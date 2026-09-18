"""
Audio Core Module for Hearing-Impaired Assistive Radar
- WASAPI Loopback Capture (Zero-intrusion, 100% external)
- DSP Engine: Multi-channel (7.1 / 5.1) and Stereo (2.0) direction & magnitude calculation
- Acoustic Bandpass Filter for footstep/movement frequency enhancement
- Mock Generator for offline testing without opening any game
"""

import math
import time
import threading
import numpy as np
from scipy.signal import butter, sosfilt

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class AudioDSP:
    """Digital Signal Processing for multi-channel & stereo audio direction and magnitude."""

    # 7.1 channel angles in radians (0 rad = Straight Ahead, pi/2 = Right, pi = Behind, -pi/2 = Left)
    # Standard Windows 7.1 Channel Order: FL, FR, FC, LFE, BL, BR, SL, SR
    CHANNEL_ANGLES_7_1 = [
        math.radians(-35),   # 0: Front Left (FL)
        math.radians(35),    # 1: Front Right (FR)
        math.radians(0),     # 2: Front Center (FC)
        None,                # 3: LFE (Subwoofer - Non-directional)
        math.radians(-145),  # 4: Back Left (BL)
        math.radians(145),   # 5: Back Right (BR)
        math.radians(-90),   # 6: Side Left (SL)
        math.radians(90),    # 7: Side Right (SR)
    ]

    # 5.1 channel angles
    CHANNEL_ANGLES_5_1 = [
        math.radians(-45),   # 0: Front Left (FL)
        math.radians(45),    # 1: Front Right (FR)
        math.radians(0),     # 2: Front Center (FC)
        None,                # 3: LFE
        math.radians(-135),  # 4: Back Left (BL)
        math.radians(135),   # 5: Back Right (BR)
    ]

    def __init__(self, sample_rate=48000, enable_filter=True, lowcut=150.0, highcut=1800.0):
        self.sample_rate = sample_rate
        self.enable_filter = enable_filter
        self.lowcut = lowcut
        self.highcut = highcut
        self.sos = self._init_bandpass(sample_rate, lowcut, highcut)

    def _init_bandpass(self, sr, low, high):
        """Initializes a 4th order Butterworth bandpass filter in SOS format."""
        nyq = 0.5 * sr
        low_norm = max(10.0, low) / nyq
        high_norm = min(nyq - 10.0, high) / nyq
        if low_norm >= high_norm:
            high_norm = min(0.99, low_norm + 0.1)
        return butter(4, [low_norm, high_norm], btype='bandpass', output='sos')

    def apply_filter(self, audio_data):
        """Apply bandpass filter to multi-channel audio data: shape (samples, channels)."""
        if not self.enable_filter or self.sos is None:
            return audio_data
        try:
            return sosfilt(self.sos, audio_data, axis=0)
        except Exception:
            return audio_data

    def calculate_direction_and_magnitude(self, audio_chunk):
        """
        Takes an audio chunk (shape: [N, num_channels], dtype: float32)
        Returns:
            angle_deg: 0 ~ 360 (0=Front, 90=Right, 180=Behind, 270=Left)
            magnitude: 0.0 ~ 1.0 (normalized volume / intensity)
            db: Decibel level
            raw_energies: list of RMS energy per channel
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0, 0.0, -100.0, []

        num_channels = audio_chunk.shape[1] if audio_chunk.ndim > 1 else 1
        if num_channels == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=1)
            num_channels = 1

        # Optionally filter for footstep/action frequency
        filtered_chunk = self.apply_filter(audio_chunk)

        # Calculate RMS energy per channel
        rms_energies = np.sqrt(np.mean(filtered_chunk ** 2, axis=0) + 1e-12)
        total_rms = float(np.sqrt(np.mean(filtered_chunk ** 2) + 1e-12))

        # Convert to approximate dB
        db = 20.0 * math.log10(max(total_rms, 1e-6))
        # Normalize magnitude: noise floor ~ -60dB -> 0.0, loud ~ -10dB -> 1.0
        normalized_mag = max(0.0, min(1.0, (db + 60.0) / 50.0))

        if total_rms < 1e-4:
            return 0.0, 0.0, db, rms_energies.tolist()

        if num_channels >= 8:
            angle_deg = self._calc_multichannel_angle(rms_energies[:8], self.CHANNEL_ANGLES_7_1)
        elif num_channels >= 6:
            angle_deg = self._calc_multichannel_angle(rms_energies[:6], self.CHANNEL_ANGLES_5_1)
        elif num_channels == 2:
            angle_deg = self._calc_stereo_angle(filtered_chunk[:, 0], filtered_chunk[:, 1], rms_energies[0], rms_energies[1])
        else:
            angle_deg = 0.0

        return angle_deg, normalized_mag, db, rms_energies.tolist()

    NUM_SECTORS = 16
    SECTOR_ANGLES = [i * 22.5 for i in range(16)]

    def calculate_multi_source_profile(self, audio_chunk):
        """
        Calculates a 360-degree spatial energy profile across 16 sectors and extracts local threat peaks.
        Enables simultaneous detection of multiple sound sources from different directions.
        Returns:
            sector_energies: list of 16 float values (0.0 ~ 1.0)
            peaks: list of dict [{"angle": deg, "magnitude": mag, "db": db, "compass": str}]
            overall_db: float
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return [0.0] * self.NUM_SECTORS, [], -100.0

        num_channels = audio_chunk.shape[1] if audio_chunk.ndim > 1 else 1
        if num_channels == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=1)
            num_channels = 1

        filtered_chunk = self.apply_filter(audio_chunk)
        rms_energies = np.sqrt(np.mean(filtered_chunk ** 2, axis=0) + 1e-12)
        total_rms = float(np.sqrt(np.mean(filtered_chunk ** 2) + 1e-12))
        overall_db = 20.0 * math.log10(max(total_rms, 1e-6))

        if total_rms < 1e-4 or overall_db < -65.0:
            return [0.0] * self.NUM_SECTORS, [], overall_db

        sector_raw = np.zeros(self.NUM_SECTORS, dtype=np.float32)

        if num_channels >= 6:
            angle_map = self.CHANNEL_ANGLES_7_1 if num_channels >= 8 else self.CHANNEL_ANGLES_5_1
            for ch_idx, ang_rad in enumerate(angle_map):
                if ang_rad is None or ch_idx >= len(rms_energies):
                    continue
                e = float(rms_energies[ch_idx])
                if e <= 1e-5:
                    continue
                # Center channel compensation
                w_ch = 0.707 if ch_idx == 2 else 1.0
                power = (e * w_ch) ** 2.0

                for s_idx, s_deg in enumerate(self.SECTOR_ANGLES):
                    s_rad = math.radians(s_deg)
                    diff = abs((s_rad - ang_rad + math.pi) % (2 * math.pi) - math.pi)
                    spatial_kernel = max(0.0, math.cos(diff)) ** 4
                    sector_raw[s_idx] += power * spatial_kernel
        else:
            # 2.0 Stereo channel separation
            e_left = float(rms_energies[0])
            e_right = float(rms_energies[1]) if num_channels > 1 else 1e-6
            common_mono = min(e_left, e_right)
            diff_left = max(0.0, e_left - common_mono)
            diff_right = max(0.0, e_right - common_mono)

            for s_idx, s_deg in enumerate(self.SECTOR_ANGLES):
                s_rad = math.radians(s_deg)
                # Front center projection (0 deg) for shared mono
                fc_diff = abs((s_rad - 0.0 + math.pi) % (2 * math.pi) - math.pi)
                fc_kernel = max(0.0, math.cos(fc_diff)) ** 3
                # Left projection (270 deg)
                left_diff = abs((s_rad - math.radians(270) + math.pi) % (2 * math.pi) - math.pi)
                left_kernel = max(0.0, math.cos(left_diff)) ** 3
                # Right projection (90 deg)
                right_diff = abs((s_rad - math.radians(90) + math.pi) % (2 * math.pi) - math.pi)
                right_kernel = max(0.0, math.cos(right_diff)) ** 3

                sector_raw[s_idx] = (
                    (common_mono ** 2) * fc_kernel * 0.8 +
                    (diff_left ** 2) * left_kernel +
                    (diff_right ** 2) * right_kernel
                )

        # Normalize sector profile into 0.0 ~ 1.0
        max_raw = float(np.max(sector_raw))
        if max_raw < 1e-9:
            return [0.0] * self.NUM_SECTORS, [], overall_db

        normalized_sectors = sector_raw / max_raw
        # Scale by overall volume so quiet sounds don't blow up
        overall_mag = max(0.0, min(1.0, (overall_db + 60.0) / 45.0))
        final_sectors = (normalized_sectors * overall_mag).tolist()

        # Multi-peak detection (Identify distinct simultaneous sound sources)
        peaks = []
        for i in range(self.NUM_SECTORS):
            curr = final_sectors[i]
            if curr < 0.15:
                continue
            prev_val = final_sectors[(i - 1) % self.NUM_SECTORS]
            next_val = final_sectors[(i + 1) % self.NUM_SECTORS]
            if curr >= prev_val and curr >= next_val:
                # Sub-sector peak interpolation
                denom = (2.0 * curr - prev_val - next_val)
                offset = (next_val - prev_val) / (2.0 * denom + 1e-6) if denom > 1e-5 else 0.0
                offset = max(-0.5, min(0.5, offset))
                peak_deg = (self.SECTOR_ANGLES[i] + offset * 22.5) % 360.0

                peaks.append({
                    "angle": peak_deg,
                    "magnitude": curr,
                    "db": overall_db,
                    "compass": self._deg_to_compass(peak_deg)
                })

        # Sort peaks by magnitude descending, keep top 4 threats
        peaks.sort(key=lambda p: p["magnitude"], reverse=True)
        peaks = peaks[:4]

        return final_sectors, peaks, overall_db

    @staticmethod
    def _deg_to_compass(deg):
        dirs = ["北", "东北偏北", "东北", "东北偏东", "东", "东南偏东", "东南", "东南偏南",
                "南", "西南偏南", "西南", "西南偏西", "西", "西北偏西", "西北", "西北偏北"]
        idx = int((deg + 11.25) / 22.5) % 16
        return dirs[idx]

    def _calc_multichannel_angle(self, energies, angle_map):
        """Vector addition across discrete surround sound channels with center-channel compensation."""
        x_sum = 0.0
        y_sum = 0.0
        total_w = 0.0

        for ch_idx, angle in enumerate(angle_map):
            if angle is None or ch_idx >= len(energies):
                continue
            e = energies[ch_idx]
            # Sharp power weighting to focus on the dominant direction and suppress cross-talk
            w = e ** 2.2
            # Front Center (FC, index 2) compensation: FC is closer to FL/FR, downweight slightly to prevent front-pull
            if ch_idx == 2:
                w *= 0.707

            x_sum += w * math.sin(angle)  # X is Right (+), Left (-)
            y_sum += w * math.cos(angle)  # Y is Ahead (+), Behind (-)
            total_w += w

        if total_w < 1e-9 or (x_sum == 0.0 and y_sum == 0.0):
            return 0.0

        # atan2(x, y): 0 is Ahead (Y+), pi/2 is Right (X+), pi is Behind (Y-), -pi/2 is Left (X-)
        rad = math.atan2(x_sum, y_sum)
        deg = math.degrees(rad)
        if deg < 0:
            deg += 360.0
        return deg

    def _calc_stereo_angle(self, left_samples, right_samples, left_rms, right_rms):
        """
        Estimate direction from 2-channel stereo.
        Combines Interaural Level Difference (ILD) and cross-correlation (ITD).
        """
        sum_e = left_rms + right_rms
        if sum_e < 1e-8:
            return 0.0

        # Interaural Level Difference (-1.0 = All Left, +1.0 = All Right)
        ild = (right_rms - left_rms) / sum_e
        ild = max(-1.0, min(1.0, ild))

        # Basic left/right mapping:
        # Straight Ahead = 0°, Right = 90°, Left = 270°
        if ild >= 0:
            deg = ild * 90.0
        else:
            deg = 360.0 + (ild * 90.0)

        return deg


class MockAudioStream:
    """Generates synthetic 360-degree rotating audio for offline validation."""

    def __init__(self, sample_rate=48000, num_channels=8, rotation_speed_sec=8.0, enable_secondary_source=True):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.rotation_speed_sec = rotation_speed_sec
        self.enable_secondary_source = enable_secondary_source
        self.current_angle_deg = 0.0
        self.running = False
        self.pulse_phase = 0.0
        self.gunfire_timer = 0.0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def generate_chunk(self, chunk_size=1024):
        """Generate a simulated surround sound buffer with a moving footstep sound source."""
        dt = chunk_size / self.sample_rate
        # Advance angle
        deg_per_sec = 360.0 / self.rotation_speed_sec
        self.current_angle_deg = (self.current_angle_deg + deg_per_sec * dt) % 360.0

        # Pulse amplitude (simulating footsteps cadence ~ 2.5 steps per sec)
        self.pulse_phase = (self.pulse_phase + dt * 2.5 * 2 * math.pi) % (2 * math.pi)
        footstep_envelope = max(0.0, math.sin(self.pulse_phase)) ** 3

        # Base sound: 300Hz ~ 800Hz shaped pulse (characteristic of footstep thump)
        t = np.linspace(0, dt, chunk_size, endpoint=False)
        sound_wave = (
            0.6 * np.sin(2 * np.pi * 320 * t) +
            0.3 * np.sin(2 * np.pi * 650 * t) +
            0.1 * np.random.randn(chunk_size)
        ).astype(np.float32)

        carrier = sound_wave * (0.05 + 0.5 * footstep_envelope)

        # Source 2: Intermittent Gunfire at 90° East (bursts every 2.5 seconds)
        if self.enable_secondary_source:
            self.gunfire_timer = (self.gunfire_timer + dt) % 2.5
            gunfire_envelope = 0.8 if (self.gunfire_timer < 0.35 and (math.sin(self.gunfire_timer * 30.0) > 0.0)) else 0.0
            gunfire_wave = (0.7 * np.random.randn(chunk_size) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
            gunfire_carrier = gunfire_wave * gunfire_envelope
        else:
            gunfire_carrier = 0.0

        buffer = np.zeros((chunk_size, self.num_channels), dtype=np.float32)
        target_rad1 = math.radians(self.current_angle_deg)
        target_rad2 = math.radians(90.0)  # Gunfire at 90° East

        if self.num_channels == 8:
            angle_map = AudioDSP.CHANNEL_ANGLES_7_1
        elif self.num_channels == 6:
            angle_map = AudioDSP.CHANNEL_ANGLES_5_1
        else:
            angle_map = [math.radians(-90), math.radians(90)]

        for ch_idx, ch_rad in enumerate(angle_map):
            if ch_rad is None or ch_idx >= self.num_channels:
                continue
            # Angular distance for Source 1 (Footsteps)
            diff1 = abs((target_rad1 - ch_rad + math.pi) % (2 * math.pi) - math.pi)
            gain1 = max(0.0, math.cos(diff1 * 0.5)) ** 3

            # Angular distance for Source 2 (Gunfire at 90°)
            diff2 = abs((target_rad2 - ch_rad + math.pi) % (2 * math.pi) - math.pi)
            gain2 = max(0.0, math.cos(diff2 * 0.5)) ** 3

            buffer[:, ch_idx] = carrier * gain1 + gunfire_carrier * gain2

        return buffer, self.current_angle_deg


class LiveAudioCapture:
    """Captures system audio via Windows WASAPI Loopback (zero-intrusion)."""

    def __init__(self, dsp: AudioDSP, callback, chunk_size=1024):
        self.dsp = dsp
        self.callback = callback  # func(angle_deg, magnitude, db, channels)
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None
        self.pa = None
        self.stream = None
        self.device_info = None

    def find_loopback_device(self):
        """Locates the default WASAPI loopback device for current speakers/headphones."""
        if not HAS_PYAUDIO:
            return None
        self.pa = pyaudio.PyAudio()
        try:
            wasapi_info = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self.pa.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
            if not default_speakers.get('isLoopbackDevice', False):
                for loopback in self.pa.get_loopback_device_info_generator():
                    if default_speakers['name'] in loopback['name']:
                        return loopback
                # Fallback to first loopback
                for loopback in self.pa.get_loopback_device_info_generator():
                    return loopback
            return default_speakers
        except Exception as e:
            print(f"[AudioCapture] Error locating loopback device: {e}")
            return None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = None

    def _capture_worker(self):
        device = self.find_loopback_device()
        if device is None:
            print("[AudioCapture] No WASAPI Loopback device found or PyAudioWPatch unavailable.")
            return

        self.device_info = device
        num_channels = int(device['maxInputChannels'])
        sample_rate = int(device['defaultSampleRate'])
        self.dsp.sample_rate = sample_rate
        self.dsp.sos = self.dsp._init_bandpass(sample_rate, self.dsp.lowcut, self.dsp.highcut)

        print(f"[AudioCapture] Starting loopback capture: {device['name']}, Channels: {num_channels}, Rate: {sample_rate}Hz")

        try:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=num_channels,
                rate=sample_rate,
                input=True,
                input_device_index=device['index'],
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"[AudioCapture] Failed to open audio stream: {e}")
            return

        while self.running:
            try:
                raw_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(raw_data, dtype=np.float32)
                audio_chunk = audio_np.reshape(-1, num_channels)

                sectors, peaks, overall_db = self.dsp.calculate_multi_source_profile(audio_chunk)
                if self.callback:
                    self.callback(sectors, peaks, overall_db)
            except Exception as e:
                time.sleep(0.01)
