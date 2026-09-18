"""
Main Entrypoint for Delta Force Multi-Source Acoustic Radar (MVP)
Orchestrates AudioDSP, MockAudioStream, LiveAudioCapture, and TacticalRadarApp HUD.
Uses thread-safe Queue for zero-latency multi-sector profile transmission.
"""

import time
import queue
import threading
import tkinter as tk
from audio_core import AudioDSP, MockAudioStream, LiveAudioCapture, HAS_PYAUDIO
from radar_ui import TacticalRadarApp


class RadarController:
    def __init__(self, root):
        self.root = root
        self.dsp = AudioDSP(sample_rate=48000, enable_filter=True)
        self.mock_stream = MockAudioStream(sample_rate=48000, num_channels=8, rotation_speed_sec=7.0)
        self.live_capture = None

        self.running = True
        self.active_mode = "live"  # Default to real-time live mode
        self.mock_thread = None

        # Thread-safe data communication
        self.data_queue = queue.Queue(maxsize=10)

        # Create UI
        self.ui = TacticalRadarApp(
            root=self.root,
            on_mode_change=self.set_mode,
            on_filter_change=self.set_filter
        )

        # Intercept window close
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        # Start queue polling loop on main UI thread (~60Hz)
        self._poll_queue()

        # Start default mode (live listening to system audio)
        self.set_mode("live")

    def _poll_queue(self):
        """Poll the data queue on the Tkinter main thread."""
        if not self.running:
            return
        try:
            latest = None
            while not self.data_queue.empty():
                latest = self.data_queue.get_nowait()
            if latest is not None:
                sectors, peaks, db, status = latest
                self.ui.update_multi_data(sectors, peaks, db, status=status)
        except Exception:
            pass

        self.root.after(16, self._poll_queue)

    def _push_data(self, sectors, peaks, db, status):
        """Push latest multi-source audio result into the queue without blocking."""
        try:
            self.data_queue.put_nowait((sectors, peaks, db, status))
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait((sectors, peaks, db, status))
            except Exception:
                pass

    def set_filter(self, enabled: bool):
        self.dsp.enable_filter = enabled
        print(f"[Controller] Footstep bandpass filter: {'ENABLED' if enabled else 'DISABLED'}")

    def set_mode(self, mode: str):
        print(f"[Controller] Switching mode to: {mode}")
        self.active_mode = mode

        # Stop existing streams
        self._stop_mock()
        self._stop_live()

        if mode == "mock":
            self._start_mock()
        elif mode == "live":
            self._start_live()

    def _start_mock(self):
        self.mock_stream.start()
        self.mock_thread = threading.Thread(target=self._mock_worker, daemon=True)
        self.mock_thread.start()
        self._push_data([0.0] * 16, [], -80, "多声源模拟中")

    def _stop_mock(self):
        self.mock_stream.stop()
        if self.mock_thread and self.mock_thread.is_alive():
            self.mock_thread.join(timeout=0.5)
        self.mock_thread = None

    def _mock_worker(self):
        chunk_size = 1024
        interval = chunk_size / 48000.0
        while self.running and self.active_mode == "mock":
            t0 = time.perf_counter()
            chunk, true_deg = self.mock_stream.generate_chunk(chunk_size)
            sectors, peaks, db = self.dsp.calculate_multi_source_profile(chunk)

            self._push_data(sectors, peaks, db, "双声源(脚步+枪声)")

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    def _start_live(self):
        if not HAS_PYAUDIO:
            self._push_data([0.0] * 16, [], -80, "错误: 未安装 PyAudioWPatch")
            return

        def live_callback(sectors, peaks, db):
            if self.active_mode == "live":
                device_name = self.live_capture.device_info['name'] if self.live_capture and self.live_capture.device_info else "声卡"
                if len(device_name) > 16:
                    device_name = device_name[:14] + ".."
                self._push_data(sectors, peaks, db, f"回环: {device_name}")

        self.live_capture = LiveAudioCapture(self.dsp, callback=live_callback)
        self.live_capture.start()
        self._push_data([0.0] * 16, [], -80, "正在监听耳机 [等待发声]")

    def _stop_live(self):
        if self.live_capture:
            self.live_capture.stop()
            self.live_capture = None

    def shutdown(self):
        print("[Controller] Shutting down acoustic radar...")
        self.running = False
        self._stop_mock()
        self._stop_live()
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()
    controller = RadarController(root)
    root.mainloop()


if __name__ == '__main__':
    main()
