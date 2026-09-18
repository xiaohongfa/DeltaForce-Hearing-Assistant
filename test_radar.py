"""
Unit and Benchmark Tests for AudioDSP and Mock Stream.
Tests 360-degree direction accuracy, magnitude scaling, and processing latency.
"""

import sys
import math
import time
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from audio_core import AudioDSP, MockAudioStream


def test_7_1_cardinal_directions():
    print("=== Testing 7.1 Cardinal Directions ===")
    dsp = AudioDSP(sample_rate=48000, enable_filter=False)

    # Angles to test: 0° (Front), 90° (Right), 180° (Back), 270° (Left)
    test_cases = [
        (0.0, "Front (FC dominant)"),
        (90.0, "Right (SR dominant)"),
        (180.0, "Back (BL+BR balanced)"),
        (270.0, "Left (SL dominant)"),
        (35.0, "Front-Right (FR dominant)"),
        (325.0, "Front-Left (FL dominant)"),
    ]

    mock = MockAudioStream(sample_rate=48000, num_channels=8, enable_secondary_source=False)
    chunk_size = 1024

    for target_deg, desc in test_cases:
        mock.current_angle_deg = target_deg
        chunk, actual_angle = mock.generate_chunk(chunk_size)

        calc_angle, mag, db, raw_e = dsp.calculate_direction_and_magnitude(chunk)

        # Angular difference taking 360 wrap into account
        diff = abs((calc_angle - target_deg + 180) % 360 - 180)
        print(f"Target: {target_deg:5.1f}° ({desc:<25}) -> Calc: {calc_angle:5.1f}°, Diff: {diff:4.1f}°, Mag: {mag:.2f}, dB: {db:5.1f}")
        assert diff < 8.0, f"Error too large for {target_deg}°: diff={diff}°"

    print(">>> 7.1 Cardinal direction tests passed!\n")


def test_stereo_ild():
    print("=== Testing 2.0 Stereo Direction (ILD) ===")
    dsp = AudioDSP(sample_rate=48000, enable_filter=False)
    chunk_size = 1024

    # Case 1: Pure Left -> should be ~270°
    left_pure = np.zeros((chunk_size, 2), dtype=np.float32)
    left_pure[:, 0] = 0.5
    angle, mag, _, _ = dsp.calculate_direction_and_magnitude(left_pure)
    print(f"Pure Left -> Calc: {angle:.1f}° (Expected: 270.0°)")
    assert abs(angle - 270.0) < 1.0

    # Case 2: Pure Right -> should be ~90°
    right_pure = np.zeros((chunk_size, 2), dtype=np.float32)
    right_pure[:, 1] = 0.5
    angle, mag, _, _ = dsp.calculate_direction_and_magnitude(right_pure)
    print(f"Pure Right -> Calc: {angle:.1f}° (Expected: 90.0°)")
    assert abs(angle - 90.0) < 1.0

    # Case 3: Balanced Center -> should be 0°
    center_pure = np.ones((chunk_size, 2), dtype=np.float32) * 0.5
    angle, mag, _, _ = dsp.calculate_direction_and_magnitude(center_pure)
    print(f"Pure Center -> Calc: {angle:.1f}° (Expected: 0.0°)")
    assert abs(angle - 0.0) < 1.0

    print(">>> Stereo ILD tests passed!\n")


def test_bandpass_filter():
    print("=== Testing Bandpass Filter (Footstep Enhancement) ===")
    sr = 48000
    dsp = AudioDSP(sample_rate=sr, enable_filter=True, lowcut=150.0, highcut=1800.0)

    # 1. Test In-band signal (500 Hz footstep frequency)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    in_band = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32).reshape(-1, 1)
    in_band_filtered = dsp.apply_filter(in_band)
    rms_in_before = np.sqrt(np.mean(in_band ** 2))
    rms_in_after = np.sqrt(np.mean(in_band_filtered[sr//4:] ** 2))
    attenuation_in = rms_in_after / rms_in_before
    print(f"In-band (500Hz) pass ratio: {attenuation_in:.3f} (Expected > 0.85)")
    assert attenuation_in > 0.85

    # 2. Test Out-of-band high frequency (8000 Hz hiss/gunshot transient)
    out_band_high = (0.5 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32).reshape(-1, 1)
    out_band_filtered = dsp.apply_filter(out_band_high)
    rms_out_after = np.sqrt(np.mean(out_band_filtered[sr//4:] ** 2))
    attenuation_out = rms_out_after / rms_in_before
    print(f"Out-of-band (8000Hz) pass ratio: {attenuation_out:.3f} (Expected < 0.1)")
    assert attenuation_out < 0.1

    print(">>> Bandpass filter tests passed!\n")


def test_latency_benchmark():
    print("=== Latency Benchmark ===")
    dsp = AudioDSP(sample_rate=48000, enable_filter=True)
    chunk = np.random.randn(1024, 8).astype(np.float32)

    # Warmup
    for _ in range(20):
        dsp.calculate_direction_and_magnitude(chunk)

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        dsp.calculate_direction_and_magnitude(chunk)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"Average DSP calculation time per 1024-sample chunk (8 channels): {avg_ms:.3f} ms")
    assert avg_ms < 5.0, f"DSP calculation took too long: {avg_ms:.3f} ms"
    print(f">>> Latency benchmark passed! (Limit: < 5.0 ms, Actual: {avg_ms:.3f} ms)\n")


def test_multi_source_simultaneous_resolution():
    print("=== Testing Multi-Source Simultaneous Resolution ===")
    dsp = AudioDSP(sample_rate=48000, enable_filter=False)
    chunk_size = 1024

    # Create two simultaneous signals: Left (SL: 270°) and Front-Right (FR: 35°)
    # In Windows 7.1 order: FL(0), FR(1), FC(2), LFE(3), BL(4), BR(5), SL(6), SR(7)
    audio = np.zeros((chunk_size, 8), dtype=np.float32)
    t = np.linspace(0, chunk_size / 48000, chunk_size, endpoint=False)

    # Source A (Left footsteps at 270°): 400Hz tone on SL (ch 6)
    audio[:, 6] = 0.5 * np.sin(2 * np.pi * 400 * t)
    # Source B (Right gunfire at 35°): 1500Hz tone on FR (ch 1)
    audio[:, 1] = 0.7 * np.sin(2 * np.pi * 1500 * t)

    sectors, peaks, overall_db = dsp.calculate_multi_source_profile(audio)
    print(f"Overall dB: {overall_db:.1f} dB, Detected Peaks: {len(peaks)}")
    for i, p in enumerate(peaks):
        print(f"  Peak {i+1}: {p['angle']:.1f}° ({p['compass']}), Mag: {p['magnitude']:.2f}")

    # Verify BOTH peaks are detected
    assert len(peaks) >= 2, f"Expected at least 2 distinct peaks, got {len(peaks)}"
    angles = [p['angle'] for p in peaks]
    has_left = any(abs((a - 270.0 + 180) % 360 - 180) < 18.0 for a in angles)
    has_fr = any(abs((a - 35.0 + 180) % 360 - 180) < 18.0 for a in angles)

    assert has_left, f"Failed to detect Left (270°) source! Detected: {angles}"
    assert has_fr, f"Failed to detect Front-Right (35°) source! Detected: {angles}"
    print(">>> Multi-source simultaneous resolution test passed!\n")


if __name__ == '__main__':
    test_7_1_cardinal_directions()
    test_stereo_ild()
    test_bandpass_filter()
    test_multi_source_simultaneous_resolution()
    test_latency_benchmark()
    print("ALL TESTS PASSED SUCCESSFULLY!")
