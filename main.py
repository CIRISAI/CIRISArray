#!/usr/bin/env python3
"""
CIRIS Array - Modular GPU Sensor Platform
==========================================

Unified entry point for different detection modes.

VALIDATED:
- Workload: GPU workload/tampering detection (mean-shift method)

THEORIZED (NOT VALIDATED):
- EMI: Electromagnetic interference spectrum analysis
- Temperature: Thermal drift detection
- VFD: Variable frequency drive interference
- Custom: User-defined frequency bands

Note: EMI/Temperature/VFD modes showed inconsistent frequency peaks
in testing (Jan 2026). Results are not reproducible and should be
considered experimental/unproven.

Author: CIRIS Research Team
License: BSL 1.1
"""

import numpy as np
import time
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from collections import deque

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy required. Install with: pip install cupy-cuda12x")
    sys.exit(1)


# =============================================================================
# DETECTION MODE CONFIGURATIONS
# =============================================================================

@dataclass
class ModeConfig:
    """Configuration for a detection mode."""
    name: str
    description: str
    sample_rate: int          # Hz
    detection_threshold: float  # σ multiplier for detection
    window_size: int          # samples for baseline
    freq_band: Tuple[float, float]  # Hz, frequency band of interest
    display_interval: float   # seconds between display updates


# Pre-defined modes
MODES = {
    'emi': ModeConfig(
        name='EMI',
        description='Electromagnetic interference (60 Hz harmonics)',
        sample_rate=500,
        detection_threshold=3.0,
        window_size=500,
        freq_band=(50, 70),
        display_interval=1.0
    ),
    'temperature': ModeConfig(
        name='Temperature',
        description='Thermal drift detection (0-0.1 Hz)',
        sample_rate=100,
        detection_threshold=2.0,
        window_size=1000,
        freq_band=(0, 0.1),
        display_interval=2.0
    ),
    'workload': ModeConfig(
        name='Workload',
        description='GPU workload detection (mean shift)',
        sample_rate=4000,
        detection_threshold=1.5,  # 50% mean shift
        window_size=400,
        freq_band=(100, 500),
        display_interval=0.5
    ),
    'vfd': ModeConfig(
        name='VFD',
        description='Variable frequency drive interference',
        sample_rate=2000,
        detection_threshold=3.0,
        window_size=2000,
        freq_band=(10, 1000),
        display_interval=1.0
    ),
    'custom': ModeConfig(
        name='Custom',
        description='User-defined parameters',
        sample_rate=1000,
        detection_threshold=3.0,
        window_size=500,
        freq_band=(0, 500),
        display_interval=1.0
    ),
}


# =============================================================================
# MINIMAL GPU TIMING SENSOR
# =============================================================================

class GPUSensor:
    """Minimal GPU timing sensor for EMI/transient detection."""

    def __init__(self, n_elements: int = 256):
        self.n = n_elements
        # Simple GPU arrays for timing measurement
        self.data = cp.random.randn(n_elements).astype(cp.float32)
        self.result = cp.zeros(1, dtype=cp.float32)

        # Warmup
        for _ in range(10):
            self._measure()

    def _measure(self) -> float:
        """Single timing measurement (microseconds)."""
        t0 = time.perf_counter_ns()
        # Simple GPU operation
        self.result[0] = cp.sum(self.data * self.data)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1000.0  # μs

    def sample(self) -> float:
        """Get one timing sample."""
        return self._measure()


# =============================================================================
# EMI SPECTRUM MODE
# =============================================================================

def emi_mode(config: ModeConfig, duration: int = 30):
    """EMI spectrum analysis (imported from ciris_sentinel)."""
    try:
        from ciris_sentinel import emi_mode as _emi_mode
        _emi_mode(duration=duration, sample_rate=config.sample_rate)
    except ImportError:
        print("ERROR: Could not import emi_mode from ciris_sentinel.py")
        _simple_emi_mode(config, duration)


def _simple_emi_mode(config: ModeConfig, duration: int):
    """Fallback simple EMI detection."""
    from scipy import signal as scipy_signal

    print("=" * 60)
    print("CIRIS EMI DETECTOR (Simple Mode)")
    print("=" * 60)

    sensor = GPUSensor()
    timings = []

    print(f"Collecting {duration}s of data at {config.sample_rate} Hz...")
    start = time.time()
    interval = 1.0 / config.sample_rate

    while time.time() - start < duration:
        timings.append(sensor.sample())
        time.sleep(interval)

    timings = np.array(timings)

    # FFT analysis
    freqs, psd = scipy_signal.welch(timings - np.mean(timings),
                                     fs=config.sample_rate, nperseg=min(len(timings)//4, 1024))

    print("\nTop frequencies:")
    peak_idx = np.argsort(psd)[-10:][::-1]
    for i in peak_idx:
        print(f"  {freqs[i]:.2f} Hz: power = {psd[i]:.2e}")


# =============================================================================
# WORKLOAD DETECTION MODE
# =============================================================================

def workload_mode(config: ModeConfig, duration: int = 60):
    """Real-time workload detection with mean-shift method."""
    print("=" * 60)
    print("CIRIS WORKLOAD DETECTOR")
    print("=" * 60)
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Detection: mean shift > {config.detection_threshold * 100:.0f}%")
    print("-" * 60)

    sensor = GPUSensor()

    # Build baseline
    print("Building baseline...")
    baseline = []
    for _ in range(config.window_size):
        baseline.append(sensor.sample())
        time.sleep(1.0 / config.sample_rate)

    baseline_mean = np.mean(baseline)
    print(f"Baseline mean: {baseline_mean:.1f} μs")
    print("-" * 60)
    print("Monitoring... (Ctrl+C to stop)")

    window = deque(maxlen=40)  # ~10ms detection window
    detections = 0
    start = time.time()
    last_print = start

    try:
        while time.time() - start < duration:
            t = sensor.sample()
            window.append(t)

            if len(window) >= 10:
                current_mean = np.mean(window)
                shift_pct = (current_mean - baseline_mean) / baseline_mean * 100

                if abs(shift_pct) > config.detection_threshold * 100:
                    detections += 1
                    direction = "↑" if shift_pct > 0 else "↓"
                    print(f"  {direction} WORKLOAD DETECTED: {shift_pct:+.0f}% shift")

            # Status update every second
            if time.time() - last_print > 1.0:
                current_mean = np.mean(window) if window else baseline_mean
                shift = (current_mean - baseline_mean) / baseline_mean * 100
                print(f"  ... mean={current_mean:.1f}μs ({shift:+.1f}%), detections={detections}")
                last_print = time.time()

            time.sleep(1.0 / config.sample_rate)

    except KeyboardInterrupt:
        pass

    print(f"\nTotal detections: {detections}")


# =============================================================================
# TEMPERATURE MODE
# =============================================================================

def temperature_mode(config: ModeConfig, duration: int = 120):
    """Long-term thermal drift monitoring."""
    print("=" * 60)
    print("CIRIS TEMPERATURE MONITOR")
    print("=" * 60)
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Update interval: {config.display_interval}s")
    print("-" * 60)

    sensor = GPUSensor()

    history = []
    start = time.time()
    last_update = start
    samples = []

    print("Monitoring thermal drift... (Ctrl+C to stop)")

    try:
        while time.time() - start < duration:
            samples.append(sensor.sample())

            if time.time() - last_update >= config.display_interval:
                if samples:
                    var = np.var(samples)
                    mean = np.mean(samples)
                    history.append((time.time() - start, var, mean))

                    # Variance inversely correlates with temperature
                    # Show trend
                    if len(history) >= 2:
                        delta_var = history[-1][1] - history[-2][1]
                        trend = "↓ WARMING" if delta_var < 0 else "↑ COOLING" if delta_var > 0 else "─ STABLE"
                    else:
                        trend = "─ STABLE"

                    elapsed = time.time() - start
                    print(f"  [{elapsed:5.0f}s] var={var:.2f}, mean={mean:.1f}μs  {trend}")

                samples = []
                last_update = time.time()

            time.sleep(1.0 / config.sample_rate)

    except KeyboardInterrupt:
        pass

    print("\nDone.")


# =============================================================================
# VFD MODE
# =============================================================================

def vfd_mode(config: ModeConfig, duration: int = 60):
    """Variable Frequency Drive interference detection."""
    from scipy import signal as scipy_signal

    print("=" * 60)
    print("CIRIS VFD DETECTOR")
    print("=" * 60)
    print("Detecting variable frequency drive harmonics...")
    print(f"Frequency band: {config.freq_band[0]}-{config.freq_band[1]} Hz")
    print("-" * 60)

    sensor = GPUSensor()

    # Collect and analyze in chunks
    chunk_duration = 5  # seconds
    start = time.time()

    try:
        while time.time() - start < duration:
            print(f"\nCollecting {chunk_duration}s chunk...")
            timings = []
            chunk_start = time.time()

            while time.time() - chunk_start < chunk_duration:
                timings.append(sensor.sample())
                time.sleep(1.0 / config.sample_rate)

            timings = np.array(timings)

            # FFT
            freqs, psd = scipy_signal.welch(timings - np.mean(timings),
                                           fs=config.sample_rate, nperseg=512)

            # Find peaks in VFD range
            mask = (freqs >= config.freq_band[0]) & (freqs <= config.freq_band[1])
            vfd_freqs = freqs[mask]
            vfd_psd = psd[mask]

            if len(vfd_psd) > 0:
                peak_idx = np.argmax(vfd_psd)
                peak_freq = vfd_freqs[peak_idx]
                peak_power = vfd_psd[peak_idx]
                noise_floor = np.median(psd)
                snr = 10 * np.log10(peak_power / noise_floor) if noise_floor > 0 else 0

                print(f"  Peak: {peak_freq:.1f} Hz, SNR: {snr:.1f} dB")

                if snr > 6:
                    print(f"  ⚠️  VFD INTERFERENCE DETECTED at {peak_freq:.1f} Hz")

    except KeyboardInterrupt:
        pass

    print("\nDone.")


# =============================================================================
# CUSTOM MODE
# =============================================================================

def custom_mode(config: ModeConfig, duration: int = 60):
    """User-customized detection mode - runs EMI-style analysis with custom params."""
    print("=" * 60)
    print("CIRIS CUSTOM MODE")
    print("=" * 60)
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Threshold: {config.detection_threshold}σ")
    print(f"Frequency band: {config.freq_band[0]}-{config.freq_band[1]} Hz")
    print("-" * 60)

    # Run VFD-style detection with custom parameters
    vfd_mode(config, duration)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CIRIS Array - Modular GPU Sensor Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  emi         - Electromagnetic interference (60 Hz harmonics)
  temperature - Thermal drift detection
  workload    - GPU workload/tampering detection
  vfd         - Variable frequency drive interference
  custom      - User-defined parameters

Examples:
  python main.py emi                      # EMI spectrum analysis
  python main.py emi --duration 60        # Run for 1 minute
  python main.py workload -d 120          # Workload detection for 2 minutes
  python main.py custom --rate 2000 --freq-low 10 --freq-high 500
        """
    )

    parser.add_argument('mode', choices=list(MODES.keys()),
                        help='Detection mode')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Duration in seconds (default: 60)')
    parser.add_argument('--rate', '-r', type=int,
                        help='Sample rate in Hz (overrides mode default)')
    parser.add_argument('--threshold', '-t', type=float,
                        help='Detection threshold in σ (overrides mode default)')
    parser.add_argument('--interval', '-i', type=float,
                        help='Display interval in seconds')
    parser.add_argument('--freq-low', type=float,
                        help='Low frequency bound (Hz)')
    parser.add_argument('--freq-high', type=float,
                        help='High frequency bound (Hz)')

    args = parser.parse_args()

    # Get base config for mode
    config = MODES[args.mode]

    # Override with command-line args
    if args.rate:
        config.sample_rate = args.rate
    if args.threshold:
        config.detection_threshold = args.threshold
    if args.interval:
        config.display_interval = args.interval
    if args.freq_low is not None and args.freq_high is not None:
        config.freq_band = (args.freq_low, args.freq_high)

    # Check CUDA
    if not cp.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
    print()

    # Dispatch to mode
    mode_handlers = {
        'emi': emi_mode,
        'temperature': temperature_mode,
        'workload': workload_mode,
        'vfd': vfd_mode,
        'custom': custom_mode,
    }

    handler = mode_handlers[args.mode]
    handler(config, args.duration)


if __name__ == '__main__':
    main()
