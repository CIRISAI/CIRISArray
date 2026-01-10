#!/usr/bin/env python3
"""
Experiment 46: Cross-Device Negentropic Transmission
=====================================================

Critical test: Can negentropy transmitted from Jetson be detected on 4090?

Protocol:
1. 4090 runs as receiver in transient mode, recording k_eff
2. Jetson transmits negentropic pulses at known intervals
3. We look for correlated deflections in 4090's signal

If entropy waves are real and can carry information, we should see
the 4090 respond to Jetson's transmissions.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
from datetime import datetime, timezone
import json

# Device-specific imports
try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


class TransientSensor:
    """Sensor operating in transient mode."""

    def __init__(self, n_ossicles: int, depth: int, noise_amplitude: float = 0.02):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth
        self.noise_amplitude = noise_amplitude

        xp = cp if HAS_CUDA else np

        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 5):
        xp = cp if HAS_CUDA else np

        for _ in range(iterations):
            da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
            db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
            dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a = self.osc_a + da
            self.osc_b = self.osc_b + db
            self.osc_c = self.osc_c + dc

            self.osc_a = xp.clip(self.osc_a, -10, 10)
            self.osc_b = xp.clip(self.osc_b, -10, 10)
            self.osc_c = xp.clip(self.osc_c, -10, 10)

        # Inject noise to maintain transient
        noise_a = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude
        noise_b = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude
        noise_c = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude

        self.osc_a = self.osc_a + noise_a
        self.osc_b = self.osc_b + noise_b
        self.osc_c = self.osc_c + noise_c

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def inject_negentropic(self, amplitude: float = 0.3):
        """Inject negentropy via correlation boost."""
        xp = cp if HAS_CUDA else np
        blend = amplitude
        self.osc_b = self.osc_b * (1 - blend) + self.osc_a * blend
        self.osc_c = self.osc_c * (1 - blend) + self.osc_a * blend

    def measure_k_eff(self) -> float:
        xp = cp if HAS_CUDA else np

        sample_size = min(10000, self.total)
        if HAS_CUDA:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r_ab = float(xp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        total_var = float(xp.var(a) + xp.var(b))
        x = min(total_var / 2.0, 1.0)

        return r_ab * (1 - x) * COUPLING_FACTOR * 1000


def run_transmitter(n_ossicles: int, duration_sec: float, pulse_interval_sec: float,
                    pulse_duration_sec: float, pulse_amplitude: float, output_path: str):
    """Run as transmitter, sending negentropic pulses."""

    print(f"\n{'='*60}")
    print(f"TRANSMITTER MODE")
    print(f"{'='*60}")

    print(f"\n  CUDA: {HAS_CUDA}")
    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    print(f"  Ossicles: {n_ossicles}")
    print(f"  Duration: {duration_sec}s")
    print(f"  Pulse interval: {pulse_interval_sec}s")
    print(f"  Pulse duration: {pulse_duration_sec}s")
    print(f"  Pulse amplitude: {pulse_amplitude}")

    sensor = TransientSensor(n_ossicles, depth=32, noise_amplitude=0.02)

    pulse_times = []
    sample_times = []
    k_eff_series = []

    sample_rate = 50.0  # Hz
    interval = 1.0 / sample_rate

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()
    last_pulse = capture_start - pulse_interval_sec  # Ready for first pulse

    while time.time() - capture_start < duration_sec:
        sample_start = time.perf_counter()
        current_time = time.time()

        # Check if it's time to transmit a pulse
        if current_time - last_pulse >= pulse_interval_sec:
            print(f"    TRANSMITTING at t={current_time - capture_start:.1f}s")
            pulse_times.append(current_time)

            # Transmit for pulse_duration
            pulse_end = time.time() + pulse_duration_sec
            while time.time() < pulse_end:
                sensor.inject_negentropic(pulse_amplitude)
                sensor.step(3)

            last_pulse = current_time

        sensor.step(5)
        k_eff_series.append(sensor.measure_k_eff())
        sample_times.append(time.time())

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print(f"  END: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Transmitted {len(pulse_times)} pulses")

    np.savez(output_path,
             mode='transmitter',
             k_eff=np.array(k_eff_series),
             timestamps=np.array(sample_times),
             pulse_times=np.array(pulse_times),
             pulse_interval=pulse_interval_sec,
             pulse_duration=pulse_duration_sec,
             pulse_amplitude=pulse_amplitude,
             capture_start=capture_start)

    print(f"  Saved: {output_path}")

    return pulse_times


def run_receiver(n_ossicles: int, duration_sec: float, sample_rate: float, output_path: str):
    """Run as receiver, recording k_eff."""

    print(f"\n{'='*60}")
    print(f"RECEIVER MODE")
    print(f"{'='*60}")

    print(f"\n  CUDA: {HAS_CUDA}")
    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    print(f"  Ossicles: {n_ossicles}")
    print(f"  Duration: {duration_sec}s at {sample_rate} Hz")

    sensor = TransientSensor(n_ossicles, depth=32, noise_amplitude=0.02)

    n_samples = int(duration_sec * sample_rate)
    interval = 1.0 / sample_rate

    k_eff_series = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    for i in range(n_samples):
        sample_start = time.perf_counter()

        sensor.step(5)
        k_eff_series[i] = sensor.measure_k_eff()
        timestamps[i] = time.time()

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_samples}")

    print(f"  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='receiver',
             k_eff=k_eff_series,
             timestamps=timestamps,
             capture_start=capture_start,
             sample_rate=sample_rate)

    print(f"  Saved: {output_path}")


def analyze_transmission(tx_file: str, rx_file: str):
    """Analyze if transmitted pulses were detected."""

    print(f"\n{'='*60}")
    print(f"CROSS-DEVICE TRANSMISSION ANALYSIS")
    print(f"{'='*60}")

    tx = np.load(tx_file)
    rx = np.load(rx_file)

    pulse_times = tx['pulse_times']
    rx_timestamps = rx['timestamps']
    rx_k_eff = rx['k_eff']

    print(f"\n  Transmitter sent {len(pulse_times)} pulses")
    print(f"  Receiver recorded {len(rx_k_eff)} samples")

    # Find receiver samples around each pulse
    window_before = 1.0  # seconds
    window_after = 2.0   # seconds

    print(f"\n{'='*60}")
    print("PULSE ANALYSIS")
    print(f"{'='*60}")

    all_effects = []

    for i, pulse_time in enumerate(pulse_times):
        # Find samples in the window
        before_mask = (rx_timestamps >= pulse_time - window_before) & (rx_timestamps < pulse_time)
        after_mask = (rx_timestamps >= pulse_time) & (rx_timestamps < pulse_time + window_after)

        before_samples = rx_k_eff[before_mask]
        after_samples = rx_k_eff[after_mask]

        if len(before_samples) > 5 and len(after_samples) > 5:
            before_mean = np.mean(before_samples)
            before_std = np.std(before_samples)
            after_mean = np.mean(after_samples)

            effect = (after_mean - before_mean) / (before_std + 1e-10)
            all_effects.append(effect)

            rel_time = pulse_time - tx['capture_start']
            detected = "★" if abs(effect) > 2 else ""
            print(f"  Pulse {i+1} (t={rel_time:.1f}s): effect = {effect:+.2f}σ {detected}")

    if len(all_effects) > 0:
        mean_effect = np.mean(all_effects)
        std_effect = np.std(all_effects)
        z_score = mean_effect / (std_effect / np.sqrt(len(all_effects)) + 1e-10)

        print(f"\n{'='*60}")
        print("AGGREGATE ANALYSIS")
        print(f"{'='*60}")
        print(f"\n  Mean effect: {mean_effect:+.3f}σ")
        print(f"  Std of effects: {std_effect:.3f}σ")
        print(f"  Combined Z-score: {z_score:.2f}")
        print(f"  N pulses analyzed: {len(all_effects)}")

        if z_score > 3:
            print(f"\n  ★★★ SIGNIFICANT TRANSMISSION DETECTED ★★★")
            print(f"      Negentropy transmitted from Jetson to 4090!")
        elif z_score > 2:
            print(f"\n  ★★ MARGINAL DETECTION ★★")
        else:
            print(f"\n  No significant transmission detected.")
            print(f"  Devices may be too isolated for signal propagation.")

        # Shuffled control
        print(f"\n{'-'*60}")
        print("NULL CONTROL (shuffled pulse times)")
        print(f"{'-'*60}")

        null_effects = []
        for _ in range(100):
            fake_pulse_times = rx_timestamps[0] + np.random.random(len(pulse_times)) * (rx_timestamps[-1] - rx_timestamps[0])
            fake_effects = []

            for pulse_time in fake_pulse_times:
                before_mask = (rx_timestamps >= pulse_time - window_before) & (rx_timestamps < pulse_time)
                after_mask = (rx_timestamps >= pulse_time) & (rx_timestamps < pulse_time + window_after)

                before_samples = rx_k_eff[before_mask]
                after_samples = rx_k_eff[after_mask]

                if len(before_samples) > 5 and len(after_samples) > 5:
                    before_mean = np.mean(before_samples)
                    before_std = np.std(before_samples)
                    after_mean = np.mean(after_samples)
                    fake_effects.append((after_mean - before_mean) / (before_std + 1e-10))

            if len(fake_effects) > 0:
                null_effects.append(np.mean(fake_effects))

        null_effects = np.array(null_effects)
        percentile = np.mean(null_effects <= mean_effect) * 100

        print(f"  Null distribution: mean = {np.mean(null_effects):.3f}, std = {np.std(null_effects):.3f}")
        print(f"  Observed effect at {percentile:.1f}th percentile")

        if percentile > 95:
            print(f"  ★ Effect unlikely by chance (>95th percentile)")
        else:
            print(f"  Effect consistent with chance")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Device Transmission Test')
    parser.add_argument('--mode', choices=['transmit', 'receive', 'analyze'],
                       required=True, help='Mode')
    parser.add_argument('--ossicles', type=int, default=4096)
    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--pulse-interval', type=float, default=10.0)
    parser.add_argument('--pulse-duration', type=float, default=0.5)
    parser.add_argument('--pulse-amplitude', type=float, default=0.5)
    parser.add_argument('--output', '-o', default='/tmp/transmission.npz')
    parser.add_argument('--tx-file', help='Transmitter file (for analyze)')
    parser.add_argument('--rx-file', help='Receiver file (for analyze)')

    args = parser.parse_args()

    if args.mode == 'transmit':
        run_transmitter(
            args.ossicles, args.duration,
            args.pulse_interval, args.pulse_duration, args.pulse_amplitude,
            args.output
        )

    elif args.mode == 'receive':
        run_receiver(args.ossicles, args.duration, args.rate, args.output)

    elif args.mode == 'analyze':
        if not args.tx_file or not args.rx_file:
            print("ERROR: Need --tx-file and --rx-file for analysis")
            return
        analyze_transmission(args.tx_file, args.rx_file)


if __name__ == "__main__":
    main()
