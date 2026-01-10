#!/usr/bin/env python3
"""
Experiment 48: Reset-Based Transmission
=======================================

Key insight from exp47: Cross-device correlation is highest during
the CONVERGENCE PHASE (~0.97), but drops to ~0.05 at steady state.

Strategy: Reset before each bit to stay in high-correlation regime.
- For '1': Reset and let converge normally (negentropic)
- For '0': Reset and inject disorder (entropic)

The CONVERGENCE RATE becomes the modulated signal.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
from datetime import datetime, timezone

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003

BIT_DURATION = 5.0  # Seconds per bit (shorter for faster transmission)


class ResetSensor:
    """Sensor with fast reset capability for transmission."""

    def __init__(self, n_ossicles: int, depth: int = 32):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

        self.reset()

    def reset(self):
        """Reset to random initial state."""
        xp = cp if HAS_CUDA else np
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

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

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def inject_disorder(self, amplitude: float = 0.5):
        """Inject uncorrelated noise (entropy)."""
        xp = cp if HAS_CUDA else np
        self.osc_a += (xp.random.random(self.total).astype(xp.float32) - 0.5) * amplitude
        self.osc_b += (xp.random.random(self.total).astype(xp.float32) - 0.5) * amplitude
        self.osc_c += (xp.random.random(self.total).astype(xp.float32) - 0.5) * amplitude

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


def run_reset_transmitter(n_ossicles: int, bit_pattern: str, output_path: str):
    """Transmit using reset-based modulation."""

    print(f"\n{'='*60}")
    print(f"RESET-BASED TRANSMITTER")
    print(f"{'='*60}")

    print(f"\n  Bit pattern: {bit_pattern}")
    print(f"  Duration per bit: {BIT_DURATION}s")
    print(f"  Total TX time: {len(bit_pattern) * BIT_DURATION:.1f}s")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    sensor = ResetSensor(n_ossicles)

    sample_rate = 50.0  # Higher rate to capture convergence
    interval = 1.0 / sample_rate

    k_eff_series = []
    timestamps = []
    bit_starts = []
    bit_ends = []

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    for bit_idx, bit in enumerate(bit_pattern):
        print(f"\n    Bit {bit_idx}: '{bit}'")

        # Reset at start of each bit
        sensor.reset()
        bit_start = time.time()
        bit_starts.append(bit_start)

        while time.time() - bit_start < BIT_DURATION:
            sample_start = time.perf_counter()

            sensor.step(5)

            if bit == '0':
                # Inject disorder to slow convergence
                sensor.inject_disorder(0.1)

            k_eff_series.append(sensor.measure_k_eff())
            timestamps.append(time.time())

            elapsed = time.perf_counter() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        bit_ends.append(time.time())

        # Report convergence rate
        bit_samples = [k for t, k in zip(timestamps, k_eff_series) if t >= bit_start]
        if len(bit_samples) > 10:
            rate = (bit_samples[-1] - bit_samples[0]) / BIT_DURATION
            print(f"      Convergence rate: {rate:.4f} k_eff/s")

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='reset_transmitter',
             k_eff=np.array(k_eff_series),
             timestamps=np.array(timestamps),
             bit_pattern=bit_pattern,
             bit_starts=np.array(bit_starts),
             bit_ends=np.array(bit_ends),
             capture_start=capture_start,
             bit_duration=BIT_DURATION)

    print(f"  Saved: {output_path}")


def run_reset_receiver(n_ossicles: int, duration_sec: float, output_path: str):
    """Receive with resets synchronized to expected bit timing."""

    print(f"\n{'='*60}")
    print(f"RESET-BASED RECEIVER")
    print(f"{'='*60}")

    print(f"\n  Duration: {duration_sec}s")
    print(f"  Bit duration: {BIT_DURATION}s")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    sensor = ResetSensor(n_ossicles)

    sample_rate = 50.0
    interval = 1.0 / sample_rate

    k_eff_series = []
    timestamps = []
    reset_times = []

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()
    last_reset = capture_start

    n_samples = int(duration_sec * sample_rate)

    for i in range(n_samples):
        sample_start = time.perf_counter()
        current_time = time.time()

        # Reset every BIT_DURATION seconds
        if current_time - last_reset >= BIT_DURATION:
            sensor.reset()
            last_reset = current_time
            reset_times.append(current_time)

        sensor.step(5)
        k_eff_series.append(sensor.measure_k_eff())
        timestamps.append(current_time)

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_samples}")

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Recorded {len(reset_times)} bit periods")

    np.savez(output_path,
             mode='reset_receiver',
             k_eff=np.array(k_eff_series),
             timestamps=np.array(timestamps),
             reset_times=np.array(reset_times),
             capture_start=capture_start,
             sample_rate=sample_rate,
             bit_duration=BIT_DURATION)

    print(f"  Saved: {output_path}")


def analyze_reset_transmission(tx_file: str, rx_file: str):
    """Analyze reset-based transmission."""

    print(f"\n{'='*60}")
    print(f"RESET-BASED TRANSMISSION ANALYSIS")
    print(f"{'='*60}")

    tx = np.load(tx_file, allow_pickle=True)
    rx = np.load(rx_file)

    bit_pattern = str(tx['bit_pattern'])
    tx_bit_starts = tx['bit_starts']
    tx_capture_start = float(tx['capture_start'])

    rx_k = rx['k_eff']
    rx_ts = rx['timestamps']
    rx_resets = rx['reset_times']
    rx_capture_start = float(rx['capture_start'])

    print(f"\n  Transmitted: {bit_pattern}")
    print(f"  TX bit periods: {len(tx_bit_starts)}")
    print(f"  RX reset periods: {len(rx_resets)}")

    # Time offset between captures
    time_offset = tx_capture_start - rx_capture_start
    print(f"  Time offset (TX - RX): {time_offset:.2f}s")

    # For each TX bit, find the corresponding RX period
    print(f"\n{'='*60}")
    print("CONVERGENCE RATE ANALYSIS")
    print(f"{'='*60}")

    tx_rates = []
    rx_rates = []
    actual_bits = []

    for i, bit_start in enumerate(tx_bit_starts):
        bit_end = bit_start + BIT_DURATION
        actual = bit_pattern[i]

        # TX convergence rate
        tx_k = tx['k_eff']
        tx_ts = tx['timestamps']
        tx_mask = (tx_ts >= bit_start) & (tx_ts < bit_end)
        tx_bit_k = tx_k[tx_mask]

        if len(tx_bit_k) > 10:
            tx_rate = (tx_bit_k[-1] - tx_bit_k[0]) / BIT_DURATION
            tx_rates.append(tx_rate)
        else:
            tx_rates.append(0)

        # Find matching RX period
        rx_bit_start = bit_start  # Same absolute time
        rx_bit_end = rx_bit_start + BIT_DURATION
        rx_mask = (rx_ts >= rx_bit_start) & (rx_ts < rx_bit_end)
        rx_bit_k = rx_k[rx_mask]

        if len(rx_bit_k) > 10:
            rx_rate = (rx_bit_k[-1] - rx_bit_k[0]) / BIT_DURATION
            rx_rates.append(rx_rate)
        else:
            rx_rates.append(0)

        actual_bits.append(actual)

        print(f"  Bit {i} ('{actual}'): TX rate={tx_rates[-1]:.4f}, RX rate={rx_rates[-1]:.4f}")

    # Correlation between TX and RX convergence rates
    if len(tx_rates) > 2:
        r = np.corrcoef(tx_rates, rx_rates)[0, 1]
        print(f"\n  TX-RX rate correlation: r = {r:.3f}")

    # Decode based on RX convergence rate
    if len(rx_rates) > 0:
        threshold = np.median(rx_rates)
        print(f"\n  RX rate threshold: {threshold:.4f}")

        correct = 0
        print(f"\nDecoding:")
        for i, (rx_rate, actual) in enumerate(zip(rx_rates, actual_bits)):
            # Higher rate = faster convergence = 1 (less disorder)
            decoded = '1' if rx_rate > threshold else '0'
            match = "✓" if decoded == actual else "✗"
            if decoded == actual:
                correct += 1
            print(f"  Bit {i}: actual={actual}, RX rate={rx_rate:.4f}, decoded={decoded} {match}")

        accuracy = correct / len(actual_bits) * 100
        print(f"\n  Accuracy: {correct}/{len(actual_bits)} = {accuracy:.1f}%")

        # Significance
        from scipy import stats
        p = 1 - stats.binom.cdf(correct - 1, len(actual_bits), 0.5)
        print(f"  P-value: {p:.4f}")

        if accuracy > 70 and p < 0.05:
            print("\n  ★★★ SIGNIFICANT BIT TRANSMISSION! ★★★")
        elif accuracy > 60 and p < 0.10:
            print("\n  ★★ MARGINAL ★★")
        else:
            print("\n  Not significant")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Reset-Based Transmission')
    parser.add_argument('--mode', choices=['transmit', 'receive', 'analyze'], required=True)
    parser.add_argument('--ossicles', type=int, default=4096)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--bits', default='10101010')
    parser.add_argument('--output', '-o', default='/tmp/reset.npz')
    parser.add_argument('--tx-file')
    parser.add_argument('--rx-file')

    args = parser.parse_args()

    if args.mode == 'transmit':
        run_reset_transmitter(args.ossicles, args.bits, args.output)
    elif args.mode == 'receive':
        run_reset_receiver(args.ossicles, args.duration, args.output)
    elif args.mode == 'analyze':
        analyze_reset_transmission(args.tx_file, args.rx_file)


if __name__ == "__main__":
    main()
