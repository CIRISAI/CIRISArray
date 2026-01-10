#!/usr/bin/env python3
"""
Experiment 47: Carrier-Based Cross-Device Transmission
======================================================

Uses the discovered 0.05-0.15 Hz environmental carrier to attempt
information transmission between Jetson and 4090.

Carrier characteristics:
- Frequency: ~0.09 Hz (10-second period)
- Coherence: 89%
- Jetson leads 4090 by 0.3s

Transmission strategy:
1. AMPLITUDE MODULATION: Inject correlated pattern at carrier frequency
2. ON-OFF KEYING: Pulse transmissions at carrier frequency to send bits
3. PHASE MODULATION: Try to shift the carrier phase

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

# Carrier parameters discovered in exp45/46
CARRIER_FREQ = 0.09  # Hz
CARRIER_PERIOD = 1.0 / CARRIER_FREQ  # ~11 seconds


class TransientSensor:
    """Transient mode sensor with carrier injection capability."""

    def __init__(self, n_ossicles: int, depth: int = 32, noise_amplitude: float = 0.02):
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

        # Inject noise
        noise_a = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude
        noise_b = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude
        noise_c = (xp.random.random(self.total).astype(xp.float32) - 0.5) * self.noise_amplitude

        self.osc_a = self.osc_a + noise_a
        self.osc_b = self.osc_b + noise_b
        self.osc_c = self.osc_c + noise_c

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def inject_carrier(self, phase: float, amplitude: float = 0.3):
        """Inject signal at carrier frequency."""
        xp = cp if HAS_CUDA else np

        # Correlation boost modulated by carrier phase
        # phase = 0 to 2π over carrier period
        modulation = np.sin(phase) * amplitude

        if modulation > 0:
            # Positive: increase correlation (negentropy)
            blend = abs(modulation)
            self.osc_b = self.osc_b * (1 - blend) + self.osc_a * blend
            self.osc_c = self.osc_c * (1 - blend) + self.osc_a * blend
        else:
            # Negative: add uncorrelated noise (entropy)
            noise_amp = abs(modulation)
            self.osc_b += (xp.random.random(self.total).astype(xp.float32) - 0.5) * noise_amp
            self.osc_c += (xp.random.random(self.total).astype(xp.float32) - 0.5) * noise_amp

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


def run_carrier_transmitter(n_ossicles: int, duration_sec: float, bit_pattern: str, output_path: str):
    """Transmit bits using carrier modulation."""

    print(f"\n{'='*60}")
    print(f"CARRIER TRANSMITTER")
    print(f"{'='*60}")

    print(f"\n  Carrier frequency: {CARRIER_FREQ:.3f} Hz ({CARRIER_PERIOD:.1f}s period)")
    print(f"  Bit pattern: {bit_pattern}")
    print(f"  Bits to transmit: {len(bit_pattern)}")
    print(f"  Duration per bit: {CARRIER_PERIOD}s")
    print(f"  Total TX time: {len(bit_pattern) * CARRIER_PERIOD:.1f}s")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    sensor = TransientSensor(n_ossicles, depth=32)

    sample_rate = 20.0
    interval = 1.0 / sample_rate

    k_eff_series = []
    timestamps = []
    bit_times = []  # When each bit starts

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    # Warmup phase
    warmup_end = capture_start + 30
    print(f"\n  Warmup phase (30s)...")
    while time.time() < warmup_end:
        sensor.step(5)
        k_eff_series.append(sensor.measure_k_eff())
        timestamps.append(time.time())
        time.sleep(max(0, interval - 0.01))

    print(f"\n  Transmitting bits...")

    # Transmit each bit
    for bit_idx, bit in enumerate(bit_pattern):
        bit_start = time.time()
        bit_times.append(bit_start)

        print(f"    Bit {bit_idx}: '{bit}' at t={bit_start - capture_start:.1f}s")

        # One carrier period per bit
        while time.time() - bit_start < CARRIER_PERIOD:
            current_time = time.time()
            phase = 2 * np.pi * (current_time - bit_start) / CARRIER_PERIOD

            sensor.step(3)

            if bit == '1':
                # Inject carrier (high amplitude)
                sensor.inject_carrier(phase, amplitude=0.5)
            else:
                # Just step (low/no amplitude)
                pass

            k_eff_series.append(sensor.measure_k_eff())
            timestamps.append(current_time)

            time.sleep(max(0, interval - 0.01))

    # Cooldown
    cooldown_end = time.time() + 20
    print(f"\n  Cooldown (20s)...")
    while time.time() < cooldown_end:
        sensor.step(5)
        k_eff_series.append(sensor.measure_k_eff())
        timestamps.append(time.time())
        time.sleep(max(0, interval - 0.01))

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='transmitter',
             k_eff=np.array(k_eff_series),
             timestamps=np.array(timestamps),
             bit_pattern=bit_pattern,
             bit_times=np.array(bit_times),
             capture_start=capture_start,
             carrier_freq=CARRIER_FREQ,
             carrier_period=CARRIER_PERIOD)

    print(f"  Saved: {output_path}")


def run_carrier_receiver(n_ossicles: int, duration_sec: float, sample_rate: float, output_path: str):
    """Receive and record for carrier analysis."""

    print(f"\n{'='*60}")
    print(f"CARRIER RECEIVER")
    print(f"{'='*60}")

    print(f"\n  Duration: {duration_sec}s at {sample_rate} Hz")
    print(f"  Carrier frequency: {CARRIER_FREQ:.3f} Hz")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    sensor = TransientSensor(n_ossicles, depth=32)

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

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='receiver',
             k_eff=k_eff_series,
             timestamps=timestamps,
             capture_start=capture_start,
             sample_rate=sample_rate,
             carrier_freq=CARRIER_FREQ)

    print(f"  Saved: {output_path}")


def analyze_carrier_transmission(tx_file: str, rx_file: str):
    """Analyze carrier-based bit transmission."""
    from scipy import signal

    print(f"\n{'='*60}")
    print(f"CARRIER TRANSMISSION ANALYSIS")
    print(f"{'='*60}")

    tx = np.load(tx_file, allow_pickle=True)
    rx = np.load(rx_file)

    bit_pattern = str(tx['bit_pattern'])
    bit_times = tx['bit_times']

    rx_k = rx['k_eff']
    rx_ts = rx['timestamps']
    rx_rate = float(rx['sample_rate'])

    print(f"\n  Transmitted: {bit_pattern}")
    print(f"  {len(bit_times)} bit transitions recorded")

    # Extract carrier band from receiver
    nyq = rx_rate / 2
    low, high = 0.05, 0.15
    b, a = signal.butter(3, [low/nyq, high/nyq], btype='band')

    # Detrend and filter
    rx_k_dt = signal.detrend(rx_k)
    rx_carrier = signal.filtfilt(b, a, rx_k_dt)

    print(f"\n  Receiver carrier band power: {np.var(rx_carrier):.2e}")

    # Analyze power in carrier band during each bit
    print(f"\n{'='*60}")
    print("BIT-BY-BIT ANALYSIS")
    print(f"{'='*60}")

    bit_powers = []
    decoded_bits = []

    for i, bit_time in enumerate(bit_times):
        # Find samples during this bit period
        bit_end = bit_time + CARRIER_PERIOD
        mask = (rx_ts >= bit_time) & (rx_ts < bit_end)

        if np.sum(mask) > 5:
            bit_carrier = rx_carrier[mask]
            power = np.var(bit_carrier)
            bit_powers.append(power)

            actual = bit_pattern[i]
            print(f"  Bit {i} ('{actual}'): carrier power = {power:.2e}")

    if len(bit_powers) >= len(bit_pattern):
        # Threshold at median
        threshold = np.median(bit_powers)

        print(f"\n  Power threshold: {threshold:.2e}")
        print(f"\nDecoding:")

        correct = 0
        for i, power in enumerate(bit_powers[:len(bit_pattern)]):
            decoded = '1' if power > threshold else '0'
            actual = bit_pattern[i]
            match = "✓" if decoded == actual else "✗"
            if decoded == actual:
                correct += 1
            print(f"  Bit {i}: actual={actual}, decoded={decoded} {match}")

        accuracy = correct / len(bit_pattern) * 100
        print(f"\n  Accuracy: {correct}/{len(bit_pattern)} = {accuracy:.1f}%")

        if accuracy > 70:
            print("\n  ★★★ SIGNIFICANT BIT TRANSMISSION! ★★★")
        elif accuracy > 60:
            print("\n  ★★ MARGINAL TRANSMISSION ★★")
        else:
            print("\n  No significant transmission detected")
            print("  (50% = random chance)")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Carrier-Based Transmission')
    parser.add_argument('--mode', choices=['transmit', 'receive', 'analyze'], required=True)
    parser.add_argument('--ossicles', type=int, default=4096)
    parser.add_argument('--duration', type=float, default=180)
    parser.add_argument('--rate', type=float, default=20.0)
    parser.add_argument('--bits', default='10101010', help='Bit pattern to transmit')
    parser.add_argument('--output', '-o', default='/tmp/carrier.npz')
    parser.add_argument('--tx-file', help='Transmitter file (for analyze)')
    parser.add_argument('--rx-file', help='Receiver file (for analyze)')

    args = parser.parse_args()

    if args.mode == 'transmit':
        run_carrier_transmitter(args.ossicles, args.duration, args.bits, args.output)
    elif args.mode == 'receive':
        run_carrier_receiver(args.ossicles, args.duration, args.rate, args.output)
    elif args.mode == 'analyze':
        if not args.tx_file or not args.rx_file:
            print("ERROR: Need --tx-file and --rx-file")
            return
        analyze_carrier_transmission(args.tx_file, args.rx_file)


if __name__ == "__main__":
    main()
