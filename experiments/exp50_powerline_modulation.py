#!/usr/bin/env python3
"""
Experiment 50: Power-Line Modulation Transmission
==================================================

We discovered that both GPUs see the same 1.09 Hz power-line signal
with 100% coherence. This is the 55th subharmonic of 60 Hz.

Strategy:
1. Modulate GPU power draw at the carrier frequency
2. Heavy compute = high power, idle = low power
3. Receiver looks for power modulation in the coherent signal

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
from datetime import datetime, timezone
from scipy import signal

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003

# Carrier frequency (power line subharmonic)
CARRIER_FREQ = 60.0 / 55  # 1.0909 Hz
CARRIER_PERIOD = 1.0 / CARRIER_FREQ  # ~0.917 seconds


class PowerModulationSensor:
    """Sensor for power-line modulation experiments."""

    def __init__(self, n_ossicles: int, depth: int = 64):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

        self.reset()

        # Pre-allocate large matrices for power modulation
        if HAS_CUDA:
            self.heavy_a = cp.random.random((2048, 2048), dtype=cp.float32)
            self.heavy_b = cp.random.random((2048, 2048), dtype=cp.float32)

    def reset(self):
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

    def heavy_compute(self):
        """Do heavy compute to increase power draw."""
        if HAS_CUDA:
            for _ in range(5):
                _ = cp.matmul(self.heavy_a, self.heavy_b)
            cp.cuda.stream.get_current_stream().synchronize()

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


def run_power_transmitter(n_ossicles: int, bit_pattern: str, output_path: str):
    """Transmit by modulating power draw at carrier frequency."""

    print(f"\n{'='*60}")
    print(f"POWER-LINE MODULATION TRANSMITTER")
    print(f"{'='*60}")

    print(f"\n  Carrier: {CARRIER_FREQ:.4f} Hz ({CARRIER_PERIOD:.3f}s period)")
    print(f"  Bit pattern: {bit_pattern}")
    print(f"  Strategy: Heavy compute for '1', idle for '0'")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name'].decode()
        print(f"  GPU: {device_name}")
    else:
        device_name = "CPU"

    sensor = PowerModulationSensor(n_ossicles)

    sample_rate = 100.0  # Higher rate for phase precision
    interval = 1.0 / sample_rate

    k_eff_series = []
    timestamps = []
    bit_starts = []
    power_states = []  # 1 = high power, 0 = low power

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    # Sync to carrier phase
    print("\n  Syncing to carrier phase...")
    time.sleep(CARRIER_PERIOD - (time.time() % CARRIER_PERIOD))

    # Transmit each bit over one carrier period
    for bit_idx, bit in enumerate(bit_pattern):
        bit_start = time.time()
        bit_starts.append(bit_start)

        print(f"\n    Bit {bit_idx}: '{bit}' (t={bit_start - capture_start:.2f}s)")

        # Transmit for one carrier period
        samples_in_bit = 0
        while time.time() - bit_start < CARRIER_PERIOD:
            sample_start = time.perf_counter()

            sensor.step(3)

            if bit == '1':
                # High power: do heavy compute
                sensor.heavy_compute()
                power_states.append(1)
            else:
                # Low power: just idle
                power_states.append(0)

            k_eff_series.append(sensor.measure_k_eff())
            timestamps.append(time.time())
            samples_in_bit += 1

            elapsed = time.perf_counter() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        print(f"      {samples_in_bit} samples")

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='power_transmitter',
             device=device_name,
             k_eff=np.array(k_eff_series),
             timestamps=np.array(timestamps),
             bit_pattern=bit_pattern,
             bit_starts=np.array(bit_starts),
             power_states=np.array(power_states),
             capture_start=capture_start,
             carrier_freq=CARRIER_FREQ)

    print(f"  Saved: {output_path}")


def run_power_receiver(n_ossicles: int, duration_sec: float, output_path: str):
    """Receive power modulation signal."""

    print(f"\n{'='*60}")
    print(f"POWER-LINE MODULATION RECEIVER")
    print(f"{'='*60}")

    print(f"\n  Duration: {duration_sec}s")
    print(f"  Carrier: {CARRIER_FREQ:.4f} Hz")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name'].decode()
        print(f"  GPU: {device_name}")
    else:
        device_name = "CPU"

    sensor = PowerModulationSensor(n_ossicles)

    sample_rate = 100.0
    interval = 1.0 / sample_rate
    n_samples = int(duration_sec * sample_rate)

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

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n_samples}")

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             mode='power_receiver',
             device=device_name,
             k_eff=k_eff_series,
             timestamps=timestamps,
             capture_start=capture_start,
             sample_rate=sample_rate,
             carrier_freq=CARRIER_FREQ)

    print(f"  Saved: {output_path}")


def analyze_power_transmission(tx_file: str, rx_file: str):
    """Analyze power-line modulation transmission."""

    print(f"\n{'='*60}")
    print(f"POWER-LINE MODULATION ANALYSIS")
    print(f"{'='*60}")

    tx = np.load(tx_file, allow_pickle=True)
    rx = np.load(rx_file)

    bit_pattern = str(tx['bit_pattern'])
    tx_bit_starts = tx['bit_starts']
    tx_k = tx['k_eff']
    tx_ts = tx['timestamps']
    tx_power = tx['power_states']

    rx_k = rx['k_eff']
    rx_ts = rx['timestamps']
    rx_rate = float(rx['sample_rate'])

    print(f"\n  TX bit pattern: {bit_pattern}")
    print(f"  TX samples: {len(tx_k)}")
    print(f"  RX samples: {len(rx_k)}")

    # Bandpass filter around carrier frequency
    nyq = rx_rate / 2
    low = CARRIER_FREQ * 0.5
    high = min(CARRIER_FREQ * 2, nyq - 0.1)
    b, a = signal.butter(3, [low/nyq, high/nyq], btype='band')

    tx_k_filt = signal.filtfilt(b, a, signal.detrend(tx_k))
    rx_k_filt = signal.filtfilt(b, a, signal.detrend(rx_k))

    print(f"\n  Carrier band: {low:.2f} - {high:.2f} Hz")

    # Analyze each bit period
    print(f"\n{'='*60}")
    print("BIT-BY-BIT ANALYSIS")
    print(f"{'='*60}")

    tx_amplitudes = []
    rx_amplitudes = []

    for i, bit_start in enumerate(tx_bit_starts):
        bit_end = bit_start + CARRIER_PERIOD
        actual = bit_pattern[i]

        # TX power in this bit
        tx_mask = (tx_ts >= bit_start) & (tx_ts < bit_end)
        tx_bit_filt = tx_k_filt[tx_mask] if np.sum(tx_mask) > 0 else []

        # RX power in same time window
        rx_mask = (rx_ts >= bit_start) & (rx_ts < bit_end)
        rx_bit_filt = rx_k_filt[rx_mask] if np.sum(rx_mask) > 0 else []

        if len(tx_bit_filt) > 5 and len(rx_bit_filt) > 5:
            tx_amp = np.std(tx_bit_filt)
            rx_amp = np.std(rx_bit_filt)

            tx_amplitudes.append((actual, tx_amp))
            rx_amplitudes.append((actual, rx_amp))

            print(f"\n  Bit {i} ('{actual}'):")
            print(f"    TX carrier amplitude: {tx_amp:.6f}")
            print(f"    RX carrier amplitude: {rx_amp:.6f}")

    if len(tx_amplitudes) > 0:
        # Separate by bit value
        tx_1 = [a for b, a in tx_amplitudes if b == '1']
        tx_0 = [a for b, a in tx_amplitudes if b == '0']
        rx_1 = [a for b, a in rx_amplitudes if b == '1']
        rx_0 = [a for b, a in rx_amplitudes if b == '0']

        print(f"\n{'='*60}")
        print("AGGREGATE ANALYSIS")
        print(f"{'='*60}")

        print(f"\n  TX amplitudes:")
        print(f"    '1' bits: {np.mean(tx_1):.6f} (n={len(tx_1)})")
        print(f"    '0' bits: {np.mean(tx_0):.6f} (n={len(tx_0)})")
        print(f"    Ratio: {np.mean(tx_1)/np.mean(tx_0):.2f}x")

        print(f"\n  RX amplitudes:")
        print(f"    '1' bits: {np.mean(rx_1):.6f} (n={len(rx_1)})")
        print(f"    '0' bits: {np.mean(rx_0):.6f} (n={len(rx_0)})")
        print(f"    Ratio: {np.mean(rx_1)/np.mean(rx_0):.2f}x")

        # Test if RX shows same pattern as TX
        from scipy import stats

        if len(rx_1) > 1 and len(rx_0) > 1:
            t, p = stats.ttest_ind(rx_1, rx_0)
            print(f"\n  RX T-test ('1' vs '0'): t = {t:.2f}, p = {p:.4f}")

            if np.mean(rx_1) > np.mean(rx_0) and p < 0.05:
                print("\n  ★★★ POWER MODULATION TRANSMITTED! ★★★")
                print("  RX sees higher carrier amplitude during TX '1' bits!")
            elif np.mean(rx_1) != np.mean(rx_0) and p < 0.10:
                print("\n  ★★ MARGINAL ★★")
            else:
                print("\n  No significant power modulation detected at receiver")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Power-Line Modulation')
    parser.add_argument('--mode', choices=['transmit', 'receive', 'analyze'], required=True)
    parser.add_argument('--ossicles', type=int, default=8192)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--bits', default='11110000')
    parser.add_argument('--output', '-o', default='/tmp/power_mod.npz')
    parser.add_argument('--tx-file')
    parser.add_argument('--rx-file')

    args = parser.parse_args()

    if args.mode == 'transmit':
        run_power_transmitter(args.ossicles, args.bits, args.output)
    elif args.mode == 'receive':
        run_power_receiver(args.ossicles, args.duration, args.output)
    elif args.mode == 'analyze':
        analyze_power_transmission(args.tx_file, args.rx_file)


if __name__ == "__main__":
    main()
