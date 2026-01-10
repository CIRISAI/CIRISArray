#!/usr/bin/env python3
"""
Experiment 42a: Jetson Nano Receiver
====================================

Deploy CIRISArray receiver on Jetson Nano for cross-device correlation.
Designed to run in sync with exp42_4090_receiver.py on the main workstation.

Uses timestamp-based synchronization for correlation analysis.

Run on Jetson:
  python3 exp42_jetson_receiver.py --output /tmp/jetson_capture.npz

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
import os
from datetime import datetime
from dataclasses import dataclass

# Try CuPy (Jetson), fall back to NumPy
try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np  # Fallback to numpy

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class JetsonConfig:
    """Configuration sized for Jetson Nano (4GB RAM, 128 CUDA cores)."""
    n_arrays: int = 4          # Fewer arrays for Jetson
    ossicles_per_array: int = 256  # Smaller arrays
    oscillator_depth: int = 32     # Smaller depth
    warmup_seconds: float = 10.0
    capture_seconds: float = 60.0
    sample_rate_hz: float = 100.0


class JetsonOssicleArray:
    """Ossicle array sized for Jetson capabilities."""

    def __init__(self, config: JetsonConfig):
        self.config = config
        self.total_ossicles = config.n_arrays * config.ossicles_per_array
        self.total_elements = self.total_ossicles * config.oscillator_depth

        print(f"  Initializing {self.total_ossicles} ossicles ({self.total_elements} elements)")

        if HAS_CUDA:
            self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
            self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
            self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        else:
            self.osc_a = np.random.random(self.total_elements).astype(np.float32) * 0.1
            self.osc_b = np.random.random(self.total_elements).astype(np.float32) * 0.1
            self.osc_c = np.random.random(self.total_elements).astype(np.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
        """Run oscillator dynamics."""
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

    def measure_k_eff(self) -> float:
        """Measure effective coupling."""
        xp = cp if HAS_CUDA else np

        # Use full array for correlation
        r_ab = float(xp.corrcoef(self.osc_a.flatten(), self.osc_b.flatten())[0, 1])
        r_bc = float(xp.corrcoef(self.osc_b.flatten(), self.osc_c.flatten())[0, 1])
        r_ca = float(xp.corrcoef(self.osc_c.flatten(), self.osc_a.flatten())[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float(xp.var(self.osc_a) + xp.var(self.osc_b) + xp.var(self.osc_c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


def run_jetson_capture(config: JetsonConfig, output_path: str) -> dict:
    """Run capture on Jetson and save results."""

    print("\n" + "="*60)
    print("JETSON NANO RECEIVER")
    print("="*60)

    print(f"\n  CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  Device: {props['name'].decode()}")

    print(f"\n  Configuration:")
    print(f"    Arrays: {config.n_arrays} × {config.ossicles_per_array} = {config.n_arrays * config.ossicles_per_array} ossicles")
    print(f"    Warmup: {config.warmup_seconds}s")
    print(f"    Capture: {config.capture_seconds}s at {config.sample_rate_hz} Hz")

    # Initialize array
    print(f"\n  Initializing...")
    array = JetsonOssicleArray(config)

    # Warmup
    print(f"  Warming up...")
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < config.warmup_seconds:
        array.step(20)

    # Capture with precise timestamps
    n_samples = int(config.capture_seconds * config.sample_rate_hz)
    print(f"\n  Starting capture of {n_samples} samples...")
    print(f"  START TIMESTAMP: {datetime.utcnow().isoformat()}")

    k_eff_series = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)  # Unix timestamps with high precision

    capture_start = time.time()  # Unix time for synchronization
    perf_start = time.perf_counter()  # High-res for intervals

    for i in range(n_samples):
        array.step(5)
        k_eff_series[i] = array.measure_k_eff()
        timestamps[i] = capture_start + (time.perf_counter() - perf_start)

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - perf_start
            rate = (i + 1) / elapsed
            print(f"    {i + 1}/{n_samples} samples ({elapsed:.1f}s, {rate:.1f} Hz)")

    actual_duration = time.perf_counter() - perf_start
    actual_rate = n_samples / actual_duration

    print(f"\n  END TIMESTAMP: {datetime.utcnow().isoformat()}")
    print(f"  Actual rate: {actual_rate:.1f} Hz")

    # Statistics
    print(f"\n  Statistics:")
    print(f"    Mean k_eff: {np.mean(k_eff_series):.6f}")
    print(f"    Std k_eff:  {np.std(k_eff_series):.6f}")
    print(f"    Min/Max:    {np.min(k_eff_series):.6f} / {np.max(k_eff_series):.6f}")

    # Save results
    results = {
        'device': 'jetson_nano',
        'k_eff': k_eff_series,
        'timestamps': timestamps,
        'capture_start_unix': capture_start,
        'actual_rate': actual_rate,
        'config': {
            'n_arrays': config.n_arrays,
            'ossicles_per_array': config.ossicles_per_array,
            'oscillator_depth': config.oscillator_depth
        }
    }

    np.savez(output_path, **results)
    print(f"\n  Saved to: {output_path}")

    print("\n" + "="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Jetson Nano CIRISArray Receiver')
    parser.add_argument('--output', '-o', default='/tmp/jetson_capture.npz',
                       help='Output file path')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                       help='Capture duration in seconds')
    parser.add_argument('--rate', '-r', type=float, default=100.0,
                       help='Sample rate in Hz')

    args = parser.parse_args()

    config = JetsonConfig(
        capture_seconds=args.duration,
        sample_rate_hz=args.rate
    )

    run_jetson_capture(config, args.output)


if __name__ == "__main__":
    main()
