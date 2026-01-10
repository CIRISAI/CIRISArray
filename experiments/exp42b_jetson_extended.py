#!/usr/bin/env python3
"""
Experiment 42b: Jetson Extended Capture
=======================================

Extended capture matching 4090 timing.
Run on Jetson: python3 exp42b_jetson_extended.py

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
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


class JetsonSensorArray:
    """Sensor array sized for Jetson."""

    def __init__(self, n_ossicles: int = 1024, depth: int = 32):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total_elements = n_ossicles * depth

        xp = cp if HAS_CUDA else np

        # Initialize with variance
        self.osc_a = xp.random.random(self.total_elements).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total_elements).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total_elements).astype(xp.float32) * 0.5 - 0.25

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
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
        xp = cp if HAS_CUDA else np

        r_ab = float(xp.corrcoef(self.osc_a.flatten(), self.osc_b.flatten())[0, 1])
        r_bc = float(xp.corrcoef(self.osc_b.flatten(), self.osc_c.flatten())[0, 1])
        r_ca = float(xp.corrcoef(self.osc_c.flatten(), self.osc_a.flatten())[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float((xp.var(self.osc_a) + xp.var(self.osc_b) + xp.var(self.osc_c)))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000

    def measure_variance(self) -> float:
        xp = cp if HAS_CUDA else np
        return float(xp.var(self.osc_a))


def run_extended_capture():
    """Run extended capture on Jetson."""

    print("\n" + "="*60)
    print("EXTENDED CROSS-DEVICE CAPTURE (JETSON)")
    print("="*60)

    print(f"\n  CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  Device: {props['name'].decode()}")

    # Short warmup
    print(f"\n  Short warmup (5s)...")
    array = JetsonSensorArray()
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < 5.0:
        array.step(10)

    # Extended capture - 120 seconds at 20 Hz
    duration = 120.0
    target_rate = 20.0
    n_samples = int(duration * target_rate)

    print(f"\n  Capturing {n_samples} samples over {duration}s at {target_rate} Hz...")
    print(f"  START: {datetime.now(timezone.utc).isoformat()}")

    k_eff_series = np.zeros(n_samples)
    variance_series = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)

    capture_start = time.time()
    interval = 1.0 / target_rate

    for i in range(n_samples):
        sample_start = time.perf_counter()

        array.step(5)
        k_eff_series[i] = array.measure_k_eff()
        variance_series[i] = array.measure_variance()
        timestamps[i] = time.time()

        # Rate control
        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 500 == 0:
            total_elapsed = time.time() - capture_start
            actual_rate = (i + 1) / total_elapsed
            print(f"    {i + 1}/{n_samples} ({total_elapsed:.1f}s, {actual_rate:.1f} Hz, var={variance_series[i]:.6f})")

    actual_duration = time.time() - capture_start
    actual_rate = n_samples / actual_duration

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Actual rate: {actual_rate:.2f} Hz")

    print(f"\n  k_eff Statistics:")
    print(f"    Mean: {np.mean(k_eff_series):.6f}")
    print(f"    Std:  {np.std(k_eff_series):.6f}")
    print(f"    Range: {np.min(k_eff_series):.6f} - {np.max(k_eff_series):.6f}")

    output_path = '/tmp/jetson_extended.npz'
    np.savez(output_path,
             device='jetson',
             k_eff=k_eff_series,
             variance=variance_series,
             timestamps=timestamps,
             capture_start_unix=capture_start,
             actual_rate=actual_rate)

    print(f"\n  Saved to: {output_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    run_extended_capture()
