#!/usr/bin/env python3
"""
Experiment 42b: Longer Cross-Device Capture
============================================

Longer capture with less warmup to catch some variance.
Run both devices for 2 minutes at lower rate.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import time
from datetime import datetime, timezone

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void batch_ossicle_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int depth, int total_elements, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float a = osc_a[idx];
    float b = osc_b[idx];
    float c = osc_c[idx];

    for (int i = 0; i < iterations; i++) {
        float da = coupling_ab * (b - a) + coupling_ca * (c - a);
        float db = coupling_ab * (a - b) + coupling_bc * (c - b);
        float dc = coupling_bc * (b - c) + coupling_ca * (a - c);

        a += da;
        b += db;
        c += dc;

        a = fmaxf(-10.0f, fminf(10.0f, a));
        b = fmaxf(-10.0f, fminf(10.0f, b));
        c = fmaxf(-10.0f, fminf(10.0f, c));
    }

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}
''', 'batch_ossicle_step')


class FastSensorArray:
    """Sensor array with variance-capturing design."""

    def __init__(self, n_ossicles: int = 32768, depth: int = 64):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total_elements = n_ossicles * depth

        # Initialize with MORE variance
        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.depth, self.total_elements, iterations)
        )

    def measure_k_eff(self) -> float:
        """Fast k_eff with subsampling."""
        sample_size = 10000
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r = float(cp.corrcoef(a, b)[0, 1])
        r = 0 if np.isnan(r) else r

        var = float(cp.var(a) + cp.var(b)) / 2
        x = min(var, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000

    def measure_variance(self) -> float:
        """Measure total variance (entropy proxy)."""
        sample_size = 10000
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)
        return float(cp.var(self.osc_a[indices]))


def run_extended_capture():
    """Run extended capture with variance tracking."""

    print("\n" + "="*70)
    print("EXTENDED CROSS-DEVICE CAPTURE (4090)")
    print("="*70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\n  Device: {props['name'].decode()}")

    # Short warmup
    print(f"\n  Short warmup (5s)...")
    array = FastSensorArray()
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < 5.0:
        array.step(10)

    # Extended capture - 120 seconds at 20 Hz (controllable rate)
    duration = 120.0  # 2 minutes
    target_rate = 20.0  # Lower rate for sync
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

        # Do work
        array.step(5)
        k_eff_series[i] = array.measure_k_eff()
        variance_series[i] = array.measure_variance()
        timestamps[i] = time.time()

        # Rate control - wait if needed
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

    # Statistics
    print(f"\n  k_eff Statistics:")
    print(f"    Mean: {np.mean(k_eff_series):.6f}")
    print(f"    Std:  {np.std(k_eff_series):.6f}")
    print(f"    Range: {np.min(k_eff_series):.6f} - {np.max(k_eff_series):.6f}")

    print(f"\n  Variance Statistics:")
    print(f"    Mean: {np.mean(variance_series):.6f}")
    print(f"    Std:  {np.std(variance_series):.6f}")

    # Save
    output_path = '/tmp/4090_extended.npz'
    np.savez(output_path,
             device='rtx_4090',
             k_eff=k_eff_series,
             variance=variance_series,
             timestamps=timestamps,
             capture_start_unix=capture_start,
             actual_rate=actual_rate)

    print(f"\n  Saved to: {output_path}")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_extended_capture()
