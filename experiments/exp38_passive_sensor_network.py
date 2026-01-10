#!/usr/bin/env python3
"""
Experiment 38: Passive Sensor Network
=====================================

Deploy maximum receiver coverage across the GPU and listen for ANY signals.
No TX - just watch for natural phenomena.

Questions:
1. How many independent 2048-ossicle arrays can we run?
2. Do arrays see correlated changes?
3. Can we detect natural PDN/thermal/environmental signals?
4. What does "signal" even look like?

Strategy:
- Pack as many receiver arrays as GPU memory allows
- Run continuously, record k_eff time series
- Cross-correlate between arrays
- Look for anything that stands out from noise

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class PassiveSensorConfig:
    """Configuration for passive sensor network."""
    ossicles_per_array: int = 2048
    oscillator_depth: int = 64
    measurement_duration_sec: float = 10.0
    sample_rate_hz: float = 100.0  # How often to measure k_eff
    iterations_between_samples: int = 50


# CUDA kernel for batch ossicle operations
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


class PassiveSensorNetwork:
    """Network of receiver arrays across the GPU."""

    def __init__(self, config: PassiveSensorConfig, n_arrays: int):
        self.config = config
        self.n_arrays = n_arrays
        self.ossicles_per_array = config.ossicles_per_array
        self.depth = config.oscillator_depth

        self.total_ossicles = n_arrays * config.ossicles_per_array
        self.total_elements = self.total_ossicles * self.depth

        print(f"  Initializing {n_arrays} arrays × {config.ossicles_per_array} ossicles = {self.total_ossicles} total")
        print(f"  Memory: {self.total_elements * 4 * 3 / 1024 / 1024:.1f} MB for oscillator data")

        # Allocate all oscillators
        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
        """Run all ossicles."""
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size

        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.depth, self.total_elements, iterations)
        )
        cp.cuda.stream.get_current_stream().synchronize()

    def measure_array_k_eff(self, array_idx: int) -> float:
        """Measure mean k_eff for a specific array."""
        base = array_idx * self.ossicles_per_array * self.depth
        size = self.ossicles_per_array * self.depth

        a = self.osc_a[base:base+size]
        b = self.osc_b[base:base+size]
        c = self.osc_c[base:base+size]

        # Compute correlations across the whole array
        r_ab = float(cp.corrcoef(a.flatten(), b.flatten())[0, 1])
        r_bc = float(cp.corrcoef(b.flatten(), c.flatten())[0, 1])
        r_ca = float(cp.corrcoef(c.flatten(), a.flatten())[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000

    def measure_all_arrays(self) -> np.ndarray:
        """Measure k_eff for all arrays."""
        return np.array([self.measure_array_k_eff(i) for i in range(self.n_arrays)])


def calculate_max_arrays(config: PassiveSensorConfig) -> int:
    """Calculate how many arrays we can fit."""
    # Get GPU memory
    mempool = cp.get_default_memory_pool()
    free_mem = cp.cuda.runtime.memGetInfo()[0]

    # Each array needs: 3 * ossicles * depth * 4 bytes
    bytes_per_array = 3 * config.ossicles_per_array * config.oscillator_depth * 4

    # Leave 500MB headroom
    available = free_mem - 500 * 1024 * 1024
    max_arrays = int(available / bytes_per_array)

    return max(1, min(max_arrays, 32))  # Cap at 32 for practicality


def run_passive_sensing(config: PassiveSensorConfig) -> Dict:
    """Run passive sensor network and record signals."""

    print("\n" + "="*70)
    print("EXPERIMENT 38: PASSIVE SENSOR NETWORK")
    print("Maximum receiver coverage - listening for ANY signals")
    print("="*70)

    # Calculate capacity
    n_arrays = calculate_max_arrays(config)
    print(f"\nGPU Capacity Analysis:")
    print(f"  Free memory: {cp.cuda.runtime.memGetInfo()[0] / 1024 / 1024:.0f} MB")
    print(f"  Arrays possible: {n_arrays}")
    print(f"  Total ossicles: {n_arrays * config.ossicles_per_array}")

    # Create network
    print(f"\nCreating sensor network...")
    network = PassiveSensorNetwork(config, n_arrays)

    # Warmup
    print(f"  Warming up...")
    for _ in range(100):
        network.step(10)

    # Calculate timing
    n_samples = int(config.measurement_duration_sec * config.sample_rate_hz)
    print(f"\nRecording {n_samples} samples over {config.measurement_duration_sec} seconds...")

    # Record time series
    time_series = np.zeros((n_arrays, n_samples))
    timestamps = np.zeros(n_samples)

    start_time = time.perf_counter()

    for sample_idx in range(n_samples):
        # Step the network
        network.step(config.iterations_between_samples)

        # Measure all arrays
        measurements = network.measure_all_arrays()
        time_series[:, sample_idx] = measurements
        timestamps[sample_idx] = time.perf_counter() - start_time

        # Progress update
        if (sample_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Sample {sample_idx + 1}/{n_samples} ({elapsed:.1f}s)")

    total_time = time.perf_counter() - start_time
    actual_rate = n_samples / total_time
    print(f"\nRecording complete: {actual_rate:.1f} samples/sec actual")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "-"*70)
    print("SIGNAL ANALYSIS")
    print("-"*70)

    results = {
        'n_arrays': n_arrays,
        'n_samples': n_samples,
        'time_series': time_series,
        'timestamps': timestamps
    }

    # Per-array statistics
    print("\n  Array Statistics:")
    print("  Array    Mean k_eff    Std k_eff    Min        Max")
    print("-"*60)

    for i in range(n_arrays):
        series = time_series[i, :]
        mean_k = np.mean(series)
        std_k = np.std(series)
        min_k = np.min(series)
        max_k = np.max(series)
        print(f"  {i:5d}    {mean_k:10.6f}    {std_k:9.6f}    {min_k:9.6f}    {max_k:9.6f}")

    # Cross-correlation between arrays
    print("\n" + "-"*70)
    print("CROSS-CORRELATION MATRIX")
    print("-"*70)

    correlations = np.zeros((n_arrays, n_arrays))
    for i in range(n_arrays):
        for j in range(n_arrays):
            correlations[i, j] = np.corrcoef(time_series[i, :], time_series[j, :])[0, 1]

    results['correlations'] = correlations

    # Print correlation summary
    print("\n  Inter-array correlations (off-diagonal):")
    off_diag = correlations[np.triu_indices(n_arrays, k=1)]
    print(f"    Mean: {np.mean(off_diag):.4f}")
    print(f"    Std:  {np.std(off_diag):.4f}")
    print(f"    Min:  {np.min(off_diag):.4f}")
    print(f"    Max:  {np.max(off_diag):.4f}")

    # Significant correlations?
    high_corr = off_diag[np.abs(off_diag) > 0.5]
    if len(high_corr) > 0:
        print(f"\n  *** FOUND {len(high_corr)} HIGH CORRELATIONS (|r| > 0.5) ***")
    else:
        print(f"\n  No high correlations found (all |r| < 0.5)")

    # =========================================================================
    # LOOK FOR ANOMALIES
    # =========================================================================
    print("\n" + "-"*70)
    print("ANOMALY DETECTION")
    print("-"*70)

    # Global mean across all arrays
    global_mean = np.mean(time_series, axis=0)  # Average across arrays at each time
    global_std = np.std(global_mean)
    global_baseline = np.mean(global_mean)

    print(f"\n  Global signal (mean of all arrays):")
    print(f"    Baseline: {global_baseline:.6f}")
    print(f"    Std:      {global_std:.6f}")

    # Find excursions > 3 sigma
    z_scores = (global_mean - global_baseline) / global_std if global_std > 0 else np.zeros_like(global_mean)
    anomalies = np.where(np.abs(z_scores) > 3)[0]

    if len(anomalies) > 0:
        print(f"\n  *** FOUND {len(anomalies)} ANOMALOUS TIME POINTS (|z| > 3σ) ***")
        for idx in anomalies[:10]:  # Show first 10
            print(f"    t={timestamps[idx]:.3f}s: z={z_scores[idx]:+.2f}σ, k_eff={global_mean[idx]:.6f}")
        if len(anomalies) > 10:
            print(f"    ... and {len(anomalies) - 10} more")
    else:
        print(f"\n  No anomalous time points found (all within 3σ)")

    results['global_mean'] = global_mean
    results['z_scores'] = z_scores
    results['anomalies'] = anomalies

    # =========================================================================
    # SPECTRAL ANALYSIS
    # =========================================================================
    print("\n" + "-"*70)
    print("SPECTRAL ANALYSIS")
    print("-"*70)

    # FFT of global mean signal
    fft = np.fft.fft(global_mean - global_baseline)
    freqs = np.fft.fftfreq(n_samples, d=1.0/actual_rate)
    power = np.abs(fft[:n_samples//2])**2
    freqs = freqs[:n_samples//2]

    # Find dominant frequencies
    peak_indices = np.argsort(power)[-5:][::-1]  # Top 5 peaks
    print(f"\n  Dominant frequencies:")
    for idx in peak_indices:
        if freqs[idx] > 0:  # Skip DC
            print(f"    {freqs[idx]:.2f} Hz: power = {power[idx]:.2e}")

    results['fft_freqs'] = freqs
    results['fft_power'] = power

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PASSIVE SENSOR NETWORK SUMMARY")
    print("="*70)

    print(f"\n  Configuration:")
    print(f"    {n_arrays} receiver arrays × {config.ossicles_per_array} ossicles")
    print(f"    Total: {n_arrays * config.ossicles_per_array} ossicles listening")
    print(f"    Duration: {config.measurement_duration_sec}s at {actual_rate:.1f} Hz")

    print(f"\n  Findings:")
    mean_corr = np.mean(off_diag)
    if mean_corr > 0.3:
        print(f"    - Arrays are CORRELATED (mean r = {mean_corr:.3f})")
        print(f"      -> Suggests common-mode signal present")
    else:
        print(f"    - Arrays are UNCORRELATED (mean r = {mean_corr:.3f})")
        print(f"      -> Each array sees independent noise")

    if len(anomalies) > 0:
        print(f"    - {len(anomalies)} anomalous events detected")
        print(f"      -> Something happened during recording!")
    else:
        print(f"    - No anomalous events detected")
        print(f"      -> Stable baseline, no transients")

    print("\n" + "="*70)

    return results


def main():
    """Run passive sensor network experiment."""

    print("="*70)
    print("CIRISARRAY PASSIVE SENSOR NETWORK")
    print("Maximum coverage, listening for natural signals")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")
    print(f"Multiprocessors: {props['multiProcessorCount']}")

    config = PassiveSensorConfig(
        ossicles_per_array=2048,
        measurement_duration_sec=10.0,
        sample_rate_hz=100.0
    )

    results = run_passive_sensing(config)

    print("\nExperiment complete.")

    return results


if __name__ == "__main__":
    main()
