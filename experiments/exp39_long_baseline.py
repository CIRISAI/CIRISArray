#!/usr/bin/env python3
"""
Experiment 39: Long Baseline Recording
======================================

Extended recording after warmup to look for real signals.
- Long warmup period to reach steady state
- Extended recording to capture rare events
- Focus on deviations from the correlated baseline

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class LongBaselineConfig:
    """Configuration for long baseline recording."""
    ossicles_per_array: int = 2048
    n_arrays: int = 16  # Fewer arrays, faster sampling
    oscillator_depth: int = 64
    warmup_seconds: float = 30.0
    recording_seconds: float = 60.0
    sample_rate_hz: float = 50.0
    iterations_between_samples: int = 20


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


class SensorNetwork:
    """Optimized sensor network."""

    def __init__(self, config: LongBaselineConfig):
        self.config = config
        self.n_arrays = config.n_arrays
        self.ossicles_per_array = config.ossicles_per_array
        self.depth = config.oscillator_depth

        self.total_ossicles = config.n_arrays * config.ossicles_per_array
        self.total_elements = self.total_ossicles * self.depth

        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1

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
        cp.cuda.stream.get_current_stream().synchronize()

    def measure_global_k_eff(self) -> float:
        """Fast global k_eff measurement."""
        # Sample subset for speed
        sample_size = min(100000, self.total_elements)
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000

    def measure_variance(self) -> float:
        """Measure total variance (entropy proxy)."""
        sample_size = min(100000, self.total_elements)
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)
        return float(cp.var(self.osc_a[indices]) + cp.var(self.osc_b[indices]) + cp.var(self.osc_c[indices]))


def run_long_baseline(config: LongBaselineConfig) -> Dict:
    """Run extended baseline recording."""

    print("\n" + "="*70)
    print("EXPERIMENT 39: LONG BASELINE RECORDING")
    print("="*70)

    print(f"\n  Configuration:")
    print(f"    {config.n_arrays} arrays × {config.ossicles_per_array} ossicles = {config.n_arrays * config.ossicles_per_array}")
    print(f"    Warmup: {config.warmup_seconds}s")
    print(f"    Recording: {config.recording_seconds}s at {config.sample_rate_hz} Hz")

    # Create network
    print(f"\n  Initializing network...")
    network = SensorNetwork(config)

    # Extended warmup
    print(f"\n  Warming up for {config.warmup_seconds}s...")
    warmup_start = time.perf_counter()
    warmup_samples = int(config.warmup_seconds * 10)  # 10 Hz during warmup

    warmup_k = []
    for i in range(warmup_samples):
        network.step(50)
        if i % 10 == 0:
            warmup_k.append(network.measure_global_k_eff())

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - warmup_start
            print(f"    Warmup: {elapsed:.1f}s / {config.warmup_seconds}s")

    # Check warmup stability
    warmup_k = np.array(warmup_k)
    warmup_drift = warmup_k[-1] - warmup_k[0]
    warmup_std = np.std(warmup_k[-10:])  # Last 10 samples
    print(f"\n  Warmup complete:")
    print(f"    Total drift: {warmup_drift:+.6f}")
    print(f"    Final std: {warmup_std:.6f}")

    # Main recording
    print(f"\n  Recording for {config.recording_seconds}s...")
    n_samples = int(config.recording_seconds * config.sample_rate_hz)

    k_eff_series = np.zeros(n_samples)
    variance_series = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)

    record_start = time.perf_counter()

    for i in range(n_samples):
        network.step(config.iterations_between_samples)

        k_eff_series[i] = network.measure_global_k_eff()
        variance_series[i] = network.measure_variance()
        timestamps[i] = time.perf_counter() - record_start

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - record_start
            rate = (i + 1) / elapsed
            print(f"    {i + 1}/{n_samples} samples ({elapsed:.1f}s, {rate:.1f} Hz)")

    actual_duration = time.perf_counter() - record_start
    actual_rate = n_samples / actual_duration
    print(f"\n  Recording complete: {actual_rate:.1f} Hz actual")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "-"*70)
    print("BASELINE ANALYSIS")
    print("-"*70)

    results = {
        'k_eff': k_eff_series,
        'variance': variance_series,
        'timestamps': timestamps
    }

    # Basic stats
    k_mean = np.mean(k_eff_series)
    k_std = np.std(k_eff_series)
    k_min = np.min(k_eff_series)
    k_max = np.max(k_eff_series)

    print(f"\n  k_eff statistics:")
    print(f"    Mean: {k_mean:.6f}")
    print(f"    Std:  {k_std:.6f}")
    print(f"    Min:  {k_min:.6f}")
    print(f"    Max:  {k_max:.6f}")
    print(f"    Range: {k_max - k_min:.6f} ({(k_max - k_min) / k_std:.1f}σ)")

    # Drift analysis
    first_quarter = np.mean(k_eff_series[:n_samples//4])
    last_quarter = np.mean(k_eff_series[-n_samples//4:])
    drift = last_quarter - first_quarter
    print(f"\n  Drift (first→last quarter): {drift:+.6f} ({drift/k_std:+.2f}σ)")

    # Anomaly detection (using rolling baseline)
    window_size = 50
    rolling_mean = np.convolve(k_eff_series, np.ones(window_size)/window_size, mode='valid')
    rolling_std = np.array([np.std(k_eff_series[max(0,i-window_size):i+1]) for i in range(len(k_eff_series))])

    # Z-scores relative to local baseline
    z_scores = np.zeros(n_samples)
    for i in range(window_size, n_samples):
        local_mean = np.mean(k_eff_series[i-window_size:i])
        local_std = np.std(k_eff_series[i-window_size:i])
        if local_std > 0:
            z_scores[i] = (k_eff_series[i] - local_mean) / local_std

    anomalies = np.where(np.abs(z_scores) > 4)[0]  # 4σ threshold

    print(f"\n  Anomalies (|z| > 4σ relative to local baseline):")
    if len(anomalies) > 0:
        print(f"    Found: {len(anomalies)} events")
        for idx in anomalies[:10]:
            print(f"      t={timestamps[idx]:.2f}s: z={z_scores[idx]:+.2f}σ, k={k_eff_series[idx]:.6f}")
        if len(anomalies) > 10:
            print(f"      ... and {len(anomalies) - 10} more")
    else:
        print(f"    None found - stable baseline!")

    results['z_scores'] = z_scores
    results['anomalies'] = anomalies

    # Spectral analysis
    print("\n" + "-"*70)
    print("SPECTRAL CONTENT")
    print("-"*70)

    # Detrend before FFT
    detrended = k_eff_series - np.linspace(k_eff_series[0], k_eff_series[-1], n_samples)
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n_samples, d=1.0/actual_rate)
    power = np.abs(fft[:n_samples//2])**2
    freqs = freqs[:n_samples//2]

    # Normalize power
    total_power = np.sum(power[1:])  # Exclude DC
    if total_power > 0:
        rel_power = power / total_power

    # Find peaks
    peak_indices = np.argsort(power[1:])[-5:][::-1] + 1  # Skip DC
    print(f"\n  Dominant frequencies:")
    for idx in peak_indices:
        pct = 100 * power[idx] / total_power if total_power > 0 else 0
        print(f"    {freqs[idx]:.3f} Hz: {pct:.1f}% of variance")

    # Check for periodic signals
    print(f"\n  Periodicity check:")
    autocorr = np.correlate(detrended, detrended, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find first significant peak after lag 0
    for lag in range(10, len(autocorr)//2):
        if autocorr[lag] > 0.3:  # Significant correlation
            period = lag / actual_rate
            print(f"    Periodic signal detected: period ≈ {period:.2f}s ({1/period:.2f} Hz)")
            break
    else:
        print(f"    No periodic signals detected")

    results['fft_freqs'] = freqs
    results['fft_power'] = power

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("LONG BASELINE SUMMARY")
    print("="*70)

    print(f"\n  Total ossicles: {config.n_arrays * config.ossicles_per_array}")
    print(f"  Recording duration: {actual_duration:.1f}s")
    print(f"  Samples collected: {n_samples}")

    print(f"\n  Signal characteristics:")
    print(f"    Baseline k_eff: {k_mean:.6f} ± {k_std:.6f}")
    print(f"    Drift: {drift:+.2f}σ over {actual_duration:.0f}s")
    print(f"    Anomalies: {len(anomalies)} events at 4σ threshold")

    if len(anomalies) == 0:
        print(f"\n  CONCLUSION: Stable baseline, no transient signals detected")
        print(f"              The array is seeing steady-state GPU operation")
    else:
        print(f"\n  CONCLUSION: {len(anomalies)} potential events detected!")
        print(f"              Further investigation needed")

    print("\n" + "="*70)

    return results


def main():
    """Run long baseline experiment."""

    print("="*70)
    print("CIRISARRAY LONG BASELINE SENSOR")
    print("Extended recording after warmup stabilization")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = LongBaselineConfig(
        n_arrays=16,
        ossicles_per_array=2048,
        warmup_seconds=30.0,
        recording_seconds=60.0,
        sample_rate_hz=50.0
    )

    results = run_long_baseline(config)

    return results


if __name__ == "__main__":
    main()
