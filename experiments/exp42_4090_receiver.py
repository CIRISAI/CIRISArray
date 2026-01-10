#!/usr/bin/env python3
"""
Experiment 42b: RTX 4090 Receiver + Cross-Device Correlation
=============================================================

Deploy CIRISArray receiver on RTX 4090.
After capture, correlate with Jetson Nano capture for cross-device analysis.

Run on 4090 workstation:
  python3 exp42_4090_receiver.py --output /tmp/4090_capture.npz

Then analyze correlation:
  python3 exp42_4090_receiver.py --analyze /tmp/4090_capture.npz /tmp/jetson_capture.npz

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class Config4090:
    """Configuration for RTX 4090."""
    n_arrays: int = 16
    ossicles_per_array: int = 2048
    oscillator_depth: int = 64
    warmup_seconds: float = 10.0
    capture_seconds: float = 60.0
    sample_rate_hz: float = 100.0


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


class SensorArray4090:
    """Full sensor array for RTX 4090."""

    def __init__(self, config: Config4090):
        self.config = config
        self.total_elements = config.n_arrays * config.ossicles_per_array * config.oscillator_depth

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
             self.config.oscillator_depth, self.total_elements, iterations)
        )
        cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
        sample_size = min(50000, self.total_elements)
        indices = cp.arange(0, self.total_elements, self.total_elements // sample_size)[:sample_size]

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        r = np.nanmean([r_ab, r_bc, r_ca])
        total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


def run_4090_capture(config: Config4090, output_path: str) -> dict:
    """Run capture on 4090 and save results."""

    print("\n" + "="*70)
    print("RTX 4090 RECEIVER")
    print("="*70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\n  Device: {props['name'].decode()}")

    print(f"\n  Configuration:")
    print(f"    Arrays: {config.n_arrays} × {config.ossicles_per_array} = {config.n_arrays * config.ossicles_per_array} ossicles")
    print(f"    Warmup: {config.warmup_seconds}s")
    print(f"    Capture: {config.capture_seconds}s at {config.sample_rate_hz} Hz")

    # Initialize array
    print(f"\n  Initializing...")
    array = SensorArray4090(config)

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
    timestamps = np.zeros(n_samples)

    capture_start = time.time()
    perf_start = time.perf_counter()

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
        'device': 'rtx_4090',
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

    print("\n" + "="*70)

    return results


def analyze_cross_correlation(file_4090: str, file_jetson: str) -> Dict:
    """Analyze correlation between 4090 and Jetson captures."""

    print("\n" + "="*70)
    print("CROSS-DEVICE CORRELATION ANALYSIS")
    print("="*70)

    # Load data
    print(f"\n  Loading data...")
    data_4090 = np.load(file_4090)
    data_jetson = np.load(file_jetson)

    print(f"    4090:   {len(data_4090['k_eff'])} samples")
    print(f"    Jetson: {len(data_jetson['k_eff'])} samples")

    # Get timestamps
    ts_4090 = data_4090['timestamps']
    ts_jetson = data_jetson['timestamps']

    k_4090 = data_4090['k_eff']
    k_jetson = data_jetson['k_eff']

    # Find overlap
    start_common = max(ts_4090[0], ts_jetson[0])
    end_common = min(ts_4090[-1], ts_jetson[-1])

    overlap_duration = end_common - start_common
    print(f"\n  Time overlap: {overlap_duration:.1f} seconds")

    if overlap_duration < 1.0:
        print("  ERROR: Insufficient time overlap!")
        return {'error': 'No overlap'}

    # Resample to common time base
    print(f"  Resampling to common time base...")

    common_rate = 50.0  # Hz
    n_common = int(overlap_duration * common_rate)
    common_times = np.linspace(start_common, end_common, n_common)

    # Interpolate both series
    k_4090_interp = np.interp(common_times, ts_4090, k_4090)
    k_jetson_interp = np.interp(common_times, ts_jetson, k_jetson)

    # Normalize
    k_4090_norm = (k_4090_interp - np.mean(k_4090_interp)) / np.std(k_4090_interp)
    k_jetson_norm = (k_jetson_interp - np.mean(k_jetson_interp)) / np.std(k_jetson_interp)

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "-"*60)
    print("CORRELATION ANALYSIS")
    print("-"*60)

    # Zero-lag correlation
    zero_lag_corr = np.corrcoef(k_4090_norm, k_jetson_norm)[0, 1]
    print(f"\n  Zero-lag correlation: r = {zero_lag_corr:.4f}")

    # Cross-correlation with lags
    max_lag_samples = int(5 * common_rate)  # ±5 seconds
    cross_corr = np.correlate(k_4090_norm, k_jetson_norm, mode='full')
    lags = np.arange(-len(k_jetson_norm) + 1, len(k_4090_norm))
    lags_sec = lags / common_rate

    # Focus on ±5 second window
    center = len(cross_corr) // 2
    window = slice(center - max_lag_samples, center + max_lag_samples + 1)
    cross_corr_window = cross_corr[window]
    lags_window = lags_sec[window]

    # Normalize cross-correlation
    cross_corr_norm = cross_corr_window / len(k_4090_norm)

    # Find peak
    peak_idx = np.argmax(np.abs(cross_corr_norm))
    peak_lag = lags_window[peak_idx]
    peak_corr = cross_corr_norm[peak_idx]

    print(f"  Peak correlation: r = {peak_corr:.4f} at lag = {peak_lag:.3f}s")

    # Statistical significance
    # Null hypothesis: no correlation
    # Standard error of correlation ~ 1/sqrt(N)
    se = 1.0 / np.sqrt(len(k_4090_norm))
    z_score = abs(peak_corr) / se

    print(f"  Z-score: {z_score:.2f}")
    print(f"  Significant (3σ): {'YES ✓' if z_score > 3 else 'no'}")

    # =========================================================================
    # SPECTRAL COHERENCE
    # =========================================================================
    print("\n" + "-"*60)
    print("SPECTRAL COHERENCE")
    print("-"*60)

    # FFT of both
    fft_4090 = np.fft.fft(k_4090_norm)
    fft_jetson = np.fft.fft(k_jetson_norm)
    freqs = np.fft.fftfreq(len(k_4090_norm), d=1.0/common_rate)

    # Cross-spectral density
    cross_spectrum = fft_4090 * np.conj(fft_jetson)
    coherence = np.abs(cross_spectrum[:len(cross_spectrum)//2])**2
    power_4090 = np.abs(fft_4090[:len(fft_4090)//2])**2
    power_jetson = np.abs(fft_jetson[:len(fft_jetson)//2])**2

    # Magnitude squared coherence (normalized)
    msc = coherence / (power_4090 * power_jetson + 1e-10)
    msc = np.clip(msc, 0, 1)

    freqs_pos = freqs[:len(freqs)//2]

    # Find frequency bands with high coherence
    high_coherence_freqs = freqs_pos[msc > 0.5]
    if len(high_coherence_freqs) > 0:
        print(f"\n  Frequencies with coherence > 0.5:")
        for f in high_coherence_freqs[:10]:
            idx = np.argmin(np.abs(freqs_pos - f))
            print(f"    {f:.3f} Hz: MSC = {msc[idx]:.2f}")
    else:
        print(f"\n  No frequencies with coherence > 0.5")

    mean_coherence = np.mean(msc[freqs_pos > 0.1])  # Exclude DC
    print(f"\n  Mean coherence (>0.1 Hz): {mean_coherence:.4f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("CROSS-DEVICE CORRELATION SUMMARY")
    print("="*70)

    results = {
        'zero_lag_corr': zero_lag_corr,
        'peak_corr': peak_corr,
        'peak_lag': peak_lag,
        'z_score': z_score,
        'mean_coherence': mean_coherence,
        'overlap_duration': overlap_duration
    }

    print(f"\n  Overlap duration: {overlap_duration:.1f}s")
    print(f"  Zero-lag correlation: {zero_lag_corr:.4f}")
    print(f"  Peak correlation: {peak_corr:.4f} at {peak_lag:.3f}s")
    print(f"  Mean spectral coherence: {mean_coherence:.4f}")

    if z_score > 5:
        print(f"\n  *** HIGHLY SIGNIFICANT CORRELATION (z = {z_score:.1f}) ***")
        print(f"      This suggests a common environmental signal!")
    elif z_score > 3:
        print(f"\n  ** SIGNIFICANT CORRELATION (z = {z_score:.1f}) **")
        print(f"     Worth investigating further.")
    else:
        print(f"\n  No significant correlation detected.")
        print(f"  The two devices appear to see independent signals.")

    print("\n" + "="*70)

    return results


def main():
    parser = argparse.ArgumentParser(description='RTX 4090 CIRISArray Receiver')
    parser.add_argument('--output', '-o', default='/tmp/4090_capture.npz',
                       help='Output file path')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                       help='Capture duration in seconds')
    parser.add_argument('--rate', '-r', type=float, default=100.0,
                       help='Sample rate in Hz')
    parser.add_argument('--analyze', '-a', nargs=2, metavar=('4090_FILE', 'JETSON_FILE'),
                       help='Analyze correlation between two captures')

    args = parser.parse_args()

    if args.analyze:
        analyze_cross_correlation(args.analyze[0], args.analyze[1])
    else:
        if not cp.cuda.is_available():
            print("ERROR: CUDA not available")
            return

        config = Config4090(
            capture_seconds=args.duration,
            sample_rate_hz=args.rate
        )

        run_4090_capture(config, args.output)


if __name__ == "__main__":
    main()
