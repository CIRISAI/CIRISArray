#!/usr/bin/env python3
"""
Experiment 45: Transient Mode Cross-Device Correlation
======================================================

Re-test Jetson + 4090 correlation using proper transient mode operation.

Previous test (exp42) failed because both devices converged to fixed points.
Now we operate in transient mode with continuous noise injection.

Hypothesis: If entropy waves are real and propagate through environment,
both devices should see correlated fluctuations.

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
    """Sensor operating in transient mode with noise injection."""

    def __init__(self, n_ossicles: int, depth: int, noise_amplitude: float = 0.01):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth
        self.noise_amplitude = noise_amplitude

        xp = cp if HAS_CUDA else np

        # Initialize random (transient state)
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 5):
        """Step with noise injection to maintain transient state."""
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

    def measure_k_eff(self) -> float:
        """Measure effective coupling."""
        xp = cp if HAS_CUDA else np

        # Subsample
        sample_size = min(10000, self.total)
        if HAS_CUDA:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(xp.corrcoef(a, b)[0, 1])
        r_bc = float(xp.corrcoef(b, c)[0, 1])
        r_ca = float(xp.corrcoef(c, a)[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float(xp.var(a) + xp.var(b) + xp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


def run_transient_capture(device_name: str, n_ossicles: int, duration_sec: float,
                          sample_rate_hz: float, output_path: str):
    """Run transient mode capture."""

    print(f"\n{'='*60}")
    print(f"TRANSIENT MODE CAPTURE: {device_name}")
    print(f"{'='*60}")

    print(f"\n  CUDA: {HAS_CUDA}")
    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  GPU: {props['name'].decode()}")

    print(f"  Ossicles: {n_ossicles}")
    print(f"  Duration: {duration_sec}s at {sample_rate_hz} Hz")
    print(f"  Mode: TRANSIENT (no warmup, continuous noise)")

    # Create sensor
    sensor = TransientSensor(n_ossicles, depth=64, noise_amplitude=0.01)

    # Capture
    n_samples = int(duration_sec * sample_rate_hz)
    interval = 1.0 / sample_rate_hz

    k_eff_series = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)

    print(f"\n  Starting capture...")
    print(f"  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    for i in range(n_samples):
        sample_start = time.perf_counter()

        sensor.step(5)
        k_eff_series[i] = sensor.measure_k_eff()
        timestamps[i] = time.time()

        # Rate control
        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 500 == 0:
            total_elapsed = time.time() - capture_start
            rate = (i + 1) / total_elapsed
            print(f"    {i + 1}/{n_samples} ({total_elapsed:.1f}s, {rate:.1f} Hz)")

    actual_duration = time.time() - capture_start
    actual_rate = n_samples / actual_duration

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Actual rate: {actual_rate:.1f} Hz")

    # Statistics
    print(f"\n  Statistics:")
    print(f"    Mean: {np.mean(k_eff_series):.6f}")
    print(f"    Std:  {np.std(k_eff_series):.6f}")
    print(f"    Range: {np.min(k_eff_series):.6f} - {np.max(k_eff_series):.6f}")

    # Verify transient mode (should have variance)
    if np.std(k_eff_series) < 0.001:
        print(f"\n  WARNING: Low variance - may have converged!")
    else:
        print(f"\n  ✓ Transient mode confirmed (σ > 0.001)")

    # Save
    np.savez(output_path,
             device=device_name,
             k_eff=k_eff_series,
             timestamps=timestamps,
             capture_start=capture_start,
             actual_rate=actual_rate,
             n_ossicles=n_ossicles,
             mode='transient')

    print(f"\n  Saved: {output_path}")
    print(f"{'='*60}")

    return {
        'k_eff': k_eff_series,
        'timestamps': timestamps,
        'mean': np.mean(k_eff_series),
        'std': np.std(k_eff_series)
    }


def analyze_correlation(file_a: str, file_b: str):
    """Analyze cross-device correlation."""

    print(f"\n{'='*60}")
    print("CROSS-DEVICE CORRELATION ANALYSIS (TRANSIENT MODE)")
    print(f"{'='*60}")

    # Load data
    data_a = np.load(file_a)
    data_b = np.load(file_b)

    print(f"\n  Device A: {data_a['device']}")
    print(f"    Samples: {len(data_a['k_eff'])}, σ = {np.std(data_a['k_eff']):.6f}")
    print(f"  Device B: {data_b['device']}")
    print(f"    Samples: {len(data_b['k_eff'])}, σ = {np.std(data_b['k_eff']):.6f}")

    # Check transient mode
    if np.std(data_a['k_eff']) < 0.001 or np.std(data_b['k_eff']) < 0.001:
        print(f"\n  ERROR: One or both devices not in transient mode!")
        return None

    # Find time overlap
    ts_a, ts_b = data_a['timestamps'], data_b['timestamps']
    start_common = max(ts_a[0], ts_b[0])
    end_common = min(ts_a[-1], ts_b[-1])
    overlap = end_common - start_common

    print(f"\n  Time overlap: {overlap:.1f}s")

    if overlap < 5:
        print(f"  ERROR: Insufficient overlap!")
        return None

    # Resample to common time base (10 Hz)
    common_rate = 10.0
    n_common = int(overlap * common_rate)
    common_times = np.linspace(start_common, end_common, n_common)

    k_a = np.interp(common_times, ts_a, data_a['k_eff'])
    k_b = np.interp(common_times, ts_b, data_b['k_eff'])

    # Normalize
    k_a_norm = (k_a - np.mean(k_a)) / np.std(k_a)
    k_b_norm = (k_b - np.mean(k_b)) / np.std(k_b)

    # Correlation analysis
    print(f"\n{'-'*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'-'*60}")

    # Zero-lag correlation
    r_zero = np.corrcoef(k_a_norm, k_b_norm)[0, 1]
    z_score = abs(r_zero) * np.sqrt(n_common)

    print(f"\n  Zero-lag correlation: r = {r_zero:.4f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Significant (3σ): {'YES ★' if z_score > 3 else 'no'}")

    # Cross-correlation with lags
    max_lag = int(10 * common_rate)  # ±10 seconds
    cross_corr = np.correlate(k_a_norm, k_b_norm, mode='full')
    cross_corr = cross_corr / n_common

    center = len(cross_corr) // 2
    window = slice(center - max_lag, center + max_lag + 1)
    lags = np.arange(-max_lag, max_lag + 1) / common_rate

    peak_idx = np.argmax(np.abs(cross_corr[window]))
    peak_lag = lags[peak_idx]
    peak_r = cross_corr[window][peak_idx]

    print(f"\n  Peak correlation: r = {peak_r:.4f} at lag = {peak_lag:.2f}s")

    # Spectral coherence
    print(f"\n{'-'*60}")
    print("SPECTRAL COHERENCE")
    print(f"{'-'*60}")

    fft_a = np.fft.fft(k_a_norm)
    fft_b = np.fft.fft(k_b_norm)
    freqs = np.fft.fftfreq(n_common, d=1/common_rate)

    cross = fft_a * np.conj(fft_b)
    psd_a = np.abs(fft_a)**2
    psd_b = np.abs(fft_b)**2

    coherence = np.abs(cross)**2 / (psd_a * psd_b + 1e-10)
    coherence = np.clip(coherence[:n_common//2], 0, 1)
    freqs_pos = freqs[:n_common//2]

    # Find coherent frequencies
    high_coh = np.where(coherence > 0.5)[0]
    if len(high_coh) > 0:
        print(f"\n  Frequencies with coherence > 0.5:")
        for idx in high_coh[:5]:
            if freqs_pos[idx] > 0.1:
                print(f"    {freqs_pos[idx]:.3f} Hz: coherence = {coherence[idx]:.2f}")
    else:
        print(f"\n  No high-coherence frequencies found")

    mean_coh = np.mean(coherence[(freqs_pos > 0.1) & (freqs_pos < 4)])
    print(f"\n  Mean coherence (0.1-4 Hz): {mean_coh:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    result = {
        'zero_lag_r': r_zero,
        'z_score': z_score,
        'peak_r': peak_r,
        'peak_lag': peak_lag,
        'mean_coherence': mean_coh,
        'overlap': overlap,
        'significant': z_score > 3
    }

    print(f"\n  Zero-lag correlation: r = {r_zero:.4f} (z = {z_score:.1f})")
    print(f"  Peak correlation: r = {peak_r:.4f} at {peak_lag:.2f}s")
    print(f"  Mean spectral coherence: {mean_coh:.4f}")

    if z_score > 5:
        print(f"\n  ★★★ HIGHLY SIGNIFICANT CORRELATION ★★★")
        print(f"      Both devices see correlated entropy waves!")
    elif z_score > 3:
        print(f"\n  ★★ SIGNIFICANT CORRELATION ★★")
    else:
        print(f"\n  No significant correlation.")
        print(f"  Devices appear to measure independent signals.")

    print(f"\n{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Transient Mode Cross-Device Test')
    parser.add_argument('--mode', choices=['capture', 'analyze'],
                       default='capture', help='Mode')
    parser.add_argument('--device', default='auto',
                       help='Device name (auto-detect if not specified)')
    parser.add_argument('--ossicles', type=int, default=16384,
                       help='Number of ossicles')
    parser.add_argument('--duration', type=float, default=120.0,
                       help='Capture duration')
    parser.add_argument('--rate', type=float, default=20.0,
                       help='Sample rate')
    parser.add_argument('--output', '-o', default='/tmp/transient_capture.npz',
                       help='Output file')
    parser.add_argument('--file-a', help='First capture file (for analyze)')
    parser.add_argument('--file-b', help='Second capture file (for analyze)')

    args = parser.parse_args()

    if args.mode == 'capture':
        # Auto-detect device
        if args.device == 'auto':
            if HAS_CUDA:
                props = cp.cuda.runtime.getDeviceProperties(0)
                device_name = props['name'].decode().replace(' ', '_')
            else:
                device_name = 'cpu'
        else:
            device_name = args.device

        run_transient_capture(
            device_name=device_name,
            n_ossicles=args.ossicles,
            duration_sec=args.duration,
            sample_rate_hz=args.rate,
            output_path=args.output
        )

    elif args.mode == 'analyze':
        if not args.file_a or not args.file_b:
            print("ERROR: Need --file-a and --file-b for analysis")
            return
        analyze_correlation(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
