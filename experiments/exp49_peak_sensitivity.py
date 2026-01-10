#!/usr/bin/env python3
"""
Experiment 49: Peak Sensitivity Measurement
============================================

Key insight: Correlation is near-perfect during convergence phase (0-30s),
then drops to noise in steady state.

Strategy:
1. Reset sensors at beginning of each measurement window
2. Capture only during first 20-30 seconds (peak sensitivity)
3. Reset and repeat for continuous monitoring
4. Use multiple short windows instead of one long capture

This maximizes time spent in high-sensitivity regime.

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

# Optimal measurement window (peak sensitivity)
WINDOW_DURATION = 20.0  # seconds per window
SAMPLE_RATE = 50.0  # Hz


class PeakSensitivitySensor:
    """Sensor optimized for peak sensitivity operation."""

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
        """Reset to random initial state (enter peak sensitivity)."""
        xp = cp if HAS_CUDA else np
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.window_start = time.time()

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

    def time_in_window(self) -> float:
        """Time since last reset."""
        return time.time() - self.window_start

    def in_peak_sensitivity(self) -> bool:
        """Check if still in peak sensitivity window."""
        return self.time_in_window() < WINDOW_DURATION


def run_peak_sensitivity_capture(n_ossicles: int, n_windows: int, output_path: str):
    """Capture multiple peak-sensitivity windows."""

    print(f"\n{'='*60}")
    print(f"PEAK SENSITIVITY CAPTURE")
    print(f"{'='*60}")

    print(f"\n  Ossicles: {n_ossicles}")
    print(f"  Window duration: {WINDOW_DURATION}s")
    print(f"  Number of windows: {n_windows}")
    print(f"  Total capture time: {n_windows * WINDOW_DURATION:.0f}s")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")

    if HAS_CUDA:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name'].decode()
        print(f"  GPU: {device_name}")
    else:
        device_name = "CPU"

    sensor = PeakSensitivitySensor(n_ossicles)

    interval = 1.0 / SAMPLE_RATE
    samples_per_window = int(WINDOW_DURATION * SAMPLE_RATE)

    all_k_eff = []
    all_timestamps = []
    window_starts = []

    print(f"\n  START: {datetime.now(timezone.utc).isoformat()}")

    capture_start = time.time()

    for w in range(n_windows):
        # Reset to enter peak sensitivity
        sensor.reset()
        window_start = time.time()
        window_starts.append(window_start)

        print(f"\n    Window {w+1}/{n_windows} at t={window_start - capture_start:.1f}s")

        window_k_eff = []
        window_timestamps = []

        # Capture during peak sensitivity
        for s in range(samples_per_window):
            sample_start = time.perf_counter()

            sensor.step(5)
            k = sensor.measure_k_eff()

            window_k_eff.append(k)
            window_timestamps.append(time.time())

            elapsed = time.perf_counter() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        all_k_eff.extend(window_k_eff)
        all_timestamps.extend(window_timestamps)

        # Report window stats
        k_arr = np.array(window_k_eff)
        print(f"      k_eff: {np.mean(k_arr):.4f} ± {np.std(k_arr):.4f}")
        print(f"      Range: {np.min(k_arr):.4f} - {np.max(k_arr):.4f}")

    print(f"\n  END: {datetime.now(timezone.utc).isoformat()}")

    np.savez(output_path,
             device=device_name,
             k_eff=np.array(all_k_eff),
             timestamps=np.array(all_timestamps),
             window_starts=np.array(window_starts),
             capture_start=capture_start,
             window_duration=WINDOW_DURATION,
             sample_rate=SAMPLE_RATE,
             n_windows=n_windows)

    print(f"  Saved: {output_path}")


def analyze_peak_correlation(file_a: str, file_b: str):
    """Analyze correlation during peak sensitivity windows."""

    print(f"\n{'='*60}")
    print(f"PEAK SENSITIVITY CORRELATION ANALYSIS")
    print(f"{'='*60}")

    data_a = np.load(file_a)
    data_b = np.load(file_b)

    print(f"\n  Device A: {data_a['device']}")
    print(f"  Device B: {data_b['device']}")
    print(f"  Windows A: {data_a['n_windows']}")
    print(f"  Windows B: {data_b['n_windows']}")

    k_a = data_a['k_eff']
    k_b = data_b['k_eff']
    ts_a = data_a['timestamps']
    ts_b = data_b['timestamps']
    windows_a = data_a['window_starts']
    windows_b = data_b['window_starts']
    window_dur = float(data_a['window_duration'])

    print(f"\n{'='*60}")
    print("WINDOW-BY-WINDOW CORRELATION")
    print(f"{'='*60}")

    correlations = []
    coherences = []

    n_windows = min(len(windows_a), len(windows_b))

    for w in range(n_windows):
        # Find samples in this window
        w_start = max(windows_a[w], windows_b[w])
        w_end = w_start + window_dur

        mask_a = (ts_a >= w_start) & (ts_a < w_end)
        mask_b = (ts_b >= w_start) & (ts_b < w_end)

        if np.sum(mask_a) > 100 and np.sum(mask_b) > 100:
            ka = k_a[mask_a]
            kb = k_b[mask_b]

            # Truncate to same length
            min_len = min(len(ka), len(kb))
            ka = ka[:min_len]
            kb = kb[:min_len]

            # Detrend
            ka_dt = signal.detrend(ka)
            kb_dt = signal.detrend(kb)

            # Correlation
            r = np.corrcoef(ka_dt, kb_dt)[0, 1]
            correlations.append(r)

            # Coherence at key frequencies
            if len(ka_dt) > 50:
                freqs, coh = signal.coherence(ka_dt, kb_dt, fs=SAMPLE_RATE, nperseg=min(64, len(ka_dt)//2))
                mean_coh = np.mean(coh[(freqs > 0.05) & (freqs < 2)])
                coherences.append(mean_coh)
            else:
                coherences.append(0)

            print(f"\n  Window {w+1}:")
            print(f"    Samples: {min_len}")
            print(f"    Detrended correlation: r = {r:.4f}")
            print(f"    Mean coherence (0.05-2 Hz): {coherences[-1]:.4f}")

    if len(correlations) > 0:
        print(f"\n{'='*60}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*60}")

        corr_arr = np.array(correlations)
        coh_arr = np.array(coherences)

        print(f"\n  Mean correlation: {np.mean(corr_arr):.4f} ± {np.std(corr_arr):.4f}")
        print(f"  Mean coherence: {np.mean(coh_arr):.4f} ± {np.std(coh_arr):.4f}")
        print(f"  Correlation range: {np.min(corr_arr):.4f} - {np.max(corr_arr):.4f}")

        # Significance
        from scipy import stats
        t, p = stats.ttest_1samp(corr_arr, 0)
        print(f"\n  T-test (r ≠ 0): t = {t:.2f}, p = {p:.4f}")

        if np.mean(corr_arr) > 0.5 and p < 0.05:
            print("\n  ★★★ STRONG CROSS-DEVICE CORRELATION ★★★")
        elif np.mean(corr_arr) > 0.3 and p < 0.05:
            print("\n  ★★ MODERATE CORRELATION ★★")
        elif p < 0.05:
            print("\n  ★ WEAK BUT SIGNIFICANT ★")
        else:
            print("\n  Not significant")

        # Best frequency band
        print(f"\n{'='*60}")
        print("BEST FREQUENCY BAND")
        print(f"{'='*60}")

        # Combine all windows for spectral analysis
        all_ka = []
        all_kb = []
        for w in range(n_windows):
            w_start = max(windows_a[w], windows_b[w])
            w_end = w_start + window_dur
            mask_a = (ts_a >= w_start) & (ts_a < w_end)
            mask_b = (ts_b >= w_start) & (ts_b < w_end)
            if np.sum(mask_a) > 100 and np.sum(mask_b) > 100:
                ka = signal.detrend(k_a[mask_a][:500])
                kb = signal.detrend(k_b[mask_b][:500])
                all_ka.extend(ka)
                all_kb.extend(kb)

        if len(all_ka) > 200:
            all_ka = np.array(all_ka)
            all_kb = np.array(all_kb)
            freqs, coh = signal.coherence(all_ka, all_kb, fs=SAMPLE_RATE, nperseg=256)

            # Find peak
            valid = (freqs > 0.02) & (freqs < 5)
            peak_idx = np.argmax(coh[valid])
            peak_freq = freqs[valid][peak_idx]
            peak_coh = coh[valid][peak_idx]

            print(f"\n  Peak coherence: {peak_coh:.4f} at {peak_freq:.3f} Hz ({1/peak_freq:.1f}s period)")

            # Top 5 frequencies
            print("\n  Top 5 coherent frequencies:")
            top_idx = np.argsort(coh[valid])[::-1][:5]
            for i, idx in enumerate(top_idx):
                f = freqs[valid][idx]
                c = coh[valid][idx]
                print(f"    {i+1}. {f:.3f} Hz ({1/f:.1f}s): coherence = {c:.3f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Peak Sensitivity Measurement')
    parser.add_argument('--mode', choices=['capture', 'analyze'], required=True)
    parser.add_argument('--ossicles', type=int, default=8192)
    parser.add_argument('--windows', type=int, default=10)
    parser.add_argument('--output', '-o', default='/tmp/peak_sensitivity.npz')
    parser.add_argument('--file-a')
    parser.add_argument('--file-b')

    args = parser.parse_args()

    if args.mode == 'capture':
        run_peak_sensitivity_capture(args.ossicles, args.windows, args.output)
    elif args.mode == 'analyze':
        if not args.file_a or not args.file_b:
            print("ERROR: Need --file-a and --file-b")
            return
        analyze_peak_correlation(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
