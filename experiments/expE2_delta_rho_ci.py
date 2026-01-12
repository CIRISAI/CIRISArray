#!/usr/bin/env python3
"""
Experiment E2: Δρ Confidence Interval
=====================================

Gap: Leading indicator Δρ=0.452 from C4 has no error bars.

Protocol:
- Run C4 leading indicator detection 30 times
- Record ρ when spatial_variance first spikes (warning)
- Record ρ when collapse occurs (k_eff < threshold)
- Compute Δρ = ρ_collapse - ρ_warning for each trial
- Report mean, std, 95% CI

Success criteria: 95% CI for Δρ that doesn't include zero

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import time
import sys
import os
import threading
from datetime import datetime, timezone
from scipy import signal, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from strain_gauge import StrainGauge, StrainGaugeConfig

# Thresholds
RHO_COLLAPSE = 0.43  # From CCA theory
KEFF_COLLAPSE = 4.0  # k_eff threshold for "collapse"
VARIANCE_SPIKE_FACTOR = 2.0  # Warning when variance > 2x baseline


def create_ramping_stress(duration: float, max_intensity: float = 1.0):
    """Create gradually increasing GPU stress."""
    start_time = time.time()
    end_time = start_time + duration

    size = 1024
    data = cp.random.random((size, size), dtype=cp.float32)

    while time.time() < end_time:
        elapsed = time.time() - start_time
        intensity = min(max_intensity, elapsed / duration)

        iters = max(1, int(intensity * 15))
        for _ in range(iters):
            data = cp.matmul(data, data)
            cp.cuda.Device().synchronize()


def compute_keff_local(rho_matrix: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute local k_eff for each sensor."""
    n = rho_matrix.shape[0]
    keff_values = []

    for i in range(n):
        local_rho = np.mean([rho_matrix[i, j] for j in range(n) if j != i and not np.isnan(rho_matrix[i, j])])
        local_rho = max(0.0, min(1.0, local_rho))
        denom = 1 + local_rho * (k - 1)
        keff_values.append(k / denom if denom > 0.1 else k / 0.1)

    return np.array(keff_values)


def run_single_trial(sensors: list, trial_num: int) -> dict:
    """Run a single collapse induction trial."""
    n_sensors = len(sensors)

    # Reset sensors
    for s in sensors:
        for _ in range(20):
            s.read()

    # Collect baseline variance
    baseline_samples = []
    for _ in range(30):
        samples = [s.read().timing_mean_us for s in sensors]
        baseline_samples.append(samples)

    baseline_arr = np.array(baseline_samples)
    if baseline_arr.shape[0] > 1:
        baseline_rho = np.corrcoef(baseline_arr.T)
        baseline_keff = compute_keff_local(baseline_rho, k=n_sensors)
        baseline_variance = np.var(baseline_keff)
    else:
        baseline_variance = 1.0

    # Start stress ramp
    stress_thread = threading.Thread(target=create_ramping_stress, args=(4.0, 1.0))
    stress_thread.start()

    sample_buffer = []
    warning_rho = None
    collapse_rho = None
    warning_time = None
    collapse_time = None

    t0 = time.perf_counter()

    for step in range(400):
        samples = [s.read().timing_mean_us for s in sensors]
        t_ms = (time.perf_counter() - t0) * 1000

        sample_buffer.append(samples)
        if len(sample_buffer) > 15:
            sample_buffer.pop(0)

        if len(sample_buffer) >= 10:
            recent = np.array(sample_buffer[-10:])

            if np.std(recent, axis=0).min() > 0.001:
                rho_matrix = np.corrcoef(recent.T)
                rho_matrix = np.nan_to_num(rho_matrix, nan=0.0)

                off_diag = rho_matrix[np.triu_indices(n_sensors, k=1)]
                mean_rho = np.mean(off_diag)

                keff_local = compute_keff_local(rho_matrix, k=n_sensors)
                spatial_variance = np.var(keff_local)
                mean_keff = np.mean(keff_local)

                # Check for warning (variance spike)
                if warning_rho is None and spatial_variance > baseline_variance * VARIANCE_SPIKE_FACTOR:
                    warning_rho = mean_rho
                    warning_time = t_ms

                # Check for collapse
                if collapse_rho is None and (mean_rho > RHO_COLLAPSE or mean_keff < KEFF_COLLAPSE):
                    collapse_rho = mean_rho
                    collapse_time = t_ms

                # Both detected - done
                if warning_rho is not None and collapse_rho is not None:
                    break

    stress_thread.join()

    # Compute delta_rho
    if warning_rho is not None and collapse_rho is not None:
        delta_rho = collapse_rho - warning_rho
    else:
        delta_rho = None

    return {
        'trial': trial_num,
        'warning_rho': warning_rho,
        'collapse_rho': collapse_rho,
        'delta_rho': delta_rho,
        'warning_time_ms': warning_time,
        'collapse_time_ms': collapse_time,
        'baseline_variance': baseline_variance,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT E2: Δρ CONFIDENCE INTERVAL")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Quantify early warning reliability with error bars")
    print("Protocol: Run 30 collapse trials, compute Δρ statistics")
    print()

    results = {}

    # Create sensor array
    print("Creating 4×4 sensor array...")
    n_sensors = 16

    sensors = []
    for i in range(n_sensors):
        config = StrainGaugeConfig(dt=0.025, warm_up_enabled=False)
        gauge = StrainGauge(config)
        sensors.append(gauge)

    # Calibrate
    print("Calibrating...")
    sensors[0].calibrate(duration=2.0)
    for s in sensors[1:]:
        s.baseline_timing_mean = sensors[0].baseline_timing_mean
        s.baseline_timing_std = sensors[0].baseline_timing_std

    # Warm up
    for s in sensors:
        for _ in range(50):
            s.read()

    # Run 30 trials
    print()
    print("=" * 70)
    print("RUNNING 30 TRIALS")
    print("=" * 70)
    print()

    n_trials = 30
    trial_results = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...", end=" ")

        # Rest between trials
        time.sleep(1.0)

        result = run_single_trial(sensors, trial + 1)
        trial_results.append(result)

        if result['delta_rho'] is not None:
            print(f"Δρ = {result['delta_rho']:.3f} "
                  f"(warning @ {result['warning_rho']:.3f}, "
                  f"collapse @ {result['collapse_rho']:.3f})")
        else:
            print("No collapse detected")

    results['trials'] = trial_results

    # Statistics
    print()
    print("=" * 70)
    print("Δρ STATISTICS")
    print("=" * 70)
    print()

    valid_delta_rhos = [r['delta_rho'] for r in trial_results if r['delta_rho'] is not None]
    n_valid = len(valid_delta_rhos)

    print(f"Valid trials: {n_valid}/{n_trials}")

    if n_valid >= 3:
        delta_rhos = np.array(valid_delta_rhos)

        mean_delta = np.mean(delta_rhos)
        std_delta = np.std(delta_rhos, ddof=1)
        se_delta = std_delta / np.sqrt(n_valid)

        # 95% CI using t-distribution
        t_crit = stats.t.ppf(0.975, df=n_valid - 1)
        ci_lower = mean_delta - t_crit * se_delta
        ci_upper = mean_delta + t_crit * se_delta

        # One-sample t-test: is Δρ > 0?
        t_stat, p_value = stats.ttest_1samp(delta_rhos, 0)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        print()
        print(f"  Mean Δρ:     {mean_delta:.4f}")
        print(f"  Std Δρ:      {std_delta:.4f}")
        print(f"  SE:          {se_delta:.4f}")
        print()
        print(f"  95% CI:      [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  CI includes zero: {'YES' if ci_lower <= 0 <= ci_upper else 'NO'}")
        print()
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value (Δρ > 0): {p_one_sided:.4f}")
        print(f"  Significant (α=0.05): {'YES' if p_one_sided < 0.05 else 'NO'}")

        results['statistics'] = {
            'n_valid': n_valid,
            'mean': float(mean_delta),
            'std': float(std_delta),
            'se': float(se_delta),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_stat': float(t_stat),
            'p_value': float(p_one_sided),
            'significant': bool(p_one_sided < 0.05),
        }

        # Warning timing stats
        valid_warnings = [r['warning_time_ms'] for r in trial_results if r['warning_time_ms'] is not None]
        valid_collapses = [r['collapse_time_ms'] for r in trial_results if r['collapse_time_ms'] is not None]

        if valid_warnings:
            print()
            print("  Warning time:  {:.0f} ± {:.0f} ms".format(np.mean(valid_warnings), np.std(valid_warnings)))
        if valid_collapses:
            print("  Collapse time: {:.0f} ± {:.0f} ms".format(np.mean(valid_collapses), np.std(valid_collapses)))
        if valid_warnings and valid_collapses:
            lead_times = np.array(valid_collapses) - np.array(valid_warnings[:len(valid_collapses)])
            print("  Lead time:     {:.0f} ± {:.0f} ms".format(np.mean(lead_times), np.std(lead_times)))

    else:
        print("  Insufficient valid trials for statistics")
        results['statistics'] = None

    # Success criteria
    print()
    print("=" * 70)
    print("E2 SUCCESS CRITERIA")
    print("=" * 70)

    if results.get('statistics'):
        stats_result = results['statistics']
        ci_excludes_zero = stats_result['ci_lower'] > 0 or stats_result['ci_upper'] < 0

        print(f"""
  ✓/✗ 95% CI excludes zero: {'✓' if ci_excludes_zero else '✗'}
  ✓/✗ Δρ significantly > 0: {'✓' if stats_result['significant'] else '✗'} (p = {stats_result['p_value']:.4f})
  ✓/✗ Sufficient samples:   {'✓' if n_valid >= 20 else '✗'} ({n_valid}/30)

  Overall: {'✓ PASS' if ci_excludes_zero and stats_result['significant'] else '✗ NEEDS MORE DATA'}
""")

        results['success'] = {
            'ci_excludes_zero': bool(ci_excludes_zero),
            'significant': bool(stats_result['significant']),
            'sufficient_samples': bool(n_valid >= 20),
        }
    else:
        print("  Insufficient data for evaluation")
        results['success'] = None

    # Summary
    print()
    print("=" * 70)
    print("E2 SUMMARY: Δρ CONFIDENCE INTERVAL")
    print("=" * 70)

    if results.get('statistics'):
        s = results['statistics']
        print(f"""
  Trials: {n_valid}/{n_trials} valid

  Δρ = {s['mean']:.3f} ± {s['std']:.3f}
  95% CI: [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]
  p-value: {s['p_value']:.4f}

  Interpretation:
  ───────────────
  Early warning signal provides Δρ = {s['mean']:.3f} advance notice
  before correlation crosses collapse threshold.
  {'This is statistically significant (p < 0.05).' if s['significant'] else 'More trials needed for significance.'}
""")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import json

    results = main()

    def safe_serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return [safe_serialize(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_serialize(v) for v in obj]
        return obj

    with open('/home/emoore/CIRISArray/experiments/expE2_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expE2_results.json")
