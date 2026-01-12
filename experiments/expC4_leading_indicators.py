#!/usr/bin/env python3
"""
Experiment C4: Spatial Leading Indicators
=========================================

Look for spatial patterns that predict collapse before ρ crosses 0.43.

Question: Do spatial patterns predict collapse before the critical threshold?

Protocol (from RATCHET_UPDATE):
- Slowly ramp toward collapse
- Record spatial patterns (variance, gradient)
- Compare patterns BEFORE vs AFTER ρ crosses 0.43
- Identify early warning signals

Success criteria:
- Spatial variance or gradient changes BEFORE ρ crosses 0.43
- Provides early warning (how many ms/steps?)
- Consistent pattern across trials

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from strain_gauge import StrainGauge, StrainGaugeConfig


# Critical correlation threshold from CCA theory
RHO_CRITICAL = 0.43


def create_ramping_stress(duration: float, ramp_rate: float = 0.5):
    """
    Create gradually increasing GPU stress.

    Stress increases linearly from 0 to 1 over duration.
    """
    start_time = time.time()
    end_time = start_time + duration

    size_base = 1024  # Larger for more stress
    data = cp.random.random((size_base, size_base), dtype=cp.float32)

    while time.time() < end_time:
        # Calculate current intensity based on elapsed time
        elapsed = time.time() - start_time
        intensity = min(1.0, (elapsed / duration) * ramp_rate)

        # Scale computation with intensity - much more aggressive
        iters = max(1, int(intensity * 20))
        for _ in range(iters):
            data = cp.matmul(data, data)
            cp.cuda.Device().synchronize()

        # No sleep - continuous stress


def compute_keff_local(rho_matrix: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute local k_eff for each sensor."""
    n = rho_matrix.shape[0]
    keff_map = np.zeros((4, 4))

    for i in range(n):
        row, col = i // 4, i % 4
        local_rho = np.mean([rho_matrix[i, j] for j in range(n) if j != i and not np.isnan(rho_matrix[i, j])])
        local_rho = max(0.0, min(1.0, local_rho))
        denom = 1 + local_rho * (k - 1)
        keff_map[row, col] = k / denom if denom > 0.1 else k / 0.1

    return keff_map


def main():
    print("=" * 70)
    print("EXPERIMENT C4: SPATIAL LEADING INDICATORS")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print(f"Goal: Find patterns that predict collapse BEFORE ρ > {RHO_CRITICAL}")
    print("Method: Slow ramp, track spatial variance and gradient")
    print()

    results = {}

    # =========================================================================
    # Create 4x4 sensor array
    # =========================================================================
    print("Creating 4×4 sensor array (16 sensors)...")
    n_sensors = 16

    sensors = []
    for i in range(n_sensors):
        config = StrainGaugeConfig(dt=0.025, warm_up_enabled=False)
        gauge = StrainGauge(config)
        sensors.append(gauge)

    # Calibrate first sensor and share baseline
    print("Calibrating (shared baseline)...")
    sensors[0].calibrate(duration=3.0)

    for s in sensors[1:]:
        s.baseline_timing_mean = sensors[0].baseline_timing_mean
        s.baseline_timing_std = sensors[0].baseline_timing_std

    # Warm up all sensors
    print("Warming up all sensors...")
    for s in sensors:
        for _ in range(50):
            s.read()
    print()

    # =========================================================================
    # Leading indicator detection
    # =========================================================================
    print("=" * 70)
    print("LEADING INDICATOR DETECTION")
    print("=" * 70)
    print()

    n_trials = 10
    all_patterns_before = []
    all_patterns_after = []
    trial_results = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")

        # Reset
        time.sleep(1.5)

        history = []
        sample_buffer = []

        # Start ramping stress
        stress_thread = threading.Thread(
            target=create_ramping_stress,
            args=(5.0, 1.0)  # 5 seconds, full ramp
        )
        stress_thread.start()

        t0 = time.perf_counter()
        collapse_detected = False
        collapse_time = None

        # Sample during ramp
        for step in range(500):  # Up to 500 steps
            samples = [s.read().timing_mean_us for s in sensors]
            t_ms = (time.perf_counter() - t0) * 1000

            sample_buffer.append(samples)

            # Keep buffer at 20 samples
            if len(sample_buffer) > 20:
                sample_buffer.pop(0)

            # Need at least 10 samples
            if len(sample_buffer) >= 10:
                recent = np.array(sample_buffer[-10:])

                # Compute correlation matrix
                if np.std(recent, axis=0).min() > 0.001:
                    rho_matrix = np.corrcoef(recent.T)

                    # Handle NaN
                    rho_matrix = np.nan_to_num(rho_matrix, nan=0.0)

                    # Mean correlation (off-diagonal)
                    off_diag = rho_matrix[np.triu_indices(n_sensors, k=1)]
                    mean_rho = np.mean(off_diag)

                    # Compute local k_eff
                    keff_map = compute_keff_local(rho_matrix, k=16)
                    keff_local = keff_map.flatten()

                    # Spatial metrics
                    spatial_variance = np.var(keff_local)
                    spatial_gradient = np.max(keff_local) - np.min(keff_local)
                    spatial_std = np.std(keff_local)

                    # Mean k_eff
                    mean_keff = np.mean(keff_local)

                    # Detect collapse
                    collapsed = mean_rho > RHO_CRITICAL

                    if collapsed and not collapse_detected:
                        collapse_detected = True
                        collapse_time = t_ms

                    record = {
                        't_ms': t_ms,
                        'mean_rho': float(mean_rho),
                        'mean_keff': float(mean_keff),
                        'spatial_variance': float(spatial_variance),
                        'spatial_gradient': float(spatial_gradient),
                        'spatial_std': float(spatial_std),
                        'collapsed': bool(collapsed),
                        'keff_map': keff_map.tolist(),
                    }
                    history.append(record)

                    # Classify as before or after collapse
                    if collapsed:
                        all_patterns_after.append(record)
                    else:
                        all_patterns_before.append(record)

        stress_thread.join()

        # Trial summary
        if collapse_detected:
            print(f"  Collapse at {collapse_time:.0f}ms, {len(history)} samples")
        else:
            print(f"  No collapse, {len(history)} samples")

        trial_results.append({
            'trial': trial + 1,
            'collapse_detected': collapse_detected,
            'collapse_time_ms': collapse_time,
            'n_samples': len(history),
            'history': history[-50:] if len(history) > 50 else history  # Keep last 50
        })

    results['trials'] = trial_results

    # =========================================================================
    # Pattern comparison: Before vs After collapse
    # =========================================================================
    print()
    print("=" * 70)
    print("PATTERN COMPARISON: Before vs After ρ > 0.43")
    print("=" * 70)
    print()

    if len(all_patterns_before) > 0 and len(all_patterns_after) > 0:
        # Compute statistics
        before_variance = np.mean([p['spatial_variance'] for p in all_patterns_before])
        after_variance = np.mean([p['spatial_variance'] for p in all_patterns_after])

        before_gradient = np.mean([p['spatial_gradient'] for p in all_patterns_before])
        after_gradient = np.mean([p['spatial_gradient'] for p in all_patterns_after])

        before_std = np.mean([p['spatial_std'] for p in all_patterns_before])
        after_std = np.mean([p['spatial_std'] for p in all_patterns_after])

        before_keff = np.mean([p['mean_keff'] for p in all_patterns_before])
        after_keff = np.mean([p['mean_keff'] for p in all_patterns_after])

        print(f"  Samples: {len(all_patterns_before)} before, {len(all_patterns_after)} after")
        print()
        print(f"  {'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15}")
        print("  " + "-" * 60)
        print(f"  {'Spatial Variance':<20} {before_variance:<15.3f} {after_variance:<15.3f} {(after_variance-before_variance)/before_variance*100:+.0f}%" if before_variance > 0 else "")
        print(f"  {'Spatial Gradient':<20} {before_gradient:<15.2f} {after_gradient:<15.2f} {(after_gradient-before_gradient)/before_gradient*100:+.0f}%" if before_gradient > 0 else "")
        print(f"  {'Spatial Std':<20} {before_std:<15.3f} {after_std:<15.3f} {(after_std-before_std)/before_std*100:+.0f}%" if before_std > 0 else "")
        print(f"  {'Mean k_eff':<20} {before_keff:<15.1f} {after_keff:<15.1f} {(after_keff-before_keff)/before_keff*100:+.0f}%" if before_keff > 0 else "")

        results['comparison'] = {
            'n_before': len(all_patterns_before),
            'n_after': len(all_patterns_after),
            'before_variance': float(before_variance),
            'after_variance': float(after_variance),
            'before_gradient': float(before_gradient),
            'after_gradient': float(after_gradient),
            'before_std': float(before_std),
            'after_std': float(after_std),
            'before_keff': float(before_keff),
            'after_keff': float(after_keff),
        }

        # =====================================================================
        # Leading indicator identification
        # =====================================================================
        print()
        print("=" * 70)
        print("LEADING INDICATOR IDENTIFICATION")
        print("=" * 70)
        print()

        # Which metric changes most BEFORE collapse?
        # Look for monotonic trend in the pre-collapse data

        # Sort pre-collapse by rho
        sorted_before = sorted(all_patterns_before, key=lambda x: x['mean_rho'])

        if len(sorted_before) >= 20:
            # Split into low-rho and high-rho (but still pre-collapse)
            n_half = len(sorted_before) // 2
            low_rho = sorted_before[:n_half]
            high_rho = sorted_before[n_half:]

            low_var = np.mean([p['spatial_variance'] for p in low_rho])
            high_var = np.mean([p['spatial_variance'] for p in high_rho])
            var_trend = high_var - low_var

            low_grad = np.mean([p['spatial_gradient'] for p in low_rho])
            high_grad = np.mean([p['spatial_gradient'] for p in high_rho])
            grad_trend = high_grad - low_grad

            low_std = np.mean([p['spatial_std'] for p in low_rho])
            high_std = np.mean([p['spatial_std'] for p in high_rho])
            std_trend = high_std - low_std

            print(f"  Trends WITHIN pre-collapse data (low ρ → high ρ):")
            print(f"    Variance trend:  {var_trend:+.4f} ({'↑' if var_trend > 0 else '↓'})")
            print(f"    Gradient trend:  {grad_trend:+.2f} ({'↑' if grad_trend > 0 else '↓'})")
            print(f"    Std trend:       {std_trend:+.4f} ({'↑' if std_trend > 0 else '↓'})")

            # Identify leading indicator
            if abs(var_trend) > abs(grad_trend) and abs(var_trend) > abs(std_trend):
                leading_indicator = 'spatial_variance'
                trend_direction = 'increases' if var_trend > 0 else 'decreases'
            elif abs(grad_trend) > abs(std_trend):
                leading_indicator = 'spatial_gradient'
                trend_direction = 'increases' if grad_trend > 0 else 'decreases'
            else:
                leading_indicator = 'spatial_std'
                trend_direction = 'increases' if std_trend > 0 else 'decreases'

            print()
            print(f"  Leading indicator: {leading_indicator} {trend_direction} before collapse")

            results['leading_indicator'] = {
                'metric': leading_indicator,
                'direction': trend_direction,
                'var_trend': float(var_trend),
                'grad_trend': float(grad_trend),
                'std_trend': float(std_trend),
            }

        # =====================================================================
        # Early warning time
        # =====================================================================
        print()
        print("=" * 70)
        print("EARLY WARNING TIME")
        print("=" * 70)
        print()

        # Find when each metric starts deviating from baseline
        # Use first 20% of pre-collapse as "baseline"
        if len(sorted_before) >= 10:
            baseline_n = max(5, len(sorted_before) // 5)
            baseline_samples = sorted_before[:baseline_n]

            baseline_var = np.mean([p['spatial_variance'] for p in baseline_samples])
            baseline_var_std = np.std([p['spatial_variance'] for p in baseline_samples])

            # Find first sample that exceeds 2σ
            early_warning_idx = None
            for i, p in enumerate(sorted_before[baseline_n:], baseline_n):
                if abs(p['spatial_variance'] - baseline_var) > 2 * baseline_var_std:
                    early_warning_idx = i
                    break

            if early_warning_idx:
                # How far before collapse threshold?
                early_warning_rho = sorted_before[early_warning_idx]['mean_rho']
                rho_margin = RHO_CRITICAL - early_warning_rho

                print(f"  Early warning signal detected:")
                print(f"    At ρ = {early_warning_rho:.3f}")
                print(f"    Margin before collapse: Δρ = {rho_margin:.3f}")
                print(f"    Position: sample {early_warning_idx}/{len(sorted_before)}")

                results['early_warning'] = {
                    'detected': True,
                    'rho_at_warning': float(early_warning_rho),
                    'rho_margin': float(rho_margin),
                    'position_pct': float(early_warning_idx / len(sorted_before) * 100),
                }
            else:
                print(f"  No early warning signal detected within 2σ threshold")
                results['early_warning'] = {'detected': False}

    else:
        print("  Insufficient data for comparison")
        results['comparison'] = None

    # =========================================================================
    # Success criteria
    # =========================================================================
    print()
    print("=" * 70)
    print("C4 SUCCESS CRITERIA")
    print("=" * 70)

    has_change = results.get('comparison') is not None and results['comparison']['before_variance'] != results['comparison']['after_variance']
    has_indicator = results.get('leading_indicator') is not None
    has_warning = results.get('early_warning', {}).get('detected', False)

    print(f"""
  ✓/✗ Pattern changes before/after: {'✓' if has_change else '✗'}
  ✓/✗ Leading indicator identified: {'✓' if has_indicator else '✗'} ({results.get('leading_indicator', {}).get('metric', 'N/A')})
  ✓/✗ Early warning detected:       {'✓' if has_warning else '✗'}

  Overall: {'✓ PASS' if has_change and has_indicator else '✗ PARTIAL'}
""")

    results['success'] = {
        'has_change': bool(has_change),
        'has_indicator': bool(has_indicator),
        'has_warning': bool(has_warning),
        'all_pass': bool(has_change and has_indicator)
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("C4 SUMMARY: LEADING INDICATORS")
    print("=" * 70)

    comp = results.get('comparison') or {}
    li = results.get('leading_indicator') or {}
    ew = results.get('early_warning') or {}

    print(f"""
  Instrument: 4×4 sensor array, gradual stress ramp
  Critical threshold: ρ = {RHO_CRITICAL}

  Key Findings:
  ─────────────
  1. Samples collected: {comp.get('n_before', 0)} before, {comp.get('n_after', 0)} after
  2. Leading indicator: {li.get('metric', 'N/A')} ({li.get('direction', 'N/A')})
  3. Early warning: {'Yes at ρ=' + str(ew.get('rho_at_warning', 0)) if has_warning else 'No'}
  4. Variance change: {comp.get('before_variance', 0):.3f} → {comp.get('after_variance', 0):.3f}
  5. k_eff change: {comp.get('before_keff', 0):.1f} → {comp.get('after_keff', 0):.1f}

  For Lean Formalization:
  ───────────────────────
  leading_indicator := {li.get('metric', 'N/A')}
  early_warning_rho := {ew.get('rho_at_warning', 'N/A')}
  rho_critical := {RHO_CRITICAL}
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

    with open('/home/emoore/CIRISArray/experiments/expC4_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expC4_results.json")
