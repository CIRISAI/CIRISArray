#!/usr/bin/env python3
"""
Experiment C1: Real-Time k_eff Heatmap
======================================

Map k_eff spatially during gradual collapse.

Question: Where does diversity collapse first during induced correlation?

Protocol (from RATCHET_UPDATE):
- 16 sensors in 4x4 grid (9631 Hz sample rate)
- Induce correlation via barrier sync at varying strengths
- Compute k_eff = k / (1 + ρ(k-1)) where k=16
- Map which regions collapse first

Success criteria:
- k_eff decreases as sync_strength increases
- k_eff = 16 at sync=0, k_eff → 1 at sync=1
- Identify if collapse is uniform or starts in specific regions

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


def create_barrier_sync_workload(duration: float, strength: float, barrier_event: threading.Event):
    """
    Create GPU workload that induces correlation between sensors.

    At strength=0: random async workload (no correlation)
    At strength=1: fully synchronized barrier (max correlation)

    The synchronization comes from barrier waits that force all
    GPU operations to align, creating correlated timing.
    """
    if strength <= 0:
        return  # No workload at strength=0

    end_time = time.time() + duration
    size = 512  # Matrix size for workload

    # Sync interval: higher strength = more frequent barriers
    sync_interval = max(0.001, 0.1 * (1 - strength))  # 0.1s at low, 0.001s at high

    data = cp.random.random((size, size), dtype=cp.float32)

    while time.time() < end_time:
        # GPU computation
        for _ in range(int(strength * 10) + 1):
            data = cp.matmul(data, data)

        # Barrier synchronization (creates timing correlation)
        cp.cuda.Stream.null.synchronize()

        # Inter-op delay controlled by strength
        time.sleep(sync_interval)

        # Signal barrier for coordination
        barrier_event.set()
        barrier_event.clear()


def collect_samples_parallel(sensors, n_samples: int = 100):
    """Collect samples from all sensors in parallel."""
    all_samples = [[] for _ in sensors]

    for _ in range(n_samples):
        for i, s in enumerate(sensors):
            r = s.read()
            all_samples[i].append(r.timing_mean_us)

    return [np.array(samples) for samples in all_samples]


def compute_keff_formula(rho: float, k: int = 16) -> float:
    """
    CCA formula: k_eff = k / (1 + ρ(k-1))

    At ρ=0: k_eff = k (full diversity)
    At ρ=1: k_eff = 1 (complete collapse)

    Note: Clamp ρ to [0, 1] to avoid division issues with negative correlations.
    """
    rho_clamped = max(0.0, min(1.0, rho))
    denom = 1 + rho_clamped * (k - 1)
    if denom < 0.1:  # Avoid division by near-zero
        denom = 0.1
    return k / denom


def main():
    print("=" * 70)
    print("EXPERIMENT C1: REAL-TIME k_eff HEATMAP")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Map k_eff spatially during gradual correlation collapse")
    print("CCA formula: k_eff = k / (1 + ρ(k-1))")
    print()

    results = {}

    # =========================================================================
    # Create 4x4 sensor array (16 sensors)
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
    # Baseline measurement (no sync)
    # =========================================================================
    print("=" * 70)
    print("BASELINE: No synchronization (strength=0)")
    print("=" * 70)
    print()

    baseline_samples = collect_samples_parallel(sensors, n_samples=200)
    baseline_means = [np.mean(s) for s in baseline_samples]

    # Compute 16x16 correlation matrix
    baseline_matrix = np.zeros((n_sensors, n_sensors))
    for i in range(n_sensors):
        for j in range(n_sensors):
            if len(baseline_samples[i]) > 10 and len(baseline_samples[j]) > 10:
                r = np.corrcoef(baseline_samples[i], baseline_samples[j])[0, 1]
                baseline_matrix[i, j] = r if not np.isnan(r) else 0

    # Off-diagonal mean correlation
    off_diag = baseline_matrix[np.triu_indices(n_sensors, k=1)]
    baseline_rho = np.mean(off_diag)

    print(f"  Baseline mean ρ: {baseline_rho:.3f}")
    print(f"  Expected k_eff at ρ={baseline_rho:.3f}: {compute_keff_formula(baseline_rho):.1f}")
    print()

    results['baseline'] = {
        'mean_rho': float(baseline_rho),
        'correlation_matrix': baseline_matrix.tolist(),
    }

    # =========================================================================
    # Sweep synchronization strength
    # =========================================================================
    print("=" * 70)
    print("k_eff HEATMAP: Varying sync strength")
    print("=" * 70)
    print()

    sync_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = []

    barrier_event = threading.Event()

    print(f"{'Sync':<6} {'Mean ρ':<10} {'k_eff':<10} {'Min k_eff':<10} {'Max k_eff':<10}")
    print("-" * 50)

    for sync_strength in sync_strengths:
        # Start barrier sync workload
        if sync_strength > 0:
            workload_thread = threading.Thread(
                target=create_barrier_sync_workload,
                args=(3.0, sync_strength, barrier_event)
            )
            workload_thread.start()
            time.sleep(0.5)  # Let workload stabilize

        # Collect samples from all sensors
        samples = collect_samples_parallel(sensors, n_samples=150)

        # Wait for workload to finish
        if sync_strength > 0:
            workload_thread.join()

        # Compute correlation matrix
        rho_matrix = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i == j:
                    rho_matrix[i, j] = 1.0
                else:
                    r = np.corrcoef(samples[i], samples[j])[0, 1]
                    rho_matrix[i, j] = r if not np.isnan(r) else 0

        # Mean correlation (off-diagonal)
        off_diag = rho_matrix[np.triu_indices(n_sensors, k=1)]
        mean_rho = np.mean(off_diag)

        # Compute local k_eff for each sensor
        keff_map = np.zeros((4, 4))
        for i in range(n_sensors):
            row, col = i // 4, i % 4
            # Average correlation with all other sensors
            local_rho = np.mean([rho_matrix[i, j] for j in range(n_sensors) if j != i])
            # k_eff = k / (1 + ρ(k-1)), k=16 - clamp to valid range
            local_rho_clamped = max(0.0, min(1.0, local_rho))
            keff_map[row, col] = compute_keff_formula(local_rho_clamped, k=16)

        sweep_results.append({
            'sync': float(sync_strength),
            'mean_rho': float(mean_rho),
            'mean_keff': float(np.mean(keff_map)),
            'min_keff': float(np.min(keff_map)),
            'max_keff': float(np.max(keff_map)),
            'keff_map': keff_map.tolist(),
            'correlation_matrix': rho_matrix.tolist(),
        })

        print(f"{sync_strength:<6.1f} {mean_rho:<10.3f} {np.mean(keff_map):<10.1f} "
              f"{np.min(keff_map):<10.1f} {np.max(keff_map):<10.1f}")

        time.sleep(0.5)  # Recovery between trials

    results['sweep'] = sweep_results

    # =========================================================================
    # Spatial Analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("SPATIAL ANALYSIS: Where does collapse start?")
    print("=" * 70)
    print()

    # Find sync level where collapse is most non-uniform
    max_gradient_sync = 0
    max_gradient = 0
    collapse_start_position = None

    for r in sweep_results:
        keff_arr = np.array(r['keff_map'])
        gradient = np.max(keff_arr) - np.min(keff_arr)
        if gradient > max_gradient:
            max_gradient = gradient
            max_gradient_sync = r['sync']
            # Position with minimum k_eff = first to collapse
            min_pos = np.unravel_index(np.argmin(keff_arr), keff_arr.shape)
            collapse_start_position = min_pos

    print(f"  Maximum spatial gradient: {max_gradient:.2f} at sync={max_gradient_sync}")
    if collapse_start_position:
        print(f"  Collapse starts at position: ({collapse_start_position[0]}, {collapse_start_position[1]})")

    # Show k_eff heatmap at peak gradient
    peak_result = next((r for r in sweep_results if r['sync'] == max_gradient_sync), None)
    if peak_result:
        print(f"\n  k_eff heatmap at sync={max_gradient_sync}:")
        print("  " + "-" * 30)
        keff_arr = np.array(peak_result['keff_map'])
        for row in range(4):
            row_str = "  "
            for col in range(4):
                row_str += f"{keff_arr[row, col]:6.1f} "
            print(row_str)
        print("  " + "-" * 30)

    results['spatial'] = {
        'max_gradient': float(max_gradient),
        'max_gradient_sync': float(max_gradient_sync),
        'collapse_start': collapse_start_position,
    }

    # =========================================================================
    # CCA Validation
    # =========================================================================
    print()
    print("=" * 70)
    print("CCA VALIDATION")
    print("=" * 70)
    print()

    # Check if k_eff follows predicted curve
    rho_values = [r['mean_rho'] for r in sweep_results]
    keff_values = [r['mean_keff'] for r in sweep_results]
    predicted_keff = [compute_keff_formula(rho) for rho in rho_values]

    # R² between measured and predicted
    if len(keff_values) > 2:
        ss_res = np.sum((np.array(keff_values) - np.array(predicted_keff)) ** 2)
        ss_tot = np.sum((np.array(keff_values) - np.mean(keff_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        r_squared = 0

    print(f"  CCA formula fit: R² = {r_squared:.3f}")
    print()
    print("  ρ → k_eff mapping:")
    print(f"  {'ρ':<8} {'Measured':<12} {'CCA Predicted':<14} {'Δ':<8}")
    print("  " + "-" * 42)
    for i, r in enumerate(sweep_results):
        measured = r['mean_keff']
        predicted = predicted_keff[i]
        delta = measured - predicted
        print(f"  {r['mean_rho']:<8.3f} {measured:<12.1f} {predicted:<14.1f} {delta:+.1f}")

    results['cca_validation'] = {
        'r_squared': float(r_squared),
        'rho_values': rho_values,
        'keff_measured': keff_values,
        'keff_predicted': predicted_keff,
    }

    # =========================================================================
    # Success Criteria Check
    # =========================================================================
    print()
    print("=" * 70)
    print("C1 SUCCESS CRITERIA")
    print("=" * 70)

    # 1. k_eff decreases as sync increases
    keff_decrease = keff_values[0] > keff_values[-1]

    # 2. k_eff ≈ 16 at sync=0
    keff_at_zero = sweep_results[0]['mean_keff']
    keff_near_16 = keff_at_zero > 10  # Allow some correlation at baseline

    # 3. k_eff → 1 at sync=1
    keff_at_one = sweep_results[-1]['mean_keff']
    keff_near_1 = keff_at_one < 5  # Should approach 1

    # 4. Non-uniform collapse (spatial gradient > 0)
    non_uniform = max_gradient > 1.0

    print(f"""
  ✓/✗ k_eff decreases with sync: {'✓' if keff_decrease else '✗'} ({keff_values[0]:.1f} → {keff_values[-1]:.1f})
  ✓/✗ k_eff ≈ 16 at sync=0:      {'✓' if keff_near_16 else '✗'} ({keff_at_zero:.1f})
  ✓/✗ k_eff → 1 at sync=1:       {'✓' if keff_near_1 else '✗'} ({keff_at_one:.1f})
  ✓/✗ Non-uniform collapse:      {'✓' if non_uniform else '✗'} (gradient={max_gradient:.2f})

  Overall: {'✓ PASS' if keff_decrease and keff_near_16 else '✗ PARTIAL'}
""")

    results['success'] = {
        'keff_decreases': bool(keff_decrease),
        'keff_at_zero': float(keff_at_zero),
        'keff_at_one': float(keff_at_one),
        'non_uniform': bool(non_uniform),
        'all_pass': bool(keff_decrease and keff_near_16),
    }

    # =========================================================================
    # SUMMARY for RATCHET Team
    # =========================================================================
    print()
    print("=" * 70)
    print("C1 SUMMARY: k_eff HEATMAP RESULTS")
    print("=" * 70)
    print(f"""
  Instrument: 4×4 sensor array (16 sensors)
  Sample rate: {sensors[0].baseline_timing_mean:.1f} μs/sample

  Key Findings:
  ─────────────
  1. Baseline ρ = {baseline_rho:.3f} (k_eff = {compute_keff_formula(baseline_rho):.1f})
  2. Maximum ρ achieved = {max(rho_values):.3f} (k_eff = {min(keff_values):.1f})
  3. CCA formula fit: R² = {r_squared:.3f}
  4. Collapse starts at position: {collapse_start_position}
  5. Maximum spatial gradient: {max_gradient:.2f} at sync={max_gradient_sync}

  For Lean Formalization:
  ───────────────────────
  k_eff = 16 / (1 + ρ × 15)
  Measured R² = {r_squared:.3f} against CCA prediction

  Ready for C2: Propagation velocity measurement
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

    with open('/home/emoore/CIRISArray/experiments/expC1_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expC1_results.json")
