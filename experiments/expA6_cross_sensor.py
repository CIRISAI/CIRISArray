#!/usr/bin/env python3
"""
Experiment A6: Cross-Sensor Consistency
========================================

Test if 16 sensors give consistent readings.

Success criteria: CV < 15% across sensors

Protocol (from RATCHET_UPDATE):
- Create 16 sensors with validated config
- Run same workload, measure mean shift from each
- Check coefficient of variation across sensors

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


def create_workload(duration: float, intensity: float):
    """Create GPU workload."""
    end_time = time.time() + duration
    size = max(64, int(2048 * np.sqrt(intensity)))
    data = cp.random.random((size, size), dtype=cp.float32)
    while time.time() < end_time:
        data = cp.matmul(data, data)
        cp.cuda.Stream.null.synchronize()


def collect_samples_fast(gauge, n_samples: int):
    """Collect timing samples."""
    samples = []
    for _ in range(n_samples):
        r = gauge.read()
        samples.append(r.timing_mean_us)
    return np.array(samples)


def main():
    print("=" * 70)
    print("EXPERIMENT A6: CROSS-SENSOR CONSISTENCY")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Success criteria: CV < 15% across 16 sensors")
    print()

    results = {}

    # =========================================================================
    # Create 16 sensors
    # =========================================================================
    print("Creating 16 sensors...")
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
        for _ in range(30):
            s.read()
    print()

    # =========================================================================
    # TEST 1: Baseline consistency (no workload)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: BASELINE CONSISTENCY (0% workload)")
    print("=" * 70)
    print()

    baseline_means = []
    for i, s in enumerate(sensors):
        samples = collect_samples_fast(s, n_samples=100)
        baseline_means.append(np.mean(samples))

    baseline_cv = (np.std(baseline_means) / np.mean(baseline_means)) * 100

    print(f"  Sensor means (μs): {[f'{m:.1f}' for m in baseline_means]}")
    print(f"  Overall mean: {np.mean(baseline_means):.2f} μs")
    print(f"  Std: {np.std(baseline_means):.2f} μs")
    print(f"  CV: {baseline_cv:.1f}%")
    print(f"  Status: {'✓ PASS' if baseline_cv < 15 else '✗ FAIL'}")

    results['baseline'] = {
        'means': [float(m) for m in baseline_means],
        'cv': float(baseline_cv),
        'pass': baseline_cv < 15,
    }

    # =========================================================================
    # TEST 2: Workload consistency (50% load)
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 2: WORKLOAD CONSISTENCY (50% workload)")
    print("=" * 70)
    print()

    # Start workload
    workload_thread = threading.Thread(target=create_workload, args=(20.0, 0.5))
    workload_thread.start()
    time.sleep(2.0)

    workload_means = []
    for i, s in enumerate(sensors):
        samples = collect_samples_fast(s, n_samples=100)
        workload_means.append(np.mean(samples))

    workload_thread.join()

    workload_cv = (np.std(workload_means) / np.mean(workload_means)) * 100

    print(f"  Sensor means (μs): {[f'{m:.1f}' for m in workload_means]}")
    print(f"  Overall mean: {np.mean(workload_means):.2f} μs")
    print(f"  Std: {np.std(workload_means):.2f} μs")
    print(f"  CV: {workload_cv:.1f}%")
    print(f"  Status: {'✓ PASS' if workload_cv < 15 else '✗ FAIL'}")

    results['workload'] = {
        'means': [float(m) for m in workload_means],
        'cv': float(workload_cv),
        'pass': workload_cv < 15,
    }

    time.sleep(2.0)

    # =========================================================================
    # TEST 3: Mean shift consistency
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 3: MEAN SHIFT CONSISTENCY")
    print("=" * 70)
    print()

    # Calculate mean shift for each sensor
    shifts = []
    shift_pcts = []
    for i in range(n_sensors):
        shift = workload_means[i] - baseline_means[i]
        shift_pct = (shift / baseline_means[i]) * 100 if baseline_means[i] > 0 else 0
        shifts.append(shift)
        shift_pcts.append(shift_pct)

    shift_cv = (np.std(shift_pcts) / np.mean(shift_pcts)) * 100 if np.mean(shift_pcts) > 0 else 0

    print(f"  Shift percentages: {[f'{p:.0f}%' for p in shift_pcts]}")
    print(f"  Mean shift: {np.mean(shift_pcts):.0f}%")
    print(f"  Shift std: {np.std(shift_pcts):.0f}%")
    print(f"  Shift CV: {shift_cv:.1f}%")
    print(f"  Status: {'✓ PASS' if shift_cv < 15 else '✗ FAIL'}")

    # Check if all sensors detect (>50% shift)
    n_detected = sum(1 for p in shift_pcts if p > 50)
    print(f"\n  Sensors detecting (>50% shift): {n_detected}/{n_sensors}")

    results['shift'] = {
        'percentages': [float(p) for p in shift_pcts],
        'mean_shift_pct': float(np.mean(shift_pcts)),
        'cv': float(shift_cv),
        'n_detected': n_detected,
        'pass': shift_cv < 15 and n_detected == n_sensors,
    }

    # =========================================================================
    # TEST 4: 4x4 Spatial Map
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 4: 4x4 SPATIAL MAP")
    print("=" * 70)
    print()

    shift_map = np.array(shift_pcts).reshape(4, 4)

    print("  Mean shift (%) by position:")
    print("  " + "-" * 30)
    for row in range(4):
        row_str = "  "
        for col in range(4):
            row_str += f"{shift_map[row, col]:6.0f}% "
        print(row_str)
    print("  " + "-" * 30)

    # Check for spatial uniformity
    spatial_std = np.std(shift_map)
    spatial_mean = np.mean(shift_map)
    spatial_cv = (spatial_std / spatial_mean) * 100 if spatial_mean > 0 else 0

    print(f"\n  Spatial CV: {spatial_cv:.1f}%")
    print(f"  Uniform detection: {'✓ YES' if spatial_cv < 20 else '✗ NO (check for hotspots)'}")

    results['spatial'] = {
        'map': shift_map.tolist(),
        'cv': float(spatial_cv),
        'uniform': spatial_cv < 20,
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("A6 SUMMARY: CROSS-SENSOR CONSISTENCY")
    print("=" * 70)

    all_pass = (results['baseline']['pass'] and
                results['workload']['pass'] and
                results['shift']['pass'])

    print(f"""
  Test Results:
  ─────────────
  Baseline CV:  {results['baseline']['cv']:.1f}% {'✓' if results['baseline']['pass'] else '✗'} (target: <15%)
  Workload CV:  {results['workload']['cv']:.1f}% {'✓' if results['workload']['pass'] else '✗'} (target: <15%)
  Shift CV:     {results['shift']['cv']:.1f}% {'✓' if results['shift']['pass'] else '✗'} (target: <15%)
  All detect:   {results['shift']['n_detected']}/16 {'✓' if results['shift']['n_detected']==16 else '✗'}

  Overall: {'✓ PASS - Sensors are consistent' if all_pass else '✗ FAIL - High variability'}

  Mean shift across all sensors: {results['shift']['mean_shift_pct']:.0f}%
  (Target: >50% for detection, Ossicle: +248%)
""")

    results['summary'] = {
        'all_pass': bool(all_pass),
        'baseline_cv': float(results['baseline']['cv']),
        'workload_cv': float(results['workload']['cv']),
        'shift_cv': float(results['shift']['cv']),
    }

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
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_serialize(v) for v in obj]
        return obj

    with open('/home/emoore/CIRISArray/experiments/expA6_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expA6_results.json")
