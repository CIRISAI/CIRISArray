#!/usr/bin/env python3
"""
Experiment E4: Operational Collapse Threshold
==============================================

Gap: "Collapse" not operationally defined - when exactly does system fail?

Protocol:
- Sweep sync_strength from 0 → 1
- At each level, measure:
  - k_eff (diversity metric)
  - Detection latency (workload detection time)
  - TRNG entropy rate (randomness quality)
  - False positive rate (spurious detections at idle)
- Find k_eff threshold where function degrades >50%

Success criteria: Identify k_eff_critical where detection/TRNG fails

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


def create_sync_stress(duration: float, strength: float):
    """Create synchronized GPU stress at given strength."""
    if strength <= 0:
        time.sleep(duration)
        return

    end_time = time.time() + duration
    size = int(512 + 512 * strength)
    data = cp.random.random((size, size), dtype=cp.float32)

    sync_interval = max(0.001, 0.05 * (1 - strength))

    while time.time() < end_time:
        iters = max(1, int(strength * 10))
        for _ in range(iters):
            data = cp.matmul(data, data)
        cp.cuda.Device().synchronize()
        time.sleep(sync_interval)


def compute_mean_rho(sensors: list, n_samples: int = 50) -> tuple:
    """Compute mean correlation and k_eff across sensors."""
    n = len(sensors)
    samples = []

    for _ in range(n_samples):
        sample = [s.read().timing_mean_us for s in sensors]
        samples.append(sample)

    samples = np.array(samples)

    if samples.shape[0] < 10:
        return 0.0, float(n)

    rho_matrix = np.corrcoef(samples.T)
    rho_matrix = np.nan_to_num(rho_matrix, nan=0.0)

    off_diag = rho_matrix[np.triu_indices(n, k=1)]
    mean_rho = np.mean(off_diag)
    mean_rho = max(0.0, min(1.0, mean_rho))

    # CCA formula
    k_eff = n / (1 + mean_rho * (n - 1))

    return float(mean_rho), float(k_eff)


def measure_detection_latency(sensors: list, n_tests: int = 5) -> float:
    """Measure workload detection latency."""
    latencies = []

    for _ in range(n_tests):
        # Baseline
        baseline_timings = []
        for _ in range(20):
            t = sensors[0].read().timing_mean_us
            baseline_timings.append(t)
        baseline_mean = np.mean(baseline_timings)
        baseline_std = np.std(baseline_timings)
        threshold = baseline_mean + 3 * baseline_std

        # Start workload and measure detection time
        t0 = time.perf_counter()
        workload_thread = threading.Thread(
            target=lambda: [cp.matmul(cp.random.random((512, 512)), cp.random.random((512, 512))) for _ in range(100)]
        )
        workload_thread.start()

        detected = False
        while time.perf_counter() - t0 < 0.5:  # 500ms timeout
            t = sensors[0].read().timing_mean_us
            if t > threshold:
                latency = (time.perf_counter() - t0) * 1000  # ms
                latencies.append(latency)
                detected = True
                break

        workload_thread.join()

        if not detected:
            latencies.append(500.0)  # Timeout

        time.sleep(0.2)

    return float(np.mean(latencies))


def measure_entropy_rate(sensors: list, n_bits: int = 1000) -> float:
    """Measure TRNG entropy rate from timing LSBs."""
    bits = []

    for _ in range(n_bits):
        t0 = time.perf_counter_ns()
        sensors[0].read()
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()

        timing_ns = t1 - t0
        bit = timing_ns & 1
        bits.append(bit)

    bits = np.array(bits)

    # Shannon entropy
    p1 = np.mean(bits)
    p0 = 1 - p1

    if p0 > 0 and p1 > 0:
        entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
    else:
        entropy = 0.0

    return float(entropy)


def measure_false_positive_rate(sensors: list, duration: float = 2.0) -> float:
    """Measure false positive rate at idle."""
    # Establish baseline
    baseline_timings = []
    for _ in range(50):
        t = sensors[0].read().timing_mean_us
        baseline_timings.append(t)

    baseline_mean = np.mean(baseline_timings)
    baseline_std = np.std(baseline_timings)
    threshold = baseline_mean + 3 * baseline_std

    # Count false positives at idle
    false_positives = 0
    n_samples = 0

    start = time.time()
    while time.time() - start < duration:
        t = sensors[0].read().timing_mean_us
        n_samples += 1
        if t > threshold:
            false_positives += 1

    fp_rate = false_positives / n_samples if n_samples > 0 else 0
    return float(fp_rate)


def main():
    print("=" * 70)
    print("EXPERIMENT E4: OPERATIONAL COLLAPSE THRESHOLD")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Define when system operationally 'collapses'")
    print("Metrics: k_eff, detection latency, entropy rate, FP rate")
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

    # Baseline measurements
    print()
    print("=" * 70)
    print("BASELINE MEASUREMENTS (sync=0)")
    print("=" * 70)
    print()

    baseline_rho, baseline_keff = compute_mean_rho(sensors)
    baseline_latency = measure_detection_latency(sensors)
    baseline_entropy = measure_entropy_rate(sensors)
    baseline_fp = measure_false_positive_rate(sensors)

    print(f"  ρ = {baseline_rho:.3f}, k_eff = {baseline_keff:.1f}")
    print(f"  Detection latency: {baseline_latency:.1f} ms")
    print(f"  Entropy rate: {baseline_entropy:.3f} bits/bit")
    print(f"  False positive rate: {baseline_fp:.4f}")

    results['baseline'] = {
        'rho': baseline_rho,
        'keff': baseline_keff,
        'latency_ms': baseline_latency,
        'entropy': baseline_entropy,
        'fp_rate': baseline_fp,
    }

    # Sweep sync strength
    print()
    print("=" * 70)
    print("SYNC STRENGTH SWEEP")
    print("=" * 70)
    print()

    sync_strengths = np.linspace(0.0, 1.0, 11)
    sweep_results = []

    print(f"{'Sync':<6} {'ρ':<8} {'k_eff':<8} {'Latency':<10} {'Entropy':<10} {'FP Rate':<10}")
    print("-" * 55)

    for sync in sync_strengths:
        # Apply sync stress
        stress_thread = threading.Thread(target=create_sync_stress, args=(2.0, sync))
        stress_thread.start()
        time.sleep(0.5)

        # Measure metrics
        rho, keff = compute_mean_rho(sensors, n_samples=30)
        latency = measure_detection_latency(sensors, n_tests=3)
        entropy = measure_entropy_rate(sensors, n_bits=500)
        fp_rate = measure_false_positive_rate(sensors, duration=1.0)

        stress_thread.join()

        sweep_results.append({
            'sync': float(sync),
            'rho': rho,
            'keff': keff,
            'latency_ms': latency,
            'entropy': entropy,
            'fp_rate': fp_rate,
        })

        print(f"{sync:<6.2f} {rho:<8.3f} {keff:<8.1f} {latency:<10.1f} {entropy:<10.3f} {fp_rate:<10.4f}")

        time.sleep(0.5)

    results['sweep'] = sweep_results

    # Find collapse thresholds
    print()
    print("=" * 70)
    print("COLLAPSE THRESHOLD ANALYSIS")
    print("=" * 70)
    print()

    # Detection latency degradation > 50%
    latency_threshold = baseline_latency * 1.5
    latency_collapse_keff = None
    for r in sweep_results:
        if r['latency_ms'] > latency_threshold:
            latency_collapse_keff = r['keff']
            break

    # Entropy degradation > 50%
    entropy_threshold = baseline_entropy * 0.5
    entropy_collapse_keff = None
    for r in sweep_results:
        if r['entropy'] < entropy_threshold:
            entropy_collapse_keff = r['keff']
            break

    # FP rate > 10%
    fp_collapse_keff = None
    for r in sweep_results:
        if r['fp_rate'] > 0.1:
            fp_collapse_keff = r['keff']
            break

    print(f"  Latency collapse (>50% degradation): k_eff < {latency_collapse_keff if latency_collapse_keff else 'N/A'}")
    print(f"  Entropy collapse (>50% degradation): k_eff < {entropy_collapse_keff if entropy_collapse_keff else 'N/A'}")
    print(f"  FP rate collapse (>10%):             k_eff < {fp_collapse_keff if fp_collapse_keff else 'N/A'}")

    # Overall critical k_eff
    collapse_values = [v for v in [latency_collapse_keff, entropy_collapse_keff, fp_collapse_keff] if v is not None]
    if collapse_values:
        keff_critical = max(collapse_values)  # Most conservative
    else:
        keff_critical = None

    print()
    if keff_critical:
        print(f"  CRITICAL k_eff: {keff_critical:.1f}")
        print(f"  System fails when k_eff drops below {keff_critical:.1f}")
    else:
        print("  No collapse detected in sweep range")

    results['thresholds'] = {
        'latency_collapse_keff': latency_collapse_keff,
        'entropy_collapse_keff': entropy_collapse_keff,
        'fp_collapse_keff': fp_collapse_keff,
        'keff_critical': keff_critical,
    }

    # Functional degradation curves
    print()
    print("=" * 70)
    print("FUNCTIONAL DEGRADATION")
    print("=" * 70)
    print()

    keff_values = [r['keff'] for r in sweep_results]
    latency_rel = [r['latency_ms'] / baseline_latency for r in sweep_results]
    entropy_rel = [r['entropy'] / baseline_entropy if baseline_entropy > 0 else 0 for r in sweep_results]

    print("  k_eff vs Function (relative to baseline):")
    print(f"  {'k_eff':<8} {'Latency':<12} {'Entropy':<12}")
    print("  " + "-" * 35)
    for i, r in enumerate(sweep_results):
        print(f"  {keff_values[i]:<8.1f} {latency_rel[i]:<12.2f}x {entropy_rel[i]:<12.2f}x")

    # Success criteria
    print()
    print("=" * 70)
    print("E4 SUCCESS CRITERIA")
    print("=" * 70)

    found_threshold = keff_critical is not None
    measurable_degradation = any(l > 1.2 for l in latency_rel) or any(e < 0.8 for e in entropy_rel)

    print(f"""
  ✓/✗ k_eff threshold identified: {'✓' if found_threshold else '✗'}
  ✓/✗ Measurable degradation:     {'✓' if measurable_degradation else '✗'}
  ✓/✗ Multiple metrics measured:  ✓ (latency, entropy, FP rate)

  Overall: {'✓ PASS' if found_threshold else '✗ PARTIAL'}
""")

    results['success'] = {
        'threshold_found': found_threshold,
        'measurable_degradation': measurable_degradation,
    }

    # Summary
    print()
    print("=" * 70)
    print("E4 SUMMARY: OPERATIONAL COLLAPSE")
    print("=" * 70)

    print(f"""
  Baseline (healthy system):
  ──────────────────────────
  k_eff = {baseline_keff:.1f}
  Detection latency = {baseline_latency:.1f} ms
  Entropy = {baseline_entropy:.3f} bits/bit
  FP rate = {baseline_fp:.4f}

  Collapse threshold:
  ───────────────────
  k_eff_critical = {keff_critical if keff_critical else 'Not reached'}

  Operational definition:
  ───────────────────────
  "Collapse" occurs when k_eff < {keff_critical if keff_critical else 'TBD'},
  causing >50% degradation in detection latency or entropy.
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

    with open('/home/emoore/CIRISArray/experiments/expE4_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expE4_results.json")
