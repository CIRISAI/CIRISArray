#!/usr/bin/env python3
"""
Experiment A9: SNR Scaling with N Sensors
==========================================

Test if N sensors improve detection by √N.

Protocol (from RATCHET_UPDATE):
- Create 16 sensors
- Measure SNR with 1, 2, 4, 8, 16 sensors (averaged)
- Check if SNR scales with √N

Success criteria: SNR scales with √N or better

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
    print("EXPERIMENT A9: SNR SCALING WITH N SENSORS")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Does averaging N sensors improve SNR by √N?")
    print()

    results = {}

    # =========================================================================
    # Create 16 sensors
    # =========================================================================
    print("Creating 16 sensors...")
    n_max = 16

    sensors = []
    for i in range(n_max):
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
    # SNR Scaling Test
    # =========================================================================
    print("=" * 70)
    print("SNR SCALING TEST")
    print("=" * 70)
    print()
    print("  Using 30% workload for moderate signal")
    print()

    sensor_counts = [1, 2, 4, 8, 16]
    scaling_results = []

    print(f"  {'N':<6} {'SNR':<12} {'Theory √N':<12} {'Ratio':<12} {'Status'}")
    print("  " + "-" * 52)

    for n in sensor_counts:
        snrs = []
        use_sensors = sensors[:n]

        for trial in range(10):
            # Collect baseline from N sensors, average
            baseline_samples = []
            for s in use_sensors:
                samples = collect_samples_fast(s, n_samples=50)
                baseline_samples.append(np.mean(samples))

            baseline_avg = np.mean(baseline_samples)
            baseline_std = np.std(baseline_samples) if len(baseline_samples) > 1 else np.std(samples)

            # Collect with workload
            workload_thread = threading.Thread(target=create_workload, args=(3.0, 0.3))
            workload_thread.start()
            time.sleep(0.5)

            workload_samples = []
            for s in use_sensors:
                samples = collect_samples_fast(s, n_samples=50)
                workload_samples.append(np.mean(samples))

            workload_thread.join()

            workload_avg = np.mean(workload_samples)

            # SNR = (signal - baseline) / noise
            signal = workload_avg - baseline_avg
            noise = baseline_std if baseline_std > 0 else 1
            snr = signal / noise
            snrs.append(snr)

            time.sleep(0.5)

        mean_snr = np.mean(snrs)
        theory_sqrt_n = np.sqrt(n)

        # Ratio to single sensor (normalized to N=1)
        if n == 1:
            snr_n1 = mean_snr
        ratio_to_sqrt_n = mean_snr / snr_n1 / theory_sqrt_n if theory_sqrt_n > 0 else 0

        status = "✓" if ratio_to_sqrt_n > 0.7 else "✗"  # Should be close to 1.0

        scaling_results.append({
            'n': n,
            'snr': float(mean_snr),
            'theory_sqrt_n': float(theory_sqrt_n),
            'ratio': float(ratio_to_sqrt_n),
        })

        print(f"  {n:<6} {mean_snr:<12.1f} {theory_sqrt_n:<12.2f} {ratio_to_sqrt_n:<12.2f} {status}")

    results['scaling'] = scaling_results

    # =========================================================================
    # Fit Power Law
    # =========================================================================
    print()
    print("=" * 70)
    print("POWER LAW FIT")
    print("=" * 70)
    print()

    n_vals = np.array([r['n'] for r in scaling_results])
    snr_vals = np.array([r['snr'] for r in scaling_results])

    # Fit SNR = A * N^β
    log_n = np.log(n_vals)
    log_snr = np.log(snr_vals)
    beta, log_a = np.polyfit(log_n, log_snr, 1)
    a = np.exp(log_a)

    # R^2
    predicted = a * np.power(n_vals, beta)
    ss_res = np.sum((snr_vals - predicted) ** 2)
    ss_tot = np.sum((snr_vals - np.mean(snr_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"  Fit: SNR = {a:.1f} × N^{beta:.2f}")
    print(f"  R² = {r_squared:.3f}")
    print()
    print(f"  Expected: β = 0.5 (√N scaling)")
    print(f"  Actual:   β = {beta:.2f}")
    print()

    if beta > 0.4 and beta < 0.6:
        print(f"  ✓ √N scaling CONFIRMED")
        scaling_confirmed = True
    elif beta > 0.6:
        print(f"  ✓ BETTER than √N scaling!")
        scaling_confirmed = True
    else:
        print(f"  ✗ Scaling is weaker than √N")
        scaling_confirmed = False

    results['fit'] = {
        'a': float(a),
        'beta': float(beta),
        'r_squared': float(r_squared),
        'scaling_confirmed': scaling_confirmed,
    }

    # =========================================================================
    # Noise Reduction Analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("NOISE REDUCTION ANALYSIS")
    print("=" * 70)
    print()

    # Calculate noise reduction at each N
    snr_n1 = scaling_results[0]['snr']
    print(f"  Noise reduction vs single sensor:")
    print(f"  {'N':<6} {'SNR/SNR₁':<12} {'Noise Reduction':<15}")
    print("  " + "-" * 33)

    for r in scaling_results:
        snr_ratio = r['snr'] / snr_n1 if snr_n1 > 0 else 0
        noise_reduction_db = 20 * np.log10(snr_ratio) if snr_ratio > 0 else 0
        print(f"  {r['n']:<6} {snr_ratio:<12.2f}x {noise_reduction_db:+.1f} dB")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("A9 SUMMARY: SNR SCALING")
    print("=" * 70)

    improvement_16 = scaling_results[-1]['snr'] / scaling_results[0]['snr'] if scaling_results[0]['snr'] > 0 else 0
    theoretical_16 = np.sqrt(16)  # = 4

    print(f"""
  SNR Scaling Results:
  ────────────────────
  Fit: SNR = {a:.1f} × N^{beta:.2f} (R² = {r_squared:.3f})

  Expected exponent: 0.5 (√N)
  Actual exponent:   {beta:.2f}

  16-sensor improvement:
    Actual:   {improvement_16:.1f}x
    Theory:   {theoretical_16:.1f}x (√16)
    Status:   {'✓ PASS' if scaling_confirmed else '✗ FAIL'}

  Conclusion:
  ───────────
  {'Array averaging works! SNR scales with √N or better.' if scaling_confirmed else 'Array averaging shows limited improvement.'}
""")

    results['summary'] = {
        'beta': float(beta),
        'improvement_16x': float(improvement_16),
        'theoretical_16x': float(theoretical_16),
        'scaling_confirmed': scaling_confirmed,
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

    with open('/home/emoore/CIRISArray/experiments/expA9_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expA9_results.json")
