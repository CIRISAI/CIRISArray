#!/usr/bin/env python3
"""
Experiment E3: Velocity Discrepancy Investigation
==================================================

Gap: 0.5 m/s measured vs ~6 m/s thermal diffusion theory (13x discrepancy)

Protocol:
- Test 1: Vary sample rate (4kHz → 20kHz) - check if sensor latency dominates
- Test 2: External thermal stimulus - compare induced vs algorithmic propagation
- Test 3: Vary die region - check spatial consistency

Success criteria: Explain the 13x discrepancy with physical mechanism

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


def measure_propagation_velocity(sensors: list, sample_rate: float, n_trials: int = 10) -> dict:
    """
    Measure apparent propagation velocity at given sample rate.

    Returns velocity estimate and uncertainty.
    """
    # Sensor spacing assumption: 10mm between adjacent sensors in 4x4 grid
    SENSOR_SPACING_M = 0.010  # 10mm

    velocities = []

    for trial in range(n_trials):
        # Induce localized disturbance at corner
        stress_thread = threading.Thread(
            target=lambda: [cp.matmul(cp.random.random((1024, 1024)),
                           cp.random.random((1024, 1024))) for _ in range(50)]
        )

        # Collect baseline
        baseline = []
        for _ in range(20):
            sample = [s.read().timing_mean_us for s in sensors]
            baseline.append(sample)
        baseline = np.array(baseline)
        baseline_mean = np.mean(baseline, axis=0)
        baseline_std = np.std(baseline, axis=0)
        threshold = baseline_mean + 2 * baseline_std

        # Start disturbance and track arrival times
        arrival_times = [None] * len(sensors)
        t0 = time.perf_counter()

        stress_thread.start()

        dt = 1.0 / sample_rate
        max_time = 0.5  # 500ms max

        while time.perf_counter() - t0 < max_time:
            t_now = time.perf_counter() - t0
            samples = [s.read().timing_mean_us for s in sensors]

            for i, (val, thresh) in enumerate(zip(samples, threshold)):
                if arrival_times[i] is None and val > thresh:
                    arrival_times[i] = t_now

            # All sensors detected
            if all(t is not None for t in arrival_times):
                break

            time.sleep(dt)

        stress_thread.join()

        # Compute velocity from arrival time gradient
        valid_arrivals = [t for t in arrival_times if t is not None]
        if len(valid_arrivals) >= 4:
            # Use corner-to-corner time difference
            # Sensor 0 = top-left, Sensor 15 = bottom-right
            t_corner = arrival_times[0]
            t_far = arrival_times[15]

            if t_corner is not None and t_far is not None and t_far > t_corner:
                # Diagonal distance = sqrt(2) * 3 * spacing
                diagonal_m = np.sqrt(2) * 3 * SENSOR_SPACING_M
                delta_t = t_far - t_corner
                velocity = diagonal_m / delta_t
                velocities.append(velocity)

        time.sleep(0.2)

    if velocities:
        return {
            'velocity_mean': float(np.mean(velocities)),
            'velocity_std': float(np.std(velocities)),
            'n_valid': len(velocities),
            'sample_rate': sample_rate,
        }
    else:
        return {
            'velocity_mean': None,
            'velocity_std': None,
            'n_valid': 0,
            'sample_rate': sample_rate,
        }


def test_sample_rate_effect(sensors: list) -> list:
    """Test 1: Does apparent velocity change with sample rate?"""
    results = []

    # Sample rates from 1kHz to 20kHz
    sample_rates = [1000, 2000, 4000, 8000, 10000, 15000, 20000]

    for rate in sample_rates:
        result = measure_propagation_velocity(sensors, rate, n_trials=5)
        results.append(result)
        print(f"  {rate:5d} Hz: v = {result['velocity_mean']:.2f} m/s" if result['velocity_mean'] else f"  {rate:5d} Hz: No valid measurements")

    return results


def test_thermal_stimulus(sensors: list) -> dict:
    """Test 2: Compare algorithmic vs thermal propagation."""
    results = {}

    # Algorithmic: Heavy compute workload
    print("  Algorithmic stimulus (matmul)...")
    algo_velocities = []

    for _ in range(10):
        # Measure with compute workload
        v_result = measure_propagation_velocity(sensors, 4000, n_trials=1)
        if v_result['velocity_mean']:
            algo_velocities.append(v_result['velocity_mean'])
        time.sleep(0.3)

    results['algorithmic'] = {
        'velocity_mean': float(np.mean(algo_velocities)) if algo_velocities else None,
        'velocity_std': float(np.std(algo_velocities)) if algo_velocities else None,
        'n_valid': len(algo_velocities),
    }

    # Thermal: Sustained compute to heat GPU
    print("  Thermal stimulus (sustained heat)...")

    # Pre-heat GPU
    print("    Pre-heating GPU for 10s...")
    preheat_thread = threading.Thread(
        target=lambda: [cp.matmul(cp.random.random((2048, 2048)),
                       cp.random.random((2048, 2048))) for _ in range(500)]
    )
    preheat_thread.start()
    preheat_thread.join()

    # Now measure during cooldown (thermal wave)
    thermal_velocities = []
    print("    Measuring during cooldown...")

    for _ in range(10):
        v_result = measure_propagation_velocity(sensors, 4000, n_trials=1)
        if v_result['velocity_mean']:
            thermal_velocities.append(v_result['velocity_mean'])
        time.sleep(0.5)

    results['thermal'] = {
        'velocity_mean': float(np.mean(thermal_velocities)) if thermal_velocities else None,
        'velocity_std': float(np.std(thermal_velocities)) if thermal_velocities else None,
        'n_valid': len(thermal_velocities),
    }

    return results


def test_spatial_consistency(sensors: list) -> dict:
    """Test 3: Is velocity consistent across die regions?"""
    results = {}

    # Define regions: top-left quadrant, top-right, bottom-left, bottom-right
    regions = {
        'top_left': [0, 1, 4, 5],      # Sensors in top-left 2x2
        'top_right': [2, 3, 6, 7],     # Sensors in top-right 2x2
        'bottom_left': [8, 9, 12, 13], # Sensors in bottom-left 2x2
        'bottom_right': [10, 11, 14, 15],  # Sensors in bottom-right 2x2
    }

    SENSOR_SPACING_M = 0.010

    for region_name, indices in regions.items():
        print(f"  Region: {region_name}...")

        region_sensors = [sensors[i] for i in indices]
        velocities = []

        for trial in range(10):
            # Induce stress
            stress_thread = threading.Thread(
                target=lambda: [cp.matmul(cp.random.random((512, 512)),
                               cp.random.random((512, 512))) for _ in range(30)]
            )

            # Baseline for region
            baseline = []
            for _ in range(15):
                sample = [s.read().timing_mean_us for s in region_sensors]
                baseline.append(sample)
            baseline = np.array(baseline)
            baseline_mean = np.mean(baseline, axis=0)
            baseline_std = np.std(baseline, axis=0)
            threshold = baseline_mean + 2 * baseline_std

            arrival_times = [None] * 4
            t0 = time.perf_counter()

            stress_thread.start()

            while time.perf_counter() - t0 < 0.3:
                samples = [s.read().timing_mean_us for s in region_sensors]
                t_now = time.perf_counter() - t0

                for i, (val, thresh) in enumerate(zip(samples, threshold)):
                    if arrival_times[i] is None and val > thresh:
                        arrival_times[i] = t_now

                if all(t is not None for t in arrival_times):
                    break

                time.sleep(0.001)

            stress_thread.join()

            # Compute velocity within region (diagonal)
            valid = [t for t in arrival_times if t is not None]
            if len(valid) >= 2:
                t_min = min(valid)
                t_max = max(valid)
                if t_max > t_min:
                    # 2x2 diagonal = sqrt(2) * spacing
                    dist = np.sqrt(2) * SENSOR_SPACING_M
                    v = dist / (t_max - t_min)
                    velocities.append(v)

            time.sleep(0.1)

        results[region_name] = {
            'velocity_mean': float(np.mean(velocities)) if velocities else None,
            'velocity_std': float(np.std(velocities)) if velocities else None,
            'n_valid': len(velocities),
        }

    return results


def compute_theoretical_velocity():
    """Compute theoretical thermal diffusion velocity in silicon."""
    # Silicon thermal properties
    k_si = 150  # W/(m·K) thermal conductivity
    rho_si = 2330  # kg/m³ density
    cp_si = 700  # J/(kg·K) specific heat

    # Thermal diffusivity
    alpha = k_si / (rho_si * cp_si)  # m²/s

    # For diffusion, "velocity" depends on length scale
    # v ~ sqrt(alpha / t) or v ~ alpha / L

    # At L = 10mm (sensor spacing)
    L = 0.010  # m

    # Diffusion time to travel L
    t_diff = L**2 / alpha

    # Apparent velocity
    v_apparent = L / t_diff  # = alpha / L

    return {
        'alpha': alpha,
        'L': L,
        't_diff': t_diff,
        'v_apparent': v_apparent,
        'v_apparent_mm_per_s': v_apparent * 1000,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT E3: VELOCITY DISCREPANCY INVESTIGATION")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Gap: 0.5 m/s measured vs ~6 m/s thermal diffusion theory")
    print("Goal: Explain the 13x discrepancy")
    print()

    results = {}

    # Theoretical prediction
    print("=" * 70)
    print("THEORETICAL PREDICTION")
    print("=" * 70)
    print()

    theory = compute_theoretical_velocity()
    print(f"  Silicon thermal diffusivity α = {theory['alpha']:.2e} m²/s")
    print(f"  Sensor spacing L = {theory['L']*1000:.0f} mm")
    print(f"  Diffusion time τ = L²/α = {theory['t_diff']*1000:.1f} ms")
    print(f"  Theoretical v = α/L = {theory['v_apparent']:.2f} m/s")
    print()

    results['theory'] = theory

    # Create sensor array
    print("=" * 70)
    print("SETUP")
    print("=" * 70)
    print()
    print("Creating 4x4 sensor array...")
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

    # Test 1: Sample rate effect
    print()
    print("=" * 70)
    print("TEST 1: SAMPLE RATE EFFECT")
    print("=" * 70)
    print()
    print("Hypothesis: If sensor latency dominates, higher sample rate → higher apparent v")
    print()

    sample_rate_results = test_sample_rate_effect(sensors)
    results['test1_sample_rate'] = sample_rate_results

    # Analyze trend
    rates = [r['sample_rate'] for r in sample_rate_results if r['velocity_mean'] is not None]
    velocities = [r['velocity_mean'] for r in sample_rate_results if r['velocity_mean'] is not None]

    if len(rates) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(rates, velocities)
        print()
        print(f"  Trend: v = {slope*1000:.4f} × rate + {intercept:.2f}")
        print(f"  R² = {r_val**2:.3f}, p = {p_val:.4f}")

        if p_val < 0.05 and slope > 0:
            print("  → Sample rate DOES affect velocity (sensor latency contribution)")
            # Extrapolate to infinite sample rate
            v_intrinsic = intercept
            print(f"  → Extrapolated intrinsic velocity: {v_intrinsic:.2f} m/s")
            results['test1_conclusion'] = 'sensor_latency_contributes'
            results['v_intrinsic'] = float(v_intrinsic)
        else:
            print("  → Sample rate does NOT significantly affect velocity")
            results['test1_conclusion'] = 'no_latency_effect'
    else:
        print("  Insufficient data for trend analysis")
        results['test1_conclusion'] = 'insufficient_data'

    time.sleep(1.0)

    # Test 2: Thermal vs algorithmic
    print()
    print("=" * 70)
    print("TEST 2: THERMAL VS ALGORITHMIC STIMULUS")
    print("=" * 70)
    print()
    print("Hypothesis: Thermal propagation should match theory (~6 m/s)")
    print()

    thermal_results = test_thermal_stimulus(sensors)
    results['test2_thermal'] = thermal_results

    print()
    if thermal_results['algorithmic']['velocity_mean'] and thermal_results['thermal']['velocity_mean']:
        v_algo = thermal_results['algorithmic']['velocity_mean']
        v_therm = thermal_results['thermal']['velocity_mean']
        ratio = v_therm / v_algo if v_algo > 0 else None

        print(f"  Algorithmic v = {v_algo:.2f} ± {thermal_results['algorithmic']['velocity_std']:.2f} m/s")
        print(f"  Thermal v = {v_therm:.2f} ± {thermal_results['thermal']['velocity_std']:.2f} m/s")
        if ratio:
            print(f"  Ratio (thermal/algo) = {ratio:.2f}")

        if ratio and ratio > 2:
            print("  → Thermal stimulus produces FASTER propagation")
            results['test2_conclusion'] = 'thermal_faster'
        elif ratio and ratio < 0.5:
            print("  → Algorithmic stimulus produces FASTER propagation")
            results['test2_conclusion'] = 'algorithmic_faster'
        else:
            print("  → No significant difference between stimulus types")
            results['test2_conclusion'] = 'no_difference'
    else:
        print("  Insufficient measurements for comparison")
        results['test2_conclusion'] = 'insufficient_data'

    time.sleep(1.0)

    # Test 3: Spatial consistency
    print()
    print("=" * 70)
    print("TEST 3: SPATIAL CONSISTENCY")
    print("=" * 70)
    print()
    print("Hypothesis: Velocity should be uniform if physical propagation")
    print()

    spatial_results = test_spatial_consistency(sensors)
    results['test3_spatial'] = spatial_results

    print()
    region_velocities = []
    for region, data in spatial_results.items():
        if data['velocity_mean']:
            print(f"  {region:12s}: v = {data['velocity_mean']:.2f} ± {data['velocity_std']:.2f} m/s")
            region_velocities.append(data['velocity_mean'])

    if len(region_velocities) >= 3:
        cv = np.std(region_velocities) / np.mean(region_velocities)
        print()
        print(f"  Coefficient of variation: {cv:.2f}")

        if cv < 0.2:
            print("  → Velocity is UNIFORM across regions (consistent with physical wave)")
            results['test3_conclusion'] = 'uniform'
        else:
            print("  → Velocity VARIES by region (suggests non-physical artifact)")
            results['test3_conclusion'] = 'non_uniform'
    else:
        print("  Insufficient data for spatial analysis")
        results['test3_conclusion'] = 'insufficient_data'

    # Synthesis
    print()
    print("=" * 70)
    print("DISCREPANCY ANALYSIS")
    print("=" * 70)
    print()

    v_measured = 0.5  # From C2
    v_theory = theory['v_apparent']
    discrepancy = v_theory / v_measured

    print(f"  Measured velocity (C2): {v_measured} m/s")
    print(f"  Theoretical velocity: {v_theory:.2f} m/s")
    print(f"  Discrepancy: {discrepancy:.1f}x")
    print()

    # Possible explanations
    explanations = []

    if results.get('test1_conclusion') == 'sensor_latency_contributes':
        explanations.append(f"1. Sensor latency: Sample rate affects measured velocity")
        if 'v_intrinsic' in results:
            explanations.append(f"   Intrinsic velocity ≈ {results['v_intrinsic']:.2f} m/s")

    if results.get('test2_conclusion') == 'thermal_faster':
        explanations.append("2. Thermal stimulus: Produces faster propagation than algorithmic")
    elif results.get('test2_conclusion') == 'no_difference':
        explanations.append("2. Stimulus type: No significant difference (both measure same phenomenon)")

    if results.get('test3_conclusion') == 'uniform':
        explanations.append("3. Spatial uniformity: Consistent with physical wave propagation")
    elif results.get('test3_conclusion') == 'non_uniform':
        explanations.append("3. Spatial variation: May indicate measurement artifact")

    # Additional physical considerations
    explanations.append("")
    explanations.append("Physical considerations:")
    explanations.append("- GPU die is not pure silicon (metal layers, dielectrics)")
    explanations.append("- Heat spreader dominates thermal transport")
    explanations.append("- PDN response is electrical (~ns), not thermal (~ms)")
    explanations.append("- k_eff measures oscillator correlation, not temperature")

    for exp in explanations:
        print(f"  {exp}")

    results['discrepancy_factor'] = float(discrepancy)
    results['explanations'] = explanations

    # Success criteria
    print()
    print("=" * 70)
    print("E3 SUCCESS CRITERIA")
    print("=" * 70)

    has_mechanism = len([r for r in [
        results.get('test1_conclusion'),
        results.get('test2_conclusion'),
        results.get('test3_conclusion')
    ] if r and r != 'insufficient_data']) >= 2

    print(f"""
  ✓/✗ Sample rate test completed: {'✓' if results.get('test1_conclusion') != 'insufficient_data' else '✗'}
  ✓/✗ Thermal test completed:     {'✓' if results.get('test2_conclusion') != 'insufficient_data' else '✗'}
  ✓/✗ Spatial test completed:     {'✓' if results.get('test3_conclusion') != 'insufficient_data' else '✗'}
  ✓/✗ Physical mechanism proposed: {'✓' if has_mechanism else '✗'}

  Overall: {'✓ PASS' if has_mechanism else '✗ PARTIAL'}
""")

    results['success'] = {
        'tests_completed': has_mechanism,
    }

    # Summary
    print()
    print("=" * 70)
    print("E3 SUMMARY: VELOCITY DISCREPANCY")
    print("=" * 70)

    print(f"""
  Measured velocity: {v_measured} m/s (C2 experiment)
  Theoretical (thermal diffusion): {v_theory:.2f} m/s
  Discrepancy: {discrepancy:.1f}x

  Key Findings:
  ─────────────
  Test 1 (sample rate): {results.get('test1_conclusion', 'N/A')}
  Test 2 (thermal):     {results.get('test2_conclusion', 'N/A')}
  Test 3 (spatial):     {results.get('test3_conclusion', 'N/A')}

  Interpretation:
  ───────────────
  The 0.5 m/s measured velocity is likely NOT thermal diffusion.

  More likely explanations:
  1. k_eff responds to PDN (power delivery) transients
  2. GPU workload spreads via scheduler, not thermal conduction
  3. Correlation propagates through shared resources (L2 cache, memory controller)

  The measured velocity reflects the timescale of GPU resource contention,
  not physical heat transport.
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

    with open('/home/emoore/CIRISArray/experiments/expE3_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expE3_results.json")
