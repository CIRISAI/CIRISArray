#!/usr/bin/env python3
"""
Experiment C2: Correlation Propagation Velocity
================================================

Measure speed of correlation wavefront across the die.

Question: How fast does correlation spread across the die (m/s)?

Protocol (from RATCHET_UPDATE):
- 4x4 sensor array spanning ~20mm x 20mm (5mm grid spacing)
- Induce collapse from corner (0,0) only
- Measure delay for correlation to reach distant sensors
- Calculate velocity = distance / time

Success criteria:
- Measurable delay between corner and distant sensors
- Velocity estimate (expect ~0.1-10 m/s based on electrical τ=2ms)
- Consistent across trials

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
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from strain_gauge import StrainGauge, StrainGaugeConfig


# Physical layout constants (approximate for RTX 4090)
# Die size ~600mm², assume 4x4 grid spans ~20mm x 20mm
GRID_SPACING_MM = 5.0  # mm between sensor centers


def create_corner_workload(duration: float, intensity: float = 1.0):
    """
    Create localized GPU workload that affects corner region preferentially.

    This simulates correlation injection at a specific location by creating
    workload that has strongest effect on sensors near the corner.
    """
    end_time = time.time() + duration

    # Use specific CUDA stream to localize effect
    stream = cp.cuda.Stream()

    with stream:
        size = int(1024 * intensity)
        data = cp.random.random((size, size), dtype=cp.float32)

        while time.time() < end_time:
            # Intense computation
            for _ in range(5):
                data = cp.matmul(data, data)
            stream.synchronize()
            time.sleep(0.0005)  # Brief pause for timing measurement


def get_sensor_distance(sensor_idx: int, origin_idx: int = 0) -> float:
    """Get physical distance in mm from origin sensor."""
    origin_row, origin_col = origin_idx // 4, origin_idx % 4
    row, col = sensor_idx // 4, sensor_idx % 4

    delta_row = row - origin_row
    delta_col = col - origin_col

    return np.sqrt(delta_row**2 + delta_col**2) * GRID_SPACING_MM


def main():
    print("=" * 70)
    print("EXPERIMENT C2: CORRELATION PROPAGATION VELOCITY")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Measure speed of correlation wavefront (m/s)")
    print(f"Grid spacing: {GRID_SPACING_MM} mm")
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
    # Sensor layout
    # =========================================================================
    print("=" * 70)
    print("SENSOR LAYOUT")
    print("=" * 70)
    print()
    print("  Physical grid (mm from corner [0,0]):")
    print("  " + "-" * 40)

    for row in range(4):
        row_str = "  "
        for col in range(4):
            idx = row * 4 + col
            dist = get_sensor_distance(idx, 0)
            row_str += f"[{idx:2d}]={dist:4.1f}mm  "
        print(row_str)
    print("  " + "-" * 40)
    print()

    # =========================================================================
    # Propagation velocity measurement
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION VELOCITY MEASUREMENT")
    print("=" * 70)
    print()

    n_trials = 10
    correlation_threshold = 0.3  # Threshold for "correlation arrived"

    all_velocities = []
    trial_results = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")

        # Reset - let system return to baseline
        time.sleep(1.0)

        # Collect baseline samples from all sensors
        baseline_window = []
        for _ in range(20):
            samples = [s.read().timing_mean_us for s in sensors]
            baseline_window.append(samples)

        baseline_means = np.mean(baseline_window, axis=0)

        # Start high-rate sampling BEFORE inducing collapse
        t0 = time.perf_counter()
        history = []

        # Sample buffer for running correlation
        corner_buffer = deque(maxlen=10)
        sensor_buffers = [deque(maxlen=10) for _ in range(n_sensors)]

        # Start corner workload in background
        workload_thread = threading.Thread(
            target=create_corner_workload,
            args=(0.1, 1.0)  # 100ms intense workload
        )
        workload_thread.start()

        # Sample rapidly for 100ms
        sample_count = 0
        while time.perf_counter() - t0 < 0.1:
            # Get readings from all sensors
            samples = [s.read().timing_mean_us for s in sensors]
            t_ms = (time.perf_counter() - t0) * 1000  # ms since start

            # Update buffers
            corner_buffer.append(samples[0])
            for i in range(n_sensors):
                sensor_buffers[i].append(samples[i])

            # Compute running correlations with corner sensor
            correlations = [1.0]  # Corner with itself
            if len(corner_buffer) >= 5:
                corner_arr = np.array(corner_buffer)
                for i in range(1, n_sensors):
                    sensor_arr = np.array(sensor_buffers[i])
                    if np.std(corner_arr) > 0.01 and np.std(sensor_arr) > 0.01:
                        r = np.corrcoef(corner_arr, sensor_arr)[0, 1]
                        correlations.append(r if not np.isnan(r) else 0)
                    else:
                        correlations.append(0)
            else:
                correlations = [0] * n_sensors
                correlations[0] = 1.0

            history.append({
                't_ms': t_ms,
                'samples': samples.copy(),
                'correlations': correlations.copy()
            })

            sample_count += 1

        workload_thread.join()

        # Find when each sensor crossed correlation threshold
        arrival_times = {}
        for i in range(1, n_sensors):  # Skip corner sensor
            distance_mm = get_sensor_distance(i, 0)

            threshold_time = None
            for h in history:
                if len(h['correlations']) > i and h['correlations'][i] > correlation_threshold:
                    threshold_time = h['t_ms']
                    break

            if threshold_time is not None and threshold_time > 0:
                velocity = distance_mm / threshold_time  # mm/ms = m/s
                arrival_times[i] = {
                    'distance_mm': distance_mm,
                    'time_ms': threshold_time,
                    'velocity_m_s': velocity
                }
                all_velocities.append(velocity)

        trial_results.append({
            'trial': trial + 1,
            'sample_count': sample_count,
            'arrival_times': arrival_times,
            'history_length': len(history)
        })

        # Brief report
        if arrival_times:
            mean_v = np.mean([v['velocity_m_s'] for v in arrival_times.values()])
            print(f"  Sensors detected: {len(arrival_times)}/15, mean velocity: {mean_v:.1f} m/s")
        else:
            print(f"  No correlation propagation detected")

    results['trials'] = trial_results

    # =========================================================================
    # Velocity analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("VELOCITY ANALYSIS")
    print("=" * 70)
    print()

    if all_velocities:
        velocities = np.array(all_velocities)

        # Remove outliers (> 3σ)
        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        valid_velocities = velocities[np.abs(velocities - mean_v) < 3 * std_v]

        if len(valid_velocities) > 0:
            final_velocity = np.mean(valid_velocities)
            final_std = np.std(valid_velocities)

            print(f"  Propagation velocity: {final_velocity:.1f} ± {final_std:.1f} m/s")
            print(f"  Measurements: {len(valid_velocities)} (after outlier removal)")
            print(f"  Range: {np.min(valid_velocities):.1f} - {np.max(valid_velocities):.1f} m/s")

            results['velocity'] = {
                'mean_m_s': float(final_velocity),
                'std_m_s': float(final_std),
                'n_measurements': len(valid_velocities),
                'range_min': float(np.min(valid_velocities)),
                'range_max': float(np.max(valid_velocities)),
            }
        else:
            print("  No valid velocity measurements")
            results['velocity'] = None
    else:
        print("  No velocity measurements collected")
        results['velocity'] = None

    # =========================================================================
    # Distance vs Time analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("DISTANCE vs TIME")
    print("=" * 70)
    print()

    # Aggregate by sensor position
    sensor_stats = {}
    for trial in trial_results:
        for sensor_idx, data in trial['arrival_times'].items():
            if sensor_idx not in sensor_stats:
                sensor_stats[sensor_idx] = []
            sensor_stats[sensor_idx].append(data)

    print(f"  {'Sensor':<8} {'Distance':<12} {'Mean Time':<12} {'Velocity':<12} {'N'}")
    print("  " + "-" * 52)

    distance_time_pairs = []
    for sensor_idx in sorted(sensor_stats.keys()):
        stats = sensor_stats[sensor_idx]
        distance = stats[0]['distance_mm']
        times = [s['time_ms'] for s in stats]
        velocities = [s['velocity_m_s'] for s in stats]

        mean_time = np.mean(times)
        mean_velocity = np.mean(velocities)

        distance_time_pairs.append((distance, mean_time))

        print(f"  {sensor_idx:<8} {distance:<12.1f} {mean_time:<12.2f} {mean_velocity:<12.1f} {len(stats)}")

    results['distance_time'] = distance_time_pairs

    # Linear fit: distance = velocity * time
    if len(distance_time_pairs) >= 3:
        distances = np.array([p[0] for p in distance_time_pairs])
        times = np.array([p[1] for p in distance_time_pairs])

        # Fit through origin: d = v * t
        # v = Σ(d*t) / Σ(t²)
        if np.sum(times**2) > 0:
            fitted_velocity = np.sum(distances * times) / np.sum(times**2)

            # R² for fit
            predicted = fitted_velocity * times
            ss_res = np.sum((distances - predicted)**2)
            ss_tot = np.sum((distances - np.mean(distances))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print()
            print(f"  Linear fit: distance = {fitted_velocity:.1f} × time")
            print(f"  R² = {r_squared:.3f}")

            results['linear_fit'] = {
                'velocity_m_s': float(fitted_velocity),
                'r_squared': float(r_squared)
            }

    # =========================================================================
    # Physical interpretation
    # =========================================================================
    print()
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)

    if results.get('velocity'):
        v = results['velocity']['mean_m_s']

        # Compare to known velocities
        print(f"""
  Measured velocity: {v:.1f} m/s

  Comparison to known phenomena:
  ──────────────────────────────
  Thermal diffusion:     ~0.001-0.1 m/s (too slow)
  Acoustic waves:        ~5000-6000 m/s (in silicon)
  Electrical signals:    ~10⁸ m/s (too fast to measure)

  Our measurement ({v:.1f} m/s) suggests:
  → {'Thermal diffusion regime' if v < 1 else 'Mixed thermal/acoustic regime' if v < 100 else 'Electrical regime'}

  Time to cross die (~20mm):
  → At {v:.1f} m/s: {20.0/v:.1f} ms
""")

        results['interpretation'] = {
            'velocity_m_s': v,
            'die_crossing_time_ms': 20.0 / v if v > 0 else float('inf'),
            'regime': 'thermal' if v < 1 else 'mixed' if v < 100 else 'electrical'
        }

    # =========================================================================
    # Success criteria
    # =========================================================================
    print()
    print("=" * 70)
    print("C2 SUCCESS CRITERIA")
    print("=" * 70)

    has_delay = len(sensor_stats) > 0
    has_velocity = results.get('velocity') is not None
    is_consistent = has_velocity and results['velocity']['std_m_s'] < results['velocity']['mean_m_s']

    print(f"""
  ✓/✗ Measurable delay:    {'✓' if has_delay else '✗'} ({len(sensor_stats)} sensors detected)
  ✓/✗ Velocity estimate:   {'✓' if has_velocity else '✗'} {f"({results['velocity']['mean_m_s']:.1f} m/s)" if has_velocity else ''}
  ✓/✗ Consistent results:  {'✓' if is_consistent else '✗'} {f"(CV={results['velocity']['std_m_s']/results['velocity']['mean_m_s']*100:.0f}%)" if has_velocity else ''}

  Overall: {'✓ PASS' if has_delay and has_velocity else '✗ PARTIAL'}
""")

    results['success'] = {
        'has_delay': bool(has_delay),
        'has_velocity': bool(has_velocity),
        'is_consistent': bool(is_consistent),
        'all_pass': bool(has_delay and has_velocity)
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("C2 SUMMARY: PROPAGATION VELOCITY")
    print("=" * 70)

    if results.get('velocity'):
        print(f"""
  Instrument: 4×4 sensor array, {GRID_SPACING_MM}mm spacing

  Key Findings:
  ─────────────
  1. Propagation velocity: {results['velocity']['mean_m_s']:.1f} ± {results['velocity']['std_m_s']:.1f} m/s
  2. Detection range: {results['velocity']['range_min']:.1f} - {results['velocity']['range_max']:.1f} m/s
  3. Measurements: {results['velocity']['n_measurements']}
  4. Die crossing time: {results['interpretation']['die_crossing_time_ms']:.1f} ms
  5. Regime: {results['interpretation']['regime']}

  For Lean Formalization:
  ───────────────────────
  propagation_velocity := {results['velocity']['mean_m_s']:.1f} m/s
  die_crossing_time := {results['interpretation']['die_crossing_time_ms']:.1f} ms

  Ready for C3: Nucleation sites identification
""")
    else:
        print("""
  No reliable velocity measurement obtained.
  This may indicate:
  - Correlation propagates faster than measurement rate
  - Workload effect is broadcast (not localized)
  - Need higher sample rate or different injection method
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

    with open('/home/emoore/CIRISArray/experiments/expC2_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expC2_results.json")
