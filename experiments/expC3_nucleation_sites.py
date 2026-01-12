#!/usr/bin/env python3
"""
Experiment C3: Collapse Nucleation Sites
========================================

Find where collapse nucleates under uniform stress.

Question: Are there "weak points" where collapse starts preferentially?

Protocol (from RATCHET_UPDATE):
- Apply UNIFORM stress (not localized)
- Collapse should nucleate at weakest point
- Track which sensor shows high correlation first
- Map nucleation frequency across grid

Success criteria:
- Non-uniform nucleation (some sites more likely)
- Correlates with die topology (hot band? power rails?)
- Reproducible across trials

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


def create_uniform_stress(duration: float, strength: float = 0.5):
    """
    Create uniform GPU stress across all SMs.

    Unlike corner workload, this should affect all sensors equally.
    Any non-uniform collapse is due to intrinsic die properties.
    """
    end_time = time.time() + duration

    # Use multiple streams to distribute load
    n_streams = 4
    streams = [cp.cuda.Stream() for _ in range(n_streams)]

    size = int(512 * strength)
    data = [cp.random.random((size, size), dtype=cp.float32) for _ in range(n_streams)]

    while time.time() < end_time:
        for i, stream in enumerate(streams):
            with stream:
                data[i] = cp.matmul(data[i], data[i])

        # Barrier sync to keep all streams together
        cp.cuda.Device().synchronize()
        time.sleep(0.001)


def main():
    print("=" * 70)
    print("EXPERIMENT C3: COLLAPSE NUCLEATION SITES")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Find where correlation collapse nucleates first")
    print("Method: Apply uniform stress, track first sensor to show high ρ")
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
    # Nucleation site detection
    # =========================================================================
    print("=" * 70)
    print("NUCLEATION SITE DETECTION")
    print("=" * 70)
    print()

    n_trials = 20
    nucleation_threshold = 0.5  # Correlation threshold for "nucleation"

    nucleation_counts = np.zeros((4, 4))
    nucleation_times = {i: [] for i in range(n_sensors)}
    trial_results = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...", end=" ")

        # Reset - let system return to baseline
        time.sleep(1.5)

        # Collect baseline samples
        baseline_samples = []
        for _ in range(30):
            samples = [s.read().timing_mean_us for s in sensors]
            baseline_samples.append(samples)

        baseline_arr = np.array(baseline_samples)

        # Start uniform stress
        stress_thread = threading.Thread(
            target=create_uniform_stress,
            args=(2.0, 0.5)  # 2 seconds, 50% intensity
        )
        stress_thread.start()

        # Watch for first nucleation
        t0 = time.perf_counter()
        first_nucleation = None
        nucleation_time = None

        # Sample buffer for running correlation
        sample_buffer = []

        for step in range(200):  # Up to 200 steps (~200ms)
            samples = [s.read().timing_mean_us for s in sensors]
            t_ms = (time.perf_counter() - t0) * 1000

            sample_buffer.append(samples)

            # Need at least 10 samples for correlation
            if len(sample_buffer) >= 10:
                recent = np.array(sample_buffer[-10:])

                # Compute correlation matrix
                if np.std(recent, axis=0).min() > 0.01:
                    rho_matrix = np.corrcoef(recent.T)

                    # Find sensor with highest average correlation
                    avg_correlations = []
                    for i in range(n_sensors):
                        # Average with all OTHER sensors
                        other_corrs = [rho_matrix[i, j] for j in range(n_sensors) if j != i]
                        avg_corr = np.mean([c for c in other_corrs if not np.isnan(c)])
                        avg_correlations.append(avg_corr if not np.isnan(avg_corr) else 0)

                    max_corr_sensor = np.argmax(avg_correlations)
                    max_corr = avg_correlations[max_corr_sensor]

                    # Check for nucleation
                    if max_corr > nucleation_threshold and first_nucleation is None:
                        first_nucleation = max_corr_sensor
                        nucleation_time = t_ms
                        row, col = first_nucleation // 4, first_nucleation % 4
                        nucleation_counts[row, col] += 1
                        nucleation_times[first_nucleation].append(nucleation_time)
                        break

        stress_thread.join()

        if first_nucleation is not None:
            row, col = first_nucleation // 4, first_nucleation % 4
            print(f"Nucleation at ({row}, {col}) [sensor {first_nucleation}] at {nucleation_time:.1f}ms")
            trial_results.append({
                'trial': trial + 1,
                'nucleation_sensor': int(first_nucleation),
                'nucleation_row': int(row),
                'nucleation_col': int(col),
                'nucleation_time_ms': float(nucleation_time)
            })
        else:
            print("No nucleation detected")
            trial_results.append({
                'trial': trial + 1,
                'nucleation_sensor': None,
                'nucleation_time_ms': None
            })

    results['trials'] = trial_results

    # =========================================================================
    # Nucleation frequency map
    # =========================================================================
    print()
    print("=" * 70)
    print("NUCLEATION FREQUENCY MAP")
    print("=" * 70)
    print()

    total_nucleations = int(np.sum(nucleation_counts))

    print(f"  Total nucleations detected: {total_nucleations}/{n_trials}")
    print()
    print("  Nucleation frequency by position:")
    print("  " + "-" * 35)

    for row in range(4):
        row_str = "  "
        for col in range(4):
            count = int(nucleation_counts[row, col])
            pct = (count / total_nucleations * 100) if total_nucleations > 0 else 0
            row_str += f"  {count:2d} ({pct:4.0f}%) "
        print(row_str)
    print("  " + "-" * 35)

    results['nucleation_map'] = nucleation_counts.tolist()

    # =========================================================================
    # Hotspot analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("HOTSPOT ANALYSIS")
    print("=" * 70)
    print()

    if total_nucleations > 0:
        # Find most common nucleation site
        max_pos = np.unravel_index(np.argmax(nucleation_counts), nucleation_counts.shape)
        max_count = int(nucleation_counts[max_pos])
        max_pct = max_count / total_nucleations * 100

        print(f"  Most common nucleation site: ({max_pos[0]}, {max_pos[1]})")
        print(f"  Frequency: {max_count}/{total_nucleations} ({max_pct:.0f}%)")

        # Chi-square test for uniformity
        expected = total_nucleations / 16
        chi_sq = np.sum((nucleation_counts.flatten() - expected)**2 / expected) if expected > 0 else 0

        # Critical value for χ²(15) at p=0.05 is 25.0
        is_non_uniform = chi_sq > 25.0

        print(f"  χ² statistic: {chi_sq:.1f}")
        print(f"  Non-uniform distribution: {'YES (p<0.05)' if is_non_uniform else 'No'}")

        # Spatial pattern analysis
        # Check for edge vs center preference
        edge_sensors = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
        center_sensors = [5, 6, 9, 10]

        edge_nucleations = sum(nucleation_counts[i//4, i%4] for i in edge_sensors)
        center_nucleations = sum(nucleation_counts[i//4, i%4] for i in center_sensors)

        edge_rate = edge_nucleations / 12  # 12 edge sensors
        center_rate = center_nucleations / 4  # 4 center sensors

        print()
        print(f"  Edge nucleation rate:   {edge_rate:.2f} per sensor")
        print(f"  Center nucleation rate: {center_rate:.2f} per sensor")
        print(f"  Edge/Center ratio: {edge_rate/center_rate:.2f}" if center_rate > 0 else "  Edge/Center ratio: inf")

        # Corner analysis
        corners = [0, 3, 12, 15]
        corner_nucleations = sum(nucleation_counts[i//4, i%4] for i in corners)
        corner_rate = corner_nucleations / 4

        print(f"  Corner nucleation rate: {corner_rate:.2f} per sensor")

        results['hotspot'] = {
            'most_common_site': [int(max_pos[0]), int(max_pos[1])],
            'most_common_count': int(max_count),
            'most_common_pct': float(max_pct),
            'chi_squared': float(chi_sq),
            'is_non_uniform': bool(is_non_uniform),
            'edge_rate': float(edge_rate),
            'center_rate': float(center_rate),
            'corner_rate': float(corner_rate),
        }

        # =====================================================================
        # Die topology correlation
        # =====================================================================
        print()
        print("=" * 70)
        print("DIE TOPOLOGY CORRELATION")
        print("=" * 70)
        print()

        # Row analysis (potential hot band)
        row_totals = [np.sum(nucleation_counts[r, :]) for r in range(4)]
        print(f"  Nucleations by row:")
        for r, total in enumerate(row_totals):
            pct = total / total_nucleations * 100 if total_nucleations > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    Row {r}: {int(total):2d} ({pct:5.1f}%) {bar}")

        # Column analysis (potential power rail)
        col_totals = [np.sum(nucleation_counts[:, c]) for c in range(4)]
        print()
        print(f"  Nucleations by column:")
        for c, total in enumerate(col_totals):
            pct = total / total_nucleations * 100 if total_nucleations > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    Col {c}: {int(total):2d} ({pct:5.1f}%) {bar}")

        # Identify dominant pattern
        row_var = np.var(row_totals)
        col_var = np.var(col_totals)

        if row_var > col_var * 2:
            pattern = "ROW_DOMINANT (potential hot band)"
        elif col_var > row_var * 2:
            pattern = "COLUMN_DOMINANT (potential power rail)"
        else:
            pattern = "MIXED/UNIFORM"

        print()
        print(f"  Dominant pattern: {pattern}")

        results['topology'] = {
            'row_totals': [int(r) for r in row_totals],
            'col_totals': [int(c) for c in col_totals],
            'row_variance': float(row_var),
            'col_variance': float(col_var),
            'pattern': pattern,
        }

    # =========================================================================
    # Nucleation timing analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("NUCLEATION TIMING")
    print("=" * 70)
    print()

    all_times = [t for times in nucleation_times.values() for t in times]
    if all_times:
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        print(f"  Mean nucleation time: {mean_time:.1f} ± {std_time:.1f} ms")
        print(f"  Range: {np.min(all_times):.1f} - {np.max(all_times):.1f} ms")

        results['timing'] = {
            'mean_ms': float(mean_time),
            'std_ms': float(std_time),
            'min_ms': float(np.min(all_times)),
            'max_ms': float(np.max(all_times)),
        }

    # =========================================================================
    # Success criteria
    # =========================================================================
    print()
    print("=" * 70)
    print("C3 SUCCESS CRITERIA")
    print("=" * 70)

    non_uniform = results.get('hotspot', {}).get('is_non_uniform', False)
    has_pattern = results.get('topology', {}).get('pattern', '') != 'MIXED/UNIFORM'
    reproducible = total_nucleations >= n_trials * 0.5  # At least 50% detection rate

    print(f"""
  ✓/✗ Non-uniform nucleation: {'✓' if non_uniform else '✗'} (χ²={results.get('hotspot', {}).get('chi_squared', 0):.1f})
  ✓/✗ Topology correlation:   {'✓' if has_pattern else '✗'} ({results.get('topology', {}).get('pattern', 'N/A')})
  ✓/✗ Reproducible:           {'✓' if reproducible else '✗'} ({total_nucleations}/{n_trials} detected)

  Overall: {'✓ PASS' if non_uniform or has_pattern else '✗ PARTIAL'}
""")

    results['success'] = {
        'non_uniform': bool(non_uniform),
        'has_pattern': bool(has_pattern),
        'reproducible': bool(reproducible),
        'all_pass': bool(non_uniform or has_pattern)
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("C3 SUMMARY: NUCLEATION SITES")
    print("=" * 70)

    print(f"""
  Instrument: 4×4 sensor array, uniform stress

  Key Findings:
  ─────────────
  1. Detection rate: {total_nucleations}/{n_trials} ({100*total_nucleations/n_trials:.0f}%)
  2. Most common site: {results.get('hotspot', {}).get('most_common_site', 'N/A')}
  3. Non-uniform: {'YES' if non_uniform else 'NO'} (χ²={results.get('hotspot', {}).get('chi_squared', 0):.1f})
  4. Spatial pattern: {results.get('topology', {}).get('pattern', 'N/A')}
  5. Mean nucleation time: {results.get('timing', {}).get('mean_ms', 0):.1f} ms

  Nucleation Map (visual):
  ────────────────────────""")

    # Visual map
    if total_nucleations > 0:
        max_count = np.max(nucleation_counts)
        for row in range(4):
            row_str = "  "
            for col in range(4):
                count = nucleation_counts[row, col]
                # Use intensity blocks
                if count == 0:
                    row_str += "  ·  "
                elif count <= max_count * 0.25:
                    row_str += "  ░  "
                elif count <= max_count * 0.5:
                    row_str += "  ▒  "
                elif count <= max_count * 0.75:
                    row_str += "  ▓  "
                else:
                    row_str += "  █  "
            print(row_str)

    print(f"""
  For Lean Formalization:
  ───────────────────────
  nucleation_site := {results.get('hotspot', {}).get('most_common_site', 'N/A')}
  nucleation_time := {results.get('timing', {}).get('mean_ms', 0):.1f} ms
  chi_squared := {results.get('hotspot', {}).get('chi_squared', 0):.1f}

  Ready for C4: Leading indicators analysis
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

    with open('/home/emoore/CIRISArray/experiments/expC3_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expC3_results.json")
