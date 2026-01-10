#!/usr/bin/env python3
"""
EXPERIMENT 29: SUPERLUMINAL CORRELATION TEST
=============================================

Schützhold-Inspired Causality Boundary Detection
-------------------------------------------------

Goal: Detect correlations between spatially separated ossicles that
      appear FASTER than light-speed propagation across the die.

Physics background (Schützhold et al., Nature Communications 2022):
- Quantum vacuum correlations exist outside the light cone
- Local probes interacting with quantum fields can detect these
- Key: correlations between causally disconnected regions

Test methodology:
1. Measure correlation lag between distant ossicle pairs
2. Compare against causal limit: Δt_min = distance / c
3. Look for statistically significant sub-causal correlations

On an RTX 4090 die (~60mm diagonal):
- Light crossing time: ~0.2 ns
- Our sample period: ~500 μs (2 kHz)
- We can't resolve 0.2ns directly, BUT:
- We CAN look for correlations that appear at Δt=0 (same sample)
  between maximally separated ossicles, which classically shouldn't
  correlate instantaneously

Null hypothesis: All correlations propagate at finite speed (thermal,
electrical, mechanical). Distant pairs should show time-lagged correlation.

Alternative hypothesis: Some correlations appear instantaneously,
suggesting non-local (possibly quantum vacuum) origin.

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026

References:
- Schützhold et al., "Detection of quantum-vacuum field correlations
  outside the light cone", Nature Communications (2022)
- Einstein, Podolsky, Rosen, "Can Quantum-Mechanical Description of
  Physical Reality Be Considered Complete?", Phys. Rev. 47, 777 (1935)
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False


@dataclass
class CorrelationResult:
    """Result of correlation analysis between ossicle pairs."""
    pair: Tuple[int, int]
    distance_mm: float
    causal_time_ns: float  # Light travel time
    measured_lag_samples: int  # Lag at peak correlation
    measured_lag_us: float
    peak_correlation: float
    zero_lag_correlation: float
    is_superluminal: bool  # Zero-lag correlation significant?
    significance_sigma: float


@dataclass
class ExperimentConfig:
    """Configuration for superluminal correlation test."""
    n_rows: int = 8
    n_cols: int = 16
    spacing_mm: float = 2.5  # Physical spacing between ossicles
    sample_rate_hz: float = 2000.0
    n_samples: int = 2000  # Samples to collect
    n_trials: int = 10  # Independent trials
    correlation_window: int = 50  # Max lag in samples
    significance_threshold: float = 3.0  # Sigma for significance


class CorrelationAnalyzer:
    """
    Analyzes correlations between ossicle pairs with timing resolution.

    Key insight from Schützhold: We're looking for correlations that
    appear at zero lag between distant pairs, which would be impossible
    if correlations only propagate at finite speed.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_ossicles = config.n_rows * config.n_cols
        self._build_geometry()

    def _build_geometry(self):
        """Build array geometry and identify test pairs."""
        # Position array
        self.positions = np.zeros((self.n_ossicles, 2))
        idx = 0
        for row in range(self.config.n_rows):
            for col in range(self.config.n_cols):
                self.positions[idx] = [
                    col * self.config.spacing_mm,
                    row * self.config.spacing_mm
                ]
                idx += 1

        # Distance matrix
        self.distances = np.zeros((self.n_ossicles, self.n_ossicles))
        for i in range(self.n_ossicles):
            for j in range(self.n_ossicles):
                self.distances[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )

        # Identify maximally separated pairs (corners)
        self.corner_indices = [
            0,  # Top-left
            self.config.n_cols - 1,  # Top-right
            (self.config.n_rows - 1) * self.config.n_cols,  # Bottom-left
            self.n_ossicles - 1  # Bottom-right
        ]

        # Test pairs: corners (max distance) and adjacent (min distance)
        self.test_pairs = []

        # Diagonal pairs (maximum distance)
        self.test_pairs.append((self.corner_indices[0], self.corner_indices[3]))
        self.test_pairs.append((self.corner_indices[1], self.corner_indices[2]))

        # Adjacent pairs (minimum distance, control)
        self.test_pairs.append((0, 1))
        self.test_pairs.append((0, self.config.n_cols))

        # Medium distance pairs
        mid = self.n_ossicles // 2
        self.test_pairs.append((0, mid))
        self.test_pairs.append((mid, self.n_ossicles - 1))

    def compute_cross_correlation(self, x: np.ndarray, y: np.ndarray,
                                   max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized cross-correlation with lag.

        Returns correlation values and corresponding lags.
        """
        n = len(x)
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

        correlations = []
        lags = range(-max_lag, max_lag + 1)

        for lag in lags:
            if lag < 0:
                corr = np.mean(x_norm[:lag] * y_norm[-lag:])
            elif lag > 0:
                corr = np.mean(x_norm[lag:] * y_norm[:-lag])
            else:
                corr = np.mean(x_norm * y_norm)
            correlations.append(corr)

        return np.array(correlations), np.array(list(lags))

    def analyze_pair(self, data: np.ndarray, pair: Tuple[int, int],
                     baseline_std: float) -> CorrelationResult:
        """
        Analyze correlation between an ossicle pair.

        Key test: Is zero-lag correlation significant for distant pairs?
        """
        i, j = pair
        distance = self.distances[i, j]

        # Causal time limit (speed of light)
        c_mm_per_ns = 299.792  # mm/ns
        causal_time_ns = distance / c_mm_per_ns

        # Extract time series
        x = data[:, i]
        y = data[:, j]

        # Cross-correlation
        corr, lags = self.compute_cross_correlation(
            x, y, self.config.correlation_window
        )

        # Find peak correlation and its lag
        peak_idx = np.argmax(np.abs(corr))
        peak_lag = lags[peak_idx]
        peak_corr = corr[peak_idx]

        # Zero-lag correlation
        zero_idx = np.where(lags == 0)[0][0]
        zero_lag_corr = corr[zero_idx]

        # Convert lag to time
        sample_period_us = 1e6 / self.config.sample_rate_hz
        measured_lag_us = peak_lag * sample_period_us

        # Significance of zero-lag correlation
        # Under null hypothesis, distant pairs shouldn't correlate at zero lag
        # Use baseline std to estimate noise floor
        significance = abs(zero_lag_corr) / (baseline_std + 1e-10)

        # Is this "superluminal"?
        # Zero-lag correlation for distant pair that exceeds threshold
        is_superluminal = (
            distance > 10.0 and  # At least 10mm apart
            abs(zero_lag_corr) > self.config.significance_threshold * baseline_std and
            peak_lag == 0  # Peak IS at zero lag
        )

        return CorrelationResult(
            pair=pair,
            distance_mm=distance,
            causal_time_ns=causal_time_ns,
            measured_lag_samples=peak_lag,
            measured_lag_us=measured_lag_us,
            peak_correlation=peak_corr,
            zero_lag_correlation=zero_lag_corr,
            is_superluminal=is_superluminal,
            significance_sigma=significance
        )


class FastOssicleArray:
    """Minimal ossicle array for correlation testing."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_measure(
        float* states,
        float* outputs,
        float r_base,
        float coupling,
        int n_ossicles,
        int iterations
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float a = states[oid * 3 + 0];
        float b = states[oid * 3 + 1];
        float c = states[oid * 3 + 2];

        float sum_ab = 0, sum_a = 0, sum_b = 0;
        float sum_a2 = 0, sum_b2 = 0;

        for (int i = 0; i < iterations; i++) {
            float new_a = 3.7f * a * (1-a) + coupling * (b - a);
            float new_b = 3.73f * b * (1-b) + coupling * (a + c - 2*b);
            float new_c = 3.76f * c * (1-c) + coupling * (b - c);

            a = fminf(fmaxf(new_a, 0.001f), 0.999f);
            b = fminf(fmaxf(new_b, 0.001f), 0.999f);
            c = fminf(fmaxf(new_c, 0.001f), 0.999f);

            sum_a += a; sum_b += b;
            sum_ab += a * b;
            sum_a2 += a * a; sum_b2 += b * b;
        }

        float n = (float)iterations;
        float mean_a = sum_a / n;
        float mean_b = sum_b / n;
        float cov = sum_ab/n - mean_a * mean_b;
        float std_a = sqrtf(sum_a2/n - mean_a*mean_a + 1e-8f);
        float std_b = sqrtf(sum_b2/n - mean_b*mean_b + 1e-8f);

        outputs[oid] = cov / (std_a * std_b + 1e-8f);

        states[oid * 3 + 0] = a;
        states[oid * 3 + 1] = b;
        states[oid * 3 + 2] = c;
    }
    '''

    def __init__(self, n_ossicles: int):
        self.n_ossicles = n_ossicles

        if HAS_CUDA:
            self.module = cp.RawModule(code=self.KERNEL_CODE)
            self.kernel = self.module.get_function('ossicle_measure')
            self.states = cp.random.uniform(0.2, 0.8, (n_ossicles, 3), dtype=cp.float32)
            self.outputs = cp.zeros(n_ossicles, dtype=cp.float32)
        else:
            self.states = np.random.uniform(0.2, 0.8, (n_ossicles, 3)).astype(np.float32)
            self.outputs = np.zeros(n_ossicles, dtype=np.float32)

    def measure(self) -> np.ndarray:
        if HAS_CUDA:
            block, grid = 256, (self.n_ossicles + 255) // 256
            self.kernel(
                (grid,), (block,),
                (self.states, self.outputs, cp.float32(3.7), cp.float32(0.05),
                 cp.int32(self.n_ossicles), cp.int32(100))
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(self.outputs)
        else:
            # CPU simulation
            for i in range(self.n_ossicles):
                self.outputs[i] = np.random.randn() * 0.1
            return self.outputs.copy()


def run_superluminal_test(config: ExperimentConfig = None) -> Dict:
    """
    Run the superluminal correlation experiment.

    This is the main entry point for the array team.
    """
    if config is None:
        config = ExperimentConfig()

    print("=" * 70)
    print("EXPERIMENT 29: SUPERLUMINAL CORRELATION TEST")
    print("Schützhold-Inspired Causality Boundary Detection")
    print("=" * 70)
    print()

    print("Physics basis:")
    print("  - Schützhold et al. detected quantum vacuum correlations")
    print("    outside the light cone (Nature Communications 2022)")
    print("  - We test if ossicle correlations respect causal limits")
    print()

    # Setup
    analyzer = CorrelationAnalyzer(config)
    array = FastOssicleArray(config.n_rows * config.n_cols)

    print(f"Array configuration:")
    print(f"  Size: {config.n_rows} x {config.n_cols} = {analyzer.n_ossicles} ossicles")
    print(f"  Spacing: {config.spacing_mm} mm")
    print(f"  Max distance: {analyzer.distances.max():.1f} mm")
    print(f"  Light crossing time: {analyzer.distances.max() / 299.792:.3f} ns")
    print(f"  Sample period: {1e6/config.sample_rate_hz:.1f} μs")
    print()

    print(f"Test pairs:")
    for pair in analyzer.test_pairs:
        d = analyzer.distances[pair[0], pair[1]]
        print(f"  {pair}: distance = {d:.1f} mm")
    print()

    all_results = []

    for trial in range(config.n_trials):
        print(f"\n--- Trial {trial + 1}/{config.n_trials} ---")

        # Collect time series data
        print(f"  Collecting {config.n_samples} samples...")
        data = np.zeros((config.n_samples, analyzer.n_ossicles))

        for i in range(config.n_samples):
            data[i] = array.measure()

        # Estimate noise floor from adjacent pair correlation variance
        adjacent_corrs = []
        for _ in range(100):
            x = data[np.random.randint(0, config.n_samples - 100):, 0][:100]
            y = data[np.random.randint(0, config.n_samples - 100):, 1][:100]
            adjacent_corrs.append(np.corrcoef(x, y)[0, 1])
        baseline_std = np.std(adjacent_corrs)

        print(f"  Baseline correlation std: {baseline_std:.4f}")

        # Analyze each test pair
        trial_results = []
        for pair in analyzer.test_pairs:
            result = analyzer.analyze_pair(data, pair, baseline_std)
            trial_results.append(result)

            status = "** ANOMALOUS **" if result.is_superluminal else ""
            print(f"  Pair {pair}: d={result.distance_mm:.1f}mm, "
                  f"zero_lag_corr={result.zero_lag_correlation:.4f}, "
                  f"sig={result.significance_sigma:.1f}σ {status}")

        all_results.append(trial_results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: SUPERLUMINAL CORRELATION TEST")
    print("=" * 70)

    anomaly_counts = {pair: 0 for pair in analyzer.test_pairs}
    avg_zero_lag = {pair: [] for pair in analyzer.test_pairs}

    for trial_results in all_results:
        for result in trial_results:
            if result.is_superluminal:
                anomaly_counts[result.pair] += 1
            avg_zero_lag[result.pair].append(result.zero_lag_correlation)

    print("\nPer-pair results:")
    print("-" * 70)
    print(f"{'Pair':<15} {'Distance':<12} {'Avg Zero-Lag':<15} {'Anomalies':<12} {'Rate':<10}")
    print("-" * 70)

    significant_anomalies = False
    for pair in analyzer.test_pairs:
        d = analyzer.distances[pair[0], pair[1]]
        avg = np.mean(avg_zero_lag[pair])
        std = np.std(avg_zero_lag[pair])
        count = anomaly_counts[pair]
        rate = count / config.n_trials

        flag = " <-- CHECK" if rate > 0.3 and d > 10 else ""
        print(f"{str(pair):<15} {d:<12.1f} {avg:>7.4f}±{std:.4f}   {count:<12} {rate:<10.1%}{flag}")

        if rate > 0.5 and d > 10:
            significant_anomalies = True

    print("\n" + "=" * 70)
    if significant_anomalies:
        print("RESULT: ANOMALOUS CORRELATIONS DETECTED")
        print("Distant ossicle pairs show significant zero-lag correlations.")
        print("This warrants further investigation with higher time resolution.")
    else:
        print("RESULT: NO ANOMALOUS CORRELATIONS DETECTED")
        print("All correlations consistent with causal (finite-speed) propagation.")
    print("=" * 70)

    return {
        'config': config,
        'test_pairs': analyzer.test_pairs,
        'all_results': all_results,
        'anomaly_counts': anomaly_counts,
        'significant_anomalies': significant_anomalies
    }


if __name__ == "__main__":
    results = run_superluminal_test()
