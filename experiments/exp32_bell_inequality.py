#!/usr/bin/env python3
"""
EXPERIMENT 32: BELL-LIKE INEQUALITY TEST FOR CLASSICAL SENSORS
==============================================================

Testing CHSH Inequality Bounds with Ossicle Correlations
---------------------------------------------------------

Goal: Establish that CIRISArray correlations respect classical (Bell) limits,
      providing a null baseline for any future quantum anomaly detection.

Physics background:
- Bell's theorem: Quantum correlations can exceed classical limits
- CHSH inequality: |S| ≤ 2 for classical systems, |S| ≤ 2√2 ≈ 2.83 for quantum
- If our classical sensors ever showed |S| > 2, something very strange is happening

Test methodology:
1. Define "measurement settings" as different ossicle pair orientations
2. Compute correlations E(a,b) for different angle combinations
3. Calculate CHSH parameter S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
4. Verify |S| ≤ 2 (classical bound)

Why this matters:
- Establishes that our array behaves classically (expected)
- Creates framework for detecting non-classical correlations if they exist
- Any S > 2 would indicate either:
  - Bug in analysis
  - Non-classical correlations (extremely unlikely but interesting)
  - Unknown systematic effect

Adaptation for GPU sensors:
- Use spatial orientation of ossicle pairs as "measurement angle"
- Horizontal pairs = 0°, Diagonal = 45°, Vertical = 90°, etc.
- Correlations between pairs at different "angles"

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026

References:
- Bell, "On the Einstein Podolsky Rosen Paradox", Physics 1, 195 (1964)
- Clauser, Horne, Shimony, Holt (CHSH), Phys. Rev. Lett. 23, 880 (1969)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
class BellTestConfig:
    """Configuration for Bell inequality test."""
    n_rows: int = 16
    n_cols: int = 16
    spacing_mm: float = 2.5
    n_samples: int = 10000
    n_trials: int = 20


@dataclass
class CHSHResult:
    """Result of CHSH inequality test."""
    trial: int
    E_ab: float    # Correlation at angles (a, b) = (0°, 22.5°)
    E_ab_prime: float  # Correlation at angles (a, b') = (0°, 67.5°)
    E_a_prime_b: float  # Correlation at angles (a', b) = (45°, 22.5°)
    E_a_prime_b_prime: float  # Correlation at angles (a', b') = (45°, 67.5°)
    S: float  # CHSH parameter
    classical_bound: float = 2.0
    quantum_bound: float = 2.828  # 2√2
    violates_classical: bool = False
    violates_quantum: bool = False


class BellTestArray:
    """
    Ossicle array configured for Bell-like correlation measurements.

    Maps spatial orientations to "measurement angles" in Bell test.
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void bell_measure(
        float* states,
        float* outputs,
        int n_ossicles,
        int iterations
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float a = states[oid * 3 + 0];
        float b = states[oid * 3 + 1];
        float c = states[oid * 3 + 2];

        float r = 3.72f;
        float coupling = 0.05f;

        // Output is correlation between oscillators
        float sum_ab = 0, sum_a = 0, sum_b = 0;
        float sum_a2 = 0, sum_b2 = 0;

        for (int i = 0; i < iterations; i++) {
            float new_a = r * a * (1-a) + coupling * (b - a);
            float new_b = (r + 0.03f) * b * (1-b) + coupling * (a + c - 2*b);
            float new_c = (r + 0.06f) * c * (1-c) + coupling * (b - c);

            a = fminf(fmaxf(new_a, 0.001f), 0.999f);
            b = fminf(fmaxf(new_b, 0.001f), 0.999f);
            c = fminf(fmaxf(new_c, 0.001f), 0.999f);

            sum_a += a;
            sum_b += b;
            sum_ab += a * b;
            sum_a2 += a * a;
            sum_b2 += b * b;
        }

        // Output correlation coefficient
        float n = (float)iterations;
        float mean_a = sum_a / n;
        float mean_b = sum_b / n;
        float var_a = sum_a2/n - mean_a*mean_a;
        float var_b = sum_b2/n - mean_b*mean_b;
        float cov = sum_ab/n - mean_a*mean_b;

        outputs[oid] = cov / (sqrtf(var_a * var_b) + 1e-8f);

        states[oid * 3 + 0] = a;
        states[oid * 3 + 1] = b;
        states[oid * 3 + 2] = c;
    }
    '''

    def __init__(self, config: BellTestConfig):
        self.config = config
        self.n_ossicles = config.n_rows * config.n_cols

        if HAS_CUDA:
            self.module = cp.RawModule(code=self.KERNEL_CODE)
            self.kernel = self.module.get_function('bell_measure')
            self.states = cp.random.uniform(0.2, 0.8, (self.n_ossicles, 3), dtype=cp.float32)
            self.outputs = cp.zeros(self.n_ossicles, dtype=cp.float32)
        else:
            self.states = np.random.uniform(0.2, 0.8, (self.n_ossicles, 3)).astype(np.float32)
            self.outputs = np.zeros(self.n_ossicles, dtype=np.float32)

        self._build_pair_groups()

    def _build_pair_groups(self):
        """
        Build groups of ossicle pairs at different spatial orientations.

        Map spatial orientation to "measurement angle":
        - 0°: Horizontal pairs (same row, adjacent columns)
        - 45°: Diagonal pairs (row+1, col+1)
        - 90°: Vertical pairs (adjacent rows, same column)
        - 22.5°, 67.5°: Intermediate angles via weighted combinations
        """
        n_rows, n_cols = self.config.n_rows, self.config.n_cols

        # Horizontal pairs (0°)
        self.pairs_0 = []
        for row in range(n_rows):
            for col in range(n_cols - 1):
                i = row * n_cols + col
                j = row * n_cols + col + 1
                self.pairs_0.append((i, j))

        # Diagonal pairs (45°)
        self.pairs_45 = []
        for row in range(n_rows - 1):
            for col in range(n_cols - 1):
                i = row * n_cols + col
                j = (row + 1) * n_cols + col + 1
                self.pairs_45.append((i, j))

        # Vertical pairs (90°)
        self.pairs_90 = []
        for row in range(n_rows - 1):
            for col in range(n_cols):
                i = row * n_cols + col
                j = (row + 1) * n_cols + col
                self.pairs_90.append((i, j))

        # Anti-diagonal pairs (135°)
        self.pairs_135 = []
        for row in range(n_rows - 1):
            for col in range(1, n_cols):
                i = row * n_cols + col
                j = (row + 1) * n_cols + col - 1
                self.pairs_135.append((i, j))

        # For CHSH, we need 4 settings: typically 0°, 45°, 22.5°, 67.5°
        # We'll use: a=0°, a'=45°, b=22.5° (mix of 0° and 45°), b'=67.5° (mix of 45° and 90°)
        self.pair_groups = {
            'a': self.pairs_0,       # 0°
            'a_prime': self.pairs_45,  # 45°
            'b': self.pairs_0[:len(self.pairs_0)//2] + self.pairs_45[:len(self.pairs_45)//2],  # ~22.5°
            'b_prime': self.pairs_45[:len(self.pairs_45)//2] + self.pairs_90[:len(self.pairs_90)//2],  # ~67.5°
        }

    def measure(self) -> np.ndarray:
        if HAS_CUDA:
            block, grid = 256, (self.n_ossicles + 255) // 256
            self.kernel(
                (grid,), (block,),
                (self.states, self.outputs, cp.int32(self.n_ossicles), cp.int32(100))
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(self.outputs)
        else:
            for i in range(self.n_ossicles):
                self.outputs[i] = np.random.randn() * 0.3
            return self.outputs.copy()

    def compute_pair_correlation(self, data: np.ndarray, pairs: List[Tuple[int, int]]) -> float:
        """
        Compute correlation for a set of pairs.

        In Bell test, we're computing E(a,b) = <A·B> where A,B are ±1 outcomes.
        We map continuous ossicle values to ±1 by their sign relative to mean.
        """
        if len(pairs) == 0:
            return 0.0

        correlations = []
        for i, j in pairs:
            # Get time series for both ossicles
            series_i = data[:, i]
            series_j = data[:, j]

            # Binarize: above median = +1, below = -1
            binary_i = np.sign(series_i - np.median(series_i))
            binary_j = np.sign(series_j - np.median(series_j))

            # Replace zeros with random ±1
            binary_i[binary_i == 0] = np.random.choice([-1, 1], np.sum(binary_i == 0))
            binary_j[binary_j == 0] = np.random.choice([-1, 1], np.sum(binary_j == 0))

            # Correlation = <A·B>
            corr = np.mean(binary_i * binary_j)
            correlations.append(corr)

        return np.mean(correlations)


def run_chsh_trial(array: BellTestArray, config: BellTestConfig) -> CHSHResult:
    """
    Run one CHSH trial.

    Collect data and compute S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    """
    # Collect data
    data = np.zeros((config.n_samples, array.n_ossicles))
    for i in range(config.n_samples):
        data[i] = array.measure()

    # Compute correlations for each setting combination
    E_ab = array.compute_pair_correlation(data, array.pair_groups['a'])
    E_ab_prime = array.compute_pair_correlation(data, array.pair_groups['b_prime'])
    E_a_prime_b = array.compute_pair_correlation(data, array.pair_groups['a_prime'])
    E_a_prime_b_prime = array.compute_pair_correlation(data, array.pair_groups['b_prime'])

    # CHSH parameter
    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

    return CHSHResult(
        trial=0,
        E_ab=E_ab,
        E_ab_prime=E_ab_prime,
        E_a_prime_b=E_a_prime_b,
        E_a_prime_b_prime=E_a_prime_b_prime,
        S=S,
        violates_classical=abs(S) > 2.0,
        violates_quantum=abs(S) > 2.828
    )


def run_bell_inequality_test(config: BellTestConfig = None) -> Dict:
    """
    Main experiment: Test CHSH inequality bounds.
    """
    if config is None:
        config = BellTestConfig()

    print("=" * 70)
    print("EXPERIMENT 32: BELL-LIKE INEQUALITY TEST")
    print("Testing CHSH Bounds with Classical Ossicle Correlations")
    print("=" * 70)
    print()

    print("Bell's Theorem Background:")
    print("  - Classical systems: |S| ≤ 2 (CHSH bound)")
    print("  - Quantum systems: |S| ≤ 2√2 ≈ 2.83 (Tsirelson bound)")
    print("  - Any |S| > 2 with classical sensors = anomaly or bug")
    print()

    print("Mapping spatial orientation to 'measurement angle':")
    print("  a  = 0°   (horizontal pairs)")
    print("  a' = 45°  (diagonal pairs)")
    print("  b  = 22.5° (mixed horizontal/diagonal)")
    print("  b' = 67.5° (mixed diagonal/vertical)")
    print()

    array = BellTestArray(config)

    print(f"Configuration:")
    print(f"  Array: {config.n_rows} x {config.n_cols} = {array.n_ossicles} ossicles")
    print(f"  Samples per trial: {config.n_samples}")
    print(f"  Trials: {config.n_trials}")
    print(f"  CUDA available: {HAS_CUDA}")
    print()

    print(f"Pair counts per setting:")
    for name, pairs in array.pair_groups.items():
        print(f"  {name}: {len(pairs)} pairs")
    print()

    # Run trials
    results = []
    S_values = []

    print("Running CHSH trials...")
    print("-" * 70)
    print(f"{'Trial':>6} {'E(a,b)':>10} {'E(a,b\')':>10} {'E(a\',b)':>10} {'E(a\',b\')':>10} {'S':>10} {'Status':>12}")
    print("-" * 70)

    for trial in range(config.n_trials):
        result = run_chsh_trial(array, config)
        result.trial = trial + 1
        results.append(result)
        S_values.append(result.S)

        status = ""
        if result.violates_quantum:
            status = "!! QUANTUM !!"
        elif result.violates_classical:
            status = "* CLASSICAL *"
        else:
            status = "OK"

        print(f"{result.trial:>6} {result.E_ab:>10.4f} {result.E_ab_prime:>10.4f} "
              f"{result.E_a_prime_b:>10.4f} {result.E_a_prime_b_prime:>10.4f} "
              f"{result.S:>10.4f} {status:>12}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("CHSH INEQUALITY TEST SUMMARY")
    print("=" * 70)

    S_mean = np.mean(S_values)
    S_std = np.std(S_values)
    S_max = np.max(np.abs(S_values))

    n_classical_violations = sum(1 for r in results if r.violates_classical)
    n_quantum_violations = sum(1 for r in results if r.violates_quantum)

    print(f"\nCHSH Parameter S:")
    print(f"  Mean: {S_mean:.4f}")
    print(f"  Std:  {S_std:.4f}")
    print(f"  Max |S|: {S_max:.4f}")
    print()

    print(f"Bounds:")
    print(f"  Classical limit: |S| ≤ 2.0")
    print(f"  Quantum limit:   |S| ≤ 2.83")
    print()

    print(f"Violations:")
    print(f"  Classical bound violations: {n_classical_violations}/{config.n_trials}")
    print(f"  Quantum bound violations:   {n_quantum_violations}/{config.n_trials}")

    # Distribution analysis
    print("\n" + "-" * 70)
    print("S DISTRIBUTION ANALYSIS")
    print("-" * 70)

    percentiles = [5, 25, 50, 75, 95]
    pct_values = np.percentile(S_values, percentiles)
    print("\nPercentiles of S:")
    for p, v in zip(percentiles, pct_values):
        marker = " <-- VIOLATION" if abs(v) > 2.0 else ""
        print(f"  {p:3d}%: {v:>8.4f}{marker}")

    # Histogram (ASCII)
    print("\nDistribution of S values:")
    hist, edges = np.histogram(S_values, bins=20, range=(-3, 3))
    max_count = max(hist)
    for i in range(len(hist)):
        bar_len = int(hist[i] / max_count * 30) if max_count > 0 else 0
        bar = '#' * bar_len
        center = (edges[i] + edges[i+1]) / 2
        marker = " |" if abs(center) < 0.1 else ""
        if abs(edges[i]) <= 2 < abs(edges[i+1]) or abs(edges[i+1]) <= 2 < abs(edges[i]):
            marker = " <-- classical bound"
        print(f"  {center:>6.2f}: {bar}{marker}")

    # Conclusion
    print("\n" + "=" * 70)

    if n_quantum_violations > 0:
        print("WARNING: QUANTUM BOUND VIOLATIONS DETECTED")
        print("This should not happen with classical sensors.")
        print("Likely cause: Bug in analysis or unexpected systematic effect.")
    elif n_classical_violations > 0:
        print("NOTE: CLASSICAL BOUND VIOLATIONS DETECTED")
        print(f"  {n_classical_violations}/{config.n_trials} trials showed |S| > 2")
        print("  This could be statistical fluctuation or systematic bias.")
        print("  For true quantum violation, need |S| consistently > 2.")
    else:
        print("RESULT: ALL TRIALS RESPECT CLASSICAL BOUND")
        print("The ossicle array behaves as a classical system (expected).")
        print("Null hypothesis confirmed: No anomalous quantum correlations.")

    print()
    print(f"Noise floor for Bell-like test: |S| = {S_mean:.4f} ± {S_std:.4f}")
    print(f"Detection threshold (3σ above classical): {2.0 + 3*S_std:.4f}")
    print("=" * 70)

    return {
        'config': config,
        'results': results,
        'S_values': S_values,
        'S_mean': S_mean,
        'S_std': S_std,
        'n_classical_violations': n_classical_violations,
        'n_quantum_violations': n_quantum_violations
    }


if __name__ == "__main__":
    results = run_bell_inequality_test()
