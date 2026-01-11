#!/usr/bin/env python3
"""
Experiment 54: Leggett-Garg Inequality Test
============================================

Tests whether CIRISArray oscillator system satisfies macrorealism:
1. System is in definite state at all times
2. Measurement doesn't disturb subsequent evolution

Classical systems: K₃ = C₁₂ + C₂₃ - C₁₃ ≤ 1
Quantum systems: Can violate K₃ ≤ 1
Nonlinear bistable: May show measurement invasiveness

Based on: RATCHET/experiments/LGI_PROTOCOL.md

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
import json
from datetime import datetime, timezone
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class LGIResult:
    """Result of LGI measurement."""
    K3: float
    K3_err: float
    C12: float
    C23: float
    C13: float
    n_trials: int
    tau: float
    violation_sigma: float  # (K3 - 1) / K3_err
    conclusion: str


class LGISensor:
    """Sensor optimized for LGI measurements."""

    def __init__(self, n_ossicles: int = 2048, depth: int = 64):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

        self.k_eff_history = []
        self.reset()

    def reset(self):
        """Reset oscillators to random initial state."""
        xp = cp if HAS_CUDA else np
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.k_eff_history = []

    def step(self, noise: float = 0.001, iterations: int = 5):
        """Evolve system one time step."""
        xp = cp if HAS_CUDA else np

        for _ in range(iterations):
            if noise > 0:
                self.osc_a += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_b += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_c += xp.random.normal(0, noise, self.total).astype(xp.float32)

            da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
            db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
            dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a = self.osc_a + da
            self.osc_b = self.osc_b + db
            self.osc_c = self.osc_c + dc

            self.osc_a = xp.clip(self.osc_a, -10, 10)
            self.osc_b = xp.clip(self.osc_b, -10, 10)
            self.osc_c = xp.clip(self.osc_c, -10, 10)

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
        """Measure k_eff (this IS the measurement that may be invasive)."""
        xp = cp if HAS_CUDA else np

        sample_size = min(10000, self.total)
        if HAS_CUDA:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r_ab = float(xp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        total_var = float(xp.var(a) + xp.var(b))
        x = min(total_var / 2.0, 1.0)

        k_eff = r_ab * (1 - x) * self.coupling_ab * 1000
        self.k_eff_history.append(k_eff)
        return k_eff

    def evolve_for(self, duration: float, sample_rate: float = 10.0, noise: float = 0.001):
        """Evolve system for given duration."""
        n_steps = int(duration * sample_rate)
        interval = 1.0 / sample_rate

        for _ in range(n_steps):
            start = time.perf_counter()
            self.step(noise=noise)
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)


def dichotomic_Q(k_eff: float, median: float) -> int:
    """Convert k_eff to dichotomic observable Q ∈ {-1, +1}."""
    return 1 if k_eff > median else -1


# =============================================================================
# PHASE 1: NULL HYPOTHESIS TESTS
# =============================================================================

def test_N1_correlation_structure(duration: float = 300, sample_rate: float = 10) -> Dict:
    """N1: Verify temporal correlations exist."""
    print("\n" + "="*60)
    print("N1: CORRELATION STRUCTURE")
    print("="*60)

    sensor = LGISensor()
    n_samples = int(duration * sample_rate)
    k_eff_series = []

    print(f"\n  Capturing {duration}s at {sample_rate} Hz...")

    for i in range(n_samples):
        start = time.perf_counter()
        sensor.step()
        k_eff_series.append(sensor.measure_k_eff())
        elapsed = time.perf_counter() - start
        if elapsed < 1/sample_rate:
            time.sleep(1/sample_rate - elapsed)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_samples}")

    k_eff = np.array(k_eff_series)

    # Compute autocorrelation
    k_centered = k_eff - np.mean(k_eff)
    autocorr = np.correlate(k_centered, k_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Find decay time (where autocorr drops to 1/e)
    tau_idx = np.where(autocorr < 1/np.e)[0]
    tau_decay = tau_idx[0] / sample_rate if len(tau_idx) > 0 else duration

    # Noise floor
    noise_floor = np.std(autocorr[int(len(autocorr)*0.8):])

    print(f"\n  Decay time: τ = {tau_decay:.1f}s")
    print(f"  Noise floor: {noise_floor:.4f}")
    print(f"  Max correlation: {autocorr[1]:.4f}")

    if tau_decay < 1:
        conclusion = "FAIL: Decay too fast to measure"
    elif autocorr[int(sample_rate)] < noise_floor * 2:
        conclusion = "FAIL: No significant temporal structure"
    else:
        conclusion = "PASS: Temporal correlations detected"

    print(f"\n  CONCLUSION: {conclusion}")

    return {
        'tau_decay': tau_decay,
        'noise_floor': noise_floor,
        'autocorr': autocorr[:int(60*sample_rate)],
        'conclusion': conclusion
    }


def test_N2_dichotomic_stability(n_trials: int = 100, duration: float = 60) -> Dict:
    """N2: Verify Q(t) is balanced and not too noisy."""
    print("\n" + "="*60)
    print("N2: DICHOTOMIC OBSERVABLE STABILITY")
    print("="*60)

    all_k_eff = []
    transition_rates = []

    for trial in range(n_trials):
        sensor = LGISensor()
        k_eff_series = []

        for _ in range(int(duration * 10)):  # 10 Hz
            sensor.step()
            k_eff_series.append(sensor.measure_k_eff())
            time.sleep(0.1)

        all_k_eff.extend(k_eff_series)

        # Count transitions
        median = np.median(k_eff_series)
        Q = [dichotomic_Q(k, median) for k in k_eff_series]
        transitions = sum(1 for i in range(1, len(Q)) if Q[i] != Q[i-1])
        transition_rates.append(transitions / duration)

        if (trial + 1) % 20 == 0:
            print(f"    {trial+1}/{n_trials} trials")

    # Overall statistics
    overall_median = np.median(all_k_eff)
    Q_all = [dichotomic_Q(k, overall_median) for k in all_k_eff]
    p_plus = sum(1 for q in Q_all if q == 1) / len(Q_all)
    bias = abs(p_plus - 0.5)

    mean_rate = np.mean(transition_rates)

    print(f"\n  P(Q=+1) = {p_plus:.3f}")
    print(f"  Bias: {bias:.3f}")
    print(f"  Mean transition rate: {mean_rate:.2f}/s")

    if bias > 0.4:
        conclusion = "FAIL: Observable too biased"
    elif mean_rate > 5:
        conclusion = "FAIL: Too many transitions (noisy)"
    elif mean_rate < 0.01:
        conclusion = "FAIL: Too few transitions (stuck)"
    else:
        conclusion = "PASS: Observable is stable and balanced"

    print(f"\n  CONCLUSION: {conclusion}")

    return {
        'p_plus': p_plus,
        'bias': bias,
        'mean_transition_rate': mean_rate,
        'conclusion': conclusion
    }


# =============================================================================
# PHASE 2: CHARACTERIZATION
# =============================================================================

def compute_correlations(sensor: LGISensor, t1: float, t2: float, t3: float,
                        measure_at_t2: bool = True, noise: float = 0.001) -> Tuple[int, int, int]:
    """
    Run single trial, return Q values at t1, t2, t3.

    If measure_at_t2=False, skip measurement at t2 (for invasiveness test).
    """
    sensor.reset()

    # Evolve to t1
    sensor.evolve_for(t1, noise=noise)
    Q1 = dichotomic_Q(sensor.measure_k_eff(), 0)  # median ~0 for k_eff

    # Evolve to t2
    sensor.evolve_for(t2 - t1, noise=noise)
    if measure_at_t2:
        Q2 = dichotomic_Q(sensor.measure_k_eff(), 0)
    else:
        Q2 = 0  # Placeholder

    # Evolve to t3
    sensor.evolve_for(t3 - t2, noise=noise)
    Q3 = dichotomic_Q(sensor.measure_k_eff(), 0)

    return Q1, Q2, Q3


def test_C2_three_time(tau: float = 11.5, n_trials: int = 500) -> Dict:
    """C2: Measure three-time correlations."""
    print("\n" + "="*60)
    print(f"C2: THREE-TIME CORRELATIONS (τ = {tau}s)")
    print("="*60)

    t1 = 5.0
    t2 = t1 + tau
    t3 = t1 + 2 * tau

    sensor = LGISensor()
    Q1_list, Q2_list, Q3_list = [], [], []

    print(f"\n  Running {n_trials} trials...")

    for trial in range(n_trials):
        Q1, Q2, Q3 = compute_correlations(sensor, t1, t2, t3, measure_at_t2=True)
        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Q3_list.append(Q3)

        if (trial + 1) % 100 == 0:
            print(f"    {trial+1}/{n_trials}")

    Q1 = np.array(Q1_list)
    Q2 = np.array(Q2_list)
    Q3 = np.array(Q3_list)

    # Correlations
    C12 = np.mean(Q1 * Q2)
    C23 = np.mean(Q2 * Q3)
    C13 = np.mean(Q1 * Q3)

    # Standard errors (bootstrap)
    n_boot = 1000
    C12_boot = [np.mean(Q1[np.random.choice(n_trials, n_trials)] *
                       Q2[np.random.choice(n_trials, n_trials)]) for _ in range(n_boot)]
    C23_boot = [np.mean(Q2[np.random.choice(n_trials, n_trials)] *
                       Q3[np.random.choice(n_trials, n_trials)]) for _ in range(n_boot)]
    C13_boot = [np.mean(Q1[np.random.choice(n_trials, n_trials)] *
                       Q3[np.random.choice(n_trials, n_trials)]) for _ in range(n_boot)]

    C12_err = np.std(C12_boot)
    C23_err = np.std(C23_boot)
    C13_err = np.std(C13_boot)

    print(f"\n  C₁₂ = {C12:.4f} ± {C12_err:.4f}")
    print(f"  C₂₃ = {C23:.4f} ± {C23_err:.4f}")
    print(f"  C₁₃ = {C13:.4f} ± {C13_err:.4f}")

    return {
        'tau': tau,
        'C12': C12, 'C12_err': C12_err,
        'C23': C23, 'C23_err': C23_err,
        'C13': C13, 'C13_err': C13_err,
        'n_trials': n_trials
    }


def test_C3_invasiveness(tau: float = 11.5, n_trials: int = 500) -> Dict:
    """C3: Test measurement invasiveness."""
    print("\n" + "="*60)
    print("C3: INVASIVENESS TEST (CRITICAL)")
    print("="*60)

    t1 = 5.0
    t2 = t1 + tau
    t3 = t1 + 2 * tau

    sensor = LGISensor()

    # Condition A: Direct C13 (no measurement at t2)
    print("\n  Condition A: Skip measurement at t₂...")
    C13_A_samples = []
    for trial in range(n_trials):
        Q1, _, Q3 = compute_correlations(sensor, t1, t2, t3, measure_at_t2=False)
        C13_A_samples.append(Q1 * Q3)
        if (trial + 1) % 100 == 0:
            print(f"    {trial+1}/{n_trials}")

    C13_A = np.mean(C13_A_samples)
    C13_A_err = np.std(C13_A_samples) / np.sqrt(n_trials)

    # Condition B: Measure at t2
    print("\n  Condition B: Measure at t₂...")
    C13_B_samples = []
    for trial in range(n_trials):
        Q1, Q2, Q3 = compute_correlations(sensor, t1, t2, t3, measure_at_t2=True)
        C13_B_samples.append(Q1 * Q3)
        if (trial + 1) % 100 == 0:
            print(f"    {trial+1}/{n_trials}")

    C13_B = np.mean(C13_B_samples)
    C13_B_err = np.std(C13_B_samples) / np.sqrt(n_trials)

    invasiveness = abs(C13_A - C13_B)
    invasiveness_sigma = invasiveness / np.sqrt(C13_A_err**2 + C13_B_err**2)

    print(f"\n  C₁₃ (without t₂ measurement): {C13_A:.4f} ± {C13_A_err:.4f}")
    print(f"  C₁₃ (with t₂ measurement):    {C13_B:.4f} ± {C13_B_err:.4f}")
    print(f"  Invasiveness: {invasiveness:.4f} ({invasiveness_sigma:.1f}σ)")

    if invasiveness < 0.05:
        conclusion = "Non-invasive (classical)"
    elif invasiveness > 0.1:
        conclusion = "INVASIVE: Measurement disturbs system"
    else:
        conclusion = "Marginal invasiveness"

    print(f"\n  CONCLUSION: {conclusion}")

    return {
        'C13_A': C13_A, 'C13_A_err': C13_A_err,
        'C13_B': C13_B, 'C13_B_err': C13_B_err,
        'invasiveness': invasiveness,
        'invasiveness_sigma': invasiveness_sigma,
        'conclusion': conclusion
    }


# =============================================================================
# PHASE 3: VALIDATION
# =============================================================================

def test_V1_primary_LGI(tau: float = 11.5, n_trials: int = 2500) -> LGIResult:
    """V1: Primary LGI test."""
    print("\n" + "="*60)
    print("V1: PRIMARY LGI TEST")
    print("="*60)
    print(f"\n  τ = {tau}s, N = {n_trials} trials")
    print(f"  Classical bound: K₃ ≤ 1")

    t1 = 5.0
    t2 = t1 + tau
    t3 = t1 + 2 * tau

    sensor = LGISensor()
    Q1_list, Q2_list, Q3_list = [], [], []

    print("\n  Running trials...")

    for trial in range(n_trials):
        Q1, Q2, Q3 = compute_correlations(sensor, t1, t2, t3)
        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Q3_list.append(Q3)

        if (trial + 1) % 500 == 0:
            # Interim check
            Q1_arr = np.array(Q1_list)
            Q2_arr = np.array(Q2_list)
            Q3_arr = np.array(Q3_list)
            K3_interim = np.mean(Q1_arr * Q2_arr) + np.mean(Q2_arr * Q3_arr) - np.mean(Q1_arr * Q3_arr)
            print(f"    {trial+1}/{n_trials} - K₃ = {K3_interim:.4f}")

    Q1 = np.array(Q1_list)
    Q2 = np.array(Q2_list)
    Q3 = np.array(Q3_list)

    # Correlations
    C12 = np.mean(Q1 * Q2)
    C23 = np.mean(Q2 * Q3)
    C13 = np.mean(Q1 * Q3)

    K3 = C12 + C23 - C13

    # Bootstrap error
    n_boot = 1000
    K3_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n_trials, n_trials)
        K3_b = np.mean(Q1[idx] * Q2[idx]) + np.mean(Q2[idx] * Q3[idx]) - np.mean(Q1[idx] * Q3[idx])
        K3_boot.append(K3_b)

    K3_err = np.std(K3_boot)
    violation_sigma = (K3 - 1) / K3_err

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\n  C₁₂ = {C12:.4f}")
    print(f"  C₂₃ = {C23:.4f}")
    print(f"  C₁₃ = {C13:.4f}")
    print(f"\n  K₃ = C₁₂ + C₂₃ - C₁₃ = {K3:.4f} ± {K3_err:.4f}")
    print(f"\n  Classical bound: K₃ ≤ 1")
    print(f"  Distance from bound: {violation_sigma:.2f}σ")

    if K3 - 2*K3_err > 1:
        conclusion = "*** LGI VIOLATION DETECTED ***"
    elif K3 + 2*K3_err < 1:
        conclusion = "CLASSICAL: K₃ < 1 confirmed"
    else:
        conclusion = "INCONCLUSIVE: K₃ near boundary"

    print(f"\n  CONCLUSION: {conclusion}")

    return LGIResult(
        K3=K3,
        K3_err=K3_err,
        C12=C12,
        C23=C23,
        C13=C13,
        n_trials=n_trials,
        tau=tau,
        violation_sigma=violation_sigma,
        conclusion=conclusion
    )


def test_V2_four_time(tau: float = 11.5, n_trials: int = 2500) -> Dict:
    """V2: Four-time LGI extension."""
    print("\n" + "="*60)
    print("V2: FOUR-TIME LGI")
    print("="*60)
    print(f"\n  τ = {tau}s, N = {n_trials} trials")
    print(f"  Classical bound: K₄ ≤ 2")

    t1 = 5.0
    t2 = t1 + tau
    t3 = t1 + 2 * tau
    t4 = t1 + 3 * tau

    sensor = LGISensor()
    Q1_list, Q2_list, Q3_list, Q4_list = [], [], [], []

    print("\n  Running trials...")

    for trial in range(n_trials):
        sensor.reset()

        sensor.evolve_for(t1)
        Q1 = dichotomic_Q(sensor.measure_k_eff(), 0)

        sensor.evolve_for(t2 - t1)
        Q2 = dichotomic_Q(sensor.measure_k_eff(), 0)

        sensor.evolve_for(t3 - t2)
        Q3 = dichotomic_Q(sensor.measure_k_eff(), 0)

        sensor.evolve_for(t4 - t3)
        Q4 = dichotomic_Q(sensor.measure_k_eff(), 0)

        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Q3_list.append(Q3)
        Q4_list.append(Q4)

        if (trial + 1) % 500 == 0:
            print(f"    {trial+1}/{n_trials}")

    Q1 = np.array(Q1_list)
    Q2 = np.array(Q2_list)
    Q3 = np.array(Q3_list)
    Q4 = np.array(Q4_list)

    C12 = np.mean(Q1 * Q2)
    C23 = np.mean(Q2 * Q3)
    C34 = np.mean(Q3 * Q4)
    C14 = np.mean(Q1 * Q4)

    K4 = C12 + C23 + C34 - C14

    # Bootstrap error
    n_boot = 1000
    K4_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n_trials, n_trials)
        K4_b = (np.mean(Q1[idx] * Q2[idx]) + np.mean(Q2[idx] * Q3[idx]) +
               np.mean(Q3[idx] * Q4[idx]) - np.mean(Q1[idx] * Q4[idx]))
        K4_boot.append(K4_b)

    K4_err = np.std(K4_boot)
    violation_sigma = (K4 - 2) / K4_err

    print(f"\n  K₄ = {K4:.4f} ± {K4_err:.4f}")
    print(f"  Distance from bound (K₄ ≤ 2): {violation_sigma:.2f}σ")

    if K4 - 2*K4_err > 2:
        conclusion = "*** K₄ VIOLATION ***"
    elif K4 + 2*K4_err < 2:
        conclusion = "CLASSICAL: K₄ < 2"
    else:
        conclusion = "INCONCLUSIVE"

    print(f"\n  CONCLUSION: {conclusion}")

    return {
        'K4': K4, 'K4_err': K4_err,
        'C12': C12, 'C23': C23, 'C34': C34, 'C14': C14,
        'violation_sigma': violation_sigma,
        'conclusion': conclusion
    }


def main():
    parser = argparse.ArgumentParser(description='LGI Test')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all', 'quick'], default='quick')
    parser.add_argument('--tau', type=float, default=11.5)
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--output', '-o', default='/tmp/lgi_results.json')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("LEGGETT-GARG INEQUALITY TEST")
    print("="*70)
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  CUDA: {HAS_CUDA}")

    results = {}

    if args.phase == 'quick':
        # Quick test: C2 + invasiveness + V1 with fewer trials
        print("\n  Running quick LGI test...")
        results['C2'] = test_C2_three_time(tau=args.tau, n_trials=200)
        results['C3'] = test_C3_invasiveness(tau=args.tau, n_trials=200)
        results['V1'] = asdict(test_V1_primary_LGI(tau=args.tau, n_trials=500))

    elif args.phase == '1':
        results['N1'] = test_N1_correlation_structure()
        results['N2'] = test_N2_dichotomic_stability()

    elif args.phase == '2':
        results['C2'] = test_C2_three_time(tau=args.tau, n_trials=args.trials)
        results['C3'] = test_C3_invasiveness(tau=args.tau, n_trials=args.trials)

    elif args.phase == '3':
        results['V1'] = asdict(test_V1_primary_LGI(tau=args.tau, n_trials=args.trials))
        results['V2'] = test_V2_four_time(tau=args.tau, n_trials=args.trials)

    elif args.phase == 'all':
        # Full protocol
        results['N1'] = test_N1_correlation_structure()
        results['N2'] = test_N2_dichotomic_stability()
        results['C2'] = test_C2_three_time(tau=args.tau, n_trials=args.trials)
        results['C3'] = test_C3_invasiveness(tau=args.tau, n_trials=args.trials)
        results['V1'] = asdict(test_V1_primary_LGI(tau=args.tau, n_trials=2500))
        results['V2'] = test_V2_four_time(tau=args.tau, n_trials=2500)

    # Save results
    # Convert numpy arrays to lists for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj

    with open(args.output, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
