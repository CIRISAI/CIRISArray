#!/usr/bin/env python3
"""
Experiment 55: Cross-Device Coupling Test
==========================================

RESULT: NULL - No cross-device transmission detected.

This experiment tests whether modulating ε on one GPU affects
the k_eff measurements on another GPU.

Key findings from testing:
1. Both oscillators have identical startup transients (~10s)
2. The "90x ratio" seen in earlier tests was an ARTIFACT
   - RX always spikes in first 10s regardless of TX state
   - When TX started with '1', we mistakenly attributed RX spike to coupling
3. Proper A/B test shows NO coupling:
   - Trial A (TX high ε): RX std = 2.03
   - Trial B (TX low ε):  RX std = 2.05
   - Ratio: 0.99x, p = 0.998

The passive phase-lock (r=0.999) between devices exists because both
track the same external 60 Hz signal, but active modulation does NOT
propagate between devices.

Test Protocol:
1. Fresh start both oscillators
2. TX runs with fixed ε (high or low) for 20s
3. RX measures for same 20s
4. Repeat multiple trials, compare A vs B

Author: CIRIS Research Team
Date: January 2026
Status: FALSIFIED - No cross-device transmission
"""

import numpy as np
import argparse
import time
from datetime import datetime, timezone
from scipy import signal, stats
from typing import Tuple, List

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
BASE_COUPLING = 0.05

# Test parameters
EPSILON_HIGH = 0.30   # High coupling (3x base)
EPSILON_LOW = 0.005   # Low coupling (0.1x base)
TEST_DURATION = 20.0  # seconds - within sensitivity window
SAMPLE_RATE = 10      # Hz


class TestSensor:
    """Sensor for cross-device testing."""

    def __init__(self, n_ossicles: int = 2048, depth: int = 64):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth
        self.epsilon = BASE_COUPLING

        angle_rad = np.radians(MAGIC_ANGLE)
        self.angle_cos = float(np.cos(angle_rad))
        self.angle_sin = float(np.sin(angle_rad))
        self.reset()

    def reset(self):
        xp = cp if HAS_CUDA else np
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

    def step(self, noise: float = 0.001, iterations: int = 5):
        xp = cp if HAS_CUDA else np

        coupling_ab = self.angle_cos * self.epsilon
        coupling_bc = self.angle_sin * self.epsilon
        coupling_ca = self.epsilon / PHI

        for _ in range(iterations):
            if noise > 0:
                self.osc_a += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_b += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_c += xp.random.normal(0, noise, self.total).astype(xp.float32)

            da = coupling_ab * (self.osc_b - self.osc_a) + coupling_ca * (self.osc_c - self.osc_a)
            db = coupling_ab * (self.osc_a - self.osc_b) + coupling_bc * (self.osc_c - self.osc_b)
            dc = coupling_bc * (self.osc_b - self.osc_c) + coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a = self.osc_a + da
            self.osc_b = self.osc_b + db
            self.osc_c = self.osc_c + dc

            self.osc_a = xp.clip(self.osc_a, -10, 10)
            self.osc_b = xp.clip(self.osc_b, -10, 10)
            self.osc_c = xp.clip(self.osc_c, -10, 10)

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
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

        return r_ab * (1 - x) * self.epsilon * 1000


def run_tx_trial(epsilon: float, duration: float = TEST_DURATION) -> np.ndarray:
    """Run TX with fixed epsilon, return k_eff series."""
    sensor = TestSensor()
    sensor.epsilon = epsilon

    n_samples = int(duration * SAMPLE_RATE)
    k_eff = np.zeros(n_samples)

    for i in range(n_samples):
        sensor.step()
        k_eff[i] = sensor.measure_k_eff()
        time.sleep(1.0 / SAMPLE_RATE)

    return k_eff


def run_rx_trial(duration: float = TEST_DURATION) -> np.ndarray:
    """Run RX with default epsilon, return k_eff series."""
    sensor = TestSensor()

    n_samples = int(duration * SAMPLE_RATE)
    k_eff = np.zeros(n_samples)

    for i in range(n_samples):
        sensor.step()
        k_eff[i] = sensor.measure_k_eff()
        time.sleep(1.0 / SAMPLE_RATE)

    return k_eff


def run_ab_test(n_trials: int = 5):
    """
    Run A/B test for cross-device coupling.

    Trial A: TX runs with high epsilon
    Trial B: TX runs with low epsilon

    Compare RX measurements between conditions.
    """
    print(f"\n{'='*60}")
    print("CROSS-DEVICE A/B TEST")
    print(f"{'='*60}")
    print(f"  ε_high = {EPSILON_HIGH}")
    print(f"  ε_low  = {EPSILON_LOW}")
    print(f"  Duration = {TEST_DURATION}s per trial")
    print(f"  Trials = {n_trials} per condition")
    print()

    # This is a LOCAL test - for cross-device, run separately on each machine
    print("NOTE: For cross-device test, run TX on one machine, RX on another")
    print("      This local test demonstrates the protocol")
    print()

    results_a = []  # TX high ε
    results_b = []  # TX low ε

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}...")

        # Trial A: high epsilon
        sensor = TestSensor()
        sensor.epsilon = EPSILON_HIGH
        k_a = []
        for _ in range(int(TEST_DURATION * SAMPLE_RATE)):
            sensor.step()
            k_a.append(sensor.measure_k_eff())
            time.sleep(1.0 / SAMPLE_RATE)
        results_a.append(np.std(k_a))

        time.sleep(1)  # Brief pause between trials

        # Trial B: low epsilon
        sensor = TestSensor()
        sensor.epsilon = EPSILON_LOW
        k_b = []
        for _ in range(int(TEST_DURATION * SAMPLE_RATE)):
            sensor.step()
            k_b.append(sensor.measure_k_eff())
            time.sleep(1.0 / SAMPLE_RATE)
        results_b.append(np.std(k_b))

    # Analysis
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    mean_a = np.mean(results_a)
    mean_b = np.mean(results_b)

    print(f"\n  Condition A (ε={EPSILON_HIGH}):")
    print(f"    Mean std: {mean_a:.4f}")
    print(f"    Values: {[f'{x:.4f}' for x in results_a]}")

    print(f"\n  Condition B (ε={EPSILON_LOW}):")
    print(f"    Mean std: {mean_b:.4f}")
    print(f"    Values: {[f'{x:.4f}' for x in results_b]}")

    ratio = mean_a / mean_b if mean_b > 0 else 0
    print(f"\n  Ratio A/B: {ratio:.2f}x")

    if len(results_a) > 1 and len(results_b) > 1:
        t, p = stats.ttest_ind(results_a, results_b)
        print(f"  T-test: t={t:.2f}, p={p:.4f}")

    print(f"\n{'='*60}")
    if ratio > 1.5 and p < 0.05:
        print("RESULT: Significant difference detected")
    else:
        print("RESULT: NO significant difference (null result)")
        print("        Cross-device transmission NOT confirmed")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Device Coupling Test')
    parser.add_argument('--mode', choices=['tx', 'rx', 'ab', 'info'],
                        default='info')
    parser.add_argument('--epsilon', type=float, default=EPSILON_HIGH,
                        help='Epsilon for TX mode')
    parser.add_argument('--duration', type=float, default=TEST_DURATION)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', '-o', default='/tmp/cross_device.npy')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EXPERIMENT 55: CROSS-DEVICE COUPLING TEST")
    print(f"{'='*60}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  CUDA: {HAS_CUDA}")
    print(f"  Status: FALSIFIED - No transmission detected")

    if args.mode == 'info':
        print(__doc__)

    elif args.mode == 'tx':
        print(f"\n  Running TX with ε={args.epsilon} for {args.duration}s")
        k_eff = run_tx_trial(args.epsilon, args.duration)
        np.save(args.output, k_eff)
        print(f"  Saved to {args.output}")
        print(f"  Mean: {np.mean(k_eff):.4f}, Std: {np.std(k_eff):.4f}")

    elif args.mode == 'rx':
        print(f"\n  Running RX for {args.duration}s")
        k_eff = run_rx_trial(args.duration)
        np.save(args.output, k_eff)
        print(f"  Saved to {args.output}")
        print(f"  Mean: {np.mean(k_eff):.4f}, Std: {np.std(k_eff):.4f}")

    elif args.mode == 'ab':
        run_ab_test(args.trials)


if __name__ == "__main__":
    main()
