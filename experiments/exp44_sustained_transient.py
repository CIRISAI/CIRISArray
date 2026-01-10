#!/usr/bin/env python3
"""
Experiment 44: Sustained Transient State
========================================

The array converges to a stable fixed point (k_eff ≈ 0.3) with zero variance.
To be useful as a sensor, we need to PREVENT this convergence and maintain
the system in a sensitive transient state.

Methods to explore:
1. Periodic reset - reinitialize oscillators at intervals
2. Noise injection - add small random perturbations continuously
3. Chaotic coupling - modify coupling to prevent convergence
4. Competing attractors - use multiple coupled subsystems
5. Driven oscillation - apply periodic forcing

Goal: Find a regime where variance is maintained AND external
perturbations can still be detected above the baseline.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class TransientConfig:
    """Configuration for sustained transient experiments."""
    n_ossicles: int = 16384  # 16k for faster iteration
    oscillator_depth: int = 64
    test_duration_sec: float = 30.0
    sample_rate_hz: float = 50.0


batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void batch_ossicle_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int depth, int total_elements, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float a = osc_a[idx];
    float b = osc_b[idx];
    float c = osc_c[idx];

    for (int i = 0; i < iterations; i++) {
        float da = coupling_ab * (b - a) + coupling_ca * (c - a);
        float db = coupling_ab * (a - b) + coupling_bc * (c - b);
        float dc = coupling_bc * (b - c) + coupling_ca * (a - c);

        a += da;
        b += db;
        c += dc;

        a = fmaxf(-10.0f, fminf(10.0f, a));
        b = fmaxf(-10.0f, fminf(10.0f, b));
        c = fmaxf(-10.0f, fminf(10.0f, c));
    }

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}
''', 'batch_ossicle_step')


inject_noise_kernel = cp.RawKernel(r'''
extern "C" __global__
void inject_noise(
    float* osc_a, float* osc_b, float* osc_c,
    float* noise, float amplitude,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    osc_a[idx] += noise[idx] * amplitude;
    osc_b[idx] += noise[idx + total_elements] * amplitude;
    osc_c[idx] += noise[idx + 2 * total_elements] * amplitude;
}
''', 'inject_noise')


class TransientArray:
    """Array with methods to maintain transient state."""

    def __init__(self, config: TransientConfig):
        self.config = config
        self.total_elements = config.n_ossicles * config.oscillator_depth

        self._init_oscillators()

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

        # Pre-allocate noise buffer
        self.noise_buffer = cp.random.random(3 * self.total_elements, dtype=cp.float32) - 0.5

    def _init_oscillators(self):
        """Initialize with random state."""
        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.5 - 0.25

    def reset(self):
        """Reset to random initial conditions."""
        self._init_oscillators()

    def partial_reset(self, fraction: float = 0.1):
        """Reset a fraction of oscillators."""
        n_reset = int(self.total_elements * fraction)
        indices = cp.random.choice(self.total_elements, n_reset, replace=False)

        self.osc_a[indices] = cp.random.random(n_reset, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b[indices] = cp.random.random(n_reset, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c[indices] = cp.random.random(n_reset, dtype=cp.float32) * 0.5 - 0.25

    def inject_noise(self, amplitude: float = 0.01):
        """Inject random noise into oscillators."""
        # Refresh noise buffer
        self.noise_buffer = cp.random.random(3 * self.total_elements, dtype=cp.float32) - 0.5

        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        inject_noise_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.noise_buffer, np.float32(amplitude), self.total_elements)
        )

    def step(self, iterations: int = 5):
        """Standard dynamics step."""
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.config.oscillator_depth, self.total_elements, iterations)
        )

    def measure(self) -> Dict[str, float]:
        """Measure current state."""
        sample_size = min(20000, self.total_elements)
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        var_total = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(var_total / 3.0, 1.0)
        k_eff = r_ab * (1 - x) * COUPLING_FACTOR * 1000

        return {
            'k_eff': k_eff,
            'variance': var_total,
            'correlation': r_ab
        }


def test_method(
    config: TransientConfig,
    name: str,
    step_func: Callable[[TransientArray], None]
) -> Dict:
    """Test a transient maintenance method."""

    array = TransientArray(config)

    n_samples = int(config.test_duration_sec * config.sample_rate_hz)
    interval = 1.0 / config.sample_rate_hz

    k_eff_series = []
    var_series = []

    start_time = time.perf_counter()

    for i in range(n_samples):
        sample_start = time.perf_counter()

        # Apply the method's step function
        step_func(array)

        # Measure
        m = array.measure()
        k_eff_series.append(m['k_eff'])
        var_series.append(m['variance'])

        # Rate control
        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    k_eff = np.array(k_eff_series)
    variance = np.array(var_series)

    return {
        'name': name,
        'k_eff_mean': np.mean(k_eff),
        'k_eff_std': np.std(k_eff),
        'k_eff_range': np.max(k_eff) - np.min(k_eff),
        'var_mean': np.mean(variance),
        'var_std': np.std(variance),
        'maintains_variance': np.std(k_eff) > 0.001,
        'time_series': k_eff
    }


def run_transient_experiments(config: TransientConfig) -> Dict:
    """Run all transient maintenance experiments."""

    print("\n" + "="*70)
    print("EXPERIMENT 44: SUSTAINED TRANSIENT STATE")
    print("Methods to prevent convergence to fixed point")
    print("="*70)

    results = []

    # Method 1: Baseline (no intervention)
    print("\n" + "-"*60)
    print("Method 1: BASELINE (no intervention)")
    print("-"*60)

    def baseline_step(array):
        array.step(5)

    r = test_method(config, "Baseline", baseline_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 2: Periodic full reset
    print("\n" + "-"*60)
    print("Method 2: PERIODIC FULL RESET (every 100 steps)")
    print("-"*60)

    reset_counter = [0]
    def periodic_reset_step(array):
        array.step(5)
        reset_counter[0] += 1
        if reset_counter[0] % 100 == 0:
            array.reset()

    reset_counter[0] = 0
    r = test_method(config, "Periodic Reset (100)", periodic_reset_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 3: Continuous noise injection (small)
    print("\n" + "-"*60)
    print("Method 3: CONTINUOUS NOISE INJECTION (amplitude=0.001)")
    print("-"*60)

    def noise_small_step(array):
        array.step(5)
        array.inject_noise(0.001)

    r = test_method(config, "Noise (0.001)", noise_small_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 4: Continuous noise injection (medium)
    print("\n" + "-"*60)
    print("Method 4: CONTINUOUS NOISE INJECTION (amplitude=0.01)")
    print("-"*60)

    def noise_medium_step(array):
        array.step(5)
        array.inject_noise(0.01)

    r = test_method(config, "Noise (0.01)", noise_medium_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 5: Continuous noise injection (large)
    print("\n" + "-"*60)
    print("Method 5: CONTINUOUS NOISE INJECTION (amplitude=0.1)")
    print("-"*60)

    def noise_large_step(array):
        array.step(5)
        array.inject_noise(0.1)

    r = test_method(config, "Noise (0.1)", noise_large_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 6: Partial reset (10% every 50 steps)
    print("\n" + "-"*60)
    print("Method 6: PARTIAL RESET (10% every 50 steps)")
    print("-"*60)

    partial_counter = [0]
    def partial_reset_step(array):
        array.step(5)
        partial_counter[0] += 1
        if partial_counter[0] % 50 == 0:
            array.partial_reset(0.1)

    partial_counter[0] = 0
    r = test_method(config, "Partial Reset (10%)", partial_reset_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Method 7: Combination - noise + partial reset
    print("\n" + "-"*60)
    print("Method 7: COMBINED (noise=0.01 + 5% reset every 100)")
    print("-"*60)

    combo_counter = [0]
    def combo_step(array):
        array.step(5)
        array.inject_noise(0.01)
        combo_counter[0] += 1
        if combo_counter[0] % 100 == 0:
            array.partial_reset(0.05)

    combo_counter[0] = 0
    r = test_method(config, "Combined", combo_step)
    results.append(r)
    print(f"  k_eff: {r['k_eff_mean']:.6f} ± {r['k_eff_std']:.6f}")
    print(f"  Maintains variance: {'YES' if r['maintains_variance'] else 'NO'}")

    # Summary
    print("\n" + "="*70)
    print("SUSTAINED TRANSIENT SUMMARY")
    print("="*70)

    print(f"\n  {'Method':<25} {'k_eff Mean':<12} {'k_eff Std':<12} {'Maintains?'}")
    print("-"*65)

    working_methods = []
    for r in results:
        maintains = "YES ✓" if r['maintains_variance'] else "no"
        print(f"  {r['name']:<25} {r['k_eff_mean']:<12.6f} {r['k_eff_std']:<12.6f} {maintains}")
        if r['maintains_variance']:
            working_methods.append(r['name'])

    if working_methods:
        print(f"\n  WORKING METHODS: {', '.join(working_methods)}")

        # Find best method (highest variance while being stable)
        best = max([r for r in results if r['maintains_variance']],
                   key=lambda x: x['k_eff_std'])
        print(f"  RECOMMENDED: {best['name']} (σ = {best['k_eff_std']:.6f})")
    else:
        print(f"\n  NO METHODS SUCCESSFULLY MAINTAINED VARIANCE")

    print("\n" + "="*70)

    return {'results': results, 'working_methods': working_methods}


def main():
    """Run sustained transient experiments."""

    print("="*70)
    print("CIRISARRAY SUSTAINED TRANSIENT STATE EXPLORATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = TransientConfig(
        test_duration_sec=20.0,  # 20s per method
        sample_rate_hz=50.0
    )

    results = run_transient_experiments(config)

    return results


if __name__ == "__main__":
    main()
