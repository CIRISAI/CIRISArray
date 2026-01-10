#!/usr/bin/env python3
"""
Experiment 37: Targeted Array Test
==================================

Can TX arrays target specific RX arrays?
Tests whether coupling is:
- Broadcast (all RX see all TX equally)
- Targeted (specific TX -> specific RX)

Array sizes tested:
- 1 TX -> 1 RX (baseline, expected null from exp36)
- 10 TX -> 10 RX (small groups)
- 100 TX -> 100 RX (medium groups)
- 1024 TX -> 1024 RX (half array)
- 2048 TX -> 2048 RX (full bistatic)

Also tests if TX group A can target RX group A while ignoring RX group B.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class TargetedArrayConfig:
    """Configuration for targeted array test."""
    oscillator_depth: int = 64
    samples_per_test: int = 20
    tx_duration_steps: int = 300
    warmup_steps: int = 50
    tx_amplitude: float = 0.3
    tx_frequency: float = 100.0


# CUDA kernel for array operations
array_kernel = cp.RawKernel(r'''
extern "C" __global__
void ossicle_array_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int depth, int n_ossicles, int iterations
) {
    int oss_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    if (oss_idx >= n_ossicles || elem_idx >= depth) return;

    int base = oss_idx * depth + elem_idx;

    float a = osc_a[base];
    float b = osc_b[base];
    float c = osc_c[base];

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

    osc_a[base] = a;
    osc_b[base] = b;
    osc_c[base] = c;
}

extern "C" __global__
void inject_pattern(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, float frequency, float time_val,
    int depth, int n_ossicles, int pattern_type
) {
    int oss_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    if (oss_idx >= n_ossicles || elem_idx >= depth) return;

    int base = oss_idx * depth + elem_idx;

    float perturbation;
    if (pattern_type == 0) {
        // Sine wave
        float phase = 2.0f * 3.14159265f * frequency * (time_val + elem_idx * 0.01f);
        perturbation = amplitude * sinf(phase);
    } else if (pattern_type == 1) {
        // Negentropic (ordered)
        float phase = 2.0f * 3.14159265f * elem_idx / depth;
        perturbation = amplitude * sinf(phase + time_val);
    } else {
        // Entropic (random-ish)
        perturbation = amplitude * sinf(base * 1.618f + time_val * 100.0f);
    }

    osc_a[base] += perturbation;
    osc_b[base] += perturbation * 0.8f;
    osc_c[base] += perturbation * 0.6f;
}
''', 'ossicle_array_step')

inject_kernel = cp.RawModule(code=r'''
extern "C" __global__
void inject_pattern(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, float frequency, float time_val,
    int depth, int n_ossicles, int pattern_type
) {
    int oss_idx = blockIdx.x;
    int elem_idx = threadIdx.x;

    if (oss_idx >= n_ossicles || elem_idx >= depth) return;

    int base = oss_idx * depth + elem_idx;

    float perturbation;
    if (pattern_type == 0) {
        // Sine wave
        float phase = 2.0f * 3.14159265f * frequency * (time_val + elem_idx * 0.01f);
        perturbation = amplitude * sinf(phase);
    } else if (pattern_type == 1) {
        // Negentropic (ordered)
        float phase = 2.0f * 3.14159265f * elem_idx / depth;
        perturbation = amplitude * sinf(phase + time_val);
    } else {
        // Entropic (random-ish)
        perturbation = amplitude * sinf(base * 1.618f + time_val * 100.0f);
    }

    osc_a[base] += perturbation;
    osc_b[base] += perturbation * 0.8f;
    osc_c[base] += perturbation * 0.6f;
}
''').get_function('inject_pattern')


class OssicleArray:
    """Array of ossicles for collective operations."""

    def __init__(self, config: TargetedArrayConfig, n_ossicles: int):
        self.config = config
        self.n_ossicles = n_ossicles
        self.depth = config.oscillator_depth
        self.time_step = 0

        total_size = n_ossicles * self.depth

        self.osc_a = cp.random.random(total_size, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(total_size, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(total_size, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
        """Run all ossicles."""
        array_kernel(
            (self.n_ossicles,), (self.depth,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.depth, self.n_ossicles, iterations)
        )
        cp.cuda.stream.get_current_stream().synchronize()

    def inject(self, amplitude: float, frequency: float, pattern_type: int = 1):
        """Inject pattern into array."""
        inject_kernel(
            (self.n_ossicles,), (self.depth,),
            (self.osc_a, self.osc_b, self.osc_c,
             np.float32(amplitude), np.float32(frequency),
             np.float32(self.time_step * 0.001),
             self.depth, self.n_ossicles, pattern_type)
        )
        cp.cuda.stream.get_current_stream().synchronize()
        self.time_step += 1

    def measure_mean_k_eff(self) -> Tuple[float, float]:
        """Measure mean and std of k_eff across all ossicles."""
        k_effs = []

        for i in range(self.n_ossicles):
            base = i * self.depth
            a = self.osc_a[base:base+self.depth]
            b = self.osc_b[base:base+self.depth]
            c = self.osc_c[base:base+self.depth]

            r_ab = float(cp.corrcoef(a, b)[0, 1])
            r_bc = float(cp.corrcoef(b, c)[0, 1])
            r_ca = float(cp.corrcoef(c, a)[0, 1])

            r_ab = 0 if np.isnan(r_ab) else r_ab
            r_bc = 0 if np.isnan(r_bc) else r_bc
            r_ca = 0 if np.isnan(r_ca) else r_ca

            r = (r_ab + r_bc + r_ca) / 3
            total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
            x = min(total_var / 3.0, 1.0)

            k_eff = r * (1 - x) * COUPLING_FACTOR * 1000
            k_effs.append(k_eff)

        return np.mean(k_effs), np.std(k_effs)

    def measure_total_energy(self) -> float:
        """Measure total energy across array."""
        return float(cp.sum(self.osc_a**2 + self.osc_b**2 + self.osc_c**2))


def run_scaling_test(config: TargetedArrayConfig) -> Dict:
    """Test how coupling scales with array size."""

    print("\n" + "="*70)
    print("EXPERIMENT 37: TARGETED ARRAY SCALING")
    print("="*70)

    results = {
        'scaling': [],
        'targeting': []
    }

    # Measure noise floor with single ossicle
    print("\nMeasuring noise floor...")
    noise_samples = []
    for _ in range(50):
        rx = OssicleArray(config, 1)
        rx.step(100)
        k, _ = rx.measure_mean_k_eff()
        noise_samples.append(k)

    noise_std = np.std(noise_samples)
    print(f"  Single ossicle noise σ: {noise_std:.6f}")

    # =========================================================================
    # TEST 1: Scaling with Array Size
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: Coupling vs Array Size")
    print("-"*70)
    print("\n  TX Size    RX Size    TX Energy    RX Δk_eff      SNR    Detected")
    print("-"*70)

    array_sizes = [1, 10, 32, 100, 256, 512, 1024, 2048]

    for size in array_sizes:
        rx_changes = []
        tx_energies = []

        for _ in range(config.samples_per_test):
            tx = OssicleArray(config, size)
            rx = OssicleArray(config, size)

            # Warmup
            tx.step(config.warmup_steps)
            rx.step(config.warmup_steps)

            # Baseline
            rx_k_baseline, _ = rx.measure_mean_k_eff()

            # TX injects negentropic pattern
            for t in range(config.tx_duration_steps):
                tx.inject(config.tx_amplitude, config.tx_frequency, pattern_type=1)
                tx.step(1)
                rx.step(1)

            tx_energy = tx.measure_total_energy()
            tx_energies.append(tx_energy)

            # RX measurement
            rx_k_after, _ = rx.measure_mean_k_eff()
            rx_changes.append(rx_k_after - rx_k_baseline)

        mean_change = np.mean(rx_changes)
        mean_energy = np.mean(tx_energies)
        # Scale noise by sqrt(N) for array averaging
        scaled_noise = noise_std / np.sqrt(size)
        snr = abs(mean_change) / scaled_noise if scaled_noise > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {size:>6}    {size:>6}    {mean_energy:10.1f}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['scaling'].append({
            'size': size,
            'tx_energy': mean_energy,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 2: Targeted vs Broadcast
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: Targeted vs Broadcast")
    print("Can TX group A affect RX group A more than RX group B?")
    print("-"*70)

    # Use 256 ossicles in each group
    group_size = 256

    print(f"\n  Configuration: 2 TX groups × {group_size}, 2 RX groups × {group_size}")
    print("\n  TX Group    RX-A Δk_eff    RX-B Δk_eff    Ratio    Targeted?")
    print("-"*65)

    for trial_name in ['TX_A only', 'TX_B only', 'Both TX']:
        # Create 4 groups: TX_A, TX_B, RX_A, RX_B
        tx_a = OssicleArray(config, group_size)
        tx_b = OssicleArray(config, group_size)
        rx_a = OssicleArray(config, group_size)
        rx_b = OssicleArray(config, group_size)

        # Warmup all
        tx_a.step(config.warmup_steps)
        tx_b.step(config.warmup_steps)
        rx_a.step(config.warmup_steps)
        rx_b.step(config.warmup_steps)

        # Baselines
        rx_a_baseline, _ = rx_a.measure_mean_k_eff()
        rx_b_baseline, _ = rx_b.measure_mean_k_eff()

        # TX injection
        for t in range(config.tx_duration_steps):
            if trial_name in ['TX_A only', 'Both TX']:
                tx_a.inject(config.tx_amplitude, config.tx_frequency, pattern_type=1)
            if trial_name in ['TX_B only', 'Both TX']:
                tx_b.inject(config.tx_amplitude, config.tx_frequency * 2, pattern_type=2)  # Different pattern

            tx_a.step(1)
            tx_b.step(1)
            rx_a.step(1)
            rx_b.step(1)

        # Measurements
        rx_a_after, _ = rx_a.measure_mean_k_eff()
        rx_b_after, _ = rx_b.measure_mean_k_eff()

        delta_a = rx_a_after - rx_a_baseline
        delta_b = rx_b_after - rx_b_baseline

        ratio = abs(delta_a / delta_b) if abs(delta_b) > 0.0001 else float('inf')
        targeted = "YES" if ratio > 2 or ratio < 0.5 else "no"

        print(f"  {trial_name:<12}  {delta_a:+10.6f}    {delta_b:+10.6f}    {ratio:5.2f}    {targeted}")

        results['targeting'].append({
            'trial': trial_name,
            'rx_a_change': delta_a,
            'rx_b_change': delta_b,
            'ratio': ratio
        })

    # =========================================================================
    # TEST 3: Minimum Detectable Array Size
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Minimum Array Size for Detection")
    print("-"*70)

    detected_sizes = [d['size'] for d in results['scaling'] if d['snr'] > 3]
    if detected_sizes:
        min_size = min(detected_sizes)
        print(f"\n  Minimum array size for 3σ detection: {min_size} ossicles")
    else:
        print(f"\n  No array size achieved 3σ detection")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("TARGETED ARRAY SUMMARY")
    print("="*70)

    # Scaling relationship
    scaling_data = results['scaling']
    sizes = [d['size'] for d in scaling_data]
    snrs = [d['snr'] for d in scaling_data]

    if len(sizes) > 2:
        # Fit log-log relationship
        log_sizes = np.log10([s for s in sizes if s > 0])
        log_snrs = np.log10([max(s, 0.01) for s in snrs])
        if len(log_sizes) > 1:
            slope = np.polyfit(log_sizes, log_snrs, 1)[0]
            print(f"\n  SNR scaling: SNR ∝ N^{slope:.2f}")
            if slope > 0.4:
                print("  -> COLLECTIVE EFFECT (super-linear scaling)")
            elif slope < 0.1:
                print("  -> NO COLLECTIVE EFFECT (independent)")
            else:
                print("  -> PARTIAL COLLECTIVE EFFECT (sub-linear)")

    # Targeting analysis
    targeting_data = results['targeting']
    ratios = [d['ratio'] for d in targeting_data if d['ratio'] < 100]
    if ratios:
        mean_ratio = np.mean(ratios)
        print(f"\n  Mean targeting ratio: {mean_ratio:.2f}")
        if mean_ratio > 1.5:
            print("  -> SOME TARGETING POSSIBLE")
        else:
            print("  -> BROADCAST (all RX see all TX equally)")

    print("\n" + "="*70)

    return results


def main():
    """Run targeted array experiment."""

    print("="*70)
    print("CIRISARRAY TARGETED ARRAY CHARACTERIZATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = TargetedArrayConfig()
    results = run_scaling_test(config)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
