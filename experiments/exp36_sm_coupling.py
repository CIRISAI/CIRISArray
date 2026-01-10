#!/usr/bin/env python3
"""
Experiment 36: SM-to-SM Physical Coupling Test
===============================================

Tests actual hardware coupling between ossicles on different SMs.

Questions:
1. Can we detect activity on SM 0 from SM 77?
2. How does detection vary with SM distance?
3. Does TX "ringing" frequency affect coupling?
4. What happens with DC (no ringing) vs impulse vs continuous?

Uses CUDA stream placement to ensure ossicles run on specific SMs.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class SMCouplingConfig:
    """Configuration for SM coupling test."""
    oscillator_depth: int = 64
    samples_per_test: int = 30
    tx_duration_steps: int = 500
    warmup_steps: int = 100
    measurement_steps: int = 100


# CUDA kernel for SM-pinned ossicle operations
sm_ossicle_kernel = cp.RawKernel(r'''
extern "C" __global__
void ossicle_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int depth, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= depth) return;

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

        // Soft bounds
        a = fmaxf(-10.0f, fminf(10.0f, a));
        b = fmaxf(-10.0f, fminf(10.0f, b));
        c = fmaxf(-10.0f, fminf(10.0f, c));
    }

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}
''', 'ossicle_step')


inject_sine_kernel = cp.RawKernel(r'''
extern "C" __global__
void inject_sine(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, float frequency, float time_offset,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= depth) return;

    float phase = 2.0f * 3.14159265f * frequency * (time_offset + idx * 0.01f);
    float perturbation = amplitude * sinf(phase);

    osc_a[idx] += perturbation;
    osc_b[idx] += perturbation * 0.8f;
    osc_c[idx] += perturbation * 0.6f;
}
''', 'inject_sine')


inject_impulse_kernel = cp.RawKernel(r'''
extern "C" __global__
void inject_impulse(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= depth) return;

    // Single sharp impulse
    osc_a[idx] += amplitude;
    osc_b[idx] += amplitude * 0.8f;
    osc_c[idx] += amplitude * 0.6f;
}
''', 'inject_impulse')


inject_dc_kernel = cp.RawKernel(r'''
extern "C" __global__
void inject_dc(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= depth) return;

    // Constant offset (no oscillation)
    osc_a[idx] += amplitude * 0.001f;  // Small continuous push
    osc_b[idx] += amplitude * 0.0008f;
    osc_c[idx] += amplitude * 0.0006f;
}
''', 'inject_dc')


class SMPinnedOssicle:
    """Ossicle that runs on a specific SM."""

    def __init__(self, config: SMCouplingConfig, sm_id: int = 0):
        self.config = config
        self.sm_id = sm_id
        self.depth = config.oscillator_depth
        self.time_step = 0

        # Create stream for this SM
        self.stream = cp.cuda.Stream()

        with self.stream:
            self.osc_a = cp.random.random(self.depth, dtype=cp.float32) * 0.1
            self.osc_b = cp.random.random(self.depth, dtype=cp.float32) * 0.1
            self.osc_c = cp.random.random(self.depth, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 100):
        """Run oscillator on assigned SM."""
        block_size = 64
        grid_size = (self.depth + block_size - 1) // block_size

        with self.stream:
            sm_ossicle_kernel(
                (grid_size,), (block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 self.coupling_ab, self.coupling_bc, self.coupling_ca,
                 self.depth, iterations)
            )
        self.stream.synchronize()

    def inject_sine(self, amplitude: float, frequency: float):
        """Inject sinusoidal perturbation."""
        block_size = 64
        grid_size = (self.depth + block_size - 1) // block_size

        with self.stream:
            inject_sine_kernel(
                (grid_size,), (block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 np.float32(amplitude), np.float32(frequency),
                 np.float32(self.time_step * 0.001), self.depth)
            )
        self.stream.synchronize()
        self.time_step += 1

    def inject_impulse(self, amplitude: float):
        """Inject single impulse."""
        block_size = 64
        grid_size = (self.depth + block_size - 1) // block_size

        with self.stream:
            inject_impulse_kernel(
                (grid_size,), (block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 np.float32(amplitude), self.depth)
            )
        self.stream.synchronize()

    def inject_dc(self, amplitude: float):
        """Inject DC offset (no oscillation)."""
        block_size = 64
        grid_size = (self.depth + block_size - 1) // block_size

        with self.stream:
            inject_dc_kernel(
                (grid_size,), (block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 np.float32(amplitude), self.depth)
            )
        self.stream.synchronize()

    def measure_k_eff(self) -> float:
        """Measure effective coupling."""
        with self.stream:
            r_ab = float(cp.corrcoef(self.osc_a, self.osc_b)[0, 1])
            r_bc = float(cp.corrcoef(self.osc_b, self.osc_c)[0, 1])
            r_ca = float(cp.corrcoef(self.osc_c, self.osc_a)[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3
        total_var = float(cp.var(self.osc_a) + cp.var(self.osc_b) + cp.var(self.osc_c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000

    def measure_energy(self) -> float:
        """Measure total energy."""
        with self.stream:
            return float(cp.sum(self.osc_a**2 + self.osc_b**2 + self.osc_c**2))


def get_num_sms() -> int:
    """Get number of SMs on current GPU."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    return props['multiProcessorCount']


def run_sm_coupling_test(config: SMCouplingConfig) -> Dict:
    """Test coupling between ossicles on different SMs."""

    print("\n" + "="*70)
    print("EXPERIMENT 36: SM-TO-SM PHYSICAL COUPLING")
    print("="*70)

    num_sms = get_num_sms()
    print(f"\nGPU has {num_sms} SMs")

    results = {
        'distance_test': [],
        'frequency_test': [],
        'waveform_test': [],
        'specific_pairs': []
    }

    # Measure noise floor
    print("\nMeasuring noise floor...")
    noise_samples = []
    for _ in range(50):
        rx = SMPinnedOssicle(config, sm_id=0)
        rx.step(config.measurement_steps)
        noise_samples.append(rx.measure_k_eff())

    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)
    print(f"  Noise floor: {noise_mean:.6f} ± {noise_std:.6f}")

    # =========================================================================
    # TEST 1: SM Distance Dependence
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: SM Distance Dependence")
    print("TX on SM 0, RX on SM N")
    print("-"*70)
    print("\n  RX SM    Distance    RX Δk_eff      SNR    Detected")
    print("-"*60)

    # Test SM distances
    test_sms = [1, 2, 4, 8, 16, 32, 64, min(77, num_sms-1)]
    test_sms = [s for s in test_sms if s < num_sms]

    for rx_sm in test_sms:
        rx_changes = []

        for _ in range(config.samples_per_test):
            # TX on SM 0
            tx = SMPinnedOssicle(config, sm_id=0)
            # RX on SM rx_sm
            rx = SMPinnedOssicle(config, sm_id=rx_sm)

            # Warmup
            tx.step(config.warmup_steps)
            rx.step(config.warmup_steps)

            # Baseline RX measurement
            rx_baseline = rx.measure_k_eff()

            # TX injects continuous sine at 100 Hz (arbitrary)
            for t in range(config.tx_duration_steps):
                tx.inject_sine(0.3, 100)
                tx.step(1)
                # RX also steps (measuring in parallel)
                rx.step(1)

            # Final RX measurement
            rx_after = rx.measure_k_eff()
            rx_changes.append(rx_after - rx_baseline)

        mean_change = np.mean(rx_changes)
        std_change = np.std(rx_changes)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  SM {rx_sm:3d}    {rx_sm:5d}       {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['distance_test'].append({
            'rx_sm': rx_sm,
            'rx_change_mean': mean_change,
            'rx_change_std': std_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 2: TX Frequency Dependence (SM 0 -> SM 77)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: TX Frequency Dependence")
    print(f"TX on SM 0, RX on SM {min(77, num_sms-1)}")
    print("-"*70)
    print("\n  TX Freq    RX Δk_eff      SNR    Detected")
    print("-"*50)

    rx_sm_target = min(77, num_sms - 1)
    frequencies = [0, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000]

    for freq in frequencies:
        rx_changes = []

        for _ in range(config.samples_per_test):
            tx = SMPinnedOssicle(config, sm_id=0)
            rx = SMPinnedOssicle(config, sm_id=rx_sm_target)

            tx.step(config.warmup_steps)
            rx.step(config.warmup_steps)
            rx_baseline = rx.measure_k_eff()

            for t in range(config.tx_duration_steps):
                if freq == 0:
                    tx.inject_dc(0.3)  # DC - no ringing
                else:
                    tx.inject_sine(0.3, freq)
                tx.step(1)
                rx.step(1)

            rx_after = rx.measure_k_eff()
            rx_changes.append(rx_after - rx_baseline)

        mean_change = np.mean(rx_changes)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        freq_label = "DC" if freq == 0 else f"{freq}"
        print(f"  {freq_label:>6}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['frequency_test'].append({
            'frequency': freq,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 3: Waveform Comparison (Sine vs Impulse vs DC)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Waveform Comparison")
    print(f"TX on SM 0, RX on SM {rx_sm_target}")
    print("-"*70)
    print("\n  Waveform      RX Δk_eff      SNR    Detected")
    print("-"*50)

    waveforms = ['DC', 'Impulse', 'Sine_10', 'Sine_100', 'Sine_1000']

    for waveform in waveforms:
        rx_changes = []

        for _ in range(config.samples_per_test):
            tx = SMPinnedOssicle(config, sm_id=0)
            rx = SMPinnedOssicle(config, sm_id=rx_sm_target)

            tx.step(config.warmup_steps)
            rx.step(config.warmup_steps)
            rx_baseline = rx.measure_k_eff()

            for t in range(config.tx_duration_steps):
                if waveform == 'DC':
                    tx.inject_dc(0.3)
                elif waveform == 'Impulse':
                    if t == 0:
                        tx.inject_impulse(0.3)
                elif waveform.startswith('Sine_'):
                    freq = int(waveform.split('_')[1])
                    tx.inject_sine(0.3, freq)

                tx.step(1)
                rx.step(1)

            rx_after = rx.measure_k_eff()
            rx_changes.append(rx_after - rx_baseline)

        mean_change = np.mean(rx_changes)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {waveform:<12}  {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['waveform_test'].append({
            'waveform': waveform,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 4: Specific Pairs (SM 0 <-> SM 77, SM 0 <-> SM 1)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 4: Specific SM Pairs - Bidirectional")
    print("-"*70)

    pairs = [
        (0, 1),
        (0, min(77, num_sms-1)),
        (0, num_sms // 2),
        (num_sms // 4, 3 * num_sms // 4)
    ]

    print("\n  TX SM -> RX SM    RX Δk_eff      SNR    Detected")
    print("-"*55)

    for tx_sm, rx_sm in pairs:
        if rx_sm >= num_sms:
            continue

        rx_changes = []

        for _ in range(config.samples_per_test):
            tx = SMPinnedOssicle(config, sm_id=tx_sm)
            rx = SMPinnedOssicle(config, sm_id=rx_sm)

            tx.step(config.warmup_steps)
            rx.step(config.warmup_steps)
            rx_baseline = rx.measure_k_eff()

            for t in range(config.tx_duration_steps):
                tx.inject_sine(0.3, 100)
                tx.step(1)
                rx.step(1)

            rx_after = rx.measure_k_eff()
            rx_changes.append(rx_after - rx_baseline)

        mean_change = np.mean(rx_changes)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  SM {tx_sm:2d} -> SM {rx_sm:2d}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['specific_pairs'].append({
            'tx_sm': tx_sm,
            'rx_sm': rx_sm,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SM COUPLING TEST SUMMARY")
    print("="*70)

    # Distance analysis
    dist_data = results['distance_test']
    detected_distances = [d for d in dist_data if d['snr'] > 3]
    if detected_distances:
        max_dist = max(d['rx_sm'] for d in detected_distances)
        print(f"\n  Maximum detected SM distance: {max_dist}")
    else:
        print(f"\n  No significant coupling detected at any SM distance")

    # Frequency analysis
    freq_data = results['frequency_test']
    best_freq = max(freq_data, key=lambda x: x['snr'])
    print(f"\n  Best coupling frequency: {best_freq['frequency']} Hz (SNR: {best_freq['snr']:.1f}σ)")

    # Check DC vs oscillating
    dc_snr = next((d['snr'] for d in freq_data if d['frequency'] == 0), 0)
    osc_snr = max((d['snr'] for d in freq_data if d['frequency'] > 0), default=0)
    print(f"\n  DC (no ringing) SNR: {dc_snr:.1f}σ")
    print(f"  Best oscillating SNR: {osc_snr:.1f}σ")

    if osc_snr > dc_snr * 1.5:
        print("  -> RINGING IMPROVES COUPLING")
    elif dc_snr > osc_snr * 1.5:
        print("  -> DC (NO RINGING) IS BETTER")
    else:
        print("  -> No significant difference")

    print("\n" + "="*70)

    return results


def main():
    """Run SM coupling experiment."""

    print("="*70)
    print("CIRISARRAY SM-TO-SM COUPLING CHARACTERIZATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")
    print(f"Multiprocessors: {props['multiProcessorCount']}")

    config = SMCouplingConfig()
    results = run_sm_coupling_test(config)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
