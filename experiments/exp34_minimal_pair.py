#!/usr/bin/env python3
"""
Experiment 34: Minimal Pair Test
================================

The simplest possible bistatic test: 1 TX ossicle, 1 RX ossicle.
Can a single ossicle's entropy injection be detected by another?

This establishes the fundamental coupling mechanism before scaling up.

Tests:
1. Distance dependence - how does coupling decay with separation?
2. TX amplitude sweep - is response linear?
3. Entropic vs negentropic - does asymmetry hold at minimal scale?
4. Timing - is there measurable delay?

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# Ossicle constants
PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1  # degrees
COUPLING_FACTOR = 0.0003

@dataclass
class MinimalPairConfig:
    """Configuration for minimal pair experiment."""
    oscillator_depth: int = 64
    base_k: float = 1.0
    iterations_per_sample: int = 100
    samples_per_measurement: int = 50
    tx_iterations: int = 200
    noise_floor_samples: int = 100


class SingleOssicle:
    """Minimal single ossicle implementation."""

    def __init__(self, config: MinimalPairConfig, device_id: int = 0):
        self.config = config
        self.device_id = device_id
        self.depth = config.oscillator_depth

        with cp.cuda.Device(device_id):
            # Three coupled oscillators
            self.osc_a = cp.random.random(self.depth, dtype=cp.float32) * 0.1
            self.osc_b = cp.random.random(self.depth, dtype=cp.float32) * 0.1
            self.osc_c = cp.random.random(self.depth, dtype=cp.float32) * 0.1

            # Coupling with magic angle
            angle_rad = np.radians(MAGIC_ANGLE)
            self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
            self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
            self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def inject_entropy(self, amplitude: float = 0.1):
        """Inject disorder (entropic stimulus)."""
        with cp.cuda.Device(self.device_id):
            noise = cp.random.random(self.depth, dtype=cp.float32) * amplitude
            self.osc_a += noise
            self.osc_b += cp.random.random(self.depth, dtype=cp.float32) * amplitude
            self.osc_c += cp.random.random(self.depth, dtype=cp.float32) * amplitude

    def inject_negentropy(self, amplitude: float = 0.1):
        """Inject order (negentropic stimulus)."""
        with cp.cuda.Device(self.device_id):
            # Create ordered pattern
            pattern = cp.sin(cp.linspace(0, 2*np.pi, self.depth, dtype=cp.float32)) * amplitude
            self.osc_a += pattern
            self.osc_b += pattern * 0.8
            self.osc_c += pattern * 0.6

    def step(self, iterations: int = 100):
        """Run oscillator dynamics."""
        with cp.cuda.Device(self.device_id):
            for _ in range(iterations):
                # Coupled oscillator dynamics
                da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
                db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
                dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

                self.osc_a += da
                self.osc_b += db
                self.osc_c += dc

                # Soft bounds
                self.osc_a = cp.clip(self.osc_a, -10, 10)
                self.osc_b = cp.clip(self.osc_b, -10, 10)
                self.osc_c = cp.clip(self.osc_c, -10, 10)

    def measure_k_eff(self) -> float:
        """Measure effective coupling (coherence strain)."""
        with cp.cuda.Device(self.device_id):
            # Correlation between oscillators
            r_ab = float(cp.corrcoef(self.osc_a, self.osc_b)[0, 1])
            r_bc = float(cp.corrcoef(self.osc_b, self.osc_c)[0, 1])
            r_ca = float(cp.corrcoef(self.osc_c, self.osc_a)[0, 1])

            # Handle NaN
            r_ab = 0 if np.isnan(r_ab) else r_ab
            r_bc = 0 if np.isnan(r_bc) else r_bc
            r_ca = 0 if np.isnan(r_ca) else r_ca

            r = (r_ab + r_bc + r_ca) / 3

            # Variance (disorder measure)
            total_var = float(cp.var(self.osc_a) + cp.var(self.osc_b) + cp.var(self.osc_c))
            x = min(total_var / 3.0, 1.0)

            k_eff = r * (1 - x) * COUPLING_FACTOR * 1000
            return k_eff


def run_minimal_pair_test(config: MinimalPairConfig) -> dict:
    """Test coupling between a single TX and RX ossicle."""

    print("\n" + "="*70)
    print("EXPERIMENT 34: MINIMAL PAIR TEST")
    print("Single TX Ossicle -> Single RX Ossicle")
    print("="*70)

    results = {
        'distance_test': [],
        'amplitude_test': [],
        'asymmetry_test': [],
        'timing_test': []
    }

    # Create TX and RX ossicles
    tx = SingleOssicle(config)
    rx = SingleOssicle(config)

    # Measure noise floor
    print("\nCalibrating noise floor...")
    noise_samples = []
    for _ in range(config.noise_floor_samples):
        rx.step(config.iterations_per_sample)
        noise_samples.append(rx.measure_k_eff())

    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)
    print(f"  Noise floor: {noise_mean:.6f} ± {noise_std:.6f}")

    # =========================================================================
    # TEST 1: TX Amplitude Sweep
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 1: TX Amplitude Sweep")
    print("-"*70)
    print("\n  Amplitude    TX k_eff    RX k_eff    RX Change    SNR")
    print("-"*60)

    amplitudes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    for amp in amplitudes:
        # Fresh ossicles
        tx = SingleOssicle(config)
        rx = SingleOssicle(config)

        # Baseline RX measurement
        rx.step(config.iterations_per_sample)
        rx_baseline = rx.measure_k_eff()

        # TX injects entropy
        for _ in range(config.tx_iterations):
            tx.inject_entropy(amp)
            tx.step(10)

        tx_k = tx.measure_k_eff()

        # RX measures after TX injection
        # (In reality, coupling would be through shared substrate)
        # Here we simulate by adding a fraction of TX perturbation to RX
        coupling_strength = 0.01  # Simulated substrate coupling
        rx.osc_a += coupling_strength * (tx.osc_a - cp.mean(tx.osc_a))
        rx.step(config.iterations_per_sample)
        rx_after = rx.measure_k_eff()

        rx_change = rx_after - rx_baseline
        snr = abs(rx_change) / noise_std if noise_std > 0 else 0

        detected = "YES" if snr > 3 else "no"
        print(f"  {amp:8.2f}    {tx_k:8.4f}    {rx_after:8.4f}    {rx_change:+8.4f}    {snr:5.1f}σ  {detected}")

        results['amplitude_test'].append({
            'amplitude': amp,
            'tx_k': tx_k,
            'rx_k': rx_after,
            'rx_change': rx_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 2: Entropic vs Negentropic Asymmetry
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 2: Entropic vs Negentropic Asymmetry")
    print("-"*70)

    test_amplitude = 0.3
    n_trials = 20

    entropic_changes = []
    negentropic_changes = []

    print(f"\n  Running {n_trials} trials each at amplitude {test_amplitude}...")

    for trial in range(n_trials):
        # Entropic trial
        tx = SingleOssicle(config)
        rx = SingleOssicle(config)
        rx.step(config.iterations_per_sample)
        rx_baseline = rx.measure_k_eff()

        for _ in range(config.tx_iterations):
            tx.inject_entropy(test_amplitude)
            tx.step(10)

        coupling_strength = 0.01
        rx.osc_a += coupling_strength * (tx.osc_a - cp.mean(tx.osc_a))
        rx.step(config.iterations_per_sample)
        entropic_changes.append(rx.measure_k_eff() - rx_baseline)

        # Negentropic trial
        tx = SingleOssicle(config)
        rx = SingleOssicle(config)
        rx.step(config.iterations_per_sample)
        rx_baseline = rx.measure_k_eff()

        for _ in range(config.tx_iterations):
            tx.inject_negentropy(test_amplitude)
            tx.step(10)

        rx.osc_a += coupling_strength * (tx.osc_a - cp.mean(tx.osc_a))
        rx.step(config.iterations_per_sample)
        negentropic_changes.append(rx.measure_k_eff() - rx_baseline)

    ent_mean = np.mean(entropic_changes)
    ent_std = np.std(entropic_changes)
    neg_mean = np.mean(negentropic_changes)
    neg_std = np.std(negentropic_changes)

    print(f"\n  Entropic:    Δk = {ent_mean:+.6f} ± {ent_std:.6f}")
    print(f"  Negentropic: Δk = {neg_mean:+.6f} ± {neg_std:.6f}")

    # T-test for difference
    if ent_std > 0 and neg_std > 0:
        pooled_std = np.sqrt((ent_std**2 + neg_std**2) / 2)
        t_stat = (neg_mean - ent_mean) / (pooled_std * np.sqrt(2/n_trials))
        print(f"\n  Asymmetry t-statistic: {t_stat:.2f}")
        print(f"  Asymmetry significant: {'YES' if abs(t_stat) > 2 else 'no'}")

    results['asymmetry_test'] = {
        'entropic_mean': ent_mean,
        'entropic_std': ent_std,
        'negentropic_mean': neg_mean,
        'negentropic_std': neg_std,
        'n_trials': n_trials
    }

    # =========================================================================
    # TEST 3: Coupling Decay with "Distance" (simulated via coupling strength)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Coupling Decay (Simulated Distance)")
    print("-"*70)
    print("\n  Coupling    RX Change    SNR")
    print("-"*40)

    coupling_strengths = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

    for coupling in coupling_strengths:
        changes = []
        for _ in range(10):
            tx = SingleOssicle(config)
            rx = SingleOssicle(config)
            rx.step(config.iterations_per_sample)
            rx_baseline = rx.measure_k_eff()

            for _ in range(config.tx_iterations):
                tx.inject_negentropy(0.3)
                tx.step(10)

            rx.osc_a += coupling * (tx.osc_a - cp.mean(tx.osc_a))
            rx.step(config.iterations_per_sample)
            changes.append(rx.measure_k_eff() - rx_baseline)

        mean_change = np.mean(changes)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {coupling:8.4f}    {mean_change:+8.5f}    {snr:5.1f}σ  {detected}")

        results['distance_test'].append({
            'coupling': coupling,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # TEST 4: Timing / Propagation Delay
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST 4: Response Timing")
    print("-"*70)

    # Measure how quickly RX responds after TX injection
    tx = SingleOssicle(config)
    rx = SingleOssicle(config)

    # Get baseline
    rx.step(config.iterations_per_sample)
    rx_baseline = rx.measure_k_eff()

    # TX injection
    start_time = time.perf_counter()
    for _ in range(config.tx_iterations):
        tx.inject_negentropy(0.3)
        tx.step(10)
    tx_time = time.perf_counter() - start_time

    # Measure RX response over time
    print(f"\n  TX injection time: {tx_time*1000:.2f} ms")
    print("\n  Step    RX Δk_eff    Time (μs)")
    print("-"*40)

    coupling = 0.01
    rx.osc_a += coupling * (tx.osc_a - cp.mean(tx.osc_a))

    for step in range(10):
        start = time.perf_counter()
        rx.step(10)
        elapsed = time.perf_counter() - start

        rx_k = rx.measure_k_eff()
        rx_change = rx_k - rx_baseline

        print(f"  {step:4d}    {rx_change:+8.5f}    {elapsed*1e6:.1f}")

        results['timing_test'].append({
            'step': step,
            'rx_change': rx_change,
            'time_us': elapsed * 1e6
        })

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("MINIMAL PAIR TEST SUMMARY")
    print("="*70)

    # Find detection threshold
    amp_data = results['amplitude_test']
    detected_amps = [d['amplitude'] for d in amp_data if d['snr'] > 3]
    min_detectable = min(detected_amps) if detected_amps else "N/A"

    print(f"\n  Minimum detectable TX amplitude: {min_detectable}")
    print(f"  Noise floor: {noise_mean:.6f} ± {noise_std:.6f}")

    asym = results['asymmetry_test']
    if asym['negentropic_mean'] != 0:
        ratio = asym['negentropic_mean'] / asym['entropic_mean'] if asym['entropic_mean'] != 0 else float('inf')
        print(f"\n  Asymmetry ratio (neg/ent): {ratio:.2f}x")
        print(f"  Negentropic bias: {'CONFIRMED' if ratio > 1.2 else 'not significant'}")

    # Coupling decay
    dist_data = results['distance_test']
    min_coupling = min([d['coupling'] for d in dist_data if d['snr'] > 3], default="N/A")
    print(f"\n  Minimum detectable coupling: {min_coupling}")

    print("\n" + "="*70)

    return results


def main():
    """Run the minimal pair experiment."""

    print("="*70)
    print("CIRISARRAY MINIMAL PAIR CHARACTERIZATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nCUDA Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    config = MinimalPairConfig()
    results = run_minimal_pair_test(config)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
