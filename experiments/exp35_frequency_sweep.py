#!/usr/bin/env python3
"""
Experiment 35: Frequency Sweep
==============================

Test what TX frequencies couple best to RX.
Does the system have resonance frequencies?

Varies:
- TX injection frequency (oscillation rate)
- TX waveform (sine, square, noise bursts)
- Measures RX response amplitude

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Dict
import time

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class FrequencySweepConfig:
    """Configuration for frequency sweep experiment."""
    oscillator_depth: int = 64
    tx_amplitude: float = 0.3
    substrate_coupling: float = 0.01
    samples_per_frequency: int = 20
    iterations_per_sample: int = 100


class FrequencyOssicle:
    """Ossicle with frequency-controlled injection."""

    def __init__(self, config: FrequencySweepConfig):
        self.config = config
        self.depth = config.oscillator_depth
        self.time_step = 0

        # Three coupled oscillators
        self.osc_a = cp.random.random(self.depth, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.depth, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.depth, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

    def inject_sine(self, frequency: float, amplitude: float, duration_steps: int):
        """Inject sinusoidal perturbation at given frequency."""
        for t in range(duration_steps):
            phase = 2 * np.pi * frequency * t / 1000  # Normalize to ~kHz range
            perturbation = amplitude * np.sin(phase)

            self.osc_a += cp.float32(perturbation)
            self.osc_b += cp.float32(perturbation * 0.8)
            self.osc_c += cp.float32(perturbation * 0.6)

            self.step(1)
            self.time_step += 1

    def inject_square(self, frequency: float, amplitude: float, duration_steps: int):
        """Inject square wave perturbation."""
        period = max(2, int(1000 / frequency)) if frequency > 0 else duration_steps
        half_period = max(1, period // 2)

        for t in range(duration_steps):
            # Square wave: +amp for half period, -amp for half period
            if (t // half_period) % 2 == 0:
                perturbation = amplitude
            else:
                perturbation = -amplitude

            self.osc_a += cp.float32(perturbation)
            self.osc_b += cp.float32(perturbation * 0.8)
            self.osc_c += cp.float32(perturbation * 0.6)

            self.step(1)
            self.time_step += 1

    def inject_burst(self, frequency: float, amplitude: float, duration_steps: int):
        """Inject noise bursts at given frequency."""
        period = max(4, int(1000 / frequency)) if frequency > 0 else duration_steps
        burst_length = max(1, period // 4)

        for t in range(duration_steps):
            if t % period < burst_length:
                # Noise burst
                noise = cp.random.random(self.depth, dtype=cp.float32) * amplitude
                self.osc_a += noise
                self.osc_b += cp.random.random(self.depth, dtype=cp.float32) * amplitude
                self.osc_c += cp.random.random(self.depth, dtype=cp.float32) * amplitude

            self.step(1)
            self.time_step += 1

    def step(self, iterations: int = 1):
        """Run oscillator dynamics."""
        for _ in range(iterations):
            da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
            db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
            dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a += da
            self.osc_b += db
            self.osc_c += dc

            self.osc_a = cp.clip(self.osc_a, -10, 10)
            self.osc_b = cp.clip(self.osc_b, -10, 10)
            self.osc_c = cp.clip(self.osc_c, -10, 10)

    def measure_k_eff(self) -> float:
        """Measure effective coupling."""
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
        """Measure total oscillator energy (amplitude)."""
        return float(cp.sum(self.osc_a**2 + self.osc_b**2 + self.osc_c**2))


def run_frequency_sweep(config: FrequencySweepConfig) -> Dict:
    """Run frequency sweep experiment."""

    print("\n" + "="*70)
    print("EXPERIMENT 35: FREQUENCY SWEEP")
    print("What TX frequencies couple best to RX?")
    print("="*70)

    results = {
        'sine': [],
        'square': [],
        'burst': []
    }

    # Frequencies to test (in arbitrary units, maps to ~Hz-kHz range)
    frequencies = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Measure noise floor
    print("\nMeasuring noise floor...")
    noise_samples = []
    for _ in range(50):
        rx = FrequencyOssicle(config)
        rx.step(config.iterations_per_sample)
        noise_samples.append(rx.measure_k_eff())

    noise_std = np.std(noise_samples)
    print(f"  Noise σ: {noise_std:.6f}")

    # =========================================================================
    # SINE WAVE SWEEP
    # =========================================================================
    print("\n" + "-"*70)
    print("SINE WAVE FREQUENCY SWEEP")
    print("-"*70)
    print("\n  Freq     TX Energy    RX Δk_eff      SNR    Detected")
    print("-"*60)

    for freq in frequencies:
        rx_changes = []
        tx_energies = []

        for _ in range(config.samples_per_frequency):
            tx = FrequencyOssicle(config)
            rx = FrequencyOssicle(config)

            # Baseline
            rx.step(config.iterations_per_sample)
            rx_baseline = rx.measure_k_eff()

            # TX injects at frequency
            tx.inject_sine(freq, config.tx_amplitude, 200)
            tx_energy = tx.measure_energy()
            tx_energies.append(tx_energy)

            # Couple to RX
            rx.osc_a += config.substrate_coupling * (tx.osc_a - cp.mean(tx.osc_a))
            rx.step(config.iterations_per_sample)

            rx_changes.append(rx.measure_k_eff() - rx_baseline)

        mean_change = np.mean(rx_changes)
        mean_energy = np.mean(tx_energies)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {freq:5d}    {mean_energy:9.2f}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['sine'].append({
            'frequency': freq,
            'tx_energy': mean_energy,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # SQUARE WAVE SWEEP
    # =========================================================================
    print("\n" + "-"*70)
    print("SQUARE WAVE FREQUENCY SWEEP")
    print("-"*70)
    print("\n  Freq     TX Energy    RX Δk_eff      SNR    Detected")
    print("-"*60)

    for freq in frequencies:
        rx_changes = []
        tx_energies = []

        for _ in range(config.samples_per_frequency):
            tx = FrequencyOssicle(config)
            rx = FrequencyOssicle(config)

            rx.step(config.iterations_per_sample)
            rx_baseline = rx.measure_k_eff()

            tx.inject_square(freq, config.tx_amplitude, 200)
            tx_energy = tx.measure_energy()
            tx_energies.append(tx_energy)

            rx.osc_a += config.substrate_coupling * (tx.osc_a - cp.mean(tx.osc_a))
            rx.step(config.iterations_per_sample)

            rx_changes.append(rx.measure_k_eff() - rx_baseline)

        mean_change = np.mean(rx_changes)
        mean_energy = np.mean(tx_energies)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {freq:5d}    {mean_energy:9.2f}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['square'].append({
            'frequency': freq,
            'tx_energy': mean_energy,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # NOISE BURST SWEEP
    # =========================================================================
    print("\n" + "-"*70)
    print("NOISE BURST FREQUENCY SWEEP")
    print("-"*70)
    print("\n  Freq     TX Energy    RX Δk_eff      SNR    Detected")
    print("-"*60)

    for freq in frequencies:
        rx_changes = []
        tx_energies = []

        for _ in range(config.samples_per_frequency):
            tx = FrequencyOssicle(config)
            rx = FrequencyOssicle(config)

            rx.step(config.iterations_per_sample)
            rx_baseline = rx.measure_k_eff()

            tx.inject_burst(freq, config.tx_amplitude, 200)
            tx_energy = tx.measure_energy()
            tx_energies.append(tx_energy)

            rx.osc_a += config.substrate_coupling * (tx.osc_a - cp.mean(tx.osc_a))
            rx.step(config.iterations_per_sample)

            rx_changes.append(rx.measure_k_eff() - rx_baseline)

        mean_change = np.mean(rx_changes)
        mean_energy = np.mean(tx_energies)
        snr = abs(mean_change) / noise_std if noise_std > 0 else 0
        detected = "YES" if snr > 3 else "no"

        print(f"  {freq:5d}    {mean_energy:9.2f}    {mean_change:+10.6f}    {snr:5.1f}σ    {detected}")

        results['burst'].append({
            'frequency': freq,
            'tx_energy': mean_energy,
            'rx_change': mean_change,
            'snr': snr
        })

    # =========================================================================
    # Summary - Find resonance peaks
    # =========================================================================
    print("\n" + "="*70)
    print("FREQUENCY SWEEP SUMMARY")
    print("="*70)

    for waveform in ['sine', 'square', 'burst']:
        data = results[waveform]
        snrs = [d['snr'] for d in data]
        freqs = [d['frequency'] for d in data]

        best_idx = np.argmax(snrs)
        best_freq = freqs[best_idx]
        best_snr = snrs[best_idx]

        detected_freqs = [d['frequency'] for d in data if d['snr'] > 3]

        print(f"\n  {waveform.upper()}:")
        print(f"    Best frequency: {best_freq} (SNR: {best_snr:.1f}σ)")
        print(f"    Detected at: {detected_freqs if detected_freqs else 'none'}")

    # Check for resonance (peaks in SNR)
    sine_snrs = [d['snr'] for d in results['sine']]
    if len(sine_snrs) > 2:
        # Look for local maxima
        peaks = []
        for i in range(1, len(sine_snrs) - 1):
            if sine_snrs[i] > sine_snrs[i-1] and sine_snrs[i] > sine_snrs[i+1]:
                peaks.append(frequencies[i])

        if peaks:
            print(f"\n  RESONANCE CANDIDATES (sine): {peaks}")
        else:
            print(f"\n  No clear resonance peaks detected")

    print("\n" + "="*70)

    return results


def main():
    """Run frequency sweep experiment."""

    print("="*70)
    print("CIRISARRAY FREQUENCY RESPONSE CHARACTERIZATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nCUDA Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    config = FrequencySweepConfig()
    results = run_frequency_sweep(config)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
