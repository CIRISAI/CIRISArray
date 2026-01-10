#!/usr/bin/env python3
"""
Experiment 41: 34 Hz Periodicity Analysis
=========================================

Deep characterization of the ~34 Hz periodic signal detected in exp39.

Questions:
1. What exactly is the frequency? (precision measurement)
2. Is it stable over time?
3. Does it have harmonics?
4. Does it change with GPU load?
5. Is it related to GPU clocks, PDN, or thermal?
6. Does it appear on all arrays equally?

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import subprocess

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class PeriodicityConfig:
    """Configuration for periodicity analysis."""
    n_arrays: int = 8
    ossicles_per_array: int = 2048
    oscillator_depth: int = 64
    warmup_seconds: float = 30.0
    high_rate_seconds: float = 10.0  # High-rate sampling for frequency precision
    sample_rate_hz: float = 500.0    # Nyquist for up to 250 Hz


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


# Compute load kernel
load_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_load(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = data[idx];
    for (int i = 0; i < iterations; i++) {
        x = sinf(x) * cosf(x * 1.1f) + 0.001f;
    }
    data[idx] = x;
}
''', 'compute_load')


class HighSpeedSensor:
    """High-speed sensor for frequency analysis."""

    def __init__(self, config: PeriodicityConfig):
        self.config = config
        self.total_elements = config.n_arrays * config.ossicles_per_array * config.oscillator_depth

        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 5):
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.config.oscillator_depth, self.total_elements, iterations)
        )

    def measure_fast(self) -> float:
        """Fast measurement using subset."""
        sample_size = 20000
        indices = cp.arange(0, self.total_elements, self.total_elements // sample_size)[:sample_size]

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r = float(cp.corrcoef(a, b)[0, 1])
        r = 0 if np.isnan(r) else r

        var = float(cp.var(a) + cp.var(b)) / 2
        x = min(var, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


def get_gpu_info() -> Dict:
    """Get GPU clock and temperature info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_clock_mhz': int(parts[0]),
                'mem_clock_mhz': int(parts[1]),
                'temp_c': int(parts[2]),
                'power_w': float(parts[3])
            }
    except:
        pass
    return {}


def run_high_rate_capture(config: PeriodicityConfig) -> Tuple[np.ndarray, float]:
    """Capture high-rate k_eff time series."""

    sensor = HighSpeedSensor(config)

    # Warmup
    print(f"  Warming up for {config.warmup_seconds}s...")
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < config.warmup_seconds:
        sensor.step(20)

    # High-rate capture
    n_samples = int(config.high_rate_seconds * config.sample_rate_hz)
    print(f"  Capturing {n_samples} samples at {config.sample_rate_hz} Hz...")

    samples = np.zeros(n_samples)
    start_time = time.perf_counter()

    for i in range(n_samples):
        sensor.step(2)  # Minimal stepping for speed
        samples[i] = sensor.measure_fast()

    actual_duration = time.perf_counter() - start_time
    actual_rate = n_samples / actual_duration

    print(f"  Captured at {actual_rate:.1f} Hz actual")

    return samples, actual_rate


def analyze_spectrum(samples: np.ndarray, sample_rate: float) -> Dict:
    """Detailed spectral analysis."""

    n = len(samples)

    # Remove DC and linear trend
    detrended = samples - np.linspace(samples[0], samples[-1], n)
    detrended = detrended - np.mean(detrended)

    # FFT
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    power = np.abs(fft[:n//2])**2
    freqs = freqs[:n//2]

    # Find peaks
    total_power = np.sum(power[1:])  # Exclude DC

    # Sort by power
    peak_indices = np.argsort(power[1:])[::-1] + 1  # Skip DC
    top_peaks = []

    for idx in peak_indices[:20]:  # Top 20 peaks
        if power[idx] > total_power * 0.001:  # > 0.1% of total
            top_peaks.append({
                'freq': freqs[idx],
                'power': power[idx],
                'pct': 100 * power[idx] / total_power
            })

    return {
        'freqs': freqs,
        'power': power,
        'total_power': total_power,
        'peaks': top_peaks
    }


def find_fundamental_and_harmonics(peaks: List[Dict]) -> Dict:
    """Identify fundamental frequency and its harmonics."""

    if not peaks:
        return {'fundamental': None, 'harmonics': []}

    # Sort by frequency
    sorted_peaks = sorted(peaks, key=lambda x: x['freq'])

    # Try to find fundamental (lowest significant frequency)
    for peak in sorted_peaks:
        if peak['freq'] > 1.0:  # Ignore sub-Hz
            fundamental = peak['freq']

            # Find harmonics
            harmonics = []
            for other in sorted_peaks:
                ratio = other['freq'] / fundamental
                # Check if it's close to an integer multiple
                if abs(ratio - round(ratio)) < 0.1 and round(ratio) > 1:
                    harmonics.append({
                        'freq': other['freq'],
                        'harmonic': int(round(ratio)),
                        'power_pct': other['pct']
                    })

            return {
                'fundamental': fundamental,
                'fundamental_power_pct': peak['pct'],
                'harmonics': harmonics
            }

    return {'fundamental': None, 'harmonics': []}


def run_load_comparison(config: PeriodicityConfig) -> Dict:
    """Compare periodicity under different GPU loads."""

    results = {}

    conditions = [
        ('idle', 0),
        ('light', 100),
        ('medium', 1000),
        ('heavy', 5000)
    ]

    for condition_name, load_iterations in conditions:
        print(f"\n  Testing under {condition_name} load...")

        sensor = HighSpeedSensor(config)

        # Warmup
        for _ in range(500):
            sensor.step(20)

        # If not idle, create background load
        if load_iterations > 0:
            load_data = cp.random.random(5000000, dtype=cp.float32)

        # Capture
        n_samples = 2000
        samples = np.zeros(n_samples)
        gpu_clocks = []

        start_time = time.perf_counter()

        for i in range(n_samples):
            # Apply load
            if load_iterations > 0:
                block_size = 256
                grid_size = (5000000 + block_size - 1) // block_size
                load_kernel((grid_size,), (block_size,),
                           (load_data, 5000000, load_iterations))

            sensor.step(2)
            samples[i] = sensor.measure_fast()

            if i % 500 == 0:
                gpu_info = get_gpu_info()
                if gpu_info:
                    gpu_clocks.append(gpu_info.get('gpu_clock_mhz', 0))

        actual_duration = time.perf_counter() - start_time
        actual_rate = n_samples / actual_duration

        # Analyze
        spectrum = analyze_spectrum(samples, actual_rate)
        harmonics = find_fundamental_and_harmonics(spectrum['peaks'][:10])

        results[condition_name] = {
            'sample_rate': actual_rate,
            'fundamental': harmonics['fundamental'],
            'mean_gpu_clock': np.mean(gpu_clocks) if gpu_clocks else None,
            'peaks': spectrum['peaks'][:5]
        }

        if load_iterations > 0:
            del load_data
            cp.get_default_memory_pool().free_all_blocks()

    return results


def run_periodicity_analysis(config: PeriodicityConfig) -> Dict:
    """Full periodicity analysis."""

    print("\n" + "="*70)
    print("EXPERIMENT 41: 34 Hz PERIODICITY ANALYSIS")
    print("="*70)

    results = {}

    # Get initial GPU state
    gpu_info = get_gpu_info()
    print(f"\n  GPU State:")
    if gpu_info:
        print(f"    Clock: {gpu_info.get('gpu_clock_mhz', 'N/A')} MHz")
        print(f"    Temp: {gpu_info.get('temp_c', 'N/A')} °C")
        print(f"    Power: {gpu_info.get('power_w', 'N/A')} W")

    results['initial_gpu'] = gpu_info

    # =========================================================================
    # HIGH-RATE CAPTURE
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 1: High-Rate Frequency Measurement")
    print("-"*70)

    samples, actual_rate = run_high_rate_capture(config)
    results['sample_rate'] = actual_rate
    results['samples'] = samples

    # Spectral analysis
    spectrum = analyze_spectrum(samples, actual_rate)
    results['spectrum'] = spectrum

    print(f"\n  Top frequencies detected:")
    print(f"  {'Freq (Hz)':<12} {'Power %':<10} {'Notes'}")
    print("-"*50)

    for i, peak in enumerate(spectrum['peaks'][:10]):
        notes = ""
        if 30 <= peak['freq'] <= 40:
            notes = "← CANDIDATE (near 34 Hz)"
        elif peak['freq'] < 1:
            notes = "(sub-Hz drift)"

        print(f"  {peak['freq']:<12.3f} {peak['pct']:<10.1f} {notes}")

    # Find fundamental and harmonics
    harmonics = find_fundamental_and_harmonics(spectrum['peaks'][:15])
    results['harmonics'] = harmonics

    if harmonics['fundamental']:
        print(f"\n  FUNDAMENTAL FREQUENCY: {harmonics['fundamental']:.3f} Hz")
        print(f"  Power: {harmonics['fundamental_power_pct']:.1f}% of total")

        if harmonics['harmonics']:
            print(f"\n  Harmonics detected:")
            for h in harmonics['harmonics']:
                print(f"    {h['harmonic']}× = {h['freq']:.3f} Hz ({h['power_pct']:.1f}%)")
    else:
        print(f"\n  No clear fundamental frequency identified")

    # =========================================================================
    # LOAD COMPARISON
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 2: Load Dependence")
    print("-"*70)

    load_results = run_load_comparison(config)
    results['load_comparison'] = load_results

    print(f"\n  {'Condition':<12} {'Fundamental':<15} {'GPU Clock':<12}")
    print("-"*50)

    for condition, data in load_results.items():
        fund = f"{data['fundamental']:.2f} Hz" if data['fundamental'] else "N/A"
        clock = f"{data['mean_gpu_clock']:.0f} MHz" if data['mean_gpu_clock'] else "N/A"
        print(f"  {condition:<12} {fund:<15} {clock:<12}")

    # Check if frequency changes with clock
    freqs = [d['fundamental'] for d in load_results.values() if d['fundamental']]
    if len(freqs) > 1:
        freq_range = max(freqs) - min(freqs)
        print(f"\n  Frequency variation: {freq_range:.2f} Hz")
        if freq_range < 2:
            print("  → Frequency is STABLE across loads")
        else:
            print("  → Frequency VARIES with load")

    # =========================================================================
    # STABILITY OVER TIME
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 3: Temporal Stability")
    print("-"*70)

    print("  Recording 5 consecutive captures...")

    stability_freqs = []
    for capture in range(5):
        sensor = HighSpeedSensor(config)
        for _ in range(300):
            sensor.step(20)

        cap_samples = np.zeros(1000)
        start = time.perf_counter()
        for i in range(1000):
            sensor.step(2)
            cap_samples[i] = sensor.measure_fast()
        cap_rate = 1000 / (time.perf_counter() - start)

        cap_spectrum = analyze_spectrum(cap_samples, cap_rate)
        cap_harmonics = find_fundamental_and_harmonics(cap_spectrum['peaks'][:10])

        if cap_harmonics['fundamental']:
            stability_freqs.append(cap_harmonics['fundamental'])
            print(f"    Capture {capture + 1}: {cap_harmonics['fundamental']:.3f} Hz")
        else:
            print(f"    Capture {capture + 1}: No clear fundamental")

    if stability_freqs:
        mean_freq = np.mean(stability_freqs)
        std_freq = np.std(stability_freqs)
        print(f"\n  Stability: {mean_freq:.3f} ± {std_freq:.3f} Hz")
        results['stability'] = {'mean': mean_freq, 'std': std_freq}

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PERIODICITY ANALYSIS SUMMARY")
    print("="*70)

    if harmonics['fundamental']:
        print(f"\n  DETECTED FREQUENCY: {harmonics['fundamental']:.3f} Hz")

        # Try to identify source
        fund = harmonics['fundamental']
        print(f"\n  Possible sources:")

        if 29 <= fund <= 35:
            print(f"    • GPU boost clock adjustment cycle (~30-33 Hz typical)")
        if 58 <= fund <= 62:
            print(f"    • Display refresh rate (60 Hz)")
        if 48 <= fund <= 52:
            print(f"    • Power line (50 Hz regions)")
        if 58 <= fund <= 62:
            print(f"    • Power line (60 Hz regions)")

        # Check relationship to GPU clock
        if gpu_info and gpu_info.get('gpu_clock_mhz'):
            gpu_clock = gpu_info['gpu_clock_mhz']
            ratio = gpu_clock / fund
            if abs(ratio - round(ratio)) < 0.5:
                print(f"    • GPU clock / {int(round(ratio))} = {gpu_clock/round(ratio):.1f} Hz")

    else:
        print(f"\n  No dominant periodic signal found")

    print("\n" + "="*70)

    return results


def main():
    """Run periodicity analysis."""

    print("="*70)
    print("CIRISARRAY 34 Hz PERIODICITY CHARACTERIZATION")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = PeriodicityConfig(
        warmup_seconds=20.0,
        high_rate_seconds=10.0,
        sample_rate_hz=500.0
    )

    results = run_periodicity_analysis(config)

    return results


if __name__ == "__main__":
    main()
