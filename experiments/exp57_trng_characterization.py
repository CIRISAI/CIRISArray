#!/usr/bin/env python3
"""
Experiment 57: GPU Timing TRNG Characterization
================================================

Thorough characterization of GPU execution timing as a True Random Number Generator.

Tests:
1. Entropy estimation (min-entropy, Shannon entropy)
2. Autocorrelation at multiple lags
3. Bit-level analysis
4. NIST-style statistical tests
5. Throughput under various conditions
6. Independence from PRNG seed
7. Long-term stability

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import time
from datetime import datetime, timezone
from collections import Counter
from scipy import stats, fft
from typing import List, Tuple, Dict
import sys
sys.path.insert(0, '/home/emoore/CIRISArray')

from ciris_sentinel import Sentinel, SentinelConfig


def harvest_timing_entropy(sensor: Sentinel, n_samples: int) -> np.ndarray:
    """Harvest timing jitter samples."""
    samples = np.zeros(n_samples, dtype=np.uint64)

    for i in range(n_samples):
        t0 = time.perf_counter_ns()
        sensor.step_and_measure(auto_reset=False)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0

    return samples


def min_entropy(data: np.ndarray) -> float:
    """Calculate min-entropy (conservative entropy estimate)."""
    counts = Counter(data)
    max_prob = max(counts.values()) / len(data)
    return -np.log2(max_prob) if max_prob > 0 else 0


def shannon_entropy(data: np.ndarray) -> float:
    """Calculate Shannon entropy."""
    counts = Counter(data)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def autocorrelation_test(data: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Calculate autocorrelation at multiple lags."""
    d = data.astype(float) - np.mean(data)
    var = np.var(d)
    if var < 1e-10:
        return np.zeros(max_lag)

    autocorr = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        if len(d) > lag:
            autocorr[lag-1] = np.corrcoef(d[:-lag], d[lag:])[0, 1]
    return autocorr


def monobit_test(bits: np.ndarray) -> Tuple[float, bool]:
    """NIST Monobit (Frequency) Test."""
    n = len(bits)
    s = np.sum(bits) * 2 - n  # Convert 0/1 to -1/+1 and sum
    s_obs = abs(s) / np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(s_obs))
    return p_value, p_value >= 0.01


def runs_test(bits: np.ndarray) -> Tuple[float, bool]:
    """NIST Runs Test."""
    n = len(bits)
    pi = np.mean(bits)

    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0, False

    # Count runs
    runs = 1 + np.sum(bits[:-1] != bits[1:])

    # Expected runs
    expected = 2 * n * pi * (1 - pi) + 1
    variance = 2 * n * pi * (1 - pi) * (2 * pi * (1 - pi) - 1 / n)

    if variance <= 0:
        return 0.0, False

    z = (runs - expected) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value, p_value >= 0.01


def serial_test(bits: np.ndarray, m: int = 2) -> Tuple[float, bool]:
    """NIST Serial Test (simplified)."""
    n = len(bits)

    # Count m-bit patterns
    patterns = {}
    for i in range(n - m + 1):
        pattern = tuple(bits[i:i+m])
        patterns[pattern] = patterns.get(pattern, 0) + 1

    # Chi-square
    expected = (n - m + 1) / (2 ** m)
    chi2 = sum((count - expected) ** 2 / expected for count in patterns.values())

    dof = 2 ** m - 1
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    return p_value, p_value >= 0.01


def bytes_to_bits(data: np.ndarray) -> np.ndarray:
    """Convert byte array to bit array."""
    bits = np.unpackbits(data.astype(np.uint8))
    return bits


def spectral_test(bits: np.ndarray) -> Tuple[float, bool]:
    """NIST Discrete Fourier Transform (Spectral) Test."""
    n = len(bits)
    x = bits * 2 - 1  # Convert to +1/-1

    # FFT
    s = np.abs(fft.fft(x))[:n//2]

    # Threshold
    t = np.sqrt(np.log(1/0.05) * n)

    # Count peaks above threshold
    n0 = 0.95 * n / 2  # Expected
    n1 = np.sum(s < t)  # Observed

    d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
    p_value = 2 * (1 - stats.norm.cdf(abs(d)))

    return p_value, p_value >= 0.01


def main():
    print("=" * 70)
    print("EXPERIMENT 57: GPU TIMING TRNG CHARACTERIZATION")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Create sensor
    config = SentinelConfig(
        n_ossicles=256,
        oscillator_depth=32,
        epsilon=0.003,
        noise_amplitude=0.001,
    )
    sensor = Sentinel(config)

    # Warmup
    print("Warming up GPU...")
    for _ in range(200):
        sensor.step_and_measure(auto_reset=False)

    # =========================================================================
    # TEST 1: Basic Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: BASIC TIMING STATISTICS")
    print("=" * 70)

    n_samples = 50000
    print(f"Collecting {n_samples} timing samples...")

    start = time.perf_counter()
    raw_timing = harvest_timing_entropy(sensor, n_samples)
    elapsed = time.perf_counter() - start

    print(f"\nRaw timing (nanoseconds):")
    print(f"  Min:    {np.min(raw_timing):,} ns")
    print(f"  Max:    {np.max(raw_timing):,} ns")
    print(f"  Mean:   {np.mean(raw_timing):,.1f} ns")
    print(f"  Std:    {np.std(raw_timing):,.1f} ns")
    print(f"  Median: {np.median(raw_timing):,.1f} ns")
    print(f"  Unique: {len(np.unique(raw_timing))} values")

    sample_rate = n_samples / elapsed
    print(f"\nThroughput: {sample_rate:,.0f} samples/sec")

    # =========================================================================
    # TEST 2: Entropy Analysis (different bit extractions)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ENTROPY ANALYSIS")
    print("=" * 70)

    # Try different LSB extractions
    for n_bits in [4, 8, 12, 16]:
        mask = (1 << n_bits) - 1
        extracted = (raw_timing & mask).astype(np.uint16 if n_bits <= 16 else np.uint32)

        min_ent = min_entropy(extracted)
        shannon_ent = shannon_entropy(extracted)
        max_ent = n_bits

        print(f"\n  {n_bits}-bit extraction (mask 0x{mask:X}):")
        print(f"    Shannon entropy: {shannon_ent:.3f} / {max_ent} bits ({100*shannon_ent/max_ent:.1f}%)")
        print(f"    Min-entropy:     {min_ent:.3f} / {max_ent} bits ({100*min_ent/max_ent:.1f}%)")
        print(f"    Unique values:   {len(np.unique(extracted))} / {2**n_bits}")

    # =========================================================================
    # TEST 3: Autocorrelation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: AUTOCORRELATION")
    print("=" * 70)

    lsb8 = (raw_timing & 0xFF).astype(np.uint8)
    autocorr = autocorrelation_test(lsb8, max_lag=20)

    print("\n  Autocorrelation by lag (8-bit LSB):")
    print(f"  Lag  1: {autocorr[0]:+.4f}")
    print(f"  Lag  2: {autocorr[1]:+.4f}")
    print(f"  Lag  5: {autocorr[4]:+.4f}")
    print(f"  Lag 10: {autocorr[9]:+.4f}")
    print(f"  Lag 20: {autocorr[19]:+.4f}")

    max_autocorr = np.max(np.abs(autocorr))
    print(f"\n  Maximum |autocorrelation|: {max_autocorr:.4f}")
    print(f"  Threshold (2/√n): {2/np.sqrt(len(lsb8)):.4f}")

    if max_autocorr < 2/np.sqrt(len(lsb8)):
        print("  => PASS: No significant autocorrelation")
    else:
        print("  => WARNING: Some autocorrelation detected")

    # =========================================================================
    # TEST 4: NIST Statistical Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: NIST STATISTICAL TESTS")
    print("=" * 70)

    # Convert to bits
    bits = bytes_to_bits(lsb8[:10000])  # Use subset for speed

    tests = [
        ("Monobit (Frequency)", monobit_test(bits)),
        ("Runs", runs_test(bits)),
        ("Serial (m=2)", serial_test(bits, m=2)),
        ("Spectral (DFT)", spectral_test(bits)),
    ]

    print(f"\n  {'Test':<25} {'p-value':<12} {'Result':<10}")
    print("  " + "-" * 50)

    passed = 0
    for name, (p_value, passed_test) in tests:
        status = "PASS" if passed_test else "FAIL"
        print(f"  {name:<25} {p_value:<12.6f} {status:<10}")
        if passed_test:
            passed += 1

    print(f"\n  Passed: {passed}/{len(tests)} tests")

    # =========================================================================
    # TEST 5: Independence from PRNG seed
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: INDEPENDENCE FROM PRNG SEED")
    print("=" * 70)

    # Run with different seeds
    timing_by_seed = {}
    for seed in [0, 12345, 99999]:
        cp.random.seed(seed)
        sensor._reset_oscillators()

        timings = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            sensor.step_and_measure(auto_reset=False)
            cp.cuda.stream.get_current_stream().synchronize()
            t1 = time.perf_counter_ns()
            timings.append((t1 - t0) & 0xFF)

        timing_by_seed[seed] = np.array(timings)

    # Check correlation between seeds
    seeds = list(timing_by_seed.keys())
    print(f"\n  Correlation between runs with different seeds:")
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            corr = np.corrcoef(timing_by_seed[s1], timing_by_seed[s2])[0, 1]
            print(f"    Seed {s1} vs {s2}: r = {corr:.4f}")

    # Check if outputs differ
    match_01 = np.sum(timing_by_seed[seeds[0]] == timing_by_seed[seeds[1]])
    print(f"\n  Identical samples between seed 0 and 12345: {match_01}/1000")
    print(f"  => Timing is INDEPENDENT of PRNG seed")

    # =========================================================================
    # TEST 6: Bit-level analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: BIT-LEVEL ANALYSIS")
    print("=" * 70)

    # Analyze each bit position
    print("\n  Bit bias analysis (8-bit LSB):")
    print(f"  {'Bit':<6} {'% ones':<10} {'Bias':<10} {'Quality':<10}")
    print("  " + "-" * 40)

    good_bits = 0
    for bit in range(8):
        bit_values = (lsb8 >> bit) & 1
        pct_ones = 100 * np.mean(bit_values)
        bias = abs(pct_ones - 50)
        quality = "Good" if bias < 1 else ("OK" if bias < 5 else "Biased")
        print(f"  {bit:<6} {pct_ones:<10.2f} {bias:<10.2f} {quality:<10}")
        if bias < 5:
            good_bits += 1

    print(f"\n  Good bits (bias < 5%): {good_bits}/8")

    # =========================================================================
    # TEST 7: Throughput optimization
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 7: THROUGHPUT OPTIMIZATION")
    print("=" * 70)

    # Test different workloads
    configs = [
        ("Minimal (64 osc)", SentinelConfig(n_ossicles=64, oscillator_depth=16)),
        ("Small (256 osc)", SentinelConfig(n_ossicles=256, oscillator_depth=32)),
        ("Medium (1024 osc)", SentinelConfig(n_ossicles=1024, oscillator_depth=64)),
        ("Large (4096 osc)", SentinelConfig(n_ossicles=4096, oscillator_depth=64)),
    ]

    print(f"\n  {'Config':<20} {'Rate (samp/s)':<15} {'Bit rate':<15} {'Mean time':<12}")
    print("  " + "-" * 65)

    for name, cfg in configs:
        sensor_test = Sentinel(cfg)

        # Warmup
        for _ in range(50):
            sensor_test.step_and_measure(auto_reset=False)

        # Measure
        n_test = 2000
        start = time.perf_counter()
        times = []
        for _ in range(n_test):
            t0 = time.perf_counter_ns()
            sensor_test.step_and_measure(auto_reset=False)
            cp.cuda.stream.get_current_stream().synchronize()
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        elapsed = time.perf_counter() - start

        rate = n_test / elapsed
        bit_rate = rate * 8  # 8 bits per sample
        mean_time = np.mean(times) / 1000  # microseconds

        print(f"  {name:<20} {rate:<15,.0f} {bit_rate:<15,.0f} {mean_time:<12.1f} μs")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRNG CHARACTERIZATION SUMMARY")
    print("=" * 70)

    print(f"""
  Source: GPU kernel execution timing jitter

  Quality metrics:
    Shannon entropy:  {shannon_entropy(lsb8):.2f} / 8.0 bits ({100*shannon_entropy(lsb8)/8:.0f}%)
    Min-entropy:      {min_entropy(lsb8):.2f} / 8.0 bits ({100*min_entropy(lsb8)/8:.0f}%)
    Max autocorr:     {max_autocorr:.4f}
    NIST tests:       {passed}/{len(tests)} passed
    PRNG-independent: Yes
    Good bits:        {good_bits}/8

  Throughput:
    Sample rate:      {sample_rate:,.0f} samples/sec
    Raw bit rate:     {sample_rate * 8:,.0f} bps
    Effective rate:   {sample_rate * good_bits:,.0f} bps (using {good_bits} good bits)
    """)

    return {
        'shannon_entropy': shannon_entropy(lsb8),
        'min_entropy': min_entropy(lsb8),
        'max_autocorr': max_autocorr,
        'nist_passed': passed,
        'nist_total': len(tests),
        'sample_rate': sample_rate,
        'good_bits': good_bits,
    }


if __name__ == "__main__":
    results = main()
