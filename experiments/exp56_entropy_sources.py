#!/usr/bin/env python3
"""
Experiment 56: Entropy Source Analysis
======================================

Identify and measure all entropy sources available from the oscillator array.

Potential sources:
1. Oscillator state LSBs - mantissa bits of osc_a, osc_b, osc_c
2. k_eff LSBs - measurement noise in correlation calculation
3. Timing jitter - GPU execution time variations
4. Correlation residuals - r_ab, r_bc, r_ca fluctuations
5. Variance fluctuations - total variance changes
6. Raw cupy RNG - baseline comparison

For each source, measure:
- Bit rate (bits/second)
- Entropy per bit (should be ~1.0 for true random)
- Autocorrelation (should be ~0 for good randomness)
- Chi-square uniformity

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import time
from datetime import datetime, timezone
from collections import Counter
from scipy import stats
from typing import Tuple
import sys
sys.path.insert(0, '/home/emoore/CIRISArray')

from ciris_sentinel import Sentinel, SentinelConfig


def extract_lsbs(arr: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """Extract LSBs from float array by viewing as int32."""
    int_view = arr.view(np.int32)
    mask = (1 << n_bits) - 1
    return (int_view & mask).astype(np.uint8)


def measure_entropy(data: np.ndarray) -> float:
    """Measure Shannon entropy in bits per symbol."""
    counts = Counter(data.flatten())
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def measure_autocorrelation(data: np.ndarray, lag: int = 1) -> float:
    """Measure autocorrelation at given lag."""
    if len(data) < lag + 10:
        return 0.0
    d = data.astype(float) - np.mean(data)
    if np.std(d) < 1e-10:
        return 0.0
    return np.corrcoef(d[:-lag], d[lag:])[0, 1]


def chi_square_uniformity(data: np.ndarray, n_bins: int = 256) -> Tuple:
    """Chi-square test for uniformity."""
    hist, _ = np.histogram(data, bins=n_bins, range=(0, 256))
    expected = len(data) / n_bins
    chi2 = np.sum((hist - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, n_bins - 1)
    return chi2, p_value


def test_entropy_source(name: str, sample_func, n_samples: int = 10000,
                        bits_per_sample: int = 8) -> dict:
    """Test a single entropy source."""
    print(f"\n  Testing {name}...")

    samples = []
    times = []

    start = time.perf_counter()
    for i in range(n_samples):
        t0 = time.perf_counter()
        sample = sample_func()
        t1 = time.perf_counter()
        samples.append(sample)
        times.append(t1 - t0)

        if i % 1000 == 0 and i > 0:
            print(f"    {i}/{n_samples}...")

    total_time = time.perf_counter() - start

    samples = np.array(samples).flatten()

    # Metrics
    entropy = measure_entropy(samples)
    max_entropy = np.log2(256) if bits_per_sample == 8 else bits_per_sample
    entropy_ratio = entropy / max_entropy

    autocorr_1 = measure_autocorrelation(samples, 1)
    autocorr_10 = measure_autocorrelation(samples, 10)

    chi2, p_value = chi_square_uniformity(samples)

    sample_rate = n_samples / total_time
    bit_rate = sample_rate * bits_per_sample
    effective_bit_rate = bit_rate * entropy_ratio

    mean_time = np.mean(times) * 1e6  # microseconds

    return {
        'name': name,
        'n_samples': n_samples,
        'total_time_s': total_time,
        'sample_rate_hz': sample_rate,
        'bits_per_sample': bits_per_sample,
        'raw_bit_rate_bps': bit_rate,
        'entropy_per_bit': entropy_ratio,
        'effective_bit_rate_bps': effective_bit_rate,
        'autocorr_lag1': autocorr_1,
        'autocorr_lag10': autocorr_10,
        'chi2': chi2,
        'chi2_p_value': p_value,
        'mean_sample_time_us': mean_time,
        'min_value': int(np.min(samples)),
        'max_value': int(np.max(samples)),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 56: ENTROPY SOURCE ANALYSIS")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Create optimized sentinel
    config = SentinelConfig(
        n_ossicles=256,
        oscillator_depth=32,
        epsilon=0.003,
        noise_amplitude=0.001,
    )
    sensor = Sentinel(config)

    print(f"Configuration:")
    print(f"  Oscillators: {config.n_ossicles} x {config.oscillator_depth} = {sensor.total}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Noise: {config.noise_amplitude}")

    # Warmup
    print(f"\nWarming up...")
    for _ in range(100):
        sensor.step_and_measure(auto_reset=False)

    results = []

    # =========================================================================
    # SOURCE 1: Oscillator state LSBs
    # =========================================================================
    def sample_osc_lsb():
        sensor.step_and_measure(auto_reset=False)
        # Get 1 byte from each of a, b, c
        a_lsb = extract_lsbs(cp.asnumpy(sensor.osc_a[:1]), 8)[0]
        return a_lsb

    results.append(test_entropy_source(
        "Oscillator LSB (8-bit)", sample_osc_lsb, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 2: k_eff LSBs
    # =========================================================================
    def sample_k_eff_lsb():
        k_eff, _, _, _ = sensor.step_and_measure(auto_reset=False)
        # Extract LSBs from k_eff float
        k_bytes = np.array([k_eff], dtype=np.float32).view(np.uint8)
        return k_bytes[0]  # Least significant byte of mantissa

    results.append(test_entropy_source(
        "k_eff LSB (8-bit)", sample_k_eff_lsb, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 3: Timing jitter
    # =========================================================================
    def sample_timing():
        t0 = time.perf_counter_ns()
        sensor.step_and_measure(auto_reset=False)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        return (t1 - t0) & 0xFF  # LSB of timing in nanoseconds

    results.append(test_entropy_source(
        "Timing jitter (8-bit)", sample_timing, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 4: r_ab residual
    # =========================================================================
    last_r_ab = [0.0]
    def sample_r_ab_delta():
        sensor.step_and_measure(auto_reset=False)
        r_ab, _, _ = sensor.get_internal_correlations()
        delta = r_ab - last_r_ab[0]
        last_r_ab[0] = r_ab
        # Convert small delta to byte
        scaled = int((delta + 0.01) * 12800) & 0xFF
        return scaled

    results.append(test_entropy_source(
        "r_ab delta (8-bit)", sample_r_ab_delta, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 5: Variance delta
    # =========================================================================
    last_var = [0.0]
    def sample_variance_delta():
        state = sensor.step_and_measure_full(auto_reset=False)
        delta = state['variance'] - last_var[0]
        last_var[0] = state['variance']
        scaled = int((delta + 0.001) * 128000) & 0xFF
        return scaled

    results.append(test_entropy_source(
        "Variance delta (8-bit)", sample_variance_delta, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 6: Multiple oscillator XOR
    # =========================================================================
    def sample_multi_xor():
        sensor.step_and_measure(auto_reset=False)
        # XOR LSBs from multiple oscillators
        a_lsb = extract_lsbs(cp.asnumpy(sensor.osc_a[:8]), 8)
        xored = a_lsb[0]
        for i in range(1, 8):
            xored ^= a_lsb[i]
        return xored

    results.append(test_entropy_source(
        "Multi-oscillator XOR (8-bit)", sample_multi_xor, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 7: Raw cupy RNG (baseline)
    # =========================================================================
    def sample_cupy_rng():
        return int(cp.random.randint(0, 256).get())

    results.append(test_entropy_source(
        "CuPy RNG baseline", sample_cupy_rng, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SOURCE 8: Combined entropy pool
    # =========================================================================
    def sample_combined():
        k_eff, _, _, _ = sensor.step_and_measure(auto_reset=False)

        # Combine multiple sources via XOR
        a_lsb = extract_lsbs(cp.asnumpy(sensor.osc_a[:1]), 8)[0]
        b_lsb = extract_lsbs(cp.asnumpy(sensor.osc_b[:1]), 8)[0]

        k_lsb = np.array([k_eff], dtype=np.float32).view(np.uint8)[0]

        timing = time.perf_counter_ns() & 0xFF

        return a_lsb ^ b_lsb ^ k_lsb ^ timing

    results.append(test_entropy_source(
        "Combined pool (XOR)", sample_combined, n_samples=5000, bits_per_sample=8
    ))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENTROPY SOURCE SUMMARY")
    print("=" * 70)

    print(f"\n{'Source':<30} {'Rate (bps)':<12} {'Ent/bit':<10} {'Eff bps':<12} {'AC(1)':<8} {'χ² p':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<30} {r['raw_bit_rate_bps']:<12.0f} "
              f"{r['entropy_per_bit']:<10.3f} {r['effective_bit_rate_bps']:<12.0f} "
              f"{r['autocorr_lag1']:<8.3f} {r['chi2_p_value']:<8.3f}")

    # Best source
    best = max(results, key=lambda x: x['effective_bit_rate_bps'] * (1 - abs(x['autocorr_lag1'])))

    print(f"\n{'=' * 70}")
    print(f"BEST SOURCE: {best['name']}")
    print(f"{'=' * 70}")
    print(f"  Effective bit rate: {best['effective_bit_rate_bps']:.0f} bps")
    print(f"  Entropy per bit: {best['entropy_per_bit']:.3f}")
    print(f"  Autocorrelation: {best['autocorr_lag1']:.4f}")
    print(f"  Chi-square p-value: {best['chi2_p_value']:.4f}")
    print(f"  Sample time: {best['mean_sample_time_us']:.1f} μs")

    # Recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    good_sources = [r for r in results
                    if r['entropy_per_bit'] > 0.9
                    and abs(r['autocorr_lag1']) < 0.1
                    and r['chi2_p_value'] > 0.01]

    if good_sources:
        print("\nSources suitable for entropy harvesting:")
        for r in sorted(good_sources, key=lambda x: -x['effective_bit_rate_bps']):
            print(f"  - {r['name']}: {r['effective_bit_rate_bps']:.0f} effective bps")
    else:
        print("\nNo sources meet all quality criteria (entropy>0.9, autocorr<0.1, p>0.01)")
        print("Best candidates:")
        for r in sorted(results, key=lambda x: -x['entropy_per_bit'])[:3]:
            print(f"  - {r['name']}: ent={r['entropy_per_bit']:.3f}, ac={r['autocorr_lag1']:.3f}")

    return results


if __name__ == "__main__":
    results = main()
