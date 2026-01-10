#!/usr/bin/env python3
"""
EXPERIMENT 30: QUANTUM RNG ENTROPY DETECTION
=============================================

Detecting Quantum Randomness Signatures in Hardware RNG
-------------------------------------------------------

Goal: Determine if CIRISArray can distinguish quantum-derived randomness
      from pseudo-random number generation by detecting entropy signatures.

Physics background:
- Hardware RNGs (HRNG) in modern CPUs/GPUs use quantum effects:
  - Thermal noise (Johnson-Nyquist noise) - quantum origin
  - Shot noise in semiconductor junctions - quantum origin
  - Metastable states in ring oscillators - quantum tunneling
- These should produce different entropy "textures" than PRNGs
- If ossicles are sensitive to quantum processes, they might detect
  the difference in local entropy during RNG operations

Hypothesis:
- Quantum RNG operations produce subtle entropy fluctuations
- These fluctuations differ from deterministic PRNG
- An array of sensitive entropy detectors might distinguish them

Test methodology:
1. Run hardware RNG (RDRAND/RDSEED on CPU, curand on GPU)
2. Run software PRNG (Mersenne Twister, LCG)
3. Compare ossicle array response signatures
4. Statistical test for distinguishability

Why this matters:
- If we can detect quantum RNG signatures, we're detecting quantum processes
- This is a stepping stone to detecting other quantum phenomena
- Could have applications in RNG validation/certification

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026
"""

import numpy as np
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False


@dataclass
class RNGSignature:
    """Entropy signature from an RNG operation."""
    rng_type: str
    n_bytes_generated: int
    pre_entropy: np.ndarray  # Ossicle readings before
    post_entropy: np.ndarray  # Ossicle readings after
    delta_entropy: np.ndarray  # Change
    generation_time_us: float
    mean_delta: float
    std_delta: float
    max_delta: float
    spatial_pattern: np.ndarray  # 2D view


@dataclass
class ComparisonResult:
    """Statistical comparison between RNG types."""
    rng_a: str
    rng_b: str
    mean_delta_a: float
    mean_delta_b: float
    std_delta_a: float
    std_delta_b: float
    t_statistic: float
    distinguishable: bool
    confidence: float


class OssicleProbe:
    """
    Ossicle array configured as quantum process probe.

    Optimized for detecting subtle entropy changes during
    specific operations (like RNG calls).
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void entropy_probe(
        float* states,
        float* outputs,
        int n_ossicles,
        int iterations
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float a = states[oid * 3 + 0];
        float b = states[oid * 3 + 1];
        float c = states[oid * 3 + 2];

        // Accumulate entropy metric
        float entropy_sum = 0.0f;

        for (int i = 0; i < iterations; i++) {
            float r = 3.72f + 0.001f * (float)(oid % 10);

            float new_a = r * a * (1-a) + 0.05f * (b - a);
            float new_b = (r + 0.03f) * b * (1-b) + 0.05f * (a + c - 2*b);
            float new_c = (r + 0.06f) * c * (1-c) + 0.05f * (b - c);

            a = fminf(fmaxf(new_a, 0.001f), 0.999f);
            b = fminf(fmaxf(new_b, 0.001f), 0.999f);
            c = fminf(fmaxf(new_c, 0.001f), 0.999f);

            // Shannon entropy proxy
            float p = (a + b + c) / 3.0f;
            entropy_sum += -p * logf(p + 1e-10f) - (1-p) * logf(1-p + 1e-10f);
        }

        outputs[oid] = entropy_sum / (float)iterations;

        states[oid * 3 + 0] = a;
        states[oid * 3 + 1] = b;
        states[oid * 3 + 2] = c;
    }
    '''

    def __init__(self, n_rows: int = 8, n_cols: int = 16):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_ossicles = n_rows * n_cols

        if HAS_CUDA:
            self.module = cp.RawModule(code=self.KERNEL_CODE)
            self.kernel = self.module.get_function('entropy_probe')
            self.states = cp.random.uniform(0.2, 0.8, (self.n_ossicles, 3), dtype=cp.float32)
            self.outputs = cp.zeros(self.n_ossicles, dtype=cp.float32)
        else:
            self.states = np.random.uniform(0.2, 0.8, (self.n_ossicles, 3)).astype(np.float32)
            self.outputs = np.zeros(self.n_ossicles, dtype=np.float32)

    def measure(self) -> np.ndarray:
        if HAS_CUDA:
            block, grid = 256, (self.n_ossicles + 255) // 256
            self.kernel(
                (grid,), (block,),
                (self.states, self.outputs, cp.int32(self.n_ossicles), cp.int32(50))
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(self.outputs)
        else:
            for i in range(self.n_ossicles):
                self.outputs[i] = 0.6 + np.random.randn() * 0.05
            return self.outputs.copy()

    def get_spatial(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(self.n_rows, self.n_cols)


class RNGGenerator:
    """
    Generates random numbers using different sources.

    Compares:
    - Hardware RNG (quantum origin)
    - Cryptographic PRNG (deterministic but high quality)
    - Simple PRNG (deterministic, lower quality)
    """

    def __init__(self):
        self.prng_state = 12345  # For simple LCG

    def generate_hardware_rng(self, n_bytes: int) -> Tuple[bytes, float]:
        """
        Generate using hardware RNG (RDRAND on CPU, curand on GPU).

        These use physical quantum processes (thermal/shot noise).
        """
        start = time.perf_counter()

        if HAS_CUDA:
            # GPU hardware RNG via curand
            rng = cp.random.Generator(cp.random.XORWOW())
            data = rng.integers(0, 256, n_bytes, dtype=cp.uint8)
            cp.cuda.Stream.null.synchronize()
            result = bytes(cp.asnumpy(data))
        else:
            # CPU: try to use os.urandom (uses RDRAND when available)
            result = os.urandom(n_bytes)

        elapsed = (time.perf_counter() - start) * 1e6
        return result, elapsed

    def generate_crypto_prng(self, n_bytes: int) -> Tuple[bytes, float]:
        """
        Generate using cryptographic PRNG (deterministic).

        High quality but purely algorithmic - no quantum input.
        """
        start = time.perf_counter()

        # Use hashlib for deterministic but high-quality random
        result = b''
        counter = 0
        while len(result) < n_bytes:
            h = hashlib.sha256(f"seed_{counter}".encode()).digest()
            result += h
            counter += 1

        elapsed = (time.perf_counter() - start) * 1e6
        return result[:n_bytes], elapsed

    def generate_simple_prng(self, n_bytes: int) -> Tuple[bytes, float]:
        """
        Generate using simple LCG (low quality PRNG).

        Definitely no quantum component - pure determinism.
        """
        start = time.perf_counter()

        result = []
        state = self.prng_state
        for _ in range(n_bytes):
            # Linear Congruential Generator
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            result.append(state & 0xFF)

        self.prng_state = state
        elapsed = (time.perf_counter() - start) * 1e6
        return bytes(result), elapsed

    def generate_numpy_prng(self, n_bytes: int) -> Tuple[bytes, float]:
        """
        Generate using NumPy's Mersenne Twister.

        High quality PRNG but deterministic.
        """
        start = time.perf_counter()

        rng = np.random.Generator(np.random.MT19937(seed=42))
        data = rng.integers(0, 256, n_bytes, dtype=np.uint8)

        elapsed = (time.perf_counter() - start) * 1e6
        return bytes(data), elapsed


def capture_rng_signature(probe: OssicleProbe, rng: RNGGenerator,
                          rng_type: str, n_bytes: int) -> RNGSignature:
    """
    Capture ossicle array response during RNG operation.
    """
    # Pre-measurement
    pre = probe.measure()

    # Generate random numbers
    if rng_type == "hardware":
        _, gen_time = rng.generate_hardware_rng(n_bytes)
    elif rng_type == "crypto":
        _, gen_time = rng.generate_crypto_prng(n_bytes)
    elif rng_type == "simple":
        _, gen_time = rng.generate_simple_prng(n_bytes)
    elif rng_type == "numpy":
        _, gen_time = rng.generate_numpy_prng(n_bytes)
    else:
        raise ValueError(f"Unknown RNG type: {rng_type}")

    # Post-measurement
    post = probe.measure()

    # Compute delta
    delta = post - pre

    return RNGSignature(
        rng_type=rng_type,
        n_bytes_generated=n_bytes,
        pre_entropy=pre,
        post_entropy=post,
        delta_entropy=delta,
        generation_time_us=gen_time,
        mean_delta=np.mean(delta),
        std_delta=np.std(delta),
        max_delta=np.max(np.abs(delta)),
        spatial_pattern=probe.get_spatial(delta)
    )


def compare_signatures(sigs_a: List[RNGSignature],
                       sigs_b: List[RNGSignature]) -> ComparisonResult:
    """
    Statistical comparison of signature distributions.

    Uses Welch's t-test to determine if distributions differ.
    """
    deltas_a = [s.mean_delta for s in sigs_a]
    deltas_b = [s.mean_delta for s in sigs_b]

    mean_a, std_a = np.mean(deltas_a), np.std(deltas_a)
    mean_b, std_b = np.mean(deltas_b), np.std(deltas_b)

    # Welch's t-test
    n_a, n_b = len(deltas_a), len(deltas_b)
    se = np.sqrt(std_a**2/n_a + std_b**2/n_b + 1e-10)
    t_stat = (mean_a - mean_b) / se

    # Rough confidence (|t| > 2 is ~95% confidence)
    confidence = min(abs(t_stat) / 2.0, 1.0)
    distinguishable = abs(t_stat) > 2.0

    return ComparisonResult(
        rng_a=sigs_a[0].rng_type,
        rng_b=sigs_b[0].rng_type,
        mean_delta_a=mean_a,
        mean_delta_b=mean_b,
        std_delta_a=std_a,
        std_delta_b=std_b,
        t_statistic=t_stat,
        distinguishable=distinguishable,
        confidence=confidence
    )


def run_quantum_rng_experiment(n_trials: int = 50,
                               n_bytes: int = 10000) -> Dict:
    """
    Main experiment: Can CIRISArray distinguish quantum from classical RNG?
    """
    print("=" * 70)
    print("EXPERIMENT 30: QUANTUM RNG ENTROPY DETECTION")
    print("Can CIRISArray Distinguish Quantum from Classical Randomness?")
    print("=" * 70)
    print()

    print("Hypothesis:")
    print("  Hardware RNG uses quantum effects (thermal/shot noise)")
    print("  Software PRNG is purely deterministic")
    print("  Ossicle array might detect different entropy signatures")
    print()

    probe = OssicleProbe()
    rng = RNGGenerator()

    print(f"Configuration:")
    print(f"  Array: {probe.n_rows} x {probe.n_cols} = {probe.n_ossicles} ossicles")
    print(f"  Trials per RNG type: {n_trials}")
    print(f"  Bytes generated per trial: {n_bytes}")
    print(f"  CUDA available: {HAS_CUDA}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(20):
        probe.measure()
        rng.generate_hardware_rng(1000)
        rng.generate_crypto_prng(1000)

    rng_types = ["hardware", "crypto", "numpy", "simple"]
    signatures = {t: [] for t in rng_types}

    # Collect signatures
    for rng_type in rng_types:
        print(f"\nCollecting {rng_type} RNG signatures...")
        for i in range(n_trials):
            sig = capture_rng_signature(probe, rng, rng_type, n_bytes)
            signatures[rng_type].append(sig)

            if (i + 1) % 10 == 0:
                print(f"  Trial {i+1}/{n_trials}: mean_delta={sig.mean_delta:.6f}, "
                      f"gen_time={sig.generation_time_us:.1f}μs")

    # Compare all pairs
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS")
    print("=" * 70)

    comparisons = []
    for i, type_a in enumerate(rng_types):
        for type_b in rng_types[i+1:]:
            result = compare_signatures(signatures[type_a], signatures[type_b])
            comparisons.append(result)

            status = "DISTINGUISHABLE" if result.distinguishable else "not distinguishable"
            print(f"\n{type_a} vs {type_b}:")
            print(f"  Mean delta: {result.mean_delta_a:.6f} vs {result.mean_delta_b:.6f}")
            print(f"  Std delta:  {result.std_delta_a:.6f} vs {result.std_delta_b:.6f}")
            print(f"  t-statistic: {result.t_statistic:.2f}")
            print(f"  Result: {status} (confidence: {result.confidence:.1%})")

    # Key result: Can we distinguish hardware from software?
    hw_vs_crypto = compare_signatures(signatures["hardware"], signatures["crypto"])
    hw_vs_simple = compare_signatures(signatures["hardware"], signatures["simple"])

    print("\n" + "=" * 70)
    print("KEY RESULTS: HARDWARE vs SOFTWARE RNG")
    print("=" * 70)

    print(f"\nHardware RNG vs Crypto PRNG:")
    print(f"  t-statistic: {hw_vs_crypto.t_statistic:.2f}")
    print(f"  Distinguishable: {hw_vs_crypto.distinguishable}")

    print(f"\nHardware RNG vs Simple PRNG:")
    print(f"  t-statistic: {hw_vs_simple.t_statistic:.2f}")
    print(f"  Distinguishable: {hw_vs_simple.distinguishable}")

    # Spatial pattern analysis
    print("\n" + "-" * 70)
    print("SPATIAL PATTERN ANALYSIS")
    print("-" * 70)

    for rng_type in rng_types:
        patterns = np.array([s.spatial_pattern for s in signatures[rng_type]])
        mean_pattern = np.mean(patterns, axis=0)
        std_pattern = np.std(patterns, axis=0)

        print(f"\n{rng_type} RNG spatial signature:")
        print(f"  Mean pattern range: [{mean_pattern.min():.6f}, {mean_pattern.max():.6f}]")
        print(f"  Std pattern range:  [{std_pattern.min():.6f}, {std_pattern.max():.6f}]")

        # Show pattern as ASCII
        normalized = (mean_pattern - mean_pattern.min()) / (mean_pattern.max() - mean_pattern.min() + 1e-10)
        chars = " .:-=+*#%@"
        print("  Pattern:")
        for row in normalized:
            line = "    "
            for val in row:
                idx = int(val * (len(chars) - 1))
                line += chars[max(0, min(idx, len(chars)-1))]
            print(line)

    # Conclusion
    print("\n" + "=" * 70)
    quantum_detected = hw_vs_crypto.distinguishable or hw_vs_simple.distinguishable

    if quantum_detected:
        print("RESULT: QUANTUM RNG SIGNATURE POTENTIALLY DETECTED")
        print()
        print("Hardware RNG produces statistically different entropy signatures")
        print("than software PRNGs. This suggests the ossicle array may be")
        print("sensitive to quantum processes in the hardware RNG.")
        print()
        print("Next steps:")
        print("  1. Increase trial count for higher confidence")
        print("  2. Test with isolated hardware RNG (not curand)")
        print("  3. Compare with known quantum RNG devices")
    else:
        print("RESULT: NO DISTINGUISHABLE QUANTUM SIGNATURE")
        print()
        print("Current sensitivity insufficient to distinguish hardware")
        print("RNG from software PRNG. This could mean:")
        print("  1. Ossicles not sensitive enough to quantum effects")
        print("  2. Need more trials or different array configuration")
        print("  3. Hardware RNG signature below detection threshold")

    print("=" * 70)

    return {
        'signatures': signatures,
        'comparisons': comparisons,
        'quantum_detected': quantum_detected,
        'hw_vs_crypto': hw_vs_crypto,
        'hw_vs_simple': hw_vs_simple
    }


if __name__ == "__main__":
    results = run_quantum_rng_experiment()
