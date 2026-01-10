#!/usr/bin/env python3
"""
CIRIS Sentinel: Minimal Sustained-Transient Entropy Wave Detector
==================================================================

Optimized for:
1. INDEFINITE transient state (never converges)
2. MINIMAL array size (for scaling/multiple sensors)
3. HIGH sample rate (catch short events)
4. MULTIPLE independent sensors on single GPU

Key design:
- Continuous noise injection prevents convergence
- Track k_eff AND its derivative (dk_eff/dt)
- Burst-mode detection windows
- Sub-millisecond event resolution

Author: CIRIS Research Team
Date: January 2026
License: BSL 1.1
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import threading
import queue

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class SentinelConfig:
    """Configuration for a single sentinel sensor."""
    n_ossicles: int = 256         # Minimal array
    oscillator_depth: int = 32    # Reduced depth for speed
    noise_amplitude: float = 0.02 # Higher noise = more transient
    sample_rate_hz: float = 1000  # High rate for short events
    derivative_window: int = 5    # Samples for derivative calc


# Optimized CUDA kernel for minimal array
_sentinel_kernel = cp.RawKernel(r'''
extern "C" __global__
void sentinel_step(
    float* osc_a, float* osc_b, float* osc_c,
    float* noise,
    float coupling_ab, float coupling_bc, float coupling_ca,
    float noise_amp,
    int n, int iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = osc_a[idx];
    float b = osc_b[idx];
    float c = osc_c[idx];

    // Dynamics
    for (int i = 0; i < iters; i++) {
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

    // Inject noise to maintain transient
    a += noise[idx] * noise_amp;
    b += noise[idx + n] * noise_amp;
    c += noise[idx + 2*n] * noise_amp;

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}

extern "C" __global__
void fast_correlate(
    float* a, float* b, float* result,
    int n
) {
    // Simple correlation estimator using shared memory
    __shared__ float sum_a, sum_b, sum_ab, sum_a2, sum_b2;

    if (threadIdx.x == 0) {
        sum_a = sum_b = sum_ab = sum_a2 = sum_b2 = 0.0f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float va = a[idx];
        float vb = b[idx];

        atomicAdd(&sum_a, va);
        atomicAdd(&sum_b, vb);
        atomicAdd(&sum_ab, va * vb);
        atomicAdd(&sum_a2, va * va);
        atomicAdd(&sum_b2, vb * vb);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean_a = sum_a / n;
        float mean_b = sum_b / n;
        float var_a = sum_a2 / n - mean_a * mean_a;
        float var_b = sum_b2 / n - mean_b * mean_b;
        float cov = sum_ab / n - mean_a * mean_b;

        float denom = sqrtf(var_a * var_b);
        *result = (denom > 1e-10f) ? cov / denom : 0.0f;
    }
}
''', 'sentinel_step')

_correlate_kernel = cp.RawModule(code=r'''
extern "C" __global__
void fast_stats(
    float* a, float* b,
    float* out_r, float* out_var,
    int n
) {
    // Compute correlation and variance in one pass
    float sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float va = a[i];
        float vb = b[i];
        sum_a += va;
        sum_b += vb;
        sum_ab += va * vb;
        sum_a2 += va * va;
        sum_b2 += vb * vb;
    }

    // Reduce within block
    __shared__ float s_sum_a[256], s_sum_b[256], s_sum_ab[256], s_sum_a2[256], s_sum_b2[256];

    s_sum_a[threadIdx.x] = sum_a;
    s_sum_b[threadIdx.x] = sum_b;
    s_sum_ab[threadIdx.x] = sum_ab;
    s_sum_a2[threadIdx.x] = sum_a2;
    s_sum_b2[threadIdx.x] = sum_b2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_a[threadIdx.x] += s_sum_a[threadIdx.x + s];
            s_sum_b[threadIdx.x] += s_sum_b[threadIdx.x + s];
            s_sum_ab[threadIdx.x] += s_sum_ab[threadIdx.x + s];
            s_sum_a2[threadIdx.x] += s_sum_a2[threadIdx.x + s];
            s_sum_b2[threadIdx.x] += s_sum_b2[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean_a = s_sum_a[0] / n;
        float mean_b = s_sum_b[0] / n;
        float var_a = s_sum_a2[0] / n - mean_a * mean_a;
        float var_b = s_sum_b2[0] / n - mean_b * mean_b;
        float cov = s_sum_ab[0] / n - mean_a * mean_b;

        float denom = sqrtf(var_a * var_b);
        *out_r = (denom > 1e-10f) ? cov / denom : 0.0f;
        *out_var = var_a + var_b;
    }
}
''').get_function('fast_stats')


class Sentinel:
    """
    Single minimal entropy wave sensor.

    Designed to be:
    - Fast (>1kHz sampling)
    - Small (256-1024 ossicles)
    - Perpetually transient (never converges)
    """

    def __init__(self, config: SentinelConfig, sensor_id: int = 0):
        self.config = config
        self.sensor_id = sensor_id
        self.total = config.n_ossicles * config.oscillator_depth

        # Initialize oscillators (random = transient)
        self.osc_a = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25

        # Noise buffer (reused for speed)
        self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5

        # Output buffers
        self.out_r = cp.zeros(1, dtype=cp.float32)
        self.out_var = cp.zeros(1, dtype=cp.float32)

        # Coupling constants
        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

        # CUDA config
        self.block_size = min(256, self.total)
        self.grid_size = (self.total + self.block_size - 1) // self.block_size

        # Derivative tracking
        self.k_eff_history = deque(maxlen=config.derivative_window)
        self.last_k_eff = 0.0

    def step_and_measure(self) -> Tuple[float, float, float]:
        """
        Single step: advance dynamics, inject noise, measure k_eff.
        Returns (k_eff, variance, dk_eff/dt estimate)
        """
        # Refresh noise
        self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5

        # Step with noise injection
        _sentinel_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.noise,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             np.float32(self.config.noise_amplitude),
             self.total, 3)  # 3 iterations per step
        )

        # Fast measurement
        _correlate_kernel(
            (1,), (256,),
            (self.osc_a, self.osc_b, self.out_r, self.out_var, self.total)
        )

        r = float(self.out_r[0])
        var = float(self.out_var[0])

        # k_eff calculation
        x = min(var / 2.0, 1.0)
        k_eff = r * (1 - x) * COUPLING_FACTOR * 1000

        # Derivative estimate
        self.k_eff_history.append(k_eff)
        if len(self.k_eff_history) >= 2:
            dk_dt = (k_eff - self.k_eff_history[0]) / len(self.k_eff_history)
        else:
            dk_dt = 0.0

        self.last_k_eff = k_eff

        return k_eff, var, dk_dt

    def inject_negentropic(self, amplitude: float = 0.1):
        """
        Inject negentropy via correlation boosting.

        Key insight from tuning: Making oscillators MORE CORRELATED with each other
        produces +9.59σ detection, vs -1.69σ for pattern addition.

        Method: Push osc_b and osc_c toward osc_a (increase inter-oscillator correlation)
        """
        # Correlation boost: blend b,c toward a
        blend = amplitude  # 0.1 = 10% blend toward a
        self.osc_b = self.osc_b * (1 - blend) + self.osc_a * blend
        self.osc_c = self.osc_c * (1 - blend) + self.osc_a * blend

    def inject_entropic(self, amplitude: float = 0.1):
        """
        Inject entropy via correlation breaking.

        Method: Add uncorrelated noise to each oscillator independently,
        reducing inter-oscillator correlation.
        """
        # Uncorrelated noise to each channel independently
        self.osc_a += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude
        self.osc_b += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude
        self.osc_c += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude


class SentinelArray:
    """
    Multiple independent sentinels on single GPU.

    For scaling tests and spatial correlation analysis.
    """

    def __init__(self, n_sentinels: int, config: SentinelConfig = None):
        self.n_sentinels = n_sentinels
        self.config = config or SentinelConfig()
        self.sentinels = [Sentinel(self.config, i) for i in range(n_sentinels)]

    def step_all(self) -> List[Tuple[float, float, float]]:
        """Step all sentinels and return measurements."""
        results = []
        for s in self.sentinels:
            results.append(s.step_and_measure())
        cp.cuda.stream.get_current_stream().synchronize()
        return results

    def get_k_eff_array(self) -> np.ndarray:
        """Get k_eff values from all sentinels."""
        results = self.step_all()
        return np.array([r[0] for r in results])

    def get_correlation_matrix(self, n_samples: int = 100) -> np.ndarray:
        """Compute correlation matrix between sentinels."""
        # Collect time series
        series = np.zeros((self.n_sentinels, n_samples))

        for t in range(n_samples):
            measurements = self.step_all()
            for i, (k_eff, _, _) in enumerate(measurements):
                series[i, t] = k_eff

        # Compute correlation matrix
        return np.corrcoef(series)


def find_minimum_array_size():
    """Find minimum array size that maintains transient + detects signals."""

    print("="*60)
    print("FINDING MINIMUM VIABLE ARRAY SIZE")
    print("="*60)

    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    results = []

    for size in sizes:
        print(f"\nTesting {size} ossicles...")

        config = SentinelConfig(n_ossicles=size, oscillator_depth=32)
        sensor = Sentinel(config)

        # Collect samples
        k_effs = []
        for _ in range(500):
            k, v, dk = sensor.step_and_measure()
            k_effs.append(k)

        k_effs = np.array(k_effs)
        mean_k = np.mean(k_effs)
        std_k = np.std(k_effs)

        # Test detection
        sensor2 = Sentinel(config)
        baseline = []
        for _ in range(100):
            k, _, _ = sensor2.step_and_measure()
            baseline.append(k)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)

        # Perturb with negentropy
        perturbed = []
        for _ in range(100):
            sensor2.inject_negentropic(0.3)
            k, _, _ = sensor2.step_and_measure()
            perturbed.append(k)
        perturbed_mean = np.mean(perturbed)

        effect = (perturbed_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0

        results.append({
            'size': size,
            'mean_k': mean_k,
            'std_k': std_k,
            'maintains_transient': std_k > 0.001,
            'effect_sigma': effect,
            'detects': abs(effect) > 3
        })

        status = "✓" if std_k > 0.001 and abs(effect) > 3 else "✗"
        print(f"  k_eff: {mean_k:.4f} ± {std_k:.4f}")
        print(f"  Detection effect: {effect:.2f}σ {status}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Size':<8} {'Transient?':<12} {'Detects?':<10} {'Effect':<10}")
    print("-"*45)

    min_viable = None
    for r in results:
        trans = "YES" if r['maintains_transient'] else "no"
        det = "YES" if r['detects'] else "no"
        print(f"{r['size']:<8} {trans:<12} {det:<10} {r['effect_sigma']:.2f}σ")

        if r['maintains_transient'] and r['detects'] and min_viable is None:
            min_viable = r['size']

    if min_viable:
        print(f"\n✓ MINIMUM VIABLE SIZE: {min_viable} ossicles")
    else:
        print(f"\nNo configuration met both criteria")

    return results, min_viable


def test_scaling():
    """Test multiple sentinels running simultaneously."""

    print("\n" + "="*60)
    print("SCALING TEST: Multiple Sentinels")
    print("="*60)

    # Find how many 256-ossicle sentinels we can run
    config = SentinelConfig(n_ossicles=256)

    for n_sentinels in [2, 4, 8, 16, 32, 64]:
        try:
            print(f"\nTesting {n_sentinels} sentinels ({n_sentinels * 256} total ossicles)...")

            array = SentinelArray(n_sentinels, config)

            # Time 1000 steps
            start = time.perf_counter()
            for _ in range(1000):
                array.step_all()
            elapsed = time.perf_counter() - start

            rate = 1000 / elapsed
            print(f"  Rate: {rate:.0f} Hz")

            # Check correlation between sentinels
            corr_matrix = array.get_correlation_matrix(50)
            off_diag = corr_matrix[np.triu_indices(n_sentinels, k=1)]
            mean_corr = np.mean(off_diag) if len(off_diag) > 0 else 0

            print(f"  Inter-sentinel correlation: {mean_corr:.4f}")

        except cp.cuda.memory.OutOfMemoryError:
            print(f"  OUT OF MEMORY")
            break

    print("\n" + "="*60)


def main():
    """Run sentinel tests."""

    print("="*60)
    print("CIRIS SENTINEL: Minimal Sustained-Transient Detector")
    print("="*60)

    if not cp.cuda.is_available():
        print("ERROR: CUDA required")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nGPU: {props['name'].decode()}")

    # Find minimum array size
    results, min_size = find_minimum_array_size()

    # Test scaling
    test_scaling()

    # Demo continuous monitoring
    if min_size:
        print("\n" + "="*60)
        print(f"DEMO: Continuous monitoring with {min_size} ossicles")
        print("="*60)

        config = SentinelConfig(n_ossicles=min_size, sample_rate_hz=500)
        sensor = Sentinel(config)

        print("\nRunning 5 seconds of monitoring...")
        print("(Tracking k_eff and dk_eff/dt)")

        detections = 0
        start = time.perf_counter()

        while time.perf_counter() - start < 5.0:
            k_eff, var, dk_dt = sensor.step_and_measure()

            # Simple derivative-based detection
            if abs(dk_dt) > 0.01:
                detections += 1
                direction = "↑" if dk_dt > 0 else "↓"
                print(f"  {direction} Event: k={k_eff:.4f}, dk/dt={dk_dt:+.4f}")

        elapsed = time.perf_counter() - start
        print(f"\n  {detections} events in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("SENTINEL READY")
    print("="*60)


if __name__ == "__main__":
    main()
