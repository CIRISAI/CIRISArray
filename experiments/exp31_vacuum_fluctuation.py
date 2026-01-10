#!/usr/bin/env python3
"""
EXPERIMENT 31: VACUUM FLUCTUATION SENSITIVITY TEST
===================================================

Casimir-Analog Detection via Idle GPU Measurements
---------------------------------------------------

Goal: Determine if CIRISArray can detect vacuum-like fluctuations
      in an idle GPU by measuring the "noise floor" characteristics.

Physics background (Schützhold, Helmholtz-Zentrum):
- Quantum vacuum is never truly empty - filled with fluctuations
- Casimir effect: Vacuum fluctuations create measurable force between plates
- Question: Do vacuum fluctuations leave signatures in GPU entropy?

Hypothesis:
- An idle GPU still has quantum fluctuations in its semiconductor junctions
- These fluctuations should have specific statistical properties:
  - Gaussian distribution (central limit theorem)
  - No temporal correlations beyond thermal timescales
  - Specific spectral density (1/f noise from quantum origin?)
- Deviations from these would indicate non-vacuum sources

Test methodology:
1. Measure array with GPU completely idle (no kernels, no memory ops)
2. Measure array during controlled thermal load
3. Measure array during memory-intensive load
4. Compare noise characteristics:
   - Distribution shape (Gaussian vs heavy-tailed)
   - Autocorrelation decay time
   - Power spectral density slope
   - Spatial correlation structure

What we're establishing:
- Noise floor for quantum detection
- Baseline statistics for anomaly detection
- Sensitivity limits of the ossicle array

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026

References:
- Schützhold, "Eavesdropping on the void", Helmholtz (2023)
- Casimir, "On the attraction between two perfectly conducting plates" (1948)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import stats, signal
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
class NoiseCharacteristics:
    """Statistical characteristics of noise measurements."""
    condition: str
    n_samples: int

    # Distribution
    mean: float
    std: float
    skewness: float
    kurtosis: float
    is_gaussian: bool
    gaussian_p_value: float

    # Temporal
    autocorr_decay_samples: int
    autocorr_decay_us: float

    # Spectral
    spectral_slope: float  # 1/f^alpha, what is alpha?
    dominant_frequency_hz: float

    # Spatial
    spatial_correlation_length_mm: float
    spatial_isotropy: float  # 1.0 = isotropic


@dataclass
class VacuumTestConfig:
    """Configuration for vacuum fluctuation test."""
    n_rows: int = 32
    n_cols: int = 32
    spacing_mm: float = 2.5
    sample_rate_hz: float = 5000.0
    n_samples: int = 10000
    idle_wait_seconds: float = 2.0


class VacuumProbe:
    """
    Ossicle array configured for vacuum fluctuation detection.

    Optimized for measuring the quietest possible signal -
    the noise floor itself.
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void vacuum_probe(
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

        // Very gentle evolution - maximize sensitivity to small perturbations
        float r = 3.569945f;  // Edge of chaos - maximum sensitivity
        float coupling = 0.01f;  // Minimal coupling

        float sum_var = 0.0f;
        float prev_a = a;

        for (int i = 0; i < iterations; i++) {
            float new_a = r * a * (1-a) + coupling * (b - a);
            float new_b = r * b * (1-b) + coupling * (a + c - 2*b);
            float new_c = r * c * (1-c) + coupling * (b - c);

            a = fminf(fmaxf(new_a, 0.001f), 0.999f);
            b = fminf(fmaxf(new_b, 0.001f), 0.999f);
            c = fminf(fmaxf(new_c, 0.001f), 0.999f);

            // Track variance as sensitivity metric
            float delta = a - prev_a;
            sum_var += delta * delta;
            prev_a = a;
        }

        outputs[oid] = sqrtf(sum_var / (float)iterations);

        states[oid * 3 + 0] = a;
        states[oid * 3 + 1] = b;
        states[oid * 3 + 2] = c;
    }
    '''

    def __init__(self, config: VacuumTestConfig):
        self.config = config
        self.n_ossicles = config.n_rows * config.n_cols

        if HAS_CUDA:
            self.module = cp.RawModule(code=self.KERNEL_CODE)
            self.kernel = self.module.get_function('vacuum_probe')
            self.states = cp.random.uniform(0.3, 0.7, (self.n_ossicles, 3), dtype=cp.float32)
            self.outputs = cp.zeros(self.n_ossicles, dtype=cp.float32)
        else:
            self.states = np.random.uniform(0.3, 0.7, (self.n_ossicles, 3)).astype(np.float32)
            self.outputs = np.zeros(self.n_ossicles, dtype=np.float32)

        self._build_geometry()

    def _build_geometry(self):
        """Build spatial geometry for correlation analysis."""
        self.positions = np.zeros((self.n_ossicles, 2))
        idx = 0
        for row in range(self.config.n_rows):
            for col in range(self.config.n_cols):
                self.positions[idx] = [
                    col * self.config.spacing_mm,
                    row * self.config.spacing_mm
                ]
                idx += 1

        # Distance matrix
        self.distances = np.zeros((self.n_ossicles, self.n_ossicles))
        for i in range(self.n_ossicles):
            for j in range(self.n_ossicles):
                self.distances[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )

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
                self.outputs[i] = 0.1 + np.random.randn() * 0.02
            return self.outputs.copy()

    def get_spatial(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(self.config.n_rows, self.config.n_cols)


def generate_thermal_load():
    """Generate thermal load on GPU."""
    if HAS_CUDA:
        # Matrix multiply to heat up GPU
        size = 4096
        a = cp.random.randn(size, size, dtype=cp.float32)
        b = cp.random.randn(size, size, dtype=cp.float32)
        for _ in range(10):
            c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()


def generate_memory_load():
    """Generate memory-intensive load on GPU."""
    if HAS_CUDA:
        # Large memory allocations and copies
        size = 100_000_000  # 100M floats = 400MB
        for _ in range(5):
            a = cp.random.randn(size, dtype=cp.float32)
            b = a.copy()
            del a, b
        cp.cuda.Stream.null.synchronize()


def analyze_noise(data: np.ndarray, condition: str,
                  config: VacuumTestConfig,
                  probe: VacuumProbe) -> NoiseCharacteristics:
    """
    Comprehensive noise analysis.

    Tests for quantum vacuum characteristics:
    - Gaussian distribution (expected for thermal/quantum noise)
    - Temporal correlations (should decay quickly)
    - Spectral density (1/f characteristics)
    - Spatial correlations (should be local)
    """
    n_samples, n_ossicles = data.shape

    # Flatten for distribution analysis
    flat = data.flatten()

    # Distribution statistics
    mean = np.mean(flat)
    std = np.std(flat)
    skewness = stats.skew(flat)
    kurtosis = stats.kurtosis(flat)

    # Gaussian test (Shapiro-Wilk on subsample)
    subsample = np.random.choice(flat, min(5000, len(flat)), replace=False)
    _, gaussian_p = stats.shapiro(subsample)
    is_gaussian = gaussian_p > 0.01

    # Temporal autocorrelation (use center ossicle)
    center_idx = n_ossicles // 2
    center_series = data[:, center_idx]
    center_series = (center_series - np.mean(center_series)) / (np.std(center_series) + 1e-10)

    autocorr = np.correlate(center_series, center_series, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Find decay time (where autocorr drops below 1/e)
    decay_threshold = 1/np.e
    decay_idx = np.where(autocorr < decay_threshold)[0]
    if len(decay_idx) > 0:
        autocorr_decay_samples = decay_idx[0]
    else:
        autocorr_decay_samples = len(autocorr)

    sample_period_us = 1e6 / config.sample_rate_hz
    autocorr_decay_us = autocorr_decay_samples * sample_period_us

    # Power spectral density
    freqs, psd = signal.welch(center_series, fs=config.sample_rate_hz, nperseg=min(1024, n_samples//4))

    # Fit 1/f^alpha in log-log space (exclude DC and high freq)
    valid = (freqs > 1) & (freqs < config.sample_rate_hz / 4)
    if np.sum(valid) > 10:
        log_f = np.log10(freqs[valid])
        log_psd = np.log10(psd[valid] + 1e-20)
        slope, _, _, _, _ = stats.linregress(log_f, log_psd)
        spectral_slope = -slope  # Convention: 1/f^alpha means negative slope
    else:
        spectral_slope = 0.0

    # Dominant frequency
    dominant_idx = np.argmax(psd[1:]) + 1  # Exclude DC
    dominant_frequency_hz = freqs[dominant_idx]

    # Spatial correlation analysis
    # Average correlation vs distance
    mean_snapshot = np.mean(data, axis=0)
    mean_snapshot = (mean_snapshot - np.mean(mean_snapshot)) / (np.std(mean_snapshot) + 1e-10)

    distance_bins = np.linspace(0, probe.distances.max(), 20)
    spatial_corrs = []

    for i in range(len(distance_bins) - 1):
        mask = (probe.distances >= distance_bins[i]) & (probe.distances < distance_bins[i+1])
        if np.sum(mask) > 0:
            corrs = []
            for j in range(n_ossicles):
                for k in range(j+1, n_ossicles):
                    if mask[j, k]:
                        corrs.append(mean_snapshot[j] * mean_snapshot[k])
            if corrs:
                spatial_corrs.append((distance_bins[i], np.mean(corrs)))

    # Find correlation length (where correlation drops below 1/e of max)
    if spatial_corrs:
        distances_arr = np.array([x[0] for x in spatial_corrs])
        corrs_arr = np.array([x[1] for x in spatial_corrs])
        if len(corrs_arr) > 0 and corrs_arr[0] > 0:
            threshold = corrs_arr[0] / np.e
            below = np.where(corrs_arr < threshold)[0]
            if len(below) > 0:
                spatial_correlation_length_mm = distances_arr[below[0]]
            else:
                spatial_correlation_length_mm = distances_arr[-1]
        else:
            spatial_correlation_length_mm = 0.0
    else:
        spatial_correlation_length_mm = 0.0

    # Spatial isotropy (compare horizontal vs vertical correlations)
    h_corrs = []
    v_corrs = []
    for i in range(n_ossicles):
        row_i, col_i = i // config.n_cols, i % config.n_cols
        for j in range(i+1, n_ossicles):
            row_j, col_j = j // config.n_cols, j % config.n_cols
            if row_i == row_j and abs(col_i - col_j) == 1:  # Horizontal neighbor
                h_corrs.append(mean_snapshot[i] * mean_snapshot[j])
            elif col_i == col_j and abs(row_i - row_j) == 1:  # Vertical neighbor
                v_corrs.append(mean_snapshot[i] * mean_snapshot[j])

    if h_corrs and v_corrs:
        spatial_isotropy = min(np.mean(h_corrs), np.mean(v_corrs)) / (max(np.mean(h_corrs), np.mean(v_corrs)) + 1e-10)
    else:
        spatial_isotropy = 1.0

    return NoiseCharacteristics(
        condition=condition,
        n_samples=n_samples,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        is_gaussian=is_gaussian,
        gaussian_p_value=gaussian_p,
        autocorr_decay_samples=autocorr_decay_samples,
        autocorr_decay_us=autocorr_decay_us,
        spectral_slope=spectral_slope,
        dominant_frequency_hz=dominant_frequency_hz,
        spatial_correlation_length_mm=spatial_correlation_length_mm,
        spatial_isotropy=spatial_isotropy
    )


def run_vacuum_fluctuation_test(config: VacuumTestConfig = None) -> Dict:
    """
    Main experiment: Characterize vacuum-like fluctuations.
    """
    if config is None:
        config = VacuumTestConfig()

    print("=" * 70)
    print("EXPERIMENT 31: VACUUM FLUCTUATION SENSITIVITY TEST")
    print("Establishing Noise Floor for Quantum Detection")
    print("=" * 70)
    print()

    print("Goal: Characterize the 'vacuum' (idle GPU) noise floor")
    print("Expected for quantum/thermal noise:")
    print("  - Gaussian distribution")
    print("  - Fast autocorrelation decay")
    print("  - 1/f spectral slope (pink noise)")
    print("  - Short spatial correlation length")
    print()

    probe = VacuumProbe(config)

    print(f"Configuration:")
    print(f"  Array: {config.n_rows} x {config.n_cols} = {probe.n_ossicles} ossicles")
    print(f"  Sample rate: {config.sample_rate_hz} Hz")
    print(f"  Samples: {config.n_samples}")
    print(f"  CUDA available: {HAS_CUDA}")
    print()

    results = {}

    # Condition 1: Idle GPU (closest to "vacuum")
    print("=" * 50)
    print("CONDITION 1: IDLE GPU (Vacuum Analog)")
    print("=" * 50)
    print(f"Waiting {config.idle_wait_seconds}s for GPU to settle...")
    time.sleep(config.idle_wait_seconds)

    print(f"Collecting {config.n_samples} samples...")
    idle_data = np.zeros((config.n_samples, probe.n_ossicles))
    for i in range(config.n_samples):
        idle_data[i] = probe.measure()
        if (i + 1) % 2000 == 0:
            print(f"  Sample {i+1}/{config.n_samples}")

    idle_chars = analyze_noise(idle_data, "idle", config, probe)
    results['idle'] = idle_chars

    print(f"\nIdle GPU characteristics:")
    print(f"  Distribution: mean={idle_chars.mean:.6f}, std={idle_chars.std:.6f}")
    print(f"  Gaussian: {idle_chars.is_gaussian} (p={idle_chars.gaussian_p_value:.4f})")
    print(f"  Skewness: {idle_chars.skewness:.4f}, Kurtosis: {idle_chars.kurtosis:.4f}")
    print(f"  Autocorr decay: {idle_chars.autocorr_decay_us:.1f} μs")
    print(f"  Spectral slope: 1/f^{idle_chars.spectral_slope:.2f}")
    print(f"  Spatial corr length: {idle_chars.spatial_correlation_length_mm:.1f} mm")

    # Condition 2: Thermal load
    print("\n" + "=" * 50)
    print("CONDITION 2: THERMAL LOAD")
    print("=" * 50)
    print("Generating thermal load...")
    generate_thermal_load()

    print(f"Collecting {config.n_samples} samples during thermal load...")
    thermal_data = np.zeros((config.n_samples, probe.n_ossicles))
    for i in range(config.n_samples):
        if i % 500 == 0 and HAS_CUDA:
            generate_thermal_load()  # Maintain thermal load
        thermal_data[i] = probe.measure()
        if (i + 1) % 2000 == 0:
            print(f"  Sample {i+1}/{config.n_samples}")

    thermal_chars = analyze_noise(thermal_data, "thermal", config, probe)
    results['thermal'] = thermal_chars

    print(f"\nThermal load characteristics:")
    print(f"  Distribution: mean={thermal_chars.mean:.6f}, std={thermal_chars.std:.6f}")
    print(f"  Gaussian: {thermal_chars.is_gaussian} (p={thermal_chars.gaussian_p_value:.4f})")
    print(f"  Autocorr decay: {thermal_chars.autocorr_decay_us:.1f} μs")
    print(f"  Spectral slope: 1/f^{thermal_chars.spectral_slope:.2f}")

    # Condition 3: Memory load
    print("\n" + "=" * 50)
    print("CONDITION 3: MEMORY LOAD")
    print("=" * 50)
    print("Generating memory load...")

    print(f"Collecting {config.n_samples} samples during memory load...")
    memory_data = np.zeros((config.n_samples, probe.n_ossicles))
    for i in range(config.n_samples):
        if i % 500 == 0 and HAS_CUDA:
            generate_memory_load()  # Maintain memory load
        memory_data[i] = probe.measure()
        if (i + 1) % 2000 == 0:
            print(f"  Sample {i+1}/{config.n_samples}")

    memory_chars = analyze_noise(memory_data, "memory", config, probe)
    results['memory'] = memory_chars

    print(f"\nMemory load characteristics:")
    print(f"  Distribution: mean={memory_chars.mean:.6f}, std={memory_chars.std:.6f}")
    print(f"  Gaussian: {memory_chars.is_gaussian} (p={memory_chars.gaussian_p_value:.4f})")
    print(f"  Autocorr decay: {memory_chars.autocorr_decay_us:.1f} μs")
    print(f"  Spectral slope: 1/f^{memory_chars.spectral_slope:.2f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("NOISE FLOOR SUMMARY")
    print("=" * 70)

    print("\n{:<15} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        "Condition", "Std Dev", "Gaussian?", "Decay(μs)", "1/f slope", "Corr(mm)"))
    print("-" * 70)

    for name, chars in results.items():
        print("{:<15} {:>10.6f} {:>10} {:>10.1f} {:>12.2f} {:>10.1f}".format(
            name,
            chars.std,
            "Yes" if chars.is_gaussian else "No",
            chars.autocorr_decay_us,
            chars.spectral_slope,
            chars.spatial_correlation_length_mm
        ))

    # Vacuum detection threshold
    print("\n" + "=" * 70)
    print("NOISE FLOOR ESTABLISHED")
    print("=" * 70)

    # Use idle as baseline
    noise_floor_std = idle_chars.std
    detection_threshold_3sigma = 3 * noise_floor_std
    detection_threshold_5sigma = 5 * noise_floor_std

    print(f"\nBaseline (idle GPU):")
    print(f"  Noise floor (1σ): {noise_floor_std:.6f}")
    print(f"  Detection threshold (3σ): {detection_threshold_3sigma:.6f}")
    print(f"  Detection threshold (5σ): {detection_threshold_5sigma:.6f}")

    print(f"\nCharacteristics of 'vacuum' state:")
    print(f"  Distribution: {'Gaussian' if idle_chars.is_gaussian else 'Non-Gaussian'}")
    print(f"  Temporal coherence: {idle_chars.autocorr_decay_us:.1f} μs")
    print(f"  Spatial coherence: {idle_chars.spatial_correlation_length_mm:.1f} mm")
    print(f"  Spectral character: 1/f^{idle_chars.spectral_slope:.2f} " +
          f"({'white' if abs(idle_chars.spectral_slope) < 0.3 else 'pink' if idle_chars.spectral_slope < 1.5 else 'brown'} noise)")

    print("\n" + "=" * 70)
    print("CONCLUSION: Noise floor characterized for quantum detection experiments")
    print("Any signal exceeding 5σ above idle baseline warrants investigation")
    print("=" * 70)

    return {
        'config': config,
        'results': results,
        'noise_floor_std': noise_floor_std,
        'detection_threshold_3sigma': detection_threshold_3sigma,
        'detection_threshold_5sigma': detection_threshold_5sigma
    }


if __name__ == "__main__":
    results = run_vacuum_fluctuation_test()
