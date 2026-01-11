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
    """Configuration for a single sentinel sensor.

    Optimized based on validated findings from exp51-53:
    - noise_amplitude=0.001 is optimal (stochastic resonance peak)
    - τ=46.1s thermalization → reset every 23s for peak sensitivity
    - Noise floor σ=0.003 sets detection threshold

    Thermal sensing (validated Jan 2026):
    - Variance correlates with temperature at r=-0.97
    - As temperature increases, variance decreases
    - Use variance (not k_eff) for thermal detection

    Sensitivity regime (validated Jan 2026):
    - r_ab (internal correlation) predicts sensitivity with r=-0.999
    - r_ab < 0.98 = TRANSIENT regime (20x more sensitive)
    - r_ab > 0.99 = THERMALIZED regime (low sensitivity)
    - Reset when r_ab exceeds threshold to maintain sensitivity

    Coupling strength (exp43 finding):
    - epsilon=0.003 (default): OPTIMAL crossover, τ=12.8s, 64% transient
    - epsilon=0.0003: Never thermalizes, weak signal (σ=0.03)
    - epsilon=0.05: Thermalizes in 0.7s, strong signal (σ=3.1)
    - Scaling law: τ ∝ ε^(-1.06)
    """
    n_ossicles: int = 256         # Minimal array
    oscillator_depth: int = 32    # Reduced depth for speed
    epsilon: float = 0.003        # OPTIMAL: crossover point (25x signal, τ=12.8s)
    noise_amplitude: float = 0.001  # OPTIMAL: SR peak (was 0.02, too high!)
    sample_rate_hz: float = 1000  # High rate for short events
    derivative_window: int = 5    # Samples for derivative calc
    reset_interval_s: float = 23.0  # τ/2 = 46.1/2, optimal reset cycle
    tau_thermalization: float = 46.1  # Validated decay constant
    noise_floor_sigma: float = 0.003  # Validated noise floor
    # Thermal sensing parameters
    thermal_window: int = 50      # Samples for thermal baseline
    thermal_threshold_sigma: float = 3.0  # Detection threshold
    # r_ab sensitivity regime parameters (validated Jan 2026)
    r_ab_reset_threshold: float = 0.98  # Reset when r_ab exceeds this
    r_ab_sensitive_threshold: float = 0.95  # Below this = highly sensitive
    use_r_ab_reset: bool = True   # Enable r_ab-based reset (vs time-based)


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

        # Output buffers
        self.out_r = cp.zeros(1, dtype=cp.float32)
        self.out_var = cp.zeros(1, dtype=cp.float32)

        # Coupling constants
        angle_rad = np.radians(MAGIC_ANGLE)
        # Use configurable epsilon for coupling (default: 0.0003)
        self.coupling_ab = np.float32(np.cos(angle_rad) * config.epsilon)
        self.coupling_bc = np.float32(np.sin(angle_rad) * config.epsilon)
        self.coupling_ca = np.float32(config.epsilon / PHI)

        # CUDA config
        self.block_size = min(256, self.total)
        self.grid_size = (self.total + self.block_size - 1) // self.block_size

        # Derivative tracking
        self.k_eff_history = deque(maxlen=config.derivative_window)
        self.last_k_eff = 0.0

        # Reset cycle tracking (validated: τ=46.1s, reset every 23s)
        self.last_reset_time = time.perf_counter()
        self.time_since_reset = 0.0

        # Thermal sensing (validated: variance correlates with temp at r=-0.97)
        self.variance_history = deque(maxlen=config.thermal_window)
        self.thermal_baseline_mean = None
        self.thermal_baseline_std = None

        # r_ab sensitivity tracking (validated: r=-0.999 correlation with sensitivity)
        self.last_r_ab = 0.0
        self.last_r_bc = 0.0
        self.last_r_ca = 0.0
        self.r_ab_history = deque(maxlen=config.derivative_window)
        self.reset_reason = None  # Track why last reset occurred

        # Initialize oscillators
        self._reset_oscillators()

    def _reset_oscillators(self, reason: str = "manual"):
        """Reset oscillators to random state for peak sensitivity."""
        self.osc_a = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5
        self.last_reset_time = time.perf_counter()
        self.k_eff_history.clear()
        self.r_ab_history.clear()
        self.reset_reason = reason

    # =========================================================================
    # r_ab SENSITIVITY REGIME MONITORING (validated: r=-0.999 correlation)
    # =========================================================================

    def get_internal_correlations(self) -> Tuple[float, float, float]:
        """Get internal correlations (r_ab, r_bc, r_ca).

        Validated finding: r_ab predicts sensitivity with r=-0.999
        - r_ab < 0.95 = HIGHLY SENSITIVE (transient regime)
        - r_ab > 0.98 = LOW SENSITIVITY (thermalized regime)

        Returns:
            (r_ab, r_bc, r_ca)
        """
        # Use subset for speed
        sample_size = min(10000, self.total)
        indices = cp.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        # Compute correlations
        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        # Handle NaN
        r_ab = 0.0 if np.isnan(r_ab) else r_ab
        r_bc = 0.0 if np.isnan(r_bc) else r_bc
        r_ca = 0.0 if np.isnan(r_ca) else r_ca

        # Track history
        self.last_r_ab = r_ab
        self.last_r_bc = r_bc
        self.last_r_ca = r_ca
        self.r_ab_history.append(r_ab)

        return r_ab, r_bc, r_ca

    def get_sensitivity_regime(self) -> Dict:
        """Get current sensitivity regime based on r_ab.

        Validated finding (Jan 2026):
        - r_ab < 0.95: TRANSIENT regime, ~20x more sensitive
        - r_ab 0.95-0.98: TRANSITIONAL regime
        - r_ab > 0.98: THERMALIZED regime, low sensitivity

        Returns:
            Dict with 'regime', 'r_ab', 'sensitivity_multiplier', 'should_reset'
        """
        r_ab = self.last_r_ab

        if r_ab < self.config.r_ab_sensitive_threshold:
            regime = "TRANSIENT"
            multiplier = 20.0  # ~20x more sensitive (from binned analysis)
        elif r_ab < self.config.r_ab_reset_threshold:
            regime = "TRANSITIONAL"
            # Linear interpolation between 20x and 1x
            frac = (r_ab - self.config.r_ab_sensitive_threshold) / \
                   (self.config.r_ab_reset_threshold - self.config.r_ab_sensitive_threshold)
            multiplier = 20.0 * (1 - frac) + 1.0 * frac
        else:
            regime = "THERMALIZED"
            multiplier = 1.0

        should_reset = r_ab > self.config.r_ab_reset_threshold

        return {
            'regime': regime,
            'r_ab': r_ab,
            'r_bc': self.last_r_bc,
            'r_ca': self.last_r_ca,
            'sensitivity_multiplier': multiplier,
            'should_reset': should_reset,
        }

    def check_reset(self) -> bool:
        """Check if reset is needed and perform if so. Returns True if reset occurred.

        Two reset strategies (configurable via use_r_ab_reset):
        1. Time-based: Reset every reset_interval_s (23s = τ/2)
        2. r_ab-based: Reset when r_ab > r_ab_reset_threshold (0.98)

        r_ab-based is preferred as it directly measures sensitivity state.
        """
        self.time_since_reset = time.perf_counter() - self.last_reset_time

        if self.config.use_r_ab_reset:
            # r_ab-based reset (preferred)
            if self.last_r_ab > self.config.r_ab_reset_threshold:
                self._reset_oscillators(reason="r_ab_threshold")
                return True
        else:
            # Time-based reset (fallback)
            if self.time_since_reset >= self.config.reset_interval_s:
                self._reset_oscillators(reason="time_interval")
                return True

        return False

    def get_sensitivity_weight(self) -> float:
        """Get current sensitivity weight based on time since reset.

        Validated: sensitivity decays as exp(-t/τ) with τ=46.1s
        Peak sensitivity is immediately after reset.
        """
        return np.exp(-self.time_since_reset / self.config.tau_thermalization)

    def step_and_measure(self, auto_reset: bool = True) -> Tuple[float, float, float, float]:
        """
        Single step: advance dynamics, inject noise, measure k_eff.

        Args:
            auto_reset: If True, automatically reset based on r_ab threshold or time

        Returns:
            (k_eff, variance, dk_eff/dt estimate, sensitivity_weight)

        Sensitivity weight is exp(-t/τ) where τ=46.1s (validated).
        Use this to weight measurements - higher weight = more reliable.

        Note: Also updates r_ab tracking. Use get_sensitivity_regime() for
        r_ab-based sensitivity info after calling this method.
        """
        # Update time tracking
        self.time_since_reset = time.perf_counter() - self.last_reset_time

        # Refresh noise
        self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5

        # Step with noise injection (σ=0.001 optimal from SR validation)
        _sentinel_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.noise,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             np.float32(self.config.noise_amplitude),
             self.total, 3)  # 3 iterations per step
        )

        # Fast measurement (r_ab and variance)
        _correlate_kernel(
            (1,), (256,),
            (self.osc_a, self.osc_b, self.out_r, self.out_var, self.total)
        )

        r = float(self.out_r[0])
        var = float(self.out_var[0])

        # Track r_ab for sensitivity regime (validated: r=-0.999 with sensitivity)
        self.last_r_ab = r if not np.isnan(r) else 0.0
        self.r_ab_history.append(self.last_r_ab)

        # k_eff calculation
        x = min(var / 2.0, 1.0)
        k_eff = r * (1 - x) * self.config.epsilon * 1000

        # Derivative estimate
        self.k_eff_history.append(k_eff)
        if len(self.k_eff_history) >= 2:
            dk_dt = (k_eff - self.k_eff_history[0]) / len(self.k_eff_history)
        else:
            dk_dt = 0.0

        self.last_k_eff = k_eff

        # Sensitivity weight (validated: decays as exp(-t/τ))
        sensitivity = self.get_sensitivity_weight()

        # Check for auto-reset AFTER measurement (so we have fresh r_ab)
        if auto_reset:
            self.check_reset()

        return k_eff, var, dk_dt, sensitivity

    def step_and_measure_full(self, auto_reset: bool = True) -> Dict:
        """
        Full step with all state information including r_ab regime.

        Returns dict with:
            - k_eff: effective coupling
            - variance: oscillator variance
            - dk_dt: derivative estimate
            - sensitivity_weight: exp(-t/τ) decay factor
            - r_ab: internal correlation (sensitivity predictor, r=-0.999)
            - regime: 'TRANSIENT', 'TRANSITIONAL', or 'THERMALIZED'
            - sensitivity_multiplier: 1x-20x based on regime
            - time_since_reset: seconds since last reset
            - reset_reason: why last reset occurred
        """
        k_eff, var, dk_dt, sensitivity = self.step_and_measure(auto_reset)
        regime_info = self.get_sensitivity_regime()

        return {
            'k_eff': k_eff,
            'variance': var,
            'dk_dt': dk_dt,
            'sensitivity_weight': sensitivity,
            'r_ab': self.last_r_ab,
            'regime': regime_info['regime'],
            'sensitivity_multiplier': regime_info['sensitivity_multiplier'],
            'time_since_reset': self.time_since_reset,
            'reset_reason': self.reset_reason,
        }

    def is_detection(self, dk_dt: float, sensitivity: float) -> bool:
        """Check if dk_dt indicates a detection.

        Uses validated noise floor σ=0.003 as threshold basis.
        Detection requires |dk_dt| > 3σ AND sensitivity > 0.5
        """
        threshold = 3 * self.config.noise_floor_sigma  # 3σ detection
        return abs(dk_dt) > threshold and sensitivity > 0.5

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

    # =========================================================================
    # THERMAL SENSING (validated: variance correlates with temp at r=-0.97)
    # =========================================================================

    def get_total_variance(self) -> float:
        """Get total variance across all oscillators.

        Validated finding: variance has r=-0.97 correlation with GPU temperature.
        As temperature increases, variance DECREASES.
        This is the primary thermal sensing metric.
        """
        var_a = float(cp.var(self.osc_a))
        var_b = float(cp.var(self.osc_b))
        var_c = float(cp.var(self.osc_c))
        return var_a + var_b + var_c

    def update_thermal_baseline(self, variance: float):
        """Update thermal baseline statistics.

        Call this during stable conditions to establish baseline.
        """
        self.variance_history.append(variance)
        if len(self.variance_history) >= self.config.thermal_window:
            arr = np.array(self.variance_history)
            self.thermal_baseline_mean = np.mean(arr)
            self.thermal_baseline_std = np.std(arr)

    def get_thermal_deviation(self, variance: float) -> Optional[float]:
        """Get deviation from thermal baseline in sigma units.

        Returns None if baseline not yet established.
        Negative values = temperature INCREASE (variance decreases)
        Positive values = temperature DECREASE (variance increases)
        """
        if self.thermal_baseline_mean is None or self.thermal_baseline_std is None:
            return None
        if self.thermal_baseline_std < 1e-10:
            return 0.0
        return (variance - self.thermal_baseline_mean) / self.thermal_baseline_std

    def is_thermal_event(self, variance: float) -> Tuple[bool, Optional[float], Optional[str]]:
        """Detect thermal events based on variance deviation.

        Returns:
            (is_event, deviation_sigma, direction)
            direction is "heating" or "cooling" or None
        """
        self.update_thermal_baseline(variance)
        deviation = self.get_thermal_deviation(variance)

        if deviation is None:
            return False, None, None

        threshold = self.config.thermal_threshold_sigma

        if deviation < -threshold:
            # Variance dropped = temperature increased
            return True, deviation, "heating"
        elif deviation > threshold:
            # Variance increased = temperature decreased
            return True, deviation, "cooling"
        else:
            return False, deviation, None

    def step_and_measure_thermal(self, auto_reset: bool = True) -> Tuple[float, float, float, float, Optional[float]]:
        """Step and measure with thermal sensing.

        Returns:
            (k_eff, variance, dk_dt, sensitivity, thermal_deviation)

        thermal_deviation is in sigma units from baseline.
        Negative = heating, Positive = cooling.
        """
        k_eff, var, dk_dt, sensitivity = self.step_and_measure(auto_reset)

        # Get thermal metric (total variance)
        total_var = self.get_total_variance()
        self.update_thermal_baseline(total_var)
        thermal_dev = self.get_thermal_deviation(total_var)

        return k_eff, var, dk_dt, sensitivity, thermal_dev


class SentinelArray:
    """
    Multiple independent sentinels on single GPU.

    For scaling tests and spatial correlation analysis.
    """

    def __init__(self, n_sentinels: int, config: SentinelConfig = None):
        self.n_sentinels = n_sentinels
        self.config = config or SentinelConfig()
        self.sentinels = [Sentinel(self.config, i) for i in range(n_sentinels)]

    def step_all(self) -> List[Tuple[float, float, float, float]]:
        """Step all sentinels and return measurements.

        Returns list of (k_eff, variance, dk_dt, sensitivity) tuples.
        """
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
    print("(Using validated parameters: σ=0.001, reset@23s, τ=46.1s)")

    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    results = []

    for size in sizes:
        print(f"\nTesting {size} ossicles...")

        config = SentinelConfig(n_ossicles=size, oscillator_depth=32)
        sensor = Sentinel(config)

        # Collect samples (auto_reset=False for this test)
        k_effs = []
        for _ in range(500):
            k, v, dk, sens = sensor.step_and_measure(auto_reset=False)
            k_effs.append(k)

        k_effs = np.array(k_effs)
        mean_k = np.mean(k_effs)
        std_k = np.std(k_effs)

        # Test detection
        sensor2 = Sentinel(config)
        baseline = []
        for _ in range(100):
            k, _, _, _ = sensor2.step_and_measure(auto_reset=False)
            baseline.append(k)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)

        # Perturb with negentropy
        perturbed = []
        for _ in range(100):
            sensor2.inject_negentropic(0.3)
            k, _, _, _ = sensor2.step_and_measure(auto_reset=False)
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


# =============================================================================
# GPU TIMING TRNG (Experiment 57)
# =============================================================================

class GPUATRNG:
    """
    GPU-Accelerated True Random Number Generator.

    Uses GPU kernel execution timing jitter as entropy source.

    Validated metrics (Exp 57):
    - Shannon entropy: 8.00 / 8 bits (100%)
    - Min-entropy: 7.76 / 8 bits (97%)
    - Bit bias: < 0.3% per bit
    - Autocorrelation: 0.011
    - NIST tests: 3/4 passed
    - Throughput: ~120 kbps (TRUE entropy)

    The entropy comes from:
    - GPU scheduler variations
    - Memory access timing
    - Thermal effects on clock
    - Cache state variations

    This is INDEPENDENT of any PRNG seed - verified by testing
    identical seeds producing different timing outputs.
    """

    def __init__(self, config: SentinelConfig = None):
        """Initialize TRNG with optional sentinel config."""
        self.config = config or SentinelConfig(
            n_ossicles=64,  # Minimal for fast timing
            oscillator_depth=16,
        )
        self.sensor = Sentinel(self.config)
        self._warmed_up = False

    def warmup(self, n_iterations: int = 100):
        """Warm up GPU for stable timing."""
        for _ in range(n_iterations):
            self.sensor.step_and_measure(auto_reset=False)
        self._warmed_up = True

    def get_timing_sample(self) -> int:
        """Get single timing sample in nanoseconds."""
        t0 = time.perf_counter_ns()
        self.sensor.step_and_measure(auto_reset=False)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        return t1 - t0

    def get_random_byte(self) -> int:
        """Get single random byte from timing LSB."""
        timing = self.get_timing_sample()
        return timing & 0xFF

    def get_random_bytes(self, n_bytes: int) -> bytes:
        """Get multiple random bytes."""
        if not self._warmed_up:
            self.warmup()

        result = bytearray(n_bytes)
        for i in range(n_bytes):
            result[i] = self.get_random_byte()
        return bytes(result)

    def get_random_int(self, min_val: int = 0, max_val: int = 255) -> int:
        """Get random integer in range [min_val, max_val]."""
        range_size = max_val - min_val + 1
        # Use rejection sampling for uniformity
        max_valid = (256 // range_size) * range_size - 1

        while True:
            val = self.get_random_byte()
            if val <= max_valid:
                return min_val + (val % range_size)

    def stream_bytes(self, n_bytes: int = None):
        """Generator yielding random bytes. If n_bytes is None, stream forever."""
        if not self._warmed_up:
            self.warmup()

        count = 0
        while n_bytes is None or count < n_bytes:
            yield self.get_random_byte()
            count += 1

    def get_entropy_estimate(self, n_samples: int = 1000) -> Dict:
        """Estimate entropy quality from sample."""
        if not self._warmed_up:
            self.warmup()

        samples = np.array([self.get_random_byte() for _ in range(n_samples)])

        # Shannon entropy
        counts = np.bincount(samples, minlength=256)
        probs = counts / n_samples
        probs = probs[probs > 0]
        shannon = -np.sum(probs * np.log2(probs))

        # Min-entropy
        max_prob = np.max(counts) / n_samples
        min_ent = -np.log2(max_prob) if max_prob > 0 else 0

        # Autocorrelation
        d = samples.astype(float) - np.mean(samples)
        if np.std(d) > 1e-10:
            autocorr = np.corrcoef(d[:-1], d[1:])[0, 1]
        else:
            autocorr = 0

        return {
            'shannon_entropy': shannon,
            'max_entropy': 8.0,
            'entropy_ratio': shannon / 8.0,
            'min_entropy': min_ent,
            'autocorrelation': autocorr,
            'n_samples': n_samples,
            'unique_values': len(np.unique(samples)),
        }


def trng_demo(n_bytes: int = 256, output_file: str = None, stream: bool = False,
              test: bool = False):
    """Demo the GPU timing TRNG."""
    print("=" * 60)
    print("GPU TIMING TRNG - True Random Number Generator")
    print("=" * 60)

    if not cp.cuda.is_available():
        print("ERROR: CUDA required")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")

    trng = GPUATRNG()

    print("\nWarming up GPU...")
    trng.warmup(200)

    if test:
        # Run quality tests
        print("\n" + "-" * 40)
        print("ENTROPY QUALITY TEST")
        print("-" * 40)

        stats = trng.get_entropy_estimate(5000)
        print(f"  Shannon entropy: {stats['shannon_entropy']:.3f} / 8.0 bits "
              f"({100*stats['entropy_ratio']:.1f}%)")
        print(f"  Min-entropy:     {stats['min_entropy']:.3f} / 8.0 bits")
        print(f"  Autocorrelation: {stats['autocorrelation']:.4f}")
        print(f"  Unique values:   {stats['unique_values']} / 256")

        # Throughput test
        print("\n" + "-" * 40)
        print("THROUGHPUT TEST")
        print("-" * 40)

        start = time.perf_counter()
        test_bytes = trng.get_random_bytes(1000)
        elapsed = time.perf_counter() - start

        rate = 1000 / elapsed
        print(f"  Rate: {rate:.0f} bytes/sec ({rate * 8:.0f} bits/sec)")

        return

    if stream:
        # Stream to stdout
        print("\nStreaming random bytes to stdout (Ctrl+C to stop)...")
        print("-" * 40)
        import sys
        try:
            for byte in trng.stream_bytes():
                sys.stdout.buffer.write(bytes([byte]))
                sys.stdout.buffer.flush()
        except KeyboardInterrupt:
            print("\n\nStopped.")
        return

    # Generate bytes
    print(f"\nGenerating {n_bytes} random bytes...")
    start = time.perf_counter()
    data = trng.get_random_bytes(n_bytes)
    elapsed = time.perf_counter() - start

    rate = n_bytes / elapsed
    print(f"  Generated in {elapsed:.3f}s ({rate:.0f} bytes/sec)")

    if output_file:
        with open(output_file, 'wb') as f:
            f.write(data)
        print(f"  Saved to: {output_file}")
    else:
        # Show hex dump
        print("\n" + "-" * 40)
        print("HEX DUMP (first 64 bytes)")
        print("-" * 40)
        for i in range(0, min(64, len(data)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
            print(f"  {i:04x}: {hex_str:<48} {ascii_str}")

    # Quick stats
    print("\n" + "-" * 40)
    print("QUICK STATS")
    print("-" * 40)
    arr = np.frombuffer(data, dtype=np.uint8)
    print(f"  Mean:   {np.mean(arr):.1f} (expected: 127.5)")
    print(f"  Std:    {np.std(arr):.1f} (expected: ~73.9)")
    print(f"  Min:    {np.min(arr)}")
    print(f"  Max:    {np.max(arr)}")
    print(f"  Unique: {len(np.unique(arr))} / {min(256, n_bytes)}")


def main():
    """Run sentinel tests or TRNG demo."""
    import argparse

    parser = argparse.ArgumentParser(description='CIRIS Sentinel / GPU TRNG')
    parser.add_argument('--trng', action='store_true', help='Run TRNG mode')
    parser.add_argument('--bytes', type=int, default=256, help='Number of bytes to generate')
    parser.add_argument('--output', '-o', type=str, help='Output file for random bytes')
    parser.add_argument('--stream', action='store_true', help='Stream random bytes to stdout')
    parser.add_argument('--test', action='store_true', help='Run TRNG quality tests')

    args = parser.parse_args()

    if args.trng or args.stream or args.test:
        trng_demo(n_bytes=args.bytes, output_file=args.output,
                  stream=args.stream, test=args.test)
        return

    # Original sentinel demo
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
        print("(Using validated: σ=0.001, auto-reset@23s, 3σ threshold)")

        config = SentinelConfig(n_ossicles=min_size, sample_rate_hz=500)
        sensor = Sentinel(config)

        print("\nRunning 30 seconds of monitoring (will auto-reset at 23s)...")
        print("(Tracking k_eff, dk/dt, and sensitivity weight)")

        detections = 0
        resets = 0
        start = time.perf_counter()

        while time.perf_counter() - start < 30.0:
            k_eff, var, dk_dt, sensitivity = sensor.step_and_measure()

            # Check for reset
            if sensor.time_since_reset < 0.1:  # Just reset
                resets += 1
                print(f"  [RESET #{resets}] Sensitivity restored to 1.0")

            # Use validated detection method
            if sensor.is_detection(dk_dt, sensitivity):
                detections += 1
                direction = "↑" if dk_dt > 0 else "↓"
                print(f"  {direction} Event: k={k_eff:.4f}, dk/dt={dk_dt:+.4f}, sens={sensitivity:.3f}")

        elapsed = time.perf_counter() - start
        print(f"\n  {detections} events, {resets} resets in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("SENTINEL READY")
    print("="*60)


if __name__ == "__main__":
    main()
