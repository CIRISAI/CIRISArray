#!/usr/bin/env python3
"""
Experiment 53: CIRISArray Validation Protocol
==============================================

Rigorous validation of CIRISArray findings through:
- PHASE 1: Null hypothesis tests (prove it's noise)
- PHASE 2: Characterization (measure everything)
- PHASE 3: Validation (test predictions)

If any Phase 1 test passes, stop - it's noise.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
import subprocess
import json
from datetime import datetime, timezone
from scipy import signal, optimize, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class ExperimentResult:
    """Standard result format for all experiments."""
    experiment_id: str
    timestamp: str
    git_commit: str
    gpu_temp: Optional[float]
    k_eff_raw: np.ndarray
    tau: float
    tau_err: float
    f_peak: float
    p_peak: float
    spectrum_freqs: np.ndarray
    spectrum_power: np.ndarray
    parameters: Dict
    conclusion: str


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:8]
    except:
        return "unknown"


def get_gpu_temp() -> Optional[float]:
    """Get GPU temperature if available."""
    if HAS_CUDA:
        try:
            # This is NVIDIA-specific
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            return float(result.stdout.strip().split('\n')[0])
        except:
            pass
    return None


class ValidationSensor:
    """Configurable sensor for validation experiments."""

    def __init__(self, n_ossicles: int, depth: int = 64,
                 coupling: float = COUPLING_FACTOR,
                 use_gpu: bool = True,
                 deterministic: bool = False):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth
        self.use_gpu = use_gpu and HAS_CUDA
        self.deterministic = deterministic

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * coupling)
        self.coupling_bc = float(np.sin(angle_rad) * coupling)
        self.coupling_ca = float(coupling / PHI)

        self.reset()

    def reset(self, seed: Optional[int] = None):
        xp = cp if self.use_gpu else np

        if seed is not None:
            if self.use_gpu:
                cp.random.seed(seed)
            np.random.seed(seed)

        if self.deterministic:
            # Fixed initialization for deterministic mode
            self.osc_a = xp.linspace(-0.25, 0.25, self.total).astype(xp.float32)
            self.osc_b = xp.linspace(0.25, -0.25, self.total).astype(xp.float32)
            self.osc_c = xp.zeros(self.total, dtype=xp.float32)
        else:
            self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
            self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
            self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

    def step(self, noise: float = 0.001, iterations: int = 5):
        xp = cp if self.use_gpu else np

        for _ in range(iterations):
            if noise > 0 and not self.deterministic:
                self.osc_a += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_b += xp.random.normal(0, noise, self.total).astype(xp.float32)
                self.osc_c += xp.random.normal(0, noise, self.total).astype(xp.float32)

            da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
            db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
            dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a = self.osc_a + da
            self.osc_b = self.osc_b + db
            self.osc_c = self.osc_c + dc

            self.osc_a = xp.clip(self.osc_a, -10, 10)
            self.osc_b = xp.clip(self.osc_b, -10, 10)
            self.osc_c = xp.clip(self.osc_c, -10, 10)

        if self.use_gpu:
            cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
        xp = cp if self.use_gpu else np

        sample_size = min(10000, self.total)
        if self.use_gpu:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r_ab = float(xp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        total_var = float(xp.var(a) + xp.var(b))
        x = min(total_var / 2.0, 1.0)

        return r_ab * (1 - x) * self.coupling_ab * 1000


def capture_k_eff(sensor: ValidationSensor, duration: float, sample_rate: float,
                  noise: float = 0.001, progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Capture k_eff time series."""
    n_samples = int(duration * sample_rate)
    interval = 1.0 / sample_rate

    k_eff = np.zeros(n_samples)
    timestamps = np.zeros(n_samples)

    start_time = time.time()

    for i in range(n_samples):
        sample_start = time.perf_counter()

        sensor.step(noise=noise)
        k_eff[i] = sensor.measure_k_eff()
        timestamps[i] = time.time() - start_time

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if progress and (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_samples}")

    return k_eff, timestamps


def compute_spectrum(k_eff: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum."""
    k_eff_detrend = signal.detrend(k_eff)
    n_fft = 2 ** int(np.ceil(np.log2(len(k_eff))) + 1)
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
    power = np.abs(np.fft.rfft(k_eff_detrend, n=n_fft)) ** 2
    return freqs, power


def fit_tau(k_eff: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float]:
    """Fit exponential decay to autocorrelation."""
    # Compute autocorrelation
    k_centered = k_eff - np.mean(k_eff)
    autocorr = np.correlate(k_centered, k_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Time axis
    dt = np.mean(np.diff(timestamps))
    t = np.arange(len(autocorr)) * dt

    # Fit exponential: r(t) = A * exp(-t/tau)
    try:
        # Only fit first half
        n_fit = len(t) // 2
        t_fit = t[:n_fit]
        r_fit = autocorr[:n_fit]

        # Remove negative values
        valid = r_fit > 0.01
        if np.sum(valid) < 10:
            return 30.0, 10.0  # Default

        t_valid = t_fit[valid]
        r_valid = r_fit[valid]

        # Linear fit to log
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_r = np.log(r_valid)

        slope, intercept, r_value, p_value, std_err = stats.linregress(t_valid, log_r)

        tau = -1.0 / slope if slope < 0 else 100.0
        tau_err = abs(tau * std_err / slope) if slope != 0 else 10.0

        return tau, tau_err

    except Exception as e:
        return 30.0, 10.0


def find_peak(freqs: np.ndarray, power: np.ndarray,
              freq_min: float = 0.005, freq_max: float = 0.5) -> Tuple[float, float]:
    """Find peak frequency and power in range."""
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if np.sum(mask) == 0:
        return 0.0, 0.0

    freqs_range = freqs[mask]
    power_range = power[mask]

    peak_idx = np.argmax(power_range)
    return freqs_range[peak_idx], power_range[peak_idx]


# =============================================================================
# PHASE 1: NULL HYPOTHESIS TESTS
# =============================================================================

def test_N1_temporal_shuffle(duration: float = 120, sample_rate: float = 10) -> ExperimentResult:
    """N1: Shuffle k_eff time series, recompute spectrum."""
    print("\n" + "="*60)
    print("N1: TEMPORAL SHUFFLE TEST")
    print("="*60)
    print("\nIf peak persists after shuffle, it's in the analysis, not the signal.")

    sensor = ValidationSensor(2048)
    k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate)

    # Original spectrum
    freqs, power_orig = compute_spectrum(k_eff, sample_rate)
    f_peak_orig, p_peak_orig = find_peak(freqs, power_orig)

    # Shuffled spectrum
    k_shuffled = k_eff.copy()
    np.random.shuffle(k_shuffled)
    _, power_shuffled = compute_spectrum(k_shuffled, sample_rate)
    f_peak_shuffled, p_peak_shuffled = find_peak(freqs, power_shuffled)

    tau, tau_err = fit_tau(k_eff, timestamps)

    # Analysis
    power_ratio = p_peak_orig / p_peak_shuffled if p_peak_shuffled > 0 else float('inf')

    print(f"\n  Original peak:  f = {f_peak_orig:.4f} Hz, power = {p_peak_orig:.2f}")
    print(f"  Shuffled peak:  f = {f_peak_shuffled:.4f} Hz, power = {p_peak_shuffled:.2f}")
    print(f"  Power ratio:    {power_ratio:.2f}x")

    if power_ratio > 3:
        conclusion = "PASS: Peak is temporal structure, not analysis artifact"
    else:
        conclusion = "FAIL: Peak persists after shuffle - analysis artifact?"

    print(f"\n  CONCLUSION: {conclusion}")

    return ExperimentResult(
        experiment_id="N1_temporal_shuffle",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=k_eff,
        tau=tau, tau_err=tau_err,
        f_peak=f_peak_orig, p_peak=p_peak_orig,
        spectrum_freqs=freqs, spectrum_power=power_orig,
        parameters={'duration': duration, 'sample_rate': sample_rate,
                   'power_ratio': power_ratio, 'f_shuffled': f_peak_shuffled},
        conclusion=conclusion
    )


def test_N2_gpu_load(duration: float = 60, sample_rate: float = 10) -> ExperimentResult:
    """N2: Compare idle vs 100% GPU load."""
    print("\n" + "="*60)
    print("N2: GPU LOAD TEST")
    print("="*60)
    print("\nComparing idle vs heavy compute load.")

    sensor = ValidationSensor(2048)

    # Idle measurement
    print("\n  Phase 1: Idle GPU...")
    k_eff_idle, ts_idle = capture_k_eff(sensor, duration, sample_rate)
    freqs, power_idle = compute_spectrum(k_eff_idle, sample_rate)
    f_peak_idle, p_peak_idle = find_peak(freqs, power_idle)
    tau_idle, _ = fit_tau(k_eff_idle, ts_idle)

    # Heavy load measurement
    print("\n  Phase 2: Heavy compute load...")
    if HAS_CUDA:
        # Create heavy compute matrices
        heavy_a = cp.random.random((4096, 4096), dtype=cp.float32)
        heavy_b = cp.random.random((4096, 4096), dtype=cp.float32)

    sensor.reset()
    k_eff_load = np.zeros(int(duration * sample_rate))
    timestamps_load = np.zeros_like(k_eff_load)
    interval = 1.0 / sample_rate
    start_time = time.time()

    for i in range(len(k_eff_load)):
        sample_start = time.perf_counter()

        # Heavy compute
        if HAS_CUDA:
            for _ in range(3):
                _ = cp.matmul(heavy_a, heavy_b)
            cp.cuda.stream.get_current_stream().synchronize()

        sensor.step()
        k_eff_load[i] = sensor.measure_k_eff()
        timestamps_load[i] = time.time() - start_time

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(k_eff_load)}")

    _, power_load = compute_spectrum(k_eff_load, sample_rate)
    f_peak_load, p_peak_load = find_peak(freqs, power_load)
    tau_load, _ = fit_tau(k_eff_load, timestamps_load)

    # Analysis
    tau_change = abs(tau_load - tau_idle) / tau_idle * 100
    f_change = abs(f_peak_load - f_peak_idle) / f_peak_idle * 100 if f_peak_idle > 0 else 0

    print(f"\n  Idle:  τ = {tau_idle:.1f}s, f_peak = {f_peak_idle:.4f} Hz")
    print(f"  Load:  τ = {tau_load:.1f}s, f_peak = {f_peak_load:.4f} Hz")
    print(f"  τ change: {tau_change:.1f}%")
    print(f"  f change: {f_change:.1f}%")

    if tau_change < 20 and f_change < 20:
        conclusion = "PASS: Spectrum stable under load (not compute artifact)"
    else:
        conclusion = "FAIL: Spectrum changes under load - compute artifact?"

    print(f"\n  CONCLUSION: {conclusion}")

    return ExperimentResult(
        experiment_id="N2_gpu_load",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=k_eff_idle,
        tau=tau_idle, tau_err=0,
        f_peak=f_peak_idle, p_peak=p_peak_idle,
        spectrum_freqs=freqs, spectrum_power=power_idle,
        parameters={'duration': duration, 'sample_rate': sample_rate,
                   'tau_idle': tau_idle, 'tau_load': tau_load,
                   'f_idle': f_peak_idle, 'f_load': f_peak_load,
                   'tau_change_pct': tau_change, 'f_change_pct': f_change},
        conclusion=conclusion
    )


def test_N3_software_rng(duration: float = 120, sample_rate: float = 10) -> ExperimentResult:
    """N3: Replace GPU chaos with numpy PRNG only."""
    print("\n" + "="*60)
    print("N3: SOFTWARE RNG TEST")
    print("="*60)
    print("\nUsing CPU numpy PRNG instead of GPU.")

    # CPU-only sensor
    sensor = ValidationSensor(2048, use_gpu=False)
    k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate)

    freqs, power = compute_spectrum(k_eff, sample_rate)
    f_peak, p_peak = find_peak(freqs, power)
    tau, tau_err = fit_tau(k_eff, timestamps)

    # Compare to expected GPU values
    # (GPU typically gives f_peak ~0.0084 Hz, τ ~46s)
    f_diff = abs(f_peak - 0.0084) / 0.0084 * 100 if f_peak > 0 else 100
    tau_diff = abs(tau - 46) / 46 * 100

    print(f"\n  CPU result: τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")
    print(f"  Expected GPU: τ ~46s, f_peak ~0.0084 Hz")
    print(f"  τ difference: {tau_diff:.1f}%")
    print(f"  f difference: {f_diff:.1f}%")

    if f_diff < 30 and tau_diff < 30:
        conclusion = "FAIL: CPU produces same spectrum - not GPU-specific"
    else:
        conclusion = "PASS: CPU gives different spectrum - GPU-specific phenomenon"

    print(f"\n  CONCLUSION: {conclusion}")

    return ExperimentResult(
        experiment_id="N3_software_rng",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=k_eff,
        tau=tau, tau_err=tau_err,
        f_peak=f_peak, p_peak=p_peak,
        spectrum_freqs=freqs, spectrum_power=power,
        parameters={'duration': duration, 'sample_rate': sample_rate,
                   'use_gpu': False, 'f_diff_pct': f_diff, 'tau_diff_pct': tau_diff},
        conclusion=conclusion
    )


def test_N4_different_seeds(n_runs: int = 10, duration: float = 60,
                            sample_rate: float = 10) -> ExperimentResult:
    """N4: Multiple runs with different seeds."""
    print("\n" + "="*60)
    print("N4: SEED VARIANCE TEST")
    print("="*60)
    print(f"\nRunning {n_runs} captures with different random seeds.")

    f_peaks = []
    taus = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs} (seed={run*12345})")
        sensor = ValidationSensor(2048)
        sensor.reset(seed=run * 12345)

        k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate, progress=False)
        freqs, power = compute_spectrum(k_eff, sample_rate)
        f_peak, _ = find_peak(freqs, power)
        tau, _ = fit_tau(k_eff, timestamps)

        f_peaks.append(f_peak)
        taus.append(tau)
        print(f"    τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")

    f_mean = np.mean(f_peaks)
    f_std = np.std(f_peaks)
    tau_mean = np.mean(taus)
    tau_std = np.std(taus)

    f_cv = f_std / f_mean * 100 if f_mean > 0 else 100
    tau_cv = tau_std / tau_mean * 100 if tau_mean > 0 else 100

    print(f"\n  f_peak: {f_mean:.4f} +/- {f_std:.4f} Hz (CV = {f_cv:.1f}%)")
    print(f"  τ:      {tau_mean:.1f} +/- {tau_std:.1f} s (CV = {tau_cv:.1f}%)")

    if f_cv < 20 and tau_cv < 30:
        conclusion = "PASS: Results consistent across seeds - stable phenomenon"
    else:
        conclusion = "FAIL: High variance across seeds - random artifact?"

    print(f"\n  CONCLUSION: {conclusion}")

    return ExperimentResult(
        experiment_id="N4_different_seeds",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=np.array(f_peaks),  # Store f_peaks as raw data
        tau=tau_mean, tau_err=tau_std,
        f_peak=f_mean, p_peak=f_std,
        spectrum_freqs=np.array(taus), spectrum_power=np.zeros_like(taus),
        parameters={'n_runs': n_runs, 'duration': duration,
                   'sample_rate': sample_rate, 'f_cv': f_cv, 'tau_cv': tau_cv},
        conclusion=conclusion
    )


def test_N5_deterministic(duration: float = 120, sample_rate: float = 10) -> ExperimentResult:
    """N5: Disable all randomness, fixed initialization."""
    print("\n" + "="*60)
    print("N5: DETERMINISTIC MODE TEST")
    print("="*60)
    print("\nRunning with fixed initialization, no noise injection.")

    sensor = ValidationSensor(2048, deterministic=True)
    k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate, noise=0)

    freqs, power = compute_spectrum(k_eff, sample_rate)
    f_peak, p_peak = find_peak(freqs, power)
    tau, tau_err = fit_tau(k_eff, timestamps)

    # Check if there's any spectral structure
    power_above_noise = p_peak / np.median(power) if np.median(power) > 0 else 0

    print(f"\n  τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")
    print(f"  Peak/median power ratio: {power_above_noise:.1f}x")

    if power_above_noise > 5:
        conclusion = "INTERESTING: Spectral structure in deterministic mode"
    else:
        conclusion = "EXPECTED: No structure without noise (confirms stochastic nature)"

    print(f"\n  CONCLUSION: {conclusion}")

    return ExperimentResult(
        experiment_id="N5_deterministic",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=k_eff,
        tau=tau, tau_err=tau_err,
        f_peak=f_peak, p_peak=p_peak,
        spectrum_freqs=freqs, spectrum_power=power,
        parameters={'duration': duration, 'sample_rate': sample_rate,
                   'deterministic': True, 'noise': 0,
                   'power_above_noise': power_above_noise},
        conclusion=conclusion
    )


# =============================================================================
# PHASE 2: CHARACTERIZATION
# =============================================================================

def test_C1_tau_vs_N(duration: float = 90, sample_rate: float = 10) -> ExperimentResult:
    """C1: Sweep N oscillators, measure τ and f_peak."""
    print("\n" + "="*60)
    print("C1: τ vs N_oscillators")
    print("="*60)

    N_values = [64, 256, 1024, 4096]
    results = {'N': [], 'tau': [], 'f_peak': []}

    for N in N_values:
        print(f"\n  N = {N} oscillators...")
        sensor = ValidationSensor(N)
        k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate, progress=False)

        freqs, power = compute_spectrum(k_eff, sample_rate)
        f_peak, _ = find_peak(freqs, power)
        tau, _ = fit_tau(k_eff, timestamps)

        results['N'].append(N)
        results['tau'].append(tau)
        results['f_peak'].append(f_peak)

        print(f"    τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")

    # Fit power law: τ ∝ N^α
    log_N = np.log(results['N'])
    log_tau = np.log(results['tau'])
    alpha, _, r_value, _, _ = stats.linregress(log_N, log_tau)

    print(f"\n  Power law fit: τ ∝ N^{alpha:.2f} (R² = {r_value**2:.3f})")

    conclusion = f"τ scales as N^{alpha:.2f}"

    return ExperimentResult(
        experiment_id="C1_tau_vs_N",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=np.array(results['tau']),
        tau=np.mean(results['tau']), tau_err=np.std(results['tau']),
        f_peak=np.mean(results['f_peak']), p_peak=0,
        spectrum_freqs=np.array(results['N']),
        spectrum_power=np.array(results['f_peak']),
        parameters={'N_values': N_values, 'results': results,
                   'alpha': alpha, 'r_squared': r_value**2},
        conclusion=conclusion
    )


def test_C2_tau_vs_coupling(duration: float = 90, sample_rate: float = 10) -> ExperimentResult:
    """C2: Sweep coupling ε, measure τ and f_peak."""
    print("\n" + "="*60)
    print("C2: τ vs coupling ε")
    print("="*60)

    epsilon_values = [0.0001, 0.0003, 0.001, 0.003]
    results = {'epsilon': [], 'tau': [], 'f_peak': []}

    for eps in epsilon_values:
        print(f"\n  ε = {eps}...")
        sensor = ValidationSensor(2048, coupling=eps)
        k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate, progress=False)

        freqs, power = compute_spectrum(k_eff, sample_rate)
        f_peak, _ = find_peak(freqs, power)
        tau, _ = fit_tau(k_eff, timestamps)

        results['epsilon'].append(eps)
        results['tau'].append(tau)
        results['f_peak'].append(f_peak)

        print(f"    τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")

    # Fit: τ ∝ ε^β
    log_eps = np.log(results['epsilon'])
    log_tau = np.log(results['tau'])
    beta, _, r_value, _, _ = stats.linregress(log_eps, log_tau)

    print(f"\n  Power law fit: τ ∝ ε^{beta:.2f} (R² = {r_value**2:.3f})")

    conclusion = f"τ scales as ε^{beta:.2f}"

    return ExperimentResult(
        experiment_id="C2_tau_vs_coupling",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=np.array(results['tau']),
        tau=np.mean(results['tau']), tau_err=np.std(results['tau']),
        f_peak=np.mean(results['f_peak']), p_peak=0,
        spectrum_freqs=np.array(results['epsilon']),
        spectrum_power=np.array(results['f_peak']),
        parameters={'epsilon_values': epsilon_values, 'results': results,
                   'beta': beta, 'r_squared': r_value**2},
        conclusion=conclusion
    )


def test_C3_tau_vs_noise(duration: float = 90, sample_rate: float = 10) -> ExperimentResult:
    """C3: Sweep noise σ, measure τ and f_peak."""
    print("\n" + "="*60)
    print("C3: τ vs noise σ")
    print("="*60)

    sigma_values = [0.0001, 0.001, 0.01, 0.1]
    results = {'sigma': [], 'tau': [], 'f_peak': []}

    for sigma in sigma_values:
        print(f"\n  σ = {sigma}...")
        sensor = ValidationSensor(2048)
        k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate,
                                          noise=sigma, progress=False)

        freqs, power = compute_spectrum(k_eff, sample_rate)
        f_peak, _ = find_peak(freqs, power)
        tau, _ = fit_tau(k_eff, timestamps)

        results['sigma'].append(sigma)
        results['tau'].append(tau)
        results['f_peak'].append(f_peak)

        print(f"    τ = {tau:.1f}s, f_peak = {f_peak:.4f} Hz")

    # Fit: τ ∝ σ^γ
    log_sigma = np.log(results['sigma'])
    log_tau = np.log(results['tau'])
    gamma, _, r_value, _, _ = stats.linregress(log_sigma, log_tau)

    print(f"\n  Power law fit: τ ∝ σ^{gamma:.2f} (R² = {r_value**2:.3f})")

    conclusion = f"τ scales as σ^{gamma:.2f}"

    return ExperimentResult(
        experiment_id="C3_tau_vs_noise",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=np.array(results['tau']),
        tau=np.mean(results['tau']), tau_err=np.std(results['tau']),
        f_peak=np.mean(results['f_peak']), p_peak=0,
        spectrum_freqs=np.array(results['sigma']),
        spectrum_power=np.array(results['f_peak']),
        parameters={'sigma_values': sigma_values, 'results': results,
                   'gamma': gamma, 'r_squared': r_value**2},
        conclusion=conclusion
    )


def test_C5_long_capture(duration: float = 3600, sample_rate: float = 10) -> ExperimentResult:
    """C5: 1-hour continuous capture."""
    print("\n" + "="*60)
    print("C5: LONG CAPTURE (1 hour)")
    print("="*60)
    print(f"\nCapturing {duration/60:.0f} minutes at {sample_rate} Hz...")

    sensor = ValidationSensor(2048)
    k_eff, timestamps = capture_k_eff(sensor, duration, sample_rate)

    freqs, power = compute_spectrum(k_eff, sample_rate)
    f_peak, p_peak = find_peak(freqs, power, freq_min=0.0001, freq_max=5)
    tau, tau_err = fit_tau(k_eff, timestamps)

    # Find multiple peaks
    peak_indices, _ = signal.find_peaks(power, height=np.median(power) * 5)
    peak_freqs = freqs[peak_indices]
    peak_powers = power[peak_indices]

    print(f"\n  τ = {tau:.1f} +/- {tau_err:.1f} s")
    print(f"  Primary peak: {f_peak:.5f} Hz")
    print(f"\n  All significant peaks:")
    sorted_idx = np.argsort(peak_powers)[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        print(f"    {i+1}. {peak_freqs[idx]:.5f} Hz (power = {peak_powers[idx]:.1f})")

    conclusion = f"Full spectrum captured, τ = {tau:.1f}s"

    return ExperimentResult(
        experiment_id="C5_long_capture",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        gpu_temp=get_gpu_temp(),
        k_eff_raw=k_eff,
        tau=tau, tau_err=tau_err,
        f_peak=f_peak, p_peak=p_peak,
        spectrum_freqs=freqs, spectrum_power=power,
        parameters={'duration': duration, 'sample_rate': sample_rate,
                   'n_peaks': len(peak_indices),
                   'peak_freqs': peak_freqs.tolist()[:10],
                   'peak_powers': peak_powers.tolist()[:10]},
        conclusion=conclusion
    )


# =============================================================================
# MAIN
# =============================================================================

def run_phase1():
    """Run all Phase 1 null hypothesis tests."""
    print("\n" + "="*70)
    print("PHASE 1: NULL HYPOTHESIS TESTS")
    print("="*70)
    print("\nIf any test FAILS, stop - the phenomenon may be an artifact.\n")

    results = []
    tests = [
        ("N1", test_N1_temporal_shuffle),
        ("N2", test_N2_gpu_load),
        ("N3", test_N3_software_rng),
        ("N4", test_N4_different_seeds),
        ("N5", test_N5_deterministic),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)

            if "FAIL" in result.conclusion:
                print(f"\n{'!'*60}")
                print(f"WARNING: {name} FAILED - investigate before proceeding!")
                print(f"{'!'*60}")
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")

    return results


def run_phase2():
    """Run all Phase 2 characterization tests."""
    print("\n" + "="*70)
    print("PHASE 2: CHARACTERIZATION")
    print("="*70)

    results = []
    tests = [
        ("C1", test_C1_tau_vs_N),
        ("C2", test_C2_tau_vs_coupling),
        ("C3", test_C3_tau_vs_noise),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='CIRISArray Validation Protocol')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all', 'quick'],
                       default='quick', help='Which phase to run')
    parser.add_argument('--test', help='Run specific test (e.g., N1, C2)')
    parser.add_argument('--output', '-o', default='/tmp/validation_results.npz')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CIRISARRAY VALIDATION PROTOCOL")
    print("="*70)
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Git:  {get_git_commit()}")
    print(f"  CUDA: {HAS_CUDA}")
    print(f"  GPU:  {get_gpu_temp()}°C" if get_gpu_temp() else "  GPU:  temp unknown")

    results = []

    if args.test:
        # Run specific test
        test_map = {
            'N1': test_N1_temporal_shuffle,
            'N2': test_N2_gpu_load,
            'N3': test_N3_software_rng,
            'N4': test_N4_different_seeds,
            'N5': test_N5_deterministic,
            'C1': test_C1_tau_vs_N,
            'C2': test_C2_tau_vs_coupling,
            'C3': test_C3_tau_vs_noise,
            'C5': test_C5_long_capture,
        }
        if args.test in test_map:
            result = test_map[args.test]()
            results.append(result)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available: {list(test_map.keys())}")
            return

    elif args.phase == 'quick':
        # Quick validation: N1, N4, C1
        print("\nRunning quick validation (N1, N4, C1)...")
        results.append(test_N1_temporal_shuffle(duration=60))
        results.append(test_N4_different_seeds(n_runs=5, duration=30))
        results.append(test_C1_tau_vs_N(duration=45))

    elif args.phase == '1':
        results = run_phase1()

    elif args.phase == '2':
        results = run_phase2()

    elif args.phase == 'all':
        results = run_phase1()
        results.extend(run_phase2())

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for r in results:
        status = "PASS" if "PASS" in r.conclusion else "FAIL" if "FAIL" in r.conclusion else "INFO"
        print(f"  [{status}] {r.experiment_id}: {r.conclusion}")

    # Save results
    save_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'git_commit': get_git_commit(),
        'n_experiments': len(results),
        'experiments': [r.experiment_id for r in results],
        'conclusions': [r.conclusion for r in results],
    }

    for r in results:
        prefix = r.experiment_id
        save_data[f'{prefix}_tau'] = r.tau
        save_data[f'{prefix}_f_peak'] = r.f_peak
        save_data[f'{prefix}_params'] = json.dumps(r.parameters)

    np.savez(args.output, **save_data)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
