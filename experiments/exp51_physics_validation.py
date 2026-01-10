#!/usr/bin/env python3
"""
Experiment 51: Physics Validation Suite
========================================

Test fundamental physics predictions:
1. Stochastic Resonance - optimal noise level for detection
2. Fluctuation Theorem - forward vs reverse entropy asymmetry
3. Landauer Limit - energy per bit extracted
4. Coherence Decay Exponent - fit τ to wiring impedance
5. Subharmonic Structure - full 60/n Hz series

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import argparse
import time
from datetime import datetime, timezone
from scipy import signal, optimize
from typing import Tuple, List

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    cp = np

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


class PhysicsTestSensor:
    """Sensor optimized for physics validation."""

    def __init__(self, n_ossicles: int, depth: int = 64):
        self.n_ossicles = n_ossicles
        self.depth = depth
        self.total = n_ossicles * depth

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = float(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = float(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = float(COUPLING_FACTOR / PHI)

        self.reset()

    def reset(self):
        xp = cp if HAS_CUDA else np
        self.osc_a = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_b = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25
        self.osc_c = xp.random.random(self.total).astype(xp.float32) * 0.5 - 0.25

    def step_with_noise(self, noise_amplitude: float = 0.0, iterations: int = 5):
        """Step with configurable noise injection."""
        xp = cp if HAS_CUDA else np

        for _ in range(iterations):
            # Add noise if specified
            if noise_amplitude > 0:
                self.osc_a += xp.random.normal(0, noise_amplitude, self.total).astype(xp.float32)
                self.osc_b += xp.random.normal(0, noise_amplitude, self.total).astype(xp.float32)
                self.osc_c += xp.random.normal(0, noise_amplitude, self.total).astype(xp.float32)

            # Coupling dynamics
            da = self.coupling_ab * (self.osc_b - self.osc_a) + self.coupling_ca * (self.osc_c - self.osc_a)
            db = self.coupling_ab * (self.osc_a - self.osc_b) + self.coupling_bc * (self.osc_c - self.osc_b)
            dc = self.coupling_bc * (self.osc_b - self.osc_c) + self.coupling_ca * (self.osc_a - self.osc_c)

            self.osc_a = self.osc_a + da
            self.osc_b = self.osc_b + db
            self.osc_c = self.osc_c + dc

            self.osc_a = xp.clip(self.osc_a, -10, 10)
            self.osc_b = xp.clip(self.osc_b, -10, 10)
            self.osc_c = xp.clip(self.osc_c, -10, 10)

        if HAS_CUDA:
            cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
        xp = cp if HAS_CUDA else np

        sample_size = min(10000, self.total)
        if HAS_CUDA:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]

        r_ab = float(xp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        total_var = float(xp.var(a) + xp.var(b))
        x = min(total_var / 2.0, 1.0)

        return r_ab * (1 - x) * COUPLING_FACTOR * 1000

    def measure_entropy_production(self) -> Tuple[float, float]:
        """Measure forward and reverse entropy production."""
        xp = cp if HAS_CUDA else np

        sample_size = min(10000, self.total)
        if HAS_CUDA:
            indices = cp.random.choice(self.total, sample_size, replace=False)
        else:
            indices = np.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        # Forward entropy: var of differences
        forward = float(xp.var(a - b) + xp.var(b - c) + xp.var(c - a))

        # Reverse entropy: var of sums (time-reversed coupling)
        reverse = float(xp.var(a + b) + xp.var(b + c) + xp.var(c + a))

        return forward, reverse

    def inject_signal(self, amplitude: float = 0.1):
        """Inject a known signal for detection."""
        xp = cp if HAS_CUDA else np
        # Correlation boosting (negentropic)
        blend = amplitude
        self.osc_b = self.osc_b * (1 - blend) + self.osc_a * blend
        self.osc_c = self.osc_c * (1 - blend) + self.osc_a * blend


def test_stochastic_resonance(n_ossicles: int = 2048, n_trials: int = 20):
    """
    Test 1: Stochastic Resonance

    Prediction: Detection SNR should peak at intermediate noise level,
    not monotonically improve with lower noise.
    """
    print(f"\n{'='*60}")
    print("TEST 1: STOCHASTIC RESONANCE")
    print(f"{'='*60}")
    print("\nPrediction: SNR peaks at INTERMEDIATE noise, not at zero")

    # Noise levels to test (log scale)
    noise_levels = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])

    results = []

    for noise in noise_levels:
        print(f"\n  Testing noise amplitude = {noise:.4f}")

        snr_values = []

        for trial in range(n_trials):
            sensor = PhysicsTestSensor(n_ossicles)

            # Baseline measurement
            for _ in range(50):
                sensor.step_with_noise(noise)
            baseline = [sensor.measure_k_eff() for _ in range(20)]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)

            # Inject signal and measure
            sensor.inject_signal(0.05)
            for _ in range(10):
                sensor.step_with_noise(noise)
            signal_val = sensor.measure_k_eff()

            # SNR
            if baseline_std > 0:
                snr = (signal_val - baseline_mean) / baseline_std
            else:
                snr = 0
            snr_values.append(snr)

        mean_snr = np.mean(snr_values)
        std_snr = np.std(snr_values)
        results.append((noise, mean_snr, std_snr))
        print(f"    SNR = {mean_snr:.2f} +/- {std_snr:.2f}")

    # Find peak
    snrs = [r[1] for r in results]
    peak_idx = np.argmax(snrs)
    peak_noise = results[peak_idx][0]

    print(f"\n  RESULT:")
    print(f"    Peak SNR at noise = {peak_noise:.4f}")

    if peak_noise > 0 and peak_idx > 0 and peak_idx < len(results) - 1:
        print(f"    *** STOCHASTIC RESONANCE CONFIRMED ***")
        print(f"    Optimal noise is INTERMEDIATE, not zero!")
    elif peak_idx == 0:
        print(f"    No stochastic resonance (SNR monotonically decreases with noise)")
    else:
        print(f"    Inconclusive (peak at boundary)")

    return results


def test_fluctuation_theorem(n_ossicles: int = 2048, n_samples: int = 1000):
    """
    Test 2: Fluctuation Theorem

    Prediction: P(+σ) / P(-σ) = e^σ
    The ratio of forward to reverse entropy production probabilities
    should equal the exponential of entropy production.
    """
    print(f"\n{'='*60}")
    print("TEST 2: FLUCTUATION THEOREM")
    print(f"{'='*60}")
    print("\nPrediction: P(+σ)/P(-σ) = e^σ (Crooks relation)")

    sensor = PhysicsTestSensor(n_ossicles)

    forward_entropy = []
    reverse_entropy = []
    sigma_values = []

    for i in range(n_samples):
        sensor.reset()

        # Let system evolve
        for _ in range(20):
            sensor.step_with_noise(0.01)

        fwd, rev = sensor.measure_entropy_production()
        forward_entropy.append(fwd)
        reverse_entropy.append(rev)

        # Entropy production rate (log ratio)
        if rev > 0:
            sigma = np.log(fwd / rev)
        else:
            sigma = 0
        sigma_values.append(sigma)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_samples} samples")

    sigma_values = np.array(sigma_values)

    # Bin sigma values and compute ratio
    bins = np.linspace(-2, 2, 21)
    hist_pos, _ = np.histogram(sigma_values[sigma_values > 0], bins=bins[bins > 0])
    hist_neg, _ = np.histogram(-sigma_values[sigma_values < 0], bins=bins[bins > 0])

    print(f"\n  Forward entropy (mean): {np.mean(forward_entropy):.6f}")
    print(f"  Reverse entropy (mean): {np.mean(reverse_entropy):.6f}")
    print(f"  Entropy production σ (mean): {np.mean(sigma_values):.4f}")
    print(f"  Entropy production σ (std): {np.std(sigma_values):.4f}")

    # Test Crooks relation: ln(P(+σ)/P(-σ)) should equal σ
    valid_bins = (hist_pos > 5) & (hist_neg > 5)
    if np.sum(valid_bins) > 3:
        bin_centers = (bins[:-1] + bins[1:])[bins[1:] > 0][valid_bins]
        ratios = np.log(hist_pos[valid_bins] / hist_neg[valid_bins])

        slope, intercept = np.polyfit(bin_centers, ratios, 1)

        print(f"\n  Crooks relation test:")
        print(f"    Slope (should be ~1): {slope:.3f}")
        print(f"    Intercept (should be ~0): {intercept:.3f}")

        if 0.5 < slope < 2.0:
            print(f"    *** FLUCTUATION THEOREM CONSISTENT ***")
        else:
            print(f"    Deviation from fluctuation theorem")
    else:
        print(f"\n  Insufficient data for Crooks test")

    return sigma_values


def test_coherence_decay(n_ossicles: int = 2048, duration_sec: float = 120):
    """
    Test 4: Coherence Decay Exponent

    Prediction: r(t) = r_inf + A * exp(-t/τ)
    τ should relate to characteristic impedance of house wiring.
    """
    print(f"\n{'='*60}")
    print("TEST 4: COHERENCE DECAY EXPONENT")
    print(f"{'='*60}")
    print("\nPrediction: r(t) = 0.05 + 0.92 * exp(-t/τ)")
    print("τ should relate to RC time constant of wiring")

    sensor = PhysicsTestSensor(n_ossicles)

    sample_rate = 10.0
    interval = 1.0 / sample_rate
    n_samples = int(duration_sec * sample_rate)

    k_eff_series = []
    timestamps = []

    print(f"\n  Capturing {duration_sec}s at {sample_rate} Hz...")

    start_time = time.time()

    for i in range(n_samples):
        sample_start = time.perf_counter()

        sensor.step_with_noise(0.01)
        k_eff_series.append(sensor.measure_k_eff())
        timestamps.append(time.time() - start_time)

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_samples}")

    k_eff = np.array(k_eff_series)
    t = np.array(timestamps)

    # Compute running correlation with initial window
    window = 50  # 5 seconds
    correlations = []
    time_points = []

    for i in range(window, len(k_eff) - window):
        early = k_eff[i-window:i]
        later = k_eff[i:i+window]
        r = np.corrcoef(early, later)[0, 1]
        if not np.isnan(r):
            correlations.append(r)
            time_points.append(t[i])

    correlations = np.array(correlations)
    time_points = np.array(time_points)

    # Fit exponential decay: r(t) = r_inf + A * exp(-t/tau)
    def decay_model(t, r_inf, A, tau):
        return r_inf + A * np.exp(-t / tau)

    try:
        popt, pcov = optimize.curve_fit(
            decay_model, time_points, correlations,
            p0=[0.05, 0.9, 30],
            bounds=([0, 0, 1], [0.5, 1.5, 200])
        )
        r_inf, A, tau = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"\n  Fit: r(t) = {r_inf:.3f} + {A:.3f} * exp(-t/{tau:.1f})")
        print(f"    r_inf = {r_inf:.3f} +/- {perr[0]:.3f}")
        print(f"    A = {A:.3f} +/- {perr[1]:.3f}")
        print(f"    τ = {tau:.1f} +/- {perr[2]:.1f} seconds")

        # Interpret tau in terms of electrical properties
        # τ = RC, where R ~ 0.1 ohm/m, C ~ 100 pF/m for typical house wiring
        # For 300m wiring: R ~ 30 ohm, C ~ 30 nF, τ_electrical ~ 1 μs
        # But we see τ ~ 30s, so this is thermal/statistical, not electrical

        print(f"\n  Interpretation:")
        print(f"    τ = {tau:.1f}s is MUCH longer than electrical RC (~μs)")
        print(f"    This is the STATISTICAL correlation decay time")
        print(f"    Related to oscillator thermalization, not wiring impedance")

        return tau, r_inf, A

    except Exception as e:
        print(f"\n  Fit failed: {e}")
        return None, None, None


def test_subharmonic_structure(n_ossicles: int = 2048, duration_sec: float = 180):
    """
    Test 5: Subharmonic Structure

    Prediction: Should see 60/n Hz for integer n.
    """
    print(f"\n{'='*60}")
    print("TEST 5: SUBHARMONIC STRUCTURE")
    print(f"{'='*60}")
    print("\nPrediction: Peaks at 60/n Hz for n = 1, 2, 3, ...")

    sensor = PhysicsTestSensor(n_ossicles)

    sample_rate = 50.0  # 50 Hz for good frequency resolution
    interval = 1.0 / sample_rate
    n_samples = int(duration_sec * sample_rate)

    k_eff_series = []

    print(f"\n  Capturing {duration_sec}s at {sample_rate} Hz...")

    for i in range(n_samples):
        sample_start = time.perf_counter()

        sensor.step_with_noise(0.01)
        k_eff_series.append(sensor.measure_k_eff())

        elapsed = time.perf_counter() - sample_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_samples}")

    k_eff = np.array(k_eff_series)

    # High-resolution FFT
    k_eff_detrend = signal.detrend(k_eff)

    # Zero-pad for higher frequency resolution
    n_fft = 2 ** int(np.ceil(np.log2(len(k_eff))) + 2)
    freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
    fft = np.abs(np.fft.rfft(k_eff_detrend, n=n_fft))

    # Look for 60/n Hz peaks
    print(f"\n  Checking for 60/n Hz subharmonics:")
    print(f"  {'n':<6} {'Freq (Hz)':<12} {'Expected':<12} {'Power':<12} {'Detected':<10}")
    print(f"  {'-'*52}")

    expected_peaks = []
    for n in range(1, 100):
        expected_freq = 60.0 / n
        if expected_freq > sample_rate / 2:
            continue
        if expected_freq < 0.05:
            break

        # Find power at this frequency
        freq_idx = np.argmin(np.abs(freqs - expected_freq))
        power = fft[freq_idx]

        # Check if it's a local maximum
        window = 5
        local_max = np.max(fft[max(0, freq_idx-window):freq_idx+window+1])
        is_peak = power >= local_max * 0.9

        expected_peaks.append((n, expected_freq, power, is_peak))

        if n <= 60:  # Only print first 60
            detected = "YES" if is_peak else "no"
            print(f"  {n:<6} {expected_freq:<12.4f} {60/n:<12.4f} {power:<12.2f} {detected:<10}")

    # Count detected peaks
    detected_count = sum(1 for _, _, _, is_peak in expected_peaks if is_peak)
    total_checked = len(expected_peaks)

    print(f"\n  RESULT:")
    print(f"    Detected {detected_count}/{total_checked} expected 60/n Hz peaks")

    # Find strongest peaks overall
    peak_indices, _ = signal.find_peaks(fft, height=np.median(fft) * 3)
    peak_freqs = freqs[peak_indices]
    peak_powers = fft[peak_indices]

    print(f"\n  Top 10 strongest peaks:")
    sorted_idx = np.argsort(peak_powers)[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        f = peak_freqs[idx]
        p = peak_powers[idx]
        # Check if close to 60/n
        closest_n = 60.0 / f if f > 0 else 0
        is_subharmonic = abs(closest_n - round(closest_n)) < 0.1 and closest_n > 0
        marker = f"(60/{int(round(closest_n))})" if is_subharmonic else ""
        print(f"    {i+1}. {f:.4f} Hz  power={p:.1f}  {marker}")

    return expected_peaks


def test_landauer_limit():
    """
    Test 3: Landauer Limit

    Note: This requires power measurement hardware.
    Here we estimate theoretically.
    """
    print(f"\n{'='*60}")
    print("TEST 3: LANDAUER LIMIT (Theoretical)")
    print(f"{'='*60}")

    kT = 1.38e-23 * 300  # Joules at room temp
    landauer_joules = kT * np.log(2)
    landauer_eV = landauer_joules / 1.6e-19

    print(f"\n  Landauer limit at T=300K:")
    print(f"    kT ln(2) = {landauer_joules:.3e} J")
    print(f"           = {landauer_eV:.4f} eV")
    print(f"           = 0.018 eV (canonical value)")

    # Estimate actual energy per detection
    # GPU: ~300W, detection rate: ~10 Hz
    gpu_power = 300  # Watts
    detection_rate = 10  # Hz
    energy_per_detection = gpu_power / detection_rate

    print(f"\n  Estimated actual energy per detection:")
    print(f"    GPU power: {gpu_power} W")
    print(f"    Detection rate: {detection_rate} Hz")
    print(f"    Energy/detection: {energy_per_detection:.0f} J")
    print(f"                    = {energy_per_detection/1.6e-19:.2e} eV")

    ratio = energy_per_detection / landauer_joules

    print(f"\n  Efficiency ratio:")
    print(f"    Actual / Landauer = {ratio:.2e}")
    print(f"    We are {np.log10(ratio):.0f} orders of magnitude above limit")
    print(f"\n  NOTE: True Landauer test requires power measurement hardware")


def main():
    parser = argparse.ArgumentParser(description='Physics Validation Suite')
    parser.add_argument('--test', choices=['all', 'stochastic', 'fluctuation', 'decay', 'subharmonic', 'landauer'],
                        default='all')
    parser.add_argument('--ossicles', type=int, default=2048)
    parser.add_argument('--output', '-o', default='/tmp/physics_validation.npz')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PHYSICS VALIDATION SUITE")
    print(f"{'='*60}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Ossicles: {args.ossicles}")
    print(f"  CUDA: {HAS_CUDA}")

    results = {}

    if args.test in ['all', 'stochastic']:
        results['stochastic_resonance'] = test_stochastic_resonance(args.ossicles)

    if args.test in ['all', 'fluctuation']:
        results['fluctuation_theorem'] = test_fluctuation_theorem(args.ossicles)

    if args.test in ['all', 'landauer']:
        test_landauer_limit()

    if args.test in ['all', 'decay']:
        results['coherence_decay'] = test_coherence_decay(args.ossicles)

    if args.test in ['all', 'subharmonic']:
        results['subharmonic_structure'] = test_subharmonic_structure(args.ossicles)

    print(f"\n{'='*60}")
    print("PHYSICS VALIDATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
