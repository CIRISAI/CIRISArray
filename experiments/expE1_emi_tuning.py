#!/usr/bin/env python3
"""
Experiment E1: EMI Detection Tuning
====================================

Goal: Tune the sensor array to clearly detect electromagnetic interference.

Target frequencies:
- 60 Hz power line fundamental
- 120, 180, 240 Hz harmonics
- 60/n Hz subharmonics (30, 20, 15, 12, 10, etc.)
- VRM switching (~0.3-1.0 Hz from exp97)

Method:
1. High-rate sampling (500+ Hz for 250 Hz Nyquist)
2. Long duration for frequency resolution
3. Multiple sensors for coherence analysis
4. Compare to theoretical EMI spectrum

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
import time
import sys
import os
from datetime import datetime, timezone
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from strain_gauge import StrainGauge, StrainGaugeConfig


# EMI target frequencies
POWER_LINE_HZ = 60.0
HARMONICS = [60, 120, 180, 240, 300]  # Power line harmonics
SUBHARMONICS = [60/n for n in range(2, 61)]  # 60/2=30 down to 60/60=1 Hz
VRM_RANGE = (0.2, 1.5)  # VRM switching frequency range


def analyze_spectrum(timings: np.ndarray, sample_rate: float) -> dict:
    """
    Compute power spectral density and identify peaks.
    """
    # Remove mean and detrend
    y = timings - np.mean(timings)
    y = signal.detrend(y)

    # Welch PSD for noise reduction
    nperseg = min(len(y) // 4, 2048)
    freqs, psd = signal.welch(y, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)

    # Find peaks
    peak_indices, properties = signal.find_peaks(psd, height=np.median(psd) * 2, distance=3)

    peaks = []
    for idx in peak_indices:
        peaks.append({
            'freq_hz': float(freqs[idx]),
            'power': float(psd[idx]),
            'snr_db': float(10 * np.log10(psd[idx] / np.median(psd)))
        })

    # Sort by power
    peaks = sorted(peaks, key=lambda x: x['power'], reverse=True)

    return {
        'freqs': freqs,
        'psd': psd,
        'peaks': peaks[:20],  # Top 20 peaks
        'noise_floor': float(np.median(psd)),
        'sample_rate': sample_rate,
        'n_samples': len(timings),
    }


def check_emi_frequencies(peaks: list, target_freqs: list, tolerance: float = 0.5) -> list:
    """
    Check which EMI frequencies are detected in peaks.
    """
    detected = []
    for target in target_freqs:
        for peak in peaks:
            if abs(peak['freq_hz'] - target) < tolerance:
                detected.append({
                    'target_hz': target,
                    'detected_hz': peak['freq_hz'],
                    'power': peak['power'],
                    'snr_db': peak['snr_db'],
                    'offset_hz': peak['freq_hz'] - target,
                })
                break
    return detected


def compute_coherence(signal1: np.ndarray, signal2: np.ndarray, sample_rate: float) -> dict:
    """
    Compute coherence between two signals.
    """
    nperseg = min(len(signal1) // 4, 1024)
    freqs, coh = signal.coherence(signal1, signal2, fs=sample_rate, nperseg=nperseg)

    return {
        'freqs': freqs,
        'coherence': coh,
        'mean_coherence': float(np.mean(coh)),
        'max_coherence': float(np.max(coh)),
        'freq_at_max': float(freqs[np.argmax(coh)]),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT E1: EMI DETECTION TUNING")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("Goal: Tune sensors to detect 60 Hz EMI and harmonics")
    print()

    results = {}

    # =========================================================================
    # Create sensor array
    # =========================================================================
    print("Creating sensor array...")
    n_sensors = 4

    sensors = []
    for i in range(n_sensors):
        config = StrainGaugeConfig(dt=0.025, warm_up_enabled=False)
        gauge = StrainGauge(config)
        sensors.append(gauge)

    # Calibrate
    print("Calibrating...")
    sensors[0].calibrate(duration=2.0)
    for s in sensors[1:]:
        s.baseline_timing_mean = sensors[0].baseline_timing_mean
        s.baseline_timing_std = sensors[0].baseline_timing_std

    # Warm up
    for s in sensors:
        for _ in range(100):
            s.read()

    # =========================================================================
    # Test 1: High sample rate for 60 Hz detection
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 1: High Sample Rate (500 Hz) for 60 Hz Detection")
    print("=" * 70)
    print()

    sample_rate = 500  # Hz - Nyquist = 250 Hz, captures up to 4th harmonic
    duration = 30  # seconds

    print(f"Collecting {duration}s at {sample_rate} Hz...")
    print(f"Nyquist: {sample_rate/2} Hz (captures harmonics up to {sample_rate//2} Hz)")

    n_samples = int(duration * sample_rate)
    all_timings = [[] for _ in sensors]

    interval = 1.0 / sample_rate
    start_time = time.time()

    for i in range(n_samples):
        target_time = start_time + i * interval

        # Read from all sensors
        for j, s in enumerate(sensors):
            r = s.read()
            all_timings[j].append(r.timing_mean_us)

        # Maintain sample rate
        elapsed = time.time() - start_time
        sleep_time = target_time + interval - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if (i + 1) % (sample_rate * 5) == 0:
            actual_rate = (i + 1) / (time.time() - start_time)
            print(f"  {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%) - actual rate: {actual_rate:.0f} Hz")

    # Convert to arrays
    timings = [np.array(t) for t in all_timings]
    actual_rate = len(timings[0]) / (time.time() - start_time)
    print(f"\nActual sample rate: {actual_rate:.1f} Hz")

    results['high_rate'] = {
        'target_rate': sample_rate,
        'actual_rate': actual_rate,
        'n_samples': len(timings[0]),
        'duration': time.time() - start_time,
    }

    # =========================================================================
    # Spectral Analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("SPECTRAL ANALYSIS")
    print("=" * 70)
    print()

    # Analyze first sensor
    spectrum = analyze_spectrum(timings[0], actual_rate)

    print(f"Frequency resolution: {actual_rate / len(timings[0]):.4f} Hz")
    print(f"Noise floor: {spectrum['noise_floor']:.2e}")
    print()

    # Top peaks
    print("Top 15 spectral peaks:")
    print(f"{'Rank':<6} {'Freq (Hz)':<12} {'Power':<15} {'SNR (dB)':<10}")
    print("-" * 45)

    for i, peak in enumerate(spectrum['peaks'][:15]):
        print(f"{i+1:<6} {peak['freq_hz']:<12.2f} {peak['power']:<15.2e} {peak['snr_db']:<10.1f}")

    results['spectrum'] = {
        'peaks': spectrum['peaks'],
        'noise_floor': spectrum['noise_floor'],
    }

    # =========================================================================
    # EMI Frequency Detection
    # =========================================================================
    print()
    print("=" * 70)
    print("EMI FREQUENCY DETECTION")
    print("=" * 70)
    print()

    # Check power line harmonics
    print("Power line harmonics (60, 120, 180, 240, 300 Hz):")
    harmonic_detected = check_emi_frequencies(spectrum['peaks'], HARMONICS, tolerance=2.0)

    if harmonic_detected:
        print(f"{'Target':<10} {'Detected':<12} {'SNR (dB)':<10} {'Offset':<10}")
        print("-" * 45)
        for d in harmonic_detected:
            print(f"{d['target_hz']:<10.0f} {d['detected_hz']:<12.2f} {d['snr_db']:<10.1f} {d['offset_hz']:+.2f}")
    else:
        print("  None detected above noise floor")

    results['harmonics'] = harmonic_detected

    # Check subharmonics
    print()
    print("Power line subharmonics (60/n Hz):")
    subharmonic_targets = [60/n for n in [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]]
    subharmonic_detected = check_emi_frequencies(spectrum['peaks'], subharmonic_targets, tolerance=0.5)

    if subharmonic_detected:
        print(f"{'Target':<10} {'Detected':<12} {'SNR (dB)':<10} {'60/n':<10}")
        print("-" * 45)
        for d in subharmonic_detected:
            n = 60 / d['target_hz']
            print(f"{d['target_hz']:<10.2f} {d['detected_hz']:<12.2f} {d['snr_db']:<10.1f} 60/{n:.0f}")
    else:
        print("  None detected above noise floor")

    results['subharmonics'] = subharmonic_detected

    # Check VRM range
    print()
    print(f"VRM range ({VRM_RANGE[0]}-{VRM_RANGE[1]} Hz):")
    vrm_peaks = [p for p in spectrum['peaks'] if VRM_RANGE[0] <= p['freq_hz'] <= VRM_RANGE[1]]

    if vrm_peaks:
        for p in vrm_peaks[:5]:
            print(f"  {p['freq_hz']:.3f} Hz: SNR = {p['snr_db']:.1f} dB")
    else:
        print("  None detected above noise floor")

    results['vrm'] = vrm_peaks

    # =========================================================================
    # Cross-Sensor Coherence at EMI Frequencies
    # =========================================================================
    print()
    print("=" * 70)
    print("CROSS-SENSOR COHERENCE")
    print("=" * 70)
    print()

    # Compute coherence between sensor pairs
    coh_result = compute_coherence(timings[0], timings[1], actual_rate)

    print(f"Sensors 0-1:")
    print(f"  Mean coherence: {coh_result['mean_coherence']:.3f}")
    print(f"  Max coherence:  {coh_result['max_coherence']:.3f} at {coh_result['freq_at_max']:.2f} Hz")

    # Check coherence at specific EMI frequencies
    print()
    print("Coherence at EMI frequencies:")
    emi_freqs = [1.0, 2.0, 10.0, 30.0, 60.0, 120.0]
    for target_f in emi_freqs:
        # Find closest frequency in coherence array
        idx = np.argmin(np.abs(coh_result['freqs'] - target_f))
        if coh_result['freqs'][idx] < actual_rate / 2:  # Within Nyquist
            coh_val = coh_result['coherence'][idx]
            print(f"  {target_f:6.1f} Hz: coherence = {coh_val:.3f}")

    results['coherence'] = {
        'mean': coh_result['mean_coherence'],
        'max': coh_result['max_coherence'],
        'freq_at_max': coh_result['freq_at_max'],
    }

    # =========================================================================
    # Test 2: Low Frequency Focus (VRM / Thermal)
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 2: Low Frequency Focus (10 Hz for VRM/Thermal)")
    print("=" * 70)
    print()

    sample_rate_low = 10  # Hz
    duration_low = 60  # seconds

    print(f"Collecting {duration_low}s at {sample_rate_low} Hz...")

    n_samples_low = int(duration_low * sample_rate_low)
    low_freq_timings = []

    interval_low = 1.0 / sample_rate_low
    start_time = time.time()

    for i in range(n_samples_low):
        target_time = start_time + i * interval_low

        r = sensors[0].read()
        low_freq_timings.append(r.timing_mean_us)

        sleep_time = target_time + interval_low - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if (i + 1) % (sample_rate_low * 10) == 0:
            print(f"  {i+1}/{n_samples_low} ({100*(i+1)/n_samples_low:.0f}%)")

    low_freq_timings = np.array(low_freq_timings)
    actual_rate_low = len(low_freq_timings) / (time.time() - start_time)

    # Analyze low frequency spectrum
    spectrum_low = analyze_spectrum(low_freq_timings, actual_rate_low)

    print()
    print(f"Low frequency peaks (< 5 Hz):")
    for peak in spectrum_low['peaks']:
        if peak['freq_hz'] < 5:
            print(f"  {peak['freq_hz']:.3f} Hz: SNR = {peak['snr_db']:.1f} dB")

    results['low_freq'] = {
        'sample_rate': actual_rate_low,
        'peaks': [p for p in spectrum_low['peaks'] if p['freq_hz'] < 5],
    }

    # =========================================================================
    # EMI Visibility Score
    # =========================================================================
    print()
    print("=" * 70)
    print("EMI VISIBILITY SCORE")
    print("=" * 70)
    print()

    # Calculate overall EMI visibility
    n_harmonics_detected = len(harmonic_detected)
    n_subharmonics_detected = len(subharmonic_detected)
    n_vrm_detected = len(vrm_peaks)

    max_snr = max([p['snr_db'] for p in spectrum['peaks']]) if spectrum['peaks'] else 0
    mean_emi_snr = np.mean([d['snr_db'] for d in harmonic_detected + subharmonic_detected]) if (harmonic_detected or subharmonic_detected) else 0

    print(f"  Power line harmonics detected: {n_harmonics_detected}/5")
    print(f"  Subharmonics detected: {n_subharmonics_detected}")
    print(f"  VRM peaks detected: {n_vrm_detected}")
    print(f"  Max spectral SNR: {max_snr:.1f} dB")
    print(f"  Mean EMI SNR: {mean_emi_snr:.1f} dB")

    # Overall visibility grade
    if n_harmonics_detected >= 3 and mean_emi_snr > 10:
        grade = "EXCELLENT"
    elif n_harmonics_detected >= 1 or mean_emi_snr > 5:
        grade = "GOOD"
    elif n_subharmonics_detected >= 2 or n_vrm_detected >= 2:
        grade = "MODERATE"
    else:
        grade = "POOR"

    print()
    print(f"  EMI Visibility: {grade}")

    results['visibility'] = {
        'harmonics_detected': n_harmonics_detected,
        'subharmonics_detected': n_subharmonics_detected,
        'vrm_detected': n_vrm_detected,
        'max_snr_db': max_snr,
        'mean_emi_snr_db': mean_emi_snr,
        'grade': grade,
    }

    # =========================================================================
    # Recommendations
    # =========================================================================
    print()
    print("=" * 70)
    print("TUNING RECOMMENDATIONS")
    print("=" * 70)
    print()

    if n_harmonics_detected >= 1:
        print("  ✓ 60 Hz harmonics VISIBLE - power line coupling detected")
        print(f"    Strongest at: {harmonic_detected[0]['detected_hz']:.1f} Hz")
    else:
        print("  ✗ 60 Hz harmonics NOT visible")
        print("    Try: Higher sample rate, longer duration, or notch filter removal")

    if n_subharmonics_detected >= 1:
        print()
        print("  ✓ Subharmonics VISIBLE - nonlinear power coupling")
        print(f"    Detected: {[d['target_hz'] for d in subharmonic_detected]}")
    else:
        print()
        print("  ✗ Subharmonics not visible at current resolution")

    if n_vrm_detected >= 1:
        print()
        print("  ✓ VRM switching frequencies VISIBLE")
        print(f"    Peaks at: {[p['freq_hz'] for p in vrm_peaks[:3]]}")
    else:
        print()
        print("  ✗ VRM not visible - may need lower sample rate for resolution")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("E1 SUMMARY: EMI DETECTION TUNING")
    print("=" * 70)

    print(f"""
  Instrument: {n_sensors} sensors, {actual_rate:.0f} Hz sample rate

  EMI Detection Results:
  ──────────────────────
  60 Hz harmonics: {n_harmonics_detected}/5 detected
  Subharmonics:    {n_subharmonics_detected} detected
  VRM peaks:       {n_vrm_detected} detected
  Max SNR:         {max_snr:.1f} dB
  Visibility:      {grade}

  Strongest Peaks:
  ────────────────""")

    for i, peak in enumerate(spectrum['peaks'][:5]):
        # Check if this matches known EMI
        emi_match = ""
        for h in HARMONICS:
            if abs(peak['freq_hz'] - h) < 2:
                emi_match = f"(60×{h//60})"
                break
        for n in range(2, 61):
            if abs(peak['freq_hz'] - 60/n) < 0.5:
                emi_match = f"(60/{n})"
                break

        print(f"  {i+1}. {peak['freq_hz']:7.2f} Hz  SNR={peak['snr_db']:5.1f} dB {emi_match}")

    print()
    print("=" * 70)

    return results


if __name__ == "__main__":
    import json

    results = main()

    def safe_serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return [safe_serialize(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_serialize(v) for v in obj]
        return obj

    with open('/home/emoore/CIRISArray/experiments/expE1_results.json', 'w') as f:
        json.dump(safe_serialize(results), f, indent=2)
    print(f"\nResults saved to expE1_results.json")
