#!/usr/bin/env python3
"""
Experiment 42: FFT Analysis with r_ab-Based Sensitivity
========================================================

Capture k_eff and r_ab time series, analyze spectral content
in the TRANSIENT regime (r_ab < 0.95) vs THERMALIZED (r_ab > 0.98)

Hypothesis: Transient regime may show different spectral structure
due to 20x higher sensitivity to perturbations.
"""

import numpy as np
import time
from datetime import datetime
from scipy import signal
import sys
sys.path.insert(0, '/home/emoore/CIRISArray')

try:
    from ciris_sentinel import Sentinel, SentinelConfig
except ImportError:
    print("ERROR: ciris_sentinel not found")
    sys.exit(1)

def run_fft_analysis(duration_s=120, sample_rate=10):
    """Capture data and run FFT analysis by regime"""
    
    config = SentinelConfig(
        use_r_ab_reset=False,  # Disable auto-reset to capture full dynamics
    )
    sensor = Sentinel(config)
    
    n_samples = int(duration_s * sample_rate)
    
    # Storage
    k_eff_data = []
    r_ab_data = []
    variance_data = []
    timestamps = []
    
    print(f"Capturing {duration_s}s of data at {sample_rate} Hz...")
    print("(Auto-reset disabled to see full thermalization)")
    
    start = time.time()
    for i in range(n_samples):
        state = sensor.step_and_measure_full()
        
        k_eff_data.append(state['k_eff'])
        r_ab_data.append(state['r_ab'])
        variance_data.append(state['variance'])
        timestamps.append(time.time() - start)
        
        if i % (sample_rate * 10) == 0:
            print(f"  {i//sample_rate}s: r_ab={state['r_ab']:.3f}, regime={state['regime']}")
        
        time.sleep(1.0 / sample_rate)
    
    k_eff = np.array(k_eff_data)
    r_ab = np.array(r_ab_data)
    variance = np.array(variance_data)
    t = np.array(timestamps)
    
    # Split by regime
    transient_mask = r_ab < 0.95
    thermalized_mask = r_ab > 0.98
    
    print(f"\nRegime breakdown:")
    print(f"  TRANSIENT (r_ab < 0.95): {np.sum(transient_mask)} samples ({100*np.mean(transient_mask):.1f}%)")
    print(f"  THERMALIZED (r_ab > 0.98): {np.sum(thermalized_mask)} samples ({100*np.mean(thermalized_mask):.1f}%)")
    
    # FFT for each regime
    print(f"\n{'='*60}")
    print("FFT ANALYSIS")
    print(f"{'='*60}")
    
    fs = sample_rate
    
    # Full signal FFT
    freqs_full, psd_full = signal.welch(k_eff, fs=fs, nperseg=min(256, len(k_eff)//2))
    
    print(f"\nFull signal ({len(k_eff)} samples):")
    peak_idx = np.argmax(psd_full[1:]) + 1  # Skip DC
    print(f"  Peak frequency: {freqs_full[peak_idx]:.3f} Hz")
    print(f"  Peak power: {psd_full[peak_idx]:.2e}")
    
    # Transient regime FFT
    if np.sum(transient_mask) > 50:
        k_eff_trans = k_eff[transient_mask]
        freqs_t, psd_t = signal.welch(k_eff_trans, fs=fs, nperseg=min(64, len(k_eff_trans)//2))
        peak_idx_t = np.argmax(psd_t[1:]) + 1
        print(f"\nTRANSIENT regime ({len(k_eff_trans)} samples):")
        print(f"  Peak frequency: {freqs_t[peak_idx_t]:.3f} Hz")
        print(f"  Peak power: {psd_t[peak_idx_t]:.2e}")
        print(f"  Variance: {np.var(k_eff_trans):.4f}")
    else:
        print("\nTRANSIENT: Not enough samples")
    
    # Thermalized regime FFT
    if np.sum(thermalized_mask) > 50:
        k_eff_therm = k_eff[thermalized_mask]
        freqs_th, psd_th = signal.welch(k_eff_therm, fs=fs, nperseg=min(64, len(k_eff_therm)//2))
        peak_idx_th = np.argmax(psd_th[1:]) + 1
        print(f"\nTHERMALIZED regime ({len(k_eff_therm)} samples):")
        print(f"  Peak frequency: {freqs_th[peak_idx_th]:.3f} Hz")
        print(f"  Peak power: {psd_th[peak_idx_th]:.2e}")
        print(f"  Variance: {np.var(k_eff_therm):.4f}")
    else:
        print("\nTHERMALIZED: Not enough samples")
    
    # r_ab dynamics
    print(f"\n{'='*60}")
    print("r_ab DYNAMICS")
    print(f"{'='*60}")
    freqs_r, psd_r = signal.welch(r_ab, fs=fs, nperseg=min(256, len(r_ab)//2))
    peak_idx_r = np.argmax(psd_r[1:]) + 1
    print(f"  r_ab peak frequency: {freqs_r[peak_idx_r]:.3f} Hz")
    print(f"  r_ab range: [{np.min(r_ab):.3f}, {np.max(r_ab):.3f}]")
    
    # Top 5 frequencies
    print(f"\n{'='*60}")
    print("TOP 5 FREQUENCIES (full signal)")
    print(f"{'='*60}")
    top_indices = np.argsort(psd_full)[-6:-1][::-1]  # Top 5 excluding DC
    for idx in top_indices:
        if idx > 0:
            print(f"  {freqs_full[idx]:.3f} Hz: power = {psd_full[idx]:.2e}")
    
    return {
        'k_eff': k_eff,
        'r_ab': r_ab,
        'variance': variance,
        'timestamps': t,
        'freqs': freqs_full,
        'psd': psd_full,
    }

if __name__ == "__main__":
    print(f"Experiment 42: r_ab-aware FFT Analysis")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    results = run_fft_analysis(duration_s=120, sample_rate=10)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
