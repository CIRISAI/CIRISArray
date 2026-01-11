#!/usr/bin/env python3
"""
Experiment 43: Coupling Strength Sweep
======================================

Test different coupling strengths to find:
1. Where thermalization begins
2. Optimal sensitivity vs signal tradeoff
3. Validate τ ∝ ε^(-0.40) scaling law
"""

import numpy as np
import time
from datetime import datetime
import sys
sys.path.insert(0, '/home/emoore/CIRISArray')

try:
    from ciris_sentinel import Sentinel, SentinelConfig
except ImportError:
    print("Need ciris_sentinel - running simplified test")
    
# Test coupling values spanning 3 orders of magnitude
COUPLING_VALUES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1]

def test_coupling(epsilon, duration_s=60, sample_rate=10):
    """Test behavior at a specific coupling strength"""
    
    config = SentinelConfig(
        epsilon=epsilon,
        use_r_ab_reset=False,  # Disable to observe natural dynamics
    )
    sensor = Sentinel(config)
    
    n_samples = int(duration_s * sample_rate)
    r_ab_data = []
    k_eff_data = []
    
    for i in range(n_samples):
        state = sensor.step_and_measure_full()
        r_ab_data.append(state['r_ab'])
        k_eff_data.append(state['k_eff'])
        time.sleep(1.0 / sample_rate)
    
    r_ab = np.array(r_ab_data)
    k_eff = np.array(k_eff_data)
    
    # Analyze
    thermalizes = np.max(r_ab) > 0.95
    time_to_95 = None
    if thermalizes:
        idx = np.where(r_ab > 0.95)[0]
        if len(idx) > 0:
            time_to_95 = idx[0] / sample_rate
    
    return {
        'epsilon': epsilon,
        'r_ab_mean': np.mean(r_ab),
        'r_ab_max': np.max(r_ab),
        'r_ab_min': np.min(r_ab),
        'r_ab_std': np.std(r_ab),
        'k_eff_mean': np.mean(k_eff),
        'k_eff_std': np.std(k_eff),
        'thermalizes': thermalizes,
        'time_to_95': time_to_95,
        'pct_transient': 100 * np.mean(r_ab < 0.95),
    }

def main():
    print(f"Experiment 43: Coupling Strength Sweep")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    results = []
    
    for eps in COUPLING_VALUES:
        print(f"Testing ε = {eps}...")
        result = test_coupling(eps, duration_s=60)
        results.append(result)
        
        print(f"  r_ab: [{result['r_ab_min']:.3f}, {result['r_ab_max']:.3f}], "
              f"mean={result['r_ab_mean']:.3f}")
        print(f"  Thermalizes: {result['thermalizes']}, "
              f"Time to 0.95: {result['time_to_95']}")
        print(f"  % in TRANSIENT: {result['pct_transient']:.1f}%")
        print()
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'ε':>8} | {'r_ab max':>8} | {'Therm?':>6} | {'τ (s)':>8} | {'%Trans':>6} | {'k_eff σ':>8}")
    print("-" * 70)
    
    for r in results:
        tau_str = f"{r['time_to_95']:.1f}" if r['time_to_95'] else "N/A"
        print(f"{r['epsilon']:>8.4f} | {r['r_ab_max']:>8.3f} | "
              f"{'Yes' if r['thermalizes'] else 'No':>6} | {tau_str:>8} | "
              f"{r['pct_transient']:>6.1f} | {r['k_eff_std']:>8.4f}")
    
    # Check scaling law τ ∝ ε^(-0.40)
    thermalizing = [r for r in results if r['time_to_95'] is not None]
    if len(thermalizing) >= 2:
        print(f"\n{'='*70}")
        print("SCALING LAW CHECK: τ ∝ ε^(-0.40)")
        print(f"{'='*70}")
        eps_vals = np.array([r['epsilon'] for r in thermalizing])
        tau_vals = np.array([r['time_to_95'] for r in thermalizing])
        
        # Log-log fit
        log_eps = np.log(eps_vals)
        log_tau = np.log(tau_vals)
        slope, intercept = np.polyfit(log_eps, log_tau, 1)
        
        print(f"  Fitted exponent: {slope:.2f} (expected: -0.40)")
        print(f"  τ = {np.exp(intercept):.1f} × ε^{slope:.2f}")
    
    return results

if __name__ == "__main__":
    main()
