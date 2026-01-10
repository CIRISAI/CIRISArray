#!/usr/bin/env python3
"""
Experiment 40: Perturbation Detection
=====================================

Test if the passive sensor network can detect known perturbations:
1. GPU memory allocation bursts
2. Compute load spikes
3. External CUDA kernel launches

This validates that we CAN see something when something happens.

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List
import time
import threading

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class PerturbationConfig:
    """Configuration for perturbation detection."""
    n_arrays: int = 8
    ossicles_per_array: int = 2048
    oscillator_depth: int = 64
    warmup_seconds: float = 10.0
    baseline_seconds: float = 5.0
    perturbation_seconds: float = 5.0
    recovery_seconds: float = 5.0
    sample_rate_hz: float = 100.0


batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void batch_ossicle_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int depth, int total_elements, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float a = osc_a[idx];
    float b = osc_b[idx];
    float c = osc_c[idx];

    for (int i = 0; i < iterations; i++) {
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

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}
''', 'batch_ossicle_step')


# Heavy compute kernel for perturbation
compute_kernel = cp.RawKernel(r'''
extern "C" __global__
void heavy_compute(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = data[idx];
    for (int i = 0; i < iterations; i++) {
        x = sinf(x) * cosf(x) + tanf(x * 0.01f);
        x = sqrtf(fabsf(x) + 1.0f);
    }
    data[idx] = x;
}
''', 'heavy_compute')


class SensorNetwork:
    """Sensor network for perturbation detection."""

    def __init__(self, config: PerturbationConfig):
        self.config = config
        self.total_elements = config.n_arrays * config.ossicles_per_array * config.oscillator_depth

        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 10):
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.config.oscillator_depth, self.total_elements, iterations)
        )
        cp.cuda.stream.get_current_stream().synchronize()

    def measure_k_eff(self) -> float:
        sample_size = min(50000, self.total_elements)
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        r = np.nanmean([r_ab, r_bc, r_ca])
        total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


def create_memory_perturbation(size_mb: int) -> cp.ndarray:
    """Allocate GPU memory as perturbation."""
    n_floats = size_mb * 1024 * 1024 // 4
    return cp.random.random(n_floats, dtype=cp.float32)


def create_compute_perturbation(duration_ms: int):
    """Run heavy compute as perturbation."""
    n = 10000000  # 10M elements
    data = cp.random.random(n, dtype=cp.float32)
    iterations = duration_ms * 10  # Rough scaling

    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    compute_kernel((grid_size,), (block_size,), (data, n, iterations))
    cp.cuda.stream.get_current_stream().synchronize()

    return data


def run_perturbation_test(config: PerturbationConfig, perturbation_type: str) -> Dict:
    """Run a single perturbation test."""

    network = SensorNetwork(config)

    # Warmup
    for _ in range(int(config.warmup_seconds * 10)):
        network.step(50)

    # Record timeline
    timeline = []
    events = []

    sample_interval = 1.0 / config.sample_rate_hz
    n_baseline = int(config.baseline_seconds * config.sample_rate_hz)
    n_perturbation = int(config.perturbation_seconds * config.sample_rate_hz)
    n_recovery = int(config.recovery_seconds * config.sample_rate_hz)

    start_time = time.perf_counter()

    # Phase 1: Baseline
    for i in range(n_baseline):
        network.step(10)
        k = network.measure_k_eff()
        t = time.perf_counter() - start_time
        timeline.append({'time': t, 'k_eff': k, 'phase': 'baseline'})

    # Phase 2: Perturbation
    perturbation_start = time.perf_counter() - start_time
    events.append({'time': perturbation_start, 'event': f'{perturbation_type}_start'})

    perturbation_data = None

    for i in range(n_perturbation):
        # Apply perturbation each iteration
        if perturbation_type == 'memory':
            if perturbation_data is None:
                perturbation_data = create_memory_perturbation(500)  # 500 MB
        elif perturbation_type == 'compute':
            create_compute_perturbation(5)  # 5ms burst
        elif perturbation_type == 'memory_churn':
            # Allocate and free repeatedly
            temp = create_memory_perturbation(100)
            del temp
            cp.get_default_memory_pool().free_all_blocks()

        network.step(10)
        k = network.measure_k_eff()
        t = time.perf_counter() - start_time
        timeline.append({'time': t, 'k_eff': k, 'phase': 'perturbation'})

    perturbation_end = time.perf_counter() - start_time
    events.append({'time': perturbation_end, 'event': f'{perturbation_type}_end'})

    # Release perturbation
    if perturbation_data is not None:
        del perturbation_data
        cp.get_default_memory_pool().free_all_blocks()

    # Phase 3: Recovery
    for i in range(n_recovery):
        network.step(10)
        k = network.measure_k_eff()
        t = time.perf_counter() - start_time
        timeline.append({'time': t, 'k_eff': k, 'phase': 'recovery'})

    return {
        'timeline': timeline,
        'events': events,
        'perturbation_type': perturbation_type
    }


def analyze_perturbation(result: Dict) -> Dict:
    """Analyze perturbation response."""

    timeline = result['timeline']

    # Extract phases
    baseline = [p['k_eff'] for p in timeline if p['phase'] == 'baseline']
    perturbation = [p['k_eff'] for p in timeline if p['phase'] == 'perturbation']
    recovery = [p['k_eff'] for p in timeline if p['phase'] == 'recovery']

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    perturbation_mean = np.mean(perturbation)
    perturbation_std = np.std(perturbation)

    recovery_mean = np.mean(recovery)

    # Calculate effect size
    if baseline_std > 0:
        effect_z = (perturbation_mean - baseline_mean) / baseline_std
    else:
        effect_z = 0

    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'perturbation_mean': perturbation_mean,
        'perturbation_std': perturbation_std,
        'recovery_mean': recovery_mean,
        'effect_z': effect_z,
        'detected': abs(effect_z) > 3
    }


def run_all_perturbation_tests(config: PerturbationConfig) -> Dict:
    """Run all perturbation types."""

    print("\n" + "="*70)
    print("EXPERIMENT 40: PERTURBATION DETECTION")
    print("Can the sensor network detect known perturbations?")
    print("="*70)

    print(f"\n  Configuration:")
    print(f"    {config.n_arrays} arrays × {config.ossicles_per_array} ossicles")
    print(f"    Phases: {config.baseline_seconds}s baseline, {config.perturbation_seconds}s perturbation, {config.recovery_seconds}s recovery")

    results = {}

    perturbation_types = ['memory', 'compute', 'memory_churn']

    for ptype in perturbation_types:
        print(f"\n" + "-"*70)
        print(f"TEST: {ptype.upper()} PERTURBATION")
        print("-"*70)

        result = run_perturbation_test(config, ptype)
        analysis = analyze_perturbation(result)

        print(f"\n  Baseline:     {analysis['baseline_mean']:.6f} ± {analysis['baseline_std']:.6f}")
        print(f"  Perturbation: {analysis['perturbation_mean']:.6f} ± {analysis['perturbation_std']:.6f}")
        print(f"  Recovery:     {analysis['recovery_mean']:.6f}")
        print(f"\n  Effect size:  {analysis['effect_z']:+.2f}σ")
        print(f"  DETECTED:     {'YES ✓' if analysis['detected'] else 'no'}")

        results[ptype] = {
            'result': result,
            'analysis': analysis
        }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PERTURBATION DETECTION SUMMARY")
    print("="*70)

    print("\n  Perturbation      Effect Size    Detected")
    print("-"*50)

    detected_count = 0
    for ptype in perturbation_types:
        analysis = results[ptype]['analysis']
        detected = "YES ✓" if analysis['detected'] else "no"
        if analysis['detected']:
            detected_count += 1
        print(f"  {ptype:<16}  {analysis['effect_z']:+8.2f}σ     {detected}")

    print(f"\n  Detection rate: {detected_count}/{len(perturbation_types)}")

    if detected_count == 0:
        print("\n  CONCLUSION: Sensor network cannot detect these perturbations")
        print("              The coupling mechanism may be too weak")
    elif detected_count == len(perturbation_types):
        print("\n  CONCLUSION: ALL perturbations detected!")
        print("              The sensor network is functioning as expected")
    else:
        print(f"\n  CONCLUSION: Partial detection ({detected_count}/{len(perturbation_types)})")
        print("              Some perturbation types are detectable")

    print("\n" + "="*70)

    return results


def main():
    """Run perturbation detection experiment."""

    print("="*70)
    print("CIRISARRAY PERTURBATION DETECTION TEST")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = PerturbationConfig()
    results = run_all_perturbation_tests(config)

    return results


if __name__ == "__main__":
    main()
