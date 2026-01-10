#!/usr/bin/env python3
"""
Experiment 43: Detection Capability Survey
==========================================

Comprehensive test of what the 65k ossicle array can detect.

Tests:
1. Compute load (matrix multiply, crypto)
2. Memory bandwidth (large copies)
3. Power state transitions
4. External process launches
5. Network activity
6. Disk I/O
7. Temperature changes (from sustained load)
8. Clock frequency changes

Author: CIRIS Research Team
Date: January 2026
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Dict, List, Callable
import time
import subprocess
import threading
import os

PHI = (1 + np.sqrt(5)) / 2
MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003


@dataclass
class DetectionConfig:
    """Configuration for detection tests."""
    n_ossicles: int = 65536  # Full array
    oscillator_depth: int = 64
    warmup_seconds: float = 15.0
    baseline_seconds: float = 5.0
    perturbation_seconds: float = 5.0
    recovery_seconds: float = 5.0
    sample_rate_hz: float = 50.0


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


class MassiveArray:
    """65k ossicle array."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.total_elements = config.n_ossicles * config.oscillator_depth

        print(f"  Initializing {config.n_ossicles} ossicles ({self.total_elements} elements)...")

        self.osc_a = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_b = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1
        self.osc_c = cp.random.random(self.total_elements, dtype=cp.float32) * 0.1

        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

    def step(self, iterations: int = 5):
        block_size = 256
        grid_size = (self.total_elements + block_size - 1) // block_size
        batch_kernel(
            (grid_size,), (block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.config.oscillator_depth, self.total_elements, iterations)
        )

    def measure(self) -> Dict[str, float]:
        """Comprehensive measurement."""
        sample_size = 50000
        indices = cp.random.choice(self.total_elements, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_ab = 0 if np.isnan(r_ab) else r_ab

        var_total = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(var_total / 3.0, 1.0)
        k_eff = r_ab * (1 - x) * COUPLING_FACTOR * 1000

        return {
            'k_eff': k_eff,
            'variance': var_total,
            'correlation': r_ab
        }


# =========================================================================
# PERTURBATION FUNCTIONS
# =========================================================================

def perturb_none(duration: float):
    """Baseline - no perturbation."""
    time.sleep(duration)


def perturb_matmul(duration: float):
    """Heavy matrix multiplication."""
    size = 4096
    A = cp.random.random((size, size), dtype=cp.float32)
    B = cp.random.random((size, size), dtype=cp.float32)

    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        C = cp.matmul(A, B)
        cp.cuda.stream.get_current_stream().synchronize()


def perturb_memory_bandwidth(duration: float):
    """High memory bandwidth - large copies."""
    size = 100 * 1024 * 1024  # 100 MB
    src = cp.random.random(size, dtype=cp.float32)
    dst = cp.empty_like(src)

    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        cp.copyto(dst, src)
        cp.cuda.stream.get_current_stream().synchronize()


def perturb_memory_alloc(duration: float):
    """Memory allocation churn."""
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        data = cp.random.random(50 * 1024 * 1024, dtype=cp.float32)  # 200 MB
        del data
        cp.get_default_memory_pool().free_all_blocks()


def perturb_cpu_load(duration: float):
    """CPU load (separate thread)."""
    import hashlib

    def cpu_work():
        end_time = time.perf_counter() + duration
        while time.perf_counter() < end_time:
            hashlib.sha256(b'x' * 10000).hexdigest()

    threads = [threading.Thread(target=cpu_work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def perturb_disk_io(duration: float):
    """Disk I/O."""
    start = time.perf_counter()
    data = b'x' * (10 * 1024 * 1024)  # 10 MB

    while time.perf_counter() - start < duration:
        with open('/tmp/ciris_test_io.bin', 'wb') as f:
            f.write(data)
        with open('/tmp/ciris_test_io.bin', 'rb') as f:
            _ = f.read()
        time.sleep(0.1)

    try:
        os.remove('/tmp/ciris_test_io.bin')
    except:
        pass


def perturb_network(duration: float):
    """Network activity (local ping flood)."""
    try:
        subprocess.run(
            ['ping', '-c', str(int(duration * 10)), '-i', '0.1', '127.0.0.1'],
            capture_output=True, timeout=duration + 2
        )
    except:
        time.sleep(duration)


def perturb_subprocess(duration: float):
    """Launch many subprocesses."""
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        subprocess.run(['echo', 'test'], capture_output=True)
        time.sleep(0.05)


def run_detection_test(
    config: DetectionConfig,
    name: str,
    perturbation_func: Callable[[float], None],
    array: MassiveArray
) -> Dict:
    """Run a single detection test."""

    n_baseline = int(config.baseline_seconds * config.sample_rate_hz)
    n_perturb = int(config.perturbation_seconds * config.sample_rate_hz)
    n_recovery = int(config.recovery_seconds * config.sample_rate_hz)

    interval = 1.0 / config.sample_rate_hz

    # Baseline
    baseline_samples = []
    for _ in range(n_baseline):
        start = time.perf_counter()
        array.step(5)
        baseline_samples.append(array.measure())
        elapsed = time.perf_counter() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    # Perturbation (in separate thread)
    perturb_samples = []
    perturb_thread = threading.Thread(
        target=perturbation_func,
        args=(config.perturbation_seconds,)
    )
    perturb_thread.start()

    for _ in range(n_perturb):
        start = time.perf_counter()
        array.step(5)
        perturb_samples.append(array.measure())
        elapsed = time.perf_counter() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    perturb_thread.join()

    # Recovery
    recovery_samples = []
    for _ in range(n_recovery):
        start = time.perf_counter()
        array.step(5)
        recovery_samples.append(array.measure())
        elapsed = time.perf_counter() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    # Analyze
    baseline_k = np.array([s['k_eff'] for s in baseline_samples])
    perturb_k = np.array([s['k_eff'] for s in perturb_samples])
    recovery_k = np.array([s['k_eff'] for s in recovery_samples])

    baseline_var = np.array([s['variance'] for s in baseline_samples])
    perturb_var = np.array([s['variance'] for s in perturb_samples])

    # Effect size
    k_effect = (np.mean(perturb_k) - np.mean(baseline_k)) / (np.std(baseline_k) + 1e-10)
    var_effect = (np.mean(perturb_var) - np.mean(baseline_var)) / (np.std(baseline_var) + 1e-10)

    return {
        'name': name,
        'baseline_k_mean': np.mean(baseline_k),
        'baseline_k_std': np.std(baseline_k),
        'perturb_k_mean': np.mean(perturb_k),
        'perturb_k_std': np.std(perturb_k),
        'k_effect_sigma': k_effect,
        'var_effect_sigma': var_effect,
        'detected': abs(k_effect) > 3 or abs(var_effect) > 3
    }


def run_detection_survey(config: DetectionConfig) -> Dict:
    """Run full detection capability survey."""

    print("\n" + "="*70)
    print("EXPERIMENT 43: DETECTION CAPABILITY SURVEY")
    print("What can 65,536 ossicles detect?")
    print("="*70)

    # Initialize array
    print(f"\n  Initializing massive array...")
    array = MassiveArray(config)

    # Warmup
    print(f"  Warming up for {config.warmup_seconds}s...")
    start = time.perf_counter()
    while time.perf_counter() - start < config.warmup_seconds:
        array.step(20)

    # Tests
    tests = [
        ("Baseline (none)", perturb_none),
        ("Matrix multiply", perturb_matmul),
        ("Memory bandwidth", perturb_memory_bandwidth),
        ("Memory alloc/free", perturb_memory_alloc),
        ("CPU load", perturb_cpu_load),
        ("Disk I/O", perturb_disk_io),
        ("Subprocess spawn", perturb_subprocess),
    ]

    results = []

    for name, func in tests:
        print(f"\n" + "-"*60)
        print(f"Testing: {name}")
        print("-"*60)

        result = run_detection_test(config, name, func, array)
        results.append(result)

        print(f"  Baseline k_eff: {result['baseline_k_mean']:.6f} ± {result['baseline_k_std']:.6f}")
        print(f"  Perturb k_eff:  {result['perturb_k_mean']:.6f} ± {result['perturb_k_std']:.6f}")
        print(f"  K_eff effect:   {result['k_effect_sigma']:+.2f}σ")
        print(f"  Variance effect: {result['var_effect_sigma']:+.2f}σ")
        print(f"  DETECTED:       {'YES ✓' if result['detected'] else 'no'}")

    # Summary
    print("\n" + "="*70)
    print("DETECTION CAPABILITY SUMMARY")
    print("="*70)

    print(f"\n  {'Test':<25} {'K Effect':<12} {'Var Effect':<12} {'Detected'}")
    print("-"*65)

    detected_count = 0
    for r in results:
        detected = "YES ✓" if r['detected'] else "no"
        if r['detected']:
            detected_count += 1
        print(f"  {r['name']:<25} {r['k_effect_sigma']:+8.2f}σ    {r['var_effect_sigma']:+8.2f}σ    {detected}")

    print(f"\n  Detection rate: {detected_count}/{len(results)}")

    if detected_count == 0:
        print("\n  CONCLUSION: Array cannot detect any tested perturbations")
        print("              The sensor may be TOO stable after warmup")
    elif detected_count == len(results) - 1:  # All except baseline
        print("\n  CONCLUSION: Array detects most perturbation types!")
        print("              It's functioning as a system activity sensor")
    else:
        print(f"\n  CONCLUSION: Partial detection ({detected_count}/{len(results)})")

    print("\n" + "="*70)

    return {'tests': results}


def main():
    """Run detection capability survey."""

    print("="*70)
    print("CIRISARRAY 65K DETECTION CAPABILITY SURVEY")
    print("="*70)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nCUDA Device: {props['name'].decode()}")

    config = DetectionConfig()
    results = run_detection_survey(config)

    return results


if __name__ == "__main__":
    main()
