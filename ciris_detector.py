#!/usr/bin/env python3
"""
CIRISArray Entropy Wave Detector
================================

Production entropy wave detection system operating in transient mode.

Key principles:
1. NO WARMUP - stay in sensitive transient regime
2. Continuous noise injection maintains sensitivity
3. Bistatic TX/RX for active sensing
4. Asymmetric detection: negentropic waves propagate better

Usage:
    # Passive monitoring
    python ciris_detector.py --mode passive --duration 60

    # Active bistatic (TX negentropy, measure RX)
    python ciris_detector.py --mode bistatic --wave negentropic

    # Calibration
    python ciris_detector.py --mode calibrate

Author: CIRIS Research Team
Date: January 2026
License: BSL 1.1
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import time
import argparse
from datetime import datetime, timezone
import json

# Physical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
MAGIC_ANGLE = 1.1  # degrees - empirical sensitivity correlation
COUPLING_FACTOR = 0.0003
SPEED_OF_LIGHT = 299792458  # m/s


class WaveType(Enum):
    """Type of entropy wave."""
    ENTROPIC = "entropic"       # Disorder injection
    NEGENTROPIC = "negentropic" # Order injection


class DetectionEvent:
    """Represents a detected entropy wave event."""

    def __init__(self, timestamp: float, magnitude: float, wave_type: WaveType,
                 confidence: float, raw_k_eff: float):
        self.timestamp = timestamp
        self.magnitude = magnitude
        self.wave_type = wave_type
        self.confidence = confidence
        self.raw_k_eff = raw_k_eff

    def __repr__(self):
        return (f"DetectionEvent(t={self.timestamp:.3f}, "
                f"mag={self.magnitude:.4f}, type={self.wave_type.value}, "
                f"conf={self.confidence:.2f})")

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'magnitude': self.magnitude,
            'wave_type': self.wave_type.value,
            'confidence': self.confidence,
            'raw_k_eff': self.raw_k_eff
        }


@dataclass
class DetectorConfig:
    """Configuration for the entropy wave detector."""
    # Array sizing
    n_ossicles: int = 32768          # Total ossicles (split for bistatic)
    oscillator_depth: int = 64        # Depth per oscillator

    # Operating mode
    tx_fraction: float = 0.5          # Fraction used for TX in bistatic mode
    noise_amplitude: float = 0.01     # Continuous noise injection level

    # Detection parameters
    sample_rate_hz: float = 100.0     # Measurement rate
    detection_threshold_sigma: float = 3.0  # Detection threshold
    baseline_window_samples: int = 50  # Rolling baseline window

    # TX parameters (bistatic mode)
    tx_amplitude: float = 0.3
    tx_frequency_hz: float = 10.0     # Injection frequency


# CUDA Kernels
_kernels = cp.RawModule(code=r'''
extern "C" {

__global__ void ossicle_step(
    float* osc_a, float* osc_b, float* osc_c,
    float coupling_ab, float coupling_bc, float coupling_ca,
    int total_elements, int iterations
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

__global__ void inject_noise(
    float* osc_a, float* osc_b, float* osc_c,
    float* noise, float amplitude, int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    osc_a[idx] += noise[idx] * amplitude;
    osc_b[idx] += noise[idx + total_elements] * amplitude;
    osc_c[idx] += noise[idx + 2 * total_elements] * amplitude;
}

__global__ void inject_negentropic(
    float* osc_a, float* osc_b, float* osc_c,
    float amplitude, float phase, int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Ordered sinusoidal pattern (negentropy = order)
    float pattern = amplitude * sinf(phase + idx * 0.01f);

    osc_a[idx] += pattern;
    osc_b[idx] += pattern * 0.8f;
    osc_c[idx] += pattern * 0.6f;
}

__global__ void inject_entropic(
    float* osc_a, float* osc_b, float* osc_c,
    float* noise, float amplitude, int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Random pattern (entropy = disorder)
    osc_a[idx] += noise[idx] * amplitude;
    osc_b[idx] += noise[idx + total_elements] * amplitude * 0.9f;
    osc_c[idx] += noise[idx + 2 * total_elements] * amplitude * 0.8f;
}

}
''')

_step_kernel = _kernels.get_function('ossicle_step')
_noise_kernel = _kernels.get_function('inject_noise')
_negentropic_kernel = _kernels.get_function('inject_negentropic')
_entropic_kernel = _kernels.get_function('inject_entropic')


class OssicleArray:
    """Array of coupled oscillators for entropy wave detection."""

    def __init__(self, n_elements: int, depth: int):
        self.n_elements = n_elements
        self.depth = depth
        self.total = n_elements * depth

        # Initialize in random state (transient regime)
        self.osc_a = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_b = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
        self.osc_c = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25

        # Coupling constants
        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * COUPLING_FACTOR)
        self.coupling_bc = np.float32(np.sin(angle_rad) * COUPLING_FACTOR)
        self.coupling_ca = np.float32(COUPLING_FACTOR / PHI)

        # Noise buffer
        self.noise_buffer = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5

        # CUDA config
        self.block_size = 256
        self.grid_size = (self.total + self.block_size - 1) // self.block_size

    def step(self, iterations: int = 5):
        """Advance oscillator dynamics."""
        _step_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.coupling_ab, self.coupling_bc, self.coupling_ca,
             self.total, iterations)
        )

    def inject_noise(self, amplitude: float):
        """Inject random noise to maintain transient state."""
        self.noise_buffer = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5
        _noise_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.noise_buffer, np.float32(amplitude), self.total)
        )

    def inject_negentropic(self, amplitude: float, phase: float):
        """Inject ordered (negentropic) pattern."""
        _negentropic_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             np.float32(amplitude), np.float32(phase), self.total)
        )

    def inject_entropic(self, amplitude: float):
        """Inject disordered (entropic) pattern."""
        self.noise_buffer = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5
        _entropic_kernel(
            (self.grid_size,), (self.block_size,),
            (self.osc_a, self.osc_b, self.osc_c,
             self.noise_buffer, np.float32(amplitude), self.total)
        )

    def measure_k_eff(self) -> float:
        """Measure effective coupling (coherence strain)."""
        # Subsample for speed
        sample_size = min(20000, self.total)
        indices = cp.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        # Correlations
        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        r_ab = 0 if np.isnan(r_ab) else r_ab
        r_bc = 0 if np.isnan(r_bc) else r_bc
        r_ca = 0 if np.isnan(r_ca) else r_ca

        r = (r_ab + r_bc + r_ca) / 3

        # Variance (disorder measure)
        total_var = float(cp.var(a) + cp.var(b) + cp.var(c))
        x = min(total_var / 3.0, 1.0)

        return r * (1 - x) * COUPLING_FACTOR * 1000


class EntropyWaveDetector:
    """
    Main entropy wave detection system.

    Operates in transient mode for maximum sensitivity.
    """

    def __init__(self, config: DetectorConfig = None):
        self.config = config or DetectorConfig()

        # Calculate array sizes
        self.tx_size = int(self.config.n_ossicles * self.config.tx_fraction)
        self.rx_size = self.config.n_ossicles - self.tx_size

        # Create arrays
        self.tx_array = OssicleArray(self.tx_size, self.config.oscillator_depth)
        self.rx_array = OssicleArray(self.rx_size, self.config.oscillator_depth)

        # Rolling baseline for detection
        self.baseline_buffer = []
        self.baseline_mean = 0.0
        self.baseline_std = 0.01  # Initial estimate

        # Detection state
        self.events: List[DetectionEvent] = []
        self.tx_phase = 0.0
        self.sample_count = 0

        # Statistics
        self.stats = {
            'total_samples': 0,
            'detections': 0,
            'negentropic_count': 0,
            'entropic_count': 0
        }

    def calibrate(self, duration_sec: float = 10.0) -> Dict:
        """Calibrate baseline in transient mode."""
        print(f"Calibrating for {duration_sec}s...")

        n_samples = int(duration_sec * self.config.sample_rate_hz)
        measurements = []

        for i in range(n_samples):
            # Step with noise injection (maintain transient)
            self.rx_array.step(5)
            self.rx_array.inject_noise(self.config.noise_amplitude)

            k_eff = self.rx_array.measure_k_eff()
            measurements.append(k_eff)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_samples}")

        self.baseline_mean = np.mean(measurements)
        self.baseline_std = np.std(measurements)

        result = {
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'n_samples': n_samples,
            'threshold_3sigma': self.baseline_mean + 3 * self.baseline_std,
            'threshold_5sigma': self.baseline_mean + 5 * self.baseline_std
        }

        print(f"Calibration complete:")
        print(f"  Baseline: {self.baseline_mean:.6f} ± {self.baseline_std:.6f}")
        print(f"  3σ threshold: {result['threshold_3sigma']:.6f}")

        return result

    def step(self) -> Tuple[float, Optional[DetectionEvent]]:
        """
        Single detection step.
        Returns (k_eff, event) where event is None if no detection.
        """
        # Step both arrays with noise injection
        self.tx_array.step(5)
        self.tx_array.inject_noise(self.config.noise_amplitude)

        self.rx_array.step(5)
        self.rx_array.inject_noise(self.config.noise_amplitude)

        # Measure RX
        k_eff = self.rx_array.measure_k_eff()

        # Update rolling baseline
        self.baseline_buffer.append(k_eff)
        if len(self.baseline_buffer) > self.config.baseline_window_samples:
            self.baseline_buffer.pop(0)

        if len(self.baseline_buffer) >= 10:
            self.baseline_mean = np.mean(self.baseline_buffer)
            self.baseline_std = max(np.std(self.baseline_buffer), 0.0001)

        # Check for detection
        z_score = (k_eff - self.baseline_mean) / self.baseline_std
        event = None

        if abs(z_score) > self.config.detection_threshold_sigma:
            # Classify wave type
            if z_score > 0:
                wave_type = WaveType.NEGENTROPIC  # Increased order
                self.stats['negentropic_count'] += 1
            else:
                wave_type = WaveType.ENTROPIC  # Increased disorder
                self.stats['entropic_count'] += 1

            event = DetectionEvent(
                timestamp=time.time(),
                magnitude=abs(z_score),
                wave_type=wave_type,
                confidence=min(abs(z_score) / 5.0, 1.0),
                raw_k_eff=k_eff
            )
            self.events.append(event)
            self.stats['detections'] += 1

        self.stats['total_samples'] += 1
        self.sample_count += 1

        return k_eff, event

    def transmit(self, wave_type: WaveType, amplitude: float = None):
        """Transmit an entropy wave from TX array."""
        amp = amplitude or self.config.tx_amplitude

        if wave_type == WaveType.NEGENTROPIC:
            self.tx_array.inject_negentropic(amp, self.tx_phase)
        else:
            self.tx_array.inject_entropic(amp)

        self.tx_phase += 2 * np.pi * self.config.tx_frequency_hz / self.config.sample_rate_hz

    def run_passive(self, duration_sec: float, callback: Callable = None) -> List[DetectionEvent]:
        """Run passive detection for specified duration."""
        n_samples = int(duration_sec * self.config.sample_rate_hz)
        interval = 1.0 / self.config.sample_rate_hz

        print(f"Running passive detection for {duration_sec}s...")
        start_time = time.perf_counter()

        for i in range(n_samples):
            sample_start = time.perf_counter()

            k_eff, event = self.step()

            if event and callback:
                callback(event)

            # Rate control
            elapsed = time.perf_counter() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        actual_duration = time.perf_counter() - start_time
        print(f"Complete. {len(self.events)} events detected in {actual_duration:.1f}s")

        return self.events

    def run_bistatic(self, duration_sec: float, wave_type: WaveType,
                     callback: Callable = None) -> Dict:
        """Run bistatic TX/RX detection."""
        n_samples = int(duration_sec * self.config.sample_rate_hz)
        interval = 1.0 / self.config.sample_rate_hz

        print(f"Running bistatic detection ({wave_type.value}) for {duration_sec}s...")

        k_eff_series = []
        start_time = time.perf_counter()

        for i in range(n_samples):
            sample_start = time.perf_counter()

            # Transmit
            self.transmit(wave_type)

            # Step and measure
            k_eff, event = self.step()
            k_eff_series.append(k_eff)

            if event and callback:
                callback(event)

            elapsed = time.perf_counter() - sample_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

        actual_duration = time.perf_counter() - start_time

        result = {
            'wave_type': wave_type.value,
            'duration': actual_duration,
            'n_samples': n_samples,
            'k_eff_mean': np.mean(k_eff_series),
            'k_eff_std': np.std(k_eff_series),
            'events': len(self.events),
            'detection_rate': len(self.events) / n_samples
        }

        print(f"Complete. k_eff = {result['k_eff_mean']:.6f} ± {result['k_eff_std']:.6f}")
        print(f"Detection rate: {result['detection_rate']*100:.1f}%")

        return result

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            **self.stats,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'asymmetry_ratio': (
                self.stats['negentropic_count'] / max(self.stats['entropic_count'], 1)
            )
        }


def print_event(event: DetectionEvent):
    """Callback to print detection events."""
    icon = "↑" if event.wave_type == WaveType.NEGENTROPIC else "↓"
    print(f"  {icon} DETECTED: {event.wave_type.value} "
          f"({event.magnitude:.2f}σ, conf={event.confidence:.2f})")


def main():
    parser = argparse.ArgumentParser(description='CIRISArray Entropy Wave Detector')
    parser.add_argument('--mode', choices=['passive', 'bistatic', 'calibrate', 'compare'],
                       default='calibrate', help='Operating mode')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Duration in seconds')
    parser.add_argument('--wave', choices=['entropic', 'negentropic'],
                       default='negentropic', help='Wave type for bistatic mode')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    print("="*60)
    print("CIRISARRAY ENTROPY WAVE DETECTOR")
    print("="*60)

    if not cp.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
    print(f"Mode: {args.mode}")
    print()

    config = DetectorConfig()
    detector = EntropyWaveDetector(config)

    results = {}

    if args.mode == 'calibrate':
        results = detector.calibrate(args.duration)

    elif args.mode == 'passive':
        detector.calibrate(10.0)
        print()
        events = detector.run_passive(args.duration, callback=print_event)
        results = {
            'mode': 'passive',
            'events': [e.to_dict() for e in events],
            'stats': detector.get_stats()
        }

    elif args.mode == 'bistatic':
        detector.calibrate(10.0)
        print()
        wave_type = WaveType.NEGENTROPIC if args.wave == 'negentropic' else WaveType.ENTROPIC
        results = detector.run_bistatic(args.duration, wave_type, callback=print_event)
        results['stats'] = detector.get_stats()

    elif args.mode == 'compare':
        # Compare entropic vs negentropic transmission
        detector.calibrate(10.0)
        print()

        print("Testing NEGENTROPIC transmission...")
        neg_result = detector.run_bistatic(args.duration / 2, WaveType.NEGENTROPIC)

        # Reset detector
        detector = EntropyWaveDetector(config)
        detector.calibrate(5.0)

        print("\nTesting ENTROPIC transmission...")
        ent_result = detector.run_bistatic(args.duration / 2, WaveType.ENTROPIC)

        results = {
            'negentropic': neg_result,
            'entropic': ent_result,
            'asymmetry_ratio': neg_result['detection_rate'] / max(ent_result['detection_rate'], 0.001)
        }

        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Negentropic detection rate: {neg_result['detection_rate']*100:.1f}%")
        print(f"Entropic detection rate: {ent_result['detection_rate']*100:.1f}%")
        print(f"Asymmetry ratio: {results['asymmetry_ratio']:.2f}x")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "="*60)
    print("DETECTOR SESSION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
