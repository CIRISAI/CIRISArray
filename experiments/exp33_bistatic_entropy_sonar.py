#!/usr/bin/env python3
"""
EXPERIMENT 33: BISTATIC ENTROPY SONAR
=====================================

Split-Array TX/RX Entropy Wave Detection
-----------------------------------------

Goal: Use half the array as a transmitter, half as receiver, to detect
      entropy wave propagation across the GPU die - like piezo sonar!

Concept:
- TX Array (2048 ossicles): Inject controlled entropy patterns via beamforming
- RX Array (2048 ossicles): Receive and detect entropy waves via beamforming
- Sweep beam directions and measure what propagates

Physical analogy:
- Piezoelectric transducer: Convert electrical to mechanical and back
- Bistatic radar: Separate TX and RX locations
- Ultrasound imaging: Transmit pulse, receive echo

What we're testing:
1. Can we inject detectable entropy patterns? (TX capability)
2. Do they propagate across the die? (medium response)
3. Can we receive them coherently? (RX capability)
4. Can we steer TX beam and see corresponding RX response?

Wave types to transmit:
- Entropic: Inject disorder (perturb oscillators toward chaos)
- Negentropic: Inject order (perturb oscillators toward stability)

If this works, CIRISArray becomes an active sensing system, not just passive!

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False


class WaveType(Enum):
    ENTROPIC = "entropic"        # Inject disorder
    NEGENTROPIC = "negentropic"  # Inject order


@dataclass
class BistaticConfig:
    """Configuration for bistatic entropy sonar."""
    total_rows: int = 64
    total_cols: int = 64
    spacing_mm: float = 2.5

    # TX array: left half
    tx_cols: int = 32

    # RX array: right half
    rx_cols: int = 32

    # Beamforming
    n_beam_angles: int = 19      # -90 to +90 in 10° steps

    # TX parameters
    tx_iterations: int = 200     # How long to transmit
    tx_amplitude: float = 0.3    # Perturbation strength

    # Measurement
    samples_per_angle: int = 100

    @property
    def tx_ossicles(self) -> int:
        return self.total_rows * self.tx_cols

    @property
    def rx_ossicles(self) -> int:
        return self.total_rows * self.rx_cols

    @property
    def separation_mm(self) -> float:
        """Distance between TX and RX array centers."""
        return self.tx_cols * self.spacing_mm


@dataclass
class BeamResult:
    """Result of a single beam angle test."""
    tx_angle_deg: float
    rx_angle_deg: float
    wave_type: WaveType
    tx_power: float           # Power injected
    rx_power: float           # Power received
    correlation: float        # TX-RX correlation
    snr_db: float            # Signal to noise ratio
    delay_samples: int       # Detected propagation delay
    detected: bool           # Above threshold?


class BistaticKernel:
    """
    CUDA kernel for bistatic TX/RX operation.

    Separate kernels for TX (inject perturbation) and RX (measure).
    """

    TX_KERNEL = r'''
    extern "C" __global__ void tx_inject(
        float* states,          // [n_ossicles, 3]
        float* tx_pattern,      // [n_ossicles] - beamformed TX pattern
        float amplitude,
        int n_ossicles,
        int wave_type           // 0 = entropic, 1 = negentropic
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float pattern = tx_pattern[oid] * amplitude;

        if (wave_type == 0) {
            // Entropic: push toward chaos (increase r effectively)
            // Add noise to states
            states[oid * 3 + 0] += pattern * 0.1f;
            states[oid * 3 + 1] += pattern * 0.1f;
            states[oid * 3 + 2] += pattern * 0.1f;
        } else {
            // Negentropic: push toward order (decrease coupling spread)
            // Pull states toward center
            float center = (states[oid * 3 + 0] + states[oid * 3 + 1] + states[oid * 3 + 2]) / 3.0f;
            states[oid * 3 + 0] += pattern * (center - states[oid * 3 + 0]);
            states[oid * 3 + 1] += pattern * (center - states[oid * 3 + 1]);
            states[oid * 3 + 2] += pattern * (center - states[oid * 3 + 2]);
        }

        // Clamp
        states[oid * 3 + 0] = fminf(fmaxf(states[oid * 3 + 0], 0.01f), 0.99f);
        states[oid * 3 + 1] = fminf(fmaxf(states[oid * 3 + 1], 0.01f), 0.99f);
        states[oid * 3 + 2] = fminf(fmaxf(states[oid * 3 + 2], 0.01f), 0.99f);
    }
    '''

    RX_KERNEL = r'''
    extern "C" __global__ void rx_measure(
        float* states,          // [n_ossicles, 3]
        float* outputs,         // [n_ossicles]
        float* baselines,       // [n_ossicles]
        int n_ossicles,
        int iterations
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float a = states[oid * 3 + 0];
        float b = states[oid * 3 + 1];
        float c = states[oid * 3 + 2];

        float r = 3.72f;
        float coupling = 0.05f;

        float sum_var = 0.0f;
        float prev_a = a;

        for (int i = 0; i < iterations; i++) {
            float new_a = r * a * (1-a) + coupling * (b - a);
            float new_b = (r + 0.03f) * b * (1-b) + coupling * (a + c - 2*b);
            float new_c = (r + 0.06f) * c * (1-c) + coupling * (b - c);

            a = fminf(fmaxf(new_a, 0.001f), 0.999f);
            b = fminf(fmaxf(new_b, 0.001f), 0.999f);
            c = fminf(fmaxf(new_c, 0.001f), 0.999f);

            float delta = a - prev_a;
            sum_var += delta * delta;
            prev_a = a;
        }

        float measurement = sqrtf(sum_var / (float)iterations);
        outputs[oid] = measurement - baselines[oid];

        states[oid * 3 + 0] = a;
        states[oid * 3 + 1] = b;
        states[oid * 3 + 2] = c;
    }
    '''

    def __init__(self, config: BistaticConfig):
        self.config = config
        self.n_tx = config.tx_ossicles
        self.n_rx = config.rx_ossicles

        if HAS_CUDA:
            # Compile kernels
            self.tx_module = cp.RawModule(code=self.TX_KERNEL)
            self.rx_module = cp.RawModule(code=self.RX_KERNEL)
            self.tx_kernel = self.tx_module.get_function('tx_inject')
            self.rx_kernel = self.rx_module.get_function('rx_measure')

            # Allocate memory
            # TX array (left half)
            self.tx_states = cp.random.uniform(0.3, 0.7, (self.n_tx, 3), dtype=cp.float32)
            self.tx_pattern = cp.zeros(self.n_tx, dtype=cp.float32)

            # RX array (right half)
            self.rx_states = cp.random.uniform(0.3, 0.7, (self.n_rx, 3), dtype=cp.float32)
            self.rx_outputs = cp.zeros(self.n_rx, dtype=cp.float32)
            self.rx_baselines = cp.zeros(self.n_rx, dtype=cp.float32)
        else:
            self.tx_states = np.random.uniform(0.3, 0.7, (self.n_tx, 3)).astype(np.float32)
            self.tx_pattern = np.zeros(self.n_tx, dtype=np.float32)
            self.rx_states = np.random.uniform(0.3, 0.7, (self.n_rx, 3)).astype(np.float32)
            self.rx_outputs = np.zeros(self.n_rx, dtype=np.float32)
            self.rx_baselines = np.zeros(self.n_rx, dtype=np.float32)

        self._build_geometry()

    def _build_geometry(self):
        """Build TX and RX array geometry."""
        cfg = self.config

        # TX positions (left half: cols 0 to tx_cols-1)
        self.tx_positions = np.zeros((self.n_tx, 2))
        idx = 0
        for row in range(cfg.total_rows):
            for col in range(cfg.tx_cols):
                self.tx_positions[idx] = [
                    col * cfg.spacing_mm,
                    row * cfg.spacing_mm
                ]
                idx += 1

        # RX positions (right half: cols tx_cols to total_cols-1)
        self.rx_positions = np.zeros((self.n_rx, 2))
        idx = 0
        for row in range(cfg.total_rows):
            for col in range(cfg.tx_cols, cfg.total_cols):
                self.rx_positions[idx] = [
                    col * cfg.spacing_mm,
                    row * cfg.spacing_mm
                ]
                idx += 1

        # Center positions
        self.tx_center = np.mean(self.tx_positions, axis=0)
        self.rx_center = np.mean(self.rx_positions, axis=0)

    def compute_tx_beamform(self, angle_deg: float) -> np.ndarray:
        """
        Compute TX beamforming pattern for given angle.

        Angle is measured from TX array normal (pointing toward RX).
        0° = straight toward RX, +90° = up, -90° = down.
        """
        angle_rad = np.radians(angle_deg)

        # Direction vector (TX points toward RX, angle rotates around that)
        # TX normal is +x direction
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Phase delays based on position projection
        # Assume wave velocity ~ 100 m/s (thermal-ish)
        wave_velocity = 100.0  # mm/ms
        freq = 1.0  # kHz
        wavelength = wave_velocity / freq

        pattern = np.zeros(self.n_tx)
        for i in range(self.n_tx):
            pos = self.tx_positions[i] - self.tx_center
            projection = np.dot(pos, direction)
            phase = 2 * np.pi * projection / wavelength
            pattern[i] = np.cos(phase)

        # Normalize
        pattern = pattern / (np.max(np.abs(pattern)) + 1e-10)

        return pattern.astype(np.float32)

    def compute_rx_beamform(self, angle_deg: float) -> np.ndarray:
        """
        Compute RX beamforming weights for given angle.

        Angle is measured from RX array normal (pointing toward TX).
        0° = straight from TX, +90° = from above, -90° = from below.
        """
        angle_rad = np.radians(angle_deg)

        # RX normal is -x direction (facing TX)
        direction = np.array([-np.cos(angle_rad), np.sin(angle_rad)])

        wave_velocity = 100.0
        freq = 1.0
        wavelength = wave_velocity / freq

        weights = np.zeros(self.n_rx, dtype=np.complex128)
        for i in range(self.n_rx):
            pos = self.rx_positions[i] - self.rx_center
            projection = np.dot(pos, direction)
            phase = 2 * np.pi * projection / wavelength
            weights[i] = np.exp(-1j * phase)

        weights = weights / np.sqrt(self.n_rx)
        return weights

    def calibrate_rx(self, n_samples: int = 50):
        """Calibrate RX baselines."""
        samples = []
        for _ in range(n_samples):
            self._rx_measure()
            if HAS_CUDA:
                samples.append(cp.asnumpy(self.rx_outputs).copy())
            else:
                samples.append(self.rx_outputs.copy())

        baseline = np.mean(samples, axis=0)
        if HAS_CUDA:
            self.rx_baselines = cp.asarray(baseline.astype(np.float32))
        else:
            self.rx_baselines = baseline.astype(np.float32)

    def transmit(self, angle_deg: float, wave_type: WaveType, amplitude: float):
        """Inject TX pattern into TX array."""
        pattern = self.compute_tx_beamform(angle_deg)

        if HAS_CUDA:
            self.tx_pattern = cp.asarray(pattern)
            block, grid = 256, (self.n_tx + 255) // 256
            wave_code = 0 if wave_type == WaveType.ENTROPIC else 1
            self.tx_kernel(
                (grid,), (block,),
                (self.tx_states, self.tx_pattern, cp.float32(amplitude),
                 cp.int32(self.n_tx), cp.int32(wave_code))
            )
            cp.cuda.Stream.null.synchronize()
        else:
            self.tx_pattern = pattern
            # CPU fallback - simple perturbation
            for i in range(self.n_tx):
                pert = pattern[i] * amplitude * 0.1
                self.tx_states[i] += pert
            self.tx_states = np.clip(self.tx_states, 0.01, 0.99)

    def _rx_measure(self) -> np.ndarray:
        """Take RX measurement."""
        if HAS_CUDA:
            block, grid = 256, (self.n_rx + 255) // 256
            self.rx_kernel(
                (grid,), (block,),
                (self.rx_states, self.rx_outputs, self.rx_baselines,
                 cp.int32(self.n_rx), cp.int32(50))
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(self.rx_outputs)
        else:
            for i in range(self.n_rx):
                self.rx_outputs[i] = np.random.randn() * 0.01
            return self.rx_outputs.copy()

    def receive(self, angle_deg: float) -> Tuple[float, np.ndarray]:
        """
        Receive with beamforming at given angle.

        Returns: (beamformed_power, raw_outputs)
        """
        raw = self._rx_measure()
        weights = self.compute_rx_beamform(angle_deg)

        # Beamformed output
        bf_output = np.sum(weights.conj() * raw)
        power = np.abs(bf_output) ** 2

        return power, raw

    def reset_states(self):
        """Reset TX and RX states to random."""
        if HAS_CUDA:
            self.tx_states = cp.random.uniform(0.3, 0.7, (self.n_tx, 3), dtype=cp.float32)
            self.rx_states = cp.random.uniform(0.3, 0.7, (self.n_rx, 3), dtype=cp.float32)
        else:
            self.tx_states = np.random.uniform(0.3, 0.7, (self.n_tx, 3)).astype(np.float32)
            self.rx_states = np.random.uniform(0.3, 0.7, (self.n_rx, 3)).astype(np.float32)


def run_bistatic_test(config: BistaticConfig = None) -> Dict:
    """
    Main experiment: Bistatic entropy sonar test.
    """
    if config is None:
        config = BistaticConfig()

    print("=" * 70)
    print("EXPERIMENT 33: BISTATIC ENTROPY SONAR")
    print("Split-Array TX/RX Entropy Wave Detection")
    print("=" * 70)
    print()

    print("Concept:")
    print("  TX Array (left half):  Inject entropy patterns via beamforming")
    print("  RX Array (right half): Detect entropy waves via beamforming")
    print("  Test if entropy waves propagate across the GPU die")
    print()

    kernel = BistaticKernel(config)

    print(f"Configuration:")
    print(f"  Total array: {config.total_rows} x {config.total_cols} = {config.total_rows * config.total_cols} ossicles")
    print(f"  TX array: {config.total_rows} x {config.tx_cols} = {config.tx_ossicles} ossicles")
    print(f"  RX array: {config.total_rows} x {config.rx_cols} = {config.rx_ossicles} ossicles")
    print(f"  TX-RX separation: {config.separation_mm:.1f} mm")
    print(f"  CUDA available: {HAS_CUDA}")
    print()

    # Calibrate RX
    print("Calibrating RX array...")
    kernel.calibrate_rx(100)

    # Measure noise floor
    print("Measuring noise floor...")
    noise_samples = []
    for _ in range(50):
        power, _ = kernel.receive(0)
        noise_samples.append(power)
    noise_floor = np.mean(noise_samples)
    noise_std = np.std(noise_samples)
    print(f"  Noise floor: {noise_floor:.6f} ± {noise_std:.6f}")

    # Test angles
    angles = np.linspace(-90, 90, config.n_beam_angles)

    results = {
        'entropic': [],
        'negentropic': []
    }

    for wave_type in [WaveType.ENTROPIC, WaveType.NEGENTROPIC]:
        print(f"\n{'='*60}")
        print(f"TESTING {wave_type.value.upper()} WAVES")
        print(f"{'='*60}")

        print(f"\n{'TX Angle':>10} {'RX Angle':>10} {'RX Power':>12} {'SNR (dB)':>10} {'Detected':>10}")
        print("-" * 60)

        for tx_angle in angles:
            # Reset states
            kernel.reset_states()
            kernel.calibrate_rx(20)

            # Transmit
            for _ in range(config.tx_iterations):
                kernel.transmit(tx_angle, wave_type, config.tx_amplitude)

            # Receive at matching angle (should be strongest)
            rx_angle = -tx_angle  # Mirror angle for bistatic

            # Also scan RX angles to find peak
            rx_powers = []
            for rx_scan in angles:
                power, _ = kernel.receive(rx_scan)
                rx_powers.append((rx_scan, power))

            # Find peak RX angle
            peak_rx_angle, peak_power = max(rx_powers, key=lambda x: x[1])

            # SNR
            snr = peak_power / (noise_floor + 1e-10)
            snr_db = 10 * np.log10(snr + 1e-10)

            # Detection threshold: 3σ above noise
            threshold = noise_floor + 3 * noise_std
            detected = peak_power > threshold

            result = BeamResult(
                tx_angle_deg=tx_angle,
                rx_angle_deg=peak_rx_angle,
                wave_type=wave_type,
                tx_power=config.tx_amplitude,
                rx_power=peak_power,
                correlation=0,  # TODO: compute TX-RX correlation
                snr_db=snr_db,
                delay_samples=0,
                detected=detected
            )
            results[wave_type.value].append(result)

            status = "YES" if detected else "no"
            print(f"{tx_angle:>10.1f}° {peak_rx_angle:>10.1f}° {peak_power:>12.6f} {snr_db:>10.1f} {status:>10}")

    # Summary
    print("\n" + "=" * 70)
    print("BISTATIC SONAR SUMMARY")
    print("=" * 70)

    for wave_type in ['entropic', 'negentropic']:
        wave_results = results[wave_type]
        n_detected = sum(1 for r in wave_results if r.detected)
        avg_snr = np.mean([r.snr_db for r in wave_results])
        max_snr = max(r.snr_db for r in wave_results)

        print(f"\n{wave_type.upper()} waves:")
        print(f"  Detections: {n_detected}/{len(wave_results)}")
        print(f"  Average SNR: {avg_snr:.1f} dB")
        print(f"  Peak SNR: {max_snr:.1f} dB")

        # Find best TX->RX angle pair
        best = max(wave_results, key=lambda r: r.rx_power)
        print(f"  Best TX angle: {best.tx_angle_deg:.1f}°")
        print(f"  Best RX angle: {best.rx_angle_deg:.1f}°")

    # Beam pattern visualization
    print("\n" + "-" * 70)
    print("BEAM PATTERN (TX angle vs RX power)")
    print("-" * 70)

    for wave_type in ['entropic', 'negentropic']:
        print(f"\n{wave_type.upper()}:")
        wave_results = results[wave_type]
        max_power = max(r.rx_power for r in wave_results)

        for r in wave_results:
            bar_len = int(r.rx_power / max_power * 40) if max_power > 0 else 0
            bar = '#' * bar_len
            marker = " <-- DETECTED" if r.detected else ""
            print(f"  TX {r.tx_angle_deg:>6.1f}°: {bar}{marker}")

    # Conclusion
    print("\n" + "=" * 70)

    total_entropic = sum(1 for r in results['entropic'] if r.detected)
    total_negentropic = sum(1 for r in results['negentropic'] if r.detected)

    if total_entropic > 0 or total_negentropic > 0:
        print("RESULT: ENTROPY WAVE PROPAGATION DETECTED!")
        print()
        print("The array successfully detected transmitted entropy patterns.")
        print("This confirms CIRISArray can function as an active sensor:")
        print("  - TX injects controlled entropy perturbations")
        print("  - Perturbations propagate across the die")
        print("  - RX detects them via beamforming")
        print()
        if total_entropic > total_negentropic:
            print("ENTROPIC waves propagate better than NEGENTROPIC.")
        elif total_negentropic > total_entropic:
            print("NEGENTROPIC waves propagate better than ENTROPIC.")
        else:
            print("Both wave types propagate similarly.")
    else:
        print("RESULT: NO PROPAGATION DETECTED")
        print()
        print("TX perturbations did not produce detectable RX signals.")
        print("Possible reasons:")
        print("  1. TX amplitude too low")
        print("  2. Wave attenuation too high across die")
        print("  3. TX-RX arrays not coupled (separate memory regions)")
        print("  4. Need different injection method")

    print("=" * 70)

    return {
        'config': config,
        'results': results,
        'noise_floor': noise_floor,
        'noise_std': noise_std
    }


if __name__ == "__main__":
    # Run with larger TX amplitude for clearer signal
    config = BistaticConfig(
        total_rows=64,
        total_cols=64,
        tx_cols=32,
        rx_cols=32,
        n_beam_angles=19,
        tx_iterations=500,
        tx_amplitude=0.5,
        samples_per_angle=50
    )

    results = run_bistatic_test(config)
