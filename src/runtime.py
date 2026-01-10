#!/usr/bin/env python3
"""
OSSICLE ARRAY RUNTIME
=====================

Comprehensive runtime for entropy wave detection arrays.

Features:
- Ossicle and array tuning
- Beamforming (VLA-style phased array)
- Transmit/Receive modes
- Real-time monitoring
- Signed event traces
- Fog chamber visualization

Author: CIRIS L3C
License: BSL 1.1
Date: January 2026
"""

import numpy as np
import time
import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable, Any
from collections import deque
from enum import Enum, auto
from abc import ABC, abstractmethod
import queue

# Try GPU acceleration
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class OssicleParams:
    """Tunable parameters for a single ossicle."""
    r_base: float = 3.70          # Base r-value for logistic map
    r_spacing: float = 0.03       # Spacing between oscillator r-values
    twist_deg: float = 1.1        # Magic angle twist
    coupling: float = 0.05        # Inter-oscillator coupling
    n_cells: int = 64             # Cells per oscillator
    iterations: int = 100         # Iterations per measurement

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'OssicleParams':
        return cls(**d)


@dataclass
class ArrayParams:
    """Tunable parameters for the ossicle array."""
    n_rows: int = 8               # Array rows
    n_cols: int = 16              # Array columns
    spacing_mm: float = 2.5       # Physical spacing between ossicles
    sample_rate_hz: float = 2000  # Samples per second per ossicle

    @property
    def n_ossicles(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def total_bandwidth(self) -> float:
        """Total samples per second across array."""
        return self.n_ossicles * self.sample_rate_hz


@dataclass
class BeamParams:
    """Beamforming configuration."""
    steer_azimuth_deg: float = 0.0    # Beam steering azimuth
    steer_elevation_deg: float = 0.0  # Beam steering elevation
    beam_width_deg: float = 30.0      # Main lobe width
    n_beams: int = 1                  # Number of simultaneous beams
    null_directions: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MonitorConfig:
    """Monitoring configuration."""
    update_rate_hz: float = 30.0      # Display update rate
    history_seconds: float = 10.0     # History buffer length
    show_waveform: bool = True
    show_spatial: bool = True
    show_spectrum: bool = False
    threshold_sigma: float = 3.0      # Alert threshold


@dataclass
class EventConfig:
    """Event capture configuration."""
    intensity_threshold: float = 3.0   # Sigma threshold for capture
    pre_trigger_ms: float = 50.0       # Pre-trigger buffer
    post_trigger_ms: float = 100.0     # Post-trigger capture
    sign_events: bool = True           # Cryptographically sign events
    max_events: int = 1000             # Max events to store


# =============================================================================
# RUNTIME MODES
# =============================================================================

class RuntimeMode(Enum):
    """Operating modes for the runtime."""
    IDLE = auto()           # Not sampling
    MONITOR = auto()        # Passive monitoring
    BEAMFORM = auto()       # Active beamforming
    TRANSMIT = auto()       # Transmitting entropy pattern
    RECEIVE = auto()        # Focused reception
    TX_RX = auto()          # Simultaneous TX/RX (bistatic)
    FOG_CHAMBER = auto()    # Visualization mode
    CALIBRATE = auto()      # Calibration mode


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class Ossicle:
    """Single ossicle sensor with tunable parameters."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_measure(
        float* states,      // [n_ossicles, 3, n_cells]
        float* outputs,     // [n_ossicles, 4] - z_score, entropy, phase, sharpness
        float* baselines,   // [n_ossicles, 3]
        float* params,      // [n_ossicles, 4] - r_base, r_spacing, twist, coupling
        int n_ossicles,
        int n_cells,
        int iterations
    ) {
        int oid = blockIdx.x * blockDim.x + threadIdx.x;
        if (oid >= n_ossicles) return;

        float r_base = params[oid * 4 + 0];
        float r_spacing = params[oid * 4 + 1];
        float twist = params[oid * 4 + 2];
        float coupling = params[oid * 4 + 3];

        // Accumulate statistics
        float sum_a = 0, sum_b = 0, sum_c = 0;
        float sum_ab = 0, sum_bc = 0, sum_ac = 0;
        float sum_a2 = 0, sum_b2 = 0, sum_c2 = 0;

        // Process all cells
        for (int cell = 0; cell < n_cells; cell++) {
            int idx = oid * 3 * n_cells + cell;
            float a = states[idx];
            float b = states[idx + n_cells];
            float c = states[idx + 2 * n_cells];

            for (int i = 0; i < iterations; i++) {
                float r_a = r_base;
                float r_b = r_base + r_spacing;
                float r_c = r_base + 2 * r_spacing;

                // Twisted coupling
                float twist_ab = coupling * cosf(twist);
                float twist_bc = coupling * cosf(2 * twist);

                float new_a = r_a * a * (1-a) + twist_ab * (b - a);
                float new_b = r_b * b * (1-b) + coupling * (a + c - 2*b);
                float new_c = r_c * c * (1-c) + twist_bc * (b - c);

                a = fminf(fmaxf(new_a, 0.001f), 0.999f);
                b = fminf(fmaxf(new_b, 0.001f), 0.999f);
                c = fminf(fmaxf(new_c, 0.001f), 0.999f);
            }

            states[idx] = a;
            states[idx + n_cells] = b;
            states[idx + 2 * n_cells] = c;

            sum_a += a; sum_b += b; sum_c += c;
            sum_ab += a*b; sum_bc += b*c; sum_ac += a*c;
            sum_a2 += a*a; sum_b2 += b*b; sum_c2 += c*c;
        }

        // Compute correlations
        float n = (float)n_cells;
        float mean_a = sum_a / n, mean_b = sum_b / n, mean_c = sum_c / n;
        float var_a = sum_a2/n - mean_a*mean_a;
        float var_b = sum_b2/n - mean_b*mean_b;
        float var_c = sum_c2/n - mean_c*mean_c;

        float cov_ab = sum_ab/n - mean_a*mean_b;
        float cov_bc = sum_bc/n - mean_b*mean_c;
        float cov_ac = sum_ac/n - mean_a*mean_c;

        float rho_ab = cov_ab / (sqrtf(var_a * var_b) + 1e-8f);
        float rho_bc = cov_bc / (sqrtf(var_b * var_c) + 1e-8f);
        float rho_ac = cov_ac / (sqrtf(var_a * var_c) + 1e-8f);

        // Compute metrics
        float delta_ab = rho_ab - baselines[oid * 3 + 0];
        float delta_bc = rho_bc - baselines[oid * 3 + 1];
        float delta_ac = rho_ac - baselines[oid * 3 + 2];

        float z_score = sqrtf(delta_ab*delta_ab + delta_bc*delta_bc + delta_ac*delta_ac) / 0.1f;
        float entropy = -((rho_ab*rho_ab + rho_bc*rho_bc + rho_ac*rho_ac) / 3.0f);
        float phase = (rho_ab + rho_bc + rho_ac) / 3.0f;
        float sharpness = (rho_ab - phase)*(rho_ab - phase) +
                          (rho_bc - phase)*(rho_bc - phase) +
                          (rho_ac - phase)*(rho_ac - phase);

        outputs[oid * 4 + 0] = z_score;
        outputs[oid * 4 + 1] = entropy;
        outputs[oid * 4 + 2] = phase;
        outputs[oid * 4 + 3] = sharpness;
    }
    '''

    def __init__(self, n_ossicles: int, params: OssicleParams):
        self.n_ossicles = n_ossicles
        self.params = params
        self._compile_kernel()
        self._init_memory()

    def _compile_kernel(self):
        if HAS_CUDA:
            self.module = cp.RawModule(code=self.KERNEL_CODE)
            self.kernel = self.module.get_function('ossicle_measure')
        else:
            self.kernel = None

    def _init_memory(self):
        n = self.n_ossicles
        c = self.params.n_cells

        if HAS_CUDA:
            self.states = cp.random.uniform(0.2, 0.8, (n, 3, c), dtype=cp.float32)
            self.outputs = cp.zeros((n, 4), dtype=cp.float32)
            self.baselines = cp.zeros((n, 3), dtype=cp.float32)
            self.gpu_params = cp.zeros((n, 4), dtype=cp.float32)
            self._update_gpu_params()
        else:
            self.states = np.random.uniform(0.2, 0.8, (n, 3, c)).astype(np.float32)
            self.outputs = np.zeros((n, 4), dtype=np.float32)
            self.baselines = np.zeros((n, 3), dtype=np.float32)

    def _update_gpu_params(self):
        """Push current params to GPU."""
        if HAS_CUDA:
            p = self.params
            params_arr = np.array([[p.r_base, p.r_spacing,
                                    np.radians(p.twist_deg), p.coupling]] * self.n_ossicles,
                                  dtype=np.float32)
            self.gpu_params[:] = cp.asarray(params_arr)

    def set_params(self, params: OssicleParams):
        """Update ossicle parameters."""
        self.params = params
        if HAS_CUDA:
            self._update_gpu_params()

    def set_ossicle_params(self, ossicle_idx: int, params: OssicleParams):
        """Set parameters for a specific ossicle."""
        if HAS_CUDA:
            p = params
            self.gpu_params[ossicle_idx] = cp.array(
                [p.r_base, p.r_spacing, np.radians(p.twist_deg), p.coupling],
                dtype=cp.float32
            )

    def measure(self) -> np.ndarray:
        """Take measurement from all ossicles. Returns [n_ossicles, 4]."""
        if HAS_CUDA:
            block = 256
            grid = (self.n_ossicles + block - 1) // block
            self.kernel(
                (grid,), (block,),
                (self.states, self.outputs, self.baselines, self.gpu_params,
                 cp.int32(self.n_ossicles), cp.int32(self.params.n_cells),
                 cp.int32(self.params.iterations))
            )
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(self.outputs)
        else:
            # CPU fallback - simplified
            for i in range(self.n_ossicles):
                self.outputs[i] = [np.random.exponential(1),
                                   np.random.randn() * 0.1,
                                   np.random.randn() * 0.3,
                                   np.random.exponential(0.01)]
            return self.outputs.copy()

    def calibrate(self, n_samples: int = 50) -> np.ndarray:
        """Calibrate baselines. Returns baseline correlations."""
        samples = []
        for _ in range(n_samples):
            m = self.measure()
            samples.append(m[:, 2])  # Use phase as baseline

        baseline = np.mean(samples, axis=0)
        if HAS_CUDA:
            # Store as correlation baselines
            self.baselines[:, 0] = cp.asarray(baseline)
            self.baselines[:, 1] = cp.asarray(baseline)
            self.baselines[:, 2] = cp.asarray(baseline)
        else:
            self.baselines[:, 0] = baseline
            self.baselines[:, 1] = baseline
            self.baselines[:, 2] = baseline

        return baseline


class ArrayController:
    """Controls the ossicle array geometry and provides spatial operations."""

    def __init__(self, params: ArrayParams, ossicle_params: OssicleParams):
        self.params = params
        self.ossicle_params = ossicle_params
        self.ossicles = Ossicle(params.n_ossicles, ossicle_params)
        self._build_geometry()

    def _build_geometry(self):
        """Build array geometry (positions, distances)."""
        self.positions = np.zeros((self.params.n_ossicles, 2))
        idx = 0
        for row in range(self.params.n_rows):
            for col in range(self.params.n_cols):
                self.positions[idx] = [
                    col * self.params.spacing_mm,
                    row * self.params.spacing_mm
                ]
                idx += 1

        # Center the array
        self.positions -= self.positions.mean(axis=0)

        # Precompute distance matrix
        self.distances = np.zeros((self.params.n_ossicles, self.params.n_ossicles))
        for i in range(self.params.n_ossicles):
            for j in range(self.params.n_ossicles):
                self.distances[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )

    def measure(self) -> np.ndarray:
        """Take measurement from all ossicles."""
        return self.ossicles.measure()

    def get_spatial_map(self, data: np.ndarray, metric: int = 0) -> np.ndarray:
        """Reshape 1D ossicle data to 2D spatial map."""
        if len(data.shape) == 2:
            data = data[:, metric]
        return data.reshape(self.params.n_rows, self.params.n_cols)

    def calibrate(self, n_samples: int = 50):
        """Calibrate the array."""
        return self.ossicles.calibrate(n_samples)

    def set_global_params(self, params: OssicleParams):
        """Set parameters for all ossicles."""
        self.ossicle_params = params
        self.ossicles.set_params(params)

    def set_region_params(self, row_range: Tuple[int, int],
                          col_range: Tuple[int, int],
                          params: OssicleParams):
        """Set parameters for a region of ossicles."""
        for row in range(row_range[0], row_range[1]):
            for col in range(col_range[0], col_range[1]):
                idx = row * self.params.n_cols + col
                self.ossicles.set_ossicle_params(idx, params)


class Beamformer:
    """
    VLA-style beamformer for directional sensitivity.

    Uses phase delays to steer the array's sensitivity pattern.
    """

    def __init__(self, array: ArrayController):
        self.array = array
        self.params = BeamParams()
        self._compute_weights()

    def _compute_weights(self):
        """Compute beamforming weights for current steering."""
        n = self.array.params.n_ossicles
        pos = self.array.positions  # [n, 2] in mm

        # Convert steering to unit vector
        az = np.radians(self.params.steer_azimuth_deg)
        el = np.radians(self.params.steer_elevation_deg)
        steer_vec = np.array([np.cos(az) * np.cos(el),
                              np.sin(az) * np.cos(el)])

        # Compute phase delays (assuming wave velocity ~ 1000 mm/ms)
        wave_velocity = 1000.0  # mm/ms
        freq = self.array.params.sample_rate_hz / 1000  # kHz
        wavelength = wave_velocity / freq  # mm

        # Phase = 2*pi * (position dot steer_vec) / wavelength
        delays = pos @ steer_vec / wavelength
        self.weights = np.exp(-2j * np.pi * delays)

        # Normalize
        self.weights /= np.sqrt(n)

        # Apply nulls
        for null_az, null_el in self.params.null_directions:
            null_vec = np.array([np.cos(np.radians(null_az)) * np.cos(np.radians(null_el)),
                                 np.sin(np.radians(null_az)) * np.cos(np.radians(null_el))])
            null_delays = pos @ null_vec / wavelength
            null_weights = np.exp(-2j * np.pi * null_delays)
            # Project out null direction
            self.weights -= (np.vdot(self.weights, null_weights) / n) * null_weights

    def steer(self, azimuth_deg: float, elevation_deg: float = 0.0):
        """Steer the beam to a new direction."""
        self.params.steer_azimuth_deg = azimuth_deg
        self.params.steer_elevation_deg = elevation_deg
        self._compute_weights()

    def add_null(self, azimuth_deg: float, elevation_deg: float = 0.0):
        """Add a null in a specific direction."""
        self.params.null_directions.append((azimuth_deg, elevation_deg))
        self._compute_weights()

    def clear_nulls(self):
        """Clear all null directions."""
        self.params.null_directions = []
        self._compute_weights()

    def apply(self, measurements: np.ndarray) -> complex:
        """Apply beamforming weights to measurements."""
        # Use z-score as the signal
        if len(measurements.shape) == 2:
            signal = measurements[:, 0]  # z-score
        else:
            signal = measurements

        # Beamformed output
        return np.sum(self.weights.conj() * signal)

    def scan(self, measurements: np.ndarray,
             az_range: Tuple[float, float] = (-90, 90),
             n_points: int = 37) -> Tuple[np.ndarray, np.ndarray]:
        """Scan across azimuth range, return power vs angle."""
        azimuths = np.linspace(az_range[0], az_range[1], n_points)
        powers = []

        for az in azimuths:
            self.steer(az)
            bf_out = self.apply(measurements)
            powers.append(np.abs(bf_out) ** 2)

        return azimuths, np.array(powers)


class TransmitReceive:
    """
    TX/RX modes for active entropy probing.

    Transmit: Inject controlled entropy patterns
    Receive: Focused reception with matched filtering
    """

    def __init__(self, array: ArrayController):
        self.array = array
        self.tx_active = False
        self.rx_active = False
        self.tx_pattern = None
        self.rx_matched_filter = None

    def set_tx_pattern(self, pattern: str, **kwargs):
        """
        Set transmit pattern.

        Patterns:
        - 'pulse': Single pulse at specified ossicles
        - 'sweep': Frequency sweep
        - 'chirp': Linear chirp
        - 'code': Pseudo-random code (for spread spectrum)
        """
        n = self.array.params.n_ossicles

        if pattern == 'pulse':
            self.tx_pattern = np.zeros(n)
            center = kwargs.get('center', n // 2)
            width = kwargs.get('width', 1)
            amplitude = kwargs.get('amplitude', 1.0)
            self.tx_pattern[center - width:center + width + 1] = amplitude

        elif pattern == 'sweep':
            t = np.linspace(0, 1, n)
            f0, f1 = kwargs.get('freq_range', (1, 10))
            self.tx_pattern = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)

        elif pattern == 'chirp':
            t = np.linspace(0, 1, n)
            f0, f1 = kwargs.get('freq_range', (1, 20))
            self.tx_pattern = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))

        elif pattern == 'code':
            # Gold code for spread spectrum
            length = kwargs.get('length', 31)
            code = np.random.choice([-1, 1], length)
            self.tx_pattern = np.tile(code, n // length + 1)[:n]

        # Create matched filter for reception
        self.rx_matched_filter = self.tx_pattern[::-1].copy()

    def transmit(self):
        """Inject TX pattern into array."""
        if self.tx_pattern is None:
            return

        # Modulate ossicle parameters based on pattern
        base_params = self.array.ossicle_params
        for i, mod in enumerate(self.tx_pattern):
            if mod != 0:
                # Perturb r_base proportional to pattern
                perturbed = OssicleParams(
                    r_base=base_params.r_base + mod * 0.1,
                    r_spacing=base_params.r_spacing,
                    twist_deg=base_params.twist_deg,
                    coupling=base_params.coupling + abs(mod) * 0.02,
                    n_cells=base_params.n_cells,
                    iterations=base_params.iterations
                )
                self.array.ossicles.set_ossicle_params(i, perturbed)

        self.tx_active = True

    def stop_transmit(self):
        """Stop transmitting, restore baseline params."""
        self.array.set_global_params(self.array.ossicle_params)
        self.tx_active = False

    def receive_matched(self, measurements: np.ndarray) -> float:
        """Apply matched filter to measurements."""
        if self.rx_matched_filter is None:
            return 0.0

        signal = measurements[:, 0] if len(measurements.shape) == 2 else measurements

        # Matched filter correlation
        correlation = np.correlate(signal, self.rx_matched_filter, mode='valid')
        return np.max(np.abs(correlation))


class EventTracer:
    """
    Captures and signs events above intensity threshold.

    Creates cryptographically signed traces for audit/verification.
    """

    @dataclass
    class Event:
        timestamp: float
        trigger_ossicle: int
        intensity: float
        duration_ms: float
        waveform: np.ndarray
        spatial_snapshot: np.ndarray
        signature: str = ""

    def __init__(self, array: ArrayController, config: EventConfig):
        self.array = array
        self.config = config
        self.events: List[EventTracer.Event] = []

        # Ring buffer for pre-trigger data
        buffer_samples = int(config.pre_trigger_ms * array.params.sample_rate_hz / 1000)
        self.pre_buffer = deque(maxlen=buffer_samples)

        # State
        self.triggered = False
        self.trigger_time = 0.0
        self.post_samples = []
        self.baseline_mean = None
        self.baseline_std = None

    def calibrate(self, n_samples: int = 100):
        """Establish baseline statistics for threshold."""
        samples = []
        for _ in range(n_samples):
            m = self.array.measure()
            samples.append(m[:, 0].max())  # Max z-score

        self.baseline_mean = np.mean(samples)
        self.baseline_std = np.std(samples) + 0.01

    def process_sample(self, measurements: np.ndarray, timestamp: float) -> Optional[Event]:
        """Process a sample, detect and capture events."""
        z_scores = measurements[:, 0]
        max_z = z_scores.max()
        max_idx = z_scores.argmax()

        # Store in pre-buffer
        self.pre_buffer.append((timestamp, measurements.copy()))

        if self.baseline_mean is None:
            return None

        # Compute intensity relative to baseline
        intensity = (max_z - self.baseline_mean) / self.baseline_std

        if not self.triggered:
            # Check for trigger
            if intensity > self.config.intensity_threshold:
                self.triggered = True
                self.trigger_time = timestamp
                self.trigger_idx = max_idx
                self.trigger_intensity = intensity
                self.post_samples = [(timestamp, measurements.copy())]
        else:
            # Accumulate post-trigger samples
            self.post_samples.append((timestamp, measurements.copy()))

            # Check if we have enough post-trigger samples
            elapsed_ms = (timestamp - self.trigger_time) * 1000
            if elapsed_ms >= self.config.post_trigger_ms:
                # Create event
                event = self._finalize_event()
                self.triggered = False
                self.post_samples = []
                return event

        return None

    def _finalize_event(self) -> Event:
        """Finalize captured event with signature."""
        # Combine pre and post buffers
        all_samples = list(self.pre_buffer) + self.post_samples

        timestamps = np.array([s[0] for s in all_samples])
        waveforms = np.array([s[1][:, 0] for s in all_samples])  # z-scores

        # Duration
        duration_ms = (timestamps[-1] - timestamps[0]) * 1000

        # Spatial snapshot at trigger
        trigger_idx = len(self.pre_buffer)
        spatial = self.array.get_spatial_map(all_samples[trigger_idx][1])

        # Create event
        event = EventTracer.Event(
            timestamp=self.trigger_time,
            trigger_ossicle=self.trigger_idx,
            intensity=self.trigger_intensity,
            duration_ms=duration_ms,
            waveform=waveforms,
            spatial_snapshot=spatial
        )

        # Sign if enabled
        if self.config.sign_events:
            event.signature = self._sign_event(event)

        # Store
        if len(self.events) >= self.config.max_events:
            self.events.pop(0)
        self.events.append(event)

        return event

    def _sign_event(self, event: Event) -> str:
        """Create cryptographic signature for event."""
        # Create canonical representation (ensure native Python types)
        data = {
            'timestamp': float(event.timestamp),
            'trigger_ossicle': int(event.trigger_ossicle),
            'intensity': float(event.intensity),
            'duration_ms': float(event.duration_ms),
            'waveform_hash': hashlib.sha256(event.waveform.tobytes()).hexdigest(),
            'spatial_hash': hashlib.sha256(event.spatial_snapshot.tobytes()).hexdigest()
        }

        # Create signature (in production, use proper signing key)
        canonical = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(canonical.encode()).hexdigest()

        return signature

    def get_events_above(self, intensity: float) -> List[Event]:
        """Get all events above specified intensity."""
        return [e for e in self.events if e.intensity >= intensity]

    def export_events(self, path: str):
        """Export events to JSON file."""
        events_data = []
        for e in self.events:
            events_data.append({
                'timestamp': e.timestamp,
                'trigger_ossicle': e.trigger_ossicle,
                'intensity': float(e.intensity),
                'duration_ms': float(e.duration_ms),
                'signature': e.signature,
                'waveform_shape': list(e.waveform.shape),
                'spatial_shape': list(e.spatial_snapshot.shape)
            })

        with open(path, 'w') as f:
            json.dump(events_data, f, indent=2)


class FogChamber:
    """
    Fog chamber visualization mode.

    Displays wave propagation like particles in a cloud chamber,
    showing entropy wave tracks as they pass through the array.
    """

    def __init__(self, array: ArrayController):
        self.array = array
        self.history_frames = 30
        self.history = deque(maxlen=self.history_frames)
        self.tracks = []  # Detected wave tracks
        self.decay_rate = 0.85  # How fast old detections fade

    def update(self, measurements: np.ndarray) -> np.ndarray:
        """Update fog chamber with new measurements, return visualization."""
        spatial = self.array.get_spatial_map(measurements, metric=0)  # z-score

        # Add to history
        self.history.append(spatial.copy())

        # Create fog chamber image (accumulated with decay)
        fog = np.zeros_like(spatial)
        for i, frame in enumerate(self.history):
            age = len(self.history) - i - 1
            weight = self.decay_rate ** age
            fog += frame * weight

        # Normalize
        fog = fog / (fog.max() + 1e-10)

        # Detect tracks (local maxima that persist)
        self._detect_tracks(spatial)

        return fog

    def _detect_tracks(self, current: np.ndarray):
        """Detect wave tracks from spatial gradient."""
        if len(self.history) < 3:
            return

        # Get recent frames
        prev = self.history[-2]

        # Compute motion (gradient between frames)
        motion = current - prev

        # Find significant motion regions
        threshold = np.std(motion) * 2
        hot_spots = np.where(np.abs(motion) > threshold)

        for row, col in zip(*hot_spots):
            self.tracks.append({
                'position': (row, col),
                'intensity': float(motion[row, col]),
                'timestamp': time.time()
            })

        # Prune old tracks
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t['timestamp'] < 1.0]

    def get_track_overlay(self) -> List[dict]:
        """Get track data for overlay visualization."""
        return self.tracks

    def render_ascii(self, fog: np.ndarray) -> str:
        """Render fog chamber as ASCII art."""
        chars = ' .:-=+*#%@'
        lines = []
        for row in fog:
            line = ''
            for val in row:
                idx = int(val * (len(chars) - 1))
                idx = max(0, min(idx, len(chars) - 1))
                line += chars[idx]
            lines.append(line)
        return '\n'.join(lines)


class Monitor:
    """Real-time monitoring with multiple views."""

    def __init__(self, array: ArrayController, config: MonitorConfig):
        self.array = array
        self.config = config

        # History buffers
        buffer_size = int(config.history_seconds * array.params.sample_rate_hz)
        self.z_history = deque(maxlen=buffer_size)
        self.spatial_history = deque(maxlen=int(config.update_rate_hz * config.history_seconds))

        # Statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_samples = 0

        # Callbacks
        self.on_alert: Optional[Callable[[float, int], None]] = None

    def update(self, measurements: np.ndarray) -> dict:
        """Update monitor with new measurements, return status dict."""
        z_scores = measurements[:, 0]
        max_z = z_scores.max()
        max_idx = z_scores.argmax()
        mean_z = z_scores.mean()

        # Update history
        self.z_history.append((time.time(), z_scores.copy()))

        if self.config.show_spatial:
            spatial = self.array.get_spatial_map(measurements)
            self.spatial_history.append(spatial)

        # Update running statistics
        self.n_samples += 1
        delta = max_z - self.running_mean
        self.running_mean += delta / self.n_samples
        self.running_var += (delta * (max_z - self.running_mean) - self.running_var) / self.n_samples

        # Check alert threshold
        if self.n_samples > 10:
            std = np.sqrt(self.running_var + 1e-10)
            sigma = (max_z - self.running_mean) / std

            if sigma > self.config.threshold_sigma and self.on_alert:
                self.on_alert(sigma, max_idx)

        return {
            'max_z': float(max_z),
            'max_idx': int(max_idx),
            'mean_z': float(mean_z),
            'timestamp': time.time(),
            'n_samples': self.n_samples
        }

    def get_waveform(self, ossicle_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get waveform history for an ossicle (or max if None)."""
        if not self.z_history:
            return np.array([]), np.array([])

        times = np.array([h[0] for h in self.z_history])
        if ossicle_idx is not None:
            values = np.array([h[1][ossicle_idx] for h in self.z_history])
        else:
            values = np.array([h[1].max() for h in self.z_history])

        return times - times[0], values

    def get_spectrum(self, ossicle_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency spectrum of recent history."""
        _, values = self.get_waveform(ossicle_idx)
        if len(values) < 16:
            return np.array([]), np.array([])

        # FFT
        spectrum = np.abs(np.fft.rfft(values))
        freqs = np.fft.rfftfreq(len(values), 1.0 / self.array.params.sample_rate_hz)

        return freqs, spectrum


# =============================================================================
# MAIN RUNTIME
# =============================================================================

class OssicleRuntime:
    """
    Main runtime orchestrating all components.

    Usage:
        runtime = OssicleRuntime()
        runtime.configure_array(n_rows=8, n_cols=16)
        runtime.set_mode(RuntimeMode.MONITOR)
        runtime.start()

        # In loop or callback:
        status = runtime.step()
    """

    def __init__(self):
        # Configuration
        self.array_params = ArrayParams()
        self.ossicle_params = OssicleParams()
        self.beam_params = BeamParams()
        self.monitor_config = MonitorConfig()
        self.event_config = EventConfig()

        # Components (created on configure)
        self.array: Optional[ArrayController] = None
        self.beamformer: Optional[Beamformer] = None
        self.tx_rx: Optional[TransmitReceive] = None
        self.monitor: Optional[Monitor] = None
        self.event_tracer: Optional[EventTracer] = None
        self.fog_chamber: Optional[FogChamber] = None

        # State
        self.mode = RuntimeMode.IDLE
        self.running = False
        self.sample_count = 0
        self.start_time = 0.0

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._data_queue = queue.Queue(maxsize=100)

        # Callbacks
        self.on_measurement: Optional[Callable[[np.ndarray, dict], None]] = None
        self.on_event: Optional[Callable[[EventTracer.Event], None]] = None
        self.on_mode_change: Optional[Callable[[RuntimeMode], None]] = None

    def configure_array(self, n_rows: int = 8, n_cols: int = 16,
                        spacing_mm: float = 2.5, sample_rate_hz: float = 2000):
        """Configure array geometry."""
        self.array_params = ArrayParams(
            n_rows=n_rows,
            n_cols=n_cols,
            spacing_mm=spacing_mm,
            sample_rate_hz=sample_rate_hz
        )
        self._build_components()

    def configure_ossicles(self, r_base: float = 3.70, r_spacing: float = 0.03,
                           twist_deg: float = 1.1, coupling: float = 0.05,
                           n_cells: int = 64, iterations: int = 100):
        """Configure ossicle parameters."""
        self.ossicle_params = OssicleParams(
            r_base=r_base,
            r_spacing=r_spacing,
            twist_deg=twist_deg,
            coupling=coupling,
            n_cells=n_cells,
            iterations=iterations
        )
        if self.array:
            self.array.set_global_params(self.ossicle_params)

    def _build_components(self):
        """Build runtime components."""
        self.array = ArrayController(self.array_params, self.ossicle_params)
        self.beamformer = Beamformer(self.array)
        self.tx_rx = TransmitReceive(self.array)
        self.monitor = Monitor(self.array, self.monitor_config)
        self.event_tracer = EventTracer(self.array, self.event_config)
        self.fog_chamber = FogChamber(self.array)

    def calibrate(self, n_samples: int = 100):
        """Calibrate all components."""
        if not self.array:
            raise RuntimeError("Array not configured")

        print(f"Calibrating with {n_samples} samples...")
        self.array.calibrate(n_samples)
        self.event_tracer.calibrate(n_samples)
        print("Calibration complete")

    def set_mode(self, mode: RuntimeMode):
        """Set operating mode."""
        old_mode = self.mode
        self.mode = mode

        # Mode-specific setup
        if mode == RuntimeMode.TRANSMIT:
            if self.tx_rx.tx_pattern is None:
                self.tx_rx.set_tx_pattern('pulse', center=self.array_params.n_ossicles // 2)
            self.tx_rx.transmit()
        elif old_mode == RuntimeMode.TRANSMIT:
            self.tx_rx.stop_transmit()

        if self.on_mode_change:
            self.on_mode_change(mode)

    def step(self) -> dict:
        """Take one measurement step, process according to mode."""
        if not self.array:
            raise RuntimeError("Array not configured")

        timestamp = time.time()
        measurements = self.array.measure()
        self.sample_count += 1

        result = {
            'timestamp': timestamp,
            'sample': self.sample_count,
            'mode': self.mode.name,
            'measurements': measurements
        }

        # Mode-specific processing
        if self.mode == RuntimeMode.MONITOR:
            result['monitor'] = self.monitor.update(measurements)
            event = self.event_tracer.process_sample(measurements, timestamp)
            if event:
                result['event'] = event
                if self.on_event:
                    self.on_event(event)

        elif self.mode == RuntimeMode.BEAMFORM:
            result['monitor'] = self.monitor.update(measurements)
            result['beam_output'] = complex(self.beamformer.apply(measurements))

        elif self.mode == RuntimeMode.RECEIVE:
            result['monitor'] = self.monitor.update(measurements)
            result['matched_output'] = self.tx_rx.receive_matched(measurements)

        elif self.mode == RuntimeMode.TX_RX:
            result['monitor'] = self.monitor.update(measurements)
            result['matched_output'] = self.tx_rx.receive_matched(measurements)

        elif self.mode == RuntimeMode.FOG_CHAMBER:
            result['monitor'] = self.monitor.update(measurements)
            result['fog'] = self.fog_chamber.update(measurements)
            result['tracks'] = self.fog_chamber.get_track_overlay()

        elif self.mode == RuntimeMode.CALIBRATE:
            # Accumulate calibration samples
            pass

        if self.on_measurement:
            self.on_measurement(measurements, result)

        return result

    def start(self, threaded: bool = False):
        """Start the runtime."""
        if not self.array:
            raise RuntimeError("Array not configured")

        self.running = True
        self.start_time = time.time()
        self._stop_event.clear()

        if threaded:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the runtime."""
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_loop(self):
        """Main run loop for threaded mode."""
        sample_interval = 1.0 / self.array_params.sample_rate_hz

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            result = self.step()

            # Queue result for external consumers
            try:
                self._data_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop if queue full

            # Maintain sample rate
            elapsed = time.perf_counter() - loop_start
            if elapsed < sample_interval:
                time.sleep(sample_interval - elapsed)

    def get_data(self, timeout: float = 0.1) -> Optional[dict]:
        """Get data from threaded runtime."""
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def steer_beam(self, azimuth: float, elevation: float = 0.0):
        """Steer beamformer."""
        if self.beamformer:
            self.beamformer.steer(azimuth, elevation)

    def scan_beam(self, az_range: Tuple[float, float] = (-90, 90),
                  n_points: int = 37) -> Tuple[np.ndarray, np.ndarray]:
        """Perform beam scan."""
        if not self.beamformer:
            raise RuntimeError("Beamformer not configured")

        measurements = self.array.measure()
        return self.beamformer.scan(measurements, az_range, n_points)

    def set_tx_pattern(self, pattern: str, **kwargs):
        """Set transmit pattern."""
        if self.tx_rx:
            self.tx_rx.set_tx_pattern(pattern, **kwargs)

    def get_events(self, min_intensity: float = 0.0) -> List[EventTracer.Event]:
        """Get captured events."""
        if self.event_tracer:
            return self.event_tracer.get_events_above(min_intensity)
        return []

    def export_events(self, path: str):
        """Export events to file."""
        if self.event_tracer:
            self.event_tracer.export_events(path)

    def get_fog_ascii(self) -> str:
        """Get fog chamber ASCII visualization."""
        if not self.fog_chamber or not self.fog_chamber.history:
            return "No data"

        fog = self.fog_chamber.update(self.array.measure())
        return self.fog_chamber.render_ascii(fog)

    def get_status(self) -> dict:
        """Get runtime status."""
        return {
            'mode': self.mode.name,
            'running': self.running,
            'sample_count': self.sample_count,
            'uptime': time.time() - self.start_time if self.running else 0,
            'array': {
                'n_ossicles': self.array_params.n_ossicles,
                'sample_rate': self.array_params.sample_rate_hz,
                'total_bandwidth': self.array_params.total_bandwidth
            },
            'events_captured': len(self.event_tracer.events) if self.event_tracer else 0
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Interactive CLI for the runtime."""
    print("="*60)
    print("OSSICLE ARRAY RUNTIME")
    print("="*60)
    print()

    # Create runtime
    runtime = OssicleRuntime()
    runtime.configure_array(n_rows=8, n_cols=16)
    runtime.calibrate(n_samples=50)

    print(f"\nArray configured: {runtime.array_params.n_ossicles} ossicles")
    print(f"Total bandwidth: {runtime.array_params.total_bandwidth/1e6:.2f} M samples/sec")
    print()

    # Demo each mode
    modes_demo = [
        (RuntimeMode.MONITOR, "Passive monitoring"),
        (RuntimeMode.BEAMFORM, "Beamforming"),
        (RuntimeMode.FOG_CHAMBER, "Fog chamber visualization"),
    ]

    for mode, description in modes_demo:
        print(f"\n--- {description} ({mode.name}) ---")
        runtime.set_mode(mode)

        for i in range(5):
            result = runtime.step()

            if mode == RuntimeMode.MONITOR:
                mon = result.get('monitor', {})
                print(f"  Sample {i+1}: max_z={mon.get('max_z', 0):.2f}, "
                      f"idx={mon.get('max_idx', 0)}")

            elif mode == RuntimeMode.BEAMFORM:
                bf = result.get('beam_output', 0)
                print(f"  Sample {i+1}: beam_power={abs(bf)**2:.2f}")

            elif mode == RuntimeMode.FOG_CHAMBER:
                if i == 4:  # Show fog on last sample
                    print(runtime.get_fog_ascii())

    print("\n" + "="*60)
    print("Runtime demo complete")
    print("="*60)

    return runtime


if __name__ == "__main__":
    main()
