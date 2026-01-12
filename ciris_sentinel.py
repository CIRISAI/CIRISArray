#!/usr/bin/env python3
"""
CIRIS Sentinel: GPU Workload Detector & TRNG
=============================================

V3.0 VALIDATED (Jan 2026): Mean-shift detection validated in A5/O1-O7.

VALIDATED ARCHITECTURE (A5, A6, A9, B1e):
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL GPU SENSOR                           │
│                                                                     │
│   GPU Kernel Timing (4000 Hz sample rate)                           │
│           │                                                         │
│           ├──► Mean Shift ──────► WORKLOAD (+2519% at 50% load)     │
│           ├──► Thermal Band ────► TEMPERATURE (0-0.1 Hz, 79.1%)     │
│           ├──► Workload Band ───► TRANSIENTS (100-500 Hz, 7.5%)     │
│           └──► Lower 4 LSBs ────► TRNG (7.99 bits/byte)             │
│                                                                     │
│   DETECTION: mean_shift > 50% (not variance ratio!)                 │
│   Workload causes GPU contention → timing TRIPLES                   │
└─────────────────────────────────────────────────────────────────────┘

VALIDATED SPECS (A5/O1-O7, Jan 2026):
- Sample rate: 4000 Hz (avoid 1900-2100 Hz interference)
- Detection latency: 1.3 ms (A5) / 2.5 ms (O4)
- Detection floor: 1% workload (+83% mean shift)
- Cross-sensor CV: 8.2% (A6)
- SNR scaling: β=0.47 ≈ √N (A9)
- 16-sensor improvement: 5.1x

DETECTION METHOD:
- OLD (WRONG): dk_dt + 3σ threshold, variance ratio
- NEW (VALIDATED): mean_shift > 50%
- At 1% workload: +83% mean shift
- At 50% workload: +2519% mean shift

TRNG (exp56-57):
- 259 kbps true random from GPU timing LSBs
- 99.5% Shannon entropy
- Independent of PRNG seed

Author: CIRIS Research Team
Date: January 2026
License: BSL 1.1
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import time
import threading
import queue
import socket
import struct
import json
import os

PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# CROSS-DEVICE TIMING INFRASTRUCTURE
# =============================================================================
# Provides sub-millisecond synchronized timestamps between main GPU and
# secondary device (e.g., Jetson Nano) WITHOUT adding overhead to measurements.
#
# Architecture:
#   1. CALIBRATION PHASE: UDP round-trip measures offset (run once/periodically)
#   2. MEASUREMENT PHASE: Apply stored offset to raw timestamps (zero network cost)
#
# Usage:
#   timing = TimingSync("192.168.50.203")
#   timing.calibrate()  # Once, before experiment
#
#   # During measurement - no network overhead:
#   local_ts = time.time()
#   remote_ts = timing.local_to_remote(local_ts)
# =============================================================================

TIMING_CALIBRATION_FILE = "/home/emoore/CIRISArray/timing_calibration.json"
TIMING_UDP_PORT = 37123


@dataclass
class TimingCalibration:
    """Stored timing calibration between two hosts."""
    local_host: str
    remote_host: str
    offset_seconds: float      # remote - local (positive = remote ahead)
    offset_std: float          # uncertainty (1σ)
    rtt_mean_ms: float         # mean round-trip time
    rtt_min_ms: float          # minimum RTT (best precision indicator)
    n_samples: int
    calibration_time: float    # when calibrated (local time)

    def to_dict(self) -> dict:
        return {
            'local_host': self.local_host,
            'remote_host': self.remote_host,
            'offset_seconds': self.offset_seconds,
            'offset_std': self.offset_std,
            'rtt_mean_ms': self.rtt_mean_ms,
            'rtt_min_ms': self.rtt_min_ms,
            'n_samples': self.n_samples,
            'calibration_time': self.calibration_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TimingCalibration':
        return cls(**d)

    def save(self, path: str = TIMING_CALIBRATION_FILE):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = TIMING_CALIBRATION_FILE) -> 'TimingCalibration':
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def precision_ms(self) -> float:
        """Estimated precision in milliseconds."""
        return self.offset_std * 1000

    @property
    def precision_us(self) -> float:
        """Estimated precision in microseconds."""
        return self.offset_std * 1e6

    @property
    def age_seconds(self) -> float:
        """How old is this calibration."""
        return time.time() - self.calibration_time


class TimingSync:
    """
    Cross-device timing synchronization.

    Provides synchronized timestamps between local machine and remote device
    (e.g., Jetson Nano) with sub-millisecond precision over WiFi.

    CRITICAL: Calibrate BEFORE experiments, then use offset during measurement.
    No network traffic during actual measurements.

    Example:
        timing = TimingSync("192.168.50.203")

        # Calibrate once (takes ~2 seconds)
        timing.calibrate(n_samples=100)
        print(f"Precision: ±{timing.precision_ms:.3f}ms")

        # During experiment - zero overhead:
        local_ts = time.time()
        remote_ts = timing.local_to_remote(local_ts)

        # Compare measurements from different devices:
        jetson_time = 1234567890.123  # From Jetson
        local_equivalent = timing.remote_to_local(jetson_time)
    """

    def __init__(self, remote_host: str, port: int = TIMING_UDP_PORT,
                 auto_load: bool = True):
        """
        Initialize timing sync.

        Args:
            remote_host: IP or hostname of secondary device
            port: UDP port for time protocol
            auto_load: If True, load existing calibration if available
        """
        self.remote_host = remote_host
        self.port = port
        self.calibration: Optional[TimingCalibration] = None
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()

        # Get local hostname
        import subprocess
        result = subprocess.run(['hostname'], capture_output=True, text=True)
        self.local_host = result.stdout.strip()

        # Try to load existing calibration
        if auto_load:
            try:
                self.calibration = TimingCalibration.load()
                if self.calibration.remote_host == remote_host:
                    print(f"[TimingSync] Loaded calibration: offset={self.calibration.offset_seconds*1000:.3f}ms, "
                          f"age={self.calibration.age_seconds/60:.1f}min")
                else:
                    print(f"[TimingSync] Stored calibration is for different host, ignoring")
                    self.calibration = None
            except FileNotFoundError:
                pass

    def _get_single_sample(self, timeout: float = 1.0) -> Tuple[float, float]:
        """
        Get single time sample via UDP.

        Returns:
            (offset, rtt) in seconds
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)

        try:
            # Send request with local timestamp
            t1 = time.time()
            sock.sendto(b'TIME', (self.remote_host, self.port))

            # Receive response
            data, _ = sock.recvfrom(1024)
            t3 = time.time()

            # Unpack server timestamp
            t2 = struct.unpack('!d', data)[0]

            # NTP-style offset calculation
            rtt = t3 - t1
            offset = t2 - (t1 + t3) / 2

            return offset, rtt
        finally:
            sock.close()

    def calibrate(self, n_samples: int = 100, verbose: bool = True) -> TimingCalibration:
        """
        Calibrate timing offset to remote host.

        Uses UDP round-trip for sub-millisecond precision.
        Run this BEFORE experiments to establish offset.

        Args:
            n_samples: Number of samples (more = better precision)
            verbose: Print progress

        Returns:
            TimingCalibration object (also stored in self.calibration)
        """
        if verbose:
            print(f"[TimingSync] Calibrating to {self.remote_host}:{self.port}...")
            print(f"[TimingSync] Taking {n_samples} samples...")

        offsets = []
        rtts = []
        failures = 0

        for i in range(n_samples):
            try:
                offset, rtt = self._get_single_sample()
                offsets.append(offset)
                rtts.append(rtt)

                if verbose and (i + 1) % 25 == 0:
                    print(f"  {i+1}/{n_samples}: offset={offset*1000:.3f}ms, RTT={rtt*1000:.2f}ms")

            except socket.timeout:
                failures += 1
                if failures > n_samples // 2:
                    raise RuntimeError(f"Too many timeouts - is UDP server running on {self.remote_host}?")

            time.sleep(0.01)  # 10ms between samples

        if len(offsets) < 3:
            raise RuntimeError("Not enough successful samples")

        offsets = np.array(offsets)
        rtts = np.array(rtts)

        # Robust statistics with outlier rejection
        offset_median = np.median(offsets)
        offset_std = np.std(offsets)

        # Filter outliers (> 2σ from median)
        mask = np.abs(offsets - offset_median) < 2 * offset_std
        offsets_clean = offsets[mask]
        rtts_clean = rtts[mask]

        self.calibration = TimingCalibration(
            local_host=self.local_host,
            remote_host=self.remote_host,
            offset_seconds=float(np.mean(offsets_clean)),
            offset_std=float(np.std(offsets_clean)),
            rtt_mean_ms=float(np.mean(rtts_clean) * 1000),
            rtt_min_ms=float(np.min(rtts_clean) * 1000),
            n_samples=len(offsets_clean),
            calibration_time=time.time(),
        )

        # Save to file
        self.calibration.save()

        if verbose:
            print(f"\n[TimingSync] CALIBRATION COMPLETE")
            print(f"  Offset:    {self.calibration.offset_seconds*1000:+.3f} ms")
            print(f"  Precision: ±{self.calibration.offset_std*1000:.3f} ms ({self.calibration.offset_std*1e6:.0f} µs)")
            print(f"  RTT:       {self.calibration.rtt_mean_ms:.2f} ms (min: {self.calibration.rtt_min_ms:.2f} ms)")
            print(f"  Samples:   {self.calibration.n_samples}")

        return self.calibration

    def local_to_remote(self, local_time: float) -> float:
        """
        Convert local timestamp to remote (Jetson) time.

        ZERO OVERHEAD - just adds stored offset.
        """
        if self.calibration is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")
        return local_time + self.calibration.offset_seconds

    def remote_to_local(self, remote_time: float) -> float:
        """
        Convert remote (Jetson) timestamp to local time.

        ZERO OVERHEAD - just subtracts stored offset.
        """
        if self.calibration is None:
            raise RuntimeError("Not calibrated. Call calibrate() first.")
        return remote_time - self.calibration.offset_seconds

    @property
    def offset_ms(self) -> float:
        """Current offset in milliseconds."""
        if self.calibration is None:
            return 0.0
        return self.calibration.offset_seconds * 1000

    @property
    def precision_ms(self) -> float:
        """Precision in milliseconds (1σ)."""
        if self.calibration is None:
            return float('inf')
        return self.calibration.offset_std * 1000

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration exists."""
        return self.calibration is not None

    def start_background_recalibration(self, interval_seconds: float = 300):
        """
        Start background thread that re-calibrates periodically.

        Useful for long experiments where clock drift matters.
        Does NOT affect ongoing measurements - just updates offset.

        Args:
            interval_seconds: How often to recalibrate (default: 5 minutes)
        """
        if self._background_thread is not None:
            print("[TimingSync] Background thread already running")
            return

        self._stop_background.clear()

        def recalibrate_loop():
            while not self._stop_background.wait(interval_seconds):
                try:
                    self.calibrate(n_samples=50, verbose=False)
                except Exception as e:
                    print(f"[TimingSync] Background recalibration failed: {e}")

        self._background_thread = threading.Thread(target=recalibrate_loop, daemon=True)
        self._background_thread.start()
        print(f"[TimingSync] Background recalibration started (every {interval_seconds}s)")

    def stop_background_recalibration(self):
        """Stop background recalibration thread."""
        if self._background_thread is None:
            return
        self._stop_background.set()
        self._background_thread.join(timeout=2)
        self._background_thread = None
        print("[TimingSync] Background recalibration stopped")

    def __repr__(self):
        if self.calibration:
            return (f"TimingSync({self.remote_host}, offset={self.offset_ms:+.3f}ms, "
                    f"precision=±{self.precision_ms:.3f}ms)")
        return f"TimingSync({self.remote_host}, NOT CALIBRATED)"


def run_timing_server(port: int = TIMING_UDP_PORT):
    """
    Run UDP time server (run this on Jetson).

    Responds to time requests with current timestamp.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))

    print(f"[TimingServer] Listening on UDP port {port}")
    print("[TimingServer] Press Ctrl+C to stop")

    try:
        while True:
            data, addr = sock.recvfrom(1024)

            # Respond immediately with current time
            t_server = time.time()
            response = struct.pack('!d', t_server)
            sock.sendto(response, addr)

    except KeyboardInterrupt:
        print("\n[TimingServer] Stopped")
    finally:
        sock.close()


MAGIC_ANGLE = 1.1
COUPLING_FACTOR = 0.0003

# Lorenz parameters (standard chaotic regime)
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0


@dataclass
class SentinelConfig:
    """Configuration for a single sentinel sensor.

    V2.1 RESONATOR MODEL (exp63-64):
    - System is a BANDPASS FILTER on chaos, not a 70 Hz oscillator
    - Q factor = 10.25, resonance = 70.7 Hz, bandwidth 40-80 Hz
    - Lyapunov = +0.178 per step (CHAOTIC confirmed)
    - 61.5% of signal is DC/drift, 35% mid-frequency chaos

    CRITICAL: All oscillators are 100% coherent!
    - Entropy is TEMPORAL only (GPU timing)
    - For independent entropy: use different GPUs (e.g., Jetson)
    - Negentropic detection: AVOID 70 Hz (asymmetry inverts!)

    Detection bandwidth (exp64):
    - Sensitive: 40-80 Hz (+20 to +27 dB gain)
    - Peak: 70 Hz (+27.5 dB)
    - Cutoff: 80 Hz (sharp rolloff)
    - Best for negentropy: < 50 Hz

    Lorenz parameters (standard chaotic regime):
    - sigma=10.0, rho=28.0, beta=8/3
    - dt=0.01 with RK4 integration for stability
    - timing_sensitivity: how much timing affects rho

    Thermal sensing (validated Jan 2026):
    - Variance correlates with temperature at r=-0.97
    - Works via DC/drift component (61.5% of signal)
    - Use variance (not k_eff) for thermal detection

    TRNG mode:
    - 259 kbps true random bits
    - 99.5% Shannon entropy
    - Use GPUATRNG class for random bytes
    """
    # Core array parameters
    n_ossicles: int = 256         # Minimal array
    oscillator_depth: int = 32    # Reduced depth for speed

    # V2.1: Lorenz chaotic dynamics (TRUE chaos!)
    use_lorenz: bool = True       # Use Lorenz (True) or legacy linear (False)
    lorenz_sigma: float = 10.0    # Lorenz σ parameter
    lorenz_rho: float = 28.0      # Lorenz ρ parameter (base value)
    lorenz_beta: float = 8.0/3.0  # Lorenz β parameter
    lorenz_dt: float = 0.025      # VALIDATED: 0.025 = critical point (Exp 114)
    timing_sensitivity: float = 0.001  # How much timing jitter affects ρ

    # Resonator characteristics (exp63-64 validated)
    resonance_freq_hz: float = 70.7    # Natural resonance frequency
    q_factor: float = 10.25            # Quality factor (ceramic-like)
    bandwidth_low_hz: float = 40.0     # -3dB lower bound
    bandwidth_high_hz: float = 80.0    # -3dB upper bound (sharp cutoff!)
    negentropic_avoid_hz: float = 70.0 # Avoid this freq for negentropy detection!

    # Legacy linear coupling parameters (use_lorenz=False)
    epsilon: float = 0.003        # OPTIMAL: crossover point (25x signal, τ=12.8s)
    noise_amplitude: float = 0.001  # OPTIMAL: SR peak (was 0.02, too high!)

    # Sampling parameters (A5/O1-O7 VALIDATED Jan 2026)
    sample_rate_hz: float = 4000  # VALIDATED: 4000 Hz optimal (avoid 1900-2100 Hz)
    derivative_window: int = 5    # Samples for derivative calc
    reset_interval_s: float = 23.0  # τ/2 = 46.1/2, optimal reset cycle
    tau_thermalization: float = 46.1  # Validated decay constant
    noise_floor_sigma: float = 0.003  # Validated noise floor

    # MEAN SHIFT DETECTION (A5/O1-O7 VALIDATED Jan 2026)
    # Primary detection method: timing mean shift > 50%
    # Workload causes GPU contention → timing TRIPLES (~7μs → ~21μs)
    mean_shift_threshold_pct: float = 50.0  # VALIDATED: >50% mean shift = detection
    detection_window_samples: int = 40  # ~10ms at 4000 Hz (O4: 2.5ms latency)

    # Thermal sensing parameters
    thermal_window: int = 50      # Samples for thermal baseline
    thermal_threshold_sigma: float = 3.0  # Detection threshold

    # r_ab sensitivity regime parameters (validated Jan 2026)
    r_ab_reset_threshold: float = 0.98  # Reset when r_ab exceeds this
    r_ab_sensitive_threshold: float = 0.95  # Below this = highly sensitive
    use_r_ab_reset: bool = True   # Enable r_ab-based reset (vs time-based)


# Optimized CUDA kernel for minimal array
_sentinel_kernel = cp.RawKernel(r'''
extern "C" __global__
void sentinel_step(
    float* osc_a, float* osc_b, float* osc_c,
    float* noise,
    float coupling_ab, float coupling_bc, float coupling_ca,
    float noise_amp,
    int n, int iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = osc_a[idx];
    float b = osc_b[idx];
    float c = osc_c[idx];

    // Dynamics
    for (int i = 0; i < iters; i++) {
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

    // Inject noise to maintain transient
    a += noise[idx] * noise_amp;
    b += noise[idx + n] * noise_amp;
    c += noise[idx + 2*n] * noise_amp;

    osc_a[idx] = a;
    osc_b[idx] = b;
    osc_c[idx] = c;
}

extern "C" __global__
void fast_correlate(
    float* a, float* b, float* result,
    int n
) {
    // Simple correlation estimator using shared memory
    __shared__ float sum_a, sum_b, sum_ab, sum_a2, sum_b2;

    if (threadIdx.x == 0) {
        sum_a = sum_b = sum_ab = sum_a2 = sum_b2 = 0.0f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float va = a[idx];
        float vb = b[idx];

        atomicAdd(&sum_a, va);
        atomicAdd(&sum_b, vb);
        atomicAdd(&sum_ab, va * vb);
        atomicAdd(&sum_a2, va * va);
        atomicAdd(&sum_b2, vb * vb);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean_a = sum_a / n;
        float mean_b = sum_b / n;
        float var_a = sum_a2 / n - mean_a * mean_a;
        float var_b = sum_b2 / n - mean_b * mean_b;
        float cov = sum_ab / n - mean_a * mean_b;

        float denom = sqrtf(var_a * var_b);
        *result = (denom > 1e-10f) ? cov / denom : 0.0f;
    }
}
''', 'sentinel_step')

# Lorenz CUDA kernel with RK4 integration (V2 upgrade)
# TRUE CHAOS: positive Lyapunov exponent (+0.007)
# Timing-coupled: rho modulated by GPU timing jitter
_lorenz_kernel = cp.RawKernel(r'''
extern "C" __global__
void lorenz_step(
    float* osc_x, float* osc_y, float* osc_z,
    float sigma, float rho_base, float beta, float dt,
    float timing_perturbation,  // From GPU timing jitter
    int n, int iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = osc_x[idx];
    float y = osc_y[idx];
    float z = osc_z[idx];

    // Timing-modulated rho (entropy injection point)
    // Small timing variations create large trajectory divergence
    float rho = rho_base + timing_perturbation;

    // RK4 integration for stability (from exp60)
    for (int i = 0; i < iters; i++) {
        // k1
        float dx1 = sigma * (y - x);
        float dy1 = x * (rho - z) - y;
        float dz1 = x * y - beta * z;

        // k2
        float x2 = x + dx1 * dt / 2;
        float y2 = y + dy1 * dt / 2;
        float z2 = z + dz1 * dt / 2;
        float dx2 = sigma * (y2 - x2);
        float dy2 = x2 * (rho - z2) - y2;
        float dz2 = x2 * y2 - beta * z2;

        // k3
        float x3 = x + dx2 * dt / 2;
        float y3 = y + dy2 * dt / 2;
        float z3 = z + dz2 * dt / 2;
        float dx3 = sigma * (y3 - x3);
        float dy3 = x3 * (rho - z3) - y3;
        float dz3 = x3 * y3 - beta * z3;

        // k4
        float x4 = x + dx3 * dt;
        float y4 = y + dy3 * dt;
        float z4 = z + dz3 * dt;
        float dx4 = sigma * (y4 - x4);
        float dy4 = x4 * (rho - z4) - y4;
        float dz4 = x4 * y4 - beta * z4;

        // Update
        x += (dx1 + 2*dx2 + 2*dx3 + dx4) * dt / 6;
        y += (dy1 + 2*dy2 + 2*dy3 + dy4) * dt / 6;
        z += (dz1 + 2*dz2 + 2*dz3 + dz4) * dt / 6;

        // Soft bound (Lorenz attractor is bounded naturally)
        x = fmaxf(-50.0f, fminf(50.0f, x));
        y = fmaxf(-50.0f, fminf(50.0f, y));
        z = fmaxf(0.0f, fminf(100.0f, z));  // z is always positive
    }

    osc_x[idx] = x;
    osc_y[idx] = y;
    osc_z[idx] = z;
}
''', 'lorenz_step')


_correlate_kernel = cp.RawModule(code=r'''
extern "C" __global__
void fast_stats(
    float* a, float* b,
    float* out_r, float* out_var,
    int n
) {
    // Compute correlation and variance in one pass
    float sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float va = a[i];
        float vb = b[i];
        sum_a += va;
        sum_b += vb;
        sum_ab += va * vb;
        sum_a2 += va * va;
        sum_b2 += vb * vb;
    }

    // Reduce within block
    __shared__ float s_sum_a[256], s_sum_b[256], s_sum_ab[256], s_sum_a2[256], s_sum_b2[256];

    s_sum_a[threadIdx.x] = sum_a;
    s_sum_b[threadIdx.x] = sum_b;
    s_sum_ab[threadIdx.x] = sum_ab;
    s_sum_a2[threadIdx.x] = sum_a2;
    s_sum_b2[threadIdx.x] = sum_b2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_a[threadIdx.x] += s_sum_a[threadIdx.x + s];
            s_sum_b[threadIdx.x] += s_sum_b[threadIdx.x + s];
            s_sum_ab[threadIdx.x] += s_sum_ab[threadIdx.x + s];
            s_sum_a2[threadIdx.x] += s_sum_a2[threadIdx.x + s];
            s_sum_b2[threadIdx.x] += s_sum_b2[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean_a = s_sum_a[0] / n;
        float mean_b = s_sum_b[0] / n;
        float var_a = s_sum_a2[0] / n - mean_a * mean_a;
        float var_b = s_sum_b2[0] / n - mean_b * mean_b;
        float cov = s_sum_ab[0] / n - mean_a * mean_b;

        float denom = sqrtf(var_a * var_b);
        *out_r = (denom > 1e-10f) ? cov / denom : 0.0f;
        *out_var = var_a + var_b;
    }
}
''').get_function('fast_stats')


class Sentinel:
    """
    Single minimal entropy wave sensor.

    V2 LORENZ UPGRADE:
    - TRUE CHAOS: Lorenz attractor (Lyapunov = +0.007)
    - TIMING-COUPLED: GPU jitter modulates dynamics
    - HARDWARE ENTROPY: Output depends on timing, not PRNG seed

    Designed to be:
    - Fast (>1kHz sampling)
    - Small (256-1024 ossicles)
    - Truly chaotic (never converges, positive Lyapunov)
    - Timing-dependent (hardware entropy source)
    """

    def __init__(self, config: SentinelConfig, sensor_id: int = 0):
        self.config = config
        self.sensor_id = sensor_id
        self.total = config.n_ossicles * config.oscillator_depth

        # Output buffers
        self.out_r = cp.zeros(1, dtype=cp.float32)
        self.out_var = cp.zeros(1, dtype=cp.float32)

        # Legacy: Coupling constants (for use_lorenz=False)
        angle_rad = np.radians(MAGIC_ANGLE)
        self.coupling_ab = np.float32(np.cos(angle_rad) * config.epsilon)
        self.coupling_bc = np.float32(np.sin(angle_rad) * config.epsilon)
        self.coupling_ca = np.float32(config.epsilon / PHI)

        # V2: Lorenz parameters (for use_lorenz=True)
        self.lorenz_sigma = np.float32(config.lorenz_sigma)
        self.lorenz_rho = np.float32(config.lorenz_rho)
        self.lorenz_beta = np.float32(config.lorenz_beta)
        self.lorenz_dt = np.float32(config.lorenz_dt)
        self.timing_sensitivity = config.timing_sensitivity

        # Timing tracking for entropy injection
        self.last_timing_ns = 0
        self.timing_history = deque(maxlen=100)
        self.timing_perturbation = 0.0

        # CUDA config
        self.block_size = min(256, self.total)
        self.grid_size = (self.total + self.block_size - 1) // self.block_size

        # Derivative tracking
        self.k_eff_history = deque(maxlen=config.derivative_window)
        self.last_k_eff = 0.0

        # Reset cycle tracking (validated: τ=46.1s, reset every 23s)
        self.last_reset_time = time.perf_counter()
        self.time_since_reset = 0.0

        # Thermal sensing (validated: variance correlates with temp at r=-0.97)
        self.variance_history = deque(maxlen=config.thermal_window)
        self.thermal_baseline_mean = None
        self.thermal_baseline_std = None

        # r_ab sensitivity tracking (validated: r=-0.999 correlation with sensitivity)
        self.last_r_ab = 0.0
        self.last_r_bc = 0.0
        self.last_r_ca = 0.0
        self.r_ab_history = deque(maxlen=config.derivative_window)
        self.reset_reason = None  # Track why last reset occurred

        # Initialize oscillators
        self._reset_oscillators()

    def _reset_oscillators(self, reason: str = "manual"):
        """Reset oscillators to random state for peak sensitivity.

        For Lorenz mode: Initialize near the attractor for immediate chaos.
        For legacy mode: Random state near origin.
        """
        if self.config.use_lorenz:
            # Lorenz: Initialize near the attractor
            # Standard Lorenz attractor lives around x,y ∈ [-20,20], z ∈ [5,45]
            self.osc_a = cp.random.random(self.total, dtype=cp.float32) * 2 - 1  # x: [-1, 1]
            self.osc_b = cp.random.random(self.total, dtype=cp.float32) * 2 - 1  # y: [-1, 1]
            self.osc_c = cp.random.random(self.total, dtype=cp.float32) * 5 + 20  # z: [20, 25] near attractor
        else:
            # Legacy: Random near origin
            self.osc_a = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
            self.osc_b = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25
            self.osc_c = cp.random.random(self.total, dtype=cp.float32) * 0.5 - 0.25

        self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5
        self.last_reset_time = time.perf_counter()
        self.last_timing_ns = 0
        self.timing_perturbation = 0.0
        self.k_eff_history.clear()
        self.r_ab_history.clear()
        self.timing_history.clear()
        self.reset_reason = reason

    # =========================================================================
    # r_ab SENSITIVITY REGIME MONITORING (validated: r=-0.999 correlation)
    # =========================================================================

    def get_internal_correlations(self) -> Tuple[float, float, float]:
        """Get internal correlations (r_ab, r_bc, r_ca).

        Validated finding: r_ab predicts sensitivity with r=-0.999
        - r_ab < 0.95 = HIGHLY SENSITIVE (transient regime)
        - r_ab > 0.98 = LOW SENSITIVITY (thermalized regime)

        Returns:
            (r_ab, r_bc, r_ca)
        """
        # Use subset for speed
        sample_size = min(10000, self.total)
        indices = cp.random.choice(self.total, sample_size, replace=False)

        a = self.osc_a[indices]
        b = self.osc_b[indices]
        c = self.osc_c[indices]

        # Compute correlations
        r_ab = float(cp.corrcoef(a, b)[0, 1])
        r_bc = float(cp.corrcoef(b, c)[0, 1])
        r_ca = float(cp.corrcoef(c, a)[0, 1])

        # Handle NaN
        r_ab = 0.0 if np.isnan(r_ab) else r_ab
        r_bc = 0.0 if np.isnan(r_bc) else r_bc
        r_ca = 0.0 if np.isnan(r_ca) else r_ca

        # Track history
        self.last_r_ab = r_ab
        self.last_r_bc = r_bc
        self.last_r_ca = r_ca
        self.r_ab_history.append(r_ab)

        return r_ab, r_bc, r_ca

    def get_sensitivity_regime(self) -> Dict:
        """Get current sensitivity regime based on r_ab.

        Validated finding (Jan 2026):
        - r_ab < 0.95: TRANSIENT regime, ~20x more sensitive
        - r_ab 0.95-0.98: TRANSITIONAL regime
        - r_ab > 0.98: THERMALIZED regime, low sensitivity

        Returns:
            Dict with 'regime', 'r_ab', 'sensitivity_multiplier', 'should_reset'
        """
        r_ab = self.last_r_ab

        if r_ab < self.config.r_ab_sensitive_threshold:
            regime = "TRANSIENT"
            multiplier = 20.0  # ~20x more sensitive (from binned analysis)
        elif r_ab < self.config.r_ab_reset_threshold:
            regime = "TRANSITIONAL"
            # Linear interpolation between 20x and 1x
            frac = (r_ab - self.config.r_ab_sensitive_threshold) / \
                   (self.config.r_ab_reset_threshold - self.config.r_ab_sensitive_threshold)
            multiplier = 20.0 * (1 - frac) + 1.0 * frac
        else:
            regime = "THERMALIZED"
            multiplier = 1.0

        should_reset = r_ab > self.config.r_ab_reset_threshold

        return {
            'regime': regime,
            'r_ab': r_ab,
            'r_bc': self.last_r_bc,
            'r_ca': self.last_r_ca,
            'sensitivity_multiplier': multiplier,
            'should_reset': should_reset,
        }

    def check_reset(self) -> bool:
        """Check if reset is needed and perform if so. Returns True if reset occurred.

        Two reset strategies (configurable via use_r_ab_reset):
        1. Time-based: Reset every reset_interval_s (23s = τ/2)
        2. r_ab-based: Reset when r_ab > r_ab_reset_threshold (0.98)

        r_ab-based is preferred as it directly measures sensitivity state.
        """
        self.time_since_reset = time.perf_counter() - self.last_reset_time

        if self.config.use_r_ab_reset:
            # r_ab-based reset (preferred)
            if self.last_r_ab > self.config.r_ab_reset_threshold:
                self._reset_oscillators(reason="r_ab_threshold")
                return True
        else:
            # Time-based reset (fallback)
            if self.time_since_reset >= self.config.reset_interval_s:
                self._reset_oscillators(reason="time_interval")
                return True

        return False

    def get_sensitivity_weight(self) -> float:
        """Get current sensitivity weight based on time since reset.

        Validated: sensitivity decays as exp(-t/τ) with τ=46.1s
        Peak sensitivity is immediately after reset.
        """
        return np.exp(-self.time_since_reset / self.config.tau_thermalization)

    def step_and_measure(self, auto_reset: bool = True) -> Tuple[float, float, float, float]:
        """
        Single step: advance dynamics, inject noise/timing, measure k_eff.

        V2 LORENZ MODE (use_lorenz=True):
        - Measures GPU timing jitter as entropy source
        - Modulates Lorenz ρ parameter with timing
        - TRUE chaos amplifies small timing variations

        LEGACY MODE (use_lorenz=False):
        - Linear diffusive coupling with PRNG noise
        - Not truly chaotic (Lyapunov = -0.01)

        Args:
            auto_reset: If True, automatically reset based on r_ab threshold or time

        Returns:
            (k_eff, variance, dk_eff/dt estimate, sensitivity_weight)

        Sensitivity weight is exp(-t/τ) where τ=46.1s (validated).
        Use this to weight measurements - higher weight = more reliable.

        Note: Also updates r_ab tracking. Use get_sensitivity_regime() for
        r_ab-based sensitivity info after calling this method.
        """
        # Update time tracking
        self.time_since_reset = time.perf_counter() - self.last_reset_time

        if self.config.use_lorenz:
            # V2: LORENZ DYNAMICS WITH TIMING COUPLING
            # Measure timing BEFORE kernel to get jitter
            t0 = time.perf_counter_ns()

            # Compute timing perturbation (detrended from last measurement)
            if self.last_timing_ns > 0:
                timing_delta = t0 - self.last_timing_ns
            else:
                timing_delta = 0
            self.last_timing_ns = t0

            # Convert timing to ρ perturbation
            # Scale: 1000ns variation → ~0.001 change in ρ
            self.timing_perturbation = timing_delta * self.timing_sensitivity * 1e-6
            self.timing_history.append(timing_delta)

            # Run Lorenz kernel with timing-modulated ρ
            _lorenz_kernel(
                (self.grid_size,), (self.block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 self.lorenz_sigma, self.lorenz_rho, self.lorenz_beta,
                 self.lorenz_dt,
                 np.float32(self.timing_perturbation),
                 self.total, 5)  # 5 RK4 iterations per step
            )
        else:
            # LEGACY: Linear diffusive coupling
            # Refresh noise
            self.noise = cp.random.random(3 * self.total, dtype=cp.float32) - 0.5

            # Step with noise injection (σ=0.001 optimal from SR validation)
            _sentinel_kernel(
                (self.grid_size,), (self.block_size,),
                (self.osc_a, self.osc_b, self.osc_c,
                 self.noise,
                 self.coupling_ab, self.coupling_bc, self.coupling_ca,
                 np.float32(self.config.noise_amplitude),
                 self.total, 3)  # 3 iterations per step
            )

        # Fast measurement (r_ab and variance)
        _correlate_kernel(
            (1,), (256,),
            (self.osc_a, self.osc_b, self.out_r, self.out_var, self.total)
        )

        r = float(self.out_r[0])
        var = float(self.out_var[0])

        # Track r_ab for sensitivity regime (validated: r=-0.999 with sensitivity)
        self.last_r_ab = r if not np.isnan(r) else 0.0
        self.r_ab_history.append(self.last_r_ab)

        # k_eff calculation
        if self.config.use_lorenz:
            # Lorenz: k_eff based on x-y correlation and variance
            # The chaotic dynamics provide natural variation
            x = min(var / 100.0, 1.0)  # Lorenz has larger variance
            k_eff = r * (1 - x) * 10.0  # Scale factor for Lorenz
        else:
            # Legacy formula
            x = min(var / 2.0, 1.0)
            k_eff = r * (1 - x) * self.config.epsilon * 1000

        # Derivative estimate
        self.k_eff_history.append(k_eff)
        if len(self.k_eff_history) >= 2:
            dk_dt = (k_eff - self.k_eff_history[0]) / len(self.k_eff_history)
        else:
            dk_dt = 0.0

        self.last_k_eff = k_eff

        # Sensitivity weight (validated: decays as exp(-t/τ))
        sensitivity = self.get_sensitivity_weight()

        # Check for auto-reset AFTER measurement (so we have fresh r_ab)
        if auto_reset:
            self.check_reset()

        return k_eff, var, dk_dt, sensitivity

    def step_and_measure_full(self, auto_reset: bool = True) -> Dict:
        """
        Full step with all state information including r_ab regime.

        Returns dict with:
            - k_eff: effective coupling
            - variance: oscillator variance
            - dk_dt: derivative estimate
            - sensitivity_weight: exp(-t/τ) decay factor
            - r_ab: internal correlation (sensitivity predictor, r=-0.999)
            - regime: 'TRANSIENT', 'TRANSITIONAL', or 'THERMALIZED'
            - sensitivity_multiplier: 1x-20x based on regime
            - time_since_reset: seconds since last reset
            - reset_reason: why last reset occurred
        """
        k_eff, var, dk_dt, sensitivity = self.step_and_measure(auto_reset)
        regime_info = self.get_sensitivity_regime()

        return {
            'k_eff': k_eff,
            'variance': var,
            'dk_dt': dk_dt,
            'sensitivity_weight': sensitivity,
            'r_ab': self.last_r_ab,
            'regime': regime_info['regime'],
            'sensitivity_multiplier': regime_info['sensitivity_multiplier'],
            'time_since_reset': self.time_since_reset,
            'reset_reason': self.reset_reason,
        }

    def is_detection(self, dk_dt: float, sensitivity: float) -> bool:
        """Check if dk_dt indicates a detection (LEGACY - use is_workload_detected).

        Uses validated noise floor σ=0.003 as threshold basis.
        Detection requires |dk_dt| > 3σ AND sensitivity > 0.5
        """
        threshold = 3 * self.config.noise_floor_sigma  # 3σ detection
        return abs(dk_dt) > threshold and sensitivity > 0.5

    def is_workload_detected(self, current_timing_mean: float) -> Tuple[bool, float]:
        """Check for workload using VALIDATED mean-shift detection (A5/O1-O7).

        Primary detection method (Jan 2026 validated):
        - Workload causes GPU contention → timing increases
        - Detection threshold: >50% mean shift from baseline
        - At 50% workload: +2519% mean shift (A5 validated)
        - At 1% workload: +83% mean shift (A5 validated)

        Args:
            current_timing_mean: Current timing mean in microseconds

        Returns:
            (detected: bool, mean_shift_pct: float)
        """
        if not hasattr(self, '_timing_baseline_mean') or self._timing_baseline_mean <= 0:
            return False, 0.0

        mean_shift = current_timing_mean - self._timing_baseline_mean
        mean_shift_pct = (mean_shift / self._timing_baseline_mean) * 100

        detected = mean_shift_pct > self.config.mean_shift_threshold_pct
        return detected, mean_shift_pct

    def calibrate_timing_baseline(self, duration: float = 3.0):
        """Calibrate timing baseline for mean-shift detection.

        Must be called during IDLE state before detection.

        Args:
            duration: Calibration duration in seconds
        """
        print(f"Calibrating timing baseline ({duration}s)... Keep system IDLE.")
        samples = []
        start = time.time()

        while time.time() - start < duration:
            # Measure kernel timing
            t0 = time.perf_counter_ns()
            self.step_and_measure(auto_reset=False)  # Run one oscillator step
            cp.cuda.stream.get_current_stream().synchronize()
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1000.0)  # Convert to μs

        self._timing_baseline_mean = np.mean(samples)
        self._timing_baseline_std = np.std(samples)

        print(f"  Baseline: {self._timing_baseline_mean:.2f} ± {self._timing_baseline_std:.2f} μs")
        print(f"  Detection threshold: >{self.config.mean_shift_threshold_pct:.0f}% shift")

    def inject_negentropic(self, amplitude: float = 0.1):
        """
        Inject negentropy via correlation boosting.

        Key insight from tuning: Making oscillators MORE CORRELATED with each other
        produces +9.59σ detection, vs -1.69σ for pattern addition.

        Method: Push osc_b and osc_c toward osc_a (increase inter-oscillator correlation)
        """
        # Correlation boost: blend b,c toward a
        blend = amplitude  # 0.1 = 10% blend toward a
        self.osc_b = self.osc_b * (1 - blend) + self.osc_a * blend
        self.osc_c = self.osc_c * (1 - blend) + self.osc_a * blend

    def inject_entropic(self, amplitude: float = 0.1):
        """
        Inject entropy via correlation breaking.

        Method: Add uncorrelated noise to each oscillator independently,
        reducing inter-oscillator correlation.
        """
        # Uncorrelated noise to each channel independently
        self.osc_a += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude
        self.osc_b += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude
        self.osc_c += (cp.random.random(self.total, dtype=cp.float32) - 0.5) * amplitude

    # =========================================================================
    # THERMAL SENSING (validated: variance correlates with temp at r=-0.97)
    # =========================================================================

    def get_total_variance(self) -> float:
        """Get total variance across all oscillators.

        Validated finding: variance has r=-0.97 correlation with GPU temperature.
        As temperature increases, variance DECREASES.
        This is the primary thermal sensing metric.
        """
        var_a = float(cp.var(self.osc_a))
        var_b = float(cp.var(self.osc_b))
        var_c = float(cp.var(self.osc_c))
        return var_a + var_b + var_c

    def update_thermal_baseline(self, variance: float):
        """Update thermal baseline statistics.

        Call this during stable conditions to establish baseline.
        """
        self.variance_history.append(variance)
        if len(self.variance_history) >= self.config.thermal_window:
            arr = np.array(self.variance_history)
            self.thermal_baseline_mean = np.mean(arr)
            self.thermal_baseline_std = np.std(arr)

    def get_thermal_deviation(self, variance: float) -> Optional[float]:
        """Get deviation from thermal baseline in sigma units.

        Returns None if baseline not yet established.
        Negative values = temperature INCREASE (variance decreases)
        Positive values = temperature DECREASE (variance increases)
        """
        if self.thermal_baseline_mean is None or self.thermal_baseline_std is None:
            return None
        if self.thermal_baseline_std < 1e-10:
            return 0.0
        return (variance - self.thermal_baseline_mean) / self.thermal_baseline_std

    def is_thermal_event(self, variance: float) -> Tuple[bool, Optional[float], Optional[str]]:
        """Detect thermal events based on variance deviation.

        Returns:
            (is_event, deviation_sigma, direction)
            direction is "heating" or "cooling" or None
        """
        self.update_thermal_baseline(variance)
        deviation = self.get_thermal_deviation(variance)

        if deviation is None:
            return False, None, None

        threshold = self.config.thermal_threshold_sigma

        if deviation < -threshold:
            # Variance dropped = temperature increased
            return True, deviation, "heating"
        elif deviation > threshold:
            # Variance increased = temperature decreased
            return True, deviation, "cooling"
        else:
            return False, deviation, None

    def step_and_measure_thermal(self, auto_reset: bool = True) -> Tuple[float, float, float, float, Optional[float]]:
        """Step and measure with thermal sensing.

        Returns:
            (k_eff, variance, dk_dt, sensitivity, thermal_deviation)

        thermal_deviation is in sigma units from baseline.
        Negative = heating, Positive = cooling.
        """
        k_eff, var, dk_dt, sensitivity = self.step_and_measure(auto_reset)

        # Get thermal metric (total variance)
        total_var = self.get_total_variance()
        self.update_thermal_baseline(total_var)
        thermal_dev = self.get_thermal_deviation(total_var)

        return k_eff, var, dk_dt, sensitivity, thermal_dev


class SentinelArray:
    """
    Multiple independent sentinels on single GPU.

    For scaling tests and spatial correlation analysis.
    """

    def __init__(self, n_sentinels: int, config: SentinelConfig = None):
        self.n_sentinels = n_sentinels
        self.config = config or SentinelConfig()
        self.sentinels = [Sentinel(self.config, i) for i in range(n_sentinels)]

    def step_all(self) -> List[Tuple[float, float, float, float]]:
        """Step all sentinels and return measurements.

        Returns list of (k_eff, variance, dk_dt, sensitivity) tuples.
        """
        results = []
        for s in self.sentinels:
            results.append(s.step_and_measure())
        cp.cuda.stream.get_current_stream().synchronize()
        return results

    def get_k_eff_array(self) -> np.ndarray:
        """Get k_eff values from all sentinels."""
        results = self.step_all()
        return np.array([r[0] for r in results])

    def get_correlation_matrix(self, n_samples: int = 100) -> np.ndarray:
        """Compute correlation matrix between sentinels."""
        # Collect time series
        series = np.zeros((self.n_sentinels, n_samples))

        for t in range(n_samples):
            measurements = self.step_all()
            for i, (k_eff, _, _) in enumerate(measurements):
                series[i, t] = k_eff

        # Compute correlation matrix
        return np.corrcoef(series)


# =============================================================================
# REFERENCE-SUBTRACTED ARCHITECTURE (exp65-67 validated)
# =============================================================================
# Isolates GPU-specific entropy (4%) from algorithmic determinism (78%)
#
# Architecture:
#   Actual Lorenz (timing-coupled) ──┐
#                                    ├──► Difference = Pure Entropy
#   Reference Lorenz (no timing) ────┘
#
# Result: 4% GPU timing entropy isolated, 78% algorithmic cancelled
# =============================================================================

class DifferentialSentinel:
    """
    Reference-subtracted chaotic resonator.

    Runs TWO Lorenz oscillators in parallel:
    1. ACTUAL: Timing-coupled (receives GPU timing perturbations)
    2. REFERENCE: Deterministic (no timing, same initial conditions)

    The DIFFERENCE isolates pure GPU timing entropy:
    - Algorithmic component (78%): CANCELLED (same in both)
    - GPU timing (4%): ISOLATED (only in actual)
    - Noise (18%): Present in difference

    V3 ARCHITECTURE (exp65-67 validated):
    - Reference subtraction removes 78% algorithmic background
    - Improves TRNG entropy density from 4% to ~100%
    - Improves detector SNR by removing common-mode

    Usage:
        sensor = DifferentialSentinel()

        # For TRNG - pure timing entropy:
        entropy = sensor.step_and_get_entropy()

        # For detection - perturbation-only signal:
        delta_k, delta_var = sensor.step_and_measure_differential()
    """

    def __init__(self, config: SentinelConfig = None):
        self.config = config or SentinelConfig()

        # Ensure Lorenz mode
        self.config.use_lorenz = True

        # Create ACTUAL sensor (timing-coupled)
        self.actual = Sentinel(self.config, sensor_id=0)

        # Create REFERENCE sensor (deterministic - no timing)
        # We'll run it with timing_sensitivity=0
        ref_config = SentinelConfig(
            n_ossicles=self.config.n_ossicles,
            oscillator_depth=self.config.oscillator_depth,
            use_lorenz=True,
            lorenz_sigma=self.config.lorenz_sigma,
            lorenz_rho=self.config.lorenz_rho,
            lorenz_beta=self.config.lorenz_beta,
            lorenz_dt=self.config.lorenz_dt,
            timing_sensitivity=0.0,  # NO timing coupling!
        )
        self.reference = Sentinel(ref_config, sensor_id=1)

        # Synchronize initial states
        self._synchronize_states()

        # Tracking
        self.entropy_history = deque(maxlen=100)
        self.delta_k_history = deque(maxlen=100)

    def _synchronize_states(self):
        """Synchronize actual and reference oscillator states."""
        # Copy actual state to reference
        self.reference.osc_a = self.actual.osc_a.copy()
        self.reference.osc_b = self.actual.osc_b.copy()
        self.reference.osc_c = self.actual.osc_c.copy()
        self.reference.last_timing_ns = 0
        self.reference.timing_perturbation = 0.0

    def reset(self):
        """Reset both oscillators with synchronized states."""
        self.actual._reset_oscillators("differential_reset")
        self._synchronize_states()

    def step_and_measure_differential(self) -> Tuple[float, float, float, float]:
        """
        Step both oscillators and return the DIFFERENCE.

        Returns:
            (delta_k_eff, delta_variance, actual_k, reference_k)

        delta_k_eff = actual_k - reference_k = pure timing effect
        """
        # Step actual (with timing)
        k_actual, var_actual, _, _ = self.actual.step_and_measure(auto_reset=False)

        # Step reference (no timing - perturbation stays 0)
        self.reference.timing_perturbation = 0.0  # Force no timing
        k_ref, var_ref, _, _ = self.reference.step_and_measure(auto_reset=False)

        # Compute differences
        delta_k = k_actual - k_ref
        delta_var = var_actual - var_ref

        self.delta_k_history.append(delta_k)

        return delta_k, delta_var, k_actual, k_ref

    def step_and_get_entropy(self) -> float:
        """
        Get pure timing entropy (for TRNG).

        Returns the timing perturbation that was applied to actual
        but not to reference - this is the GPU-specific entropy.
        """
        # Step actual to get timing measurement
        self.actual.step_and_measure(auto_reset=False)

        # The timing perturbation IS the entropy
        entropy = self.actual.timing_perturbation

        # Also step reference to keep them in sync iteration-wise
        self.reference.timing_perturbation = 0.0
        self.reference.step_and_measure(auto_reset=False)

        self.entropy_history.append(entropy)

        return entropy

    def get_entropy_bytes(self, n_bytes: int) -> bytes:
        """
        Get random bytes from pure timing entropy.

        Much higher entropy density than raw k_eff (100% vs 4%).
        """
        result = bytearray(n_bytes)

        for i in range(n_bytes):
            # Collect timing samples and extract LSBs
            entropy = self.step_and_get_entropy()

            # Convert to integer via timing nanoseconds
            timing_ns = self.actual.timing_history[-1] if self.actual.timing_history else 0
            result[i] = timing_ns & 0xFF

        return bytes(result)

    def get_detection_signal(self, n_samples: int = 10) -> float:
        """
        Get detection signal with reference subtraction.

        Higher SNR than raw k_eff because 78% algorithmic background removed.
        """
        deltas = []
        for _ in range(n_samples):
            delta_k, _, _, _ = self.step_and_measure_differential()
            deltas.append(delta_k)

        return np.mean(deltas)

    def get_stats(self) -> Dict:
        """Get statistics on differential measurements."""
        if len(self.delta_k_history) < 10:
            return {'error': 'Not enough samples'}

        deltas = np.array(self.delta_k_history)

        return {
            'delta_k_mean': float(np.mean(deltas)),
            'delta_k_std': float(np.std(deltas)),
            'delta_k_max': float(np.max(np.abs(deltas))),
            'n_samples': len(deltas),
            'entropy_contribution': float(np.std(deltas)),  # This IS the 4%
        }


def find_minimum_array_size():
    """Find minimum array size that maintains transient + detects signals."""

    print("="*60)
    print("FINDING MINIMUM VIABLE ARRAY SIZE")
    print("="*60)
    print("(Using validated parameters: σ=0.001, reset@23s, τ=46.1s)")

    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    results = []

    for size in sizes:
        print(f"\nTesting {size} ossicles...")

        config = SentinelConfig(n_ossicles=size, oscillator_depth=32)
        sensor = Sentinel(config)

        # Collect samples (auto_reset=False for this test)
        k_effs = []
        for _ in range(500):
            k, v, dk, sens = sensor.step_and_measure(auto_reset=False)
            k_effs.append(k)

        k_effs = np.array(k_effs)
        mean_k = np.mean(k_effs)
        std_k = np.std(k_effs)

        # Test detection
        sensor2 = Sentinel(config)
        baseline = []
        for _ in range(100):
            k, _, _, _ = sensor2.step_and_measure(auto_reset=False)
            baseline.append(k)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)

        # Perturb with negentropy
        perturbed = []
        for _ in range(100):
            sensor2.inject_negentropic(0.3)
            k, _, _, _ = sensor2.step_and_measure(auto_reset=False)
            perturbed.append(k)
        perturbed_mean = np.mean(perturbed)

        effect = (perturbed_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0

        results.append({
            'size': size,
            'mean_k': mean_k,
            'std_k': std_k,
            'maintains_transient': std_k > 0.001,
            'effect_sigma': effect,
            'detects': abs(effect) > 3
        })

        status = "✓" if std_k > 0.001 and abs(effect) > 3 else "✗"
        print(f"  k_eff: {mean_k:.4f} ± {std_k:.4f}")
        print(f"  Detection effect: {effect:.2f}σ {status}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Size':<8} {'Transient?':<12} {'Detects?':<10} {'Effect':<10}")
    print("-"*45)

    min_viable = None
    for r in results:
        trans = "YES" if r['maintains_transient'] else "no"
        det = "YES" if r['detects'] else "no"
        print(f"{r['size']:<8} {trans:<12} {det:<10} {r['effect_sigma']:.2f}σ")

        if r['maintains_transient'] and r['detects'] and min_viable is None:
            min_viable = r['size']

    if min_viable:
        print(f"\n✓ MINIMUM VIABLE SIZE: {min_viable} ossicles")
    else:
        print(f"\nNo configuration met both criteria")

    return results, min_viable


def test_scaling():
    """Test multiple sentinels running simultaneously."""

    print("\n" + "="*60)
    print("SCALING TEST: Multiple Sentinels")
    print("="*60)

    # Find how many 256-ossicle sentinels we can run
    config = SentinelConfig(n_ossicles=256)

    for n_sentinels in [2, 4, 8, 16, 32, 64]:
        try:
            print(f"\nTesting {n_sentinels} sentinels ({n_sentinels * 256} total ossicles)...")

            array = SentinelArray(n_sentinels, config)

            # Time 1000 steps
            start = time.perf_counter()
            for _ in range(1000):
                array.step_all()
            elapsed = time.perf_counter() - start

            rate = 1000 / elapsed
            print(f"  Rate: {rate:.0f} Hz")

            # Check correlation between sentinels
            corr_matrix = array.get_correlation_matrix(50)
            off_diag = corr_matrix[np.triu_indices(n_sentinels, k=1)]
            mean_corr = np.mean(off_diag) if len(off_diag) > 0 else 0

            print(f"  Inter-sentinel correlation: {mean_corr:.4f}")

        except cp.cuda.memory.OutOfMemoryError:
            print(f"  OUT OF MEMORY")
            break

    print("\n" + "="*60)


# =============================================================================
# GPU TIMING TRNG (Experiment 57)
# =============================================================================

class GPUATRNG:
    """
    GPU-Accelerated True Random Number Generator.

    Uses GPU kernel execution timing jitter as entropy source.

    Validated metrics (Exp 57):
    - Shannon entropy: 8.00 / 8 bits (100%)
    - Min-entropy: 7.76 / 8 bits (97%)
    - Bit bias: < 0.3% per bit
    - Autocorrelation: 0.011
    - NIST tests: 3/4 passed
    - Throughput: ~120 kbps (TRUE entropy)

    The entropy comes from:
    - GPU scheduler variations
    - Memory access timing
    - Thermal effects on clock
    - Cache state variations

    This is INDEPENDENT of any PRNG seed - verified by testing
    identical seeds producing different timing outputs.
    """

    def __init__(self, config: SentinelConfig = None):
        """Initialize TRNG with optional sentinel config."""
        self.config = config or SentinelConfig(
            n_ossicles=64,  # Minimal for fast timing
            oscillator_depth=16,
        )
        self.sensor = Sentinel(self.config)
        self._warmed_up = False

    def warmup(self, n_iterations: int = 100):
        """Warm up GPU for stable timing."""
        for _ in range(n_iterations):
            self.sensor.step_and_measure(auto_reset=False)
        self._warmed_up = True

    def get_timing_sample(self) -> int:
        """Get single timing sample in nanoseconds."""
        t0 = time.perf_counter_ns()
        self.sensor.step_and_measure(auto_reset=False)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        return t1 - t0

    def get_random_byte(self) -> int:
        """Get single random byte from timing LSB."""
        timing = self.get_timing_sample()
        return timing & 0xFF

    def get_random_bytes(self, n_bytes: int) -> bytes:
        """Get multiple random bytes."""
        if not self._warmed_up:
            self.warmup()

        result = bytearray(n_bytes)
        for i in range(n_bytes):
            result[i] = self.get_random_byte()
        return bytes(result)

    def get_random_int(self, min_val: int = 0, max_val: int = 255) -> int:
        """Get random integer in range [min_val, max_val]."""
        range_size = max_val - min_val + 1
        # Use rejection sampling for uniformity
        max_valid = (256 // range_size) * range_size - 1

        while True:
            val = self.get_random_byte()
            if val <= max_valid:
                return min_val + (val % range_size)

    def stream_bytes(self, n_bytes: int = None):
        """Generator yielding random bytes. If n_bytes is None, stream forever."""
        if not self._warmed_up:
            self.warmup()

        count = 0
        while n_bytes is None or count < n_bytes:
            yield self.get_random_byte()
            count += 1

    def get_entropy_estimate(self, n_samples: int = 1000) -> Dict:
        """Estimate entropy quality from sample."""
        if not self._warmed_up:
            self.warmup()

        samples = np.array([self.get_random_byte() for _ in range(n_samples)])

        # Shannon entropy
        counts = np.bincount(samples, minlength=256)
        probs = counts / n_samples
        probs = probs[probs > 0]
        shannon = -np.sum(probs * np.log2(probs))

        # Min-entropy
        max_prob = np.max(counts) / n_samples
        min_ent = -np.log2(max_prob) if max_prob > 0 else 0

        # Autocorrelation
        d = samples.astype(float) - np.mean(samples)
        if np.std(d) > 1e-10:
            autocorr = np.corrcoef(d[:-1], d[1:])[0, 1]
        else:
            autocorr = 0

        return {
            'shannon_entropy': shannon,
            'max_entropy': 8.0,
            'entropy_ratio': shannon / 8.0,
            'min_entropy': min_ent,
            'autocorrelation': autocorr,
            'n_samples': n_samples,
            'unique_values': len(np.unique(samples)),
        }


def trng_demo(n_bytes: int = 256, output_file: str = None, stream: bool = False,
              test: bool = False):
    """Demo the GPU timing TRNG."""
    print("=" * 60)
    print("GPU TIMING TRNG - True Random Number Generator")
    print("=" * 60)

    if not cp.cuda.is_available():
        print("ERROR: CUDA required")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")

    trng = GPUATRNG()

    print("\nWarming up GPU...")
    trng.warmup(200)

    if test:
        # Run quality tests
        print("\n" + "-" * 40)
        print("ENTROPY QUALITY TEST")
        print("-" * 40)

        stats = trng.get_entropy_estimate(5000)
        print(f"  Shannon entropy: {stats['shannon_entropy']:.3f} / 8.0 bits "
              f"({100*stats['entropy_ratio']:.1f}%)")
        print(f"  Min-entropy:     {stats['min_entropy']:.3f} / 8.0 bits")
        print(f"  Autocorrelation: {stats['autocorrelation']:.4f}")
        print(f"  Unique values:   {stats['unique_values']} / 256")

        # Throughput test
        print("\n" + "-" * 40)
        print("THROUGHPUT TEST")
        print("-" * 40)

        start = time.perf_counter()
        test_bytes = trng.get_random_bytes(1000)
        elapsed = time.perf_counter() - start

        rate = 1000 / elapsed
        print(f"  Rate: {rate:.0f} bytes/sec ({rate * 8:.0f} bits/sec)")

        return

    if stream:
        # Stream to stdout
        print("\nStreaming random bytes to stdout (Ctrl+C to stop)...")
        print("-" * 40)
        import sys
        try:
            for byte in trng.stream_bytes():
                sys.stdout.buffer.write(bytes([byte]))
                sys.stdout.buffer.flush()
        except KeyboardInterrupt:
            print("\n\nStopped.")
        return

    # Generate bytes
    print(f"\nGenerating {n_bytes} random bytes...")
    start = time.perf_counter()
    data = trng.get_random_bytes(n_bytes)
    elapsed = time.perf_counter() - start

    rate = n_bytes / elapsed
    print(f"  Generated in {elapsed:.3f}s ({rate:.0f} bytes/sec)")

    if output_file:
        with open(output_file, 'wb') as f:
            f.write(data)
        print(f"  Saved to: {output_file}")
    else:
        # Show hex dump
        print("\n" + "-" * 40)
        print("HEX DUMP (first 64 bytes)")
        print("-" * 40)
        for i in range(0, min(64, len(data)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
            print(f"  {i:04x}: {hex_str:<48} {ascii_str}")

    # Quick stats
    print("\n" + "-" * 40)
    print("QUICK STATS")
    print("-" * 40)
    arr = np.frombuffer(data, dtype=np.uint8)
    print(f"  Mean:   {np.mean(arr):.1f} (expected: 127.5)")
    print(f"  Std:    {np.std(arr):.1f} (expected: ~73.9)")
    print(f"  Min:    {np.min(arr)}")
    print(f"  Max:    {np.max(arr)}")
    print(f"  Unique: {len(np.unique(arr))} / {min(256, n_bytes)}")


# =============================================================================
# RAW TIMING TRNG V2 (Experiments 70-72 validated)
# =============================================================================
# Bypasses Lorenz entirely - raw timing LSBs are the true entropy source.
#
# Key findings (Exp 69-72):
# - Lorenz DESTROYS entropy (7.96 bits in → 0.01 bits out)
# - Raw timing LSBs: true entropy from GPU jitter
# - Optimal: 4 LSBs without debiasing → 6/6 NIST, 470 kbps
# - 8 LSBs has pattern in upper bits → fails NIST
# - Von Neumann debiasing only needed if using >7 LSBs
#
# Throughput: ~470 kbps true random (4 LSBs, no debiasing)
# =============================================================================

class RawTimingTRNG:
    """
    True Random Number Generator using raw GPU timing LSBs.

    V2 Architecture (Exp 70-72 validated):
    - Extracts LSBs directly from GPU timing measurements
    - Optimal: 4-7 LSBs without debiasing
    - NO Lorenz processing (which destroys entropy)

    Metrics (4 LSBs, no debiasing - DEFAULT):
    - NIST tests: 6/6 pass
    - Entropy: 7.99 bits/byte
    - Throughput: ~470 kbps true random

    The entropy comes from:
    - GPU scheduler jitter
    - Memory access timing variations
    - Thermal effects on clock
    - Cache state variations

    Usage:
        trng = RawTimingTRNG()  # Uses optimal 4 LSBs
        random_bytes = trng.get_random_bytes(1024)
        # Or stream:
        for byte in trng.stream_bytes():
            process(byte)
    """

    def __init__(self, n_lsbs: int = 4, use_debiasing: bool = False):
        """
        Initialize raw timing TRNG.

        Args:
            n_lsbs: Number of LSBs to extract per timing sample (1-8)
            use_debiasing: Apply von Neumann debiasing (recommended for 6/6 NIST)
        """
        self.n_lsbs = min(8, max(1, n_lsbs))
        self.use_debiasing = use_debiasing
        self._warmed_up = False

        # Minimal GPU workload for timing measurement
        self._workload_size = 1024
        self._workload = cp.zeros(self._workload_size, dtype=cp.float32)

        # Bit buffer for debiasing
        self._bit_buffer = []

        # Stats tracking
        self.samples_collected = 0
        self.bits_generated = 0
        self.bits_discarded = 0

    def warmup(self, n_iterations: int = 100):
        """Warm up GPU for stable timing."""
        for _ in range(n_iterations):
            self._get_timing_ns()
        self._warmed_up = True

    def _get_timing_ns(self) -> int:
        """Get single timing measurement in nanoseconds."""
        t0 = time.perf_counter_ns()
        # Simple GPU operation to measure
        self._workload = cp.sin(self._workload)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        return t1 - t0

    def _extract_bits(self, timing_ns: int) -> list:
        """Extract LSBs from timing value."""
        bits = []
        for i in range(self.n_lsbs):
            bits.append((timing_ns >> i) & 1)
        return bits

    def _von_neumann_debias(self, bits: list) -> list:
        """
        Apply von Neumann debiasing.

        Pairs of bits:
        - 01 → output 0
        - 10 → output 1
        - 00, 11 → discard

        Removes any bias but halves throughput on average.
        """
        output = []
        i = 0
        while i + 1 < len(bits):
            b0, b1 = bits[i], bits[i+1]
            if b0 == 0 and b1 == 1:
                output.append(0)
            elif b0 == 1 and b1 == 0:
                output.append(1)
            # else: discard (00 or 11)
            else:
                self.bits_discarded += 2
            i += 2
        return output

    def _bits_to_byte(self, bits: list) -> int:
        """Convert 8 bits to a byte."""
        result = 0
        for i, bit in enumerate(bits[:8]):
            result |= (bit << i)
        return result

    def get_random_bits(self, n_bits: int) -> list:
        """Get n random bits."""
        if not self._warmed_up:
            self.warmup()

        result = []

        while len(result) < n_bits:
            # Get timing and extract LSBs
            timing = self._get_timing_ns()
            self.samples_collected += 1
            bits = self._extract_bits(timing)

            if self.use_debiasing:
                # Add to buffer and debias
                self._bit_buffer.extend(bits)
                if len(self._bit_buffer) >= 16:  # Process in chunks
                    before_len = len(self._bit_buffer)
                    debiased = self._von_neumann_debias(self._bit_buffer)
                    result.extend(debiased)
                    self.bits_generated += len(debiased)
                    self._bit_buffer = []
            else:
                result.extend(bits)
                self.bits_generated += len(bits)

        return result[:n_bits]

    def get_random_byte(self) -> int:
        """Get single random byte."""
        bits = self.get_random_bits(8)
        return self._bits_to_byte(bits)

    def get_random_bytes(self, n_bytes: int) -> bytes:
        """Get multiple random bytes."""
        if not self._warmed_up:
            self.warmup()

        result = bytearray(n_bytes)
        for i in range(n_bytes):
            result[i] = self.get_random_byte()
        return bytes(result)

    def stream_bytes(self, n_bytes: int = None):
        """Generator yielding random bytes."""
        if not self._warmed_up:
            self.warmup()

        count = 0
        while n_bytes is None or count < n_bytes:
            yield self.get_random_byte()
            count += 1

    def get_random_int(self, min_val: int = 0, max_val: int = 255) -> int:
        """Get random integer in range [min_val, max_val]."""
        range_size = max_val - min_val + 1
        # Use rejection sampling for uniformity
        max_valid = (256 // range_size) * range_size - 1

        while True:
            val = self.get_random_byte()
            if val <= max_valid:
                return min_val + (val % range_size)

    def get_stats(self) -> dict:
        """Get TRNG statistics."""
        efficiency = (self.bits_generated / (self.bits_generated + self.bits_discarded)
                     if self.bits_generated + self.bits_discarded > 0 else 0)
        return {
            'samples_collected': self.samples_collected,
            'bits_generated': self.bits_generated,
            'bits_discarded': self.bits_discarded,
            'debiasing_efficiency': efficiency,
            'n_lsbs': self.n_lsbs,
            'use_debiasing': self.use_debiasing,
        }

    def get_entropy_estimate(self, n_samples: int = 1000) -> dict:
        """Estimate entropy quality."""
        if not self._warmed_up:
            self.warmup()

        samples = np.array([self.get_random_byte() for _ in range(n_samples)])

        # Shannon entropy
        counts = np.bincount(samples, minlength=256)
        probs = counts / n_samples
        probs = probs[probs > 0]
        shannon = -np.sum(probs * np.log2(probs))

        # Min-entropy
        max_prob = np.max(counts) / n_samples
        min_ent = -np.log2(max_prob) if max_prob > 0 else 0

        # Autocorrelation
        d = samples.astype(float) - np.mean(samples)
        if np.std(d) > 1e-10:
            autocorr = np.corrcoef(d[:-1], d[1:])[0, 1]
        else:
            autocorr = 0

        # Bit balance
        bits = np.unpackbits(samples.astype(np.uint8))
        bit_balance = np.mean(bits)

        return {
            'shannon_entropy': shannon,
            'max_entropy': 8.0,
            'entropy_ratio': shannon / 8.0,
            'min_entropy': min_ent,
            'autocorrelation': autocorr,
            'bit_balance': bit_balance,
            'n_samples': n_samples,
            'unique_values': len(np.unique(samples)),
        }


def emi_mode(duration: int = 30, sample_rate: int = 500):
    """
    EMI spectrum analysis mode.

    Detects electromagnetic interference from power grid (60 Hz + harmonics),
    VRM switching frequencies, and subharmonics.

    Args:
        duration: Capture duration in seconds
        sample_rate: Sample rate in Hz (default 500 for 250 Hz Nyquist)
    """
    from scipy import signal as scipy_signal

    print("=" * 60)
    print("CIRIS SENTINEL: EMI Spectrum Analysis")
    print("=" * 60)
    print(f"\nSample rate: {sample_rate} Hz (Nyquist: {sample_rate//2} Hz)")
    print(f"Duration: {duration} seconds")
    print(f"Frequency resolution: {1/duration:.4f} Hz")

    # Create sensor
    config = SentinelConfig(n_ossicles=256)
    sensor = Sentinel(config)

    # Warmup
    print("\nWarming up sensor...")
    for _ in range(100):
        sensor.step_and_measure(auto_reset=False)

    # Collect samples
    print(f"\nCollecting EMI data...")
    timings = []
    interval = 1.0 / sample_rate
    start = time.time()

    n_samples = duration * sample_rate
    for i in range(n_samples):
        target = start + i * interval

        # Measure timing
        t0 = time.perf_counter_ns()
        sensor.step_and_measure(auto_reset=False)
        cp.cuda.stream.get_current_stream().synchronize()
        t1 = time.perf_counter_ns()
        timings.append((t1 - t0) / 1000.0)  # μs

        # Maintain sample rate
        sleep_time = target + interval - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if (i + 1) % (sample_rate * 5) == 0:
            pct = 100 * (i + 1) / n_samples
            print(f"  {pct:.0f}%...")

    timings = np.array(timings)
    actual_rate = len(timings) / (time.time() - start)
    print(f"\nActual rate: {actual_rate:.0f} Hz")

    # Compute PSD
    print("\n" + "=" * 60)
    print("EMI SPECTRUM ANALYSIS")
    print("=" * 60)

    y = timings - np.mean(timings)
    y = scipy_signal.detrend(y)

    nperseg = min(len(y) // 4, 2048)
    freqs, psd = scipy_signal.welch(y, fs=actual_rate, nperseg=nperseg)

    # Find peaks
    peak_idx, _ = scipy_signal.find_peaks(psd, height=np.median(psd) * 2, distance=3)
    noise_floor = np.median(psd)

    # EMI target frequencies
    HARMONICS = [60, 120, 180, 240]
    SUBHARMONICS = [60/n for n in [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]]

    # Check for EMI
    print("\n60 Hz HARMONICS:")
    print("-" * 40)
    for target in HARMONICS:
        if target > actual_rate / 2:
            continue
        # Find closest peak
        mask = (freqs > target - 2) & (freqs < target + 2)
        if np.any(mask):
            idx = np.argmax(psd[mask])
            freq = freqs[mask][idx]
            power = psd[mask][idx]
            snr = 10 * np.log10(power / noise_floor) if noise_floor > 0 else 0
            if snr > 3:
                print(f"  {target:4.0f} Hz: detected at {freq:.2f} Hz, SNR = {snr:.1f} dB")
            else:
                print(f"  {target:4.0f} Hz: below noise floor")
        else:
            print(f"  {target:4.0f} Hz: not in range")

    print("\nSUBHARMONICS (60/n Hz):")
    print("-" * 40)
    detected_sub = []
    for target in sorted(SUBHARMONICS, reverse=True):
        mask = (freqs > target - 0.5) & (freqs < target + 0.5)
        if np.any(mask):
            idx = np.argmax(psd[mask])
            freq = freqs[mask][idx]
            power = psd[mask][idx]
            snr = 10 * np.log10(power / noise_floor) if noise_floor > 0 else 0
            if snr > 3:
                n = int(round(60 / target))
                detected_sub.append((target, freq, snr))
                print(f"  60/{n:2d} = {target:5.2f} Hz: detected at {freq:.2f} Hz, SNR = {snr:.1f} dB")

    print("\nVRM FREQUENCIES (< 2 Hz):")
    print("-" * 40)
    vrm_peaks = []
    for i in peak_idx:
        if freqs[i] < 2.0:
            snr = 10 * np.log10(psd[i] / noise_floor) if noise_floor > 0 else 0
            if snr > 3:
                vrm_peaks.append((freqs[i], snr))
                print(f"  {freqs[i]:.3f} Hz: SNR = {snr:.1f} dB")

    print("\nTOP 10 SPECTRAL PEAKS:")
    print("-" * 40)
    sorted_peaks = sorted(zip(freqs[peak_idx], psd[peak_idx]),
                         key=lambda x: -x[1])[:10]
    for f, p in sorted_peaks:
        snr = 10 * np.log10(p / noise_floor)
        # Identify if EMI
        emi = ""
        if abs(f - 60) < 2:
            emi = "← 60 Hz"
        elif abs(f - 120) < 2:
            emi = "← 120 Hz"
        elif abs(f - 180) < 2:
            emi = "← 180 Hz"
        else:
            for n in range(2, 61):
                if abs(f - 60/n) < 0.5:
                    emi = f"← 60/{n}"
                    break
        print(f"  {f:7.2f} Hz: SNR = {snr:5.1f} dB {emi}")

    # Summary
    print("\n" + "=" * 60)
    print("EMI SUMMARY")
    print("=" * 60)
    n_harmonics = sum(1 for h in HARMONICS if h <= actual_rate/2 and
                      any(abs(freqs[i] - h) < 2 and
                          10*np.log10(psd[i]/noise_floor) > 3 for i in peak_idx))
    print(f"  60 Hz harmonics detected: {n_harmonics}/4")
    print(f"  Subharmonics detected: {len(detected_sub)}")
    print(f"  VRM peaks detected: {len(vrm_peaks)}")
    print(f"  Noise floor: {noise_floor:.2e}")

    if n_harmonics >= 1 or len(detected_sub) >= 3:
        print("\n  Status: EMI VISIBLE")
    else:
        print("\n  Status: EMI below detection threshold")

    print("=" * 60)


def main():
    """Run sentinel tests, TRNG demo, or timing server."""
    import argparse

    parser = argparse.ArgumentParser(description='CIRIS Sentinel / GPU TRNG / Timing Server')

    # TRNG options
    parser.add_argument('--trng', action='store_true', help='Run TRNG mode')
    parser.add_argument('--bytes', type=int, default=256, help='Number of bytes to generate')
    parser.add_argument('--output', '-o', type=str, help='Output file for random bytes')
    parser.add_argument('--stream', action='store_true', help='Stream random bytes to stdout')
    parser.add_argument('--test', action='store_true', help='Run TRNG quality tests')

    # Timing infrastructure options
    parser.add_argument('--timing-server', action='store_true',
                        help='Run UDP timing server (for Jetson)')
    parser.add_argument('--timing-calibrate', action='store_true',
                        help='Calibrate timing to remote host')
    parser.add_argument('--timing-host', type=str, default='192.168.50.203',
                        help='Remote host for timing calibration')
    parser.add_argument('--timing-samples', type=int, default=100,
                        help='Number of calibration samples')

    # EMI detection options
    parser.add_argument('--emi', action='store_true',
                        help='Run EMI spectrum analysis mode')
    parser.add_argument('--emi-duration', type=int, default=30,
                        help='EMI capture duration in seconds')
    parser.add_argument('--emi-rate', type=int, default=500,
                        help='EMI sample rate in Hz (default 500)')

    args = parser.parse_args()

    # Timing server mode (run on Jetson)
    if args.timing_server:
        run_timing_server()
        return

    # Timing calibration mode
    if args.timing_calibrate:
        timing = TimingSync(args.timing_host, auto_load=False)
        timing.calibrate(n_samples=args.timing_samples)
        return

    if args.trng or args.stream or args.test:
        trng_demo(n_bytes=args.bytes, output_file=args.output,
                  stream=args.stream, test=args.test)
        return

    # EMI mode
    if args.emi:
        emi_mode(duration=args.emi_duration, sample_rate=args.emi_rate)
        return

    # Original sentinel demo
    print("="*60)
    print("CIRIS SENTINEL: Minimal Sustained-Transient Detector")
    print("="*60)

    if not cp.cuda.is_available():
        print("ERROR: CUDA required")
        return

    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nGPU: {props['name'].decode()}")

    # Find minimum array size
    results, min_size = find_minimum_array_size()

    # Test scaling
    test_scaling()

    # Demo continuous monitoring
    if min_size:
        print("\n" + "="*60)
        print(f"DEMO: Continuous monitoring with {min_size} ossicles")
        print("="*60)
        print("(Using validated: σ=0.001, auto-reset@23s, 3σ threshold)")

        config = SentinelConfig(n_ossicles=min_size, sample_rate_hz=500)
        sensor = Sentinel(config)

        print("\nRunning 30 seconds of monitoring (will auto-reset at 23s)...")
        print("(Tracking k_eff, dk/dt, and sensitivity weight)")

        detections = 0
        resets = 0
        start = time.perf_counter()

        while time.perf_counter() - start < 30.0:
            k_eff, var, dk_dt, sensitivity = sensor.step_and_measure()

            # Check for reset
            if sensor.time_since_reset < 0.1:  # Just reset
                resets += 1
                print(f"  [RESET #{resets}] Sensitivity restored to 1.0")

            # Use validated detection method
            if sensor.is_detection(dk_dt, sensitivity):
                detections += 1
                direction = "↑" if dk_dt > 0 else "↓"
                print(f"  {direction} Event: k={k_eff:.4f}, dk/dt={dk_dt:+.4f}, sens={sensitivity:.3f}")

        elapsed = time.perf_counter() - start
        print(f"\n  {detections} events, {resets} resets in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("SENTINEL READY")
    print("="*60)


if __name__ == "__main__":
    main()
