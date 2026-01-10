"""
Entropy Wave Detection Runtime
==============================

Comprehensive runtime for ossicle array entropy wave detection.

Usage:
    from src.runtime import OssicleRuntime, RuntimeMode

    runtime = OssicleRuntime()
    runtime.configure_array(n_rows=8, n_cols=16)
    runtime.calibrate()
    runtime.set_mode(RuntimeMode.MONITOR)

    while True:
        result = runtime.step()
        print(f"Max z-score: {result['monitor']['max_z']}")

Components:
    - OssicleRuntime: Main orchestrator
    - ArrayController: Array geometry and tuning
    - Beamformer: VLA-style phased array beamforming
    - TransmitReceive: TX/RX modes for active probing
    - Monitor: Real-time monitoring with alerts
    - EventTracer: Signed event capture
    - FogChamber: Wave track visualization

Author: CIRIS L3C
License: BSL 1.1
"""

from .runtime import (
    # Main runtime
    OssicleRuntime,
    RuntimeMode,

    # Configuration
    OssicleParams,
    ArrayParams,
    BeamParams,
    MonitorConfig,
    EventConfig,

    # Components
    ArrayController,
    Beamformer,
    TransmitReceive,
    Monitor,
    EventTracer,
    FogChamber,

    # Core
    Ossicle,
)

__version__ = "0.1.0"
__author__ = "CIRIS L3C"
