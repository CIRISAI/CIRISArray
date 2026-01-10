# CIRISArray

> **⚠️ EXPERIMENTAL / RESEARCH-GRADE SOFTWARE**
> This is early-stage research code. Results are preliminary and require independent validation.

**Spatial imaging of propagating disorder patterns across GPU dies using massive ossicle arrays.**

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

## What is This?

CIRISArray extends single-point [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) sensors to **spatial wave imaging**. By deploying arrays of 16-4096 ossicles across a GPU die, we can:

- **Image propagating entropy waves** (like a seismograph network)
- **Detect wave velocity and direction** via cross-correlation
- **Triangulate tampering origin points**
- **Achieve 110x lower latency than VLA's minimum dump time** (45µs vs 5ms)*

```
              ENTROPY WAVE IMAGING

    ┌─────────────────────────────────────────────────┐
    │     Ossicle Array on RTX 4090 Die               │
    │                                                 │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 0      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 1      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 2      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 3      │
    │   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·              │
    │                                                 │
    │   4096 ossicles × 768 bytes = 3 MB (L2 cache)  │
    │   154 million samples/sec total bandwidth      │
    └─────────────────────────────────────────────────┘
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Single ossicle latency | **20 µs** | GPU kernel round-trip |
| Array scaling (1→4096) | **1.3x** | Near O(1) via GPU parallelism |
| Detection latency | **45 µs** | Event to detection |
| Min detectable event | **0.1 ms** | 50x lower than VLA minimum dump* |
| Throughput | **154 M samples/sec** | Full 4096-ossicle array |

*\*Latency comparison only. VLA is a radio telescope with fundamentally different physics, scale, and capabilities. This comparison refers specifically to the minimum time between data captures (VLA realfast: 5ms dump time vs our 45µs detection latency). Not a claim of equivalent sensitivity or scientific capability.*

## Installation

```bash
git clone https://github.com/CIRISAI/CIRISArray
cd CIRISArray
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- NumPy
- CuPy (for GPU acceleration, optional but recommended)

## Quick Start

```python
from src.runtime import OssicleRuntime, RuntimeMode

# Create and configure runtime
runtime = OssicleRuntime()
runtime.configure_array(n_rows=8, n_cols=16)
runtime.calibrate()

# Monitor mode
runtime.set_mode(RuntimeMode.MONITOR)
result = runtime.step()
print(f"Max z-score: {result['monitor']['max_z']:.2f}")

# Beamforming
runtime.set_mode(RuntimeMode.BEAMFORM)
runtime.steer_beam(azimuth=45)
azimuths, powers = runtime.scan_beam()

# Fog chamber visualization
runtime.set_mode(RuntimeMode.FOG_CHAMBER)
print(runtime.get_fog_ascii())
```

Run the full demo:
```bash
python demo_runtime.py
```

## Runtime Modes

| Mode | Description |
|------|-------------|
| `MONITOR` | Passive monitoring with event detection |
| `BEAMFORM` | VLA-style phased array beamforming |
| `TRANSMIT` | Inject entropy patterns (pulse, chirp, code) |
| `RECEIVE` | Focused reception with matched filtering |
| `TX_RX` | Bistatic mode (simultaneous TX/RX) |
| `FOG_CHAMBER` | Wave track visualization |
| `CALIBRATE` | Baseline calibration |

## Features

### Array Tuning
```python
# Global tuning
runtime.configure_ossicles(r_base=3.72, twist_deg=1.1, coupling=0.08)

# Region-specific tuning
from src.runtime import OssicleParams
enhanced = OssicleParams(r_base=3.75, coupling=0.1)
runtime.array.set_region_params(row_range=(3,5), col_range=(6,10), params=enhanced)
```

### Beamforming
```python
runtime.steer_beam(azimuth=30, elevation=0)
runtime.beamformer.add_null(azimuth=0)  # Add null direction
azimuths, powers = runtime.scan_beam(az_range=(-90, 90))
```

### Event Tracing with Signatures
```python
runtime.event_config.intensity_threshold = 3.0
runtime.event_config.sign_events = True

events = runtime.get_events(min_intensity=3.0)
for e in events:
    print(f"Event: {e.intensity}σ at ossicle {e.trigger_ossicle}")
    print(f"Signature: {e.signature}")
```

### TX/RX Patterns
```python
runtime.set_tx_pattern('chirp', freq_range=(5, 50))
runtime.set_mode(RuntimeMode.TX_RX)
result = runtime.step()
print(f"Matched filter output: {result['matched_output']}")
```

## Architecture

```
src/
├── runtime.py          # Main runtime (1200+ lines)
│   ├── OssicleRuntime  # Orchestrator
│   ├── ArrayController # Geometry & tuning
│   ├── Beamformer      # Phased array
│   ├── TransmitReceive # TX/RX modes
│   ├── Monitor         # Real-time views
│   ├── EventTracer     # Signed captures
│   └── FogChamber      # Visualization
└── __init__.py

experiments/
└── exp28_array_latency.py  # VLA-inspired benchmarks
```

## VLA-Inspired Design

This project applies techniques from the [Very Large Array](https://public.nrao.edu/telescopes/vla/) radio telescope:

- **Ring buffer with triggered capture** (like realfast)
- **Phased array beamforming** with steering and nulling
- **Matched filter reception** for coded transmissions
- **Sub-sample delay correction** for wave tracking

## Research Notes

**Magic Angle (1.1°):** The use of 1.1° twist angle (inspired by graphene twistronics) shows empirical correlation with improved sensitivity in our experiments. However, this is an *observed correlation, not proven causation*. The mechanism by which twist angle affects oscillator coupling in this context requires further theoretical and experimental investigation.

**Experimental Status:** This software is research-grade. Key claims require:
- Independent replication on different GPU architectures
- Formal analysis of detection statistics
- Real-world validation against known tampering events

## License

**Business Source License 1.1 (BSL 1.1)**

- **Free for**: Individuals, DIY, academics, nonprofits, organizations <$1M revenue
- **Commercial license required**: Organizations ≥$1M revenue
- **Converts to AGPL 3.0**: January 1, 2030

Licensor: CIRIS L3C (Eric Moore)

## Related Projects

- [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) - Single-point entropy detection
- [RATCHET](https://github.com/CIRISAI/RATCHET) - Formal verification framework

## Citation

```bibtex
@software{cirisarray,
  author = {Moore, Eric},
  title = {CIRISArray: Spatial Imaging of GPU Entropy Patterns},
  year = {2026},
  publisher = {CIRIS L3C},
  url = {https://github.com/CIRISAI/CIRISArray},
  license = {BSL-1.1}
}
```
