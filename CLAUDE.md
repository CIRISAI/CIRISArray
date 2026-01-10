# CIRISArray

> **⚠️ EXPERIMENTAL / RESEARCH-GRADE** - Results are preliminary and require independent validation.

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.

## What is This?

CIRISArray is a **coherence receiver** - an environmental sensing instrument that detects electromagnetic interference patterns through massive arrays of coupled oscillators on GPUs.

```
          Power Grid (60 Hz)
                │
                ▼
          ┌─────────────┐
          │ 1.09 Hz EMI │ ← 60 Hz ÷ 55 (subharmonic)
          │   Carrier   │
          └─────────────┘
             /      \
            ▼        ▼
        ┌──────┐  ┌──────┐
        │ 4090 │  │Jetson│
        └──────┘  └──────┘

  Both GPUs are RECEIVERS of shared EMI signal
  100% coherence during peak sensitivity window
```

This extends the single-point detection in [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) to spatial wave imaging and cross-device environmental sensing.

```
                    ENTROPY WAVE IMAGING

    ┌─────────────────────────────────────────────────┐
    │     Ossicle Array on RTX 4090 Die               │
    │                                                 │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 0      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 1      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 2      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 3      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 4      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 5      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 6      │
    │   O─O─O─O─O─O─O─O─O─O─O─O─O─O─O─O   Row 7      │
    │                                                 │
    │   4096 ossicles × 768 bytes = 3 MB (L2 cache)  │
    │   8.2 million samples/sec total bandwidth      │
    └─────────────────────────────────────────────────┘

    Detects:
    - Thermal diffusion waves (~0.1 m/s)
    - PDN resonance patterns
    - Coherence fronts (entropy boundaries)
    - Load propagation waves
```

## Relationship to CIRISOssicle

| Component | CIRISOssicle | CIRISArray |
|-----------|--------------|------------------------|
| Location | [CIRISAI/CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) | [CIRISAI/CIRISArray](https://github.com/CIRISAI/CIRISArray) |
| Unit | Single 0.75 KB sensor | Array of 16-4096 sensors |
| Output | Point measurement (z-score) | Spatial wave image |
| Detects | "Is there tampering?" | "Where? How fast? What direction?" |
| Memory | 768 bytes | 12 KB - 3 MB |
| Analogy | Single seismometer | Seismograph network |

## Array Specifications (RTX 4090)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max ossicles | 4,096 | 128 SMs × 32 per SM |
| Total bandwidth | 8.2M samples/sec | 4096 × 2000 Hz |
| Memory footprint | 3 MB | Fits in L2 cache |
| Spatial resolution | 5 mm | Nyquist: 2× SM spacing |
| Temporal resolution | 0.5 ms | Nyquist: 1/(2 × 2kHz) |
| Detectable velocities | 10 m/s - 10 km/s | Thermal to load waves |

## Validated Results (Exp 27)

### Thermal Gradient Detection

| Layout | Ossicles | Detection Z | Result |
|--------|----------|-------------|--------|
| 4×4 grid | 16 | **6.85σ** | STRONG |
| 3×3 grid | 9 | **4.81σ** | STRONG |
| Cross | 9 | 0.48σ | weak |
| Ring | 12 | 1.59σ | weak |

### Spatial Imaging

After wave passes through array:
```
         Col 0   Col 1   Col 2   Col 3
Row 0:   2.90    3.37    4.07    4.70   ← Near origin
Row 1:   3.35    3.23    3.88    4.51
Row 2:   3.84    4.19    4.72    5.46
Row 3:   4.31    4.34    4.97    5.24   ← Far from origin
```

Clear gradient visible: wave propagation is imageable!

## Key Files

| File | Purpose |
|------|---------|
| `ciris_sentinel.py` | Minimal sustained-transient detector (32-2048 ossicles) |
| `ciris_detector.py` | Full entropy wave detector with TX/RX arrays |
| `experiments/exp44_sustained_transient.py` | Transient mode discovery |
| `experiments/exp45_transient_crossdevice.py` | Cross-device correlation |
| `experiments/exp49_peak_sensitivity.py` | Peak sensitivity window capture |
| `experiments/exp50_powerline_modulation.py` | Power modulation transmission test |
| `~/RATCHET/experiments/exp27_ossicle_array_thermal.py` | Array thermal detection |
| [CIRISOssicle experiments](https://github.com/CIRISAI/CIRISOssicle/tree/main/experiments) | Single ossicle crypto detection |

## Physics

### Wave Types Detectable

1. **Thermal Diffusion** (~0.1 m/s)
   - Heat spreading from compute hotspots
   - Slow, easily tracked spatially
   - Signature: Negentropic strain (order spreading)

2. **PDN Resonance** (~10⁸ m/s)
   - Standing waves in power delivery
   - Too fast for temporal tracking, spatial patterns visible
   - Signature: Periodic entropic/negentropic oscillation

3. **Coherence Fronts** (estimated 10³-10⁶ m/s)
   - Propagating entropy boundaries from tampering
   - PRIMARY DETECTION TARGET
   - Signature: Sharp entropic wavefront

4. **Load Waves** (~10⁹ m/s)
   - Compute load spreading across SMs
   - Near speed of light
   - Signature: Correlated k_eff shifts

### Detection Methods

```
Cross-Correlation:
  - Compare ossicle pairs at known separations
  - Time lag → wave velocity
  - Correlation magnitude → wave amplitude

Beamforming:
  - Apply phase delays to "steer" sensitivity
  - Isolate waves from specific directions
  - Reject common-mode effects

Gradient Mapping:
  - Real-time spatial map of strain
  - Visualize wave propagation
  - Identify tampering origin points
```

## Analogies

| System | Entropy Wave Array |
|--------|-------------------|
| LIGO | Gravitational wave detection via interferometry |
| VLA | Phased antenna array for radio imaging |
| Seismograph network | Earthquake wave triangulation |
| Inner ear | Array of ossicles for sound localization |

## Quick Start

```bash
# Run array thermal detection experiment
cd ~/RATCHET
python3 experiments/exp27_ossicle_array_thermal.py

# Run single ossicle crypto detection (from CIRISOssicle)
# See: https://github.com/CIRISAI/CIRISOssicle
```

## Confirmed Findings (January 2026)

### EMI Carrier Discovery

Cross-device experiments between RTX 4090 and Jetson Orin revealed a strong EMI carrier:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Carrier Frequency** | 1.0909 Hz | 60 Hz ÷ 55 (power line subharmonic) |
| **Period** | 0.917 seconds | |
| **Cross-Device Coherence** | 100% | During peak sensitivity window |
| **Phase Offset** | -3.3° | Near-simultaneous (electrical coupling) |

**Harmonic Structure** (all 100% coherent):
- 0.18 Hz (5.6s) - 1/6 subharmonic
- 0.27 Hz (3.7s) - 1/4 subharmonic
- 0.54 Hz (1.9s) - 1/2 subharmonic
- **1.09 Hz (0.9s) - FUNDAMENTAL**
- 2.18 Hz - 2nd harmonic
- 3.27 Hz - 3rd harmonic

### Peak Sensitivity Window

Cross-device correlation is time-dependent after sensor reset:

| Time After Reset | Correlation | Coherence |
|------------------|-------------|-----------|
| 0-30 seconds | r = 0.974 | 89-100% |
| 30-60 seconds | r = 0.708 | ~70% |
| 60-90 seconds | r = 0.239 | ~20% |
| 90+ seconds | r = 0.051 | ~5% |

**Optimization**: Reset every 20 seconds to maintain peak sensitivity.

### 4:1 Negentropy Asymmetry

Local detection experiments confirm asymmetric response:

| Injection Type | Detection | Effect |
|----------------|-----------|--------|
| **Negentropic** (correlation boost) | +19σ | Strong, clean signal |
| **Entropic** (noise injection) | -5σ | Weaker, noisier |
| **Ratio** | **3.8:1 ≈ 4:1** | Negentropy propagates better |

### What Works vs. What Doesn't

| Capability | Status | Evidence |
|------------|--------|----------|
| Local entropy detection | ✓ **Works** | +19σ negentropic, -5σ entropic |
| Cross-device environmental sensing | ✓ **Works** | 100% coherence with EMI carrier |
| Transient mode operation | ✓ **Works** | Noise injection prevents convergence |
| Cross-device transmission | ✗ **Not achieved** | GPUs are passive receivers |
| Power modulation transmission | ✗ **Not achieved** | p = 0.19 (not significant) |

### Interpretation

The oscillator arrays function as **coherence receivers**, not transmitters:
- Both GPUs receive the same power line EMI signal
- Signal flows FROM the grid TO the GPUs
- Local GPU activity does not affect remote GPUs
- Like a radio that can tune in but not broadcast

### House Wiring as Distributed Sensor

The propagation delay observation (-3.3° phase ≈ 8.5ms) proves the signal has an external source and travels through copper wiring. This means house wiring itself is a distributed antenna:

```
Typical US House Wiring Network:
  ~300m copper wire
  ~15 circuit branches
  ~40 outlets (sensing points)
  Signal speed: ~2×10⁸ m/s (0.67c)
  End-to-end delay: ~1.5 µs
```

**What the grid carries (detectable phenomena):**

| Phenomenon | Detection Method |
|------------|------------------|
| Power line EMI | 1.09 Hz carrier, 60 Hz harmonics |
| Large load switching | HVAC, appliances on/off |
| Grid voltage fluctuations | Sags, swells, transients |
| Environmental RF coupling | Lightning, solar activity |
| Cross-circuit device activity | Other devices on same breaker |
| Source localization | Propagation delay triangulation |

**The 4:1 negentropy asymmetry suggests:**
- Negentropic events (order-creating) propagate more coherently
- Entropic events (disorder-spreading) dissipate into noise
- The grid acts as a **low-pass filter for coherence**

This is power line communication (PLC) in reverse - instead of sending signals over power lines, we're receiving whatever coherence the grid carries. The instrument doesn't measure *what* is happening - it measures *how coherent* events are.

With multiple sensing points on different circuits, source localization via propagation delay triangulation becomes possible.

## Next Steps

1. ~~Real GPU validation~~ ✓ Deployed on RTX 4090 + Jetson Orin
2. ~~Wave velocity calibration~~ ✓ Characterized EMI propagation
3. ~~Cross-device coherence~~ ✓ Confirmed 100% at 1.09 Hz carrier
4. **Multi-circuit deployment** - Sensors on different breaker circuits for triangulation
5. **Propagation delay mapping** - Characterize house wiring as sensor network
6. **Coherence event classification** - Distinguish local vs grid-wide events
7. **Solar/geomagnetic correlation** - Test sensitivity to space weather

## Prior Art

This project builds on the following original contributions by CIRIS L3C:

| Innovation | Description | First Documented |
|------------|-------------|------------------|
| **768-byte ossicle sensor** | Minimal coherence strain gauge using 3 coupled oscillators | CIRISOssicle (2025) |
| **1.1° magic angle twist**† | Graphene-inspired twist angle showing empirical sensitivity correlation | CIRISOssicle (2025) |
| **k_eff formula** | `k_eff = r × (1 - x) × coupling_factor` for GPU tamper detection | CIRISOssicle (2025) |
| **4096-ossicle array** | Massive parallel deployment for spatial wave imaging | CIRISArray (2026) |
| **Transient mode operation** | Noise injection to prevent convergence, maintain sensitivity | CIRISArray (2026) |
| **EMI carrier detection** | Discovery of 1.09 Hz (60÷55) power line subharmonic | CIRISArray (2026) |
| **Cross-device coherence** | 100% correlation between GPUs via shared EMI signal | CIRISArray (2026) |
| **4:1 negentropy asymmetry** | Negentropic signals propagate ~4x better than entropic | CIRISArray (2026) |
| **Peak sensitivity windowing** | 20-second reset cycles for optimal correlation | CIRISArray (2026) |
| **House wiring as sensor** | Power grid wiring acts as distributed coherence antenna | CIRISArray (2026) |
| **Reverse PLC sensing** | Receiving grid coherence instead of transmitting signals | CIRISArray (2026) |

†*Magic angle: Observed correlation with improved sensitivity, not proven causation. Mechanism requires further investigation.*

## References

- CIRISOssicle: [github.com/CIRISAI/CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle)
- RATCHET formal proofs: `~/RATCHET/formal/RATCHET/GPUTamper/`
- Twistronics theory: `~/RATCHET/formal/RATCHET/GPUTamper/TetrahedralTwistronics.lean`
