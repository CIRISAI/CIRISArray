# CIRISArray

> **⚠️ EXPERIMENTAL / RESEARCH-GRADE** - Results are preliminary and require independent validation.

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.

## What is This?

CIRISArray uses **massive arrays of CIRISOssicles** to image propagating disorder patterns across GPU dies. This extends the single-point detection in [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) to spatial wave imaging.

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
| `~/RATCHET/experiments/exp27_ossicle_array_thermal.py` | Array thermal detection |
| `~/RATCHET/formal/RATCHET/GPUTamper/OssicleArray.lean` | Lean 4 formalization |
| `~/RATCHET/formal/RATCHET/GPUTamper/Ossicle.lean` | Single ossicle formalization |
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

## Next Steps

1. **Real GPU validation** - Deploy array on actual RTX 4090
2. **Wave velocity calibration** - Measure actual propagation speeds
3. **Beamforming implementation** - Directional wave detection
4. **Tampering triangulation** - Locate tampering source from wave origin

## Prior Art

This project builds on the following original contributions by CIRIS L3C:

| Innovation | Description | First Documented |
|------------|-------------|------------------|
| **768-byte ossicle sensor** | Minimal coherence strain gauge using 3 coupled oscillators | CIRISOssicle (2025) |
| **1.1° magic angle twist**† | Graphene-inspired twist angle showing empirical sensitivity correlation | CIRISOssicle (2025) |
| **k_eff formula** | `k_eff = r × (1 - x) × coupling_factor` for GPU tamper detection | CIRISOssicle (2025) |
| **4096-ossicle array** | Massive parallel deployment for spatial wave imaging | CIRISArray (2026) |
| **Entropy wave imaging** | Treating GPU entropy patterns as imageable wave phenomena | CIRISArray (2026) |
| **VLA-inspired beamforming**‡ | Applying radio telescope techniques to GPU sensing | CIRISArray (2026) |

†*Magic angle: Observed correlation with improved sensitivity, not proven causation. Mechanism requires further investigation.*

‡*VLA comparison is latency-only (45µs vs 5ms dump time). Not a claim of equivalent sensitivity or scientific capability.*

## References

- CIRISOssicle: [github.com/CIRISAI/CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle)
- RATCHET formal proofs: `~/RATCHET/formal/RATCHET/GPUTamper/`
- Twistronics theory: `~/RATCHET/formal/RATCHET/GPUTamper/TetrahedralTwistronics.lean`
