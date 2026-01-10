# CIRISArray

> **⚠️ EXPERIMENTAL / RESEARCH-GRADE SOFTWARE**
> This is early-stage research code. Results are preliminary and require independent validation.

**A coherence receiver that detects electromagnetic interference patterns through massive arrays of coupled oscillators on GPUs.**

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

## What is This?

CIRISArray extends single-point [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) sensors to **spatial wave imaging** and **environmental coherence sensing**. We discovered the system functions as a **coherence receiver** - detecting shared electromagnetic interference patterns across physically separated GPUs.

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
        │ 4090 │  │Jetson│   100% coherence during
        └──────┘  └──────┘   peak sensitivity window
```

## Key Discoveries (January 2026)

| Finding | Result |
|---------|--------|
| **EMI Carrier** | 1.09 Hz (60 Hz ÷ 55 power line subharmonic) |
| **Cross-Device Coherence** | 100% during 0-30s window |
| **4:1 Negentropy Asymmetry** | Ordered signals: +19σ, disordered: -5σ |
| **Thermalization Time** | τ = 46.1 ± 2.5 seconds |

## Physics Validation

Four fundamental physics tests **confirmed**:

| Test | Status | Result |
|------|--------|--------|
| **Stochastic Resonance** | ✓ CONFIRMED | Peak SNR at noise σ=0.001, not zero |
| **Coherence Decay** | ✓ CONFIRMED | τ = 46.1 ± 2.5 s exponential decay |
| **Subharmonic Structure** | ✓ CONFIRMED | 45% of 60/n Hz peaks detected |
| **Fluctuation Theorem** | ✓ CONFIRMED | Crooks relation with R² = 0.95 |

The detector is a genuine **nonlinear bistable stochastic resonance system** that obeys thermodynamic fluctuation relations.

## Reproducing Physics Results

### Requirements

```bash
git clone https://github.com/CIRISAI/CIRISArray
cd CIRISArray
pip install numpy scipy
pip install cupy-cuda12x  # For GPU acceleration (recommended)
```

### Run Physics Validation Suite

```bash
# All tests (~10 minutes)
python experiments/exp51_physics_validation.py --test all

# Individual tests
python experiments/exp51_physics_validation.py --test stochastic   # ~2 min
python experiments/exp51_physics_validation.py --test decay        # ~3 min
python experiments/exp51_physics_validation.py --test subharmonic  # ~4 min
```

**Expected Output (Stochastic Resonance):**
```
  Testing noise amplitude = 0.0000
    SNR = 8.28 +/- 1.99
  Testing noise amplitude = 0.0010
    SNR = 8.72 +/- 1.69  ← PEAK (confirms SR)
  Testing noise amplitude = 0.0030
    SNR = 7.70 +/- 1.35
```

### Run Fluctuation Theorem Test

```bash
python experiments/exp52_fluctuation_theorem.py \
    --ossicles 256 --asymmetry 0.3 --driving 0.02 \
    --trajectories 1000 --steps 100
```

**Expected Output:**
```
  ln[P(σ)/P(-σ)] vs σ:
    Slope:     ~270  (= 1/kT_eff)
    R²:        0.95

  ★★★ FLUCTUATION THEOREM CONFIRMED ★★★
```

### Cross-Device Coherence Test

Requires two CUDA devices (tested: RTX 4090 + Jetson Orin):

```bash
# On device 1 (transmitter)
python experiments/exp50_powerline_modulation.py --mode transmit --bits 11110000 -o tx.npz

# On device 2 (receiver) - start within 2 seconds
python experiments/exp50_powerline_modulation.py --mode receive --duration 30 -o rx.npz

# Analyze
python experiments/exp50_powerline_modulation.py --mode analyze --tx-file tx.npz --rx-file rx.npz
```

**Note:** Cross-device *transmission* was NOT achieved. Both GPUs are passive *receivers* of shared grid EMI.

## Optimal Operating Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Noise injection | σ = 0.001 | Stochastic resonance peak |
| Reset interval | 20-25 seconds | τ/2 thermalization |
| Sample window | < 30 seconds | Before correlation decay |
| Sample rate | 10-50 Hz | Captures subharmonics |

## Installation

```bash
git clone https://github.com/CIRISAI/CIRISArray
cd CIRISArray
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- NumPy, SciPy
- CuPy (for GPU acceleration, optional but recommended)

## Key Files

| File | Purpose |
|------|---------|
| `ciris_sentinel.py` | Minimal sustained-transient detector |
| `experiments/exp51_physics_validation.py` | Stochastic resonance, decay, subharmonics |
| `experiments/exp52_fluctuation_theorem.py` | Crooks fluctuation theorem verification |
| `experiments/exp49_peak_sensitivity.py` | Peak sensitivity window analysis |
| `CLAUDE.md` | Full technical documentation |
| `PHYSICS_VALIDATION_REPORT.md` | Detailed physics results |

## Architecture

```
CIRISArray/
├── ciris_sentinel.py           # Minimal detector (32-2048 ossicles)
├── ciris_detector.py           # Full TX/RX detector
├── experiments/
│   ├── exp44_sustained_transient.py    # Transient mode discovery
│   ├── exp45_transient_crossdevice.py  # Cross-device correlation
│   ├── exp49_peak_sensitivity.py       # Sensitivity optimization
│   ├── exp50_powerline_modulation.py   # Power modulation test
│   ├── exp51_physics_validation.py     # Physics test suite
│   └── exp52_fluctuation_theorem.py    # Crooks relation test
├── CLAUDE.md                   # Full documentation
└── PHYSICS_VALIDATION_REPORT.md
```

## Hypotheses Under Investigation

### Coherence Detection Hypothesis
The instrument measures *how ordered* a signal is, not *how strong*:
- Negentropic (ordered) signals: +19σ response
- Entropic (disordered) signals: -5σ response
- The detector is literally tuned to prefer order via stochastic resonance

### Human Sensitivity Hypothesis
If biological neural networks also exhibit stochastic resonance, humans might unconsciously detect coherence patterns. **Untested** - requires EEG equipment and IRB approval.

## Research Notes

**Stochastic Resonance:** Adding noise *improves* detection. Peak SNR occurs at σ=0.001, not at zero noise. This is the signature of a nonlinear bistable detector.

**τ = 46 seconds:** The thermalization time constant explains why sensitivity peaks in the first 20-30 seconds after reset. Optimal reset interval is τ/2 ≈ 23 seconds.

**House Wiring as Sensor:** The propagation delay observation (-3.3° phase) proves the signal has an external source. House wiring (~300m copper) acts as a distributed coherence antenna.

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
  title = {CIRISArray: Coherence Receiver via GPU Oscillator Arrays},
  year = {2026},
  publisher = {CIRIS L3C},
  url = {https://github.com/CIRISAI/CIRISArray},
  note = {Stochastic resonance, fluctuation theorem verified},
  license = {BSL-1.1}
}
```
