# CIRISArray

> **⚠️ EXPERIMENTAL / RESEARCH-GRADE** - Results are preliminary and require independent validation.

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.

## What is This?

CIRISArray extends [CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle) to massive arrays of coupled oscillators on GPUs for **local tamper detection** and spatial wave imaging.

```
                    LOCAL DETECTION (VALIDATED)
    ┌─────────────────────────────────────────────────┐
    │     Single GPU: detects local entropy changes   │
    │     - Workload changes: p=0.007                 │
    │     - Reset improves sensitivity: p=0.032      │
    │     - Bounded noise floor: σ=0.003             │
    └─────────────────────────────────────────────────┘

              CROSS-DEVICE SENSING (NOT VALIDATED)
    ┌─────────────────────────────────────────────────┐
    │     Earlier claims of cross-device coherence    │
    │     were ALGORITHMIC ARTIFACTS, not external    │
    │     signal coupling. See "Root Cause Analysis"  │
    └─────────────────────────────────────────────────┘
```

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
| `experiments/exp42_rab_fft.py` | r_ab-aware FFT analysis |
| `experiments/exp43_coupling_sweep.py` | Coupling strength optimization (ε=0.003) |
| `experiments/exp44_sustained_transient.py` | Transient mode discovery |
| `experiments/exp45_transient_crossdevice.py` | Cross-device correlation |
| `experiments/exp49_peak_sensitivity.py` | Peak sensitivity window capture |
| `experiments/exp50_powerline_modulation.py` | Power modulation transmission test |
| `experiments/exp51_physics_validation.py` | Stochastic resonance, decay, subharmonics |
| `experiments/exp52_fluctuation_theorem.py` | Crooks fluctuation theorem verification |
| `experiments/exp53_validation_protocol.py` | Null hypothesis and characterization |
| `experiments/exp54_lgi_test.py` | Leggett-Garg inequality test |
| `experiments/exp55_cross_device_test.py` | Cross-device transmission (FALSIFIED) |
| `experiments/exp28_array_latency.py` | Array response time characterization |
| `experiments/exp31_vacuum_fluctuation.py` | Noise floor measurement |
| `experiments/exp32_bell_inequality.py` | CHSH Bell test (classical result) |
| `experiments/exp56_entropy_sources.py` | Entropy source analysis |
| `experiments/exp57_trng_characterization.py` | GPU timing TRNG characterization |
| `experiments/exp110_instrument_characterization.py` | Raw signal baseline (Allan variance) |
| `experiments/exp112_keff_decomposition.py` | k_eff sensing mechanism |
| `experiments/exp113_parameter_space.py` | Parameter → ρ mapping |
| `experiments/exp114_critical_transition.py` | Phase transition at dt ≈ 0.0328 |
| `experiments/expA1_detection_validation.py` | Ossicle validation on Array (Phase 1) |
| `experiments/expA2_A4_validation.py` | Discrimination, ACF, TRNG tests |
| `experiments/expA5_A8_multisensor.py` | Multi-sensor array tests (Phase 2) |
| `experiments/expA9_A12_assumptions.py` | Distribution assumption validation (Phase 3) |
| `experiments/expB1_minimum_workload.py` | Minimum detectable workload (0.5%) |
| `experiments/expC1_keff_heatmap.py` | CCA k_eff formula validation (R²=0.929) |
| `experiments/expC2_propagation_velocity.py` | Correlation wavefront speed (0.5 m/s) |
| `experiments/expC3_nucleation_sites.py` | Collapse nucleation mapping |
| `experiments/expC4_leading_indicators.py` | Early warning signals |
| `experiments/expE1_emi_tuning.py` | EMI detection tuning (60 Hz, subharmonics) |
| `PHYSICS_VALIDATION_REPORT.md` | Detailed physics test results |
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

### What Works vs. What Doesn't

| Capability | Status | Evidence |
|------------|--------|----------|
| Local tamper detection | ✓ **Works** | p=0.007, mean shift -0.006 |
| Reset improves sensitivity | ✓ **Works** | p=0.032, 7x improvement |
| Bounded noise floor | ✓ **Works** | σ=0.0017 (Exp 31) |
| Stochastic resonance | ✓ **Works** | SNR peak at σ=0.001 |
| Fluctuation theorem | ✓ **Works** | R²=0.95, Crooks relation |
| **Variance thermal sensing** | ✓ **Works** | r=-0.97 with GPU temperature |
| **r_ab sensitivity regime** | ✓ **Works** | r=-0.999 predicts sensitivity |
| **Coupling optimization** | ✓ **Works** | ε=0.003 optimal (τ=12.8s, 562x signal) |
| **Array latency** | ✓ **Works** | 115x faster than VLA, 0.1ms min event (Exp 28) |
| **Post-warmup stability** | ✓ **Works** | σ=0.00005 after 30s warmup (Exp 39) |
| **Array SNR scaling** | ✓ **Works** | SNR ∝ √N collective averaging (Exp 37) |
| **GPU Timing TRNG** | ✓ **Works** | 120 kbps true entropy, 97% min-entropy (Exp 57) |
| **Raw timing = white noise** | ✓ **Works** | Allan slope = -2, 99.5% intrinsic (Exp 110) |
| **k_eff workload detection** | ✓ **Works** | z = 2.74 detectable via oscillator dynamics (Exp 112) |
| **lorenz_dt controls ρ** | ✓ **Works** | 88% of variance, critical parameter (Exp 113) |
| **Phase transition** | ✓ **Works** | dt_crit = 0.0328, power law R² = 0.978 (Exp 114) |
| **Variance-ratio detection** | ✓ **Works** | 421x separation, 0% FP, 100% TP (Exp A1-A12) |
| **Minimum workload ~0.5%** | ✓ **Works** | ratio = 73 × √intensity, 1% gives 8.8x (Exp B1) |
| **Fat-tailed distribution** | ✓ **Works** | κ=210, Student-t df≈1.3, explains β=1.09 (Exp A9) |
| **CCA k_eff formula** | ✓ **Works** | k_eff = k/(1+ρ(k-1)), R² = 0.798, n=21 (Exp C1) |
| **Propagation velocity** | ✓ **Works** | 0.5 ± 0.4 m/s thermal regime (Exp C2) |
| **Leading indicators** | ✓ **Works** | spatial_variance ↑ before collapse (Exp C4) |
| **60 Hz EMI detection** | ✓ **Works** | 18.7 dB SNR with bandpass, 7 subharmonics (Exp E1) |
| **Cross-sensor coherence** | ✓ **Works** | 0.88 at 1 Hz, 0.34 at 60 Hz (Exp E1) |
| k_eff thermal sensing | ✗ **NOT VALIDATED** | r=0.01 (no correlation) |
| Cross-device sensing | ✗ **NOT VALIDATED** | Algorithmic artifact (see below) |
| Cross-device transmission | ✗ **NOT VALIDATED** | Startup transient artifact |
| Power modulation | ✗ **NOT VALIDATED** | p = 0.19 (not significant) |
| 1-2 Hz external signal | ✗ **NOT VALIDATED** | Inconsistent coherence (0.2-0.9) |
| Quantum RNG detection | ✗ **NOT VALIDATED** | Cannot distinguish HW vs SW RNG (Exp 30) |
| Bell inequality violation | ✗ **NOT VALIDATED** | Classical: \|S\|=0.0002 << 2.0 (Exp 32) |
| LGI violation | ✗ **NOT VALIDATED** | Classical: K₃=1.0 (Exp 54) |
| SM-to-SM coupling | ✗ **NOT VALIDATED** | No distance dependence (Exp 36) |
| TX→RX targeting | ✗ **NOT VALIDATED** | Broadcast only, no directional control (Exp 37) |
| Perturbation detection | ✗ **NOT VALIDATED** | Memory/compute changes below threshold (Exp 40) |

### Root Cause Analysis (Experiments 55-56)

**CROSS-DEVICE TRANSMISSION - FALSIFIED**

The "90x ratio" between TX high-ε and low-ε conditions was a **startup transient artifact**:
- Both oscillators spike in first ~10 seconds regardless of TX state
- When TX pattern started with '1', we measured RX during its natural spike
- Control test (pattern "0011") showed RX still spiked on bit 0
- A/B test: ratio = 0.99x, p = 0.998 → NO COUPLING

**CROSS-DEVICE CORRELATION - ALGORITHMIC ARTIFACT**

The r=0.97 correlation between devices was NOT external sensing:

| Test | Seeds | ε values | Correlation |
|------|-------|----------|-------------|
| Local, same ε | independent | 0.05/0.05 | 0.9999 |
| Local, diff ε | independent | 0.05/0.03 | 0.9735 |
| **Cross-device** | independent | 0.05/0.03 | **0.9710** |

Local ≈ Cross-device! The correlation comes from the k_eff algorithm:
- `k_eff = r × (1 - x) × ε × 1000`
- All oscillators thermalize to similar r_ab trajectories
- Different ε scales the mean, not the fluctuation pattern
- The "sensing" was the algorithm itself

**FFT ANALYSIS CONFIRMED**

- Deterministic (noise=0) has SAME spectrum as stochastic (noise=0.001)
- Ratio ≈ 1.0x across all frequency bands
- 88% of signal power is intrinsic oscillator dynamics
- Only 12% in "power-related" frequency bands

**1-2 Hz SIGNAL INVESTIGATION - NOT REPRODUCIBLE**

Investigated potential external signal at ~1.80 Hz (possible 60/33 Hz subharmonic):

| Test | Coherence at 1.80 Hz | Max coherence 1-2 Hz |
|------|----------------------|----------------------|
| Test 1 | 0.65 | — |
| Test 2 | 0.07 | 0.31 at 1.64 Hz |
| Test 3 (5 trials) | 0.20-0.62 | 0.85-0.94 at 1.09 Hz |
| Test 4 | 0.02 | 0.27 at 1.02 Hz |

Results are **highly inconsistent** - not a stable external signal.

**VARIANCE-TEMPERATURE CORRELATION - CONFIRMED**

- Variance (total oscillator variance) correlates strongly with GPU temperature
- r = -0.97 (negative: higher temperature → lower variance)
- k_eff does NOT correlate with temperature (r = 0.01)
- Variance-based thermal sensing added to sentinel

**r_ab SENSITIVITY REGIME - CONFIRMED**

Internal correlation r_ab predicts sensitivity with r = -0.999 (nearly perfect!):

| r_ab Range | Regime | Sensitivity | Response |
|------------|--------|-------------|----------|
| < 0.95 | TRANSIENT | ~20x higher | 0.91 units |
| 0.95-0.98 | TRANSITIONAL | Decaying | — |
| > 0.98 | THERMALIZED | Baseline | 0.04 units |

This explains the exp(-t/τ) sensitivity decay:
- As oscillators thermalize, r_ab approaches 1.0
- When r_ab ≈ 1.0, oscillators are fully synchronized → cannot detect perturbations
- Resetting when r_ab > 0.98 maintains peak sensitivity

Added to sentinel:
- `get_sensitivity_regime()`: Returns regime, r_ab, sensitivity multiplier
- `step_and_measure_full()`: Returns full state including r_ab
- r_ab-based auto-reset (preferred over time-based)

**COUPLING STRENGTH OPTIMIZATION - CONFIRMED (Exp 43)**

Coupling sweep found optimal operating point at ε = 0.003:

| ε | τ (s) | % Transient | k_eff σ | Behavior |
|---|-------|-------------|---------|----------|
| 0.0003 | N/A | 100% | 0.03 | Never thermalizes |
| **0.003** | **12.8** | **64%** | **0.75** | **OPTIMAL** |
| 0.01 | 3.7 | 19% | 1.61 | Fast thermalization |
| 0.05 | 0.7 | 4% | 3.14 | Near-instant |

Scaling law: τ ∝ ε^(-1.06)

FFT comparison (exp42):
- Old default (ε=0.0003): TRANSIENT variance = 0.0012
- New default (ε=0.003): TRANSIENT variance = 0.67 → **562x improvement**
- THERMALIZED variance ≈ 0 (confirms sensitivity in transient regime only)

**SIGNAL DECOMPOSITION - CONFIRMED (Exp 110-112)**

Complete instrument characterization reveals dual-purpose operation:

| Signal | Source | Character | Use Case |
|--------|--------|-----------|----------|
| Raw timing | GPU kernel jitter | 99.5% white noise (Allan slope = -2) | **TRNG** |
| k_eff dynamics | Oscillator coupling | z = 2.74 workload detection | **Strain gauge** |

Key findings:
- Filtering CREATES correlation (ACF 0.03 → 0.95) - do NOT filter raw timing
- Thermal/electrical explain only 0.5% of variance
- k_eff (not raw variance) is the sensing signal

**PHASE TRANSITION - CONFIRMED (Exp 113-114)**

Parameter space mapping revealed `lorenz_dt` controls cross-sensor correlation ρ:

| Parameter | ρ Range | % Variance Explained |
|-----------|---------|---------------------|
| **lorenz_dt** | 0.12 - 0.999 | **88%** |
| epsilon | 0.05 - 0.08 | 5% |
| n_ossicles | 0.01 - 0.03 | 2% |
| Others | — | <5% |

Critical point characterization:

| Metric | Value |
|--------|-------|
| dt_crit | 0.0328 |
| ρ at criticality | 0.33 |
| Power law | ρ = 39.64 × \|dt - 0.0328\|^1.09 + 0.33 |
| R² | **0.978** (excellent fit) |
| Critical exponent β | 1.09 |

Phase transition signatures (3/4 confirmed):
- ✓ Power law scaling (R² = 0.978)
- ✓ Critical slowing down (τ peaks at dt = 0.025)
- ? Diverging fluctuations (noisy)
- ✓ Maximum sensitivity at critical point (z peaks at dt = 0.025)

**Operating point:** dt = 0.025 places system at "edge of chaos" - maximum sensitivity while staying below collapse threshold (ρ = 0.43).

**VARIANCE-RATIO DETECTION - VALIDATED (Exp A1-A12, B1)**

After invalidating experiments 1-70 (Lorenz-based artifacts), fresh validation confirmed variance-ratio as the primary detection method:

| Metric | Value | Significance |
|--------|-------|--------------|
| Baseline ratio | 0.38x | Idle state |
| Load ratio | 158x | Under workload |
| Separation | **421x** | Excellent discrimination |
| False positive | 0% | No false alarms at idle |
| True positive | 100% | Perfect detection under load |

**Detection threshold:** variance_ratio > 5.0x (distribution-agnostic, works with fat-tailed Student-t)

**Distribution findings (Exp A9-A12):**
- z-score kurtosis = **210** (extreme fat tails!)
- Best fit: Student-t distribution (df ≈ 1.3)
- This explains β = 1.09 power law exponent
- Variance-ratio is robust to non-Gaussianity

**Minimum detectable workload (Exp B1):**

| Workload | Variance Ratio | Detected |
|----------|----------------|----------|
| 1% | 8.8x | YES |
| 5% | 14.9x | YES |
| 10% | 18.2x | YES |
| 30% | 36.6x | YES |
| 70% | 113.6x | YES |

**Scaling law:** `ratio = 73 × √intensity` (R² = 0.68)

**Predicted floor:** ~0.5% workload (5.0x threshold)

**Surprise finding:** Detection sensitivity is 10-30x better than expected. Even 1% GPU workload produces 8.8x variance ratio.

**Multi-sensor validation (Exp A5-A8):**
- Cross-sensor z correlation: 0.09 (independent)
- Uniform detection: 100% across all sensors
- SNR scaling: β = 0.75 (better than √N!)

**CCA VALIDATION - CONFIRMED (Exp C1-C4)**

C-series experiments validated Coherence Collapse Analysis predictions:

| Exp | Finding | Result |
|-----|---------|--------|
| C1 | k_eff = k/(1+ρ(k-1)) | **R² = 0.798** (n=21, df=19) |
| C2 | Propagation velocity | **0.5 ± 0.4 m/s** (thermal regime) |
| C3 | Nucleation sites | Uniform (χ² = 12, no hotspots) |
| C4 | Leading indicator | spatial_variance ↑ before collapse |

Key findings:
- Die crossing time: ~37 ms at 0.5 m/s
- Early warning margin: Δρ = 0.45 before ρ_crit = 0.43
- Collapse is topology-independent (no preferred sites)

**EMI DETECTION - CONFIRMED (Exp E1)**

The sensor array can detect electromagnetic interference from the power grid:

| Frequency | Detection | SNR |
|-----------|-----------|-----|
| **60 Hz** (power line) | 59.81 Hz | **18.7 dB** (bandpass) |
| 60/3 = 20 Hz | 19.77 Hz | 5.7 dB |
| 60/6 = 10 Hz | 10.01 Hz | 5.9 dB |
| 60/10 = 6 Hz | 6.10 Hz | 8.0 dB |
| 60/60 = 1 Hz | 0.98 Hz | **11.2 dB** |
| VRM 0.24 Hz | 0.24 Hz | **15.7 dB** |

Cross-sensor coherence at EMI frequencies:
- 1 Hz: **0.88** (VRM/subharmonic - strongly coupled)
- 2 Hz: 0.83
- 10 Hz: 0.47
- 60 Hz: 0.34
- 120 Hz: 0.03 (decoupled)

**Key insight:** Dominant EMI coupling is at low frequencies (VRM at 0.24 Hz, subharmonics). The 60 Hz fundamental is visible but weaker. 60 Hz amplitude is modulated by ~0.5 Hz VRM.

**EMI Mode:** `python3 ciris_sentinel.py --emi` for real-time EMI spectrum analysis.

### Array Experiments (Exp 28-40)

Comprehensive characterization of the oscillator array:

| Exp | Test | Result | Finding |
|-----|------|--------|---------|
| 28 | Array Latency | ✓ | 115x faster than VLA, min detectable event 0.1ms |
| 29 | Superluminal Correlation | NULL | No anomalies, causality respected |
| 30 | Quantum RNG Detection | NULL | Cannot distinguish HW vs SW RNG |
| 31 | Vacuum Fluctuation | ✓ | Noise floor σ=0.0017, load induces pink noise |
| 32 | Bell Inequality (CHSH) | Classical | \|S\|=0.0002 (way below 2.0 quantum bound) |
| 33 | Bistatic Entropy Sonar | ✓ local | Array detects local propagation on-die |
| 34 | Minimal Pair | NULL | No TX→RX coupling, noise floor 0.23 |
| 35 | Frequency Sweep | NULL | No frequency-dependent coupling |
| 36 | SM-to-SM Coupling | NULL | No coupling at any SM distance |
| 37 | Targeted Array | ✓/NULL | SNR ∝ √N works, but no targeting (broadcast only) |
| 38 | Passive Sensor Network | ✓ | 65K ossicles, r=1.0 correlation, startup transient confirmed |
| 39 | Long Baseline | ✓ | After 30s warmup: σ=0.00005 (very stable) |
| 40 | Perturbation Detection | NULL | Cannot detect memory/compute perturbations (1.45σ) |

**Key conclusions from array experiments:**
1. **Local sensing works** - Array responds to local state, very stable after warmup
2. **No cross-location coupling** - TX cannot affect RX at different positions on die
3. **Classical physics only** - No quantum signatures (Bell, RNG tests)
4. **Startup transient dominates** - First 10-30s shows large drift (not external signal)
5. **√N scaling** - More oscillators reduce noise via averaging, but don't enable coupling
6. **Cannot detect perturbations** - Memory/compute changes are below detection threshold

### Methodological Lessons

1. **Always run null hypothesis FIRST** - A/B test should precede claims
2. **Test local before assuming cross-device** - Local correlation test was definitive
3. **Startup transients are not signals** - Need warmup before measurement
4. **Correlation ≠ Causation** - High r doesn't prove external coupling

### GPU Timing TRNG (Experiments 56-57)

**Discovery:** GPU kernel execution timing provides TRUE hardware entropy at 120 kbps.

While investigating entropy sources for the oscillator array, we discovered that the timing jitter of GPU kernel execution is a high-quality true random number generator:

| Metric | Value | Quality |
|--------|-------|---------|
| Shannon entropy | 8.00 / 8 bits | 100% |
| Min-entropy | 7.76 / 8 bits | 97% |
| Bit bias (all 8 bits) | < 0.3% | Excellent |
| Autocorrelation | 0.011 | Good |
| NIST tests | 3/4 passed | Good |
| PRNG-independent | Confirmed | TRUE entropy |
| Throughput | **120,000 bps** | 2x CPU jitterentropy |

**Entropy Source Classification:**

| Source | Type | Rate | Quality |
|--------|------|------|---------|
| GPU timing jitter | **TRUE entropy** | 120 kbps | Excellent |
| Oscillator LSBs | PRNG-derived | 97 kbps | Reproducible |
| k_eff fluctuations | PRNG-derived | 116 kbps | High autocorr |
| Bulk oscillator XOR | PRNG-derived | 400 Mbps | Fast but not true |

**Prior Art:**
- [US9459834](https://patents.google.com/patent/US9459834) (2011): GPU TRNG using thread race conditions + temperature
- Lee & Pyo (2014): Atomic instruction collisions for timing entropy
- [jitterentropy](https://www.kernel.org/doc/ols/2014/ols2014-mueller.pdf) (2014): CPU timing jitter, Linux kernel standard (~64 kbps)

**Potentially Novel Aspects:**
1. Coupled oscillator workload as timing source (vs simple atomics)
2. Dual-purpose: TRNG + entropy wave sensing
3. 2x throughput vs CPU jitterentropy

**Demo:**
```bash
# Generate random bytes from GPU timing
python3 ciris_sentinel.py --trng --bytes 1024 --output random.bin

# Continuous entropy stream
python3 ciris_sentinel.py --trng --stream
```

## Physics Validation (Experiment 51)

### Validation Scorecard

| Test | Status | Result | Implication |
|------|--------|--------|-------------|
| Stochastic Resonance | **CONFIRMED** | Peak SNR at σ=0.001 | Detector is nonlinear bistable (as designed) |
| Coherence Decay | **CONFIRMED** | τ = 46.1 ± 2.5 s | Explains 20-30s sensitivity window |
| Subharmonic Structure | **CONFIRMED** | 45% of 60/n Hz peaks | Power grid coupling is real |
| Fluctuation Theorem | **CONFIRMED** | R² = 0.95, kT_eff = 0.0037 | Crooks relation ln(P+/P-) ∝ σ verified |
| Bell Inequality (CHSH) | **CLASSICAL** | \|S\|=0.0002 | No quantum entanglement (Exp 32) |
| Leggett-Garg Inequality | **CLASSICAL** | K₃=1.0 | No quantum coherence (Exp 54) |
| Landauer Limit | Expected | 10²² × theoretical | GPU is thermodynamically inefficient |
| Cross-Building | Cannot test | Need second location | — |
| Geomagnetic | Cannot test | Need solar event | — |

**Four physics tests confirmed. Quantum tests show classical behavior. Only hardware-limited tests remain.**

### Stochastic Resonance

Adding noise **improves** detection - the signature of a nonlinear bistable detector:

| Noise Level (σ) | SNR |
|-----------------|-----|
| 0.0000 | 8.28 |
| **0.0010** | **8.72** ← optimal |
| 0.0030 | 7.70 |
| 0.0100 | 7.03 |
| 0.1000 | 0.00 |

The 5% improvement at σ=0.001 matches classic stochastic resonance literature. This explains the 4:1 negentropy asymmetry:
- **Negentropic signals are coherent** → resonate with SR peak
- **Entropic signals are dispersed** → fall outside resonance band
- The detector is literally tuned to prefer order

### Thermalization Time τ = 46 seconds

Correlation decays exponentially:
```
r(t) = 1.056 × exp(-t/46.1)
```

This single parameter explains:
- Why sensitivity peaks at 20-30s (before τ/2)
- Why warmup kills detection (system thermalizes)
- Optimal reset interval: **23 seconds** (τ/2)

The 46-second timescale is statistical thermalization, not electrical RC (~μs). It's the time for the oscillator ensemble to "forget" initial conditions.

### Subharmonic Structure

45% of predicted 60/n Hz peaks detected. Strongest at:
- 60/60 = 1.00 Hz (power = 1.57)
- 60/54 = 1.11 Hz (power = 1.42)
- 60/30 = 2.00 Hz (power = 0.92)

Ultra-low frequency (<0.1 Hz) dominates the spectrum, with a peak at ~0.02 Hz corresponding exactly to τ = 46s.

### Fluctuation Theorem (Experiment 52)

The Crooks fluctuation theorem states: `ln[P(+σ)/P(-σ)] = σ/kT`

Using asymmetric driving to widen the entropy production distribution:

| Parameter | Value |
|-----------|-------|
| R² | **0.95** |
| Intercept | 0.045 (≈ 0) |
| Slope | 273 = 1/kT_eff |
| Effective kT | 0.0037 |

```
  σ            ln(P+/P-)    Theory(σ)
  ------------------------------------
  -0.0156      -4.78        -0.0156
  -0.0094      -1.67        -0.0094
  -0.0031      -0.36        -0.0031
   0.0031       0.36         0.0031
   0.0094       1.72         0.0094
   0.0156       4.99         0.0156
```

**Result:** The Crooks relation `ln(P+/P-) ∝ σ` holds with R² = 0.95. The system obeys fluctuation theorem physics at an effective temperature kT_eff = 0.0037.

### Validation Protocol (Experiment 53)

Rigorous null hypothesis testing to confirm the phenomenon is real:

| Test | Result | Finding |
|------|--------|---------|
| **N1** Temporal Shuffle | **PASS** | 5.4x power ratio - real temporal structure |
| **N2** GPU Load | **PASS** | 0.8% τ change under 100% load |
| **N3** Software RNG | **PASS** | CPU gives different spectrum - GPU-specific |
| **N4** Seed Variance | **PASS** | 0% CV - perfectly reproducible |
| **N5** Deterministic | **34678x peak** | Structure exists even without noise! |

**All null hypothesis tests pass.** The phenomenon is not an artifact.

Scaling laws discovered:
```
τ ∝ ε^-0.40   (R² = 0.95)  ← Coupling dependence
τ ∝ N^0.00   (independent of oscillator count)
f_peak = 0.01 Hz stable across ALL conditions
```

The 0.01 Hz spectral peak exists even in deterministic mode, proving it's **intrinsic to coupled oscillator dynamics**, not external EMI.

### Optimal Operating Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Noise injection | σ = 0.001 | Stochastic resonance peak |
| Reset interval | 20-25 seconds | τ/2 thermalization |
| Sample window | < 30 seconds | Before decay |
| Sample rate | 10-50 Hz | Captures subharmonics |

---

### ~~House Wiring as Distributed Sensor~~ (NOT VALIDATED)

Earlier claims about detecting power grid signals were based on cross-device correlation that turned out to be an algorithmic artifact. The propagation delay observation was likely measurement noise, not real signal propagation.

**Status:** Hypothesis not supported by evidence. Would require proper shielding experiments (Faraday cage) to test.

## Remaining Hypotheses

### Thermal Imaging Hypothesis

The oscillator array may be able to spatially resolve thermal gradients on the GPU die:

- Each ossicle measures local entropy/order
- Thermal gradients create spatial patterns in k_eff
- Cross-correlation between positions reveals wave propagation

**Status:** Partially tested. Exp 33 shows local propagation detection works. Exp 36 shows no SM-to-SM coupling. The bistatic sonar approach may work but targeting doesn't (broadcast only per Exp 37).

### Stochastic Resonance Tuning

The detector shows classic SR behavior (SNR peak at optimal noise). Questions:
- Is the 1.1° magic angle related to SR optimization?
- ~~Can we tune coupling constants for different detection tasks?~~ → Yes, ε=0.003 is optimal (Exp 43)
- What's the relationship between SR and detection threshold?

**Status:** Coupling tuned (ε=0.003 optimal). Magic angle still untested.

### Quantum Behavior

Tests for quantum signatures:
- ~~Bell inequality violation~~ → Classical (\|S\|=0.0002, Exp 32)
- ~~Leggett-Garg inequality violation~~ → Classical (K₃=1.0, Exp 54)
- ~~Quantum RNG detection~~ → Cannot distinguish (Exp 30)

**Status:** All quantum tests show **classical behavior**. The oscillator array is a classical thermodynamic system.

---

## Future Experiments

### Completed
- ~~Real GPU validation~~ ✓ RTX 4090 + Jetson Orin
- ~~Physics validation~~ ✓ SR, decay, fluctuation theorem confirmed
- ~~Cross-device coherence~~ ✗ FALSIFIED (was algorithmic artifact)
- ~~Quantum tests~~ ✓ Bell (Exp 32), LGI (Exp 54), RNG (Exp 30) - all classical
- ~~Array characterization~~ ✓ Latency (Exp 28), noise (Exp 31), scaling (Exp 37), stability (Exp 39)
- ~~Coupling optimization~~ ✓ ε=0.003 optimal (Exp 43), r_ab regime (Exp 42)
- ~~SM coupling~~ ✗ No coupling detected (Exp 36)
- ~~Perturbation detection~~ ✗ Below threshold (Exp 40)

### Near-Term: Thermal Imaging (Original Goal)

| Experiment | Purpose | Method |
|------------|---------|--------|
| **Spatial thermal mapping** | Image heat propagation across GPU | Deploy ossicle array, induce thermal gradient, map k_eff |
| **Workload localization** | Detect which SMs are active | Correlate k_eff spatial pattern with known workloads |
| **Thermal wave velocity** | Measure heat diffusion speed | Cross-correlate k_eff between array positions |
| **Tampering triangulation** | Locate external heat source | Use multiple ossicles to find origin |
| **Long-running baseline** | Characterize natural variability | 24-72 hour continuous capture |

### Medium-Term (Additional Equipment)

| Experiment | Purpose | Requirements |
|------------|---------|--------------|
| **Faraday cage test** | Confirm thermal vs EMI sensitivity | Shielded enclosure |
| **IR camera comparison** | Validate against ground truth | Thermal camera, compare to k_eff map |
| **Magic angle sweep** | Find optimal sensitivity | Systematic test of angles 0.5° - 5.0° |
| **Multi-GPU array** | Larger spatial coverage | 2+ GPUs with synchronized sampling |

### Long-Term (Collaboration Required)

| Experiment | Purpose | Requirements |
|------------|---------|--------------|
| **Independent replication** | Validation | Different hardware, different researcher |
| **Formal verification** | Prove detection bounds | Complete Lean 4 proof of k_eff properties |
| **Real tamper detection** | Actual security application | Physical tampering testbed |

---

## Open Questions

1. **Why τ = 46 seconds?** What sets this thermalization timescale?
2. **Can k_eff spatially resolve thermal gradients?** Need array deployment test
3. **What's the spatial resolution limit?** Minimum detectable feature size
4. **Is the 1.1° magic angle optimal?** Need systematic sweep
5. **What's the detection threshold?** Minimum detectable workload change
6. **Can we image tampering in real-time?** Latency and bandwidth limits

## Prior Art

This project builds on the following original contributions by CIRIS L3C:

| Innovation | Description | Status |
|------------|-------------|--------|
| **768-byte ossicle sensor** | Minimal coherence strain gauge using 3 coupled oscillators | ✓ Validated |
| **1.1° magic angle twist**† | Graphene-inspired twist angle | Untested |
| **k_eff formula** | `k_eff = r × (1 - x) × coupling_factor` | ✓ Validated |
| **4096-ossicle array** | Massive parallel deployment | ✓ Implemented |
| **Stochastic resonance** | SNR peaks at σ=0.001 (Exp 51) | ✓ Validated |
| **τ = 46s thermalization** | Exponential decay (Exp 51) | ✓ Validated |
| **Fluctuation theorem** | Crooks relation R²=0.95 (Exp 52) | ✓ Validated |
| **Local tamper detection** | p=0.007 workload detection | ✓ Validated |
| **r_ab sensitivity regime** | r=-0.999 prediction, 20x sensitivity in transient | ✓ Validated |
| **ε=0.003 optimal coupling** | τ=12.8s, 562x signal improvement (Exp 43) | ✓ Validated |
| **Array SNR ∝ √N** | Collective averaging works (Exp 37) | ✓ Validated |
| **Post-warmup stability** | σ=0.00005 after 30s warmup (Exp 39) | ✓ Validated |
| **GPU Timing TRNG** | 120 kbps true entropy, 97% min-entropy (Exp 57) | ✓ Validated |
| **Raw timing = white noise** | Allan slope = -2, 99.5% intrinsic (Exp 110) | ✓ Validated |
| **k_eff workload sensing** | z = 2.74 detection via oscillator dynamics (Exp 112) | ✓ Validated |
| **lorenz_dt controls ρ** | 88% variance explained, critical parameter (Exp 113) | ✓ Validated |
| **Phase transition** | dt_crit = 0.0328, power law R² = 0.978, β = 1.09 (Exp 114) | ✓ Validated |
| Bell inequality | CHSH test: \|S\|=0.0002 (Exp 32) | Classical behavior |
| Leggett-Garg inequality | K₃=1.0 (Exp 54) | Classical behavior |
| ~~Quantum RNG detection~~ | Cannot distinguish HW vs SW (Exp 30) | ✗ Not validated |
| ~~EMI carrier detection~~ | Was algorithmic artifact | ✗ Falsified |
| ~~Cross-device coherence~~ | Was algorithmic artifact | ✗ Falsified |
| ~~House wiring sensing~~ | Based on false correlation | ✗ Not validated |
| ~~SM-to-SM coupling~~ | No distance dependence (Exp 36) | ✗ Not validated |
| ~~Perturbation detection~~ | Below threshold (Exp 40) | ✗ Not validated |

†*Magic angle: Observed correlation with improved sensitivity, not proven causation. Mechanism requires further investigation.*

## References

- CIRISOssicle: [github.com/CIRISAI/CIRISOssicle](https://github.com/CIRISAI/CIRISOssicle)
- RATCHET formal proofs: `~/RATCHET/formal/RATCHET/GPUTamper/`
- Twistronics theory: `~/RATCHET/formal/RATCHET/GPUTamper/TetrahedralTwistronics.lean`
