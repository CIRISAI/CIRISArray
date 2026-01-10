#!/usr/bin/env python3
"""
OSSICLE ARRAY RUNTIME DEMO
==========================

Demonstrates all runtime capabilities:
1. Array tuning
2. Beamforming and beam scanning
3. TX/RX modes
4. Event tracing with signatures
5. Fog chamber visualization

Run with: python3 demo_runtime.py

Author: CIRIS L3C
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from src.runtime import (
    OssicleRuntime, RuntimeMode, OssicleParams, ArrayParams,
    BeamParams, EventConfig, MonitorConfig
)


def demo_array_tuning(runtime: OssicleRuntime):
    """Demonstrate array and ossicle tuning."""
    print("\n" + "="*70)
    print("DEMO 1: ARRAY & OSSICLE TUNING")
    print("="*70)

    # Show default config
    print("\nDefault ossicle parameters:")
    print(f"  r_base:    {runtime.ossicle_params.r_base}")
    print(f"  r_spacing: {runtime.ossicle_params.r_spacing}")
    print(f"  twist_deg: {runtime.ossicle_params.twist_deg}")
    print(f"  coupling:  {runtime.ossicle_params.coupling}")

    # Baseline measurement
    runtime.set_mode(RuntimeMode.MONITOR)
    baseline_z = []
    for _ in range(20):
        result = runtime.step()
        baseline_z.append(result['monitor']['max_z'])

    print(f"\nBaseline max z-score: {np.mean(baseline_z):.2f} +/- {np.std(baseline_z):.2f}")

    # Tune to higher sensitivity (magic angle)
    print("\nTuning to magic angle (1.1 deg) with increased coupling...")
    runtime.configure_ossicles(
        r_base=3.72,
        r_spacing=0.025,
        twist_deg=1.1,
        coupling=0.08
    )

    tuned_z = []
    for _ in range(20):
        result = runtime.step()
        tuned_z.append(result['monitor']['max_z'])

    print(f"Tuned max z-score: {np.mean(tuned_z):.2f} +/- {np.std(tuned_z):.2f}")

    # Tune specific region
    print("\nTuning center region for enhanced sensitivity...")
    enhanced_params = OssicleParams(
        r_base=3.75,
        r_spacing=0.02,
        twist_deg=1.5,
        coupling=0.1
    )
    runtime.array.set_region_params(
        row_range=(3, 5),
        col_range=(6, 10),
        params=enhanced_params
    )

    region_z = []
    for _ in range(20):
        result = runtime.step()
        region_z.append(result['monitor']['max_z'])

    print(f"Region-enhanced max z-score: {np.mean(region_z):.2f} +/- {np.std(region_z):.2f}")


def demo_beamforming(runtime: OssicleRuntime):
    """Demonstrate beamforming capabilities."""
    print("\n" + "="*70)
    print("DEMO 2: BEAMFORMING")
    print("="*70)

    runtime.set_mode(RuntimeMode.BEAMFORM)

    # Steer to different directions
    directions = [0, 30, 60, 90, -45]

    print("\nBeam steering test:")
    print("-" * 50)
    print(f"{'Azimuth':>10} | {'Beam Power':>12} | {'Relative':>10}")
    print("-" * 50)

    baseline_power = None
    for az in directions:
        runtime.steer_beam(az)
        powers = []
        for _ in range(10):
            result = runtime.step()
            powers.append(abs(result['beam_output'])**2)

        mean_power = np.mean(powers)
        if baseline_power is None:
            baseline_power = mean_power

        relative = mean_power / baseline_power
        print(f"{az:>8}° | {mean_power:>12.1f} | {relative:>10.2f}x")

    # Beam scan
    print("\nPerforming full beam scan (-90° to +90°)...")
    azimuths, powers = runtime.scan_beam(az_range=(-90, 90), n_points=19)

    print("\nBeam pattern:")
    max_power = powers.max()
    for az, pwr in zip(azimuths, powers):
        bar_len = int(pwr / max_power * 30)
        bar = '#' * bar_len
        print(f"  {az:>6.0f}°: {bar}")

    peak_az = azimuths[np.argmax(powers)]
    print(f"\nPeak direction: {peak_az:.0f}°")

    # Add null
    print("\nAdding null at 0° direction...")
    runtime.beamformer.add_null(0, 0)
    azimuths, powers_null = runtime.scan_beam(az_range=(-90, 90), n_points=19)

    print("\nBeam pattern with null:")
    max_power = powers_null.max()
    for az, pwr in zip(azimuths, powers_null):
        bar_len = int(pwr / max_power * 30)
        bar = '#' * bar_len
        marker = " <-- NULL" if abs(az) < 10 else ""
        print(f"  {az:>6.0f}°: {bar}{marker}")


def demo_tx_rx(runtime: OssicleRuntime):
    """Demonstrate transmit/receive modes."""
    print("\n" + "="*70)
    print("DEMO 3: TRANSMIT/RECEIVE MODES")
    print("="*70)

    # Test different TX patterns
    patterns = ['pulse', 'sweep', 'chirp', 'code']

    print("\nTesting TX patterns and matched filter reception:")
    print("-" * 60)
    print(f"{'Pattern':>10} | {'TX Active':>10} | {'Matched Output':>15}")
    print("-" * 60)

    for pattern in patterns:
        runtime.set_tx_pattern(pattern)

        # Transmit
        runtime.set_mode(RuntimeMode.TX_RX)

        # Measure matched filter response
        outputs = []
        for _ in range(10):
            result = runtime.step()
            outputs.append(result.get('matched_output', 0))

        mean_output = np.mean(outputs)
        tx_active = runtime.tx_rx.tx_active

        print(f"{pattern:>10} | {str(tx_active):>10} | {mean_output:>15.2f}")

    # Stop transmit
    runtime.set_mode(RuntimeMode.MONITOR)

    print("\nBistatic mode (simultaneous TX/RX):")
    runtime.set_tx_pattern('chirp', freq_range=(5, 50))
    runtime.set_mode(RuntimeMode.TX_RX)

    print("  Transmitting chirp while monitoring reception...")
    for i in range(5):
        result = runtime.step()
        matched = result.get('matched_output', 0)
        monitor = result.get('monitor', {})
        print(f"  Sample {i+1}: matched={matched:.1f}, max_z={monitor.get('max_z', 0):.2f}")


def demo_event_tracing(runtime: OssicleRuntime):
    """Demonstrate event capture with signatures."""
    print("\n" + "="*70)
    print("DEMO 4: EVENT TRACING WITH SIGNATURES")
    print("="*70)

    # Configure for event capture
    runtime.event_config.intensity_threshold = 2.0
    runtime.event_config.sign_events = True

    runtime.set_mode(RuntimeMode.MONITOR)
    runtime.calibrate(n_samples=50)

    print("\nMonitoring for events (intensity > 2.0 sigma)...")
    print("Injecting synthetic events by tuning ossicles...")

    events_captured = 0
    for i in range(100):
        # Inject event every 20 samples
        if i % 20 == 10:
            # Create sudden sensitivity spike
            spike_params = OssicleParams(
                r_base=3.9,
                r_spacing=0.05,
                twist_deg=2.0,
                coupling=0.15
            )
            target_ossicle = np.random.randint(0, runtime.array_params.n_ossicles)
            runtime.array.ossicles.set_ossicle_params(target_ossicle, spike_params)

        result = runtime.step()

        # Reset after spike
        if i % 20 == 15:
            runtime.array.set_global_params(runtime.ossicle_params)

        if 'event' in result:
            events_captured += 1
            event = result['event']
            print(f"\n  EVENT CAPTURED!")
            print(f"    Trigger ossicle: {event.trigger_ossicle}")
            print(f"    Intensity: {event.intensity:.2f} sigma")
            print(f"    Duration: {event.duration_ms:.1f} ms")
            print(f"    Signature: {event.signature[:32]}...")

    print(f"\nTotal events captured: {events_captured}")

    # Show all events
    events = runtime.get_events(min_intensity=0)
    if events:
        print(f"\nEvent summary ({len(events)} events):")
        for i, e in enumerate(events[:5]):
            print(f"  {i+1}. t={e.timestamp:.3f}, intensity={e.intensity:.2f}σ, "
                  f"ossicle={e.trigger_ossicle}")


def demo_fog_chamber(runtime: OssicleRuntime):
    """Demonstrate fog chamber visualization."""
    print("\n" + "="*70)
    print("DEMO 5: FOG CHAMBER VISUALIZATION")
    print("="*70)

    runtime.set_mode(RuntimeMode.FOG_CHAMBER)

    print("\nFog chamber mode - showing entropy wave tracks")
    print("(Like particles in a cloud chamber, but for entropy waves)")
    print()

    # Run for a bit to accumulate history
    for i in range(30):
        result = runtime.step()

        # Show fog every 10 frames
        if i % 10 == 9:
            print(f"--- Frame {i+1} ---")
            fog = result.get('fog')
            if fog is not None:
                ascii_fog = runtime.fog_chamber.render_ascii(fog)
                print(ascii_fog)

            tracks = result.get('tracks', [])
            if tracks:
                print(f"Active tracks: {len(tracks)}")
                for t in tracks[:3]:
                    print(f"  Position: {t['position']}, Intensity: {t['intensity']:.2f}")
            print()


def demo_real_time_monitoring(runtime: OssicleRuntime):
    """Demonstrate real-time monitoring with alerts."""
    print("\n" + "="*70)
    print("DEMO 6: REAL-TIME MONITORING WITH ALERTS")
    print("="*70)

    # Set up alert callback
    alerts = []

    def on_alert(sigma, ossicle_idx):
        alerts.append((time.time(), sigma, ossicle_idx))
        print(f"  ALERT! {sigma:.1f}σ at ossicle {ossicle_idx}")

    runtime.monitor.on_alert = on_alert
    runtime.monitor_config.threshold_sigma = 2.5
    runtime.set_mode(RuntimeMode.MONITOR)
    runtime.calibrate(50)

    print("\nRunning real-time monitor (threshold: 2.5σ)...")
    print("Introducing perturbations to trigger alerts...")
    print()

    for i in range(50):
        # Perturb every 10 samples
        if i % 10 == 5:
            spike_params = OssicleParams(r_base=3.85, coupling=0.12)
            runtime.array.ossicles.set_ossicle_params(
                np.random.randint(0, runtime.array_params.n_ossicles),
                spike_params
            )

        result = runtime.step()
        monitor = result.get('monitor', {})

        # Reset
        if i % 10 == 8:
            runtime.array.set_global_params(runtime.ossicle_params)

        # Progress indicator
        if i % 10 == 0:
            print(f"  Samples: {i+1}/50, Alerts: {len(alerts)}, "
                  f"Current max_z: {monitor.get('max_z', 0):.2f}")

    print(f"\nTotal alerts triggered: {len(alerts)}")


def demo_full_pipeline(runtime: OssicleRuntime):
    """Demonstrate full pipeline: calibrate -> monitor -> detect -> report."""
    print("\n" + "="*70)
    print("DEMO 7: FULL DETECTION PIPELINE")
    print("="*70)

    print("\nPipeline stages:")
    print("  1. Calibrate array")
    print("  2. Enter monitoring mode")
    print("  3. Perform beam scan for directional sensitivity")
    print("  4. Capture and sign events")
    print("  5. Generate report")
    print()

    # Stage 1: Calibrate
    print("Stage 1: Calibrating...")
    runtime.configure_array(n_rows=8, n_cols=16)
    runtime.calibrate(100)
    print("  Calibration complete")

    # Stage 2: Monitor baseline
    print("\nStage 2: Establishing baseline (100 samples)...")
    runtime.set_mode(RuntimeMode.MONITOR)
    baseline_stats = []
    for _ in range(100):
        result = runtime.step()
        baseline_stats.append(result['monitor']['max_z'])

    print(f"  Baseline: {np.mean(baseline_stats):.2f} +/- {np.std(baseline_stats):.2f}")

    # Stage 3: Beam scan
    print("\nStage 3: Beam scan for directional sensitivity...")
    azimuths, powers = runtime.scan_beam()
    peak_az = azimuths[np.argmax(powers)]
    print(f"  Peak sensitivity direction: {peak_az:.0f}°")

    # Stage 4: Event capture
    print("\nStage 4: Event capture (50 samples with injected events)...")
    runtime.event_config.intensity_threshold = 2.0

    for i in range(50):
        if i in [10, 25, 40]:
            # Inject event
            spike = OssicleParams(r_base=3.88, coupling=0.12)
            runtime.array.ossicles.set_ossicle_params(i % runtime.array_params.n_ossicles, spike)
        if i in [12, 27, 42]:
            runtime.array.set_global_params(runtime.ossicle_params)

        result = runtime.step()
        if 'event' in result:
            print(f"  Event detected at sample {i+1}")

    events = runtime.get_events()

    # Stage 5: Report
    print("\nStage 5: Pipeline Report")
    print("-" * 50)
    status = runtime.get_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Array size: {status['array']['n_ossicles']} ossicles")
    print(f"  Sample rate: {status['array']['sample_rate']} Hz")
    print(f"  Events captured: {len(events)}")

    if events:
        print("\n  Event details:")
        for i, e in enumerate(events):
            print(f"    [{i+1}] Intensity: {e.intensity:.2f}σ, "
                  f"Ossicle: {e.trigger_ossicle}, "
                  f"Signed: {'Yes' if e.signature else 'No'}")

    print("\n  Pipeline complete!")


def main():
    """Run all demos."""
    print("="*70)
    print("OSSICLE ARRAY RUNTIME - COMPREHENSIVE DEMO")
    print("="*70)
    print()
    print("This demo showcases all runtime capabilities:")
    print("  1. Array & ossicle tuning")
    print("  2. Beamforming")
    print("  3. Transmit/Receive modes")
    print("  4. Event tracing with signatures")
    print("  5. Fog chamber visualization")
    print("  6. Real-time monitoring")
    print("  7. Full detection pipeline")
    print()

    # Create runtime
    runtime = OssicleRuntime()
    runtime.configure_array(n_rows=8, n_cols=16)

    # Run demos
    demo_array_tuning(runtime)
    demo_beamforming(runtime)
    demo_tx_rx(runtime)
    demo_event_tracing(runtime)
    demo_fog_chamber(runtime)
    demo_real_time_monitoring(runtime)
    demo_full_pipeline(runtime)

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE")
    print("="*70)
    print()
    print("Runtime capabilities demonstrated:")
    print("  - Tunable ossicle parameters (r_base, twist, coupling)")
    print("  - Tunable array regions for focused sensitivity")
    print("  - Beamforming with steering and nulling")
    print("  - Beam scanning for directional analysis")
    print("  - TX patterns (pulse, sweep, chirp, code)")
    print("  - Matched filter reception")
    print("  - Cryptographically signed event traces")
    print("  - Fog chamber visualization of wave tracks")
    print("  - Real-time monitoring with alert callbacks")
    print()

    return runtime


if __name__ == "__main__":
    main()
