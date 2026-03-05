"""
PlantIQ — Power Curve Generator
=================================
Generates 2000 synthetic power consumption curves (1800 timesteps each)
corresponding to 30-minute batch process runs at 1Hz sampling rate.

Reads batch_data.csv to get fault_type labels and batch parameters,
then generates physically realistic power curves with appropriate
fault signatures.

Outputs:
  - data/power_curves/B0000.npy ... B1999.npy  (2000 files)

Curve shape (normal profile):
  - Startup ramp:   0–120s     (power rises from ~2 kW to ~5 kW)
  - Plateau:        120–1620s  (steady ~5–7 kW with minor fluctuation)
  - Cooldown:       1620–1800s (power drops back to ~2 kW)

Fault signatures:
  - bearing_wear:       gradual baseline rise (+0.003 kW/s)
  - wet_material:       irregular spikes in first 600s
  - calibration_needed: elevated flat baseline (+0.6 kW constant)
"""

import os
import sys
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TIMESTEPS = 1800           # 30 minutes at 1Hz
RANDOM_SEED = 42
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_CSV = os.path.join(DATA_DIR, "batch_data.csv")
CURVES_DIR = os.path.join(DATA_DIR, "power_curves")

np.random.seed(RANDOM_SEED)


def generate_normal_curve(
    batch_size: float = 500.0,
    conveyor_speed: float = 75.0,
    temperature: float = 183.0,
) -> np.ndarray:
    """
    Generate a normal (healthy) power consumption curve.

    The curve follows a realistic industrial batch profile:
    - Startup phase:  exponential ramp (0–120s)
    - Plateau phase:  steady-state with small random fluctuations (120–1620s)
    - Cooldown phase: exponential decay (1620–1800s)

    Parameters influence the curve:
    - batch_size:     affects plateau height (heavier batches draw more power)
    - conveyor_speed: affects plateau height (faster speed = more power)
    - temperature:    affects startup ramp rate
    """
    t = np.arange(TIMESTEPS, dtype=np.float64)

    # Base plateau power: scaled by batch/speed parameters
    size_factor = 0.002 * (batch_size - 300)       # 0–0.8 kW range
    speed_factor = 0.03 * (conveyor_speed - 60)     # 0–1.05 kW range
    plateau_power = 5.0 + size_factor + speed_factor  # ~5.0–6.85 kW

    # Startup ramp: exponential rise with temperature-influenced rate
    temp_factor = 0.008 * (temperature - 175)        # Faster ramp at higher temp
    startup_tau = 80.0 - temp_factor * 10            # Time constant
    startup = plateau_power * (1 - np.exp(-t / startup_tau))

    # Cooldown decay: exponential fall starting at t=1620
    cooldown_start = 1620
    cooldown_tau = 60.0
    cooldown = np.where(
        t > cooldown_start,
        plateau_power * np.exp(-(t - cooldown_start) / cooldown_tau),
        plateau_power
    )

    # Combine phases
    curve = np.minimum(startup, cooldown)

    # Add realistic Gaussian noise (small fluctuations)
    noise = np.random.normal(0, 0.12, size=TIMESTEPS)
    curve += noise

    # Add minor periodic vibration pattern (machine resonance)
    vibration = 0.08 * np.sin(2 * np.pi * t / 120)    # 2-minute cycle
    curve += vibration

    # Ensure physical bounds (power can't be negative)
    curve = np.clip(curve, 0.5, 9.5)

    return curve.astype(np.float32)


def apply_bearing_wear(curve: np.ndarray) -> np.ndarray:
    """
    Bearing wear fault: gradual baseline rise (+0.003 kW/second).

    This simulates increasing friction in a degrading bearing.
    The rise is linear and cumulative — by the end of 30 minutes,
    the baseline has risen by ~5.4 kW above normal.
    """
    t = np.arange(len(curve), dtype=np.float64)

    # Gradual linear rise in power draw
    rise = 0.003 * t

    # Add slight acceleration in the second half (bearing degradation compounds)
    acceleration = 0.0000005 * t ** 2
    acceleration = np.clip(acceleration, 0, 1.5)

    # Combine fault signature with base curve
    faulted = curve + rise + acceleration

    return np.clip(faulted, 0.5, 15.0).astype(np.float32)


def apply_wet_material(curve: np.ndarray) -> np.ndarray:
    """
    Wet material fault: irregular spikes in the first 600 seconds.

    High-moisture raw material causes irregular power surges as the
    material enters the conveyor system. Spikes are random in timing
    and amplitude, concentrated in the first third of the batch.
    """
    faulted = curve.copy()

    # Generate random spike locations in first 600 seconds
    num_spikes = np.random.randint(15, 40)
    spike_positions = np.random.randint(0, 600, size=num_spikes)

    for pos in spike_positions:
        # Each spike has random amplitude (0.5–2.5 kW above baseline)
        amplitude = np.random.uniform(0.5, 2.5)
        # Spike width: 3–15 seconds
        width = np.random.randint(3, 15)
        end = min(pos + width, len(faulted))

        # Gaussian-shaped spike
        spike_t = np.arange(end - pos)
        spike_center = width / 2
        spike = amplitude * np.exp(-0.5 * ((spike_t - spike_center) / (width / 4)) ** 2)
        faulted[pos:end] += spike

    # Also add general elevated noise in first 600s
    faulted[:600] += np.random.uniform(0.2, 0.8, size=600)

    return np.clip(faulted, 0.5, 12.0).astype(np.float32)


def apply_calibration_needed(curve: np.ndarray) -> np.ndarray:
    """
    Calibration fault: elevated flat baseline (+0.6 kW constant).

    A miscalibrated sensor or machine draws a constant offset of
    additional power throughout the entire batch. The curve shape
    is normal but shifted upward.
    """
    # Constant offset
    offset = np.random.uniform(0.4, 0.8)
    faulted = curve + offset

    # Slightly reduced noise variability (machine is "stuck" at a level)
    # Flatten some of the natural variation
    smoothing = 0.3
    for i in range(1, len(faulted)):
        faulted[i] = smoothing * faulted[i - 1] + (1 - smoothing) * faulted[i]

    return np.clip(faulted, 0.5, 10.5).astype(np.float32)


def generate_curve_for_batch(
    fault_type: str,
    batch_size: float = 500.0,
    conveyor_speed: float = 75.0,
    temperature: float = 183.0,
) -> np.ndarray:
    """Generate a power curve with the appropriate fault signature applied."""

    # Start with a normal base curve influenced by batch parameters
    curve = generate_normal_curve(batch_size, conveyor_speed, temperature)

    # Apply fault signature based on label
    if fault_type == "normal":
        return curve
    elif fault_type == "bearing_wear":
        return apply_bearing_wear(curve)
    elif fault_type == "wet_material":
        return apply_wet_material(curve)
    elif fault_type == "calibration_needed":
        return apply_calibration_needed(curve)
    else:
        # Fallback: return normal curve
        print(f"[WARN] Unknown fault type '{fault_type}', generating normal curve.")
        return curve


def main():
    """Generate 2000 power curve .npy files from batch_data.csv."""

    # Step 1: Load batch data
    if not os.path.exists(BATCH_CSV):
        print(f"[ERROR] batch_data.csv not found at {BATCH_CSV}")
        print("[ERROR] Run generate_batch_data.py first!")
        sys.exit(1)

    df = pd.read_csv(BATCH_CSV)
    num_batches = len(df)
    print(f"[PlantIQ] Generating {num_batches} power curves ({TIMESTEPS} timesteps each)...")

    # Step 2: Create output directory
    os.makedirs(CURVES_DIR, exist_ok=True)

    # Step 3: Generate curves
    fault_counts = {"normal": 0, "bearing_wear": 0, "wet_material": 0, "calibration_needed": 0}
    total_size_bytes = 0

    for idx, row in df.iterrows():
        batch_id = row["batch_id"]
        fault_type = row["fault_type"]

        # Generate curve using batch parameters for realistic variation
        curve = generate_curve_for_batch(
            fault_type=fault_type,
            batch_size=row["batch_size"],
            conveyor_speed=row["conveyor_speed"],
            temperature=row["temperature"],
        )

        # Save as .npy file
        filepath = os.path.join(CURVES_DIR, f"{batch_id}.npy")
        np.save(filepath, curve)
        total_size_bytes += os.path.getsize(filepath)

        # Track fault type counts
        fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1

        # Progress indicator every 200 batches
        if (idx + 1) % 200 == 0:
            print(f"  [{idx + 1}/{num_batches}] curves generated...")

    # Step 4: Print summary
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  POWER CURVE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory: {CURVES_DIR}")
    print(f"  Files generated:  {num_batches}")
    print(f"  Timesteps/curve:  {TIMESTEPS}")
    print(f"  Total disk size:  {total_size_mb:.1f} MB")
    print(f"\n  FAULT TYPE BREAKDOWN:")
    for ft, count in sorted(fault_counts.items()):
        pct = count / num_batches * 100
        print(f"    {ft:25s}: {count:5d} ({pct:.1f}%)")

    # Step 5: Sanity checks — load a few curves and verify shape/range
    print(f"\n  SANITY CHECKS:")
    sample_files = [f"B{i:04d}.npy" for i in [0, 500, 1000, 1500, 1999]]
    for fname in sample_files:
        fpath = os.path.join(CURVES_DIR, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)
            print(f"    {fname}: shape={arr.shape}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
