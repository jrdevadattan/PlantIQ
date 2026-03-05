"""
PlantIQ — Synthetic Batch Data Generator
=========================================
Generates 2000 realistic manufacturing batch records with domain-accurate
physics relationships between input parameters and output targets.

Outputs:
  - data/batch_data.csv  (2000 rows × 20 columns)

Domain ranges (from README spec):
  temperature:     175–195 °C   (optimal: 183)
  conveyor_speed:  60–95 %      (optimal: 75)
  hold_time:       10–30 min    (optimal: 18)
  batch_size:      300–700 kg
  material_type:   0/1/2        (TypeA / TypeB / TypeC)
  hour_of_day:     6–21
  operator_exp:    0/1/2        (junior / mid / senior)

Targets:
  quality_score:     60–100 %
  yield_pct:         70–100 %
  performance_pct:   60–100 %
  energy_kwh:        25–55 kWh
  co2_kg:            energy × 0.82
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
NUM_BATCHES = 2000
RANDOM_SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "batch_data.csv")

# Optimal parameter values (domain knowledge)
OPTIMAL_TEMP = 183.0       # °C
OPTIMAL_SPEED = 75.0       # %
OPTIMAL_HOLD = 18.0        # minutes
EMISSION_FACTOR = 0.82     # kg CO₂e per kWh

np.random.seed(RANDOM_SEED)


def generate_batch_ids(n: int) -> list[str]:
    """Generate sequential batch IDs with timestamps spanning ~6 months."""
    # Start date: September 1, 2025 — batches spread over ~6 months
    start_date = datetime(2025, 9, 1, 6, 0, 0)
    ids = []
    timestamps = []

    for i in range(n):
        # ~10 batches per day, spaced 1–3 hours apart
        offset_hours = i * np.random.uniform(1.0, 2.5)
        batch_time = start_date + timedelta(hours=offset_hours)
        batch_id = f"B{i:04d}"
        ids.append(batch_id)
        timestamps.append(batch_time.strftime("%Y-%m-%d %H:%M:%S"))

    return ids, timestamps


def generate_inputs(n: int) -> dict:
    """Generate operator-controlled input parameters with realistic distributions."""

    # Temperature: centered at 183 with spread — most operators aim near optimal
    temperature = np.clip(
        np.random.normal(loc=183, scale=4.5, size=n), 175, 195
    ).round(1)

    # Conveyor speed: centered at 75% with some variance
    conveyor_speed = np.clip(
        np.random.normal(loc=75, scale=7, size=n), 60, 95
    ).round(1)

    # Hold time: centered at 18 min, right-skewed (some long holds)
    hold_time = np.clip(
        np.random.normal(loc=18, scale=4, size=n), 10, 30
    ).round(1)

    # Batch size: roughly uniform across 300–700 kg range
    batch_size = np.clip(
        np.random.normal(loc=500, scale=80, size=n), 300, 700
    ).round(0)

    # Material type: 50% TypeA, 30% TypeB, 20% TypeC
    material_type = np.random.choice([0, 1, 2], size=n, p=[0.50, 0.30, 0.20])

    # Hour of day: biased toward morning/afternoon shifts (6–21)
    hour_of_day = np.random.choice(range(6, 22), size=n, p=_shift_distribution())

    # Operator experience: 30% junior, 45% mid, 25% senior
    operator_exp = np.random.choice([0, 1, 2], size=n, p=[0.30, 0.45, 0.25])

    return {
        "temperature": temperature,
        "conveyor_speed": conveyor_speed,
        "hold_time": hold_time,
        "batch_size": batch_size,
        "material_type": material_type,
        "hour_of_day": hour_of_day,
        "operator_exp": operator_exp,
    }


def _shift_distribution() -> list[float]:
    """Realistic hour distribution: more batches during morning/afternoon."""
    # Hours 6–21 (16 hours)
    # Morning (6–13): higher density
    # Afternoon (14–17): medium density
    # Evening (18–21): lower density
    weights = [
        0.06, 0.08, 0.09, 0.09, 0.08, 0.07, 0.07, 0.06,  # 6–13 (morning)
        0.06, 0.06, 0.06, 0.05,                              # 14–17 (afternoon)
        0.05, 0.04, 0.04, 0.04,                              # 18–21 (evening)
    ]
    total = sum(weights)
    return [w / total for w in weights]


def compute_derived_features(inputs: dict) -> dict:
    """Compute the 7 engineered features per the spec."""

    temperature = inputs["temperature"]
    conveyor_speed = inputs["conveyor_speed"]
    hold_time = inputs["hold_time"]
    batch_size = inputs["batch_size"]
    hour_of_day = inputs["hour_of_day"]

    # Interaction term
    temp_speed_product = (temperature * conveyor_speed).round(1)

    # Distance from optimal temperature
    temp_deviation = np.abs(temperature - OPTIMAL_TEMP).round(1)

    # Distance from optimal speed
    speed_deviation = np.abs(conveyor_speed - OPTIMAL_SPEED).round(1)

    # Hold time per unit mass
    hold_per_kg = (hold_time / batch_size).round(6)

    # Shift encoding: 0=morning(6–13), 1=afternoon(14–21), 2=night(22–5)
    shift = np.where(hour_of_day < 14, 0, np.where(hour_of_day < 22, 1, 2))

    # Hours into shift
    shift_start = np.where(shift == 0, 6, np.where(shift == 1, 14, 22))
    hours_into_shift = (hour_of_day - shift_start).astype(int)

    return {
        "temp_speed_product": temp_speed_product,
        "temp_deviation": temp_deviation,
        "speed_deviation": speed_deviation,
        "hold_per_kg": hold_per_kg,
        "shift": shift,
        "hours_into_shift": hours_into_shift,
    }


def compute_targets(inputs: dict, derived: dict, n: int) -> dict:
    """
    Generate realistic output targets with physics-based relationships.

    Key relationships (domain knowledge):
    - Quality ↑ when temp near 183, speed near 75, experienced operator
    - Yield ↑ when quality is high & hold time is reasonable
    - Performance ↑ when speed is moderate & temp is stable
    - Energy ↑ with higher hold_time, batch_size, speed, material type
    """

    temperature = inputs["temperature"]
    conveyor_speed = inputs["conveyor_speed"]
    hold_time = inputs["hold_time"]
    batch_size = inputs["batch_size"]
    material_type = inputs["material_type"]
    operator_exp = inputs["operator_exp"]

    temp_deviation = derived["temp_deviation"]
    speed_deviation = derived["speed_deviation"]
    shift = derived["shift"]

    # ── Quality Score (60–100%) ──
    # Base: 95% at optimal. Penalized by deviations, boosted by experience.
    quality_base = 95.0
    quality = (
        quality_base
        - 0.8 * temp_deviation           # Temp away from 183 hurts quality
        - 0.4 * speed_deviation           # Speed away from 75 hurts quality
        - 0.15 * (hold_time - OPTIMAL_HOLD)  # Too long/short hurts
        + 1.5 * operator_exp              # Senior operators produce better quality
        - 1.0 * (material_type == 2)      # TypeC material is harder to process
        - 0.5 * (shift == 1)             # Afternoon shift slightly worse
        - 1.2 * (shift == 2)             # Night shift worse
        + np.random.normal(0, 1.2, size=n)  # Random noise
    )
    quality = np.clip(quality, 60, 100).round(1)

    # ── Yield Percentage (70–100%) ──
    # Closely correlated with quality but with its own dynamics
    yield_base = 96.0
    yield_pct = (
        yield_base
        - 0.6 * temp_deviation
        - 0.3 * speed_deviation
        - 0.2 * np.abs(hold_time - OPTIMAL_HOLD)
        + 1.0 * operator_exp
        - 0.8 * (material_type == 2)
        - 0.4 * (material_type == 1)
        - 0.3 * (shift == 2)
        + 0.5 * (quality - 90) * 0.1      # Quality correlation
        + np.random.normal(0, 1.0, size=n)
    )
    yield_pct = np.clip(yield_pct, 70, 100).round(1)

    # ── Performance Percentage (60–100%) ──
    # Machine efficiency — affected by speed settings and maintenance
    perf_base = 94.0
    performance = (
        perf_base
        - 0.5 * speed_deviation
        - 0.3 * temp_deviation
        - 0.1 * np.abs(hold_time - OPTIMAL_HOLD)
        + 1.2 * operator_exp
        - 0.6 * (material_type == 2)
        - 0.5 * (shift == 2)
        + np.random.normal(0, 1.5, size=n)
    )
    performance = np.clip(performance, 60, 100).round(1)

    # ── Energy Consumption (25–55 kWh) ──
    # Physics-based: energy is driven by hold time, speed, batch size, material
    energy_base = 32.0
    energy = (
        energy_base
        + 0.5 * (hold_time - OPTIMAL_HOLD)             # Longer hold = more energy
        + 0.15 * (conveyor_speed - OPTIMAL_SPEED)       # Higher speed = more energy
        + 0.008 * (batch_size - 400)                     # Bigger batch = more energy
        + 2.0 * (material_type == 1)                     # TypeB uses slightly more
        + 4.0 * (material_type == 2)                     # TypeC uses significantly more
        + 0.1 * temp_deviation                           # Non-optimal temp wastes energy
        - 0.8 * operator_exp                             # Experienced operators are efficient
        + 0.5 * (shift == 2)                             # Night shift less efficient
        + np.random.normal(0, 1.5, size=n)
    )
    energy = np.clip(energy, 25, 55).round(1)

    # ── CO₂ Emissions (derived) ──
    co2_kg = (energy * EMISSION_FACTOR).round(2)

    # ── Energy per kg (historical derived feature) ──
    energy_per_kg = (energy / batch_size).round(6)

    return {
        "quality_score": quality,
        "yield_pct": yield_pct,
        "performance_pct": performance,
        "energy_kwh": energy,
        "co2_kg": co2_kg,
        "energy_per_kg": energy_per_kg,
    }


def assign_fault_labels(n: int) -> np.ndarray:
    """
    Assign fault types to batches for power curve generation.
    Distribution: ~75% normal, ~10% bearing_wear, ~10% wet_material, ~5% calibration_needed
    """
    return np.random.choice(
        ["normal", "bearing_wear", "wet_material", "calibration_needed"],
        size=n,
        p=[0.75, 0.10, 0.10, 0.05],
    )


def main():
    """Generate batch_data.csv with 2000 synthetic batch records."""
    print(f"[PlantIQ] Generating {NUM_BATCHES} synthetic batch records...")
    print(f"[PlantIQ] Random seed: {RANDOM_SEED}")

    # Step 1: Generate batch IDs and timestamps
    batch_ids, timestamps = generate_batch_ids(NUM_BATCHES)

    # Step 2: Generate operator-controlled inputs
    inputs = generate_inputs(NUM_BATCHES)

    # Step 3: Compute derived features
    derived = compute_derived_features(inputs)

    # Step 4: Compute targets with physics-based relationships
    targets = compute_targets(inputs, derived, NUM_BATCHES)

    # Step 5: Assign fault labels (used by power curve generator)
    fault_types = assign_fault_labels(NUM_BATCHES)

    # Step 6: Assemble DataFrame
    df = pd.DataFrame({
        "batch_id": batch_ids,
        "timestamp": timestamps,
        # Inputs
        "temperature": inputs["temperature"],
        "conveyor_speed": inputs["conveyor_speed"],
        "hold_time": inputs["hold_time"],
        "batch_size": inputs["batch_size"],
        "material_type": inputs["material_type"],
        "hour_of_day": inputs["hour_of_day"],
        "operator_exp": inputs["operator_exp"],
        # Derived features
        "temp_speed_product": derived["temp_speed_product"],
        "temp_deviation": derived["temp_deviation"],
        "speed_deviation": derived["speed_deviation"],
        "hold_per_kg": derived["hold_per_kg"],
        "shift": derived["shift"],
        "hours_into_shift": derived["hours_into_shift"],
        # Targets
        "quality_score": targets["quality_score"],
        "yield_pct": targets["yield_pct"],
        "performance_pct": targets["performance_pct"],
        "energy_kwh": targets["energy_kwh"],
        "co2_kg": targets["co2_kg"],
        "energy_per_kg": targets["energy_per_kg"],
        # Fault label (for power curve generation + anomaly model training)
        "fault_type": fault_types,
    })

    # Step 7: Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    # Step 8: Print summary statistics
    print(f"\n{'='*60}")
    print(f"  BATCH DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"  Records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\n  INPUT PARAMETER RANGES:")
    for col in ["temperature", "conveyor_speed", "hold_time", "batch_size"]:
        print(f"    {col:20s}  {df[col].min():8.1f} — {df[col].max():.1f}  (mean: {df[col].mean():.1f})")
    print(f"\n  TARGET DISTRIBUTIONS:")
    for col in ["quality_score", "yield_pct", "performance_pct", "energy_kwh", "co2_kg"]:
        print(f"    {col:20s}  {df[col].min():8.1f} — {df[col].max():.1f}  (mean: {df[col].mean():.1f})")
    print(f"\n  MATERIAL TYPE DISTRIBUTION:")
    for mt, label in [(0, "TypeA"), (1, "TypeB"), (2, "TypeC")]:
        count = (df["material_type"] == mt).sum()
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  OPERATOR EXPERIENCE DISTRIBUTION:")
    for exp, label in [(0, "Junior"), (1, "Mid"), (2, "Senior")]:
        count = (df["operator_exp"] == exp).sum()
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  FAULT TYPE DISTRIBUTION:")
    for ft in ["normal", "bearing_wear", "wet_material", "calibration_needed"]:
        count = (df["fault_type"] == ft).sum()
        print(f"    {ft:25s}: {count} ({count/len(df)*100:.1f}%)")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    main()
