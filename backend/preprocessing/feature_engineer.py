"""
PlantIQ — Stage 3: Feature Engineering
========================================
Computes 7 derived features from raw batch inputs that capture
domain-specific relationships the ML model should learn from.

Derived features (per README spec):
  1. temp_speed_product   — interaction: temperature × conveyor_speed
  2. temp_deviation       — |temperature - 183| (distance from optimal 183°C)
  3. speed_deviation      — |conveyor_speed - 75| (distance from optimal 75%)
  4. hold_per_kg          — hold_time / batch_size (process intensity)
  5. shift                — 0=morning(6-14), 1=afternoon(14-22), 2=night(22-6)
  6. hours_into_shift     — hours elapsed since current shift started
  7. energy_per_kg        — energy_kwh / batch_size (efficiency metric, historical only)

These features encode domain knowledge directly into the feature space,
letting XGBoost learn optimal decision boundaries faster.

Spec: README Layer 2 — "7 derived feats"
"""

import numpy as np
import pandas as pd


# Domain constants
OPTIMAL_TEMP = 183.0   # °C
OPTIMAL_SPEED = 75.0   # %


class FeatureEngineer:
    """Computes derived manufacturing features from raw batch inputs.

    Parameters
    ----------
    include_energy_per_kg : bool
        Whether to compute energy_per_kg. Set True for historical/training data
        (where energy_kwh is known), False for prediction inputs (where it isn't).
    """

    def __init__(self, include_energy_per_kg: bool = True):
        self.include_energy_per_kg = include_energy_per_kg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived features and add them to the DataFrame.

        If a derived column already exists, it is overwritten
        (ensures consistency with the canonical computation).

        Parameters
        ----------
        df : pd.DataFrame
            Batch data with at least: temperature, conveyor_speed,
            hold_time, batch_size, hour_of_day, and optionally energy_kwh.

        Returns
        -------
        pd.DataFrame
            DataFrame with derived feature columns added/updated.
        """
        df = df.copy()
        created = []

        # 1. temp_speed_product — interaction term
        if "temperature" in df.columns and "conveyor_speed" in df.columns:
            df["temp_speed_product"] = (df["temperature"] * df["conveyor_speed"]).round(1)
            created.append("temp_speed_product")

        # 2. temp_deviation — distance from optimal temperature
        if "temperature" in df.columns:
            df["temp_deviation"] = abs(df["temperature"] - OPTIMAL_TEMP).round(1)
            created.append("temp_deviation")

        # 3. speed_deviation — distance from optimal speed
        if "conveyor_speed" in df.columns:
            df["speed_deviation"] = abs(df["conveyor_speed"] - OPTIMAL_SPEED).round(1)
            created.append("speed_deviation")

        # 4. hold_per_kg — process intensity ratio
        if "hold_time" in df.columns and "batch_size" in df.columns:
            df["hold_per_kg"] = (df["hold_time"] / df["batch_size"]).round(6)
            created.append("hold_per_kg")

        # 5. shift — time-of-day categorization
        if "hour_of_day" in df.columns:
            df["shift"] = df["hour_of_day"].apply(_compute_shift)
            created.append("shift")

        # 6. hours_into_shift — fatigue/warmup proxy
        if "hour_of_day" in df.columns:
            df["hours_into_shift"] = df["hour_of_day"].apply(_compute_hours_into_shift)
            created.append("hours_into_shift")

        # 7. energy_per_kg — efficiency metric (historical data only)
        if self.include_energy_per_kg and "energy_kwh" in df.columns and "batch_size" in df.columns:
            df["energy_per_kg"] = (df["energy_kwh"] / df["batch_size"]).round(6)
            created.append("energy_per_kg")

        # 8. co2_kg — derived carbon emission (if energy available)
        if "energy_kwh" in df.columns and "co2_kg" not in df.columns:
            df["co2_kg"] = (df["energy_kwh"] * 0.82).round(2)
            created.append("co2_kg")

        print(f"  [FeatureEngineer] Created/updated {len(created)} features: {', '.join(created)}")
        return df


def _compute_shift(hour: int) -> int:
    """Map hour-of-day to shift category.

    Shift schedule:
      Morning:   06:00 – 13:59  → 0
      Afternoon: 14:00 – 21:59  → 1
      Night:     22:00 – 05:59  → 2
    """
    if 6 <= hour < 14:
        return 0   # morning
    elif 14 <= hour < 22:
        return 1   # afternoon
    else:
        return 2   # night


def _compute_hours_into_shift(hour: int) -> int:
    """Compute how many hours into the current shift.

    This serves as a fatigue/warmup proxy — operators
    are more alert at shift start, may fatigue toward end.
    """
    if 6 <= hour < 14:
        return hour - 6
    elif 14 <= hour < 22:
        return hour - 14
    else:
        return hour - 22 if hour >= 22 else hour + 2
