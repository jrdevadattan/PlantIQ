"""
PlantIQ — Stage 2: IQR Outlier Capping
========================================
Detects outliers using the Interquartile Range (IQR) method and
caps them to the fence values rather than removing rows.

Capping is preferred over removal in manufacturing because:
- Every batch record is valuable for training
- Extreme values may be systematic (e.g. hot shift, large batch)
  rather than data errors
- Removing rows loses temporal ordering needed for TimeSeriesSplit

Method:
  Q1 = 25th percentile, Q3 = 75th percentile
  IQR = Q3 - Q1
  Lower fence = Q1 - 1.5 × IQR
  Upper fence = Q3 + 1.5 × IQR
  Values below lower fence → clipped to lower fence
  Values above upper fence → clipped to upper fence

Spec: README Layer 2 — "IQR outlier cap"
"""

import numpy as np
import pandas as pd


# Columns to apply IQR capping on — continuous numeric features
# Categorical (material_type, operator_exp) and derived features are excluded
OUTLIER_COLS = [
    "temperature", "conveyor_speed", "hold_time", "batch_size",
    "quality_score", "yield_pct", "performance_pct", "energy_kwh",
]


class OutlierCapper:
    """IQR-based outlier capper for manufacturing batch data.

    Parameters
    ----------
    iqr_multiplier : float
        Multiplier for the IQR to determine fences (default: 1.5).
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self._fences: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "OutlierCapper":
        """Compute IQR fences from the training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training batch data (post-imputation).

        Returns
        -------
        self
        """
        cols = [c for c in OUTLIER_COLS if c in df.columns]
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self._fences[col] = (lower, upper)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using the pre-computed fences.

        Parameters
        ----------
        df : pd.DataFrame
            Batch data to cap (may be train or new data).

        Returns
        -------
        pd.DataFrame
            DataFrame with outlier values clipped to fence boundaries.
        """
        df = df.copy()
        total_capped = 0

        for col, (lower, upper) in self._fences.items():
            if col not in df.columns:
                continue

            below = (df[col] < lower).sum()
            above = (df[col] > upper).sum()
            capped = below + above
            total_capped += capped

            df[col] = df[col].clip(lower=lower, upper=upper)

            if capped > 0:
                print(f"    {col}: {capped} values capped "
                      f"({below} below {lower:.1f}, {above} above {upper:.1f})")

        print(f"  [OutlierCapper] Total values capped: {total_capped}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit fences then cap in one call.

        Parameters
        ----------
        df : pd.DataFrame
            Training batch data (post-imputation).

        Returns
        -------
        pd.DataFrame
            DataFrame with outlier values clipped.
        """
        return self.fit(df).transform(df)

    @property
    def fences(self) -> dict[str, tuple[float, float]]:
        """Return the computed fences for inspection/logging."""
        return dict(self._fences)
