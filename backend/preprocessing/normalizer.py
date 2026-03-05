"""
PlantIQ — Stage 4: Feature Normalization
==========================================
Applies StandardScaler (zero-mean, unit-variance) to numeric features
so that XGBoost and other models operate on comparable scales.

While tree-based models (XGBoost) don't strictly require normalization,
it helps with:
- SHAP value interpretation (contributions are in standardized units)
- LSTM Autoencoder training (neural networks are sensitive to scale)
- Consistent pipeline — one scaler used across all downstream models

The fitted scaler is saved to disk with joblib so that the same
transformation can be applied at inference time (API requests).

Spec: README Layer 2 — "Normalization"
       README models/trained/ — "scaler.pkl"
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# Features to normalize — all numeric inputs + derived features
# Targets are NOT normalized (predictions should be in original units)
FEATURE_COLS = [
    "temperature", "conveyor_speed", "hold_time", "batch_size",
    "hour_of_day", "operator_exp", "material_type",
    "temp_speed_product", "temp_deviation", "speed_deviation",
    "hold_per_kg", "shift", "hours_into_shift",
]

# Target columns — kept in original scale
TARGET_COLS = [
    "quality_score", "yield_pct", "performance_pct", "energy_kwh",
]


class BatchNormalizer:
    """StandardScaler wrapper with save/load capability.

    Parameters
    ----------
    artifact_dir : str
        Directory to save/load the scaler artifact (default: models/trained/).
    """

    def __init__(self, artifact_dir: str | None = None):
        self._scaler = StandardScaler()
        self._feature_cols: list[str] = []
        self._is_fitted = False

        # Default artifact path: backend/models/trained/scaler.pkl
        if artifact_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifact_dir = os.path.join(base, "models", "trained")
        self.artifact_dir = artifact_dir
        self.artifact_path = os.path.join(artifact_dir, "scaler.pkl")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and return normalized DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training batch data with feature columns present.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns normalized (zero-mean, unit-var).
            Non-feature columns (batch_id, timestamp, targets, etc.) are unchanged.
        """
        df = df.copy()
        self._feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        # Fit and transform feature columns
        df[self._feature_cols] = self._scaler.fit_transform(df[self._feature_cols])
        self._is_fitted = True

        print(f"  [Normalizer] Scaled {len(self._feature_cols)} features "
              f"(mean → 0, std → 1)")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted scaler to new data.

        Parameters
        ----------
        df : pd.DataFrame
            New batch data to normalize.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns normalized using training statistics.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit_transform() first or load().")

        df = df.copy()
        cols = [c for c in self._feature_cols if c in df.columns]
        df[cols] = self._scaler.transform(df[cols])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization to get original-scale values.

        Parameters
        ----------
        df : pd.DataFrame
            Normalized data to denormalize.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns in original scale.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted.")

        df = df.copy()
        cols = [c for c in self._feature_cols if c in df.columns]
        df[cols] = self._scaler.inverse_transform(df[cols])
        return df

    def save(self) -> str:
        """Save the fitted scaler to disk.

        Returns
        -------
        str
            Path to the saved artifact.
        """
        os.makedirs(self.artifact_dir, exist_ok=True)
        artifact = {
            "scaler": self._scaler,
            "feature_cols": self._feature_cols,
        }
        joblib.dump(artifact, self.artifact_path)
        print(f"  [Normalizer] Saved scaler to {self.artifact_path}")
        return self.artifact_path

    def load(self) -> "BatchNormalizer":
        """Load a previously fitted scaler from disk.

        Returns
        -------
        self
        """
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(f"Scaler not found at {self.artifact_path}")

        artifact = joblib.load(self.artifact_path)
        self._scaler = artifact["scaler"]
        self._feature_cols = artifact["feature_cols"]
        self._is_fitted = True
        print(f"  [Normalizer] Loaded scaler from {self.artifact_path}")
        return self
