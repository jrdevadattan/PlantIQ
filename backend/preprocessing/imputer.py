"""
PlantIQ — Stage 1: KNN Missing Value Imputation
=================================================
Uses scikit-learn's KNNImputer to fill missing values using the
k-nearest-neighbours approach. This is preferable to mean/median
imputation because it preserves local correlations between features
(e.g. high temperature batches tend to have higher energy).

Spec: README Layer 2 — Data Pipeline — "KNN imputation"
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# Columns that KNN imputation should operate on (numeric only).
# Categorical columns (material_type, operator_exp) use mode imputation.
NUMERIC_COLS = [
    "temperature", "conveyor_speed", "hold_time", "batch_size",
    "hour_of_day",
]

CATEGORICAL_COLS = [
    "material_type", "operator_exp",
]

TARGET_COLS = [
    "quality_score", "yield_pct", "performance_pct", "energy_kwh",
]


class BatchImputer:
    """KNN-based imputer for manufacturing batch data.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours for the KNN algorithm (default: 5).
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._knn = KNNImputer(n_neighbors=n_neighbors, weights="distance")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the imputer on the data and fill missing values in-place.

        Parameters
        ----------
        df : pd.DataFrame
            Raw batch data (may contain NaN values).

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.
        """
        df = df.copy()
        missing_before = df.isnull().sum().sum()

        # 1. KNN imputation on numeric columns
        num_cols_present = [c for c in NUMERIC_COLS + TARGET_COLS if c in df.columns]
        if num_cols_present and df[num_cols_present].isnull().any().any():
            df[num_cols_present] = self._knn.fit_transform(df[num_cols_present])

        # 2. Mode imputation for categorical columns
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)

        # 3. Verify no remaining nulls in key columns
        missing_after = df[num_cols_present + [c for c in CATEGORICAL_COLS if c in df.columns]].isnull().sum().sum()

        print(f"  [Imputer] Missing values: {missing_before} → {missing_after}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted imputer to new data.

        Parameters
        ----------
        df : pd.DataFrame
            New batch data (may contain NaN values).

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.
        """
        df = df.copy()
        num_cols_present = [c for c in NUMERIC_COLS + TARGET_COLS if c in df.columns]
        if num_cols_present and df[num_cols_present].isnull().any().any():
            df[num_cols_present] = self._knn.transform(df[num_cols_present])

        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)

        return df
