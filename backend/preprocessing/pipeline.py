"""
PlantIQ — Master Preprocessing Pipeline
=========================================
Orchestrates the 4-stage data pipeline:

  Stage 1: KNN Imputation     → Fill missing values using k-nearest neighbours
  Stage 2: IQR Outlier Capping → Cap extreme values to IQR fences
  Stage 3: Feature Engineering → Compute 7 derived domain features
  Stage 4: Normalization       → StandardScaler (zero-mean, unit-variance)

Additionally handles:
  - Train/test split with temporal ordering (TimeSeriesSplit-compatible)
  - Saving/loading pipeline artifacts (scaler, outlier fences)
  - Producing both normalized (for training) and original-scale (for SHAP) outputs

Usage:
  python -m preprocessing.pipeline           # Process data/batch_data.csv
  python preprocessing/pipeline.py           # Same — run from backend/ dir

Spec: README — "preprocessing/pipeline.py — 4-stage pipeline"
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Add parent directory to path for module imports when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.imputer import BatchImputer
from preprocessing.outlier_detector import OutlierCapper
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.normalizer import BatchNormalizer, FEATURE_COLS, TARGET_COLS


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")

INPUT_CSV = os.path.join(DATA_DIR, "batch_data.csv")
OUTPUT_TRAIN_CSV = os.path.join(DATA_DIR, "train_processed.csv")
OUTPUT_TEST_CSV = os.path.join(DATA_DIR, "test_processed.csv")
OUTPUT_TRAIN_NORM_CSV = os.path.join(DATA_DIR, "train_normalized.csv")
OUTPUT_TEST_NORM_CSV = os.path.join(DATA_DIR, "test_normalized.csv")

TEST_SIZE = 0.2        # 80/20 split
RANDOM_STATE = 42      # Reproducibility


class PreprocessingPipeline:
    """Master 4-stage preprocessing pipeline for PlantIQ batch data.

    Attributes
    ----------
    imputer : BatchImputer
        Stage 1 — KNN missing value imputer.
    outlier_capper : OutlierCapper
        Stage 2 — IQR outlier capper.
    feature_engineer : FeatureEngineer
        Stage 3 — Derived feature calculator.
    normalizer : BatchNormalizer
        Stage 4 — StandardScaler normalizer.
    """

    def __init__(self):
        self.imputer = BatchImputer(n_neighbors=5)
        self.outlier_capper = OutlierCapper(iqr_multiplier=1.5)
        self.feature_engineer = FeatureEngineer(include_energy_per_kg=True)
        self.normalizer = BatchNormalizer(artifact_dir=ARTIFACT_DIR)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full 4-stage pipeline on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw batch data loaded from CSV.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (processed_df, normalized_df) — both have the same rows.
            processed_df is in original scale (for SHAP, display).
            normalized_df has features scaled (for model training).
        """
        print("\n" + "=" * 60)
        print("  PREPROCESSING PIPELINE — FIT + TRANSFORM")
        print("=" * 60)

        # Stage 1: Impute missing values
        print("\n  Stage 1/4: KNN Imputation")
        df = self.imputer.fit_transform(df)

        # Stage 2: Cap outliers
        print("\n  Stage 2/4: IQR Outlier Capping")
        df = self.outlier_capper.fit_transform(df)

        # Stage 3: Feature engineering
        print("\n  Stage 3/4: Feature Engineering")
        df = self.feature_engineer.transform(df)

        # Keep a copy in original scale before normalization
        processed_df = df.copy()

        # Stage 4: Normalize features
        print("\n  Stage 4/4: Normalization")
        normalized_df = self.normalizer.fit_transform(df)

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print(f"  Rows: {len(processed_df)} | "
              f"Features: {len([c for c in FEATURE_COLS if c in processed_df.columns])} | "
              f"Targets: {len([c for c in TARGET_COLS if c in processed_df.columns])}")
        print("=" * 60)

        return processed_df, normalized_df

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the already-fitted pipeline to new data (inference).

        Parameters
        ----------
        df : pd.DataFrame
            New batch data (e.g. from API request).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (processed_df, normalized_df)
        """
        df = self.imputer.transform(df)
        df = self.outlier_capper.transform(df)
        df = self.feature_engineer.transform(df)
        processed_df = df.copy()
        normalized_df = self.normalizer.transform(df)
        return processed_df, normalized_df

    def save_artifacts(self) -> dict[str, str]:
        """Save all pipeline artifacts to disk.

        Returns
        -------
        dict[str, str]
            Mapping of artifact name → file path.
        """
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        paths = {}

        # Save scaler
        paths["scaler"] = self.normalizer.save()

        # Save outlier fences
        fences_path = os.path.join(ARTIFACT_DIR, "outlier_fences.json")
        fences_serializable = {
            col: {"lower": float(lo), "upper": float(hi)}
            for col, (lo, hi) in self.outlier_capper.fences.items()
        }
        with open(fences_path, "w") as f:
            json.dump(fences_serializable, f, indent=2)
        paths["outlier_fences"] = fences_path
        print(f"  [Pipeline] Saved outlier fences to {fences_path}")

        # Save pipeline metadata
        meta_path = os.path.join(ARTIFACT_DIR, "pipeline_meta.json")
        meta = {
            "feature_cols": [c for c in FEATURE_COLS],
            "target_cols": [c for c in TARGET_COLS],
            "imputer_neighbors": self.imputer.n_neighbors,
            "iqr_multiplier": self.outlier_capper.iqr_multiplier,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        paths["pipeline_meta"] = meta_path
        print(f"  [Pipeline] Saved pipeline metadata to {meta_path}")

        return paths

    def load_artifacts(self) -> "PreprocessingPipeline":
        """Load previously saved pipeline artifacts for inference.

        Returns
        -------
        self
        """
        self.normalizer.load()

        fences_path = os.path.join(ARTIFACT_DIR, "outlier_fences.json")
        if os.path.exists(fences_path):
            with open(fences_path) as f:
                fences_data = json.load(f)
            self.outlier_capper._fences = {
                col: (v["lower"], v["upper"]) for col, v in fences_data.items()
            }
            print(f"  [Pipeline] Loaded outlier fences from {fences_path}")

        return self


def run_pipeline(csv_path: str = INPUT_CSV) -> dict:
    """Execute the full pipeline: load CSV → process → split → save.

    Parameters
    ----------
    csv_path : str
        Path to the raw batch CSV file.

    Returns
    -------
    dict
        Summary statistics and file paths.
    """
    # ── Load raw data ──
    print(f"\nLoading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    # ── Temporal train/test split ──
    # Sort by timestamp to respect temporal ordering
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Split: first 80% = train, last 20% = test (temporal split)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"\n  Temporal split: {len(df_train)} train / {len(df_test)} test")

    # ── Run pipeline on training data ──
    pipeline = PreprocessingPipeline()
    train_processed, train_normalized = pipeline.fit_transform(df_train)

    # ── Transform test data using fitted pipeline ──
    print("\n  Applying fitted pipeline to test set...")
    test_processed, test_normalized = pipeline.transform(df_test)

    # ── Save outputs ──
    print("\n  Saving processed datasets...")
    train_processed.to_csv(OUTPUT_TRAIN_CSV, index=False)
    test_processed.to_csv(OUTPUT_TEST_CSV, index=False)
    train_normalized.to_csv(OUTPUT_TRAIN_NORM_CSV, index=False)
    test_normalized.to_csv(OUTPUT_TEST_NORM_CSV, index=False)
    print(f"    → {OUTPUT_TRAIN_CSV}")
    print(f"    → {OUTPUT_TEST_CSV}")
    print(f"    → {OUTPUT_TRAIN_NORM_CSV}")
    print(f"    → {OUTPUT_TEST_NORM_CSV}")

    # ── Save pipeline artifacts ──
    print("\n  Saving pipeline artifacts...")
    artifact_paths = pipeline.save_artifacts()

    # ── Summary statistics ──
    summary = _compute_summary(df_train, df_test, train_processed, test_processed,
                               train_normalized, artifact_paths)
    _print_summary(summary)

    return summary


def _compute_summary(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    proc_train: pd.DataFrame,
    proc_test: pd.DataFrame,
    norm_train: pd.DataFrame,
    artifact_paths: dict,
) -> dict:
    """Compute summary statistics for the pipeline run."""
    feature_cols = [c for c in FEATURE_COLS if c in proc_train.columns]
    target_cols = [c for c in TARGET_COLS if c in proc_train.columns]

    summary = {
        "raw_rows": len(raw_train) + len(raw_test),
        "train_rows": len(proc_train),
        "test_rows": len(proc_test),
        "feature_count": len(feature_cols),
        "target_count": len(target_cols),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "artifacts": artifact_paths,
        "train_stats": {},
        "normalized_stats": {},
    }

    # Target distribution stats (original scale)
    for col in target_cols:
        summary["train_stats"][col] = {
            "mean": float(proc_train[col].mean()),
            "std": float(proc_train[col].std()),
            "min": float(proc_train[col].min()),
            "max": float(proc_train[col].max()),
        }

    # Normalized feature stats (should be mean≈0, std≈1)
    for col in feature_cols:
        if col in norm_train.columns:
            summary["normalized_stats"][col] = {
                "mean": float(norm_train[col].mean()),
                "std": float(norm_train[col].std()),
            }

    return summary


def _print_summary(summary: dict) -> None:
    """Print a formatted pipeline summary."""
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total raw rows:     {summary['raw_rows']}")
    print(f"  Train set:          {summary['train_rows']} rows")
    print(f"  Test set:           {summary['test_rows']} rows")
    print(f"  Feature columns:    {summary['feature_count']}")
    print(f"  Target columns:     {summary['target_count']}")

    print(f"\n  TARGET DISTRIBUTIONS (train, original scale):")
    for col, stats in summary["train_stats"].items():
        print(f"    {col:20s}  mean={stats['mean']:7.2f}  "
              f"std={stats['std']:5.2f}  "
              f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")

    print(f"\n  NORMALIZED FEATURES (should be mean≈0, std≈1):")
    for col, stats in summary["normalized_stats"].items():
        ok = abs(stats["mean"]) < 0.01 and abs(stats["std"] - 1.0) < 0.01
        marker = "✓" if ok else "⚠"
        print(f"    {col:20s}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}  {marker}")

    print(f"\n  ARTIFACTS SAVED:")
    for name, path in summary["artifacts"].items():
        print(f"    {name}: {path}")

    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
