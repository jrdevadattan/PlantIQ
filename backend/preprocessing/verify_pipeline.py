"""
PlantIQ F1.2 — Preprocessing Pipeline Verification Script
Validates all outputs, artifacts, and data integrity.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")

errors = []


def check(label, condition, msg_fail=""):
    if condition:
        print(f"[PASS] {label}")
    else:
        errors.append(msg_fail or label)
        print(f"[FAIL] {label} — {msg_fail}")


def main():
    print("=" * 60)
    print("  PLANTIQ F1.2 — PREPROCESSING PIPELINE VERIFICATION")
    print("=" * 60)

    # ── 1. Output files exist ──
    print("\n--- Output File Checks ---")
    expected_files = {
        "train_processed.csv": os.path.join(DATA_DIR, "train_processed.csv"),
        "test_processed.csv": os.path.join(DATA_DIR, "test_processed.csv"),
        "train_normalized.csv": os.path.join(DATA_DIR, "train_normalized.csv"),
        "test_normalized.csv": os.path.join(DATA_DIR, "test_normalized.csv"),
        "scaler.pkl": os.path.join(ARTIFACT_DIR, "scaler.pkl"),
        "outlier_fences.json": os.path.join(ARTIFACT_DIR, "outlier_fences.json"),
        "pipeline_meta.json": os.path.join(ARTIFACT_DIR, "pipeline_meta.json"),
    }
    for name, path in expected_files.items():
        check(f"File exists: {name}", os.path.exists(path), f"Missing: {path}")

    # ── 2. Train/test split correctness ──
    print("\n--- Train/Test Split ---")
    train = pd.read_csv(expected_files["train_processed.csv"])
    test = pd.read_csv(expected_files["test_processed.csv"])
    total = len(train) + len(test)

    check(f"Total rows = 2000", total == 2000, f"Got {total}")
    check(f"Train rows = 1600", len(train) == 1600, f"Got {len(train)}")
    check(f"Test rows = 400", len(test) == 400, f"Got {len(test)}")

    # Temporal ordering: last train timestamp < first test timestamp
    if "timestamp" in train.columns and "timestamp" in test.columns:
        last_train = train["timestamp"].iloc[-1]
        first_test = test["timestamp"].iloc[0]
        check("Temporal split respected (train before test)",
              last_train <= first_test,
              f"Last train={last_train}, first test={first_test}")

    # No duplicate batch_ids across train/test
    train_ids = set(train["batch_id"])
    test_ids = set(test["batch_id"])
    overlap = train_ids & test_ids
    check("No batch_id overlap between train/test", len(overlap) == 0,
          f"{len(overlap)} overlapping IDs")

    # ── 3. No missing values after imputation ──
    print("\n--- Data Quality (Post-Pipeline) ---")
    key_cols = ["temperature", "conveyor_speed", "hold_time", "batch_size",
                "quality_score", "yield_pct", "performance_pct", "energy_kwh"]
    train_nulls = train[key_cols].isnull().sum().sum()
    test_nulls = test[key_cols].isnull().sum().sum()
    check("Train: no nulls in key columns", train_nulls == 0, f"{train_nulls} nulls")
    check("Test: no nulls in key columns", test_nulls == 0, f"{test_nulls} nulls")

    # ── 4. Outlier capping worked ──
    print("\n--- Outlier Capping ---")
    with open(expected_files["outlier_fences.json"]) as f:
        fences = json.load(f)

    for col, bounds in fences.items():
        lower, upper = bounds["lower"], bounds["upper"]
        if col in train.columns:
            mn, mx = train[col].min(), train[col].max()
            ok = mn >= lower - 0.01 and mx <= upper + 0.01  # small float tolerance
            check(f"Train {col}: [{mn:.2f}, {mx:.2f}] within fences [{lower:.2f}, {upper:.2f}]",
                  ok, f"Out of bounds")

    # ── 5. Feature engineering produced all 7 derived features ──
    print("\n--- Derived Features ---")
    derived_cols = ["temp_speed_product", "temp_deviation", "speed_deviation",
                    "hold_per_kg", "shift", "hours_into_shift", "energy_per_kg"]
    for col in derived_cols:
        check(f"Derived feature present: {col}", col in train.columns,
              f"Missing from train_processed.csv")

    # Verify shift values are valid
    if "shift" in train.columns:
        valid_shifts = {0, 1}  # morning and afternoon (our hour_of_day range is 6-21)
        actual_shifts = set(train["shift"].unique())
        check(f"Shift values valid: {actual_shifts}",
              actual_shifts.issubset({0, 1, 2}), f"Invalid shifts: {actual_shifts}")

    # ── 6. Normalization correctness ──
    print("\n--- Normalization ---")
    train_norm = pd.read_csv(expected_files["train_normalized.csv"])
    feature_cols = ["temperature", "conveyor_speed", "hold_time", "batch_size",
                    "hour_of_day", "operator_exp", "material_type",
                    "temp_speed_product", "temp_deviation", "speed_deviation",
                    "hold_per_kg", "shift", "hours_into_shift"]

    for col in feature_cols:
        if col in train_norm.columns:
            mean = train_norm[col].mean()
            std = train_norm[col].std()
            check(f"Norm {col:20s}: mean={mean:+.4f} std={std:.4f}",
                  abs(mean) < 0.05 and abs(std - 1.0) < 0.05,
                  f"Bad normalization: mean={mean:.4f}, std={std:.4f}")

    # Targets should NOT be normalized
    target_cols = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]
    for col in target_cols:
        if col in train_norm.columns:
            mean = train_norm[col].mean()
            check(f"Target {col} NOT normalized (mean={mean:.1f})",
                  abs(mean) > 5,  # targets have large means like 91, 92, 33
                  f"Target appears normalized: mean={mean:.4f}")

    # ── 7. Scaler artifact loads correctly ──
    print("\n--- Artifact Integrity ---")
    try:
        artifact = joblib.load(expected_files["scaler.pkl"])
        check("scaler.pkl loads successfully", True)
        check("scaler.pkl contains 'scaler' key", "scaler" in artifact,
              f"Keys: {list(artifact.keys())}")
        check("scaler.pkl contains 'feature_cols' key", "feature_cols" in artifact,
              f"Keys: {list(artifact.keys())}")
    except Exception as e:
        errors.append(f"Failed to load scaler.pkl: {e}")
        print(f"[FAIL] scaler.pkl — {e}")

    # Pipeline metadata
    try:
        with open(expected_files["pipeline_meta.json"]) as f:
            meta = json.load(f)
        check("pipeline_meta.json loads successfully", True)
        check(f"Feature count in meta = {len(meta.get('feature_cols', []))}",
              len(meta.get("feature_cols", [])) == 13,
              f"Expected 13, got {len(meta.get('feature_cols', []))}")
        check(f"Target count in meta = {len(meta.get('target_cols', []))}",
              len(meta.get("target_cols", [])) == 4,
              f"Expected 4, got {len(meta.get('target_cols', []))}")
    except Exception as e:
        errors.append(f"Failed to load pipeline_meta.json: {e}")

    # ── 8. Final verdict ──
    print("\n" + "=" * 60)
    if errors:
        print(f"  RESULT: FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  RESULT: ALL CHECKS PASSED ✓")
        print(f"  Pipeline: 4 stages × {total} rows → {len(train)} train + {len(test)} test")
        print(f"  Features: 13 normalized | Targets: 4 original scale")
        print(f"  Artifacts: scaler.pkl, outlier_fences.json, pipeline_meta.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
