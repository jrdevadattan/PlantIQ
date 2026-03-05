"""
F1.3 Verification Script — Multi-Target XGBoost Predictor
Runs comprehensive checks on model artifacts, metrics, and predictions.
"""
import os
import sys
import json
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(BASE)
TRAINED_DIR = os.path.join(BASE, "trained")
DATA_DIR = os.path.join(BACKEND, "data")

# Add backend to path for imports
sys.path.insert(0, BACKEND)

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    """Run a single check and track pass/fail."""
    global passed, failed, total
    total += 1
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        failed += 1
    else:
        passed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{total:02d}] {status}  {name}{suffix}")


def main():
    print("=" * 60)
    print("  F1.3 VERIFICATION — Multi-Target XGBoost Predictor")
    print("=" * 60)

    # ── Section 1: Artifact Existence ─────────────────────────────────────
    print("\n  § 1. Artifact Existence")

    model_path = os.path.join(TRAINED_DIR, "multi_target.pkl")
    check("Model file exists (multi_target.pkl)", os.path.exists(model_path))

    report_path = os.path.join(TRAINED_DIR, "evaluation_report.json")
    check("Evaluation report exists", os.path.exists(report_path))

    meta_path = os.path.join(TRAINED_DIR, "pipeline_meta.json")
    check("Pipeline meta exists", os.path.exists(meta_path))

    scaler_path = os.path.join(TRAINED_DIR, "scaler.pkl")
    check("Scaler artifact exists", os.path.exists(scaler_path))

    # ── Section 2: Model Artifact Integrity ───────────────────────────────
    print("\n  § 2. Model Artifact Integrity")

    model_size = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
    check("Model file > 1 MB", model_size > 1.0, f"{model_size:.1f} MB")
    check("Model file < 50 MB", model_size < 50.0, f"{model_size:.1f} MB")

    # Load model
    from models.multi_target_predictor import MultiTargetPredictor
    predictor = MultiTargetPredictor()
    try:
        predictor.load()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        print(f"    [ERROR] Failed to load model: {e}")
    check("Model loads without error", model_loaded)

    if model_loaded:
        # Check it has 4 estimators (one per target)
        n_estimators = len(predictor.model.estimators_)
        check("Model has 4 estimators (one per target)", n_estimators == 4, f"found {n_estimators}")

        # Check each estimator is XGBRegressor
        from xgboost import XGBRegressor
        all_xgb = all(isinstance(e, XGBRegressor) for e in predictor.model.estimators_)
        check("All estimators are XGBRegressor", all_xgb)

    # ── Section 3: Evaluation Report Integrity ────────────────────────────
    print("\n  § 3. Evaluation Report Integrity")

    with open(report_path, "r") as f:
        report = json.load(f)

    expected_sections = ["train_metrics", "cv_metrics", "test_metrics", "feature_importance", "hyperparameters"]
    for section in expected_sections:
        check(f"Report contains '{section}'", section in report)

    targets = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]
    for target in targets:
        check(f"Test metrics contain '{target}'", target in report.get("test_metrics", {}))

    # ── Section 4: Accuracy Thresholds (>90% per README spec) ─────────────
    print("\n  § 4. Accuracy Thresholds (>90% required)")

    test_metrics = report.get("test_metrics", {})

    for target in targets:
        m = test_metrics.get(target, {})
        acc = m.get("accuracy", 0)
        check(f"{target} accuracy > 90%", acc > 90.0, f"{acc:.1f}%")

    # Cross-validation should also pass
    cv_metrics = report.get("cv_metrics", {})
    for target in targets:
        m = cv_metrics.get(target, {})
        acc = m.get("accuracy", 0)
        check(f"{target} CV accuracy > 90%", acc > 90.0, f"{acc:.1f}%")

    # ── Section 5: Feature Importance Non-Zero ────────────────────────────
    print("\n  § 5. Feature Importance Validation")

    importance = report.get("feature_importance", {})
    check("Feature importance has 4 targets", len(importance) == 4)

    for target in targets:
        imp = importance.get(target, {})
        non_zero = sum(1 for v in imp.values() if v > 0)
        check(f"{target} has non-zero importances", non_zero > 0, f"{non_zero}/13 features")

    # Top feature for energy should be material_type or hold_time (domain knowledge)
    energy_imp = importance.get("energy_kwh", {})
    if energy_imp:
        top_energy = max(energy_imp, key=energy_imp.get)
        check(
            "Energy top feature is material_type or hold_time",
            top_energy in ("material_type", "hold_time"),
            f"found '{top_energy}'"
        )

    # ── Section 6: Hyperparameters Match README Spec ──────────────────────
    print("\n  § 6. Hyperparameter Compliance (README-locked)")

    hp = report.get("hyperparameters", {})
    check("n_estimators = 300", hp.get("n_estimators") == 300, f"got {hp.get('n_estimators')}")
    check("learning_rate = 0.05", hp.get("learning_rate") == 0.05, f"got {hp.get('learning_rate')}")
    check("max_depth = 6", hp.get("max_depth") == 6, f"got {hp.get('max_depth')}")
    check("subsample = 0.8", hp.get("subsample") == 0.8, f"got {hp.get('subsample')}")
    check("colsample_bytree = 0.8", hp.get("colsample_bytree") == 0.8, f"got {hp.get('colsample_bytree')}")
    check("random_state = 42", hp.get("random_state") == 42, f"got {hp.get('random_state')}")

    # ── Section 7: Prediction Sanity (Single Batch) ───────────────────────
    print("\n  § 7. Prediction Sanity Check")

    if model_loaded:
        sample_input = {
            "temperature": 183.0,
            "conveyor_speed": 75.0,
            "hold_time": 18.0,
            "batch_size": 500.0,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        }
        try:
            pred = predictor.predict_single(sample_input)
            pred_ok = True
        except Exception as e:
            pred_ok = False
            pred = {}
            print(f"    [ERROR] Prediction failed: {e}")

        check("Single prediction returns without error", pred_ok)

        if pred_ok:
            # Domain-reasonable ranges
            q = pred.get("quality_score", 0)
            check("quality_score in [60, 100]", 60 <= q <= 100, f"{q:.1f}")

            y = pred.get("yield_pct", 0)
            check("yield_pct in [70, 100]", 70 <= y <= 100, f"{y:.1f}")

            p = pred.get("performance_pct", 0)
            check("performance_pct in [60, 100]", 60 <= p <= 100, f"{p:.1f}")

            e = pred.get("energy_kwh", 0)
            check("energy_kwh in [25, 55]", 25 <= e <= 55, f"{e:.1f}")

            co2 = pred.get("co2_kg", 0)
            expected_co2 = e * 0.82
            check(
                "co2_kg = energy × 0.82",
                abs(co2 - expected_co2) < 0.01,
                f"{co2:.2f} vs expected {expected_co2:.2f}"
            )

    # ── Section 8: Data Integrity ─────────────────────────────────────────
    print("\n  § 8. Training Data Integrity")

    import pandas as pd
    train = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

    check("Train has 1600 rows", len(train) == 1600, f"found {len(train)}")
    check("Test has 400 rows", len(test) == 400, f"found {len(test)}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]

    all_cols_present = all(c in train.columns for c in feature_cols + target_cols)
    check("All feature + target columns in train set", all_cols_present)

    no_nans = train[feature_cols + target_cols].isna().sum().sum() == 0
    check("No NaN values in training features/targets", no_nans)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — F1.3 Multi-Target XGBoost verified!")
    else:
        print(f"  ⚠️  {failed} check(s) failed — review above.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
