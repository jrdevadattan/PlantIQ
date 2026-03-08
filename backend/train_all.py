#!/usr/bin/env python3
"""
PlantIQ — One-Click Training Script
=====================================
Trains ALL ML models in the correct dependency order.

Usage:
    cd backend
    python train_all.py

This script will:
    1. Generate synthetic batch data (2000 batches)
    2. Generate power curve signals (2000 .npy files)
    3. Run the 4-stage preprocessing pipeline
    4. Train the Multi-Target XGBoost predictor
    5. Train the LSTM Autoencoder (anomaly detection)
    6. Train the Fault Classifier (RandomForest)
    7. Generate conformal calibration intervals
    8. Initialize adaptive energy/carbon targets
    9. Discover golden batch signatures
    10. Run final evaluation

Total time: ~2-5 minutes depending on hardware.
"""

import os
import sys
import time

# ── Ensure we're in the backend directory ─────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

TRAINED_DIR = os.path.join(BACKEND_DIR, "models", "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)


def banner(msg: str) -> None:
    """Print a visible step banner."""
    print(f"\n{'='*60}")
    print(f"  Step: {msg}")
    print(f"{'='*60}\n")


def run_step(description: str, func) -> bool:
    """Run a training step with timing and error handling."""
    banner(description)
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"\n  ✅ {description} — completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {description} — FAILED after {elapsed:.1f}s")
        print(f"     Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════
# Step 1: Generate synthetic batch data
# ═══════════════════════════════════════════════════════════════
def step_generate_data() -> None:
    csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  batch_data.csv already exists ({len(df)} rows) — skipping")
        print(f"  (Delete data/batch_data.csv to regenerate)")
        return

    from data.generate_batch_data import main as generate_data_main
    generate_data_main()

    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  Generated {len(df)} batches → data/batch_data.csv")


# ═══════════════════════════════════════════════════════════════
# Step 2: Generate power curves
# ═══════════════════════════════════════════════════════════════
def step_generate_power_curves() -> None:
    curves_dir = os.path.join(BACKEND_DIR, "data", "power_curves")
    if os.path.exists(curves_dir):
        n = len([f for f in os.listdir(curves_dir) if f.endswith('.npy')])
        if n >= 100:
            print(f"  Power curves already exist ({n} files) — skipping")
            print(f"  (Delete data/power_curves/ to regenerate)")
            return

    from data.generate_power_curves import main as generate_curves_main
    generate_curves_main()

    n = len([f for f in os.listdir(curves_dir) if f.endswith('.npy')])
    print(f"  Generated {n} power curve files → data/power_curves/")


# ═══════════════════════════════════════════════════════════════
# Step 3: Run preprocessing pipeline
# ═══════════════════════════════════════════════════════════════
def step_preprocess() -> None:
    from preprocessing.pipeline import run_pipeline
    result = run_pipeline()
    print(f"  Preprocessing complete:")
    print(f"    Train set: {result.get('train_rows', '?')} rows")
    print(f"    Test set:  {result.get('test_rows', '?')} rows")
    print(f"    Features:  {result.get('feature_count', result.get('n_features', '?'))}")


# ═══════════════════════════════════════════════════════════════
# Step 4: Train XGBoost Multi-Target Predictor
# ═══════════════════════════════════════════════════════════════
def step_train_xgboost() -> None:
    from models.multi_target_predictor import train_and_evaluate
    metrics = train_and_evaluate()

    print(f"  XGBoost training complete!")
    for target in ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]:
        vals = metrics.get(target, {})
        r2 = vals.get('r2', 0)
        mae = vals.get('mae', 0)
        print(f"    {target}: R²={r2:.4f}, MAE={mae:.4f}")

    pkl_path = os.path.join(TRAINED_DIR, "multi_target.pkl")
    if os.path.exists(pkl_path):
        size_mb = os.path.getsize(pkl_path) / 1024 / 1024
        print(f"  Saved: models/trained/multi_target.pkl ({size_mb:.1f} MB)")
    else:
        raise FileNotFoundError("multi_target.pkl was not created!")


# ═══════════════════════════════════════════════════════════════
# Step 5: Train LSTM Autoencoder
# ═══════════════════════════════════════════════════════════════
def step_train_lstm() -> None:
    try:
        import torch
        print(f"  PyTorch {torch.__version__} available")
    except ImportError:
        print("  ⚠️  PyTorch not installed — skipping LSTM training")
        print("     Anomaly detection will use statistical fallback")
        print("     Install: pip install torch")
        return

    from models.lstm_autoencoder import train_autoencoder, save_model

    model, threshold, metadata = train_autoencoder()
    save_model(model, threshold, metadata)

    print(f"  LSTM Autoencoder training complete!")
    print(f"    Threshold: {threshold:.6f}")

    pt_path = os.path.join(TRAINED_DIR, "lstm_autoencoder.pt")
    if os.path.exists(pt_path):
        size_kb = os.path.getsize(pt_path) / 1024
        print(f"  Saved: models/trained/lstm_autoencoder.pt ({size_kb:.0f} KB)")


# ═══════════════════════════════════════════════════════════════
# Step 6: Train Fault Classifier
# ═══════════════════════════════════════════════════════════════
def step_train_fault_classifier() -> None:
    from models.fault_classifier import train as train_fc

    metrics = train_fc()
    print(f"  Fault Classifier training complete!")
    acc = metrics.get('test_accuracy_pct', metrics.get('accuracy', 0))
    print(f"    Accuracy: {acc:.1f}%")

    pkl_path = os.path.join(TRAINED_DIR, "fault_classifier.pkl")
    if os.path.exists(pkl_path):
        size_kb = os.path.getsize(pkl_path) / 1024
        print(f"  Saved: models/trained/fault_classifier.pkl ({size_kb:.0f} KB)")


# ═══════════════════════════════════════════════════════════════
# Step 7: Generate conformal calibration
# ═══════════════════════════════════════════════════════════════
def step_conformal_calibration() -> None:
    eval_path = os.path.join(TRAINED_DIR, "evaluation_report.json")
    if not os.path.exists(eval_path):
        print("  ⚠️  No evaluation_report.json — skipping conformal calibration")
        return

    from models.conformal_calibrator import generate_calibration
    generate_calibration()


# ═══════════════════════════════════════════════════════════════
# Step 8: Initialize adaptive targets
# ═══════════════════════════════════════════════════════════════
def step_init_targets() -> None:
    from models.adaptive_targets import AdaptiveTargetEngine
    import pandas as pd

    csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
    if not os.path.exists(csv_path):
        print("  ⚠️  No batch_data.csv — skipping adaptive targets")
        return

    engine = AdaptiveTargetEngine()
    df = pd.read_csv(csv_path)
    baseline = engine.initialize_from_data(df)
    print(f"  Adaptive targets initialized from {len(df)} batches")
    if isinstance(baseline, dict):
        for key, val in baseline.items():
            if isinstance(val, dict) and "mean" in val:
                print(f"    {key}: mean={val['mean']}, p90={val.get('p90', '?')}")


# ═══════════════════════════════════════════════════════════════
# Step 9: Discover golden signatures
# ═══════════════════════════════════════════════════════════════
def step_golden_signatures() -> None:
    from models.golden_signature import GoldenSignatureManager
    import pandas as pd

    csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
    if not os.path.exists(csv_path):
        print("  ⚠️  No batch_data.csv — skipping golden signatures")
        return

    df = pd.read_csv(csv_path)
    manager = GoldenSignatureManager()

    input_cols = [
        "temperature", "conveyor_speed", "hold_time", "batch_size",
        "material_type", "hour_of_day", "operator_exp",
    ]
    target_cols = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]

    sig = manager.discover_signatures(df, input_cols, target_cols)
    print(f"  Golden signature discovered!")
    print(f"    Composite score: {sig.get('composite_score', '?')}")
    print(f"    Pareto-optimal batches: {sig.get('n_pareto_optimal', '?')}")
    if "golden_parameters" in sig:
        for param, vals in sig["golden_parameters"].items():
            opt = vals.get("optimal", "?")
            print(f"    {param}: optimal={opt}")


# ═══════════════════════════════════════════════════════════════
# Step 10: Final evaluation
# ═══════════════════════════════════════════════════════════════
def step_evaluate() -> None:
    import json

    eval_path = os.path.join(TRAINED_DIR, "evaluation_report.json")
    if not os.path.exists(eval_path):
        print("  No evaluation report found — skipping")
        return

    with open(eval_path) as f:
        report = json.load(f)

    # Metrics may be under "test_metrics" or at top level
    metrics_section = report.get("test_metrics", report)

    print(f"  Final Model Evaluation Report:")
    print(f"  {'Target':<20} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Acc%':>8}")
    print(f"  {'-'*52}")
    for target in ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]:
        m = metrics_section.get(target, {})
        r2 = m.get('r2', m.get('accuracy', 0) / 100 if 'accuracy' in m else 0)
        mae = m.get('mae', 0)
        rmse = m.get('rmse', 0)
        acc = m.get('accuracy', 0)
        print(f"  {target:<20} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f} {acc:>7.2f}%")


# ═══════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════
def main() -> bool:
    print("""
╔══════════════════════════════════════════════════════════╗
║           PlantIQ — One-Click Model Training             ║
║                                                          ║
║  This will train all ML models from scratch.             ║
║  Estimated time: 2-5 minutes.                            ║
╚══════════════════════════════════════════════════════════╝
    """)

    total_start = time.time()
    results = []

    steps = [
        ("Generate synthetic batch data (2000 batches)",     step_generate_data),
        ("Generate power curve signals (2000 .npy files)",   step_generate_power_curves),
        ("Run 4-stage preprocessing pipeline",               step_preprocess),
        ("Train Multi-Target XGBoost predictor",             step_train_xgboost),
        ("Train LSTM Autoencoder (anomaly detection)",       step_train_lstm),
        ("Train Fault Classifier (RandomForest)",            step_train_fault_classifier),
        ("Generate conformal calibration intervals",         step_conformal_calibration),
        ("Initialize adaptive energy/carbon targets",        step_init_targets),
        ("Discover golden batch signatures",                 step_golden_signatures),
        ("Run final evaluation",                             step_evaluate),
    ]

    for desc, func in steps:
        ok = run_step(desc, func)
        results.append((desc, ok))

    # ── Final Summary ─────────────────────────────────────────
    total_elapsed = time.time() - total_start
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Steps: {passed}/{len(results)} passed, {failed} failed")
    print()

    # List all generated artifacts
    print("  Generated artifacts in models/trained/:")
    for f in sorted(os.listdir(TRAINED_DIR)):
        fpath = os.path.join(TRAINED_DIR, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"    ✅ {f} ({size/1024/1024:.1f} MB)")
        elif size > 1024:
            print(f"    ✅ {f} ({size/1024:.0f} KB)")
        else:
            print(f"    ✅ {f} ({size} bytes)")

    if failed > 0:
        print(f"\n  ⚠️  {failed} step(s) failed — check errors above")
        print(f"     The API will still start but some features may use fallbacks")
    else:
        print(f"\n  🎉 All models trained successfully!")
        print(f"     Start the API:  cd backend && python main.py")
        print(f"     Start the UI:   cd frontend && npm run dev")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
