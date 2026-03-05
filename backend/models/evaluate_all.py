"""
PlantIQ — Model Evaluation Report Generator
=============================================
Loads all trained models and produces a comprehensive performance report
matching the README's expected evaluation output format.

Currently evaluates:
  - Multi-Target XGBoost Predictor (4 targets)

Future additions (Tier 2):
  - LSTM Autoencoder (anomaly detection F1)
  - Fault Type Classifier (classification accuracy)

Usage:
  python models/evaluate_all.py

Spec: README — "python models/evaluate_all.py"
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "evaluation_report.json")


def main():
    """Print the formatted model performance report."""
    if not os.path.exists(REPORT_PATH):
        print("ERROR: No evaluation report found. Run training first:")
        print("  python models/multi_target_predictor.py --train")
        sys.exit(1)

    with open(REPORT_PATH) as f:
        report = json.load(f)

    test_metrics = report.get("test_metrics", {})
    cv_metrics = report.get("cv_metrics", {})

    # ── Formatted report matching README spec ──
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║         MODEL PERFORMANCE REPORT                 ║")
    print("╠═══════════════════════╦══════════╦═══════════════╣")
    print("║ Target                ║ Accuracy ║ Status        ║")
    print("╠═══════════════════════╬══════════╬═══════════════╣")

    all_pass = True
    for target in ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]:
        m = test_metrics.get(target, {})
        acc = m.get("accuracy", 0)
        status = "✅ PASS" if acc >= 90 else "❌ FAIL"
        if acc < 90:
            all_pass = False
        print(f"║ {target:21s} ║  {acc:5.1f}%  ║ {status:13s} ║")

    print("╠═══════════════════════╬══════════╬═══════════════╣")

    # Placeholder for future models (LSTM, Fault Classifier)
    print("║ LSTM Anomaly F1       ║  -.---   ║ ⏳ PENDING    ║")
    print("║ Fault Classifier Acc  ║  --.-%   ║ ⏳ PENDING    ║")

    print("╚═══════════════════════╩══════════╩═══════════════╝")

    if all_pass:
        print("All targets exceed 90% accuracy requirement ✅")
    else:
        print("⚠️  Some targets below 90% — review needed")

    # ── Detailed metrics table ──
    print("\n┌─ DETAILED TEST SET METRICS ─────────────────────────┐")
    print(f"│ {'Target':20s} │ {'MAE':>7s} │ {'RMSE':>7s} │ {'MAPE':>6s} │ {'Acc':>7s} │")
    print(f"├{'─'*22}┼{'─'*9}┼{'─'*9}┼{'─'*8}┼{'─'*9}┤")
    for target in ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]:
        m = test_metrics.get(target, {})
        unit = "kWh" if target == "energy_kwh" else "%"
        print(f"│ {target:20s} │ {m.get('mae', 0):5.2f}{unit:>2s} │ "
              f"{m.get('rmse', 0):5.2f}{unit:>2s} │ {m.get('mape', 0):5.1f}% │ "
              f"{m.get('accuracy', 0):6.1f}% │")
    print(f"└{'─'*22}┴{'─'*9}┴{'─'*9}┴{'─'*8}┴{'─'*9}┘")

    # ── Cross-validation metrics ──
    print(f"\n┌─ CROSS-VALIDATION METRICS (TimeSeriesSplit, 5 folds) ┐")
    print(f"│ {'Target':20s} │ {'MAE':>7s} │ {'MAPE':>6s} │ {'Acc':>7s} │")
    print(f"├{'─'*22}┼{'─'*9}┼{'─'*8}┼{'─'*9}┤")
    for target in ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]:
        m = cv_metrics.get(target, {})
        unit = "kWh" if target == "energy_kwh" else "%"
        print(f"│ {target:20s} │ {m.get('mae', 0):5.2f}{unit:>2s} │ "
              f"{m.get('mape', 0):5.1f}% │ {m.get('accuracy', 0):6.1f}% │")
    print(f"└{'─'*22}┴{'─'*9}┴{'─'*8}┴{'─'*9}┘")

    # ── Feature importance ──
    importance = report.get("feature_importance", {})
    if importance:
        print("\n┌─ TOP FEATURE IMPORTANCE PER TARGET ─────────────────┐")
        for target, imp in importance.items():
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
            bars = "  ".join(f"{f}({v:.2f})" for f, v in sorted_imp)
            print(f"│ {target:20s} │ {bars}")
        print(f"└{'─'*54}┘")

    # ── Training info ──
    print(f"\nTraining: {report.get('train_rows', '?')} rows | "
          f"Test: {report.get('test_rows', '?')} rows")
    params = report.get("xgboost_params", {})
    print(f"XGBoost: n_estimators={params.get('n_estimators')}, "
          f"lr={params.get('learning_rate')}, "
          f"max_depth={params.get('max_depth')}")


if __name__ == "__main__":
    main()
