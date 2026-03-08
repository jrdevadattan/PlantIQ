#!/usr/bin/env python3
"""
PlantIQ — Hackathon Data Training Pipeline
============================================
One-command script to train ALL models on the REAL hackathon data
from the 69997ffba83f5_problem_statement folder.

Uses:
  - _h_batch_production_data.xlsx → 60 batches × 15 columns (batch-level)
  - _h_batch_process_data.xlsx    → 60 batches × ~200 rows each (time-series)

Steps:
  1. Load & validate both Excel files
  2. Engineer pharmaceutical domain features
  3. Generate power curve .npy files from ALL 60 batches
  4. Train XGBoost multi-target predictor (7 quality targets)
  5. Compute quality compliance scores
  6. Set T001 as the golden signature (reference optimal batch)
  7. Train energy pattern analysis model
  8. Initialize adaptive targets from real data
  9. Generate conformal calibration intervals
 10. Save evaluation report with per-target accuracy

Golden Signature Batch: T001
  - Hardness: 95 N
  - Dissolution Rate: 89.3%
  - Content Uniformity: 98.7%
  - Friability: 0.65%
  - Tablet Weight: 199.8 mg
  - All metrics within pharmaceutical spec limits

Usage:
  cd backend
  python3 train_hackathon.py          # Train everything
  python3 train_hackathon.py --quick  # Skip power curve generation
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd

# Ensure backend/ is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

TRAINED_DIR = os.path.join(BACKEND_DIR, "models", "trained")
POWER_CURVES_DIR = os.path.join(BACKEND_DIR, "data", "power_curves")
os.makedirs(TRAINED_DIR, exist_ok=True)
os.makedirs(POWER_CURVES_DIR, exist_ok=True)

# Golden signature batch
GOLDEN_BATCH_ID = "T001"


# ──────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────
def _json_safe(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj).__name__} not JSON serializable")


def banner(step_num: int, title: str):
    """Print a step banner."""
    print(f"\n{'─' * 60}")
    print(f"  Step {step_num}/10: {title}")
    print(f"{'─' * 60}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def warn(msg: str):
    print(f"  ⚠️  {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")


# ══════════════════════════════════════════════════════
# Step 1: Load & Validate Hackathon Data
# ══════════════════════════════════════════════════════
def step_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both hackathon Excel files and validate structure."""
    banner(1, "Load & Validate Hackathon Data")

    from data.hackathon_adapter import HackathonDataAdapter

    adapter = HackathonDataAdapter()

    # Load production data (60 batches × 15 columns)
    production_df = adapter.load_production_data()
    ok(f"Production data: {production_df.shape[0]} batches × {production_df.shape[1]} columns")

    # Load ALL process data (60 batches, each ~200 rows)
    process_df = adapter.load_process_data()
    ok(f"Process data: {process_df.shape[0]} total rows from "
       f"{process_df['Batch_ID'].nunique()} batches")

    # Validate T001 exists
    t001_prod = production_df[production_df["Batch_ID"] == GOLDEN_BATCH_ID]
    t001_proc = process_df[process_df["Batch_ID"] == GOLDEN_BATCH_ID]
    if t001_prod.empty or t001_proc.empty:
        fail(f"Golden batch {GOLDEN_BATCH_ID} not found!")
        sys.exit(1)

    ok(f"Golden batch {GOLDEN_BATCH_ID}: "
       f"Hardness={t001_prod.iloc[0]['Hardness']}N, "
       f"Dissolution={t001_prod.iloc[0]['Dissolution_Rate']}%, "
       f"CU={t001_prod.iloc[0]['Content_Uniformity']}%, "
       f"Friability={t001_prod.iloc[0]['Friability']}%")

    return production_df, process_df


# ══════════════════════════════════════════════════════
# Step 2: Feature Engineering
# ══════════════════════════════════════════════════════
def step_engineer_features(production_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer pharmaceutical domain features from production data."""
    banner(2, "Feature Engineering (Pharmaceutical Domain)")

    from data.hackathon_adapter import HackathonDataAdapter

    adapter = HackathonDataAdapter()
    df = adapter.engineer_features(production_df)
    df = adapter.compute_quality_compliance(df)

    derived_cols = [
        "granulation_intensity", "drying_intensity", "compression_speed_ratio",
        "binder_per_time", "drying_efficiency", "force_lubricant_ratio",
        "machine_load_index",
    ]
    ok(f"Engineered {len(derived_cols)} derived features")
    ok(f"Quality compliance computed: avg composite score = "
       f"{df['composite_quality_score'].mean():.1f}%")

    return df


# ══════════════════════════════════════════════════════
# Step 3: Generate Power Curves from Process Data
# ══════════════════════════════════════════════════════
def step_generate_power_curves(process_df: pd.DataFrame) -> int:
    """Extract Power_Consumption_kW from each batch and save as .npy files.

    These are used by the LSTM Autoencoder for energy pattern anomaly
    detection. Each batch gets a standardized 1800-point curve (resampled
    from the raw ~200 minute-by-minute readings to match the expected format).
    """
    banner(3, "Generate Power Curves from Process Data")

    batch_ids = sorted(process_df["Batch_ID"].unique())
    count = 0

    for batch_id in batch_ids:
        batch_data = process_df[process_df["Batch_ID"] == batch_id]
        power = batch_data["Power_Consumption_kW"].values

        if len(power) < 10:
            warn(f"Batch {batch_id}: only {len(power)} power readings, skipping")
            continue

        # Resample to 1800 points (standard format for LSTM autoencoder)
        # Each original minute gets ~8.5 seconds resolution (1800 / ~210 ≈ 8.6)
        x_original = np.linspace(0, 1, len(power))
        x_target = np.linspace(0, 1, 1800)
        power_1800 = np.interp(x_target, x_original, power)

        # Save as .npy with the batch numbering convention (B0000, B0001, etc.)
        batch_num = int(batch_id.replace("T", "").replace("t", "")) - 1  # T001 → 0
        npy_path = os.path.join(POWER_CURVES_DIR, f"B{batch_num:04d}.npy")
        np.save(npy_path, power_1800)
        count += 1

    ok(f"Generated {count} power curve files (1800 points each)")
    ok(f"Saved to {POWER_CURVES_DIR}")

    # Also save T001's curve separately as the golden reference
    t001_data = process_df[process_df["Batch_ID"] == GOLDEN_BATCH_ID]
    t001_power = t001_data["Power_Consumption_kW"].values
    t001_resampled = np.interp(
        np.linspace(0, 1, 1800),
        np.linspace(0, 1, len(t001_power)),
        t001_power,
    )
    golden_path = os.path.join(TRAINED_DIR, "golden_power_curve_T001.npy")
    np.save(golden_path, t001_resampled)
    ok(f"Golden power curve (T001) saved: {golden_path}")

    return count


# ══════════════════════════════════════════════════════
# Step 4: Train Multi-Target XGBoost Predictor
# ══════════════════════════════════════════════════════
def step_train_predictor(engineered_df: pd.DataFrame) -> dict:
    """Train XGBoost multi-target model on ALL 7 quality targets.

    Inputs (14 features):
      7 raw: Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time,
             Compression_Force, Machine_Speed, Lubricant_Conc
      7 derived: granulation_intensity, drying_intensity, compression_speed_ratio,
                 binder_per_time, drying_efficiency, force_lubricant_ratio,
                 machine_load_index

    Outputs (7 quality targets):
      Moisture_Content, Tablet_Weight, Hardness, Friability,
      Disintegration_Time, Dissolution_Rate, Content_Uniformity
    """
    banner(4, "Train Multi-Target XGBoost Predictor")

    from data.hackathon_adapter import (
        HACKATHON_INPUT_COLS, HACKATHON_TARGET_COLS, QUALITY_SPECS,
        train_on_hackathon_data,
    )

    results = train_on_hackathon_data(verbose=True)

    # Check if we meet the >90% accuracy target
    avg_accuracy = results.get("overall", {}).get("avg_accuracy_pct", 0)
    if avg_accuracy >= 90:
        ok(f"≥90% accuracy target MET: {avg_accuracy:.2f}%")
    else:
        warn(f"Accuracy is {avg_accuracy:.2f}% — below 90% target "
             f"(expected with only 60 samples)")

    return results


# ══════════════════════════════════════════════════════
# Step 5: Create Batch Metadata CSV for Pipeline
# ══════════════════════════════════════════════════════
def step_create_batch_csv(
    production_df: pd.DataFrame,
    process_df: pd.DataFrame,
) -> str:
    """Create a batch_data.csv that merges production + process summary data.

    This CSV is consumed by the preprocessing pipeline, LSTM autoencoder,
    and fault classifier. It bridges the hackathon data to our existing
    pipeline's expected format.
    """
    banner(5, "Create Unified Batch CSV")

    from data.hackathon_adapter import HACKATHON_INPUT_COLS, HACKATHON_TARGET_COLS

    df = production_df.copy()

    # Add process-level summary per batch
    energy_per_batch = {}
    duration_per_batch = {}
    avg_temp_per_batch = {}
    avg_vibration_per_batch = {}

    for batch_id in df["Batch_ID"].unique():
        batch_proc = process_df[process_df["Batch_ID"] == batch_id]
        if batch_proc.empty:
            continue

        # Energy (kWh) = average power × duration in hours
        avg_power = batch_proc["Power_Consumption_kW"].mean()
        duration_min = batch_proc["Time_Minutes"].max() - batch_proc["Time_Minutes"].min() + 1
        energy_kwh = avg_power * duration_min / 60.0

        energy_per_batch[batch_id] = round(energy_kwh, 3)
        duration_per_batch[batch_id] = int(duration_min)
        avg_temp_per_batch[batch_id] = round(batch_proc["Temperature_C"].mean(), 2)
        avg_vibration_per_batch[batch_id] = round(batch_proc["Vibration_mm_s"].mean(), 3)

    df["energy_kwh"] = df["Batch_ID"].map(energy_per_batch)
    df["duration_minutes"] = df["Batch_ID"].map(duration_per_batch)
    df["avg_process_temp"] = df["Batch_ID"].map(avg_temp_per_batch)
    df["avg_vibration"] = df["Batch_ID"].map(avg_vibration_per_batch)
    df["co2_kg"] = (df["energy_kwh"] * 0.82).round(3)

    # Map batch IDs to power curve file indices
    batch_ids_sorted = sorted(df["Batch_ID"].unique())
    batch_to_idx = {bid: i for i, bid in enumerate(batch_ids_sorted)}
    df["batch_id"] = df["Batch_ID"].map(
        lambda x: f"B{batch_to_idx.get(x, 0):04d}"
    )

    # Assign fault_type based on vibration/energy patterns
    # T001 is normal (golden). Others classified by deviation from T001.
    t001_energy = energy_per_batch.get("T001", 0)
    t001_vibration = avg_vibration_per_batch.get("T001", 0)

    def classify_fault(row):
        """Simple rule-based fault classification for hackathon data."""
        vib = row.get("avg_vibration", 0) or 0
        eng = row.get("energy_kwh", 0) or 0

        # High vibration → bearing_wear
        if vib > t001_vibration * 1.5:
            return "bearing_wear"
        # High energy with normal vibration → calibration_needed
        if eng > t001_energy * 1.3 and vib <= t001_vibration * 1.2:
            return "calibration_needed"
        # Very variable moisture → wet_material
        if row.get("Moisture_Content", 2.0) > 3.0:
            return "wet_material"
        return "normal"

    df["fault_type"] = df.apply(classify_fault, axis=1)

    # Add timestamp (synthetic, for pipeline compatibility)
    df["timestamp"] = pd.date_range(start="2026-01-01", periods=len(df), freq="4h")

    csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
    df.to_csv(csv_path, index=False)
    ok(f"Batch CSV: {df.shape[0]} rows × {df.shape[1]} columns")
    ok(f"Saved to {csv_path}")

    fault_counts = df["fault_type"].value_counts().to_dict()
    ok(f"Fault distribution: {fault_counts}")

    return csv_path


# ══════════════════════════════════════════════════════
# Step 6: Set T001 as Golden Signature
# ══════════════════════════════════════════════════════
def step_set_golden_signature(engineered_df: pd.DataFrame):
    """Set batch T001 as the golden reference signature.

    T001 has all quality metrics within pharmaceutical spec limits:
      - Hardness: 95 N (spec: 80–120)
      - Dissolution Rate: 89.3% (spec: 85–100)
      - Content Uniformity: 98.7% (spec: 95–105)
      - Friability: 0.65% (spec: 0–1.0)
      - Tablet Weight: 199.8 mg (spec: 198–202)
      - Disintegration Time: 8.2 min (spec: 5–15)
      - Moisture Content: 2.1% (spec: 1–3)
    """
    banner(6, f"Set {GOLDEN_BATCH_ID} as Golden Signature")

    from data.hackathon_adapter import HACKATHON_INPUT_COLS, HACKATHON_TARGET_COLS
    from models.golden_signature import GoldenSignatureManager

    gsm = GoldenSignatureManager()

    # Discover signatures from full data (Pareto-optimal analysis)
    input_cols = HACKATHON_INPUT_COLS + [
        "granulation_intensity", "drying_intensity", "compression_speed_ratio",
        "binder_per_time", "drying_efficiency", "force_lubricant_ratio",
        "machine_load_index",
    ]
    # Only use the input cols that exist in the dataframe
    input_cols = [c for c in input_cols if c in engineered_df.columns]
    target_cols = [c for c in HACKATHON_TARGET_COLS if c in engineered_df.columns]

    # Discover golden signatures from ALL data first (Pareto analysis)
    signature = gsm.discover_signatures(
        df=engineered_df,
        input_cols=input_cols,
        target_cols=target_cols,
        n_top=5,
    )

    ok(f"Pareto analysis: {signature['n_pareto_optimal']} Pareto-optimal batches")
    ok(f"Top batches: {signature.get('top_batch_ids', [])}")

    # Now explicitly set T001 as the golden reference
    t001 = engineered_df[engineered_df["Batch_ID"] == GOLDEN_BATCH_ID].iloc[0]

    t001_params = {col: float(t001[col]) for col in input_cols if col in t001.index}
    t001_targets = {col: float(t001[col]) for col in target_cols if col in t001.index}

    # Create a dedicated T001 golden signature
    t001_signature = {
        "scenario_id": "T001_golden_reference",
        "created_at": pd.Timestamp.now().isoformat(),
        "n_batches_analyzed": len(engineered_df),
        "n_pareto_optimal": 1,
        "golden_batch_id": GOLDEN_BATCH_ID,
        "golden_parameters": {
            col: {
                "optimal": round(float(t001[col]), 4),
                "min_range": round(float(engineered_df[col].quantile(0.1)), 4),
                "max_range": round(float(engineered_df[col].quantile(0.9)), 4),
                "std": round(float(engineered_df[col].std()), 4),
            }
            for col in input_cols if col in t001.index
        },
        "expected_targets": {
            col: {
                "expected": round(float(t001[col]), 4),
                "best": round(float(t001[col]), 4),
                "direction": (
                    "minimize" if col in ("Friability", "Disintegration_Time")
                    else "target" if col in ("Content_Uniformity", "Moisture_Content", "Tablet_Weight")
                    else "maximize"
                ),
            }
            for col in target_cols if col in t001.index
        },
        "composite_score": round(float(
            engineered_df[engineered_df["Batch_ID"] == GOLDEN_BATCH_ID]
            .get("composite_quality_score", pd.Series([85.0])).iloc[0]
        ), 4),
        "top_batch_ids": [GOLDEN_BATCH_ID],
    }

    gsm.signatures["T001_golden_reference"] = t001_signature
    gsm.history.append({
        "action": "set_golden_reference",
        "scenario_id": "T001_golden_reference",
        "batch_id": GOLDEN_BATCH_ID,
        "timestamp": pd.Timestamp.now().isoformat(),
    })
    gsm.save()

    ok(f"T001 golden signature saved with {len(t001_params)} parameters")
    ok(f"T001 targets: {t001_targets}")

    # Compare all other batches against T001
    alignment_scores = []
    for _, row in engineered_df.iterrows():
        if row["Batch_ID"] == GOLDEN_BATCH_ID:
            continue
        batch_params = {col: float(row[col]) for col in input_cols if col in row.index}
        batch_targets = {col: float(row[col]) for col in target_cols if col in row.index}
        comparison = gsm.compare_batch(
            batch_params, batch_targets,
            scenario_id="T001_golden_reference",
        )
        if comparison.get("has_signature"):
            alignment_scores.append(comparison["alignment_score"])

    if alignment_scores:
        ok(f"Fleet alignment to T001: avg={np.mean(alignment_scores):.1f}%, "
           f"min={np.min(alignment_scores):.1f}%, max={np.max(alignment_scores):.1f}%")


# ══════════════════════════════════════════════════════
# Step 7: Train Energy Pattern Analysis
# ══════════════════════════════════════════════════════
def step_energy_pattern_analysis(process_df: pd.DataFrame) -> dict:
    """Analyse energy consumption patterns across all batches.

    Per problem statement: "Create models that analyse power consumption
    patterns for asset and process reliability insights."
    """
    banner(7, "Energy Pattern Analysis (All Batches)")

    from data.hackathon_adapter import HackathonDataAdapter

    results = {
        "per_batch": {},
        "golden_reference": {},
        "fleet_summary": {},
    }

    batch_ids = sorted(process_df["Batch_ID"].unique())

    # Analyse T001 first (golden reference)
    adapter = HackathonDataAdapter()
    t001_data = process_df[process_df["Batch_ID"] == GOLDEN_BATCH_ID]
    adapter._process_df = t001_data
    t001_analysis = adapter.analyze_process_phases()
    t001_attribution = adapter.attribute_energy_patterns()

    results["golden_reference"] = {
        "batch_id": GOLDEN_BATCH_ID,
        "total_energy_kwh": t001_analysis["total_energy_kwh"],
        "total_co2_kg": t001_analysis["total_co2_kg"],
        "attribution": t001_attribution["overall_attribution"],
        "phase_energy": t001_analysis["phase_energy_breakdown"],
    }
    ok(f"T001 energy: {t001_analysis['total_energy_kwh']:.2f} kWh, "
       f"CO₂: {t001_analysis['total_co2_kg']:.2f} kg")

    # Analyse all batches and compare to T001
    energy_values = []
    anomaly_counts = []

    for batch_id in batch_ids:
        batch_data = process_df[process_df["Batch_ID"] == batch_id]
        adapter._process_df = batch_data

        try:
            analysis = adapter.analyze_process_phases()
            attribution = adapter.attribute_energy_patterns()

            results["per_batch"][batch_id] = {
                "energy_kwh": analysis["total_energy_kwh"],
                "co2_kg": analysis["total_co2_kg"],
                "attribution": attribution["overall_attribution"],
                "n_anomaly_indicators": len(analysis.get("anomaly_indicators", [])),
                "n_asset_issues": len(attribution.get("asset_health_indicators", [])),
                "n_process_issues": len(attribution.get("process_deviation_indicators", [])),
            }
            energy_values.append(analysis["total_energy_kwh"])
            anomaly_counts.append(len(analysis.get("anomaly_indicators", [])))
        except Exception as e:
            warn(f"Batch {batch_id} analysis failed: {e}")

    results["fleet_summary"] = {
        "total_batches": len(batch_ids),
        "avg_energy_kwh": round(float(np.mean(energy_values)), 3),
        "std_energy_kwh": round(float(np.std(energy_values)), 3),
        "min_energy_kwh": round(float(np.min(energy_values)), 3),
        "max_energy_kwh": round(float(np.max(energy_values)), 3),
        "avg_co2_kg": round(float(np.mean(energy_values) * 0.82), 3),
        "batches_with_anomalies": sum(1 for c in anomaly_counts if c > 0),
    }

    ok(f"Fleet energy: avg={results['fleet_summary']['avg_energy_kwh']:.2f} kWh "
       f"(±{results['fleet_summary']['std_energy_kwh']:.2f})")
    ok(f"Batches with anomaly indicators: "
       f"{results['fleet_summary']['batches_with_anomalies']}/{len(batch_ids)}")

    # Save energy analysis report
    report_path = os.path.join(TRAINED_DIR, "energy_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    ok(f"Energy analysis saved: {report_path}")

    return results


# ══════════════════════════════════════════════════════
# Step 8: Initialize Adaptive Targets
# ══════════════════════════════════════════════════════
def step_adaptive_targets(engineered_df: pd.DataFrame, process_df: pd.DataFrame):
    """Set adaptive energy/carbon targets based on real batch data."""
    banner(8, "Initialize Adaptive Targets")

    from models.adaptive_targets import AdaptiveTargetEngine

    # We need energy_kwh per batch for the adaptive targets
    ate = AdaptiveTargetEngine()

    # Build a DataFrame with energy data
    energy_data = []
    for _, row in engineered_df.iterrows():
        batch_id = row["Batch_ID"]
        batch_proc = process_df[process_df["Batch_ID"] == batch_id]
        if batch_proc.empty:
            continue

        avg_power = batch_proc["Power_Consumption_kW"].mean()
        duration = batch_proc["Time_Minutes"].max() - batch_proc["Time_Minutes"].min() + 1
        energy_kwh = avg_power * duration / 60.0

        energy_data.append({
            "batch_id": batch_id,
            "energy_kwh": round(energy_kwh, 3),
            "quality_score": row.get("composite_quality_score", 85.0),
            "yield_pct": row.get("Dissolution_Rate", 90.0),
            "performance_pct": row.get("Hardness", 90.0),
        })

    energy_df = pd.DataFrame(energy_data)

    try:
        ate.initialize_from_data(energy_df)
        ok(f"Adaptive targets initialized from {len(energy_df)} batches")
        ok(f"Baseline energy: {energy_df['energy_kwh'].mean():.2f} kWh")
    except Exception as e:
        warn(f"Adaptive targets initialization: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════
# Step 9: Train Fault Classifier on Power Curves
# ══════════════════════════════════════════════════════
def step_train_fault_classifier(batch_csv_path: str):
    """Train RandomForest fault classifier on power curve features."""
    banner(9, "Train Fault Classifier on Power Curves")

    try:
        from models.fault_classifier import train as train_fc
        result = train_fc()
        ok(f"Fault classifier trained: {result}")
    except Exception as e:
        warn(f"Fault classifier training: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════
# Step 10: Final Evaluation Report
# ══════════════════════════════════════════════════════
def step_final_report(predictor_results: dict, energy_results: dict):
    """Generate comprehensive evaluation report."""
    banner(10, "Final Evaluation Report")

    from data.hackathon_adapter import QUALITY_SPECS

    report = {
        "project": "PlantIQ — AI-Driven Manufacturing Intelligence",
        "track": "Track A: Predictive Modelling Specialization",
        "data_source": "hackathon_problem_statement",
        "golden_batch": GOLDEN_BATCH_ID,
        "prediction_results": predictor_results,
        "energy_analysis": {
            "golden_energy_kwh": energy_results.get("golden_reference", {}).get("total_energy_kwh"),
            "fleet_avg_energy_kwh": energy_results.get("fleet_summary", {}).get("avg_energy_kwh"),
            "batches_with_anomalies": energy_results.get("fleet_summary", {}).get("batches_with_anomalies"),
        },
        "models_trained": [],
        "artifacts": [],
    }

    # Check which artifacts exist
    artifact_checks = [
        ("hackathon_model.pkl", "XGBoost Multi-Target Predictor"),
        ("hackathon_scaler.pkl", "Feature Scaler"),
        ("hackathon_evaluation.json", "Evaluation Report"),
        ("golden_signatures.json", "Golden Signatures (T001 reference)"),
        ("golden_power_curve_T001.npy", "T001 Power Curve Reference"),
        ("energy_analysis_report.json", "Energy Pattern Analysis"),
        ("fault_classifier.pkl", "Fault Classifier"),
        ("adaptive_targets.json", "Adaptive Targets"),
    ]

    for filename, description in artifact_checks:
        path = os.path.join(TRAINED_DIR, filename)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        report["artifacts"].append({
            "file": filename,
            "description": description,
            "exists": exists,
            "size_bytes": size,
        })
        if exists:
            ok(f"{filename} ({size:,} bytes) — {description}")
            report["models_trained"].append(description)
        else:
            warn(f"{filename} — {description} (NOT FOUND)")

    # Print target accuracy summary
    per_target = predictor_results.get("per_target", {})
    if per_target:
        print(f"\n  {'Target':<25} {'MAE':>8} {'RMSE':>8} {'Accuracy':>10} {'Spec':>15}")
        print(f"  {'─' * 70}")
        for target, metrics in per_target.items():
            spec = QUALITY_SPECS.get(target, {})
            spec_str = f"{spec.get('min', '?')}–{spec.get('max', '?')} {spec.get('unit', '')}"
            print(f"  {target:<25} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f} "
                  f"{metrics['accuracy_pct']:>9.2f}% {spec_str:>15}")

        overall = predictor_results.get("overall", {})
        print(f"\n  Overall Accuracy: {overall.get('avg_accuracy_pct', 0):.2f}%")
        print(f"  Meets ≥90% Target: {'✅ YES' if overall.get('meets_90pct_target') else '⚠️  NO'}")

    # Save final report
    report_path = os.path.join(TRAINED_DIR, "hackathon_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)
    ok(f"Final report: {report_path}")


# ══════════════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════════════
def main():
    """Run the complete hackathon training pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="PlantIQ — Hackathon Training Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Skip power curve generation (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PlantIQ — Hackathon Training Pipeline")
    print("  Data: 69997ffba83f5_problem_statement")
    print(f"  Golden Signature Batch: {GOLDEN_BATCH_ID}")
    print("=" * 60)

    t_start = time.time()
    results = {}

    # Step 1: Load data
    production_df, process_df = step_load_data()

    # Step 2: Feature engineering
    engineered_df = step_engineer_features(production_df)

    # Step 3: Generate power curves
    if not args.quick:
        n_curves = step_generate_power_curves(process_df)
    else:
        warn("Skipping power curve generation (--quick mode)")

    # Step 4: Train multi-target predictor
    predictor_results = step_train_predictor(engineered_df)

    # Step 5: Create unified batch CSV
    batch_csv = step_create_batch_csv(production_df, process_df)

    # Step 6: Set T001 as golden signature
    step_set_golden_signature(engineered_df)

    # Step 7: Energy pattern analysis
    energy_results = step_energy_pattern_analysis(process_df)

    # Step 8: Adaptive targets
    step_adaptive_targets(engineered_df, process_df)

    # Step 9: Fault classifier
    step_train_fault_classifier(batch_csv)

    # Step 10: Final report
    step_final_report(predictor_results, energy_results)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Training complete in {elapsed:.1f}s")
    print(f"  Golden batch: {GOLDEN_BATCH_ID}")
    print(f"  Models saved to: {TRAINED_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
