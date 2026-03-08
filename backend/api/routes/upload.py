"""
PlantIQ — Hackathon Data Upload Endpoint
==========================================
POST /upload — Accept CSV or Excel files with pharmaceutical batch data,
               run feature engineering + XGBoost prediction, and return results.

Accepts the hackathon data format:
  Input columns:  Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time,
                  Compression_Force, Machine_Speed, Lubricant_Conc
  Target columns: Moisture_Content, Tablet_Weight, Hardness, Friability,
                  Disintegration_Time, Dissolution_Rate, Content_Uniformity

Also compares each uploaded batch against the T001 golden signature.
"""
from __future__ import annotations

import io
import os
import sys
import json
import logging
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Query

logger = logging.getLogger("plantiq.upload")

router = APIRouter(tags=["upload"])

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)

# ── Hackathon column schema ──────────────────────────────────
HACKATHON_INPUT_COLS = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
    "Compression_Force", "Machine_Speed", "Lubricant_Conc",
]

HACKATHON_TARGET_COLS = [
    "Moisture_Content", "Tablet_Weight", "Hardness", "Friability",
    "Disintegration_Time", "Dissolution_Rate", "Content_Uniformity",
]

DERIVED_COLS = [
    "granulation_intensity", "drying_intensity", "compression_speed_ratio",
    "binder_per_time", "drying_efficiency", "force_lubricant_ratio",
    "machine_load_index",
]

ALL_FEATURE_COLS = HACKATHON_INPUT_COLS + DERIVED_COLS

# Flexible column aliases so users can use different names
COLUMN_ALIASES = {
    "granulation_time": "Granulation_Time",
    "binder_amount": "Binder_Amount",
    "drying_temp": "Drying_Temp",
    "drying_time": "Drying_Time",
    "compression_force": "Compression_Force",
    "machine_speed": "Machine_Speed",
    "lubricant_conc": "Lubricant_Conc",
    "gran_time": "Granulation_Time",
    "granulation": "Granulation_Time",
    "binder": "Binder_Amount",
    "binder_amt": "Binder_Amount",
    "dry_temp": "Drying_Temp",
    "drying_temperature": "Drying_Temp",
    "dry_time": "Drying_Time",
    "comp_force": "Compression_Force",
    "compression": "Compression_Force",
    "speed": "Machine_Speed",
    "machine_spd": "Machine_Speed",
    "lubricant": "Lubricant_Conc",
    "lubricant_concentration": "Lubricant_Conc",
    "lub_conc": "Lubricant_Conc",
    "moisture": "Moisture_Content",
    "moisture_content": "Moisture_Content",
    "tablet_weight": "Tablet_Weight",
    "weight": "Tablet_Weight",
    "hardness": "Hardness",
    "friability": "Friability",
    "disintegration_time": "Disintegration_Time",
    "disintegration": "Disintegration_Time",
    "dissolution_rate": "Dissolution_Rate",
    "dissolution": "Dissolution_Rate",
    "content_uniformity": "Content_Uniformity",
    "uniformity": "Content_Uniformity",
}

# Valid ranges for inputs
VALID_RANGES = {
    "Granulation_Time": (5, 35),
    "Binder_Amount": (4, 16),
    "Drying_Temp": (35, 80),
    "Drying_Time": (10, 55),
    "Compression_Force": (3, 22),
    "Machine_Speed": (80, 300),
    "Lubricant_Conc": (0.3, 3.5),
}

# Pharmaceutical quality spec limits
QUALITY_SPECS = {
    "Moisture_Content":    {"min": 1.0, "max": 3.0, "unit": "%",   "optimal": 2.0},
    "Tablet_Weight":       {"min": 198, "max": 202, "unit": "mg",  "optimal": 200},
    "Hardness":            {"min": 80,  "max": 120, "unit": "N",   "optimal": 100},
    "Friability":          {"min": 0.0, "max": 1.0, "unit": "%",   "optimal": 0.5},
    "Disintegration_Time": {"min": 5,   "max": 15,  "unit": "min", "optimal": 10},
    "Dissolution_Rate":    {"min": 85,  "max": 100, "unit": "%",   "optimal": 92},
    "Content_Uniformity":  {"min": 95,  "max": 105, "unit": "%",   "optimal": 100},
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map user column names to internal names via alias table."""
    renamed = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace("-", "_")
        if clean in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[clean]
    if renamed:
        df = df.rename(columns=renamed)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 derived pharmaceutical features."""
    df = df.copy()
    for col in HACKATHON_INPUT_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    df["granulation_intensity"] = (df["Granulation_Time"] * df["Binder_Amount"]).round(2)
    df["drying_intensity"] = (df["Drying_Temp"] * df["Drying_Time"]).round(2)
    df["compression_speed_ratio"] = (
        df["Compression_Force"] / df["Machine_Speed"].clip(lower=1)
    ).round(4)
    df["binder_per_time"] = (
        df["Binder_Amount"] / df["Granulation_Time"].clip(lower=1)
    ).round(4)
    df["drying_efficiency"] = (
        df["Drying_Temp"] / df["Drying_Time"].clip(lower=1)
    ).round(4)
    df["force_lubricant_ratio"] = (
        df["Compression_Force"] / df["Lubricant_Conc"].clip(lower=0.01)
    ).round(4)
    df["machine_load_index"] = (df["Machine_Speed"] * df["Compression_Force"]).round(2)
    return df


def _check_quality_compliance(preds: dict) -> dict:
    """Check if predicted values are within pharmaceutical spec limits."""
    compliance = {}
    for target, value in preds.items():
        if target in QUALITY_SPECS:
            spec = QUALITY_SPECS[target]
            in_spec = spec["min"] <= value <= spec["max"]
            deviation = abs(value - spec["optimal"]) / spec["optimal"] * 100
            compliance[target] = {
                "value": round(value, 4),
                "in_spec": in_spec,
                "spec_range": f"{spec['min']}–{spec['max']} {spec['unit']}",
                "optimal": spec["optimal"],
                "deviation_pct": round(deviation, 2),
            }
    return compliance


def _compare_to_golden(batch_params: dict) -> Optional[dict]:
    """Compare uploaded batch against T001 golden signature."""
    try:
        sig_path = os.path.join(BACKEND_DIR, "models", "trained", "golden_signatures.json")
        if not os.path.exists(sig_path):
            return None
        with open(sig_path) as f:
            data = json.load(f)
        sig = data.get("signatures", {}).get("T001_golden_reference")
        if not sig:
            return None

        deviations = {}
        for param, spec in sig.get("golden_parameters", {}).items():
            if param in batch_params:
                current = batch_params[param]
                optimal = spec["optimal"]
                dev_pct = abs(current - optimal) / abs(optimal) * 100 if optimal != 0 else 0
                deviations[param] = {
                    "your_value": round(current, 4),
                    "golden_value": optimal,
                    "deviation_pct": round(dev_pct, 2),
                    "in_range": spec["min_range"] <= current <= spec["max_range"],
                }

        avg_dev = np.mean([d["deviation_pct"] for d in deviations.values()]) if deviations else 0
        return {
            "golden_batch": "T001",
            "alignment_score": round(max(0, 100 - avg_dev), 2),
            "parameter_deviations": deviations,
        }
    except Exception as e:
        logger.warning(f"Golden comparison failed: {e}")
        return None


@router.post("/upload")
async def upload_data(
    file: UploadFile = File(..., description="CSV or Excel file with batch data"),
):
    """Upload pharmaceutical batch data (CSV/Excel) and get quality predictions.

    **Required columns** (7 input parameters):
    - Granulation_Time (min), Binder_Amount, Drying_Temp (°C),
    - Drying_Time (min), Compression_Force (kN), Machine_Speed (RPM),
    - Lubricant_Conc

    **Returns** predictions for 7 quality targets plus golden signature comparison.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: '{file.filename}'. Use .csv, .xlsx, or .xls",
        )

    try:
        content = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    logger.info(f"[Upload] '{file.filename}' — {len(df)} rows × {len(df.columns)} cols")

    df = _normalize_columns(df)

    missing = [c for c in HACKATHON_INPUT_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Missing required columns: {missing}",
                "required": HACKATHON_INPUT_COLS,
                "received": list(df.columns),
                "tip": "Use: Granulation_Time, Binder_Amount, Drying_Temp, "
                       "Drying_Time, Compression_Force, Machine_Speed, Lubricant_Conc",
            },
        )

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="File contains no data rows.")

    warnings = []
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            out = df[(df[col] < lo) | (df[col] > hi)]
            if len(out) > 0:
                warnings.append(f"{col}: {len(out)} values outside [{lo}, {hi}]")

    df_feat = _engineer_features(df)

    # ── Load hackathon model and predict ─────────────────────
    predictions = []
    try:
        import joblib
        model_path = os.path.join(BACKEND_DIR, "models", "trained", "hackathon_model.pkl")
        scaler_path = os.path.join(BACKEND_DIR, "models", "trained", "hackathon_scaler.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Run: python3 train_hackathon.py",
            )

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        X = df_feat[ALL_FEATURE_COLS].values
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        for i in range(len(y_pred)):
            row_pred = {}
            for j, target in enumerate(HACKATHON_TARGET_COLS):
                row_pred[target] = round(float(y_pred[i][j]), 4)

            compliance = _check_quality_compliance(row_pred)
            batch_params = {col: float(df_feat.iloc[i][col]) for col in ALL_FEATURE_COLS}
            golden = _compare_to_golden(batch_params)

            batch_id = None
            if "Batch_ID" in df.columns:
                batch_id = str(df.iloc[i]["Batch_ID"])

            predictions.append({
                "batch_id": batch_id or f"Row_{i + 1}",
                "inputs": {col: round(float(df.iloc[i][col]), 4) for col in HACKATHON_INPUT_COLS if col in df.columns},
                "predicted_targets": row_pred,
                "quality_compliance": compliance,
                "golden_comparison": golden,
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Prediction failed: {e}", exc_info=True)
        warnings.append(f"Prediction failed: {str(e)}")

    # ── Summary ──────────────────────────────────────────────
    input_summary = {}
    for col in HACKATHON_INPUT_COLS:
        if col in df.columns:
            vals = df[col].dropna()
            input_summary[col] = {
                "min": round(float(vals.min()), 2),
                "max": round(float(vals.max()), 2),
                "mean": round(float(vals.mean()), 2),
            }

    pred_summary = {}
    if predictions:
        for target in HACKATHON_TARGET_COLS:
            vals = [p["predicted_targets"][target] for p in predictions]
            spec = QUALITY_SPECS.get(target, {})
            in_spec = sum(1 for v in vals if spec.get("min", -999) <= v <= spec.get("max", 999))
            pred_summary[target] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "in_spec_count": in_spec,
                "total": len(vals),
                "spec": f"{spec.get('min', '?')}–{spec.get('max', '?')} {spec.get('unit', '')}",
            }

    return {
        "status": "success",
        "filename": file.filename,
        "rows_uploaded": len(df),
        "columns_detected": list(df.columns),
        "input_summary": input_summary,
        "prediction_summary": pred_summary,
        "predictions": predictions,
        "warnings": warnings,
    }
