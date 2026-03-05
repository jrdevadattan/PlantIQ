"""
PlantIQ — Hackathon Data Routes
==================================
POST /hackathon/train               — Train XGBoost on real hackathon production data
POST /hackathon/predict             — Predict pharma quality targets from inputs
GET  /hackathon/production-data     — Load & inspect production data (60 batches)
GET  /hackathon/process-analysis    — Analyze time-series process data (phases, energy)
GET  /hackathon/energy-attribution  — Attribute energy patterns to asset vs. process
GET  /hackathon/quality-compliance  — Compute pharmaceutical quality compliance scores
GET  /hackathon/power-curve         — Extract power curve for LSTM anomaly detection

Provides full access to the real hackathon Excel data for validation
and demonstration of the PlantIQ pipeline on pharmaceutical tablet
manufacturing data.
"""

from __future__ import annotations

import os
import logging
import json
import joblib

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import HackathonTrainRequest, HackathonPredictRequest

logger = logging.getLogger("plantiq.hackathon")


def _sanitize_numpy(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
router = APIRouter(prefix="/hackathon", tags=["hackathon-data"])

# ── Paths ────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
HACKATHON_MODEL = os.path.join(ARTIFACT_DIR, "hackathon_model.pkl")
HACKATHON_SCALER = os.path.join(ARTIFACT_DIR, "hackathon_scaler.pkl")


# ──────────────────────────────────────────────────────────────
# POST /hackathon/train
# ──────────────────────────────────────────────────────────────
@router.post("/train")
async def train_hackathon_model(request: HackathonTrainRequest):
    """Train a multi-target XGBoost model on real hackathon production data.

    Uses the 60-batch pharmaceutical tablet dataset:
      Inputs:  Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time,
               Compression_Force, Machine_Speed, Lubricant_Conc
      Targets: Moisture_Content, Tablet_Weight, Hardness, Friability,
               Disintegration_Time, Dissolution_Rate, Content_Uniformity

    Saves:
      - hackathon_model.pkl (trained MultiOutputRegressor)
      - hackathon_scaler.pkl (fitted StandardScaler)
      - hackathon_evaluation.json (per-target accuracy metrics)
    """
    try:
        from data.hackathon_adapter import train_on_hackathon_data
        results = train_on_hackathon_data(verbose=request.verbose)
    except Exception as e:
        logger.error("Hackathon model training failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return {"status": "success", "evaluation": _sanitize_numpy(results)}


# ──────────────────────────────────────────────────────────────
# POST /hackathon/predict
# ──────────────────────────────────────────────────────────────
@router.post("/predict")
async def predict_hackathon(request: HackathonPredictRequest):
    """Predict pharmaceutical quality targets from 7 input parameters.

    Requires the hackathon model to be trained first (POST /hackathon/train).
    Returns all 7 quality predictions + compliance assessment.
    """
    if not os.path.exists(HACKATHON_MODEL):
        raise HTTPException(
            status_code=400,
            detail="Hackathon model not trained yet. Call POST /hackathon/train first.",
        )

    try:
        model = joblib.load(HACKATHON_MODEL)
        scaler = joblib.load(HACKATHON_SCALER)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load hackathon model: {str(e)}")

    # Build input vector
    from data.hackathon_adapter import HACKATHON_INPUT_COLS, HACKATHON_TARGET_COLS, QUALITY_SPECS

    raw_params = {
        "Granulation_Time": request.granulation_time,
        "Binder_Amount": request.binder_amount,
        "Drying_Temp": request.drying_temp,
        "Drying_Time": request.drying_time,
        "Compression_Force": request.compression_force,
        "Machine_Speed": request.machine_speed,
        "Lubricant_Conc": request.lubricant_conc,
    }

    # Engineer derived features
    from data.hackathon_adapter import HackathonDataAdapter
    adapter = HackathonDataAdapter()
    input_df = pd.DataFrame([raw_params])
    input_df = adapter.engineer_features(input_df)

    # Scale features (use all columns that the model was trained on)
    feature_cols = list(input_df.columns)
    X_scaled = scaler.transform(input_df[feature_cols])

    # Predict
    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Build output
    predictions = {}
    compliance = {}
    for i, target in enumerate(HACKATHON_TARGET_COLS):
        val = float(round(y_pred[0][i], 2))
        predictions[target] = val

        spec = QUALITY_SPECS.get(target, {})
        if spec:
            in_spec = spec["min"] <= val <= spec["max"]
            deviation = abs(val - spec["optimal"]) / spec["optimal"] * 100
            compliance[target] = {
                "value": val,
                "in_spec": in_spec,
                "spec_range": f"{spec['min']}–{spec['max']} {spec['unit']}",
                "deviation_from_optimal_pct": round(deviation, 1),
            }

    specs_met = sum(1 for c in compliance.values() if c["in_spec"])
    total_specs = len(compliance)

    return {
        "status": "success",
        "predictions": predictions,
        "compliance": compliance,
        "specs_met": specs_met,
        "total_specs": total_specs,
        "overall_quality_pct": round(specs_met / max(total_specs, 1) * 100, 1),
    }


# ──────────────────────────────────────────────────────────────
# GET /hackathon/production-data
# ──────────────────────────────────────────────────────────────
@router.get("/production-data")
async def get_production_data():
    """Load and return the hackathon production data (60 batches).

    Returns batch records with all 7 inputs and 7 quality outputs,
    plus engineered features and compliance scores.
    """
    try:
        from data.hackathon_adapter import HackathonDataAdapter
        adapter = HackathonDataAdapter()
        df = adapter.load_production_data()
        df = adapter.engineer_features(df)
        df = adapter.compute_quality_compliance(df)
    except Exception as e:
        logger.error("Failed to load production data: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data load failed: {str(e)}")

    # Convert to JSON-safe format
    records = df.to_dict(orient="records")
    # Convert numpy types
    for r in records:
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(round(v, 4))
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)

    return {
        "status": "success",
        "count": len(records),
        "batches": records,
    }


# ──────────────────────────────────────────────────────────────
# GET /hackathon/process-analysis
# ──────────────────────────────────────────────────────────────
@router.get("/process-analysis")
async def get_process_analysis():
    """Analyze time-series process data by manufacturing phase.

    Returns per-phase sensor statistics, energy breakdown,
    phase transitions, and anomaly indicators for batch T001.
    """
    try:
        from data.hackathon_adapter import analyze_hackathon_process_data
        result = analyze_hackathon_process_data(verbose=False)
    except Exception as e:
        logger.error("Process analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Process analysis failed: {str(e)}")

    return {"status": "success", **result}


# ──────────────────────────────────────────────────────────────
# GET /hackathon/energy-attribution
# ──────────────────────────────────────────────────────────────
@router.get("/energy-attribution")
async def get_energy_attribution():
    """Attribute energy pattern changes to asset vs. process factors.

    Energy Pattern Intelligence from the problem statement:
      "Methods to attribute energy pattern changes to specific
       equipment conditions or process deviations."

    Returns per-phase energy analysis, asset health indicators,
    process deviation indicators, and recommendations.
    """
    try:
        from data.hackathon_adapter import HackathonDataAdapter
        adapter = HackathonDataAdapter()
        process_df = adapter.load_process_data()
        attribution = adapter.attribute_energy_patterns(process_df)
    except Exception as e:
        logger.error("Energy attribution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Energy attribution failed: {str(e)}")

    return {"status": "success", **attribution}


# ──────────────────────────────────────────────────────────────
# GET /hackathon/quality-compliance
# ──────────────────────────────────────────────────────────────
@router.get("/quality-compliance")
async def get_quality_compliance():
    """Compute pharmaceutical quality compliance for all 60 batches.

    Checks each quality target against pharma specification limits:
      - Moisture_Content: 1.0–3.0%
      - Tablet_Weight: 198–202 mg
      - Hardness: 80–120 N
      - Friability: 0.0–1.0%
      - Disintegration_Time: 5–15 min
      - Dissolution_Rate: 85–100%
      - Content_Uniformity: 95–105%

    Returns per-batch compliance scores with individual spec pass/fail.
    """
    try:
        from data.hackathon_adapter import (
            HackathonDataAdapter,
            HACKATHON_TARGET_COLS,
            QUALITY_SPECS,
        )
        adapter = HackathonDataAdapter()
        df = adapter.load_production_data()
        df = adapter.compute_quality_compliance(df)
    except Exception as e:
        logger.error("Quality compliance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")

    # Summary stats
    total = len(df)
    fully_compliant = int((df["specs_met"] == df["total_specs"]).sum())

    # Per-target pass rate
    target_pass_rates = {}
    for t in HACKATHON_TARGET_COLS:
        col = f"{t}_in_spec"
        if col in df.columns:
            target_pass_rates[t] = {
                "pass_rate_pct": round(df[col].mean() * 100, 1),
                "spec": f"{QUALITY_SPECS[t]['min']}–{QUALITY_SPECS[t]['max']} {QUALITY_SPECS[t]['unit']}",
            }

    return {
        "status": "success",
        "total_batches": total,
        "fully_compliant": fully_compliant,
        "compliance_rate_pct": round(fully_compliant / max(total, 1) * 100, 1),
        "per_target_pass_rates": target_pass_rates,
        "mean_composite_quality": round(float(df["composite_quality_score"].mean()), 1),
    }


# ──────────────────────────────────────────────────────────────
# GET /hackathon/power-curve
# ──────────────────────────────────────────────────────────────
@router.get("/power-curve")
async def get_power_curve():
    """Extract power consumption curve from process data.

    Returns the Power_Consumption_kW time series for batch T001,
    suitable for LSTM Autoencoder anomaly detection.
    """
    try:
        from data.hackathon_adapter import HackathonDataAdapter
        adapter = HackathonDataAdapter()
        process_df = adapter.load_process_data()
        curve = adapter.get_power_curve(process_df)
    except Exception as e:
        logger.error("Power curve extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Power curve extraction failed: {str(e)}")

    return {
        "status": "success",
        "batch_id": "T001",
        "length": len(curve),
        "power_readings_kw": [round(float(v), 3) for v in curve],
        "total_energy_kwh": round(float(np.sum(curve) / 60), 2),
    }
