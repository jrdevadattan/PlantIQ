"""
PlantIQ — Prediction Routes
==============================
POST /predict/batch     — Pre-batch prediction (all 4 targets)
POST /predict/realtime  — Mid-batch updated prediction with sliding window

Both endpoints return predictions + confidence intervals + carbon budget.
"""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionValues,
    ConfidenceInterval,
    CarbonBudget,
    RealtimePredictionRequest,
    RealtimePredictionResponse,
    AlertInfo,
)

router = APIRouter(prefix="/predict", tags=["predictions"])

# Carbon budget constant (per README: energy × 0.82 CO₂ factor)
CO2_FACTOR = 0.82
DAILY_CO2_BUDGET_KG = 4200.0  # Total daily budget
BATCHES_PER_DAY = 100         # Approximate batches per day
BATCH_CO2_BUDGET = DAILY_CO2_BUDGET_KG / BATCHES_PER_DAY  # 42.0 kg per batch

# Confidence interval width (% of prediction — conservative estimate)
CI_PCT = 0.07  # ±7% for energy, tighter for others


def _generate_batch_id() -> str:
    """Generate a unique batch ID with timestamp."""
    return f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _get_predictor():
    """Lazy-import the global predictor instance from main."""
    from main import predictor
    if predictor is None or not predictor.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is starting up.")
    return predictor


def _compute_carbon_budget(energy_kwh: float) -> CarbonBudget:
    """Compute carbon budget status for a predicted energy value."""
    co2 = energy_kwh * CO2_FACTOR
    headroom = BATCH_CO2_BUDGET - co2

    if co2 <= BATCH_CO2_BUDGET * 0.8:
        status = "ON_TRACK"
    elif co2 <= BATCH_CO2_BUDGET:
        status = "WARNING"
    else:
        status = "OVER_BUDGET"

    return CarbonBudget(
        batch_budget_kg=BATCH_CO2_BUDGET,
        predicted_usage_kg=round(co2, 1),
        status=status,
        headroom_kg=round(headroom, 1),
    )


def _compute_confidence_intervals(predictions: dict) -> dict[str, ConfidenceInterval]:
    """Compute confidence intervals for each target prediction."""
    intervals = {}

    # Different CI widths per target (energy is harder to predict precisely)
    ci_widths = {
        "energy_kwh": 0.07,
        "quality_score": 0.02,
        "yield_pct": 0.02,
        "performance_pct": 0.03,
    }

    for target in ["energy_kwh", "quality_score", "yield_pct", "performance_pct"]:
        val = predictions.get(target, 0)
        width = ci_widths.get(target, 0.05)
        intervals[target] = ConfidenceInterval(
            lower=round(val * (1 - width), 1),
            upper=round(val * (1 + width), 1),
        )

    return intervals


# ──────────────────────────────────────────────────────────────
# POST /predict/batch
# ──────────────────────────────────────────────────────────────
@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict all 4 targets from batch setup parameters.

    Call this before or at the start of a batch.  Returns predictions,
    confidence intervals, and carbon budget status.
    """
    predictor = _get_predictor()

    # Convert request to dict for predictor
    params = request.model_dump()

    try:
        raw_preds = predictor.predict_single(params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Build response
    predictions = PredictionValues(
        quality_score=raw_preds["quality_score"],
        yield_pct=raw_preds["yield_pct"],
        performance_pct=raw_preds["performance_pct"],
        energy_kwh=raw_preds["energy_kwh"],
        co2_kg=raw_preds.get("co2_kg", raw_preds["energy_kwh"] * CO2_FACTOR),
    )

    return BatchPredictionResponse(
        batch_id=_generate_batch_id(),
        predictions=predictions,
        confidence_intervals=_compute_confidence_intervals(raw_preds),
        carbon_budget=_compute_carbon_budget(raw_preds["energy_kwh"]),
    )


# ──────────────────────────────────────────────────────────────
# POST /predict/realtime
# ──────────────────────────────────────────────────────────────
@router.post("/realtime", response_model=RealtimePredictionResponse)
async def predict_realtime(request: RealtimePredictionRequest) -> RealtimePredictionResponse:
    """Update prediction mid-batch using actual data collected so far.

    Uses a sliding window blend: trusts model more at start,
    trusts actual consumption rate more as the batch progresses.
    Per README Component 4 — Sliding Window Real-Time Forecaster.
    """
    predictor = _get_predictor()

    original = request.original_params.model_dump()
    partial = request.partial_data

    # Get baseline model prediction
    try:
        model_preds = predictor.predict_single(original)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Compute progress
    total_minutes = original["hold_time"]
    progress_pct = round(min(partial.elapsed_minutes / total_minutes * 100, 100.0), 1)

    # Sliding window blend (per README spec)
    progress_frac = partial.elapsed_minutes / total_minutes
    blend_weight = min(progress_frac * 2, 0.8)

    # Extrapolate energy from actual consumption rate
    if partial.elapsed_minutes > 0:
        rate_kwh_per_min = partial.energy_so_far / partial.elapsed_minutes
        extrapolated_energy = rate_kwh_per_min * total_minutes
    else:
        extrapolated_energy = model_preds["energy_kwh"]

    # Blended energy prediction
    adjusted_energy = (1 - blend_weight) * model_preds["energy_kwh"] + blend_weight * extrapolated_energy
    adjusted_energy = round(adjusted_energy, 1)

    # Adjust other targets proportionally based on energy deviation
    energy_ratio = adjusted_energy / max(model_preds["energy_kwh"], 0.1)
    # Higher energy → slightly lower quality/yield/performance
    quality_adj = model_preds["quality_score"] * (2 - energy_ratio) if energy_ratio > 1 else model_preds["quality_score"]
    yield_adj = model_preds["yield_pct"] * (2 - energy_ratio) if energy_ratio > 1 else model_preds["yield_pct"]
    perf_adj = model_preds["performance_pct"] * (2 - energy_ratio) if energy_ratio > 1 else model_preds["performance_pct"]

    co2_adjusted = round(adjusted_energy * CO2_FACTOR, 1)

    updated = PredictionValues(
        quality_score=round(quality_adj, 1),
        yield_pct=round(yield_adj, 1),
        performance_pct=round(perf_adj, 1),
        energy_kwh=adjusted_energy,
        co2_kg=co2_adjusted,
    )

    # Confidence shrinks as we get more data
    ci_energy = abs(adjusted_energy - model_preds["energy_kwh"])
    ci_str = f"±{max(ci_energy * 0.5, 0.5):.1f} kWh"

    # Alert logic
    alert = None
    deviation_pct = ((adjusted_energy - model_preds["energy_kwh"]) / max(model_preds["energy_kwh"], 0.1)) * 100

    if deviation_pct > 15:
        alert = AlertInfo(
            severity="WARNING",
            message=f"Energy trending {deviation_pct:.1f}% above target",
            recommended_action=f"Reduce conveyor speed from {original['conveyor_speed']:.0f}% to {max(original['conveyor_speed'] - 6, 60):.0f}%",
            estimated_saving_kwh=round(adjusted_energy * 0.05, 1),
            quality_impact_pct=-0.3,
        )
    elif deviation_pct > 5:
        alert = AlertInfo(
            severity="WATCH",
            message=f"Energy trending {deviation_pct:.1f}% above target",
        )

    return RealtimePredictionResponse(
        progress_pct=progress_pct,
        updated_predictions=updated,
        confidence=ci_str,
        alert=alert,
    )
