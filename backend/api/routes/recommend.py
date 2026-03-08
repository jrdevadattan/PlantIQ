"""
PlantIQ — Recommendation & Decision Engine API Routes
========================================================
POST /recommend/generate     — Generate machine-level recommendations from SHAP
POST /recommend/validate     — Validate batch input parameters (4-gate)
POST /recommend/confidence   — Compute multi-factor confidence score
POST /recommend/alerts       — Check predictions + anomaly for alerts
"""

from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/recommend", tags=["decision-engine"])


# ══════════════════════════════════════════════════════════════
# Request / Response Schemas
# ══════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    """POST /recommend/generate request."""
    input_params: dict = Field(..., description="Batch input parameters (temperature, conveyor_speed, etc.)")
    shap_contributions: list[dict] = Field(
        ...,
        description="SHAP contributions list — [{feature, contribution, direction}, ...]",
    )
    target: str = Field(default="energy_kwh", description="Optimisation target")
    batch_id: Optional[str] = Field(default=None, description="Batch ID for tracking")
    max_recommendations: int = Field(default=3, ge=1, le=7, description="Max recommendations to return")


class ValidateRequest(BaseModel):
    """POST /recommend/validate request."""
    temperature: Optional[float] = None
    conveyor_speed: Optional[float] = None
    hold_time: Optional[float] = None
    batch_size: Optional[float] = None
    material_type: Optional[int] = None
    hour_of_day: Optional[int] = None
    operator_exp: Optional[int] = None


class ConfidenceRequest(BaseModel):
    """POST /recommend/confidence request."""
    ood_fields: list[str] = Field(default_factory=list, description="Fields outside training distribution")
    drift_detected: bool = Field(default=False, description="Whether model drift has been flagged")
    feature_issue: bool = Field(default=False, description="Whether feature computation failed")


class AlertCheckRequest(BaseModel):
    """POST /recommend/alerts request."""
    batch_id: str = Field(..., description="Batch identifier")
    predictions: dict = Field(..., description="Predicted values (energy_kwh, quality_score, etc.)")
    energy_target_kwh: float = Field(default=42.0, description="Energy target for the batch")
    anomaly_score: Optional[float] = Field(default=None, description="Anomaly score (if available)")
    fault_type: Optional[str] = Field(default=None, description="Fault type from classifier")
    fault_detail: Optional[str] = Field(default=None, description="Fault detail description")
    shap_top_feature: Optional[dict] = Field(default=None, description="Top SHAP contributor")
    recommendation: Optional[str] = Field(default=None, description="Recommended action text")


# ══════════════════════════════════════════════════════════════
# POST /recommend/generate
# ══════════════════════════════════════════════════════════════

@router.post("/generate")
async def generate_recommendations(request: RecommendRequest):
    """Generate ranked, machine-level recommendations from SHAP contributions.

    Returns a RecommendationSet with up to `max_recommendations` items,
    each containing the exact physical control to adjust, the direction,
    estimated savings, and a plain-English instruction string.
    """
    from decision_engine.recommendation_engine import RecommendationEngine

    engine = RecommendationEngine()

    try:
        rec_set = engine.generate(
            input_params=request.input_params,
            shap_contributions=request.shap_contributions,
            target=request.target,
            batch_id=request.batch_id,
            max_recommendations=request.max_recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

    return {
        "batch_id": rec_set.batch_id,
        "target_focus": rec_set.target_focus,
        "total_estimated_saving_kwh": rec_set.total_estimated_saving_kwh,
        "summary": rec_set.summary,
        "recommendations": [
            {
                "rank": r.rank,
                "parameter": r.parameter,
                "current_value": r.current_value,
                "recommended_value": r.recommended_value,
                "adjustment": r.adjustment,
                "direction": r.direction,
                "machine": r.machine,
                "control": r.control,
                "instruction": r.instruction,
                "estimated_energy_saving_kwh": r.estimated_energy_saving_kwh,
                "estimated_quality_impact_pct": r.estimated_quality_impact_pct,
                "estimated_yield_impact_pct": r.estimated_yield_impact_pct,
                "response_time_min": r.response_time_min,
                "shap_contribution": r.shap_contribution,
                "safety_note": r.safety_note,
            }
            for r in rec_set.recommendations
        ],
    }


# ══════════════════════════════════════════════════════════════
# POST /recommend/validate
# ══════════════════════════════════════════════════════════════

@router.post("/validate")
async def validate_inputs(request: ValidateRequest):
    """Validate batch input parameters through 4 sequential gates.

    Gates:
      1. Presence — all 7 fields must be non-null
      2. Type — all values must be numeric
      3. Physical — within machine operating range (hard stop)
      4. Training — within model training distribution (soft warning)

    Returns validation result with errors and/or warnings.
    """
    from decision_engine.input_validator import InputValidator

    validator = InputValidator()
    params = request.model_dump()

    result = validator.validate(params)
    return result.to_dict()


# ══════════════════════════════════════════════════════════════
# POST /recommend/confidence
# ══════════════════════════════════════════════════════════════

@router.post("/confidence")
async def compute_confidence(request: ConfidenceRequest):
    """Compute multi-factor confidence score for a prediction.

    Starts at base 0.95 and applies penalties:
      - OOD fields: −35% (scaled by count)
      - Drift detected: −10%
      - Feature issue: −15%

    Returns score (0–1), indicator (green/amber/red), and penalty details.
    """
    from decision_engine.confidence_scorer import ConfidenceScorer

    scorer = ConfidenceScorer()

    report = scorer.compute(
        ood_fields=request.ood_fields if request.ood_fields else None,
        drift_detected=request.drift_detected,
        feature_issue=request.feature_issue,
    )

    return report.to_dict()


# ══════════════════════════════════════════════════════════════
# POST /recommend/alerts
# ══════════════════════════════════════════════════════════════

@router.post("/alerts")
async def check_alerts(request: AlertCheckRequest):
    """Check predictions and anomaly scores against thresholds.

    Generates structured alert records for:
      - Energy overrun (>15% WARNING, >25% CRITICAL)
      - Quality risk (<80% WARNING, <70% CRITICAL)
      - Anomaly score (>0.15 WATCH, >0.30 WARNING, >0.60 CRITICAL)

    Returns list of triggered alerts (may be empty).
    """
    from decision_engine.alert_engine import AlertEngine

    engine = AlertEngine(energy_target_kwh=request.energy_target_kwh)

    alerts = []

    # Check predictions
    pred_alerts = engine.check_predictions(
        batch_id=request.batch_id,
        predictions=request.predictions,
        energy_target_kwh=request.energy_target_kwh,
        shap_top_feature=request.shap_top_feature,
        recommendation=request.recommendation,
    )
    alerts.extend(pred_alerts)

    # Check anomaly if score provided
    if request.anomaly_score is not None:
        anomaly_alerts = engine.check_anomaly(
            batch_id=request.batch_id,
            anomaly_score=request.anomaly_score,
            fault_type=request.fault_type,
            fault_detail=request.fault_detail,
            recommendation=request.recommendation,
        )
        alerts.extend(anomaly_alerts)

    return {
        "batch_id": request.batch_id,
        "alert_count": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "batch_id": a.batch_id,
                "timestamp": a.timestamp,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "technical_detail": a.technical_detail,
                "root_cause": a.root_cause,
                "recommended_action": a.recommended_action,
                "estimated_saving_kwh": a.estimated_saving_kwh,
                "quality_impact_pct": a.quality_impact_pct,
            }
            for a in alerts
        ],
    }
