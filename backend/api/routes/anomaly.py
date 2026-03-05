"""
PlantIQ — Anomaly Detection Route
====================================
POST /anomaly/detect — Analyze power curve segment for anomalies.

Note: The LSTM Autoencoder (F2.1) and Fault Classifier (F2.2) are Tier 2
features not yet implemented.  This route provides a statistical mock
implementation that produces correctly-shaped responses matching the
README API spec, to be replaced when models are built.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    DiagnosisInfo,
)

router = APIRouter(tags=["anomaly"])

# Anomaly thresholds per README spec
ANOMALY_THRESHOLD = 0.30

# Severity mapping
def _severity(score: float) -> str:
    """Map anomaly score to severity level per README spec."""
    if score < 0.15:
        return "NORMAL"
    elif score < 0.30:
        return "WATCH"
    elif score < 0.60:
        return "WARNING"
    else:
        return "CRITICAL"


# Fault diagnosis lookup
FAULT_ACTIONS = {
    "normal": {
        "human_readable": "Power curve is within normal operating parameters",
        "recommended_action": "No action required",
        "energy_impact": 0.0,
        "quality_impact": 0.0,
    },
    "bearing_wear": {
        "human_readable": "Power curve shows gradual baseline rise consistent with bearing degradation",
        "recommended_action": "Schedule maintenance within 5 days",
        "energy_impact": 2.1,
        "quality_impact": -3.2,
    },
    "wet_material": {
        "human_readable": "Power curve shows irregular spikes consistent with high raw material moisture content",
        "recommended_action": "Extend drying phase by 4 minutes",
        "energy_impact": 1.8,
        "quality_impact": -8.4,
    },
    "calibration_needed": {
        "human_readable": "Power curve shows elevated flat baseline indicating machine calibration drift",
        "recommended_action": "Machine calibration required",
        "energy_impact": 3.5,
        "quality_impact": -5.1,
    },
}


def _statistical_anomaly_detect(readings: list[float]) -> tuple[float, str, float]:
    """Statistical anomaly detection (placeholder for LSTM Autoencoder).

    Uses simple statistical features to estimate anomaly score and
    classify fault type.  Will be replaced by LSTM + RandomForest
    when Tier 2 features are built.

    Returns
    -------
    tuple[float, str, float]
        (anomaly_score, fault_type, fault_confidence)
    """
    arr = np.array(readings, dtype=np.float64)

    if len(arr) < 2:
        return 0.0, "normal", 0.99

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    max_val = float(np.max(arr))

    # Trend slope (linear fit)
    t = np.arange(len(arr))
    slope = float(np.polyfit(t, arr, 1)[0]) if len(arr) > 1 else 0.0

    # Spike count — sharp changes
    diffs = np.abs(np.diff(arr))
    spike_count = int(np.sum(diffs > 0.5))

    # Coefficient of variation
    cv = std_val / max(mean_val, 0.01)

    # ── Scoring heuristics ──────────────────────────
    score = 0.0
    fault_type = "normal"
    confidence = 0.85

    # Bearing wear: gradual rising baseline
    if slope > 0.003:
        score += min(slope * 100, 0.5)
        fault_type = "bearing_wear"
        confidence = min(0.6 + slope * 50, 0.95)

    # Wet material: lots of spikes in first half
    spike_ratio = spike_count / max(len(arr), 1)
    if spike_ratio > 0.1:
        score += min(spike_ratio * 3, 0.5)
        if fault_type == "normal" or spike_ratio > 0.2:
            fault_type = "wet_material"
            confidence = min(0.6 + spike_ratio * 2, 0.95)

    # Calibration needed: elevated flat baseline
    if mean_val > 5.5 and cv < 0.1:
        score += min((mean_val - 5.0) * 0.15, 0.4)
        fault_type = "calibration_needed"
        confidence = min(0.7 + (mean_val - 5.0) * 0.05, 0.95)

    # General anomaly from high variance
    if cv > 0.3:
        score += min(cv * 0.3, 0.2)

    # Clamp to [0, 1]
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.15:
        fault_type = "normal"
        confidence = 0.95

    return round(score, 4), fault_type, round(confidence, 2)


@router.post("/anomaly/detect", response_model=AnomalyDetectResponse)
async def detect_anomaly(request: AnomalyDetectRequest) -> AnomalyDetectResponse:
    """Analyze a power curve segment for anomalies and diagnose fault type.

    Currently uses statistical heuristics.  Will be upgraded to
    LSTM Autoencoder + RandomForest fault classifier in Tier 2.
    """
    if not request.power_readings:
        raise HTTPException(status_code=400, detail="power_readings must not be empty")

    anomaly_score, fault_type, fault_confidence = _statistical_anomaly_detect(
        request.power_readings
    )

    is_anomaly = anomaly_score >= ANOMALY_THRESHOLD
    severity = _severity(anomaly_score)

    fault_info = FAULT_ACTIONS.get(fault_type, FAULT_ACTIONS["normal"])

    diagnosis = DiagnosisInfo(
        fault_type=fault_type,
        confidence=fault_confidence,
        human_readable=fault_info["human_readable"],
        recommended_action=fault_info["recommended_action"],
        estimated_energy_impact_kwh=fault_info["energy_impact"],
        estimated_quality_impact_pct=fault_info["quality_impact"],
    )

    return AnomalyDetectResponse(
        anomaly_score=anomaly_score,
        threshold=ANOMALY_THRESHOLD,
        is_anomaly=is_anomaly,
        severity=severity,
        diagnosis=diagnosis,
    )
