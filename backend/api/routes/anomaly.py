"""
PlantIQ — Anomaly Detection Route
====================================
POST /anomaly/detect — Analyze power curve segment for anomalies.

Uses the LSTM Autoencoder (F2.1) when the trained model is available,
falling back to statistical heuristics if the LSTM model hasn't been
trained yet.  Fault diagnosis uses pattern-matching on the anomaly
features.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    DiagnosisInfo,
)

logger = logging.getLogger("plantiq.anomaly")
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

    Uses the LSTM Autoencoder when available for reconstruction-based anomaly
    scoring. Falls back to statistical heuristics if LSTM model not loaded.
    """
    if not request.power_readings:
        raise HTTPException(status_code=400, detail="power_readings must not be empty")

    # Try LSTM first, fall back to statistical method
    used_lstm = False
    try:
        from main import lstm_model
        if lstm_model is not None:
            from models.lstm_autoencoder import compute_anomaly_score
            lstm_result = compute_anomaly_score(
                model=lstm_model["model"],
                curve=np.array(request.power_readings, dtype=np.float32),
                threshold=lstm_model["threshold"],
                normal_mean=lstm_model["normal_mean"],
                normal_std=lstm_model["normal_std"],
            )
            anomaly_score = lstm_result["anomaly_score"]
            used_lstm = True
            logger.info(
                "LSTM anomaly score for %s: %.4f (threshold=%.4f)",
                request.batch_id, anomaly_score, lstm_result["threshold"],
            )

            # Use trained RandomForest classifier if available, else heuristic
            fault_type, fault_confidence = _classify_fault_ml(
                request.power_readings, anomaly_score
            )
        else:
            anomaly_score, fault_type, fault_confidence = _statistical_anomaly_detect(
                request.power_readings
            )
    except ImportError:
        anomaly_score, fault_type, fault_confidence = _statistical_anomaly_detect(
            request.power_readings
        )
    except Exception as e:
        logger.warning("LSTM detection failed, using statistical fallback: %s", e)
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


def _classify_fault_ml(
    readings: list[float], anomaly_score: float
) -> tuple[str, float]:
    """Classify fault type using the trained RandomForest classifier (F2.2).

    Falls back to the rule-based heuristic if the ML classifier is
    not loaded.

    Returns
    -------
    tuple[str, float]
        (fault_type, confidence)
    """
    # Low anomaly score → skip classification, always normal
    if anomaly_score < 0.15:
        return "normal", 0.95

    # Try trained RandomForest first
    try:
        from main import fault_classifier
        if fault_classifier is not None and fault_classifier.is_loaded:
            curve_arr = np.array(readings, dtype=np.float64)
            fault_type, confidence = fault_classifier.classify(curve_arr)
            logger.info(
                "RF fault classification: %s (confidence=%.4f)",
                fault_type, confidence,
            )
            return fault_type, confidence
    except ImportError:
        pass
    except Exception as e:
        logger.warning("RF classifier failed, using heuristic: %s", e)

    # Fallback: rule-based heuristic
    return _classify_fault_heuristic(readings, anomaly_score)


def _classify_fault_heuristic(
    readings: list[float], anomaly_score: float
) -> tuple[str, float]:
    """Rule-based fault classification fallback.

    Used when the trained RandomForest classifier is not available.

    Returns
    -------
    tuple[str, float]
        (fault_type, confidence)
    """
    arr = np.array(readings, dtype=np.float64)

    if anomaly_score < 0.15:
        return "normal", 0.95

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    # Trend slope
    t = np.arange(len(arr))
    slope = float(np.polyfit(t, arr, 1)[0]) if len(arr) > 1 else 0.0

    # Spike count
    diffs = np.abs(np.diff(arr))
    spike_ratio = float(np.sum(diffs > 0.5)) / max(len(arr), 1)

    # Coefficient of variation
    cv = std_val / max(mean_val, 0.01)

    # Decision logic
    if slope > 0.003:
        return "bearing_wear", min(0.7 + anomaly_score * 0.3, 0.95)
    elif spike_ratio > 0.1:
        return "wet_material", min(0.65 + spike_ratio * 2, 0.95)
    elif mean_val > 5.5 and cv < 0.1:
        return "calibration_needed", min(0.7 + (mean_val - 5.0) * 0.05, 0.95)
    else:
        return "bearing_wear" if slope > 0.001 else "calibration_needed", 0.6
