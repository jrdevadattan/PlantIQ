"""
PlantIQ — Feedback Loop Engine
==================================
Component 1.5 — Rolling accuracy tracking with drift detection.

Maintains a 30-day rolling MAPE for each of the five prediction targets.
If any target exceeds 10% MAPE, an alert fires.
If 10 consecutive batches exceed the threshold, a retrain flag is set.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from database.models import PredictionRecord, FeedbackMetric, AuditLog


TARGET_KEYS = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]

# Threshold: 10% MAPE triggers a drift alert
MAPE_THRESHOLD = 10.0
# Sustained degradation: 10 consecutive batches above threshold → retrain
CONSECUTIVE_THRESHOLD = 10


def compute_rolling_metrics(
    db: Session,
    *,
    window_days: int = 30,
    max_batches: int = 200,
) -> List[FeedbackMetric]:
    """Compute rolling MAPE for each target from recent closed predictions.

    Returns list of FeedbackMetric instances (one per target).
    Persists results to the feedback_metrics table.
    """
    cutoff = datetime.utcnow() - timedelta(days=window_days)

    closed = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.status == "closed",
            PredictionRecord.actual_outcomes.isnot(None),
            PredictionRecord.outcome_recorded_at >= cutoff,
        )
        .order_by(PredictionRecord.outcome_recorded_at.desc())
        .limit(max_batches)
        .all()
    )

    if not closed:
        return []

    metrics_list: List[FeedbackMetric] = []

    for target in TARGET_KEYS:
        errors: List[float] = []
        abs_errors: List[float] = []

        for record in closed:
            pred_errors = record.prediction_errors or {}
            target_error = pred_errors.get(target, {})
            pct = target_error.get("percentage_error")
            abs_err = target_error.get("absolute_error")
            if pct is not None:
                errors.append(pct)
            if abs_err is not None:
                abs_errors.append(abs_err)

        if not errors:
            continue

        rolling_mape = sum(errors) / len(errors)
        rolling_mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

        # Count consecutive degraded batches (most recent first)
        consecutive = 0
        for pct in errors:
            if pct > MAPE_THRESHOLD:
                consecutive += 1
            else:
                break

        drift_detected = rolling_mape > MAPE_THRESHOLD
        retrain_flag = consecutive >= CONSECUTIVE_THRESHOLD

        metric = FeedbackMetric(
            target_name=target,
            rolling_mape=rolling_mape,
            rolling_mae=rolling_mae,
            batch_count=len(errors),
            drift_detected=drift_detected,
            consecutive_degraded=consecutive,
            retrain_flag=retrain_flag,
        )
        db.add(metric)
        metrics_list.append(metric)

        # Log drift detection events
        if drift_detected:
            audit = AuditLog(
                event_type="drift_detected",
                details={
                    "target": target,
                    "rolling_mape": round(rolling_mape, 2),
                    "consecutive_degraded": consecutive,
                    "retrain_flag": retrain_flag,
                    "batch_count": len(errors),
                },
            )
            db.add(audit)

    db.commit()
    for m in metrics_list:
        db.refresh(m)

    return metrics_list


def get_latest_metrics(db: Session) -> Dict[str, dict]:
    """Get the most recent feedback metric for each target.

    Returns dict keyed by target_name → metric dict.
    """
    result = {}
    for target in TARGET_KEYS:
        metric = (
            db.query(FeedbackMetric)
            .filter(FeedbackMetric.target_name == target)
            .order_by(FeedbackMetric.computed_at.desc())
            .first()
        )
        if metric:
            result[target] = metric.to_dict()
    return result


def get_metric_history(
    db: Session,
    target_name: str,
    *,
    limit: int = 30,
) -> List[dict]:
    """Get historical MAPE values for a specific target (for charting)."""
    metrics = (
        db.query(FeedbackMetric)
        .filter(FeedbackMetric.target_name == target_name)
        .order_by(FeedbackMetric.computed_at.desc())
        .limit(limit)
        .all()
    )
    return [m.to_dict() for m in reversed(metrics)]


def check_drift_status(db: Session) -> Dict[str, dict]:
    """Quick drift status check for all targets.

    Returns dict with overall status and per-target details.
    """
    latest = get_latest_metrics(db)
    any_drift = any(m.get("drift_detected") for m in latest.values())
    any_retrain = any(m.get("retrain_flag") for m in latest.values())

    return {
        "overall_status": "critical" if any_retrain else ("warning" if any_drift else "healthy"),
        "any_drift_detected": any_drift,
        "any_retrain_needed": any_retrain,
        "targets": latest,
    }
