"""
PlantIQ — Outcome Recorder
==============================
Component 1.3 (QC outcome fields) — Record actual QC measurements
and compute prediction errors for closed batches.

When a batch finishes, the QC team enters actual quality, yield,
performance, and energy readings.  The recorder:
  1. Stores actual values on the PredictionRecord
  2. Computes per-target prediction errors (absolute + percentage)
  3. Marks the record as closed
  4. Triggers the FeedbackLoop to update rolling MAPE
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from sqlalchemy.orm import Session

from database.models import PredictionRecord, AuditLog


TARGET_KEYS = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]


def record_outcome(
    db: Session,
    batch_id: str,
    actual_quality: float,
    actual_yield: float,
    actual_performance: float,
    actual_energy: float,
    recorded_by: str = "qc_operator",
) -> PredictionRecord:
    """Record actual QC results for a batch and compute prediction errors.

    Raises ValueError if batch not found or already has outcomes recorded.
    Returns the updated PredictionRecord.
    """
    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.batch_id == batch_id)
        .first()
    )
    if record is None:
        raise ValueError(f"Prediction not found: {batch_id}")
    if record.actual_outcomes is not None:
        raise ValueError(f"Outcomes already recorded for: {batch_id}")

    # ── Store actual outcomes ────────────────────────
    actuals = {
        "quality_score": actual_quality,
        "yield_pct": actual_yield,
        "performance_pct": actual_performance,
        "energy_kwh": actual_energy,
    }
    record.actual_outcomes = actuals

    # ── Compute prediction errors ────────────────────
    predictions = record.predictions or {}
    errors = {}
    for key in TARGET_KEYS:
        predicted = predictions.get(key)
        actual = actuals.get(key)
        if predicted is not None and actual is not None:
            abs_error = abs(predicted - actual)
            pct_error = (abs_error / actual * 100) if actual != 0 else 0.0
            errors[key] = {
                "predicted": round(predicted, 4),
                "actual": round(actual, 4),
                "absolute_error": round(abs_error, 4),
                "percentage_error": round(pct_error, 2),
            }
    record.prediction_errors = errors

    # ── Timestamps + close ───────────────────────────
    record.outcome_recorded_at = datetime.utcnow()
    record.outcome_recorded_by = recorded_by
    record.status = "closed"

    # ── Audit ────────────────────────────────────────
    audit = AuditLog(
        event_type="outcome_recorded",
        batch_id=batch_id,
        actor=recorded_by,
        details={
            "actuals": actuals,
            "errors_summary": {
                k: v["percentage_error"]
                for k, v in errors.items()
            },
        },
    )
    db.add(audit)

    db.commit()
    db.refresh(record)
    return record


def get_closed_predictions_with_errors(
    db: Session,
    *,
    limit: int = 100,
    since: Optional[datetime] = None,
) -> list:
    """Retrieve closed predictions that have outcomes recorded.

    Useful for the FeedbackLoop to compute rolling MAPE.
    """
    query = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.status == "closed",
            PredictionRecord.actual_outcomes.isnot(None),
        )
    )
    if since:
        query = query.filter(PredictionRecord.outcome_recorded_at >= since)

    return (
        query.order_by(PredictionRecord.outcome_recorded_at.desc())
        .limit(limit)
        .all()
    )


def compute_error_summary(record: PredictionRecord) -> Dict[str, float]:
    """Compute a flat error summary dict from a closed prediction."""
    errors = record.prediction_errors or {}
    return {
        key: errors.get(key, {}).get("percentage_error", 0.0)
        for key in TARGET_KEYS
    }
