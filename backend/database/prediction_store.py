"""
PlantIQ — Prediction Store
==============================
Component 1.3 — Immutable prediction records with full provenance.

Rules:
  - Every /predict/batch call creates one PredictionRecord.
  - After status="closed", no field is modified.
  - Corrections create new records with `correction_of` pointing to the original.
  - Query by batch_id, date range, or status.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import PredictionRecord, AuditLog


def _generate_batch_id() -> str:
    """Generate a unique batch ID: BATCH_YYYYMMDD_HHMMSS_xxxx."""
    now = datetime.utcnow()
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"BATCH_{now.strftime('%Y%m%d_%H%M%S')}_{short_uuid}"


def save_prediction(
    db: Session,
    *,
    input_params: dict,
    predictions: dict,
    batch_id: Optional[str] = None,
    model_version: Optional[str] = None,
    confidence_intervals: Optional[dict] = None,
    confidence_score: Optional[float] = None,
    shap_breakdown: Optional[list] = None,
    shap_summary: Optional[str] = None,
    cost_translation: Optional[dict] = None,
    carbon_budget: Optional[dict] = None,
    distribution_warnings: Optional[list] = None,
    derived_features: Optional[dict] = None,
) -> PredictionRecord:
    """Persist a new prediction record and log it.

    Returns the created PredictionRecord ORM instance.
    """
    if batch_id is None:
        batch_id = _generate_batch_id()

    record = PredictionRecord(
        batch_id=batch_id,
        model_version=model_version,
        input_params=input_params,
        derived_features=derived_features,
        predictions=predictions,
        confidence_intervals=confidence_intervals,
        confidence_score=confidence_score,
        shap_breakdown=shap_breakdown,
        shap_summary=shap_summary,
        cost_translation=cost_translation,
        carbon_budget=carbon_budget,
        distribution_warnings=distribution_warnings,
        status="open",
    )
    db.add(record)

    # Audit trail
    audit = AuditLog(
        event_type="prediction_created",
        batch_id=batch_id,
        details={
            "model_version": model_version,
            "input_params": input_params,
            "predicted_quality": predictions.get("quality_score"),
            "predicted_energy": predictions.get("energy_kwh"),
        },
    )
    db.add(audit)

    db.commit()
    db.refresh(record)
    return record


def get_prediction(db: Session, batch_id: str) -> Optional[PredictionRecord]:
    """Retrieve a single prediction by batch_id."""
    return (
        db.query(PredictionRecord)
        .filter(PredictionRecord.batch_id == batch_id)
        .first()
    )


def list_predictions(
    db: Session,
    *,
    status: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[PredictionRecord]:
    """Query predictions with optional filters."""
    query = db.query(PredictionRecord)

    if status:
        query = query.filter(PredictionRecord.status == status)
    if since:
        query = query.filter(PredictionRecord.created_at >= since)
    if until:
        query = query.filter(PredictionRecord.created_at <= until)

    return (
        query.order_by(PredictionRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def close_prediction(db: Session, batch_id: str) -> PredictionRecord:
    """Mark a prediction as closed (immutable).

    After closure, no fields should be modified.
    Raises ValueError if already closed or not found.
    """
    record = get_prediction(db, batch_id)
    if record is None:
        raise ValueError(f"Prediction not found: {batch_id}")
    if record.status == "closed":
        raise ValueError(f"Prediction already closed: {batch_id}")

    record.status = "closed"

    audit = AuditLog(
        event_type="batch_closed",
        batch_id=batch_id,
        details={"previous_status": "open"},
    )
    db.add(audit)

    db.commit()
    db.refresh(record)
    return record


def count_predictions(
    db: Session,
    *,
    status: Optional[str] = None,
    since: Optional[datetime] = None,
) -> int:
    """Count prediction records with optional filters."""
    query = db.query(PredictionRecord)
    if status:
        query = query.filter(PredictionRecord.status == status)
    if since:
        query = query.filter(PredictionRecord.created_at >= since)
    return query.count()


def get_recent_predictions(
    db: Session, hours: int = 24, limit: int = 20
) -> List[PredictionRecord]:
    """Get predictions from the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return (
        db.query(PredictionRecord)
        .filter(PredictionRecord.created_at >= cutoff)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .all()
    )
