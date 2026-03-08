"""
PlantIQ — Audit Store
========================
Component 10 — Complete batch audit queries.

Per README §10 — Audit Architecture:
  "One question is the acid test: 'What exactly happened with Batch
   B-20250613-0042?' If the system can answer that question completely
   from stored records alone, the audit architecture is sufficient."

This module provides that single-query capability by aggregating data
across PredictionRecord, AlertRecord, and AuditLog tables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import PredictionRecord, AlertRecord, AuditLog


def get_complete_batch_audit(db: Session, batch_id: str) -> Optional[dict]:
    """Answer: 'What exactly happened with Batch <batch_id>?'

    Returns a comprehensive audit record containing:
      - Prediction details (inputs, outputs, model version, SHAP)
      - Cost translation
      - Actual outcomes and prediction errors
      - All alerts and their lifecycle states
      - Complete event timeline from the audit log
    """
    # ── Prediction record ────────────────────────────
    prediction = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.batch_id == batch_id)
        .first()
    )
    if prediction is None:
        return None

    # ── Alerts ───────────────────────────────────────
    alerts = (
        db.query(AlertRecord)
        .filter(AlertRecord.batch_id == batch_id)
        .order_by(AlertRecord.fired_at)
        .all()
    )

    # ── Audit events ─────────────────────────────────
    events = (
        db.query(AuditLog)
        .filter(AuditLog.batch_id == batch_id)
        .order_by(AuditLog.timestamp)
        .all()
    )

    # ── Compose the complete audit ───────────────────
    return {
        "batch_id": batch_id,
        "status": prediction.status,
        "created_at": prediction.created_at.isoformat() if prediction.created_at else None,

        "prediction": {
            "model_version": prediction.model_version,
            "input_params": prediction.input_params,
            "derived_features": prediction.derived_features,
            "predictions": prediction.predictions,
            "confidence_intervals": prediction.confidence_intervals,
            "confidence_score": prediction.confidence_score,
            "shap_breakdown": prediction.shap_breakdown,
            "shap_summary": prediction.shap_summary,
            "cost_translation": prediction.cost_translation,
            "carbon_budget": prediction.carbon_budget,
            "distribution_warnings": prediction.distribution_warnings,
        },

        "outcomes": {
            "recorded": prediction.actual_outcomes is not None,
            "actual_values": prediction.actual_outcomes,
            "prediction_errors": prediction.prediction_errors,
            "recorded_at": (
                prediction.outcome_recorded_at.isoformat()
                if prediction.outcome_recorded_at else None
            ),
            "recorded_by": prediction.outcome_recorded_by,
        },

        "alerts": {
            "count": len(alerts),
            "records": [a.to_dict() for a in alerts],
        },

        "timeline": [e.to_dict() for e in events],
    }


def get_audit_events(
    db: Session,
    *,
    event_type: Optional[str] = None,
    batch_id: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    actor: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """Query audit events with flexible filters."""
    query = db.query(AuditLog)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)
    if batch_id:
        query = query.filter(AuditLog.batch_id == batch_id)
    if since:
        query = query.filter(AuditLog.timestamp >= since)
    if until:
        query = query.filter(AuditLog.timestamp <= until)
    if actor:
        query = query.filter(AuditLog.actor == actor)

    events = (
        query.order_by(AuditLog.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [e.to_dict() for e in events]


def get_audit_summary(db: Session) -> dict:
    """Get high-level audit statistics."""
    total_predictions = db.query(PredictionRecord).count()
    open_predictions = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.status == "open")
        .count()
    )
    closed_predictions = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.status == "closed")
        .count()
    )
    total_alerts = db.query(AlertRecord).count()
    active_alerts = (
        db.query(AlertRecord)
        .filter(AlertRecord.state != "resolved")
        .count()
    )
    total_events = db.query(AuditLog).count()

    return {
        "predictions": {
            "total": total_predictions,
            "open": open_predictions,
            "closed": closed_predictions,
        },
        "alerts": {
            "total": total_alerts,
            "active": active_alerts,
        },
        "audit_events": total_events,
    }


def log_event(
    db: Session,
    event_type: str,
    *,
    batch_id: Optional[str] = None,
    actor: Optional[str] = None,
    details: Optional[dict] = None,
) -> AuditLog:
    """Append a new event to the audit log."""
    event = AuditLog(
        event_type=event_type,
        batch_id=batch_id,
        actor=actor,
        details=details,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event
