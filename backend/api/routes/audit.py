"""
PlantIQ — Audit & Persistence API Routes
============================================
REST endpoints for the C-Tier persistence/auditability layer.

Endpoints:
  ── Predictions ──────────────────────────────────────
  GET  /store/predictions              — List stored predictions
  GET  /store/predictions/{batch_id}   — Get one prediction
  POST /store/predictions/{batch_id}/close — Close (make immutable)

  ── Outcomes ─────────────────────────────────────────
  POST /outcomes/record                — Record QC actuals for a batch

  ── Alerts ───────────────────────────────────────────
  GET  /alerts                         — List active alerts
  GET  /alerts/{alert_id}              — Get single alert
  GET  /alerts/batch/{batch_id}        — Get alerts for a batch
  PATCH /alerts/{alert_id}/transition  — Advance alert state
  GET  /alerts/stats                   — Alert counts by state
  POST /alerts/check-escalations       — Run escalation checks

  ── Feedback / Drift ─────────────────────────────────
  GET  /feedback/status                — Current drift status
  POST /feedback/compute               — Recompute rolling metrics
  GET  /feedback/history/{target}      — MAPE history for a target

  ── Model Registry ───────────────────────────────────
  GET  /registry/models                — List model versions
  GET  /registry/models/{version_id}   — Get model details
  POST /registry/models                — Register a new model

  ── Audit ────────────────────────────────────────────
  GET  /audit/{batch_id}               — Complete batch audit (THE question)
  GET  /audit/events                   — Query audit events
  GET  /audit/summary                  — High-level audit stats
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from database import prediction_store
from database import outcome_recorder
from database import alert_store
from database import feedback_loop
from database import model_registry
from database import audit_store

router = APIRouter(tags=["Persistence & Audit"])


# ═══════════════════════════════════════════════════════
# Pydantic Schemas
# ═══════════════════════════════════════════════════════

class OutcomeRequest(BaseModel):
    batch_id: str = Field(..., description="Batch to record outcomes for")
    actual_quality: float = Field(..., ge=0, le=100)
    actual_yield: float = Field(..., ge=0, le=100)
    actual_performance: float = Field(..., ge=0, le=100)
    actual_energy: float = Field(..., ge=0)
    recorded_by: str = Field(default="qc_operator")


class AlertTransitionRequest(BaseModel):
    new_state: str = Field(..., description="Target state: delivered/seen/acknowledged/acted_upon/resolved")
    actor: Optional[str] = None
    action_taken: Optional[str] = Field(None, description="For acted_upon: followed/declined/escalated")
    action_note: Optional[str] = None


class RegisterModelRequest(BaseModel):
    version_id: str = Field(..., description="Unique model version ID")
    model_type: str = Field(..., description="e.g. multi_target_xgboost, lstm_autoencoder")
    training_data_fingerprint: Optional[str] = None
    training_data_rows: Optional[int] = None
    training_metrics: Optional[dict] = None
    metadata_extra: Optional[dict] = None
    status: str = Field(default="active")


# ═══════════════════════════════════════════════════════
# Prediction Store Endpoints
# ═══════════════════════════════════════════════════════

@router.get("/store/predictions")
def list_stored_predictions(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List stored prediction records with optional filters."""
    records = prediction_store.list_predictions(
        db, status=status, limit=limit, offset=offset
    )
    total = prediction_store.count_predictions(db, status=status)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": [r.to_dict() for r in records],
    }


@router.get("/store/predictions/{batch_id}")
def get_stored_prediction(batch_id: str, db: Session = Depends(get_db)):
    """Retrieve a single prediction record."""
    record = prediction_store.get_prediction(db, batch_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {batch_id}")
    return record.to_dict()


@router.post("/store/predictions/{batch_id}/close")
def close_stored_prediction(batch_id: str, db: Session = Depends(get_db)):
    """Mark a prediction as closed (immutable)."""
    try:
        record = prediction_store.close_prediction(db, batch_id)
        return {"status": "closed", "batch_id": batch_id, "record": record.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════
# Outcome Recording
# ═══════════════════════════════════════════════════════

@router.post("/outcomes/record")
def record_batch_outcome(req: OutcomeRequest, db: Session = Depends(get_db)):
    """Record actual QC results for a batch and compute prediction errors."""
    try:
        record = outcome_recorder.record_outcome(
            db,
            batch_id=req.batch_id,
            actual_quality=req.actual_quality,
            actual_yield=req.actual_yield,
            actual_performance=req.actual_performance,
            actual_energy=req.actual_energy,
            recorded_by=req.recorded_by,
        )
        return {
            "status": "recorded",
            "batch_id": req.batch_id,
            "prediction_errors": record.prediction_errors,
            "actual_outcomes": record.actual_outcomes,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════
# Alert Lifecycle
# ═══════════════════════════════════════════════════════

@router.get("/alerts")
def list_alerts(
    severity: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    """List active (unresolved) alerts."""
    alerts = alert_store.list_active_alerts(db, severity=severity, limit=limit)
    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.get("/alerts/stats")
def alert_stats(db: Session = Depends(get_db)):
    """Alert counts grouped by lifecycle state."""
    return alert_store.count_alerts_by_state(db)


@router.get("/alerts/batch/{batch_id}")
def alerts_for_batch(batch_id: str, db: Session = Depends(get_db)):
    """Get all alerts for a specific batch."""
    alerts = alert_store.get_alerts_for_batch(db, batch_id)
    return {
        "batch_id": batch_id,
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.get("/alerts/{alert_id}")
def get_single_alert(alert_id: str, db: Session = Depends(get_db)):
    """Get details of a single alert."""
    alert = alert_store.get_alert(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
    return alert.to_dict()


@router.patch("/alerts/{alert_id}/transition")
def transition_single_alert(
    alert_id: str,
    req: AlertTransitionRequest,
    db: Session = Depends(get_db),
):
    """Advance an alert to the next lifecycle state."""
    try:
        alert = alert_store.transition_alert(
            db,
            alert_id,
            req.new_state,
            actor=req.actor,
            action_taken=req.action_taken,
            action_note=req.action_note,
        )
        return {"status": "transitioned", "alert": alert.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/alerts/check-escalations")
def run_escalation_check(db: Session = Depends(get_db)):
    """Check for alerts that need escalation due to timeout."""
    escalated = alert_store.check_escalations(db)
    return {
        "escalated_count": len(escalated),
        "escalated_alerts": [a.to_dict() for a in escalated],
    }


# ═══════════════════════════════════════════════════════
# Feedback / Drift Detection
# ═══════════════════════════════════════════════════════

@router.get("/feedback/status")
def feedback_status(db: Session = Depends(get_db)):
    """Current drift status for all prediction targets."""
    return feedback_loop.check_drift_status(db)


@router.post("/feedback/compute")
def compute_feedback_metrics(
    window_days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Recompute rolling MAPE metrics from closed predictions."""
    metrics = feedback_loop.compute_rolling_metrics(db, window_days=window_days)
    return {
        "computed_count": len(metrics),
        "metrics": [m.to_dict() for m in metrics],
    }


@router.get("/feedback/history/{target_name}")
def feedback_history(
    target_name: str,
    limit: int = Query(default=30, le=100),
    db: Session = Depends(get_db),
):
    """Historical MAPE values for a specific target (for charting)."""
    history = feedback_loop.get_metric_history(db, target_name, limit=limit)
    return {"target": target_name, "history": history}


# ═══════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════

@router.get("/registry/models")
def list_model_versions(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    """List registered model versions."""
    models = model_registry.list_models(
        db, model_type=model_type, status=status, limit=limit
    )
    return {
        "count": len(models),
        "models": [m.to_dict() for m in models],
    }


@router.get("/registry/models/{version_id}")
def get_model_version_detail(version_id: str, db: Session = Depends(get_db)):
    """Get details of a specific model version."""
    model = model_registry.get_model_version(db, version_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model version not found: {version_id}")
    return model.to_dict()


@router.post("/registry/models")
def register_model_version(req: RegisterModelRequest, db: Session = Depends(get_db)):
    """Register a new model version in the registry."""
    try:
        model = model_registry.register_model(
            db,
            version_id=req.version_id,
            model_type=req.model_type,
            training_data_fingerprint=req.training_data_fingerprint,
            training_data_rows=req.training_data_rows,
            training_metrics=req.training_metrics,
            metadata_extra=req.metadata_extra,
            status=req.status,
        )
        return {"status": "registered", "model": model.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════
# Audit (THE question: "What happened with Batch X?")
# ═══════════════════════════════════════════════════════

@router.get("/audit/summary")
def audit_summary(db: Session = Depends(get_db)):
    """High-level audit statistics."""
    return audit_store.get_audit_summary(db)


@router.get("/audit/events")
def query_audit_events(
    event_type: Optional[str] = None,
    batch_id: Optional[str] = None,
    actor: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Query audit events with flexible filters."""
    return audit_store.get_audit_events(
        db,
        event_type=event_type,
        batch_id=batch_id,
        actor=actor,
        limit=limit,
        offset=offset,
    )


@router.get("/audit/{batch_id}")
def get_batch_audit(batch_id: str, db: Session = Depends(get_db)):
    """Answer: 'What exactly happened with Batch <batch_id>?'

    Returns the complete audit record with prediction, outcomes,
    alerts, and timeline.
    """
    audit = audit_store.get_complete_batch_audit(db, batch_id)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"No records found for batch: {batch_id}")
    return audit
