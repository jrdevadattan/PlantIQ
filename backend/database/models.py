"""
PlantIQ — Database ORM Models
================================
SQLAlchemy table definitions for the Data Foundation Layer.

Tables:
  1. prediction_records  — Every prediction with full provenance (Component 1.3)
  2. model_versions      — Model registry with lifecycle tracking (Component 1.4)
  3. alert_records       — Alert lifecycle with 6-state tracking (Component 3.4)
  4. feedback_metrics    — Rolling accuracy per target (Component 1.5)
  5. audit_log           — Append-only event log (Component 10)

All records are append-only after batch closure.  Corrections create
new linked records — originals are never modified.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    Index,
)
from sqlalchemy.types import TypeDecorator

from database import Base


# ═══════════════════════════════════════════════════════
# Custom JSON column type (SQLite-compatible)
# ═══════════════════════════════════════════════════════

class JSONType(TypeDecorator):
    """Store Python dicts/lists as JSON strings in SQLite."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


# ═══════════════════════════════════════════════════════
# Table 1: Prediction Records (Component 1.3)
# ═══════════════════════════════════════════════════════

class PredictionRecord(Base):
    """Immutable prediction record with complete provenance.

    Per README Component 1.3 — Prediction Store:
      "A complete record contains the batch identity and timestamp,
       the model version and training data fingerprint, all ten input
       features, all five predictions with confidence scores, the
       complete SHAP breakdown, the cost translation in rupees, any
       distribution warnings, and placeholder fields for QC outcomes."

    After a batch is marked closed, no field is modified.
    Corrections create new records referencing the original.
    """
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(16), default="open", nullable=False)  # open | closed

    # ── Model provenance ─────────────────────────────
    model_version = Column(String(64), nullable=True)
    model_type = Column(String(64), default="multi_target_xgboost")

    # ── Inputs ───────────────────────────────────────
    input_params = Column(JSONType, nullable=False)       # 7 operator inputs
    derived_features = Column(JSONType, nullable=True)    # computed features

    # ── Predictions ──────────────────────────────────
    predictions = Column(JSONType, nullable=False)        # quality, yield, perf, energy, co2
    confidence_intervals = Column(JSONType, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # ── Explainability ───────────────────────────────
    shap_breakdown = Column(JSONType, nullable=True)      # per-feature SHAP values
    shap_summary = Column(Text, nullable=True)            # plain English summary

    # ── Cost translation ─────────────────────────────
    cost_translation = Column(JSONType, nullable=True)    # INR cost, projections
    carbon_budget = Column(JSONType, nullable=True)       # CO₂ budget status

    # ── Distribution warnings ────────────────────────
    distribution_warnings = Column(JSONType, nullable=True)

    # ── Actual outcomes (filled by Outcome Recorder) ──
    actual_outcomes = Column(JSONType, nullable=True)     # actual quality, yield, perf, energy
    prediction_errors = Column(JSONType, nullable=True)   # per-target error
    outcome_recorded_at = Column(DateTime, nullable=True)
    outcome_recorded_by = Column(String(128), nullable=True)

    # ── Correction linkage ───────────────────────────
    correction_of = Column(String(64), nullable=True)     # batch_id of original if this is a correction
    correction_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_pred_status", "status"),
        Index("idx_pred_created", "created_at"),
    )

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status,
            "model_version": self.model_version,
            "input_params": self.input_params,
            "predictions": self.predictions,
            "confidence_intervals": self.confidence_intervals,
            "cost_translation": self.cost_translation,
            "carbon_budget": self.carbon_budget,
            "shap_breakdown": self.shap_breakdown,
            "shap_summary": self.shap_summary,
            "actual_outcomes": self.actual_outcomes,
            "prediction_errors": self.prediction_errors,
            "outcome_recorded_at": self.outcome_recorded_at.isoformat() if self.outcome_recorded_at else None,
            "outcome_recorded_by": self.outcome_recorded_by,
        }


# ═══════════════════════════════════════════════════════
# Table 2: Model Versions (Component 1.4)
# ═══════════════════════════════════════════════════════

class ModelVersion(Base):
    """Model registry with complete lifecycle tracking.

    Per README Component 1.4 — Model Registry:
      "Store every version of every model with complete metadata so
       that any historical prediction can be traced to the exact
       model that produced it."
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(String(64), unique=True, nullable=False, index=True)
    model_type = Column(String(64), nullable=False)  # multi_target_xgboost, lstm_autoencoder, etc.

    # ── Lifecycle ────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_at = Column(DateTime, nullable=True)
    retired_at = Column(DateTime, nullable=True)
    status = Column(String(16), default="active", nullable=False)  # active | retired | staging

    # ── Training provenance ──────────────────────────
    training_data_fingerprint = Column(String(128), nullable=True)  # SHA-256 of training data
    training_data_rows = Column(Integer, nullable=True)
    training_data_rows_real = Column(Integer, nullable=True)
    training_data_rows_augmented = Column(Integer, nullable=True)

    # ── Metrics ──────────────────────────────────────
    training_metrics = Column(JSONType, nullable=True)     # per-target MAE, MAPE, R²
    deployment_metrics = Column(JSONType, nullable=True)   # rolling MAPE in production

    # ── Metadata ─────────────────────────────────────
    metadata_extra = Column(JSONType, nullable=True)       # feature_cols, target_cols, hyperparams

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "status": self.status,
            "training_data_fingerprint": self.training_data_fingerprint,
            "training_data_rows": self.training_data_rows,
            "training_metrics": self.training_metrics,
            "deployment_metrics": self.deployment_metrics,
        }


# ═══════════════════════════════════════════════════════
# Table 3: Alert Records (Component 3.4)
# ═══════════════════════════════════════════════════════

class AlertRecord(Base):
    """Alert lifecycle with 6-state tracking.

    Per README Component 3.4 — Alert Acknowledgement System:
      "Every alert passes through six states: fired, delivered, seen,
       acknowledged, acted-upon, and resolved."

    Escalation: if not seen within timeout → supervisor.
    If supervisor doesn't ack within 5 min → plant manager.
    """
    __tablename__ = "alert_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, nullable=False, index=True)
    batch_id = Column(String(64), nullable=False, index=True)

    # ── Alert content ────────────────────────────────
    alert_type = Column(String(32), nullable=False)       # energy_overrun, anomaly, drift
    severity = Column(String(16), nullable=False)         # WATCH, WARNING, CRITICAL
    message = Column(Text, nullable=False)
    technical_detail = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    estimated_saving_kwh = Column(Float, nullable=True)
    quality_impact_pct = Column(Float, nullable=True)

    # ── 6-state lifecycle ────────────────────────────
    state = Column(String(16), default="fired", nullable=False)
    fired_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    delivered_at = Column(DateTime, nullable=True)
    seen_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(128), nullable=True)
    acted_upon_at = Column(DateTime, nullable=True)
    action_taken = Column(String(16), nullable=True)      # followed, declined, escalated
    action_note = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # ── Escalation ───────────────────────────────────
    escalation_level = Column(Integer, default=0)         # 0=none, 1=supervisor, 2=plant_manager
    escalated_at = Column(DateTime, nullable=True)
    escalated_to = Column(String(128), nullable=True)

    __table_args__ = (
        Index("idx_alert_batch", "batch_id"),
        Index("idx_alert_state", "state"),
        Index("idx_alert_severity", "severity"),
    )

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "batch_id": self.batch_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "estimated_saving_kwh": self.estimated_saving_kwh,
            "state": self.state,
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "seen_at": self.seen_at.isoformat() if self.seen_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "acted_upon_at": self.acted_upon_at.isoformat() if self.acted_upon_at else None,
            "action_taken": self.action_taken,
            "action_note": self.action_note,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalation_level": self.escalation_level,
        }


# ═══════════════════════════════════════════════════════
# Table 4: Feedback Metrics (Component 1.5)
# ═══════════════════════════════════════════════════════

class FeedbackMetric(Base):
    """Rolling accuracy metrics per prediction target.

    Per README Component 1.5 — Feedback Loop Engine:
      "The engine maintains a 30-day rolling MAPE for each of the
       five prediction targets. If any target exceeds 10% MAPE,
       an alert goes to the data team."
    """
    __tablename__ = "feedback_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    target_name = Column(String(32), nullable=False)      # quality_score, yield_pct, etc.

    # ── Accuracy metrics ─────────────────────────────
    rolling_mape = Column(Float, nullable=False)          # Mean Abs % Error
    rolling_mae = Column(Float, nullable=False)           # Mean Abs Error
    batch_count = Column(Integer, nullable=False)         # batches in rolling window

    # ── Drift detection ──────────────────────────────
    drift_detected = Column(Boolean, default=False)
    consecutive_degraded = Column(Integer, default=0)     # consecutive batches above threshold
    retrain_flag = Column(Boolean, default=False)         # True if 10+ consecutive degraded

    __table_args__ = (
        Index("idx_fb_target", "target_name"),
        Index("idx_fb_computed", "computed_at"),
    )

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "rolling_mape": round(self.rolling_mape, 4),
            "rolling_mae": round(self.rolling_mae, 4),
            "batch_count": self.batch_count,
            "drift_detected": self.drift_detected,
            "consecutive_degraded": self.consecutive_degraded,
            "retrain_flag": self.retrain_flag,
        }


# ═══════════════════════════════════════════════════════
# Table 5: Audit Log (Component 10)
# ═══════════════════════════════════════════════════════

class AuditLog(Base):
    """Append-only event log for complete traceability.

    Per README §10 — Audit Architecture:
      "The immutability rule is absolute. Once a batch is marked
       closed, no field in its audit record can be modified.
       Corrections are appended as new records."

    Event types:
      prediction_created, outcome_recorded, batch_closed,
      alert_fired, alert_acknowledged, alert_escalated,
      model_deployed, model_retired, drift_detected,
      correction_appended
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(32), nullable=False)
    batch_id = Column(String(64), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    actor = Column(String(128), nullable=True)            # who triggered this event
    details = Column(JSONType, nullable=True)             # event-specific data

    __table_args__ = (
        Index("idx_audit_type", "event_type"),
        Index("idx_audit_time", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actor": self.actor,
            "details": self.details,
        }
