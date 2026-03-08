"""
PlantIQ — Model Registry
============================
Component 1.4 — Version-tracked models with lifecycle management.

Maintains a complete registry of every model version so that any
historical prediction can be traced to the exact model that produced it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import ModelVersion, AuditLog


def register_model(
    db: Session,
    *,
    version_id: str,
    model_type: str,
    training_data_fingerprint: Optional[str] = None,
    training_data_rows: Optional[int] = None,
    training_metrics: Optional[dict] = None,
    metadata_extra: Optional[dict] = None,
    status: str = "active",
) -> ModelVersion:
    """Register a new model version in the registry."""
    existing = get_model_version(db, version_id)
    if existing:
        raise ValueError(f"Model version already exists: {version_id}")

    model = ModelVersion(
        version_id=version_id,
        model_type=model_type,
        status=status,
        training_data_fingerprint=training_data_fingerprint,
        training_data_rows=training_data_rows,
        training_metrics=training_metrics,
        metadata_extra=metadata_extra,
        deployed_at=datetime.utcnow() if status == "active" else None,
    )
    db.add(model)

    audit = AuditLog(
        event_type="model_registered",
        details={
            "version_id": version_id,
            "model_type": model_type,
            "status": status,
        },
    )
    db.add(audit)

    db.commit()
    db.refresh(model)
    return model


def deploy_model(db: Session, version_id: str) -> ModelVersion:
    """Mark a model version as deployed/active."""
    model = get_model_version(db, version_id)
    if model is None:
        raise ValueError(f"Model version not found: {version_id}")

    model.status = "active"
    model.deployed_at = datetime.utcnow()

    audit = AuditLog(
        event_type="model_deployed",
        details={"version_id": version_id},
    )
    db.add(audit)

    db.commit()
    db.refresh(model)
    return model


def retire_model(db: Session, version_id: str, reason: Optional[str] = None) -> ModelVersion:
    """Retire a model version — it won't be used for new predictions."""
    model = get_model_version(db, version_id)
    if model is None:
        raise ValueError(f"Model version not found: {version_id}")
    if model.status == "retired":
        raise ValueError(f"Model already retired: {version_id}")

    model.status = "retired"
    model.retired_at = datetime.utcnow()

    audit = AuditLog(
        event_type="model_retired",
        details={"version_id": version_id, "reason": reason},
    )
    db.add(audit)

    db.commit()
    db.refresh(model)
    return model


def get_model_version(db: Session, version_id: str) -> Optional[ModelVersion]:
    """Look up a single model version."""
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.version_id == version_id)
        .first()
    )


def get_active_model(db: Session, model_type: str) -> Optional[ModelVersion]:
    """Get the currently active model for a given type."""
    return (
        db.query(ModelVersion)
        .filter(
            ModelVersion.model_type == model_type,
            ModelVersion.status == "active",
        )
        .order_by(ModelVersion.deployed_at.desc())
        .first()
    )


def list_models(
    db: Session,
    *,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[ModelVersion]:
    """List model versions with optional filters."""
    query = db.query(ModelVersion)
    if model_type:
        query = query.filter(ModelVersion.model_type == model_type)
    if status:
        query = query.filter(ModelVersion.status == status)
    return query.order_by(ModelVersion.created_at.desc()).limit(limit).all()


def update_deployment_metrics(
    db: Session, version_id: str, metrics: dict
) -> ModelVersion:
    """Update rolling production metrics for a deployed model."""
    model = get_model_version(db, version_id)
    if model is None:
        raise ValueError(f"Model version not found: {version_id}")

    model.deployment_metrics = metrics
    db.commit()
    db.refresh(model)
    return model
