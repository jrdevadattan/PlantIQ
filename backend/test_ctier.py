"""
PlantIQ — C-Tier Comprehensive Tests
========================================
Tests all 7 C-tier features:
  1. Database Layer (SQLite + SQLAlchemy ORM)
  2. Prediction Store (save / query / close / immutability)
  3. Model Registry (register / deploy / retire)
  4. Outcome Recorder (record QC actuals, compute errors)
  5. Feedback Loop Engine (rolling MAPE, drift detection)
  6. Alert Store (6-state lifecycle, escalation)
  7. Audit Store (complete batch audit — THE question)

Run with:
    cd backend
    python3 test_ctier.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import traceback
from datetime import datetime, timedelta

# Ensure backend is on path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Use in-memory SQLite for tests (override before importing database)
os.environ["PLANTIQ_DB_URL"] = "sqlite:///:memory:"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, PredictionRecord, ModelVersion, AlertRecord, FeedbackMetric, AuditLog


# ── Test database setup ─────────────────────────────────
_test_engine = create_engine("sqlite:///:memory:", echo=False)
_TestSession = sessionmaker(bind=_test_engine)

def _fresh_db():
    """Create all tables and return a new session."""
    Base.metadata.drop_all(_test_engine)
    Base.metadata.create_all(_test_engine)
    return _TestSession()


# ═══════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════

_passed = 0
_failed = 0


def _run_test(name: str, func):
    global _passed, _failed
    try:
        func()
        print(f"  ✅ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        traceback.print_exc()
        _failed += 1


# ═══════════════════════════════════════════════════════
# 1. Database Layer Tests
# ═══════════════════════════════════════════════════════

def test_tables_created():
    db = _fresh_db()
    # Verify all 5 tables exist
    table_names = Base.metadata.tables.keys()
    assert "prediction_records" in table_names
    assert "model_versions" in table_names
    assert "alert_records" in table_names
    assert "feedback_metrics" in table_names
    assert "audit_log" in table_names
    db.close()


def test_json_type_column():
    db = _fresh_db()
    record = PredictionRecord(
        batch_id="TEST_JSON_001",
        input_params={"temperature": 185.0, "batch_size": 500},
        predictions={"quality_score": 92.5, "energy_kwh": 38.0},
    )
    db.add(record)
    db.commit()

    loaded = db.query(PredictionRecord).filter_by(batch_id="TEST_JSON_001").first()
    assert loaded.input_params["temperature"] == 185.0
    assert loaded.predictions["quality_score"] == 92.5
    db.close()


# ═══════════════════════════════════════════════════════
# 2. Prediction Store Tests
# ═══════════════════════════════════════════════════════

def test_save_prediction():
    db = _fresh_db()
    from database.prediction_store import save_prediction, get_prediction

    record = save_prediction(
        db,
        batch_id="BATCH_TEST_001",
        input_params={"temperature": 183.0, "conveyor_speed": 75.0, "hold_time": 18.0},
        predictions={"quality_score": 91.5, "yield_pct": 88.2, "performance_pct": 85.0, "energy_kwh": 38.5},
        model_version="xgb-v1.0.0",
        confidence_score=0.87,
        cost_translation={"predicted_cost_inr": 327.25},
    )
    assert record.batch_id == "BATCH_TEST_001"
    assert record.status == "open"
    assert record.predictions["quality_score"] == 91.5

    # Verify it's in the database
    loaded = get_prediction(db, "BATCH_TEST_001")
    assert loaded is not None
    assert loaded.model_version == "xgb-v1.0.0"
    db.close()


def test_list_and_count_predictions():
    db = _fresh_db()
    from database.prediction_store import save_prediction, list_predictions, count_predictions

    for i in range(5):
        save_prediction(
            db,
            batch_id=f"BATCH_LIST_{i:03d}",
            input_params={"temperature": 180 + i},
            predictions={"quality_score": 90 + i},
        )

    records = list_predictions(db, limit=10)
    assert len(records) == 5

    count = count_predictions(db)
    assert count == 5

    count_open = count_predictions(db, status="open")
    assert count_open == 5
    db.close()


def test_close_prediction():
    db = _fresh_db()
    from database.prediction_store import save_prediction, close_prediction

    save_prediction(
        db,
        batch_id="BATCH_CLOSE_001",
        input_params={"temperature": 185},
        predictions={"quality_score": 90},
    )
    closed = close_prediction(db, "BATCH_CLOSE_001")
    assert closed.status == "closed"

    # Trying to close again should raise
    try:
        close_prediction(db, "BATCH_CLOSE_001")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    db.close()


def test_auto_batch_id():
    db = _fresh_db()
    from database.prediction_store import save_prediction

    record = save_prediction(
        db,
        input_params={"temperature": 183},
        predictions={"quality_score": 90},
    )
    assert record.batch_id.startswith("BATCH_")
    assert len(record.batch_id) > 10
    db.close()


# ═══════════════════════════════════════════════════════
# 3. Model Registry Tests
# ═══════════════════════════════════════════════════════

def test_register_model():
    db = _fresh_db()
    from database.model_registry import register_model, get_model_version

    model = register_model(
        db,
        version_id="xgb-v1.0.0",
        model_type="multi_target_xgboost",
        training_data_rows=800,
        training_metrics={"quality_mae": 1.2, "energy_mae": 2.1},
    )
    assert model.version_id == "xgb-v1.0.0"
    assert model.status == "active"

    loaded = get_model_version(db, "xgb-v1.0.0")
    assert loaded is not None
    assert loaded.training_metrics["quality_mae"] == 1.2
    db.close()


def test_retire_model():
    db = _fresh_db()
    from database.model_registry import register_model, retire_model, get_model_version

    register_model(db, version_id="xgb-v1.0.0", model_type="multi_target_xgboost")
    retired = retire_model(db, "xgb-v1.0.0", reason="Replaced by v1.1.0")
    assert retired.status == "retired"
    assert retired.retired_at is not None

    # Double-retire should fail
    try:
        retire_model(db, "xgb-v1.0.0")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    db.close()


def test_get_active_model():
    db = _fresh_db()
    from database.model_registry import register_model, retire_model, get_active_model

    register_model(db, version_id="xgb-v1.0.0", model_type="multi_target_xgboost")
    register_model(db, version_id="xgb-v1.1.0", model_type="multi_target_xgboost")

    active = get_active_model(db, "multi_target_xgboost")
    assert active is not None
    # Most recent deployment should be returned
    assert active.version_id in ("xgb-v1.0.0", "xgb-v1.1.0")
    db.close()


def test_duplicate_version_rejected():
    db = _fresh_db()
    from database.model_registry import register_model

    register_model(db, version_id="xgb-v1.0.0", model_type="multi_target_xgboost")
    try:
        register_model(db, version_id="xgb-v1.0.0", model_type="multi_target_xgboost")
        assert False, "Should have raised ValueError for duplicate"
    except ValueError:
        pass
    db.close()


# ═══════════════════════════════════════════════════════
# 4. Outcome Recorder Tests
# ═══════════════════════════════════════════════════════

def test_record_outcome():
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.outcome_recorder import record_outcome

    save_prediction(
        db,
        batch_id="BATCH_QC_001",
        input_params={"temperature": 183},
        predictions={"quality_score": 91.5, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 38.0},
    )

    record = record_outcome(
        db,
        batch_id="BATCH_QC_001",
        actual_quality=90.0,
        actual_yield=87.5,
        actual_performance=84.0,
        actual_energy=39.0,
        recorded_by="qc_team_lead",
    )
    assert record.status == "closed"
    assert record.actual_outcomes["quality_score"] == 90.0
    assert record.prediction_errors is not None
    assert "quality_score" in record.prediction_errors

    # Quality error: |91.5 - 90.0| = 1.5, pct = 1.5/90 * 100 = 1.67%
    q_err = record.prediction_errors["quality_score"]
    assert abs(q_err["absolute_error"] - 1.5) < 0.01
    assert abs(q_err["percentage_error"] - 1.67) < 0.1
    db.close()


def test_outcome_duplicate_rejected():
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.outcome_recorder import record_outcome

    save_prediction(
        db,
        batch_id="BATCH_QC_002",
        input_params={"temperature": 183},
        predictions={"quality_score": 91.5, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 38.0},
    )
    record_outcome(db, "BATCH_QC_002", 90.0, 87.5, 84.0, 39.0)

    try:
        record_outcome(db, "BATCH_QC_002", 91.0, 88.0, 85.0, 38.0)
        assert False, "Should have raised ValueError for duplicate outcome"
    except ValueError:
        pass
    db.close()


# ═══════════════════════════════════════════════════════
# 5. Feedback Loop Engine Tests
# ═══════════════════════════════════════════════════════

def test_compute_rolling_metrics():
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.outcome_recorder import record_outcome
    from database.feedback_loop import compute_rolling_metrics, get_latest_metrics

    # Create 5 predictions with outcomes
    for i in range(5):
        save_prediction(
            db,
            batch_id=f"BATCH_FB_{i:03d}",
            input_params={"temperature": 183},
            predictions={"quality_score": 90.0, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 38.0},
        )
        # Actuals with ~2% error
        record_outcome(
            db,
            f"BATCH_FB_{i:03d}",
            actual_quality=88.0 + i * 0.5,
            actual_yield=87.0 + i * 0.3,
            actual_performance=84.0 + i * 0.2,
            actual_energy=39.0 - i * 0.1,
        )

    metrics = compute_rolling_metrics(db)
    assert len(metrics) > 0

    latest = get_latest_metrics(db)
    assert "quality_score" in latest
    assert latest["quality_score"]["batch_count"] == 5
    db.close()


def test_drift_detection():
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.outcome_recorder import record_outcome
    from database.feedback_loop import compute_rolling_metrics, check_drift_status

    # Create predictions with large errors (>10% MAPE) to trigger drift
    for i in range(12):
        save_prediction(
            db,
            batch_id=f"BATCH_DRIFT_{i:03d}",
            input_params={"temperature": 183},
            predictions={"quality_score": 90.0, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 38.0},
        )
        # 15% error on quality → should trigger drift
        record_outcome(
            db,
            f"BATCH_DRIFT_{i:03d}",
            actual_quality=76.5,  # ~15% error from 90
            actual_yield=87.0,
            actual_performance=84.0,
            actual_energy=39.0,
        )

    compute_rolling_metrics(db)
    status = check_drift_status(db)
    assert status["any_drift_detected"] is True
    # quality_score should show drift
    assert status["targets"]["quality_score"]["drift_detected"] is True
    # 12 consecutive → retrain flag
    assert status["targets"]["quality_score"]["retrain_flag"] is True
    db.close()


# ═══════════════════════════════════════════════════════
# 6. Alert Store Tests
# ═══════════════════════════════════════════════════════

def test_fire_and_get_alert():
    db = _fresh_db()
    from database.alert_store import fire_alert, get_alert

    alert = fire_alert(
        db,
        batch_id="BATCH_ALERT_001",
        alert_type="energy_overrun",
        severity="WARNING",
        message="Energy 43.5 kWh exceeds target 42.0 kWh",
        recommended_action="Reduce conveyor speed by 5%",
        estimated_saving_kwh=1.5,
    )
    assert alert.state == "fired"
    assert alert.severity == "WARNING"

    loaded = get_alert(db, alert.alert_id)
    assert loaded is not None
    assert loaded.message == "Energy 43.5 kWh exceeds target 42.0 kWh"
    db.close()


def test_alert_lifecycle():
    db = _fresh_db()
    from database.alert_store import fire_alert, transition_alert

    alert = fire_alert(
        db,
        batch_id="BATCH_LC_001",
        alert_type="anomaly",
        severity="CRITICAL",
        message="Anomaly score 0.72 — critical",
    )
    alert_id = alert.alert_id

    # Progress through all 6 states
    t = transition_alert(db, alert_id, "delivered")
    assert t.state == "delivered"

    t = transition_alert(db, alert_id, "seen")
    assert t.state == "seen"

    t = transition_alert(db, alert_id, "acknowledged", actor="operator_john")
    assert t.state == "acknowledged"
    assert t.acknowledged_by == "operator_john"

    t = transition_alert(db, alert_id, "acted_upon", action_taken="followed", action_note="Reduced speed")
    assert t.state == "acted_upon"
    assert t.action_taken == "followed"

    t = transition_alert(db, alert_id, "resolved")
    assert t.state == "resolved"
    assert t.resolved_at is not None
    db.close()


def test_alert_no_backward_transition():
    db = _fresh_db()
    from database.alert_store import fire_alert, transition_alert

    alert = fire_alert(
        db,
        batch_id="BATCH_BACK_001",
        alert_type="drift",
        severity="WATCH",
        message="Drift detected",
    )
    transition_alert(db, alert.alert_id, "acknowledged")

    # Try to go backward
    try:
        transition_alert(db, alert.alert_id, "delivered")
        assert False, "Should have raised ValueError for backward transition"
    except ValueError:
        pass
    db.close()


def test_alerts_for_batch():
    db = _fresh_db()
    from database.alert_store import fire_alert, get_alerts_for_batch

    fire_alert(db, batch_id="BATCH_MULTI_001", alert_type="energy_overrun", severity="WARNING", message="Alert 1")
    fire_alert(db, batch_id="BATCH_MULTI_001", alert_type="anomaly", severity="CRITICAL", message="Alert 2")
    fire_alert(db, batch_id="BATCH_MULTI_002", alert_type="drift", severity="WATCH", message="Alert 3")

    alerts = get_alerts_for_batch(db, "BATCH_MULTI_001")
    assert len(alerts) == 2
    db.close()


def test_alert_count_by_state():
    db = _fresh_db()
    from database.alert_store import fire_alert, transition_alert, count_alerts_by_state

    a1 = fire_alert(db, batch_id="B1", alert_type="x", severity="WARNING", message="m1")
    a2 = fire_alert(db, batch_id="B2", alert_type="x", severity="CRITICAL", message="m2")
    transition_alert(db, a2.alert_id, "resolved")

    counts = count_alerts_by_state(db)
    assert counts["fired"] == 1
    assert counts["resolved"] == 1
    db.close()


# ═══════════════════════════════════════════════════════
# 7. Audit Store Tests — THE question
# ═══════════════════════════════════════════════════════

def test_complete_batch_audit():
    """THE acid test: 'What exactly happened with Batch X?'"""
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.outcome_recorder import record_outcome
    from database.alert_store import fire_alert, transition_alert
    from database.audit_store import get_complete_batch_audit

    batch_id = "BATCH_AUDIT_001"

    # 1. Prediction created
    save_prediction(
        db,
        batch_id=batch_id,
        input_params={"temperature": 183, "conveyor_speed": 75, "hold_time": 18},
        predictions={"quality_score": 91.5, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 43.5},
        model_version="xgb-v1.0.0",
        cost_translation={"predicted_cost_inr": 369.75},
        shap_breakdown=[{"feature": "temperature", "contribution": 3.2}],
    )

    # 2. Alert fired (energy overrun)
    alert = fire_alert(
        db,
        batch_id=batch_id,
        alert_type="energy_overrun",
        severity="WARNING",
        message="Energy 43.5 kWh exceeds target 42.0 kWh",
        recommended_action="Reduce conveyor speed by 5%",
    )
    transition_alert(db, alert.alert_id, "acknowledged", actor="operator_smith")

    # 3. QC outcome recorded
    record_outcome(db, batch_id, 90.0, 87.5, 84.0, 44.0, "qc_lead")

    # NOW: answer THE question
    audit = get_complete_batch_audit(db, batch_id)

    assert audit is not None
    assert audit["batch_id"] == batch_id
    assert audit["status"] == "closed"

    # Prediction data present
    assert audit["prediction"]["model_version"] == "xgb-v1.0.0"
    assert audit["prediction"]["input_params"]["temperature"] == 183
    assert audit["prediction"]["predictions"]["quality_score"] == 91.5
    assert audit["prediction"]["cost_translation"]["predicted_cost_inr"] == 369.75
    assert len(audit["prediction"]["shap_breakdown"]) == 1

    # Outcomes present
    assert audit["outcomes"]["recorded"] is True
    assert audit["outcomes"]["actual_values"]["quality_score"] == 90.0
    assert audit["outcomes"]["recorded_by"] == "qc_lead"

    # Alerts present
    assert audit["alerts"]["count"] == 1
    assert audit["alerts"]["records"][0]["severity"] == "WARNING"
    assert audit["alerts"]["records"][0]["state"] == "acknowledged"

    # Timeline has events
    assert len(audit["timeline"]) >= 3  # prediction_created, alert_fired, outcome_recorded

    db.close()


def test_audit_events_query():
    db = _fresh_db()
    from database.audit_store import log_event, get_audit_events

    log_event(db, "prediction_created", batch_id="B001", actor="system")
    log_event(db, "alert_fired", batch_id="B001", actor="system")
    log_event(db, "outcome_recorded", batch_id="B001", actor="qc_team")
    log_event(db, "prediction_created", batch_id="B002", actor="system")

    # All events
    all_events = get_audit_events(db)
    assert len(all_events) == 4

    # Filter by batch
    b001_events = get_audit_events(db, batch_id="B001")
    assert len(b001_events) == 3

    # Filter by type
    pred_events = get_audit_events(db, event_type="prediction_created")
    assert len(pred_events) == 2
    db.close()


def test_audit_summary():
    db = _fresh_db()
    from database.prediction_store import save_prediction
    from database.alert_store import fire_alert
    from database.audit_store import get_audit_summary

    save_prediction(db, batch_id="B1", input_params={}, predictions={})
    save_prediction(db, batch_id="B2", input_params={}, predictions={})
    fire_alert(db, batch_id="B1", alert_type="x", severity="WARNING", message="m")

    summary = get_audit_summary(db)
    assert summary["predictions"]["total"] == 2
    assert summary["predictions"]["open"] == 2
    assert summary["alerts"]["total"] == 1
    db.close()


# ═══════════════════════════════════════════════════════
# 8. to_dict() serialization tests
# ═══════════════════════════════════════════════════════

def test_prediction_to_dict():
    db = _fresh_db()
    from database.prediction_store import save_prediction

    record = save_prediction(
        db,
        batch_id="BATCH_DICT_001",
        input_params={"temperature": 183},
        predictions={"quality_score": 91.5},
        model_version="xgb-v1.0.0",
    )
    d = record.to_dict()
    assert d["batch_id"] == "BATCH_DICT_001"
    assert d["model_version"] == "xgb-v1.0.0"
    assert isinstance(d["created_at"], str)  # ISO string
    assert d["actual_outcomes"] is None
    db.close()


def test_alert_to_dict():
    db = _fresh_db()
    from database.alert_store import fire_alert

    alert = fire_alert(
        db,
        batch_id="B1",
        alert_type="anomaly",
        severity="CRITICAL",
        message="Test",
    )
    d = alert.to_dict()
    assert d["state"] == "fired"
    assert d["severity"] == "CRITICAL"
    assert isinstance(d["fired_at"], str)
    db.close()


# ═══════════════════════════════════════════════════════
# API Endpoint Tests (via FastAPI TestClient)
# ═══════════════════════════════════════════════════════

def test_api_endpoints():
    """Test the REST API endpoints through FastAPI TestClient.

    Uses a standalone FastAPI app (no model loading) to avoid XGBoost/libomp dependency.
    """
    import tempfile
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import get_db
    from database.models import Base as ModelsBase
    from api.routes.audit import router as audit_router

    # Use a temp file database to avoid issues with in-memory and threading
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    try:
        api_engine = create_engine(f"sqlite:///{tmp_db_path}", connect_args={"check_same_thread": False})
        ApiSession = sessionmaker(autocommit=False, autoflush=False, bind=api_engine)

        # Create all tables
        ModelsBase.metadata.create_all(bind=api_engine)

        def override_get_db():
            session = ApiSession()
            try:
                yield session
            finally:
                session.close()

        # Standalone test app — audit routes only (no model loading)
        test_app = FastAPI()
        test_app.include_router(audit_router)
        test_app.dependency_overrides[get_db] = override_get_db

        client = TestClient(test_app)

        # ── 1. Store: list predictions (empty) ─────────────
        r = client.get("/store/predictions")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0

        # ── 2. Create a prediction via prediction_store directly ──
        api_db = ApiSession()
        from database.prediction_store import save_prediction
        save_prediction(
            api_db,
            batch_id="API_BATCH_001",
            input_params={"temperature": 183},
            predictions={"quality_score": 91.5, "yield_pct": 88.0, "performance_pct": 85.0, "energy_kwh": 38.5},
            model_version="xgb-v1.0.0",
        )
        api_db.close()

        # ── 3. Get stored prediction ───────────────────────
        r = client.get("/store/predictions/API_BATCH_001")
        assert r.status_code == 200
        assert r.json()["batch_id"] == "API_BATCH_001"

        # ── 4. Record outcome ──────────────────────────────
        r = client.post("/outcomes/record", json={
            "batch_id": "API_BATCH_001",
            "actual_quality": 90.0,
            "actual_yield": 87.5,
            "actual_performance": 84.0,
            "actual_energy": 39.0,
            "recorded_by": "test_qc",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"

        # ── 5. Audit: get complete batch audit ─────────────
        r = client.get("/audit/API_BATCH_001")
        assert r.status_code == 200
        audit = r.json()
        assert audit["batch_id"] == "API_BATCH_001"
        assert audit["outcomes"]["recorded"] is True

        # ── 6. Alert endpoints ─────────────────────────────
        r = client.get("/alerts")
        assert r.status_code == 200

        r = client.get("/alerts/stats")
        assert r.status_code == 200

        # ── 7. Feedback status ─────────────────────────────
        r = client.get("/feedback/status")
        assert r.status_code == 200

        # ── 8. Model registry ──────────────────────────────
        r = client.post("/registry/models", json={
            "version_id": "xgb-test-v1",
            "model_type": "multi_target_xgboost",
            "training_data_rows": 800,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "registered"

        r = client.get("/registry/models")
        assert r.status_code == 200
        assert r.json()["count"] == 1

        # ── 9. Audit summary ──────────────────────────────
        r = client.get("/audit/summary")
        assert r.status_code == 200
        assert r.json()["predictions"]["total"] >= 1

        # ── 10. 404 for non-existent batch ────────────────
        r = client.get("/store/predictions/NON_EXISTENT")
        assert r.status_code == 404

    finally:
        # Cleanup temp database
        os.unlink(tmp_db_path)


# ═══════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PlantIQ C-Tier Tests — Persistence & Auditability Layer")
    print("=" * 60)

    print("\n📦 1. Database Layer")
    _run_test("Tables created correctly", test_tables_created)
    _run_test("JSON column type works", test_json_type_column)

    print("\n📋 2. Prediction Store")
    _run_test("Save prediction", test_save_prediction)
    _run_test("List and count predictions", test_list_and_count_predictions)
    _run_test("Close prediction (immutability)", test_close_prediction)
    _run_test("Auto-generate batch ID", test_auto_batch_id)

    print("\n🏷️  3. Model Registry")
    _run_test("Register model version", test_register_model)
    _run_test("Retire model version", test_retire_model)
    _run_test("Get active model", test_get_active_model)
    _run_test("Duplicate version rejected", test_duplicate_version_rejected)

    print("\n📊 4. Outcome Recorder")
    _run_test("Record QC outcome + compute errors", test_record_outcome)
    _run_test("Duplicate outcome rejected", test_outcome_duplicate_rejected)

    print("\n📈 5. Feedback Loop Engine")
    _run_test("Compute rolling MAPE metrics", test_compute_rolling_metrics)
    _run_test("Drift detection (10+ consecutive)", test_drift_detection)

    print("\n🔔 6. Alert Store")
    _run_test("Fire and get alert", test_fire_and_get_alert)
    _run_test("Full 6-state lifecycle", test_alert_lifecycle)
    _run_test("No backward transitions", test_alert_no_backward_transition)
    _run_test("Alerts for batch", test_alerts_for_batch)
    _run_test("Count by state", test_alert_count_by_state)

    print("\n🔍 7. Audit Store — THE question")
    _run_test("Complete batch audit", test_complete_batch_audit)
    _run_test("Audit events query", test_audit_events_query)
    _run_test("Audit summary", test_audit_summary)

    print("\n🔄 8. Serialization")
    _run_test("PredictionRecord.to_dict()", test_prediction_to_dict)
    _run_test("AlertRecord.to_dict()", test_alert_to_dict)

    print("\n🌐 9. API Endpoints")
    _run_test("REST API endpoint integration", test_api_endpoints)

    print("\n" + "=" * 60)
    total = _passed + _failed
    print(f"Results: {_passed}/{total} passed, {_failed} failed")
    if _failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED — check output above")
    print("=" * 60 + "\n")

    sys.exit(0 if _failed == 0 else 1)
