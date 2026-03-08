"""
PlantIQ — D-Tier Test Suite
==============================
Tests all Decision Engine modules (Components 3.2–3.5) and their
API endpoints.

Run with:
    cd backend
    python -m pytest test_dtier.py -v

Pattern: same as test_ctier.py — standalone FastAPI app for API tests
to avoid XGBoost/libomp dependency.
"""

from __future__ import annotations

import os
import sys
import unittest

# ── Ensure backend on path ───────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ══════════════════════════════════════════════════════════════
# UNIT TESTS — Recommendation Engine
# ══════════════════════════════════════════════════════════════

class TestRecommendationEngine(unittest.TestCase):
    """Test decision_engine.recommendation_engine."""

    def setUp(self):
        from decision_engine.recommendation_engine import RecommendationEngine
        self.engine = RecommendationEngine()
        self.base_params = {
            "temperature": 190.0,
            "conveyor_speed": 85.0,
            "hold_time": 15.0,
            "batch_size": 500.0,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        }
        self.shap_contributions = [
            {"feature": "temperature", "contribution": 3.5, "direction": "increases"},
            {"feature": "conveyor_speed", "contribution": 2.1, "direction": "increases"},
            {"feature": "hold_time", "contribution": -0.8, "direction": "decreases"},
        ]

    def test_generate_returns_recommendation_set(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.target_focus, "energy_kwh")
        self.assertGreater(len(result.recommendations), 0)
        self.assertIsNotNone(result.summary)

    def test_recommendations_are_ranked(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
        )
        ranks = [r.rank for r in result.recommendations]
        self.assertEqual(ranks, sorted(ranks))

    def test_max_recommendations_respected(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
            max_recommendations=1,
        )
        self.assertLessEqual(len(result.recommendations), 1)

    def test_each_rec_has_instruction(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
        )
        for rec in result.recommendations:
            self.assertIsNotNone(rec.instruction)
            self.assertGreater(len(rec.instruction), 10)

    def test_each_rec_has_machine_info(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
        )
        for rec in result.recommendations:
            self.assertIsNotNone(rec.machine)
            self.assertIsNotNone(rec.control)

    def test_recommended_values_within_safe_range(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
        )
        from decision_engine.recommendation_engine import MACHINE_MAP
        for rec in result.recommendations:
            if rec.parameter in MACHINE_MAP:
                lo, hi = MACHINE_MAP[rec.parameter]["safe_range"]
                self.assertGreaterEqual(rec.recommended_value, lo)
                self.assertLessEqual(rec.recommended_value, hi)

    def test_batch_id_propagated(self):
        result = self.engine.generate(
            input_params=self.base_params,
            shap_contributions=self.shap_contributions,
            target="energy_kwh",
            batch_id="TEST-123",
        )
        self.assertEqual(result.batch_id, "TEST-123")


# ══════════════════════════════════════════════════════════════
# UNIT TESTS — Alert Engine
# ══════════════════════════════════════════════════════════════

class TestAlertEngine(unittest.TestCase):
    """Test decision_engine.alert_engine."""

    def setUp(self):
        from decision_engine.alert_engine import AlertEngine
        self.engine = AlertEngine(energy_target_kwh=42.0)

    def test_no_alerts_when_normal(self):
        alerts = self.engine.check_predictions(
            batch_id="B-OK",
            predictions={"energy_kwh": 40.0, "quality_score": 92.0},
        )
        self.assertEqual(len(alerts), 0)

    def test_energy_warning_at_15pct(self):
        # 42 * 1.16 = 48.72 → 16% over → WARNING
        alerts = self.engine.check_predictions(
            batch_id="B-WARN",
            predictions={"energy_kwh": 48.72, "quality_score": 92.0},
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "WARNING")
        self.assertEqual(alerts[0].alert_type, "energy_overrun")

    def test_energy_critical_at_25pct(self):
        # 42 * 1.26 = 52.92 → 26% over → CRITICAL
        alerts = self.engine.check_predictions(
            batch_id="B-CRIT",
            predictions={"energy_kwh": 52.92, "quality_score": 92.0},
        )
        energy_alerts = [a for a in alerts if a.alert_type == "energy_overrun"]
        self.assertEqual(len(energy_alerts), 1)
        self.assertEqual(energy_alerts[0].severity, "CRITICAL")

    def test_quality_warning(self):
        alerts = self.engine.check_predictions(
            batch_id="B-QW",
            predictions={"energy_kwh": 38.0, "quality_score": 75.0},
        )
        q_alerts = [a for a in alerts if a.alert_type == "quality_risk"]
        self.assertEqual(len(q_alerts), 1)
        self.assertEqual(q_alerts[0].severity, "WARNING")

    def test_quality_critical(self):
        alerts = self.engine.check_predictions(
            batch_id="B-QC",
            predictions={"energy_kwh": 38.0, "quality_score": 65.0},
        )
        q_alerts = [a for a in alerts if a.alert_type == "quality_risk"]
        self.assertEqual(len(q_alerts), 1)
        self.assertEqual(q_alerts[0].severity, "CRITICAL")

    def test_anomaly_critical(self):
        alerts = self.engine.check_anomaly(
            batch_id="B-AC",
            anomaly_score=0.72,
            fault_type="bearing_wear",
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "CRITICAL")
        self.assertEqual(alerts[0].alert_type, "anomaly")

    def test_anomaly_warning(self):
        alerts = self.engine.check_anomaly(
            batch_id="B-AW",
            anomaly_score=0.45,
            fault_type="wet_material",
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "WARNING")

    def test_anomaly_watch(self):
        alerts = self.engine.check_anomaly(
            batch_id="B-AWatch",
            anomaly_score=0.20,
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "WATCH")

    def test_anomaly_normal_no_alert(self):
        alerts = self.engine.check_anomaly(
            batch_id="B-AN",
            anomaly_score=0.10,
        )
        self.assertEqual(len(alerts), 0)

    def test_alert_has_complete_structure(self):
        alerts = self.engine.check_predictions(
            batch_id="B-STRUCT",
            predictions={"energy_kwh": 55.0, "quality_score": 92.0},
            shap_top_feature={"feature": "temperature", "contribution": 3.2},
            recommendation="Lower temperature on Drying Oven",
        )
        self.assertGreater(len(alerts), 0)
        a = alerts[0]
        self.assertIsNotNone(a.alert_id)
        self.assertIsNotNone(a.timestamp)
        self.assertIsNotNone(a.message)
        self.assertIsNotNone(a.technical_detail)
        self.assertIsNotNone(a.root_cause)
        self.assertIn("temperature", a.root_cause)

    def test_drift_alert(self):
        alerts = self.engine.check_drift(
            batch_id="B-DRIFT",
            target_name="energy_kwh",
            rolling_mape=15.0,
            consecutive_degraded=12,
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "CRITICAL")
        self.assertEqual(alerts[0].alert_type, "drift")


# ══════════════════════════════════════════════════════════════
# UNIT TESTS — Input Validator
# ══════════════════════════════════════════════════════════════

class TestInputValidator(unittest.TestCase):
    """Test decision_engine.input_validator."""

    def setUp(self):
        from decision_engine.input_validator import InputValidator
        self.validator = InputValidator()
        self.valid_params = {
            "temperature": 183.0,
            "conveyor_speed": 75.0,
            "hold_time": 18.0,
            "batch_size": 500.0,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        }

    def test_valid_params_pass_all_gates(self):
        result = self.validator.validate(self.valid_params)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

    def test_missing_field_fails_presence(self):
        params = {k: v for k, v in self.valid_params.items() if k != "temperature"}
        result = self.validator.validate(params)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].gate, "presence")

    def test_null_field_fails_presence(self):
        params = dict(self.valid_params, temperature=None)
        result = self.validator.validate(params)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].gate, "presence")

    def test_string_value_fails_type(self):
        params = dict(self.valid_params, temperature="hot")
        result = self.validator.validate(params)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].gate, "type")

    def test_out_of_physical_range_fails(self):
        params = dict(self.valid_params, temperature=999.0)
        result = self.validator.validate(params)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].gate, "physical")

    def test_out_of_training_range_warns(self):
        # 170 °C is within physical range (100–250) but outside training (175–195)
        params = dict(self.valid_params, temperature=170.0)
        result = self.validator.validate(params)
        self.assertTrue(result.is_valid)  # soft warning, not hard stop
        self.assertTrue(result.has_warnings)
        self.assertIn("temperature", result.ood_fields)

    def test_multiple_ood_fields(self):
        params = dict(self.valid_params, temperature=170.0, conveyor_speed=55.0)
        result = self.validator.validate(params)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.ood_fields), 2)

    def test_to_dict_structure(self):
        result = self.validator.validate(self.valid_params)
        d = result.to_dict()
        self.assertIn("is_valid", d)
        self.assertIn("errors", d)
        self.assertIn("warnings", d)
        self.assertIn("ood_fields", d)


# ══════════════════════════════════════════════════════════════
# UNIT TESTS — Confidence Scorer
# ══════════════════════════════════════════════════════════════

class TestConfidenceScorer(unittest.TestCase):
    """Test decision_engine.confidence_scorer."""

    def setUp(self):
        from decision_engine.confidence_scorer import ConfidenceScorer
        self.scorer = ConfidenceScorer()

    def test_base_confidence_green(self):
        report = self.scorer.compute()
        self.assertAlmostEqual(report.score, 0.95, places=2)
        self.assertEqual(report.indicator, "green")
        self.assertEqual(report.label, "High")

    def test_single_ood_reduces_to_amber(self):
        report = self.scorer.compute(ood_fields=["temperature"])
        self.assertAlmostEqual(report.score, 0.60, places=2)
        self.assertEqual(report.indicator, "amber")

    def test_ood_plus_drift_goes_red(self):
        report = self.scorer.compute(ood_fields=["temperature"], drift_detected=True)
        self.assertLess(report.score, 0.60)
        self.assertEqual(report.indicator, "red")

    def test_all_penalties_stacked(self):
        report = self.scorer.compute(
            ood_fields=["temperature"],
            drift_detected=True,
            feature_issue=True,
        )
        self.assertLess(report.score, 0.40)
        self.assertEqual(report.indicator, "red")
        self.assertEqual(len(report.penalties_applied), 3)

    def test_no_penalties_applied_list_empty(self):
        report = self.scorer.compute()
        self.assertEqual(len(report.penalties_applied), 0)

    def test_score_clamped_to_zero(self):
        # Extreme case: many OOD fields + all penalties
        report = self.scorer.compute(
            ood_fields=["temperature", "conveyor_speed", "hold_time", "batch_size"],
            drift_detected=True,
            feature_issue=True,
        )
        self.assertGreaterEqual(report.score, 0.0)

    def test_to_dict_structure(self):
        report = self.scorer.compute(ood_fields=["temperature"])
        d = report.to_dict()
        self.assertIn("score", d)
        self.assertIn("indicator", d)
        self.assertIn("label", d)
        self.assertIn("penalties_applied", d)
        self.assertIn("details", d)


# ══════════════════════════════════════════════════════════════
# API TESTS — Using standalone FastAPI app (no XGBoost needed)
# ══════════════════════════════════════════════════════════════

def _build_test_app():
    """Build a standalone FastAPI app with only D-tier routes.
    Avoids importing main.py which loads XGBoost models.
    """
    from fastapi import FastAPI
    from api.routes.recommend import router as recommend_router
    app = FastAPI()
    app.include_router(recommend_router)
    return app


class TestRecommendAPI(unittest.TestCase):
    """API tests for POST /recommend/* endpoints."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_generate_endpoint(self):
        resp = self.client.post("/recommend/generate", json={
            "input_params": {
                "temperature": 190.0,
                "conveyor_speed": 85.0,
                "hold_time": 15.0,
                "batch_size": 500.0,
                "material_type": 0,
                "hour_of_day": 10,
                "operator_exp": 2,
            },
            "shap_contributions": [
                {"feature": "temperature", "contribution": 3.5, "direction": "increases"},
                {"feature": "conveyor_speed", "contribution": 2.1, "direction": "increases"},
            ],
            "target": "energy_kwh",
            "batch_id": "API-TEST-001",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["batch_id"], "API-TEST-001")
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)
        self.assertIn("summary", data)

    def test_validate_endpoint_valid(self):
        resp = self.client.post("/recommend/validate", json={
            "temperature": 183.0,
            "conveyor_speed": 75.0,
            "hold_time": 18.0,
            "batch_size": 500.0,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_valid"])

    def test_validate_endpoint_missing_field(self):
        resp = self.client.post("/recommend/validate", json={
            "temperature": 183.0,
            # missing other fields
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_valid"])
        self.assertGreater(len(data["errors"]), 0)

    def test_validate_endpoint_ood_warning(self):
        resp = self.client.post("/recommend/validate", json={
            "temperature": 170.0,  # outside training but inside physical
            "conveyor_speed": 75.0,
            "hold_time": 18.0,
            "batch_size": 500.0,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_valid"])
        self.assertGreater(len(data["warnings"]), 0)
        self.assertIn("temperature", data["ood_fields"])

    def test_confidence_endpoint_green(self):
        resp = self.client.post("/recommend/confidence", json={
            "ood_fields": [],
            "drift_detected": False,
            "feature_issue": False,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["indicator"], "green")
        self.assertGreaterEqual(data["score"], 0.85)

    def test_confidence_endpoint_with_penalties(self):
        resp = self.client.post("/recommend/confidence", json={
            "ood_fields": ["temperature"],
            "drift_detected": True,
            "feature_issue": False,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["indicator"], "red")
        self.assertGreater(len(data["penalties_applied"]), 0)

    def test_alerts_endpoint_no_alerts(self):
        resp = self.client.post("/recommend/alerts", json={
            "batch_id": "B-SAFE",
            "predictions": {"energy_kwh": 40.0, "quality_score": 92.0},
            "energy_target_kwh": 42.0,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["alert_count"], 0)

    def test_alerts_endpoint_energy_overrun(self):
        resp = self.client.post("/recommend/alerts", json={
            "batch_id": "B-OVER",
            "predictions": {"energy_kwh": 55.0, "quality_score": 92.0},
            "energy_target_kwh": 42.0,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["alert_count"], 0)
        self.assertEqual(data["alerts"][0]["severity"], "CRITICAL")

    def test_alerts_endpoint_with_anomaly(self):
        resp = self.client.post("/recommend/alerts", json={
            "batch_id": "B-ANOM",
            "predictions": {"energy_kwh": 40.0, "quality_score": 92.0},
            "energy_target_kwh": 42.0,
            "anomaly_score": 0.72,
            "fault_type": "bearing_wear",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["alert_count"], 0)
        anomaly_alerts = [a for a in data["alerts"] if a["alert_type"] == "anomaly"]
        self.assertEqual(len(anomaly_alerts), 1)
        self.assertEqual(anomaly_alerts[0]["severity"], "CRITICAL")


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
