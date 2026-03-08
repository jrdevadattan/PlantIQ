"""
PlantIQ — Cost Translator API Test
=====================================
Spins up a minimal FastAPI with only the cost route
and tests all 3 endpoints via TestClient (no ML models needed).

Run: python3 test_cost_api.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routes.cost import router

# Minimal app with only cost routes
app = FastAPI(title="Cost Translator Test")
app.include_router(router)
client = TestClient(app)


def test_translate_endpoint():
    """POST /cost/translate — full cost breakdown."""
    print("=== API Test 1: POST /cost/translate ===")

    resp = client.post("/cost/translate", json={"energy_kwh": 45.0})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    data = resp.json()
    print(f"  Status:              {resp.status_code}")
    print(f"  predicted_cost_inr:  {data['predicted_cost_inr']}")
    print(f"  energy_variance_pct: {data['energy_variance_pct']}%")
    print(f"  co2_kg:              {data['co2_kg']}")
    print(f"  co2_status:          {data['co2_status']}")
    print(f"  monthly_cost_inr:    {data['monthly_cost_inr']}")
    print(f"  monthly_batches:     {data['monthly_batches']}")
    print(f"  summary length:      {len(data['summary'])} chars")

    # Verify all required fields present
    required = [
        "predicted_energy_kwh", "energy_target_kwh", "energy_variance_kwh",
        "energy_variance_pct", "tariff_inr_per_kwh", "predicted_cost_inr",
        "target_cost_inr", "cost_variance_inr", "cost_variance_pct",
        "co2_kg", "co2_budget_kg", "co2_variance_kg", "co2_status",
        "batches_per_day", "operating_days_per_month", "monthly_batches",
        "monthly_energy_kwh", "monthly_cost_inr", "monthly_co2_kg",
        "monthly_target_cost_inr", "monthly_savings_inr",
        "savings_if_optimized_pct", "potential_monthly_saving_inr",
        "potential_annual_saving_inr", "summary",
    ]
    for field in required:
        assert field in data, f"Missing field: {field}"

    assert data["predicted_cost_inr"] == 382.5  # 45.0 * 8.50
    assert data["co2_kg"] == 36.9  # 45.0 * 0.82
    print("  PASSED\n")


def test_translate_with_overrides():
    """POST /cost/translate with what-if overrides."""
    print("=== API Test 2: POST /cost/translate (with overrides) ===")

    resp = client.post("/cost/translate", json={
        "energy_kwh": 50.0,
        "tariff_inr_per_kwh": 12.0,
        "batches_per_day": 15,
        "operating_days_per_month": 20,
    })
    assert resp.status_code == 200

    data = resp.json()
    print(f"  tariff:           INR {data['tariff_inr_per_kwh']}/kWh")
    print(f"  cost:             INR {data['predicted_cost_inr']}")
    print(f"  monthly_batches:  {data['monthly_batches']}")
    print(f"  monthly_cost:     INR {data['monthly_cost_inr']}")

    assert data["tariff_inr_per_kwh"] == 12.0
    assert data["monthly_batches"] == 300  # 15 * 20
    assert data["predicted_cost_inr"] == 600.0  # 50 * 12
    print("  PASSED\n")


def test_translate_batch_endpoint():
    """POST /cost/translate-batch — lightweight batch enrichment."""
    print("=== API Test 3: POST /cost/translate-batch ===")

    resp = client.post("/cost/translate-batch", json={
        "batch_id": "BATCH_20260308_120000",
        "energy_kwh": 40.0,
        "quality_score": 92.3,
    })
    assert resp.status_code == 200

    data = resp.json()
    print(f"  batch_id:             {data['batch_id']}")
    print(f"  cost_inr:             INR {data['cost_inr']}")
    print(f"  co2_kg:               {data['co2_kg']}")
    print(f"  co2_status:           {data['co2_status']}")
    print(f"  cost_variance_inr:    INR {data['cost_variance_inr']}")
    print(f"  monthly_projection:   INR {data['monthly_projection_inr']}")

    assert data["batch_id"] == "BATCH_20260308_120000"
    assert data["cost_inr"] == 340.0  # 40 * 8.50
    print("  PASSED\n")


def test_config_endpoint():
    """GET /cost/config — current configuration."""
    print("=== API Test 4: GET /cost/config ===")

    resp = client.get("/cost/config")
    assert resp.status_code == 200

    data = resp.json()
    print(f"  tariff:            INR {data['tariff_inr_per_kwh']}/kWh")
    print(f"  co2_factor:        {data['co2_factor_kg_per_kwh']} kg/kWh")
    print(f"  energy_target:     {data['energy_target_kwh']} kWh")
    print(f"  co2_budget:        {data['co2_budget_kg']} kg")
    print(f"  batches/day:       {data['batches_per_day']}")
    print(f"  operating_days:    {data['operating_days_per_month']}")
    print(f"  optimization:      {data['optimization_headroom_pct']}%")

    assert data["tariff_inr_per_kwh"] == 8.5
    assert data["co2_factor_kg_per_kwh"] == 0.82
    print("  PASSED\n")


def test_validation_errors():
    """Test Pydantic validation on bad inputs."""
    print("=== API Test 5: Input validation ===")

    # Missing required field
    resp = client.post("/cost/translate", json={})
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    print(f"  Missing energy_kwh:  {resp.status_code} (correct)")

    # Negative energy
    resp = client.post("/cost/translate", json={"energy_kwh": -5.0})
    assert resp.status_code == 422
    print(f"  Negative energy:     {resp.status_code} (correct)")

    # Over max energy
    resp = client.post("/cost/translate", json={"energy_kwh": 999.0})
    assert resp.status_code == 422
    print(f"  Over max energy:     {resp.status_code} (correct)")

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("PlantIQ Cost Translator — API Endpoint Tests")
    print("=" * 60)
    print()

    test_translate_endpoint()
    test_translate_with_overrides()
    test_translate_batch_endpoint()
    test_config_endpoint()
    test_validation_errors()

    print("=" * 60)
    print("ALL 5 API TESTS PASSED")
    print("=" * 60)
