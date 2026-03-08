"""
PlantIQ — Cost Translator Test Suite
======================================
Run: python3 test_cost_translator.py
"""

import sys
import os

# Ensure backend is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_engine.cost_translator import CostTranslator, CostTranslatorConfig


def test_below_target():
    """Batch consuming less than target energy."""
    t = CostTranslator()
    r = t.translate(energy_kwh=38.5)

    print("=== Test 1: Below target (38.5 kWh) ===")
    print(f"  Energy:  {r.predicted_energy_kwh} kWh  (target: {r.energy_target_kwh})")
    print(f"  Cost:    INR {r.predicted_cost_inr}  (target: INR {r.target_cost_inr})")
    print(f"  Var:     INR {r.cost_variance_inr} ({r.cost_variance_pct}%)")
    print(f"  CO2:     {r.co2_kg} kg  (budget: {r.co2_budget_kg}, status: {r.co2_status})")
    print(f"  Monthly: INR {r.monthly_cost_inr:,.0f}  ({r.monthly_batches} batches)")
    print(f"  ROI:     INR {r.potential_monthly_saving_inr:,.0f}/month")
    print()

    assert r.energy_variance_kwh < 0, "Should be below target"
    assert r.cost_variance_inr < 0, "Cost variance should be negative (saving)"
    # 38.5 kWh * 0.82 = 31.57 kg, budget = 34.44 → 91.6% → WARNING (>80%)
    assert r.co2_status == "WARNING", f"Expected WARNING, got {r.co2_status}"
    assert r.monthly_batches == 200, f"Expected 200, got {r.monthly_batches}"
    print("  PASSED\n")


def test_above_target():
    """Batch consuming more than target energy."""
    t = CostTranslator()
    r = t.translate(energy_kwh=48.0)

    print("=== Test 2: Above target (48.0 kWh) ===")
    print(f"  Energy:  {r.predicted_energy_kwh} kWh  (target: {r.energy_target_kwh})")
    print(f"  Cost:    INR {r.predicted_cost_inr}  (target: INR {r.target_cost_inr})")
    print(f"  Var:     INR {r.cost_variance_inr} ({r.cost_variance_pct}%)")
    print(f"  CO2:     {r.co2_kg} kg  (status: {r.co2_status})")
    print()

    assert r.energy_variance_kwh > 0, "Should be above target"
    assert r.cost_variance_inr > 0, "Cost variance should be positive (overspend)"
    # 48.0 kWh * 0.82 = 39.36 kg, budget = 34.44 → 114% → OVER_BUDGET
    assert r.co2_status == "OVER_BUDGET", f"Expected OVER_BUDGET, got {r.co2_status}"
    print("  PASSED\n")


def test_what_if_scenario():
    """What-if: different tariff and batch count."""
    t = CostTranslator()
    r = t.translate(energy_kwh=45.0, tariff_override=10.0, batches_per_day=12)

    print("=== Test 3: What-if scenario (INR 10/kWh, 12 batches/day) ===")
    print(f"  Cost:    INR {r.predicted_cost_inr}  (tariff: INR {r.tariff_inr_per_kwh}/kWh)")
    print(f"  Monthly: INR {r.monthly_cost_inr:,.0f}  ({r.monthly_batches} batches)")
    print(f"  Annual potential saving: INR {r.potential_annual_saving_inr:,.0f}")
    print()

    assert r.tariff_inr_per_kwh == 10.0, "Tariff override not applied"
    assert r.monthly_batches == 300, f"Expected 300 (12*25), got {r.monthly_batches}"
    assert r.predicted_cost_inr == 450.0, f"Expected 450, got {r.predicted_cost_inr}"
    print("  PASSED\n")


def test_summary_text():
    """Plain English summary generation."""
    t = CostTranslator()

    # Over target
    r = t.translate(energy_kwh=48.0)
    summary = t.summary_text(r)
    print("=== Test 4: Summary text (over target) ===")
    print(f"  {summary}")
    print()

    assert "ABOVE" in summary, "Should mention above target"
    assert "INR" in summary or "₹" in summary, "Should mention cost in rupees"
    assert "CO" in summary, "Should mention CO2"

    # Below target
    r2 = t.translate(energy_kwh=30.0)
    summary2 = t.summary_text(r2)
    print(f"  {summary2}")
    print()

    assert "BELOW" in summary2, "Should mention below target"
    print("  PASSED\n")


def test_edge_cases():
    """Edge cases: zero energy, exact target, boundary values."""
    t = CostTranslator()

    # Zero energy
    r = t.translate(energy_kwh=0.0)
    print("=== Test 5: Edge cases ===")
    print(f"  Zero kWh:  cost=INR {r.predicted_cost_inr}, co2={r.co2_kg}")
    assert r.predicted_cost_inr == 0.0
    assert r.co2_kg == 0.0
    assert r.co2_status == "ON_TRACK"

    # Exact target
    r2 = t.translate(energy_kwh=42.0)
    print(f"  Exact target: var=INR {r2.cost_variance_inr}, co2_status={r2.co2_status}")
    assert r2.energy_variance_kwh == 0.0
    assert r2.cost_variance_inr == 0.0

    print()
    print("  PASSED\n")


def test_custom_config():
    """Custom configuration."""
    cfg = CostTranslatorConfig(
        tariff_inr_per_kwh=12.0,
        co2_factor=0.90,
        energy_target_kwh=50.0,
        batches_per_day=10,
        operating_days_per_month=22,
    )
    t = CostTranslator(config=cfg)
    r = t.translate(energy_kwh=45.0)

    print("=== Test 6: Custom config ===")
    print(f"  Tariff: INR {r.tariff_inr_per_kwh}/kWh")
    print(f"  Target: {r.energy_target_kwh} kWh")
    print(f"  CO2:    {r.co2_kg} kg  (factor: 0.90)")
    print(f"  Monthly batches: {r.monthly_batches}  (10 * 22 = 220)")
    print()

    assert r.tariff_inr_per_kwh == 12.0
    assert r.co2_kg == 40.5  # 45 * 0.90
    assert r.monthly_batches == 220
    assert r.energy_variance_kwh < 0  # 45 < 50 target
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("PlantIQ Cost Translator — Test Suite")
    print("=" * 60)
    print()

    test_below_target()
    test_above_target()
    test_what_if_scenario()
    test_summary_text()
    test_edge_cases()
    test_custom_config()

    print("=" * 60)
    print("ALL 6 TESTS PASSED")
    print("=" * 60)
