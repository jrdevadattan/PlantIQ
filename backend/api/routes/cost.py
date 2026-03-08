"""
PlantIQ — Cost Translation Routes (Decision Engine Layer 3)
==============================================================
POST /cost/translate        — Translate energy kWh to ₹, CO₂, projections
POST /cost/translate-batch  — Translate a prediction response in-place
GET  /cost/config           — Current tariff and planning configuration

Per README §6 Component 3.1:
  "Convert every energy prediction from kilowatt-hours into Indian Rupees,
   compute the variance against the batch cost target, and project monthly
   cost at the current rate."
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from decision_engine.cost_translator import (
    CostTranslator,
    CostTranslatorConfig,
    CostBreakdown,
)


router = APIRouter(prefix="/cost", tags=["cost-translator"])

# ── Module-level translator instance (default config) ────────
_translator = CostTranslator()


# ═══════════════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════════════

class CostTranslateRequest(BaseModel):
    """POST /cost/translate request.

    Minimum: just energy_kwh.  Optional overrides let the
    frontend or operator explore what-if scenarios.
    """
    energy_kwh: float = Field(
        ..., ge=0, le=200,
        description="Predicted or actual energy consumption (kWh)",
    )
    energy_target_kwh: Optional[float] = Field(
        default=None, ge=0, le=200,
        description="Override batch energy target (kWh). Uses default if omitted.",
    )
    tariff_inr_per_kwh: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Override electricity tariff (₹/kWh). Uses default if omitted.",
    )
    batches_per_day: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Override daily batch count for monthly projection.",
    )
    operating_days_per_month: Optional[int] = Field(
        default=None, ge=1, le=31,
        description="Override operating days per month.",
    )


class CostTranslateResponse(BaseModel):
    """POST /cost/translate response — full cost breakdown."""

    # Energy
    predicted_energy_kwh: float
    energy_target_kwh: float
    energy_variance_kwh: float
    energy_variance_pct: float

    # Cost in ₹
    tariff_inr_per_kwh: float
    predicted_cost_inr: float
    target_cost_inr: float
    cost_variance_inr: float
    cost_variance_pct: float

    # Carbon
    co2_kg: float
    co2_budget_kg: float
    co2_variance_kg: float
    co2_status: str

    # Monthly projection
    batches_per_day: int
    operating_days_per_month: int
    monthly_batches: int
    monthly_energy_kwh: float
    monthly_cost_inr: float
    monthly_co2_kg: float
    monthly_target_cost_inr: float
    monthly_savings_inr: float

    # ROI
    savings_if_optimized_pct: float
    potential_monthly_saving_inr: float
    potential_annual_saving_inr: float

    # Plain English
    summary: str


class CostConfigResponse(BaseModel):
    """GET /cost/config response — current translator configuration."""
    tariff_inr_per_kwh: float
    co2_factor_kg_per_kwh: float
    energy_target_kwh: float
    co2_budget_kg: float
    batches_per_day: int
    operating_days_per_month: int
    optimization_headroom_pct: float


class BatchCostRequest(BaseModel):
    """POST /cost/translate-batch — translate from a prediction response."""
    batch_id: str = Field(..., description="Batch identifier")
    energy_kwh: float = Field(..., ge=0, description="Predicted energy (kWh)")
    quality_score: Optional[float] = Field(default=None, description="Predicted quality")
    yield_pct: Optional[float] = Field(default=None, description="Predicted yield")
    performance_pct: Optional[float] = Field(default=None, description="Predicted performance")


class BatchCostResponse(BaseModel):
    """POST /cost/translate-batch response — enriched with cost data."""
    batch_id: str
    energy_kwh: float
    cost_inr: float
    co2_kg: float
    co2_status: str
    cost_variance_inr: float
    monthly_projection_inr: float
    summary: str


# ═══════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════

@router.post("/translate", response_model=CostTranslateResponse)
async def translate_cost(request: CostTranslateRequest) -> CostTranslateResponse:
    """Translate an energy prediction into full cost, carbon, and projections.

    This is the primary Cost Translator endpoint (README Component 3.1).
    Accepts energy in kWh, returns ₹ cost, CO₂, monthly projection, and ROI.

    Optional overrides let operators explore what-if scenarios:
    - "What if the tariff increases to ₹10/kWh?"
    - "What if we run 12 batches/day instead of 8?"
    """
    try:
        breakdown = _translator.translate(
            energy_kwh=request.energy_kwh,
            energy_target_kwh=request.energy_target_kwh,
            tariff_override=request.tariff_inr_per_kwh,
            batches_per_day=request.batches_per_day,
            operating_days=request.operating_days_per_month,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost translation failed: {str(e)}")

    summary = _translator.summary_text(breakdown)

    return CostTranslateResponse(
        # Energy
        predicted_energy_kwh=breakdown.predicted_energy_kwh,
        energy_target_kwh=breakdown.energy_target_kwh,
        energy_variance_kwh=breakdown.energy_variance_kwh,
        energy_variance_pct=breakdown.energy_variance_pct,
        # Cost
        tariff_inr_per_kwh=breakdown.tariff_inr_per_kwh,
        predicted_cost_inr=breakdown.predicted_cost_inr,
        target_cost_inr=breakdown.target_cost_inr,
        cost_variance_inr=breakdown.cost_variance_inr,
        cost_variance_pct=breakdown.cost_variance_pct,
        # Carbon
        co2_kg=breakdown.co2_kg,
        co2_budget_kg=breakdown.co2_budget_kg,
        co2_variance_kg=breakdown.co2_variance_kg,
        co2_status=breakdown.co2_status,
        # Monthly
        batches_per_day=breakdown.batches_per_day,
        operating_days_per_month=breakdown.operating_days_per_month,
        monthly_batches=breakdown.monthly_batches,
        monthly_energy_kwh=breakdown.monthly_energy_kwh,
        monthly_cost_inr=breakdown.monthly_cost_inr,
        monthly_co2_kg=breakdown.monthly_co2_kg,
        monthly_target_cost_inr=breakdown.monthly_target_cost_inr,
        monthly_savings_inr=breakdown.monthly_savings_inr,
        # ROI
        savings_if_optimized_pct=breakdown.savings_if_optimized_pct,
        potential_monthly_saving_inr=breakdown.potential_monthly_saving_inr,
        potential_annual_saving_inr=breakdown.potential_annual_saving_inr,
        # Summary
        summary=summary,
    )


@router.post("/translate-batch", response_model=BatchCostResponse)
async def translate_batch_cost(request: BatchCostRequest) -> BatchCostResponse:
    """Quick cost enrichment for a batch prediction.

    Lighter endpoint — takes a batch_id + energy_kwh and returns
    the essential cost fields needed by the dashboard.
    """
    try:
        breakdown = _translator.translate(energy_kwh=request.energy_kwh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost translation failed: {str(e)}")

    summary = _translator.summary_text(breakdown)

    return BatchCostResponse(
        batch_id=request.batch_id,
        energy_kwh=breakdown.predicted_energy_kwh,
        cost_inr=breakdown.predicted_cost_inr,
        co2_kg=breakdown.co2_kg,
        co2_status=breakdown.co2_status,
        cost_variance_inr=breakdown.cost_variance_inr,
        monthly_projection_inr=breakdown.monthly_cost_inr,
        summary=summary,
    )


@router.get("/config", response_model=CostConfigResponse)
async def get_cost_config() -> CostConfigResponse:
    """Return the current Cost Translator configuration.

    Useful for the frontend to display the active tariff,
    CO₂ factor, and planning assumptions.
    """
    cfg = _translator.config
    return CostConfigResponse(
        tariff_inr_per_kwh=cfg.tariff_inr_per_kwh,
        co2_factor_kg_per_kwh=cfg.co2_factor,
        energy_target_kwh=cfg.energy_target_kwh,
        co2_budget_kg=cfg.co2_budget_kg,
        batches_per_day=cfg.batches_per_day,
        operating_days_per_month=cfg.operating_days_per_month,
        optimization_headroom_pct=cfg.optimization_headroom_pct,
    )
