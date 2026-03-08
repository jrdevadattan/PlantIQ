"""
PlantIQ — Cost Translator (Decision Engine Component 3.1)
============================================================
Per README §6 — Component 3.1:

  "One job: Convert every energy prediction from kilowatt-hours into
   Indian Rupees, compute the variance against the batch cost target,
   and project monthly cost at the current rate."

Outputs per batch:
  ✓ Predicted energy cost in ₹
  ✓ Gap against the cost target
  ✓ CO₂ emission (energy × 0.82 kg/kWh)
  ✓ Carbon variance vs. batch budget
  ✓ Monthly cost projection across planned daily batch count
  ✓ ROI calculation: if recommendations reduce energy by X%, monthly saving = ₹Y

This module is a pure function layer — no state, no side effects.
Route handlers call it after prediction to enrich the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════
# Configurable Constants (single source of truth)
# ═══════════════════════════════════════════════════════

# Electricity tariff — India industrial average (₹/kWh)
# Source: CERC tariff schedule.  Change this one number
# and all projections update automatically.
DEFAULT_TARIFF_INR_PER_KWH = 8.50

# CO₂ emission factor (kg CO₂e per kWh of grid electricity)
# Source: CEA CO₂ Baseline Database v18.0 (India grid)
CO2_FACTOR_KG_PER_KWH = 0.82

# Default planning assumptions
DEFAULT_BATCHES_PER_DAY = 8     # Typical pharma batch plant
DEFAULT_OPERATING_DAYS = 25     # Per month (excludes Sundays + holidays)

# Batch-level targets (from golden batch / regulatory baseline)
DEFAULT_ENERGY_TARGET_KWH = 42.0    # kWh per batch target
DEFAULT_CO2_BUDGET_KG = DEFAULT_ENERGY_TARGET_KWH * CO2_FACTOR_KG_PER_KWH  # 34.44 kg


# ═══════════════════════════════════════════════════════
# Output Data Classes
# ═══════════════════════════════════════════════════════

@dataclass
class CostBreakdown:
    """Complete cost translation for a single batch prediction.

    Every field has a clear physical meaning documented inline.
    """

    # ── Energy ────────────────────────────────────────
    predicted_energy_kwh: float          # Raw model prediction
    energy_target_kwh: float             # Batch-level target
    energy_variance_kwh: float           # predicted − target (positive = over)
    energy_variance_pct: float           # variance as % of target

    # ── Cost in ₹ ─────────────────────────────────────
    tariff_inr_per_kwh: float            # Applied electricity rate
    predicted_cost_inr: float            # energy × tariff
    target_cost_inr: float               # target energy × tariff
    cost_variance_inr: float             # predicted − target cost
    cost_variance_pct: float             # variance as % of target cost

    # ── Carbon ────────────────────────────────────────
    co2_kg: float                        # energy × emission factor
    co2_budget_kg: float                 # per-batch CO₂ budget
    co2_variance_kg: float               # predicted − budget
    co2_status: str                      # ON_TRACK / WARNING / OVER_BUDGET

    # ── Monthly Projection ────────────────────────────
    batches_per_day: int                 # Planning parameter
    operating_days_per_month: int        # Planning parameter
    monthly_batches: int                 # batches_per_day × operating_days
    monthly_energy_kwh: float            # total projected monthly energy
    monthly_cost_inr: float              # total projected monthly cost
    monthly_co2_kg: float                # total projected monthly emissions
    monthly_target_cost_inr: float       # what it *should* cost per month
    monthly_savings_inr: float           # target − predicted (positive = saving)

    # ── ROI ───────────────────────────────────────────
    savings_if_optimized_pct: float      # Assumed optimization headroom (%)
    potential_monthly_saving_inr: float  # monthly_cost × optimization %
    potential_annual_saving_inr: float   # monthly × 12


@dataclass
class CostTranslatorConfig:
    """All tuneable parameters for the Cost Translator.

    Change tariff, emission factor, batch count, or targets
    in one place — all downstream computations update automatically.
    """
    tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH
    co2_factor: float = CO2_FACTOR_KG_PER_KWH
    energy_target_kwh: float = DEFAULT_ENERGY_TARGET_KWH
    co2_budget_kg: float = DEFAULT_CO2_BUDGET_KG
    batches_per_day: int = DEFAULT_BATCHES_PER_DAY
    operating_days_per_month: int = DEFAULT_OPERATING_DAYS
    optimization_headroom_pct: float = 6.0   # Conservative 6% from recommendations


# ═══════════════════════════════════════════════════════
# Cost Translator — Pure Function
# ═══════════════════════════════════════════════════════

class CostTranslator:
    """Converts energy predictions to cost, carbon, and projections.

    Per README Component 3.1:
      - Energy kWh → Indian Rupees (₹)
      - Variance against batch cost target
      - CO₂ emission at 0.82 kg/kWh
      - Carbon variance
      - Monthly projection at current batch rate
      - ROI: if recommendations reduce energy by X%, saving = ₹Y

    Usage
    -----
    >>> translator = CostTranslator()
    >>> result = translator.translate(energy_kwh=45.3)
    >>> print(f"Cost: ₹{result.predicted_cost_inr:.2f}")
    >>> print(f"Monthly projection: ₹{result.monthly_cost_inr:,.0f}")
    """

    def __init__(self, config: Optional[CostTranslatorConfig] = None):
        self.config = config or CostTranslatorConfig()

    def translate(
        self,
        energy_kwh: float,
        *,
        energy_target_kwh: Optional[float] = None,
        tariff_override: Optional[float] = None,
        batches_per_day: Optional[int] = None,
        operating_days: Optional[int] = None,
    ) -> CostBreakdown:
        """Translate a single energy prediction into full cost breakdown.

        Parameters
        ----------
        energy_kwh : float
            Predicted energy consumption for the batch (kWh).
        energy_target_kwh : float, optional
            Override the default batch energy target.
        tariff_override : float, optional
            Override the configured ₹/kWh tariff.
        batches_per_day : int, optional
            Override default batch count for monthly projection.
        operating_days : int, optional
            Override default operating days per month.

        Returns
        -------
        CostBreakdown
            Complete cost, carbon, and projection analysis.
        """
        cfg = self.config

        # ── Resolve parameters (overrides > config > defaults) ──
        tariff = tariff_override or cfg.tariff_inr_per_kwh
        target = energy_target_kwh or cfg.energy_target_kwh
        bpd = batches_per_day or cfg.batches_per_day
        opd = operating_days or cfg.operating_days_per_month

        # ── Energy variance ──────────────────────────────────
        energy_var = energy_kwh - target
        energy_var_pct = (energy_var / target * 100) if target > 0 else 0.0

        # ── Cost ─────────────────────────────────────────────
        predicted_cost = energy_kwh * tariff
        target_cost = target * tariff
        cost_var = predicted_cost - target_cost
        cost_var_pct = (cost_var / target_cost * 100) if target_cost > 0 else 0.0

        # ── Carbon ───────────────────────────────────────────
        co2 = energy_kwh * cfg.co2_factor
        co2_budget = cfg.co2_budget_kg
        co2_var = co2 - co2_budget

        # CO₂ status thresholds (per README alert severity)
        if co2 <= co2_budget * 0.80:
            co2_status = "ON_TRACK"
        elif co2 <= co2_budget:
            co2_status = "WARNING"
        else:
            co2_status = "OVER_BUDGET"

        # ── Monthly projection ───────────────────────────────
        monthly_batches = bpd * opd
        monthly_energy = energy_kwh * monthly_batches
        monthly_cost = predicted_cost * monthly_batches
        monthly_co2 = co2 * monthly_batches
        monthly_target_cost = target_cost * monthly_batches
        monthly_savings = monthly_target_cost - monthly_cost  # positive = good

        # ── ROI from optimization ────────────────────────────
        opt_pct = cfg.optimization_headroom_pct
        potential_monthly = monthly_cost * (opt_pct / 100.0)
        potential_annual = potential_monthly * 12

        return CostBreakdown(
            # Energy
            predicted_energy_kwh=round(energy_kwh, 2),
            energy_target_kwh=round(target, 2),
            energy_variance_kwh=round(energy_var, 2),
            energy_variance_pct=round(energy_var_pct, 1),
            # Cost
            tariff_inr_per_kwh=round(tariff, 2),
            predicted_cost_inr=round(predicted_cost, 2),
            target_cost_inr=round(target_cost, 2),
            cost_variance_inr=round(cost_var, 2),
            cost_variance_pct=round(cost_var_pct, 1),
            # Carbon
            co2_kg=round(co2, 2),
            co2_budget_kg=round(co2_budget, 2),
            co2_variance_kg=round(co2_var, 2),
            co2_status=co2_status,
            # Monthly
            batches_per_day=bpd,
            operating_days_per_month=opd,
            monthly_batches=monthly_batches,
            monthly_energy_kwh=round(monthly_energy, 1),
            monthly_cost_inr=round(monthly_cost, 2),
            monthly_co2_kg=round(monthly_co2, 1),
            monthly_target_cost_inr=round(monthly_target_cost, 2),
            monthly_savings_inr=round(monthly_savings, 2),
            # ROI
            savings_if_optimized_pct=round(opt_pct, 1),
            potential_monthly_saving_inr=round(potential_monthly, 2),
            potential_annual_saving_inr=round(potential_annual, 2),
        )

    def translate_multiple(
        self,
        energy_values: list[float],
        **kwargs,
    ) -> list[CostBreakdown]:
        """Translate multiple energy predictions at once."""
        return [self.translate(e, **kwargs) for e in energy_values]

    def summary_text(self, breakdown: CostBreakdown) -> str:
        """Generate a human-readable cost summary for the operator.

        Returns a plain-English paragraph suitable for display in the
        dashboard or inclusion in a shift report.
        """
        b = breakdown
        lines = []

        # ── Headline ─────────────────────────────────────────
        if b.energy_variance_kwh <= 0:
            lines.append(
                f"This batch is predicted to consume {b.predicted_energy_kwh:.1f} kWh, "
                f"which is {abs(b.energy_variance_kwh):.1f} kWh BELOW the target of "
                f"{b.energy_target_kwh:.1f} kWh. Good."
            )
        else:
            lines.append(
                f"This batch is predicted to consume {b.predicted_energy_kwh:.1f} kWh, "
                f"which is {b.energy_variance_kwh:.1f} kWh ABOVE the target of "
                f"{b.energy_target_kwh:.1f} kWh ({b.energy_variance_pct:+.1f}%)."
            )

        # ── Cost ─────────────────────────────────────────────
        lines.append(
            f"Estimated cost: ₹{b.predicted_cost_inr:.2f} "
            f"(target: ₹{b.target_cost_inr:.2f}, "
            f"variance: ₹{b.cost_variance_inr:+.2f})."
        )

        # ── Carbon ───────────────────────────────────────────
        lines.append(
            f"CO₂ emissions: {b.co2_kg:.1f} kg "
            f"(budget: {b.co2_budget_kg:.1f} kg, status: {b.co2_status})."
        )

        # ── Monthly ──────────────────────────────────────────
        lines.append(
            f"At {b.batches_per_day} batches/day over {b.operating_days_per_month} "
            f"operating days, monthly projection: ₹{b.monthly_cost_inr:,.0f} "
            f"({b.monthly_energy_kwh:,.0f} kWh, {b.monthly_co2_kg:,.0f} kg CO₂)."
        )

        # ── ROI hint ─────────────────────────────────────────
        if b.potential_monthly_saving_inr > 0:
            lines.append(
                f"If recommendations reduce energy by {b.savings_if_optimized_pct:.0f}%, "
                f"monthly saving: ₹{b.potential_monthly_saving_inr:,.0f} "
                f"(₹{b.potential_annual_saving_inr:,.0f}/year)."
            )

        return " ".join(lines)
