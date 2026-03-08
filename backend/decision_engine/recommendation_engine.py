"""
PlantIQ — Recommendation Engine (Decision Engine Component 3.2)
=================================================================
Per README §6 — Component 3.2:

  "Convert SHAP attribution into a specific, physical, machine-level
   instruction that an operator can follow without additional
   interpretation."

A model-level recommendation: "reduce compression force."
A machine-level recommendation: "reduce compression force on Compression
  Unit B — left panel RPM dial, from position 7 to 6, after the current
  compression cycle completes in ~6 min, saving ~3.4 kWh and reducing
  hardness from 109 N to ~95 N, within the acceptable range."

This module produces the second version by combining:
  1. Top SHAP contributor (from ShapExplainer)
  2. Machine configuration map (parameter → physical control)
  3. Quality impact model (if I change X, what happens to Y?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════
# Machine Configuration Map
# ═══════════════════════════════════════════════════════
# Maps each parameter to its physical machine control,
# adjustment direction, and operational constraints.

MACHINE_MAP = {
    "temperature": {
        "machine": "Drying Oven — Panel A",
        "control": "temperature setpoint dial (digital, °C)",
        "adjustment_step": 2.0,
        "unit": "°C",
        "optimal": 183.0,
        "safe_range": (175.0, 195.0),
        "response_time_min": 4,
        "note": "Allow 4 min for thermal stabilisation after adjustment",
    },
    "conveyor_speed": {
        "machine": "Main Conveyor — Control Station 2",
        "control": "speed potentiometer (% of rated speed)",
        "adjustment_step": 5.0,
        "unit": "%",
        "optimal": 75.0,
        "safe_range": (60.0, 95.0),
        "response_time_min": 1,
        "note": "Adjust between batch transitions to avoid product-in-transit issues",
    },
    "hold_time": {
        "machine": "Batch Timer — MES Terminal",
        "control": "batch hold duration (minutes)",
        "adjustment_step": 2.0,
        "unit": "min",
        "optimal": 18.0,
        "safe_range": (10.0, 30.0),
        "response_time_min": 0,
        "note": "Extend only if quality reading is below threshold at hold end",
    },
    "batch_size": {
        "machine": "Hopper Feed System — Weigh Scale",
        "control": "target fill weight (kg)",
        "adjustment_step": 50.0,
        "unit": "kg",
        "optimal": 500.0,
        "safe_range": (300.0, 700.0),
        "response_time_min": 0,
        "note": "Only adjustable before batch start; mid-batch changes not permitted",
    },
    "material_type": {
        "machine": "Raw Material Staging Area",
        "control": "formulation selector (Type A / B / C)",
        "adjustment_step": 1,
        "unit": "type",
        "optimal": 0,
        "safe_range": (0, 2),
        "response_time_min": 0,
        "note": "Material type is fixed per production order; not adjustable mid-batch",
    },
    "hour_of_day": {
        "machine": "Production Scheduling System",
        "control": "batch start time scheduling",
        "adjustment_step": 1,
        "unit": "hour",
        "optimal": 10,
        "safe_range": (6, 21),
        "response_time_min": 0,
        "note": "Reschedule future batches to optimal shift windows",
    },
    "operator_exp": {
        "machine": "Shift Roster System",
        "control": "operator assignment",
        "adjustment_step": 1,
        "unit": "level",
        "optimal": 2,
        "safe_range": (0, 2),
        "response_time_min": 0,
        "note": "Assign senior operator for high-complexity batches",
    },
}

# ═══════════════════════════════════════════════════════
# Quality Impact Matrix
# ═══════════════════════════════════════════════════════
# Empirical sensitivity: how much does each target change
# per unit change in each parameter?  (Derived from SHAP
# global importance and domain knowledge.)

QUALITY_IMPACT_PER_UNIT = {
    "temperature": {
        "energy_kwh": 0.8,           # +1°C → +0.8 kWh
        "quality_score": -0.3,       # +1°C above optimal → −0.3% quality
        "yield_pct": -0.1,
        "performance_pct": -0.2,
    },
    "conveyor_speed": {
        "energy_kwh": 0.4,           # +1% speed → +0.4 kWh
        "quality_score": -0.2,
        "yield_pct": -0.15,
        "performance_pct": 0.3,
    },
    "hold_time": {
        "energy_kwh": 1.2,           # +1 min → +1.2 kWh
        "quality_score": 0.4,        # longer hold → better quality (up to a point)
        "yield_pct": 0.1,
        "performance_pct": -0.5,     # longer hold → lower throughput
    },
    "batch_size": {
        "energy_kwh": 0.05,          # +1 kg → +0.05 kWh
        "quality_score": -0.01,
        "yield_pct": -0.02,
        "performance_pct": -0.01,
    },
}


# ═══════════════════════════════════════════════════════
# Output Data Classes
# ═══════════════════════════════════════════════════════

@dataclass
class Recommendation:
    """A single, actionable, machine-level recommendation."""
    rank: int                                 # 1 = most impactful
    parameter: str                            # e.g. "temperature"
    current_value: float                      # operator's input
    recommended_value: float                  # what we suggest
    adjustment: float                         # delta (signed)
    direction: str                            # "increase" | "decrease"

    # Machine-level detail
    machine: str                              # e.g. "Drying Oven — Panel A"
    control: str                              # physical control description
    instruction: str                          # plain-English step-by-step

    # Impact estimates
    estimated_energy_saving_kwh: float
    estimated_quality_impact_pct: float
    estimated_yield_impact_pct: float

    # Timing
    response_time_min: int
    timing_note: str

    # SHAP provenance
    shap_contribution: float                  # absolute SHAP value
    shap_direction: str                       # "increases_energy" etc.

    # Safety
    within_safe_range: bool
    safety_note: str


@dataclass
class RecommendationSet:
    """Complete set of recommendations for a batch prediction."""
    batch_id: Optional[str]
    target_focus: str                         # which target we're optimising
    recommendation_count: int
    total_estimated_saving_kwh: float
    total_quality_impact_pct: float
    recommendations: List[Recommendation]
    summary: str                              # plain-English summary


# ═══════════════════════════════════════════════════════
# Recommendation Engine
# ═══════════════════════════════════════════════════════

class RecommendationEngine:
    """Convert SHAP attributions into machine-level operator instructions.

    Usage:
        engine = RecommendationEngine()
        recs = engine.generate(
            input_params={"temperature": 190, "conveyor_speed": 85, ...},
            shap_contributions=[
                {"feature": "temperature", "contribution": 3.2, "direction": "increases_energy"},
                ...
            ],
            target="energy_kwh",
        )
    """

    def __init__(self):
        self.machine_map = MACHINE_MAP
        self.impact_matrix = QUALITY_IMPACT_PER_UNIT

    def generate(
        self,
        *,
        input_params: dict,
        shap_contributions: List[dict],
        target: str = "energy_kwh",
        batch_id: Optional[str] = None,
        max_recommendations: int = 3,
    ) -> RecommendationSet:
        """Generate ranked, machine-level recommendations.

        Parameters
        ----------
        input_params : dict
            The 7 operator inputs for this batch.
        shap_contributions : list[dict]
            SHAP feature contributions from ShapExplainer.
            Each dict has: feature, contribution, direction, value.
        target : str
            Which target to optimise (default: energy_kwh).
        batch_id : str | None
            Optional batch identifier for traceability.
        max_recommendations : int
            Maximum number of recommendations to return.

        Returns
        -------
        RecommendationSet
        """
        # Filter to adjustable parameters with SHAP data
        adjustable = {"temperature", "conveyor_speed", "hold_time", "batch_size"}
        actionable_contribs = [
            c for c in shap_contributions
            if c["feature"] in adjustable and abs(c.get("contribution", 0)) > 0.01
        ]

        # Sort by absolute contribution (largest impact first)
        actionable_contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        recommendations = []
        total_saving = 0.0
        total_quality_impact = 0.0

        for i, contrib in enumerate(actionable_contribs[:max_recommendations]):
            feature = contrib["feature"]
            shap_val = contrib["contribution"]
            current_val = input_params.get(feature, contrib.get("value", 0))

            rec = self._build_recommendation(
                rank=i + 1,
                feature=feature,
                current_value=current_val,
                shap_contribution=shap_val,
                shap_direction=contrib.get("direction", ""),
                target=target,
            )
            recommendations.append(rec)
            total_saving += rec.estimated_energy_saving_kwh
            total_quality_impact += rec.estimated_quality_impact_pct

        summary = self._build_summary(
            recommendations, target, total_saving, total_quality_impact
        )

        return RecommendationSet(
            batch_id=batch_id,
            target_focus=target,
            recommendation_count=len(recommendations),
            total_estimated_saving_kwh=round(total_saving, 2),
            total_quality_impact_pct=round(total_quality_impact, 2),
            recommendations=recommendations,
            summary=summary,
        )

    def _build_recommendation(
        self,
        *,
        rank: int,
        feature: str,
        current_value: float,
        shap_contribution: float,
        shap_direction: str,
        target: str,
    ) -> Recommendation:
        """Build a single machine-level recommendation."""
        machine_info = self.machine_map.get(feature, {})
        impact_info = self.impact_matrix.get(feature, {})

        optimal = machine_info.get("optimal", current_value)
        step = machine_info.get("adjustment_step", 1.0)
        safe_min, safe_max = machine_info.get("safe_range", (current_value, current_value))

        # Determine direction: should we increase or decrease?
        # For energy_kwh target: if SHAP says this feature increases energy,
        # we want to move it toward optimal (which should decrease energy)
        if "increases" in shap_direction and target == "energy_kwh":
            # Feature is pushing energy UP — move toward optimal
            if current_value > optimal:
                direction = "decrease"
                adjustment = -step
            else:
                direction = "increase"
                adjustment = step
        elif "decreases" in shap_direction and target == "energy_kwh":
            # Feature is pulling energy DOWN — keep going
            direction = "maintain" if abs(current_value - optimal) < step else (
                "decrease" if current_value > optimal else "increase"
            )
            adjustment = -step if current_value > optimal else step
        else:
            # General case: move toward optimal
            if abs(current_value - optimal) < step * 0.5:
                direction = "maintain"
                adjustment = 0
            elif current_value > optimal:
                direction = "decrease"
                adjustment = -step
            else:
                direction = "increase"
                adjustment = step

        recommended_value = current_value + adjustment
        recommended_value = max(safe_min, min(safe_max, recommended_value))
        actual_adjustment = recommended_value - current_value
        within_safe = safe_min <= recommended_value <= safe_max

        # Estimate impact
        energy_impact_per_unit = impact_info.get("energy_kwh", 0)
        quality_impact_per_unit = impact_info.get("quality_score", 0)
        yield_impact_per_unit = impact_info.get("yield_pct", 0)

        est_energy_saving = abs(actual_adjustment) * energy_impact_per_unit
        est_quality_impact = actual_adjustment * quality_impact_per_unit
        est_yield_impact = actual_adjustment * yield_impact_per_unit

        # If we're reducing a parameter that increases energy, the saving is positive
        if direction == "decrease" and "increases" in shap_direction:
            est_energy_saving = abs(est_energy_saving)
        elif direction == "increase" and "decreases" in shap_direction:
            est_energy_saving = abs(est_energy_saving)

        instruction = self._build_instruction(
            feature=feature,
            direction=direction,
            current_value=current_value,
            recommended_value=recommended_value,
            machine_info=machine_info,
            est_energy_saving=est_energy_saving,
        )

        safety_note = (
            f"Within safe range ({safe_min}–{safe_max} {machine_info.get('unit', '')})"
            if within_safe
            else f"⚠️ Outside safe range ({safe_min}–{safe_max}). Manual override required."
        )

        return Recommendation(
            rank=rank,
            parameter=feature,
            current_value=round(current_value, 2),
            recommended_value=round(recommended_value, 2),
            adjustment=round(actual_adjustment, 2),
            direction=direction,
            machine=machine_info.get("machine", "Unknown"),
            control=machine_info.get("control", "Unknown"),
            instruction=instruction,
            estimated_energy_saving_kwh=round(est_energy_saving, 2),
            estimated_quality_impact_pct=round(est_quality_impact, 2),
            estimated_yield_impact_pct=round(est_yield_impact, 2),
            response_time_min=machine_info.get("response_time_min", 0),
            timing_note=machine_info.get("note", ""),
            shap_contribution=round(shap_contribution, 4),
            shap_direction=shap_direction,
            within_safe_range=within_safe,
            safety_note=safety_note,
        )

    def _build_instruction(
        self,
        *,
        feature: str,
        direction: str,
        current_value: float,
        recommended_value: float,
        machine_info: dict,
        est_energy_saving: float,
    ) -> str:
        """Build a plain-English, machine-level instruction."""
        machine = machine_info.get("machine", "the machine")
        control = machine_info.get("control", "the control")
        unit = machine_info.get("unit", "")
        response = machine_info.get("response_time_min", 0)
        note = machine_info.get("note", "")

        if direction == "maintain":
            return (
                f"Current {feature.replace('_', ' ')} of {current_value}{unit} "
                f"on {machine} is at or near optimal. No adjustment needed."
            )

        verb = "Reduce" if direction == "decrease" else "Increase"
        delta = abs(recommended_value - current_value)

        parts = [
            f"{verb} {feature.replace('_', ' ')} on {machine} — "
            f"{control}, from {current_value}{unit} to {recommended_value}{unit} "
            f"(change of {'+' if direction == 'increase' else '-'}{delta:.1f}{unit}).",
        ]

        if est_energy_saving > 0:
            parts.append(
                f"Estimated saving: {est_energy_saving:.1f} kWh per batch."
            )

        if response > 0:
            parts.append(
                f"Allow {response} minute{'s' if response > 1 else ''} for stabilisation."
            )

        if note:
            parts.append(f"Note: {note}")

        return " ".join(parts)

    def _build_summary(
        self,
        recommendations: List[Recommendation],
        target: str,
        total_saving: float,
        total_quality_impact: float,
    ) -> str:
        """Build a plain-English summary of all recommendations."""
        if not recommendations:
            return "No actionable adjustments identified. Current parameters are near optimal."

        n = len(recommendations)
        params = ", ".join(r.parameter.replace("_", " ") for r in recommendations)

        parts = [
            f"{n} adjustment{'s' if n > 1 else ''} recommended targeting {params}.",
            f"Combined estimated energy saving: {total_saving:.1f} kWh per batch.",
        ]

        if total_quality_impact != 0:
            direction = "improvement" if total_quality_impact > 0 else "trade-off"
            parts.append(
                f"Quality {direction}: {total_quality_impact:+.1f}% "
                f"(within acceptable range)."
            )

        # Top recommendation highlight
        top = recommendations[0]
        parts.append(
            f"Priority action: {top.instruction.split('.')[0]}."
        )

        return " ".join(parts)
