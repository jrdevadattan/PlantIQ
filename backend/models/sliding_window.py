"""
PlantIQ — Sliding Window Real-Time Forecaster (F2.3)
======================================================
Updates predictions every 30 seconds during a live batch using a
blend of model prediction and actual-data extrapolation.

Per README Component 4:
  At batch start (0% done):  →  Trust model predictions (100% model weight)
  Mid-batch (33% done):      →  Blend model + actual rate (model still dominant)
  Late batch (67%+ done):    →  Trust actual consumption rate (80% actual weight)

Blend formula:
  blend_weight = min(progress_pct * 2, 0.8)
  adjusted = (1 - blend_weight) * model_prediction + blend_weight * extrapolated

Usage:
    from models.sliding_window import SlidingWindowForecaster

    forecaster = SlidingWindowForecaster()
    result = forecaster.update(
        model_predictions={...},
        original_params={...},
        elapsed_minutes=10.0,
        energy_so_far=13.7,
        avg_power_kw=1.37,
        anomaly_events=0,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("plantiq.sliding_window")

# ── Constants ────────────────────────────────────────────────
CO2_FACTOR = 0.82                # kWh → kg CO₂e
MAX_BLEND_WEIGHT = 0.80          # Maximum trust in actual data
BLEND_RATE = 2.0                 # How quickly blend shifts to actual
ALERT_WATCH_THRESHOLD = 5.0      # % deviation → WATCH alert
ALERT_WARNING_THRESHOLD = 15.0   # % deviation → WARNING alert
MIN_CI_KWH = 0.5                # Minimum confidence interval width


@dataclass
class SlidingWindowResult:
    """Result of a sliding window prediction update.

    Attributes
    ----------
    progress_pct : float
        Batch completion percentage (0–100)
    adjusted_energy_kwh : float
        Blended energy prediction
    adjusted_quality : float
        Adjusted quality score
    adjusted_yield : float
        Adjusted yield percentage
    adjusted_performance : float
        Adjusted performance percentage
    adjusted_co2_kg : float
        CO₂ estimate from adjusted energy
    blend_weight : float
        Weight applied to actual data (0.0–0.8)
    confidence_str : str
        Human-readable confidence interval (e.g., "±2.1 kWh")
    energy_deviation_pct : float
        Percentage deviation from model prediction
    extrapolated_energy : float
        Energy extrapolated purely from actual rate
    alert_severity : Optional[str]
        None, "WATCH", or "WARNING"
    alert_message : Optional[str]
        Alert description
    recommended_action : Optional[str]
        Suggested operator response
    estimated_saving_kwh : Optional[float]
        Potential saving from action
    quality_impact_pct : Optional[float]
        Quality impact of recommended action
    """
    progress_pct: float = 0.0
    adjusted_energy_kwh: float = 0.0
    adjusted_quality: float = 0.0
    adjusted_yield: float = 0.0
    adjusted_performance: float = 0.0
    adjusted_co2_kg: float = 0.0
    blend_weight: float = 0.0
    confidence_str: str = ""
    energy_deviation_pct: float = 0.0
    extrapolated_energy: float = 0.0
    alert_severity: Optional[str] = None
    alert_message: Optional[str] = None
    recommended_action: Optional[str] = None
    estimated_saving_kwh: Optional[float] = None
    quality_impact_pct: Optional[float] = None


class SlidingWindowForecaster:
    """Real-time prediction updater using sliding window blend.

    The forecaster blends the model's pre-batch prediction with
    extrapolation from actual consumption data collected mid-batch.
    At the start, the model is trusted fully. As the batch progresses,
    actual data is trusted more (up to 80% weight).

    This is loaded once at API startup and reused for all realtime
    prediction requests.
    """

    def __init__(
        self,
        co2_factor: float = CO2_FACTOR,
        max_blend_weight: float = MAX_BLEND_WEIGHT,
        blend_rate: float = BLEND_RATE,
    ):
        self.co2_factor = co2_factor
        self.max_blend_weight = max_blend_weight
        self.blend_rate = blend_rate
        logger.info(
            "SlidingWindowForecaster initialized (blend_rate=%.1f, max_weight=%.2f)",
            self.blend_rate, self.max_blend_weight,
        )

    def compute_blend_weight(self, progress_fraction: float) -> float:
        """Compute the blend weight for actual vs. model data.

        Per README: blend_weight = min(progress_pct * 2, 0.8)
        At 0% progress → weight = 0.0 (100% model)
        At 40% progress → weight = 0.8 (20% model, 80% actual)
        At 100% progress → weight = 0.8 (capped)

        Parameters
        ----------
        progress_fraction : float
            Batch completion as fraction (0.0–1.0)

        Returns
        -------
        float
            Blend weight for actual data (0.0–0.8)
        """
        return min(progress_fraction * self.blend_rate, self.max_blend_weight)

    def extrapolate_energy(
        self,
        energy_so_far: float,
        elapsed_minutes: float,
        total_minutes: float,
    ) -> float:
        """Extrapolate total energy from current consumption rate.

        Parameters
        ----------
        energy_so_far : float
            kWh consumed so far
        elapsed_minutes : float
            Minutes since batch start
        total_minutes : float
            Expected total batch duration

        Returns
        -------
        float
            Extrapolated total energy (kWh)
        """
        if elapsed_minutes <= 0:
            return 0.0

        rate_kwh_per_min = energy_so_far / elapsed_minutes
        return rate_kwh_per_min * total_minutes

    def blend_predictions(
        self,
        model_energy: float,
        extrapolated_energy: float,
        blend_weight: float,
    ) -> float:
        """Blend model prediction with actual extrapolation.

        Parameters
        ----------
        model_energy : float
            Model's pre-batch energy prediction (kWh)
        extrapolated_energy : float
            Energy extrapolated from actual consumption
        blend_weight : float
            Weight for actual data (0.0–0.8)

        Returns
        -------
        float
            Blended energy prediction (kWh)
        """
        return (1 - blend_weight) * model_energy + blend_weight * extrapolated_energy

    def adjust_targets(
        self,
        model_quality: float,
        model_yield: float,
        model_performance: float,
        adjusted_energy: float,
        model_energy: float,
    ) -> tuple[float, float, float]:
        """Adjust quality/yield/performance based on energy deviation.

        Higher energy than expected → slightly lower quality/yield/performance.
        The relationship: over-consuming energy often means something is off
        (bearing wear, wet material, etc.) which impacts quality.

        Parameters
        ----------
        model_quality, model_yield, model_performance : float
            Original model predictions
        adjusted_energy, model_energy : float
            Adjusted and original energy predictions

        Returns
        -------
        tuple[float, float, float]
            (adjusted_quality, adjusted_yield, adjusted_performance)
        """
        energy_ratio = adjusted_energy / max(model_energy, 0.1)

        if energy_ratio > 1:
            # Higher energy → slightly lower targets
            quality = model_quality * (2 - energy_ratio)
            yield_pct = model_yield * (2 - energy_ratio)
            performance = model_performance * (2 - energy_ratio)
        else:
            # Lower or equal energy → keep model predictions
            quality = model_quality
            yield_pct = model_yield
            performance = model_performance

        return (
            round(max(quality, 0.0), 1),
            round(max(yield_pct, 0.0), 1),
            round(max(performance, 0.0), 1),
        )

    def compute_confidence(
        self,
        adjusted_energy: float,
        model_energy: float,
    ) -> str:
        """Compute confidence interval string.

        Confidence shrinks as model and actual converge (more data = less uncertainty).

        Parameters
        ----------
        adjusted_energy, model_energy : float
            Blended and original model predictions

        Returns
        -------
        str
            Human-readable confidence (e.g., "±2.1 kWh")
        """
        ci = abs(adjusted_energy - model_energy) * 0.5
        ci = max(ci, MIN_CI_KWH)
        return f"±{ci:.1f} kWh"

    def generate_alert(
        self,
        deviation_pct: float,
        adjusted_energy: float,
        conveyor_speed: float,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[float]]:
        """Generate alert based on energy deviation.

        Parameters
        ----------
        deviation_pct : float
            Percentage energy deviation from model
        adjusted_energy : float
            Current blended energy prediction
        conveyor_speed : float
            Original conveyor speed parameter

        Returns
        -------
        tuple
            (severity, message, recommended_action, estimated_saving, quality_impact)
        """
        if deviation_pct > ALERT_WARNING_THRESHOLD:
            target_speed = max(conveyor_speed - 6, 60)
            return (
                "WARNING",
                f"Energy trending {deviation_pct:.1f}% above target",
                f"Reduce conveyor speed from {conveyor_speed:.0f}% to {target_speed:.0f}%",
                round(adjusted_energy * 0.05, 1),
                -0.3,
            )
        elif deviation_pct > ALERT_WATCH_THRESHOLD:
            return (
                "WATCH",
                f"Energy trending {deviation_pct:.1f}% above target",
                None,
                None,
                None,
            )

        return None, None, None, None, None

    def update(
        self,
        model_predictions: dict,
        original_params: dict,
        elapsed_minutes: float,
        energy_so_far: float,
        avg_power_kw: float = 0.0,
        anomaly_events: int = 0,
    ) -> SlidingWindowResult:
        """Perform a full sliding window prediction update.

        This is the main method called by the /predict/realtime endpoint.

        Parameters
        ----------
        model_predictions : dict
            Model's pre-batch predictions with keys:
            quality_score, yield_pct, performance_pct, energy_kwh
        original_params : dict
            Original batch parameters (temperature, conveyor_speed, hold_time, etc.)
        elapsed_minutes : float
            Minutes elapsed since batch start
        energy_so_far : float
            kWh consumed so far
        avg_power_kw : float
            Average power draw (kW) — informational
        anomaly_events : int
            Number of anomaly events — informational

        Returns
        -------
        SlidingWindowResult
            Complete updated prediction with alerts
        """
        total_minutes = original_params.get("hold_time", 30.0)

        # Progress tracking
        progress_frac = min(elapsed_minutes / total_minutes, 1.0)
        progress_pct = round(progress_frac * 100, 1)

        # Blend weight computation
        blend_weight = self.compute_blend_weight(progress_frac)

        # Energy extrapolation from actual rate
        if elapsed_minutes > 0:
            extrapolated = self.extrapolate_energy(energy_so_far, elapsed_minutes, total_minutes)
        else:
            extrapolated = model_predictions.get("energy_kwh", 0.0)

        # Blended energy prediction
        model_energy = model_predictions.get("energy_kwh", 0.0)
        adjusted_energy = round(
            self.blend_predictions(model_energy, extrapolated, blend_weight), 1
        )

        # Adjust other targets
        adj_quality, adj_yield, adj_perf = self.adjust_targets(
            model_quality=model_predictions.get("quality_score", 0.0),
            model_yield=model_predictions.get("yield_pct", 0.0),
            model_performance=model_predictions.get("performance_pct", 0.0),
            adjusted_energy=adjusted_energy,
            model_energy=model_energy,
        )

        # CO₂ estimate
        co2 = round(adjusted_energy * self.co2_factor, 1)

        # Confidence interval
        confidence_str = self.compute_confidence(adjusted_energy, model_energy)

        # Energy deviation
        deviation_pct = (
            ((adjusted_energy - model_energy) / max(model_energy, 0.1)) * 100
        )

        # Alert logic
        conveyor_speed = original_params.get("conveyor_speed", 75.0)
        severity, message, action, saving, q_impact = self.generate_alert(
            deviation_pct, adjusted_energy, conveyor_speed
        )

        logger.debug(
            "SlidingWindow update: progress=%.1f%%, blend=%.2f, "
            "model=%.1f, extrap=%.1f, adjusted=%.1f kWh",
            progress_pct, blend_weight, model_energy, extrapolated, adjusted_energy,
        )

        return SlidingWindowResult(
            progress_pct=progress_pct,
            adjusted_energy_kwh=adjusted_energy,
            adjusted_quality=adj_quality,
            adjusted_yield=adj_yield,
            adjusted_performance=adj_perf,
            adjusted_co2_kg=co2,
            blend_weight=round(blend_weight, 4),
            confidence_str=confidence_str,
            energy_deviation_pct=round(deviation_pct, 2),
            extrapolated_energy=round(extrapolated, 1),
            alert_severity=severity,
            alert_message=message,
            recommended_action=action,
            estimated_saving_kwh=saving,
            quality_impact_pct=q_impact,
        )
