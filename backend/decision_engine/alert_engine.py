"""
PlantIQ — Alert Engine (Decision Engine Component 3.3)
========================================================
Per README §6 — Component 3.3:

  "Monitor anomaly scores and forecast deviations, fire alerts at the
   correct severity level, and ensure every alert contains a complete,
   structured record."

Thresholds:
  • Forecast >15% over energy target → WARNING
  • Forecast >25% over target, OR anomaly score >0.60 → CRITICAL

Every alert record contains:
  alert_id, batch_id, timestamp, type, severity, plain-English
  description, technical detail, root cause from SHAP, specific
  physical action (from RecommendationEngine), estimated saving,
  quality impact.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════
# Thresholds (per README §6 Component 3.3)
# ═══════════════════════════════════════════════════════

ENERGY_WARNING_PCT = 15.0      # >15% over target → WARNING
ENERGY_CRITICAL_PCT = 25.0     # >25% over target → CRITICAL
ANOMALY_CRITICAL_SCORE = 0.60  # anomaly score >0.60 → CRITICAL
ANOMALY_WARNING_SCORE = 0.30   # anomaly score >0.30 → WARNING
ANOMALY_WATCH_SCORE = 0.15     # anomaly score >0.15 → WATCH

QUALITY_WARNING_THRESHOLD = 80.0   # predicted quality <80 → WARNING
QUALITY_CRITICAL_THRESHOLD = 70.0  # predicted quality <70 → CRITICAL


# ═══════════════════════════════════════════════════════
# Output Data Class
# ═══════════════════════════════════════════════════════

@dataclass
class AlertRecord:
    """Complete structured alert per README spec."""
    alert_id: str
    batch_id: str
    timestamp: str
    alert_type: str             # energy_overrun, anomaly, quality_risk, drift
    severity: str               # WATCH, WARNING, CRITICAL
    message: str                # plain-English description
    technical_detail: str       # detection method and values
    root_cause: Optional[str]   # from SHAP attribution
    recommended_action: Optional[str]  # machine-level instruction
    estimated_saving_kwh: Optional[float]
    quality_impact_pct: Optional[float]


# ═══════════════════════════════════════════════════════
# Alert Engine
# ═══════════════════════════════════════════════════════

class AlertEngine:
    """Monitor predictions and anomaly scores, fire structured alerts.

    Usage:
        engine = AlertEngine()

        # Check predictions for energy/quality alerts
        alerts = engine.check_predictions(
            batch_id="B-001",
            predictions={"energy_kwh": 48.0, "quality_score": 75.0, ...},
            energy_target_kwh=42.0,
            shap_top_feature={"feature": "temperature", "contribution": 3.2},
            recommendation="Reduce temperature on Drying Oven...",
        )

        # Check anomaly score
        alerts += engine.check_anomaly(
            batch_id="B-001",
            anomaly_score=0.72,
            fault_type="bearing_wear",
            fault_detail="Gradual baseline rise detected",
            recommendation="Schedule maintenance within 5 days",
        )
    """

    def __init__(
        self,
        *,
        energy_target_kwh: float = 42.0,
    ):
        self.energy_target_kwh = energy_target_kwh

    def check_predictions(
        self,
        *,
        batch_id: str,
        predictions: dict,
        energy_target_kwh: Optional[float] = None,
        shap_top_feature: Optional[dict] = None,
        recommendation: Optional[str] = None,
        estimated_saving_kwh: Optional[float] = None,
        quality_impact_pct: Optional[float] = None,
    ) -> List[AlertRecord]:
        """Check predicted values against thresholds and generate alerts.

        Returns list of AlertRecord (may be empty if everything is normal).
        """
        target = energy_target_kwh or self.energy_target_kwh
        alerts: List[AlertRecord] = []

        # ── Energy overrun check ─────────────────────────
        energy = predictions.get("energy_kwh", 0)
        if target > 0:
            deviation_pct = ((energy - target) / target) * 100

            if deviation_pct > ENERGY_CRITICAL_PCT:
                alerts.append(self._build_alert(
                    batch_id=batch_id,
                    alert_type="energy_overrun",
                    severity="CRITICAL",
                    message=(
                        f"Energy prediction {energy:.1f} kWh exceeds target "
                        f"{target:.1f} kWh by {deviation_pct:.1f}% — "
                        f"immediate intervention recommended."
                    ),
                    technical_detail=(
                        f"Predicted: {energy:.1f} kWh | Target: {target:.1f} kWh | "
                        f"Deviation: +{deviation_pct:.1f}% (threshold: {ENERGY_CRITICAL_PCT}%)"
                    ),
                    shap_top_feature=shap_top_feature,
                    recommendation=recommendation,
                    estimated_saving_kwh=estimated_saving_kwh,
                    quality_impact_pct=quality_impact_pct,
                ))
            elif deviation_pct > ENERGY_WARNING_PCT:
                alerts.append(self._build_alert(
                    batch_id=batch_id,
                    alert_type="energy_overrun",
                    severity="WARNING",
                    message=(
                        f"Energy prediction {energy:.1f} kWh exceeds target "
                        f"{target:.1f} kWh by {deviation_pct:.1f}%. "
                        f"Consider parameter adjustment."
                    ),
                    technical_detail=(
                        f"Predicted: {energy:.1f} kWh | Target: {target:.1f} kWh | "
                        f"Deviation: +{deviation_pct:.1f}% (threshold: {ENERGY_WARNING_PCT}%)"
                    ),
                    shap_top_feature=shap_top_feature,
                    recommendation=recommendation,
                    estimated_saving_kwh=estimated_saving_kwh,
                    quality_impact_pct=quality_impact_pct,
                ))

        # ── Quality risk check ───────────────────────────
        quality = predictions.get("quality_score", 100)
        if quality < QUALITY_CRITICAL_THRESHOLD:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="quality_risk",
                severity="CRITICAL",
                message=(
                    f"Predicted quality {quality:.1f}% is below critical "
                    f"threshold ({QUALITY_CRITICAL_THRESHOLD}%). "
                    f"Batch may fail QC inspection."
                ),
                technical_detail=(
                    f"Predicted quality: {quality:.1f}% | "
                    f"Critical threshold: {QUALITY_CRITICAL_THRESHOLD}%"
                ),
                shap_top_feature=shap_top_feature,
                recommendation=recommendation,
            ))
        elif quality < QUALITY_WARNING_THRESHOLD:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="quality_risk",
                severity="WARNING",
                message=(
                    f"Predicted quality {quality:.1f}% is below warning "
                    f"threshold ({QUALITY_WARNING_THRESHOLD}%). "
                    f"Monitor closely."
                ),
                technical_detail=(
                    f"Predicted quality: {quality:.1f}% | "
                    f"Warning threshold: {QUALITY_WARNING_THRESHOLD}%"
                ),
                shap_top_feature=shap_top_feature,
                recommendation=recommendation,
            ))

        return alerts

    def check_anomaly(
        self,
        *,
        batch_id: str,
        anomaly_score: float,
        fault_type: Optional[str] = None,
        fault_detail: Optional[str] = None,
        recommendation: Optional[str] = None,
        estimated_saving_kwh: Optional[float] = None,
        quality_impact_pct: Optional[float] = None,
    ) -> List[AlertRecord]:
        """Check anomaly score and generate alerts if above thresholds."""
        alerts: List[AlertRecord] = []

        if anomaly_score >= ANOMALY_CRITICAL_SCORE:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="anomaly",
                severity="CRITICAL",
                message=(
                    f"Anomaly score {anomaly_score:.2f} indicates critical "
                    f"deviation from normal power curve. "
                    f"Fault type: {fault_type or 'unknown'}. "
                    f"Immediate investigation required."
                ),
                technical_detail=(
                    f"Anomaly score: {anomaly_score:.4f} | "
                    f"Critical threshold: {ANOMALY_CRITICAL_SCORE} | "
                    f"Fault type: {fault_type or 'unclassified'} | "
                    f"Detail: {fault_detail or 'N/A'}"
                ),
                recommendation=recommendation,
                estimated_saving_kwh=estimated_saving_kwh,
                quality_impact_pct=quality_impact_pct,
            ))
        elif anomaly_score >= ANOMALY_WARNING_SCORE:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="anomaly",
                severity="WARNING",
                message=(
                    f"Anomaly score {anomaly_score:.2f} above warning threshold. "
                    f"Fault type: {fault_type or 'unknown'}. "
                    f"Investigation recommended."
                ),
                technical_detail=(
                    f"Anomaly score: {anomaly_score:.4f} | "
                    f"Warning threshold: {ANOMALY_WARNING_SCORE} | "
                    f"Fault type: {fault_type or 'unclassified'}"
                ),
                recommendation=recommendation,
            ))
        elif anomaly_score >= ANOMALY_WATCH_SCORE:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="anomaly",
                severity="WATCH",
                message=(
                    f"Anomaly score {anomaly_score:.2f} slightly elevated. "
                    f"Monitor next batches for trend."
                ),
                technical_detail=(
                    f"Anomaly score: {anomaly_score:.4f} | "
                    f"Watch threshold: {ANOMALY_WATCH_SCORE}"
                ),
            ))

        return alerts

    def check_drift(
        self,
        *,
        batch_id: str,
        target_name: str,
        rolling_mape: float,
        consecutive_degraded: int,
    ) -> List[AlertRecord]:
        """Generate alert if model drift is detected."""
        alerts: List[AlertRecord] = []

        if consecutive_degraded >= 10:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="drift",
                severity="CRITICAL",
                message=(
                    f"Model drift detected for {target_name}: "
                    f"{consecutive_degraded} consecutive batches exceed 10% MAPE "
                    f"(rolling MAPE: {rolling_mape:.1f}%). "
                    f"Retraining recommended."
                ),
                technical_detail=(
                    f"Target: {target_name} | Rolling MAPE: {rolling_mape:.2f}% | "
                    f"Consecutive degraded: {consecutive_degraded} | Threshold: 10"
                ),
                recommendation="Flag model for retraining review. Current model may be unreliable.",
            ))
        elif rolling_mape > 10.0:
            alerts.append(self._build_alert(
                batch_id=batch_id,
                alert_type="drift",
                severity="WARNING",
                message=(
                    f"Accuracy degradation for {target_name}: "
                    f"rolling MAPE {rolling_mape:.1f}% exceeds 10% threshold."
                ),
                technical_detail=(
                    f"Target: {target_name} | Rolling MAPE: {rolling_mape:.2f}% | "
                    f"Consecutive degraded: {consecutive_degraded}"
                ),
            ))

        return alerts

    def _build_alert(
        self,
        *,
        batch_id: str,
        alert_type: str,
        severity: str,
        message: str,
        technical_detail: str,
        shap_top_feature: Optional[dict] = None,
        recommendation: Optional[str] = None,
        estimated_saving_kwh: Optional[float] = None,
        quality_impact_pct: Optional[float] = None,
    ) -> AlertRecord:
        """Construct a complete alert record."""
        root_cause = None
        if shap_top_feature:
            feat = shap_top_feature.get("feature", "unknown")
            contrib = shap_top_feature.get("contribution", 0)
            root_cause = (
                f"Top SHAP contributor: {feat} "
                f"(contribution: {contrib:+.2f})"
            )

        return AlertRecord(
            alert_id=f"ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4].upper()}",
            batch_id=batch_id,
            timestamp=datetime.utcnow().isoformat(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            technical_detail=technical_detail,
            root_cause=root_cause,
            recommended_action=recommendation,
            estimated_saving_kwh=estimated_saving_kwh,
            quality_impact_pct=quality_impact_pct,
        )
