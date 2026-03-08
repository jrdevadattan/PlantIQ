"""
PlantIQ — Confidence Scorer (Decision Engine Component 3.5)
=============================================================
Per README §6 — Component 3.5:

  "Multi-factor confidence score for every prediction.
   Starts at base 0.95. Reductions:
     • Input out-of-distribution   −35%
     • Model drift detected        −10%
     • Feature computation issue   −15%

   Produces green / amber / red indicator:
     > 0.85 → green (high)
     0.60–0.85 → amber (medium)
     < 0.60 → red (low)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

BASE_CONFIDENCE = 0.95

# Penalty weights (multiplicative factors)
PENALTY_OOD = 0.35             # per OOD field, max 1 field penalty applied
PENALTY_DRIFT = 0.10           # if drift detected
PENALTY_FEATURE_ISSUE = 0.15   # if feature computation failed

# Indicator thresholds
GREEN_THRESHOLD = 0.85
AMBER_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════
# Output Data Class
# ═══════════════════════════════════════════════════════

@dataclass
class ConfidenceReport:
    """Structured confidence assessment for a prediction."""
    score: float                      # 0.0–1.0
    indicator: str                    # "green", "amber", "red"
    label: str                        # "High", "Medium", "Low"
    penalties_applied: List[str]      # human-readable descriptions
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "indicator": self.indicator,
            "label": self.label,
            "penalties_applied": self.penalties_applied,
            "details": {k: round(v, 4) for k, v in self.details.items()},
        }


# ═══════════════════════════════════════════════════════
# Confidence Scorer
# ═══════════════════════════════════════════════════════

class ConfidenceScorer:
    """Multi-factor confidence scoring for predictions.

    Usage:
        scorer = ConfidenceScorer()
        report = scorer.compute(
            ood_fields=["temperature", "hold_time"],
            drift_detected=False,
            feature_issue=False,
        )

        print(report.score)      # e.g. 0.6175
        print(report.indicator)  # "amber"
        print(report.label)      # "Medium"
    """

    def __init__(self, *, base_confidence: float = BASE_CONFIDENCE):
        self.base_confidence = base_confidence

    def compute(
        self,
        *,
        ood_fields: Optional[List[str]] = None,
        drift_detected: bool = False,
        feature_issue: bool = False,
    ) -> ConfidenceReport:
        """Compute multi-factor confidence score.

        Args:
            ood_fields: Fields outside the training distribution.
            drift_detected: True if feedback loop flagged drift.
            feature_issue: True if derived features couldn't be computed.

        Returns:
            ConfidenceReport with score, indicator, penalties.
        """
        score = self.base_confidence
        penalties: List[str] = []
        details: Dict[str, float] = {"base": self.base_confidence}

        # ── Penalty: Out-of-Distribution inputs ─────────
        if ood_fields:
            n_ood = len(ood_fields)
            # Scale penalty: first field gets full weight, subsequent fields
            # get diminishing penalty (cap at ~50% total OOD reduction)
            ood_penalty = min(PENALTY_OOD * (1 + 0.3 * (n_ood - 1)), 0.50)
            score -= ood_penalty
            penalties.append(
                f"Input OOD ({n_ood} field{'s' if n_ood > 1 else ''}: "
                f"{', '.join(ood_fields)}) → −{ood_penalty:.0%}"
            )
            details["ood_penalty"] = ood_penalty

        # ── Penalty: Model drift ────────────────────────
        if drift_detected:
            score -= PENALTY_DRIFT
            penalties.append(f"Model drift detected → −{PENALTY_DRIFT:.0%}")
            details["drift_penalty"] = PENALTY_DRIFT

        # ── Penalty: Feature computation issue ──────────
        if feature_issue:
            score -= PENALTY_FEATURE_ISSUE
            penalties.append(f"Feature computation issue → −{PENALTY_FEATURE_ISSUE:.0%}")
            details["feature_penalty"] = PENALTY_FEATURE_ISSUE

        # ── Clamp to [0, 1] ────────────────────────────
        score = max(0.0, min(1.0, score))
        details["final_score"] = score

        # ── Determine indicator ─────────────────────────
        indicator, label = self._classify(score)

        return ConfidenceReport(
            score=score,
            indicator=indicator,
            label=label,
            penalties_applied=penalties,
            details=details,
        )

    def _classify(self, score: float):
        """Map score to indicator + label."""
        if score >= GREEN_THRESHOLD:
            return "green", "High"
        elif score >= AMBER_THRESHOLD:
            return "amber", "Medium"
        else:
            return "red", "Low"
