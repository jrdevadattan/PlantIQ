"""
PlantIQ — Input Validator (Decision Engine Component 3.4)
==========================================================
Per README §6 — Component 3.4:

  "Every prediction request must pass 4 sequential validation gates
   before reaching the model."

Gates (sequential):
  1. Presence  — all 7 required fields exist
  2. Type      — all values are numeric
  3. Physical  — within machine operating limits (hard stop)
  4. Training  — within model training distribution (soft warning)

Gates 1–3 are HARD STOPS → reject with 400 + details.
Gate 4 is a SOFT WARNING → allow but flag + reduce confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════
# Required Fields (per README §8 Domain Knowledge)
# ═══════════════════════════════════════════════════════

REQUIRED_FIELDS = [
    "temperature",
    "conveyor_speed",
    "hold_time",
    "batch_size",
    "material_type",
    "hour_of_day",
    "operator_exp",
]

# ═══════════════════════════════════════════════════════
# Physical limits: absolute machine operating range
# (hard stop — request is rejected if outside these)
# ═══════════════════════════════════════════════════════

PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "temperature":    (100.0, 250.0),    # °C — machine range
    "conveyor_speed": (10.0, 100.0),     # % — motor range
    "hold_time":      (1.0, 60.0),       # minutes — physical max
    "batch_size":     (50.0, 1500.0),    # kg — hopper capacity
    "material_type":  (0, 2),            # enum 0/1/2
    "hour_of_day":    (0, 23),           # clock hours
    "operator_exp":   (0, 2),            # enum 0/1/2
}

# ═══════════════════════════════════════════════════════
# Training distribution: model's training range
# (soft warning — request is allowed but flagged)
# ═══════════════════════════════════════════════════════

TRAINING_RANGE: Dict[str, Tuple[float, float]] = {
    "temperature":    (175.0, 195.0),
    "conveyor_speed": (60.0, 95.0),
    "hold_time":      (10.0, 30.0),
    "batch_size":     (300.0, 700.0),
    "material_type":  (0, 2),
    "hour_of_day":    (6, 21),
    "operator_exp":   (0, 2),
}


# ═══════════════════════════════════════════════════════
# Output Data Classes
# ═══════════════════════════════════════════════════════

@dataclass
class ValidationError:
    """A single validation failure."""
    gate: str          # "presence", "type", "physical", "training"
    field: str         # which field failed
    message: str       # human-readable description
    severity: str      # "error" (hard stop) or "warning" (soft)


@dataclass
class ValidationResult:
    """Complete validation report for an input request."""
    is_valid: bool                        # True if all hard gates pass
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    ood_fields: List[str] = field(default_factory=list)  # out-of-distribution

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [
                {"gate": e.gate, "field": e.field, "message": e.message, "severity": e.severity}
                for e in self.errors
            ],
            "warnings": [
                {"gate": w.gate, "field": w.field, "message": w.message, "severity": w.severity}
                for w in self.warnings
            ],
            "ood_fields": self.ood_fields,
        }


# ═══════════════════════════════════════════════════════
# Input Validator
# ═══════════════════════════════════════════════════════

class InputValidator:
    """4-gate sequential input validator.

    Usage:
        validator = InputValidator()
        result = validator.validate({
            "temperature": 185,
            "conveyor_speed": 75,
            "hold_time": 18,
            "batch_size": 500,
            "material_type": 0,
            "hour_of_day": 10,
            "operator_exp": 2,
        })

        if not result.is_valid:
            raise HTTPException(400, detail=result.to_dict())

        if result.has_warnings:
            # reduce confidence by ~35%
            pass
    """

    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Run all 4 gates sequentially. Short-circuit on hard failure."""
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        ood_fields: List[str] = []

        # ── Gate 1: Presence ─────────────────────────────
        presence_errors = self._check_presence(params)
        if presence_errors:
            errors.extend(presence_errors)
            return ValidationResult(is_valid=False, errors=errors)

        # ── Gate 2: Type ─────────────────────────────────
        type_errors = self._check_types(params)
        if type_errors:
            errors.extend(type_errors)
            return ValidationResult(is_valid=False, errors=errors)

        # ── Gate 3: Physical bounds ──────────────────────
        physical_errors = self._check_physical(params)
        if physical_errors:
            errors.extend(physical_errors)
            return ValidationResult(is_valid=False, errors=errors)

        # ── Gate 4: Training distribution (soft) ─────────
        dist_warnings, ood = self._check_training_dist(params)
        warnings.extend(dist_warnings)
        ood_fields.extend(ood)

        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            ood_fields=ood_fields,
        )

    # ──────────────────────────────────────────────────────
    # Gate 1: Presence
    # ──────────────────────────────────────────────────────

    def _check_presence(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Ensure all 7 required fields are present."""
        errors = []
        for f in REQUIRED_FIELDS:
            if f not in params or params[f] is None:
                errors.append(ValidationError(
                    gate="presence",
                    field=f,
                    message=f"Required field '{f}' is missing or null.",
                    severity="error",
                ))
        return errors

    # ──────────────────────────────────────────────────────
    # Gate 2: Type
    # ──────────────────────────────────────────────────────

    def _check_types(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Ensure all values are numeric (int or float)."""
        errors = []
        for f in REQUIRED_FIELDS:
            val = params.get(f)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(ValidationError(
                    gate="type",
                    field=f,
                    message=f"Field '{f}' must be numeric, got {type(val).__name__}: {val!r}.",
                    severity="error",
                ))
        return errors

    # ──────────────────────────────────────────────────────
    # Gate 3: Physical bounds (hard stop)
    # ──────────────────────────────────────────────────────

    def _check_physical(self, params: Dict[str, Any]) -> List[ValidationError]:
        """Check values are within machine operating range."""
        errors = []
        for f, (lo, hi) in PHYSICAL_LIMITS.items():
            val = params.get(f)
            if val is not None and (val < lo or val > hi):
                errors.append(ValidationError(
                    gate="physical",
                    field=f,
                    message=(
                        f"Field '{f}' value {val} is outside machine operating "
                        f"range [{lo}, {hi}]."
                    ),
                    severity="error",
                ))
        return errors

    # ──────────────────────────────────────────────────────
    # Gate 4: Training distribution (soft warning)
    # ──────────────────────────────────────────────────────

    def _check_training_dist(
        self,
        params: Dict[str, Any],
    ) -> Tuple[List[ValidationError], List[str]]:
        """Check values are within the model's training distribution."""
        warnings = []
        ood_fields = []

        for f, (lo, hi) in TRAINING_RANGE.items():
            val = params.get(f)
            if val is not None and (val < lo or val > hi):
                ood_fields.append(f)
                warnings.append(ValidationError(
                    gate="training",
                    field=f,
                    message=(
                        f"Field '{f}' value {val} is outside training range "
                        f"[{lo}, {hi}]. Model confidence may be reduced."
                    ),
                    severity="warning",
                ))

        return warnings, ood_fields
