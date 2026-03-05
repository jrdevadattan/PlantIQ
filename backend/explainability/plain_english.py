"""
PlantIQ — Plain English Converter
====================================
Converts raw SHAP feature contributions into human-readable summaries
that operators can actually understand and act on.

Per README spec:
  "Hold time added 3.8 kWh to this prediction because it was
   22 minutes — 4 minutes longer than the optimal 18 minutes
   for this material type."

Public API:
  PlainEnglishConverter
    .convert(explanation)   → adds 'plain_english' + 'summary' to explanation dict
    .feature_sentence(...)  → single feature explanation sentence
    .generate_summary(...)  → top-level summary paragraph

Usage:
  Used internally by ShapExplainer and the /explain/{batch_id} API route.
  Can also run standalone for testing:

    python explainability/plain_english.py
"""

import os
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# Domain knowledge for natural language generation
# ──────────────────────────────────────────────
OPTIMAL_VALUES = {
    "temperature": 183.0,
    "conveyor_speed": 75.0,
    "hold_time": 18.0,
    "batch_size": 500.0,
}

FEATURE_UNITS = {
    "temperature": "°C",
    "conveyor_speed": "%",
    "hold_time": "min",
    "batch_size": "kg",
    "material_type": "",
    "hour_of_day": "",
    "operator_exp": "",
    "temp_speed_product": "",
    "temp_deviation": "°C",
    "speed_deviation": "%",
    "hold_per_kg": "min/kg",
    "shift": "",
    "hours_into_shift": "hrs",
}

MATERIAL_LABELS = {0: "Type-A (standard)", 1: "Type-B (hard)", 2: "Type-C (high moisture)"}
OPERATOR_LABELS = {0: "junior", 1: "mid-level", 2: "senior"}
SHIFT_LABELS = {0: "morning", 1: "afternoon", 2: "evening"}

# Target-specific vocabulary
TARGET_VOCAB = {
    "quality_score": {"verb_up": "increased", "verb_down": "decreased", "noun": "quality", "unit": "%"},
    "yield_pct": {"verb_up": "increased", "verb_down": "decreased", "noun": "yield", "unit": "%"},
    "performance_pct": {"verb_up": "improved", "verb_down": "reduced", "noun": "performance", "unit": "%"},
    "energy_kwh": {"verb_up": "increased", "verb_down": "reduced", "noun": "energy consumption", "unit": "kWh"},
}

# How many top features to include in the summary
SUMMARY_TOP_N = 3


class PlainEnglishConverter:
    """Converts SHAP explanation dicts into human-readable text.

    Designed to produce operator-friendly explanations that:
    - State what happened in plain language
    - Reference specific parameter values and optima
    - Suggest actionable changes where possible
    """

    def convert(self, explanation: dict) -> dict:
        """Add plain_english and summary fields to an explanation dict.

        Parameters
        ----------
        explanation : dict
            Output from ShapExplainer.explain_single() for a single target.
            Must contain: target, feature_contributions, baseline_prediction,
            final_prediction, unit.

        Returns
        -------
        dict
            Same dict with added 'plain_english' per contribution + 'summary' field.
        """
        target = explanation["target"]
        contributions = explanation["feature_contributions"]
        prediction = explanation["final_prediction"]
        baseline = explanation["baseline_prediction"]
        unit = explanation.get("unit", "")

        # Add plain_english to each contribution
        for contrib in contributions:
            contrib["plain_english"] = self.feature_sentence(
                feature=contrib["feature"],
                value=contrib["value"],
                contribution=contrib["contribution"],
                target=target,
                unit=unit,
            )

        # Generate overall summary
        explanation["summary"] = self.generate_summary(
            target=target,
            prediction=prediction,
            baseline=baseline,
            unit=unit,
            contributions=contributions,
        )

        return explanation

    def feature_sentence(
        self,
        feature: str,
        value: float,
        contribution: float,
        target: str,
        unit: str,
    ) -> str:
        """Generate a plain English sentence for one feature's contribution.

        Parameters
        ----------
        feature : str
            Feature name (e.g., 'hold_time').
        value : float
            Actual feature value for this batch.
        contribution : float
            SHAP contribution value.
        target : str
            Target name (e.g., 'energy_kwh').
        unit : str
            Unit string for the target.

        Returns
        -------
        str
            Human-readable explanation sentence.
        """
        vocab = TARGET_VOCAB.get(target, {"verb_up": "increased", "verb_down": "decreased", "noun": target, "unit": unit})
        abs_contrib = abs(contribution)
        direction = vocab["verb_up"] if contribution > 0 else vocab["verb_down"]

        # Skip negligible contributions
        if abs_contrib < 0.01:
            return f"{self._display(feature)} had negligible impact on {vocab['noun']}."

        # ── Feature-specific sentences ──────────────────────────────────

        # Temperature
        if feature == "temperature":
            optimal = OPTIMAL_VALUES.get("temperature", 183.0)
            diff = value - optimal
            if abs(diff) < 0.5:
                return (f"Temperature of {value:.0f}°C is near optimal ({optimal:.0f}°C), "
                        f"contributing {contribution:+.1f} {unit} to {vocab['noun']}.")
            else:
                above_below = "above" if diff > 0 else "below"
                return (f"Temperature of {value:.0f}°C ({abs(diff):.0f}°C {above_below} optimal {optimal:.0f}°C) "
                        f"{direction} predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Conveyor speed
        if feature == "conveyor_speed":
            optimal = OPTIMAL_VALUES.get("conveyor_speed", 75.0)
            diff = value - optimal
            above_below = "above" if diff > 0 else "below"
            return (f"Conveyor speed of {value:.0f}% ({abs(diff):.0f}% {above_below} optimal {optimal:.0f}%) "
                    f"{direction} predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Hold time
        if feature == "hold_time":
            optimal = OPTIMAL_VALUES.get("hold_time", 18.0)
            diff = value - optimal
            if abs(diff) < 0.5:
                return (f"Hold time of {value:.0f} min is near optimal ({optimal:.0f} min), "
                        f"contributing {contribution:+.1f} {unit} to {vocab['noun']}.")
            longer_shorter = "longer" if diff > 0 else "shorter"
            return (f"Hold time of {value:.0f} min — {abs(diff):.0f} min {longer_shorter} than "
                    f"optimal {optimal:.0f} min — {direction} predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Material type
        if feature == "material_type":
            mat_val = int(value)
            mat_label = MATERIAL_LABELS.get(mat_val, f"type {mat_val}")
            return (f"{mat_label} material {direction} predicted {vocab['noun']} by "
                    f"{abs_contrib:.1f} {unit}.")

        # Operator experience
        if feature == "operator_exp":
            exp_val = int(value)
            exp_label = OPERATOR_LABELS.get(exp_val, f"level {exp_val}")
            return (f"{exp_label.capitalize()} operator experience {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Batch size
        if feature == "batch_size":
            return (f"Batch size of {value:.0f} kg {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Hour of day
        if feature == "hour_of_day":
            return (f"Production at hour {value:.0f} {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Derived features — more generic sentences
        if feature == "temp_deviation":
            return (f"Temperature deviation of {value:.1f}°C from optimal {direction} "
                    f"predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

        if feature == "speed_deviation":
            return (f"Speed deviation of {value:.1f}% from optimal {direction} "
                    f"predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

        if feature == "temp_speed_product":
            return (f"Temperature-speed interaction {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        if feature == "hold_per_kg":
            return (f"Hold time per kg ({value:.4f} min/kg) {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        if feature == "shift":
            shift_label = SHIFT_LABELS.get(int(value), f"shift {int(value)}")
            return (f"{shift_label.capitalize()} shift {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        if feature == "hours_into_shift":
            return (f"Being {value:.1f} hours into the shift {direction} predicted "
                    f"{vocab['noun']} by {abs_contrib:.1f} {unit}.")

        # Fallback
        return (f"{self._display(feature)} = {value:.2f} {direction} "
                f"predicted {vocab['noun']} by {abs_contrib:.1f} {unit}.")

    def generate_summary(
        self,
        target: str,
        prediction: float,
        baseline: float,
        unit: str,
        contributions: list[dict],
    ) -> str:
        """Generate a top-level summary paragraph.

        Parameters
        ----------
        target : str
            Target name.
        prediction : float
            Actual prediction value.
        baseline : float
            Baseline (average) prediction.
        unit : str
            Unit string.
        contributions : list[dict]
            Sorted feature contributions (largest first).

        Returns
        -------
        str
            Summary paragraph.
        """
        vocab = TARGET_VOCAB.get(target, {"noun": target, "unit": unit})
        delta = prediction - baseline

        # Top contributors
        top = contributions[:SUMMARY_TOP_N]

        if not top:
            return f"Predicted {vocab['noun']} is {prediction} {unit} (baseline: {baseline} {unit})."

        # Build the main driver sentence
        top_feat = top[0]
        top_name = self._display(top_feat["feature"])
        top_contrib = top_feat["contribution"]

        if abs(delta) < 0.1:
            opening = (f"Predicted {vocab['noun']} of {prediction} {unit} is very close to "
                       f"the baseline ({baseline} {unit}).")
        elif delta > 0:
            opening = (f"{top_name} is the dominant factor, contributing "
                       f"{top_contrib:+.1f} {unit} to predicted {vocab['noun']} of {prediction} {unit} "
                       f"(baseline: {baseline} {unit}).")
        else:
            opening = (f"{top_name} is the dominant factor, contributing "
                       f"{top_contrib:+.1f} {unit} to predicted {vocab['noun']} of {prediction} {unit} "
                       f"(baseline: {baseline} {unit}).")

        # Add secondary drivers
        if len(top) > 1:
            secondary = []
            for c in top[1:]:
                name = self._display(c["feature"])
                contrib = c["contribution"]
                secondary.append(f"{name} ({contrib:+.1f} {unit})")
            secondary_str = " and ".join(secondary)
            opening += f" Other significant factors: {secondary_str}."

        # Add actionable suggestion for energy target
        if target == "energy_kwh" and delta > 1.0:
            # Find the largest positive contributor that's actionable
            for c in top:
                feat = c["feature"]
                if feat in OPTIMAL_VALUES and c["contribution"] > 0.5:
                    optimal = OPTIMAL_VALUES[feat]
                    feat_unit = FEATURE_UNITS.get(feat, "")
                    opening += (f" Reducing {self._display(feat).lower()} toward "
                                f"{optimal}{feat_unit} could lower energy consumption.")
                    break

        return opening

    @staticmethod
    def _display(feature: str) -> str:
        """Get human-friendly display name for a feature."""
        display_map = {
            "temperature": "Temperature",
            "conveyor_speed": "Conveyor speed",
            "hold_time": "Hold time",
            "batch_size": "Batch size",
            "material_type": "Material type",
            "hour_of_day": "Hour of day",
            "operator_exp": "Operator experience",
            "temp_speed_product": "Temperature-speed interaction",
            "temp_deviation": "Temperature deviation",
            "speed_deviation": "Speed deviation",
            "hold_per_kg": "Hold time per kg",
            "shift": "Shift period",
            "hours_into_shift": "Hours into shift",
        }
        return display_map.get(feature, feature.replace("_", " ").title())


# ──────────────────────────────────────────────
# CLI — Standalone demo
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PLAIN ENGLISH CONVERTER — Demo")
    print("=" * 65)

    from explainability.shap_explainer import ShapExplainer

    explainer = ShapExplainer()
    converter = PlainEnglishConverter()

    # Demo with a suboptimal batch
    params = {
        "temperature": 190.0,
        "conveyor_speed": 85.0,
        "hold_time": 25.0,
        "batch_size": 600.0,
        "material_type": 2,
        "hour_of_day": 15,
        "operator_exp": 0,
    }

    print("\n  Input parameters:")
    for k, v in params.items():
        print(f"    {k:20s} = {v}")

    for target in ["energy_kwh", "quality_score"]:
        explanation = explainer.explain_single(params, target=target)
        explanation = converter.convert(explanation)

        print(f"\n  {'─' * 61}")
        print(f"  TARGET: {target}")
        print(f"  Prediction: {explanation['final_prediction']} {explanation['unit']}")
        print(f"  Baseline:   {explanation['baseline_prediction']} {explanation['unit']}")
        print(f"  {'═' * 61}")

        # Show top 5 contributions with plain English
        for c in explanation["feature_contributions"][:5]:
            contrib = c["contribution"]
            sign = "+" if contrib >= 0 else ""
            print(f"\n    {c['display_name']:22s}  {sign}{contrib:.3f} {explanation['unit']}")
            print(f"    → {c['plain_english']}")

        print(f"\n  SUMMARY:")
        print(f"    {explanation['summary']}")

    print(f"\n  {'═' * 61}")
    print("  ✅ Plain English Converter operational")
    print(f"  {'═' * 61}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
