"""
PlantIQ — SHAP Explainability Engine
======================================
Computes per-feature SHAP contributions for every prediction made by the
multi-target XGBoost model.  Uses TreeExplainer (exact, fast on tree models).

Key output per README spec:
  For each prediction → SHAP values → feature contributions → plain English

Architecture:
  TreeExplainer wraps each of the 4 XGBRegressor estimators inside the
  MultiOutputRegressor.  This gives exact Shapley values in O(TLD) time
  where T=trees, L=leaves, D=depth.

Public API:
  ShapExplainer
    .explain_single(params)       → full explanation dict for 1 batch
    .explain_batch(X, target)     → SHAP matrix for many samples
    .get_baseline(target)         → expected (average) prediction
    .global_importance(target)    → mean |SHAP| per feature (global)

Usage:
  python explainability/shap_explainer.py            # Demo explanation
  python explainability/shap_explainer.py --batch_id 42  # Explain batch #42

Output:
  Structured dict matching GET /explain/{batch_id}?target=energy API spec.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
from models.multi_target_predictor import MultiTargetPredictor, FEATURE_COLS, TARGET_COLS

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
TRAIN_CSV = os.path.join(DATA_DIR, "train_processed.csv")


# Domain knowledge for plain-English generation
OPTIMAL_VALUES = {
    "temperature": {"value": 183.0, "unit": "°C"},
    "conveyor_speed": {"value": 75.0, "unit": "%"},
    "hold_time": {"value": 18.0, "unit": "min"},
    "batch_size": {"value": 500.0, "unit": "kg"},
    "material_type": {"value": 0, "unit": "type", "labels": {0: "Type-A", 1: "Type-B", 2: "Type-C"}},
    "hour_of_day": {"value": 10, "unit": "hour"},
    "operator_exp": {"value": 2, "unit": "level", "labels": {0: "junior", 1: "mid", 2: "senior"}},
}

TARGET_UNITS = {
    "quality_score": "%",
    "yield_pct": "%",
    "performance_pct": "%",
    "energy_kwh": "kWh",
}

# Friendly display names for features
FEATURE_DISPLAY = {
    "temperature": "Temperature",
    "conveyor_speed": "Conveyor Speed",
    "hold_time": "Hold Time",
    "batch_size": "Batch Size",
    "material_type": "Material Type",
    "hour_of_day": "Hour of Day",
    "operator_exp": "Operator Experience",
    "temp_speed_product": "Temp × Speed",
    "temp_deviation": "Temperature Deviation",
    "speed_deviation": "Speed Deviation",
    "hold_per_kg": "Hold Time per kg",
    "shift": "Shift Period",
    "hours_into_shift": "Hours into Shift",
}


class ShapExplainer:
    """SHAP-based explainability for the multi-target XGBoost model.

    Uses shap.TreeExplainer on each of the 4 XGBRegressor estimators
    to compute exact Shapley values.

    Attributes
    ----------
    predictor : MultiTargetPredictor
        Loaded multi-target model.
    explainers : dict[str, shap.TreeExplainer]
        One TreeExplainer per target.
    baselines : dict[str, float]
        Expected (mean) prediction per target.
    """

    def __init__(self):
        self.predictor = MultiTargetPredictor()
        self.predictor.load()
        self.explainers: dict[str, shap.TreeExplainer] = {}
        self.baselines: dict[str, float] = {}
        self._init_explainers()

    def _init_explainers(self):
        """Create a TreeExplainer for each target model."""
        for i, target in enumerate(TARGET_COLS):
            estimator = self.predictor.model.estimators_[i]
            explainer = shap.TreeExplainer(estimator)
            self.explainers[target] = explainer
            # Baseline = expected_value (mean prediction over training data)
            base = explainer.expected_value
            if isinstance(base, np.ndarray):
                base = float(base[0])
            else:
                base = float(base)
            self.baselines[target] = round(base, 2)

    def explain_single(
        self,
        params: dict,
        target: str | None = None,
    ) -> dict:
        """Generate a full SHAP explanation for a single batch.

        Parameters
        ----------
        params : dict
            Raw batch parameters (7 operator inputs).
            Example: {temperature: 183, conveyor_speed: 76, hold_time: 22, ...}
        target : str | None
            Specific target to explain.  If None, explains all 4 targets.

        Returns
        -------
        dict
            Structured explanation matching GET /explain/{batch_id} API spec.
        """
        from preprocessing.feature_engineer import FeatureEngineer

        # Build feature vector
        row = pd.DataFrame([params])
        fe = FeatureEngineer(include_energy_per_kg=False)
        row = fe.transform(row)

        # Ensure all feature columns exist
        for col in FEATURE_COLS:
            if col not in row.columns:
                row[col] = 0

        X = row[FEATURE_COLS]

        targets_to_explain = [target] if target else TARGET_COLS
        explanations = {}

        for t in targets_to_explain:
            if t not in self.explainers:
                raise ValueError(f"Unknown target '{t}'. Must be one of {TARGET_COLS}")

            # Compute SHAP values
            shap_values = self.explainers[t].shap_values(X.values)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # shap_values shape: (1, n_features) — single sample
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values

            # Get prediction
            pred = self.predictor.predict(X.values)[0]
            target_idx = TARGET_COLS.index(t)
            prediction = round(float(pred[target_idx]), 2)
            baseline = self.baselines[t]
            unit = TARGET_UNITS.get(t, "")

            # Build feature contributions (sorted by absolute impact)
            contributions = []
            for j, feat in enumerate(FEATURE_COLS):
                contrib = float(sv[j])
                feat_value = float(X[feat].iloc[0])

                # Determine direction
                if contrib > 0:
                    direction = f"increases_{t.split('_')[0]}"
                elif contrib < 0:
                    direction = f"decreases_{t.split('_')[0]}"
                else:
                    direction = "neutral"

                contributions.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY.get(feat, feat),
                    "value": round(feat_value, 2),
                    "contribution": round(contrib, 4),
                    "direction": direction,
                })

            # Sort by absolute contribution (largest impact first)
            contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

            explanations[t] = {
                "target": t,
                "baseline_prediction": baseline,
                "final_prediction": prediction,
                "unit": unit,
                "feature_contributions": contributions,
            }

        # Return single target or all
        if target:
            return explanations[target]
        return explanations

    def explain_batch(
        self,
        X: pd.DataFrame | np.ndarray,
        target: str,
    ) -> np.ndarray:
        """Compute SHAP values for multiple samples at once.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix, shape (n_samples, 13).
        target : str
            Target to explain.

        Returns
        -------
        np.ndarray
            SHAP values, shape (n_samples, n_features).
        """
        if target not in self.explainers:
            raise ValueError(f"Unknown target '{target}'. Must be one of {TARGET_COLS}")

        if isinstance(X, pd.DataFrame):
            X_vals = X[FEATURE_COLS].values
        else:
            X_vals = X

        shap_values = self.explainers[target].shap_values(X_vals)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return shap_values

    def get_baseline(self, target: str) -> float:
        """Get the expected (average) prediction for a target.

        Parameters
        ----------
        target : str
            Target name.

        Returns
        -------
        float
            Baseline prediction.
        """
        if target not in self.baselines:
            raise ValueError(f"Unknown target '{target}'.")
        return self.baselines[target]

    def global_importance(self, target: str, n_samples: int = 500) -> dict[str, float]:
        """Compute global feature importance as mean |SHAP| values.

        Parameters
        ----------
        target : str
            Target to analyze.
        n_samples : int
            Number of training samples to compute mean |SHAP| over.

        Returns
        -------
        dict[str, float]
            {feature_name: mean_absolute_shap_value}, sorted descending.
        """
        # Load training data
        train_df = pd.read_csv(TRAIN_CSV)
        X_train = train_df[FEATURE_COLS].values

        # Subsample if large
        if len(X_train) > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), n_samples, replace=False)
            X_train = X_train[idx]

        shap_values = self.explain_batch(X_train, target)
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        importance = {}
        for j, feat in enumerate(FEATURE_COLS):
            importance[feat] = round(float(mean_abs[j]), 4)

        # Sort descending
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance


# ──────────────────────────────────────────────
# CLI — Demo Explanation
# ──────────────────────────────────────────────
def _format_bar(value: float, max_val: float, width: int = 30) -> str:
    """Create a simple ASCII bar for contribution visualization."""
    if max_val == 0:
        return ""
    n_blocks = int(abs(value) / max_val * width)
    n_blocks = max(1, n_blocks) if abs(value) > 0.01 else 0
    return "█" * n_blocks


def main():
    parser = argparse.ArgumentParser(description="SHAP Explainability Engine")
    parser.add_argument("--target", type=str, default=None,
                        help="Target to explain (quality_score, yield_pct, performance_pct, energy_kwh)")
    parser.add_argument("--batch_id", type=int, default=None,
                        help="Batch index from test set to explain (0-399)")
    args = parser.parse_args()

    print("=" * 65)
    print("  SHAP EXPLAINABILITY ENGINE — PlantIQ")
    print("=" * 65)

    # Initialize explainer
    print("\n  Initializing SHAP TreeExplainers...")
    explainer = ShapExplainer()
    print(f"  ✅ {len(explainer.explainers)} explainers ready")
    print(f"  Baselines: {explainer.baselines}")

    # ── Single-batch explanation ──────────────────────────────────────────
    if args.batch_id is not None:
        # Use actual test data
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
        if args.batch_id < 0 or args.batch_id >= len(test_df):
            print(f"\n  ❌ batch_id must be 0–{len(test_df) - 1}")
            return 1

        row = test_df.iloc[args.batch_id]
        params = {col: float(row[col]) for col in FEATURE_COLS}
        print(f"\n  Explaining batch #{args.batch_id} from test set")
    else:
        # Demo with sample parameters (slightly suboptimal batch)
        params = {
            "temperature": 188.0,
            "conveyor_speed": 82.0,
            "hold_time": 22.0,
            "batch_size": 550.0,
            "material_type": 2,
            "hour_of_day": 14,
            "operator_exp": 1,
        }
        print("\n  Demo batch (suboptimal parameters):")
        for k, v in params.items():
            print(f"    {k:20s} = {v}")

    # Explain all targets or specific target
    targets = [args.target] if args.target else TARGET_COLS

    for target in targets:
        result = explainer.explain_single(params, target=target)
        unit = result["unit"]
        baseline = result["baseline_prediction"]
        prediction = result["final_prediction"]
        delta = prediction - baseline

        print(f"\n  {'─' * 61}")
        print(f"  TARGET: {target}")
        print(f"  Prediction: {prediction} {unit}  (baseline: {baseline} {unit})")
        print(f"  Delta from baseline: {delta:+.2f} {unit}")
        print(f"  {'═' * 61}")

        contribs = result["feature_contributions"]
        max_val = max(abs(c["contribution"]) for c in contribs) if contribs else 1.0

        for c in contribs:
            contrib = c["contribution"]
            bar = _format_bar(contrib, max_val)
            sign = "+" if contrib >= 0 else ""
            feat_name = c["display_name"]
            print(f"  {feat_name:22s} {sign}{contrib:>7.3f} {unit:4s}  {bar}")

        print(f"  {'═' * 61}")

    # ── Global importance ──────────────────────────────────────────────────
    print(f"\n  {'─' * 61}")
    print("  GLOBAL FEATURE IMPORTANCE (mean |SHAP| over 500 samples)")
    print(f"  {'─' * 61}")

    for target in TARGET_COLS:
        gi = explainer.global_importance(target, n_samples=500)
        top5 = list(gi.items())[:5]
        names = "  ".join(f"{FEATURE_DISPLAY.get(f, f)}({v:.3f})" for f, v in top5)
        print(f"  {target:20s} │ {names}")

    print(f"\n  {'═' * 61}")
    print("  ✅ SHAP Explainability Engine operational")
    print(f"  {'═' * 61}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
