"""
PlantIQ — Golden Signature Framework
======================================
Implements the Golden Signature management system as specified in the
hackathon problem statement:

  "A golden signature is a set of optimised features for the given
   multi-objective target. The requirement is to design and implement
   comprehensive optimized parameters for target combinations such as:
   Best yield with lowest energy, Optimal quality with best yield,
   Maximum performance with minimal environmental impact."

Features:
  1. Golden Signature Discovery — Pareto-optimal parameter sets from historical data
  2. Multi-Objective Optimization — Find best trade-offs between conflicting targets
  3. Dynamic Benchmarking — Compare current batches against golden signatures
  4. Self-Improvement — Auto-update signatures when new data exceeds benchmarks
  5. Scenario Analysis — "What-if" for any combination of primary + secondary targets

Supports both synthetic PlantIQ data and real hackathon pharmaceutical data.

Spec: Problem Statement — "Golden Signature Management"
       README — "golden batch fingerprints for ideal parameter combinations"
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
SIGNATURE_FILE = os.path.join(ARTIFACT_DIR, "golden_signatures.json")

# Default target directions (higher is better unless specified)
DEFAULT_TARGET_DIRECTIONS = {
    # Synthetic data targets
    "quality_score": "maximize",
    "yield_pct": "maximize",
    "performance_pct": "maximize",
    "energy_kwh": "minimize",
    "co2_kg": "minimize",
    # Hackathon data targets
    "Hardness": "maximize",
    "Dissolution_Rate": "maximize",
    "Content_Uniformity": "target",   # target = close to 100%
    "Friability": "minimize",
    "Moisture_Content": "target",     # target = close to 2%
    "Tablet_Weight": "target",        # target = close to 200mg
    "Disintegration_Time": "minimize",
}

# Optimal target values for "target" direction
TARGET_OPTIMA = {
    "Content_Uniformity": 100.0,
    "Moisture_Content": 2.0,
    "Tablet_Weight": 200.0,
}


class GoldenSignatureManager:
    """Manages golden batch signatures for multi-objective optimization.

    A golden signature represents the optimal set of input parameters
    that achieves the best trade-off between selected target objectives.
    Multiple signatures can exist for different priority combinations.

    Attributes
    ----------
    signatures : dict
        Stored golden signatures keyed by scenario ID.
    history : list
        Historical log of signature updates.
    """

    def __init__(self):
        self.signatures: dict = {}
        self.history: list = []
        self._load_if_exists()

    def _load_if_exists(self):
        """Load saved signatures from disk if they exist."""
        if os.path.exists(SIGNATURE_FILE):
            try:
                with open(SIGNATURE_FILE, "r") as f:
                    data = json.load(f)
                self.signatures = data.get("signatures", {})
                self.history = data.get("history", [])
                print(f"[GoldenSignature] Loaded {len(self.signatures)} saved signatures")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[GoldenSignature] Warning: could not load signatures: {e}")

    def save(self):
        """Persist current signatures to disk."""
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        data = {
            "signatures": self.signatures,
            "history": self.history,
            "last_updated": datetime.now().isoformat(),
        }
        with open(SIGNATURE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[GoldenSignature] Saved {len(self.signatures)} signatures to {SIGNATURE_FILE}")

    # ──────────────────────────────────────────────
    # Core: Discover Golden Signatures
    # ──────────────────────────────────────────────

    def discover_signatures(
        self,
        df: pd.DataFrame,
        input_cols: list[str],
        target_cols: list[str],
        target_directions: Optional[dict] = None,
        n_top: int = 5,
    ) -> dict:
        """Discover golden signatures from historical batch data.

        Uses Pareto-optimal front identification to find batches that
        represent the best trade-offs across all specified targets.

        Parameters
        ----------
        df : pd.DataFrame
            Historical batch data with input and target columns.
        input_cols : list[str]
            Column names for controllable input parameters.
        target_cols : list[str]
            Column names for output targets to optimize.
        target_directions : dict, optional
            Per-target optimization direction: "maximize", "minimize", or "target".
        n_top : int
            Number of top batches to consider for golden signatures.

        Returns
        -------
        dict
            Discovered signatures with Pareto-optimal parameter sets.
        """
        if target_directions is None:
            target_directions = DEFAULT_TARGET_DIRECTIONS

        df = df.copy()

        # Compute composite score for each batch
        df["_composite_score"] = self._compute_composite_scores(
            df, target_cols, target_directions
        )

        # Sort by composite score (higher is better)
        df_sorted = df.sort_values("_composite_score", ascending=False)
        top_batches = df_sorted.head(n_top)

        # Find Pareto-optimal batches
        pareto_indices = self._find_pareto_front(
            df, target_cols, target_directions
        )
        pareto_batches = df.iloc[pareto_indices]

        # Golden signature = mean of top Pareto-optimal input parameters
        golden_params = {}
        for col in input_cols:
            if col in pareto_batches.columns:
                values = pareto_batches[col].values
                golden_params[col] = {
                    "optimal": round(float(np.mean(values)), 4),
                    "min_range": round(float(np.min(values)), 4),
                    "max_range": round(float(np.max(values)), 4),
                    "std": round(float(np.std(values)), 4),
                }

        # Golden targets (what the optimal params achieve)
        golden_targets = {}
        for col in target_cols:
            if col in pareto_batches.columns:
                values = pareto_batches[col].values
                golden_targets[col] = {
                    "expected": round(float(np.mean(values)), 4),
                    "best": round(float(
                        np.max(values) if target_directions.get(col) == "maximize"
                        else np.min(values) if target_directions.get(col) == "minimize"
                        else values[np.argmin(np.abs(values - TARGET_OPTIMA.get(col, np.mean(values))))]
                    ), 4),
                    "direction": target_directions.get(col, "maximize"),
                }

        scenario_id = "_".join(target_cols)
        signature = {
            "scenario_id": scenario_id,
            "created_at": datetime.now().isoformat(),
            "n_batches_analyzed": len(df),
            "n_pareto_optimal": len(pareto_indices),
            "golden_parameters": golden_params,
            "expected_targets": golden_targets,
            "composite_score": round(float(pareto_batches["_composite_score"].mean()), 4),
            "top_batch_ids": (
                top_batches["Batch_ID"].tolist()
                if "Batch_ID" in top_batches.columns
                else top_batches.index.tolist()
            ),
        }

        # Store and save
        self.signatures[scenario_id] = signature
        self.history.append({
            "action": "discover",
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            "composite_score": signature["composite_score"],
        })
        self.save()

        return signature

    # ──────────────────────────────────────────────
    # Compare Batch Against Golden Signature
    # ──────────────────────────────────────────────

    def compare_batch(
        self,
        batch_params: dict,
        batch_targets: dict,
        scenario_id: Optional[str] = None,
    ) -> dict:
        """Compare a single batch against the golden signature.

        Returns deviation analysis and recommendations for parameter
        adjustments to move closer to the golden signature.

        Parameters
        ----------
        batch_params : dict
            Current batch input parameters {param_name: value}.
        batch_targets : dict
            Current batch target outcomes {target_name: value}.
        scenario_id : str, optional
            Which golden signature to compare against. Uses first available.

        Returns
        -------
        dict
            Comparison with deviations, score, and recommendations.
        """
        if scenario_id and scenario_id in self.signatures:
            sig = self.signatures[scenario_id]
        elif self.signatures:
            sig = list(self.signatures.values())[0]
        else:
            return {
                "error": "No golden signatures available. Run discover_signatures() first.",
                "has_signature": False,
            }

        # Parameter deviations
        param_deviations = {}
        for param, spec in sig["golden_parameters"].items():
            if param in batch_params:
                current = batch_params[param]
                optimal = spec["optimal"]
                deviation = current - optimal
                deviation_pct = abs(deviation / optimal * 100) if optimal != 0 else 0

                param_deviations[param] = {
                    "current": round(float(current), 4),
                    "optimal": optimal,
                    "deviation": round(float(deviation), 4),
                    "deviation_pct": round(float(deviation_pct), 2),
                    "in_range": spec["min_range"] <= current <= spec["max_range"],
                    "adjustment": "increase" if deviation < 0 else "decrease" if deviation > 0 else "optimal",
                }

        # Target deviations
        target_deviations = {}
        for target, spec in sig["expected_targets"].items():
            if target in batch_targets:
                current = batch_targets[target]
                expected = spec["expected"]
                direction = spec["direction"]

                deviation = current - expected
                deviation_pct = abs(deviation / expected * 100) if expected != 0 else 0

                # Performance assessment
                if direction == "maximize":
                    performance = "above_target" if current >= expected else "below_target"
                elif direction == "minimize":
                    performance = "above_target" if current <= expected else "below_target"
                else:  # target
                    performance = "on_target" if deviation_pct < 5 else "off_target"

                target_deviations[target] = {
                    "current": round(float(current), 4),
                    "golden_expected": expected,
                    "deviation": round(float(deviation), 4),
                    "deviation_pct": round(float(deviation_pct), 2),
                    "performance": performance,
                }

        # Generate recommendations
        recommendations = self._generate_recommendations(param_deviations, target_deviations)

        # Overall alignment score (0–100)
        if param_deviations:
            avg_param_dev = np.mean([
                d["deviation_pct"] for d in param_deviations.values()
            ])
            alignment_score = max(0, 100 - avg_param_dev)
        else:
            alignment_score = 0.0

        return {
            "has_signature": True,
            "scenario_id": sig["scenario_id"],
            "alignment_score": round(float(alignment_score), 2),
            "parameter_deviations": param_deviations,
            "target_deviations": target_deviations,
            "recommendations": recommendations,
            "golden_signature_date": sig["created_at"],
        }

    # ──────────────────────────────────────────────
    # Self-Improvement: Update Signatures
    # ──────────────────────────────────────────────

    def update_if_better(
        self,
        batch_params: dict,
        batch_targets: dict,
        target_cols: list[str],
        target_directions: Optional[dict] = None,
    ) -> dict:
        """Check if a new batch exceeds the current golden signature.

        If it does, update the golden signature benchmarks.
        This implements the "Continuous Learning" requirement.

        Parameters
        ----------
        batch_params : dict
            New batch input parameters.
        batch_targets : dict
            New batch outcomes.
        target_cols : list[str]
            Targets to evaluate.
        target_directions : dict, optional
            Optimization directions.

        Returns
        -------
        dict
            Update result with whether signature was improved.
        """
        if target_directions is None:
            target_directions = DEFAULT_TARGET_DIRECTIONS

        scenario_id = "_".join(target_cols)
        if scenario_id not in self.signatures:
            return {"updated": False, "reason": "No existing signature for this scenario"}

        sig = self.signatures[scenario_id]

        # Compute new batch's composite score
        new_score = 0.0
        current_score = sig["composite_score"]

        for target in target_cols:
            if target not in batch_targets or target not in sig["expected_targets"]:
                continue

            current_val = batch_targets[target]
            expected_val = sig["expected_targets"][target]["expected"]
            direction = target_directions.get(target, "maximize")

            if direction == "maximize":
                new_score += current_val / max(expected_val, 0.01)
            elif direction == "minimize":
                new_score += expected_val / max(current_val, 0.01)
            else:  # target
                optimal = TARGET_OPTIMA.get(target, expected_val)
                new_score += 1.0 - abs(current_val - optimal) / max(optimal, 0.01)

        new_score /= max(len(target_cols), 1)

        if new_score > current_score:
            # Update the golden signature with this batch's parameters
            for param, value in batch_params.items():
                if param in sig["golden_parameters"]:
                    old_optimal = sig["golden_parameters"][param]["optimal"]
                    # Exponential moving average (blend 30% new, 70% old)
                    sig["golden_parameters"][param]["optimal"] = round(
                        0.7 * old_optimal + 0.3 * float(value), 4
                    )

            for target, value in batch_targets.items():
                if target in sig["expected_targets"]:
                    old_expected = sig["expected_targets"][target]["expected"]
                    sig["expected_targets"][target]["expected"] = round(
                        0.7 * old_expected + 0.3 * float(value), 4
                    )

            sig["composite_score"] = round(float(new_score), 4)
            sig["last_improved"] = datetime.now().isoformat()

            self.history.append({
                "action": "auto_update",
                "scenario_id": scenario_id,
                "timestamp": datetime.now().isoformat(),
                "old_score": round(float(current_score), 4),
                "new_score": round(float(new_score), 4),
                "improvement_pct": round((new_score - current_score) / current_score * 100, 2),
            })
            self.save()

            return {
                "updated": True,
                "old_score": round(float(current_score), 4),
                "new_score": round(float(new_score), 4),
                "improvement_pct": round((new_score - current_score) / current_score * 100, 2),
            }

        return {
            "updated": False,
            "reason": f"Current score ({new_score:.4f}) does not exceed golden ({current_score:.4f})",
        }

    # ──────────────────────────────────────────────
    # Multi-Objective Scenario Analysis
    # ──────────────────────────────────────────────

    def get_scenario_recommendations(
        self,
        df: pd.DataFrame,
        input_cols: list[str],
        primary_targets: list[str],
        secondary_targets: Optional[list[str]] = None,
        primary_weight: float = 0.7,
    ) -> dict:
        """Generate recommendations for a specific target combination scenario.

        "The user can choose any combination of primary targets and secondary targets."

        Parameters
        ----------
        df : pd.DataFrame
            Historical data.
        input_cols : list[str]
            Input parameter columns.
        primary_targets : list[str]
            Primary optimization targets (weighted higher).
        secondary_targets : list[str], optional
            Secondary targets (weighted lower).
        primary_weight : float
            Weight for primary targets (default 0.7, secondary gets 0.3).

        Returns
        -------
        dict
            Scenario-specific golden signature and recommendations.
        """
        all_targets = primary_targets + (secondary_targets or [])
        target_directions = {t: DEFAULT_TARGET_DIRECTIONS.get(t, "maximize") for t in all_targets}

        # Discover with weighted scoring
        sig = self.discover_signatures(
            df, input_cols, all_targets, target_directions
        )

        # Add scenario-specific metadata
        sig["primary_targets"] = primary_targets
        sig["secondary_targets"] = secondary_targets or []
        sig["primary_weight"] = primary_weight

        # Human-readable scenario description
        primary_str = " + ".join(primary_targets)
        secondary_str = " + ".join(secondary_targets) if secondary_targets else "none"
        sig["scenario_description"] = (
            f"Optimize for {primary_str} (primary, weight={primary_weight}) "
            f"with {secondary_str} (secondary)"
        )

        return sig

    # ──────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────

    def _compute_composite_scores(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        target_directions: dict,
    ) -> pd.Series:
        """Compute a single composite score for each batch row."""
        scores = pd.Series(0.0, index=df.index)

        for col in target_cols:
            if col not in df.columns:
                continue

            direction = target_directions.get(col, "maximize")
            values = df[col].values

            if direction == "maximize":
                # Normalize to 0–1 (higher is better)
                vmin, vmax = values.min(), values.max()
                if vmax > vmin:
                    normalized = (values - vmin) / (vmax - vmin)
                else:
                    normalized = np.ones_like(values) * 0.5
            elif direction == "minimize":
                # Invert so lower is better
                vmin, vmax = values.min(), values.max()
                if vmax > vmin:
                    normalized = 1.0 - (values - vmin) / (vmax - vmin)
                else:
                    normalized = np.ones_like(values) * 0.5
            else:  # target
                optimal = TARGET_OPTIMA.get(col, np.mean(values))
                max_dev = max(abs(values.max() - optimal), abs(values.min() - optimal), 1e-6)
                normalized = 1.0 - np.abs(values - optimal) / max_dev

            scores += normalized

        return scores / max(len(target_cols), 1)

    def _find_pareto_front(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        target_directions: dict,
    ) -> list[int]:
        """Find Pareto-optimal batch indices (no batch dominates another)."""
        # Prepare objective matrix (all "maximize" direction)
        objectives = []
        for col in target_cols:
            if col not in df.columns:
                continue
            values = df[col].values.copy()
            direction = target_directions.get(col, "maximize")
            if direction == "minimize":
                values = -values  # Negate so we always maximize
            elif direction == "target":
                optimal = TARGET_OPTIMA.get(col, np.mean(values))
                values = -np.abs(values - optimal)  # Closer to optimal = better
            objectives.append(values)

        if not objectives:
            return list(range(min(5, len(df))))

        obj_matrix = np.column_stack(objectives)
        n = len(obj_matrix)

        # Pareto dominance check
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # j dominates i if j is >= in all objectives and > in at least one
                if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                    is_pareto[i] = False
                    break

        pareto_indices = np.where(is_pareto)[0].tolist()
        return pareto_indices if pareto_indices else list(range(min(5, n)))

    def _generate_recommendations(
        self,
        param_deviations: dict,
        target_deviations: dict,
    ) -> list[str]:
        """Generate human-readable recommendations based on deviations."""
        recs = []

        # Parameter adjustment recommendations
        large_deviations = [
            (p, d) for p, d in param_deviations.items()
            if d["deviation_pct"] > 10
        ]
        large_deviations.sort(key=lambda x: x[1]["deviation_pct"], reverse=True)

        for param, dev in large_deviations[:3]:
            action = dev["adjustment"]
            pct = dev["deviation_pct"]
            target_val = dev["optimal"]

            if action == "increase":
                recs.append(
                    f"Increase {param} by ~{pct:.0f}% "
                    f"(current: {dev['current']:.1f}, golden: {target_val:.1f})"
                )
            elif action == "decrease":
                recs.append(
                    f"Decrease {param} by ~{pct:.0f}% "
                    f"(current: {dev['current']:.1f}, golden: {target_val:.1f})"
                )

        # Target performance recommendations
        underperforming = [
            (t, d) for t, d in target_deviations.items()
            if d["performance"] in ("below_target", "off_target")
        ]
        for target, dev in underperforming:
            recs.append(
                f"{target} is {dev['performance'].replace('_', ' ')} "
                f"(current: {dev['current']:.2f}, expected: {dev['golden_expected']:.2f})"
            )

        if not recs:
            recs.append("Batch is well-aligned with the golden signature. No adjustments needed.")

        return recs

    # ──────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────

    def get_all_signatures(self) -> dict:
        """Return all stored golden signatures."""
        return {
            "count": len(self.signatures),
            "signatures": self.signatures,
            "history_count": len(self.history),
            "recent_history": self.history[-10:] if self.history else [],
        }


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    """CLI: Discover golden signatures from hackathon data."""
    import argparse

    parser = argparse.ArgumentParser(description="PlantIQ — Golden Signature Framework")
    parser.add_argument("--discover", action="store_true", help="Discover signatures from hackathon data")
    parser.add_argument("--synthetic", action="store_true", help="Discover from synthetic data")
    parser.add_argument("--show", action="store_true", help="Show existing signatures")
    args = parser.parse_args()

    manager = GoldenSignatureManager()

    if args.show or (not args.discover and not args.synthetic):
        sigs = manager.get_all_signatures()
        print(f"\n=== Golden Signatures ({sigs['count']}) ===")
        for sid, sig in sigs["signatures"].items():
            print(f"\n  Scenario: {sid}")
            print(f"  Score: {sig['composite_score']}")
            print(f"  Pareto-optimal batches: {sig['n_pareto_optimal']}")
            if "golden_parameters" in sig:
                print("  Golden Parameters:")
                for p, v in sig["golden_parameters"].items():
                    print(f"    {p}: {v['optimal']} [{v['min_range']}–{v['max_range']}]")

    if args.discover:
        from data.hackathon_adapter import HackathonDataAdapter, HACKATHON_INPUT_COLS, HACKATHON_TARGET_COLS

        print("\n" + "=" * 60)
        print("  DISCOVERING GOLDEN SIGNATURES (Hackathon Data)")
        print("=" * 60)

        adapter = HackathonDataAdapter()
        df = adapter.load_production_data()
        df = adapter.engineer_features(df)

        # Scenario 1: Best quality with lowest waste
        sig1 = manager.discover_signatures(
            df, HACKATHON_INPUT_COLS,
            ["Content_Uniformity", "Dissolution_Rate", "Friability"],
        )
        print(f"\n  Scenario 1 (Quality Focus): score = {sig1['composite_score']}")

        # Scenario 2: Best hardness with optimal weight
        sig2 = manager.discover_signatures(
            df, HACKATHON_INPUT_COLS,
            ["Hardness", "Tablet_Weight", "Moisture_Content"],
        )
        print(f"  Scenario 2 (Physical Focus): score = {sig2['composite_score']}")

        # Scenario 3: All targets
        sig3 = manager.discover_signatures(
            df, HACKATHON_INPUT_COLS,
            HACKATHON_TARGET_COLS,
        )
        print(f"  Scenario 3 (All Targets): score = {sig3['composite_score']}")

    if args.synthetic:
        print("\n" + "=" * 60)
        print("  DISCOVERING GOLDEN SIGNATURES (Synthetic Data)")
        print("=" * 60)

        csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
        if not os.path.exists(csv_path):
            print("  Error: synthetic data not found")
            return

        df = pd.read_csv(csv_path)
        input_cols = [
            "temperature", "conveyor_speed", "hold_time", "batch_size",
            "material_type", "hour_of_day", "operator_exp",
        ]
        target_cols = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]

        sig = manager.discover_signatures(df, input_cols, target_cols)
        print(f"  Synthetic signature score: {sig['composite_score']}")
        print("  Golden Parameters:")
        for p, v in sig["golden_parameters"].items():
            print(f"    {p}: {v['optimal']} [{v['min_range']}–{v['max_range']}]")


if __name__ == "__main__":
    main()
