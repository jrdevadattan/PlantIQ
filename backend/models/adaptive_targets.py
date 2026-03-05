"""
PlantIQ — Adaptive Target Setting Engine
==========================================
Implements dynamic carbon emission and energy targets as specified in
the hackathon problem statement:

  "Integrate regulatory requirements and sustainability commitments
   into dynamic goal-setting mechanisms."
  "Establish dynamic carbon emission targets aligned with regulatory
   and organizational requirements."

Features:
  1. Regulatory Framework — Define carbon intensity limits, reduction targets
  2. Dynamic Targets — Adaptive batch-level goals based on historical performance
  3. Rolling Benchmarks — Moving-window performance tracking
  4. Compliance Monitoring — Real-time check against regulatory thresholds
  5. Carbon Budget Allocation — Per-batch carbon budgets from annual limits
  6. Business Impact — Energy savings (kWh/batch), emission reductions (kg CO₂e)

Spec: Problem Statement — Universal Objectives #1
       README — "Adaptive carbon targets based on rolling performance"
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime


BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "models", "trained")
TARGETS_FILE = os.path.join(ARTIFACT_DIR, "adaptive_targets.json")


# ──────────────────────────────────────────────
# Default Regulatory & Sustainability Configuration
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    # CO₂ emission factor (kg CO₂e per kWh of electricity)
    "co2_factor_kg_per_kwh": 0.82,

    # Annual carbon reduction target (% reduction year-over-year)
    "annual_reduction_target_pct": 5.0,

    # Maximum allowed energy per batch (kWh) — regulatory limit
    "max_energy_per_batch_kwh": 55.0,

    # Maximum allowed CO₂ per batch (kg) — derived from energy limit
    "max_co2_per_batch_kg": 45.1,  # 55 × 0.82

    # Rolling window for performance benchmarking (number of batches)
    "rolling_window": 50,

    # Target percentile for adaptive goals (75th = stretch goal)
    "target_percentile": 75,

    # Severity thresholds for carbon budget status
    "carbon_budget_thresholds": {
        "on_track": 0.90,    # <= 90% of budget = green
        "caution": 1.0,      # 90–100% = amber
        "exceeded": 1.10,    # > 100% = red
    },

    # Energy cost per kWh (for ROI calculations)
    "energy_cost_per_kwh": 0.12,  # USD

    # Carbon credit cost per kg CO₂e
    "carbon_credit_cost_per_kg": 0.05,  # USD
}


class AdaptiveTargetEngine:
    """Engine for dynamic, regulation-aware energy and carbon targets.

    Computes adaptive batch-level targets based on:
      - Regulatory carbon limits (annual/per-batch caps)
      - Historical rolling performance (moving benchmarks)
      - Sustainability reduction commitments (year-over-year goals)
      - Real-time batch performance vs. targets

    Attributes
    ----------
    config : dict
        Regulatory and sustainability configuration.
    baseline : dict
        Historical baseline metrics (set from training data).
    rolling_history : list
        Recent batch performance for rolling benchmarks.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.baseline: dict = {}
        self.rolling_history: list = []
        self._load_if_exists()

    def _load_if_exists(self):
        """Load saved targets from disk."""
        if os.path.exists(TARGETS_FILE):
            try:
                with open(TARGETS_FILE, "r") as f:
                    data = json.load(f)
                self.baseline = data.get("baseline", {})
                self.rolling_history = data.get("rolling_history", [])
                self.config.update(data.get("config_overrides", {}))
            except (json.JSONDecodeError, IOError):
                pass

    def save(self):
        """Persist adaptive targets to disk."""
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        data = {
            "baseline": self.baseline,
            "rolling_history": self.rolling_history[-200:],  # Keep last 200
            "config_overrides": {},
            "last_updated": datetime.now().isoformat(),
        }
        with open(TARGETS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ──────────────────────────────────────────────
    # Baseline Initialization
    # ──────────────────────────────────────────────

    def initialize_from_data(
        self,
        df: pd.DataFrame,
        energy_col: str = "energy_kwh",
        quality_col: str = "quality_score",
        yield_col: str = "yield_pct",
        performance_col: str = "performance_pct",
    ) -> dict:
        """Initialize baseline targets from historical batch data.

        Parameters
        ----------
        df : pd.DataFrame
            Historical batch data with energy and quality columns.
        energy_col, quality_col, yield_col, performance_col : str
            Column names for the relevant metrics.

        Returns
        -------
        dict
            Computed baseline targets.
        """
        baseline = {}

        for col_name, col in [
            ("energy", energy_col),
            ("quality", quality_col),
            ("yield", yield_col),
            ("performance", performance_col),
        ]:
            if col not in df.columns:
                continue

            values = df[col].dropna().values
            baseline[col_name] = {
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "p25": round(float(np.percentile(values, 25)), 4),
                "p75": round(float(np.percentile(values, 75)), 4),
                "p90": round(float(np.percentile(values, 90)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "n_samples": len(values),
            }

        # CO₂ baseline
        if energy_col in df.columns:
            co2_values = df[energy_col].dropna().values * self.config["co2_factor_kg_per_kwh"]
            baseline["co2"] = {
                "mean": round(float(np.mean(co2_values)), 4),
                "median": round(float(np.median(co2_values)), 4),
                "p75": round(float(np.percentile(co2_values, 75)), 4),
                "annual_estimate_kg": round(float(np.mean(co2_values) * 365 * 3), 2),  # ~3 batches/day
            }

        self.baseline = baseline
        self.save()

        print(f"[AdaptiveTarget] Initialized baseline from {len(df)} batches:")
        if "energy" in baseline:
            print(f"  Energy: mean = {baseline['energy']['mean']:.2f} kWh, "
                  f"P75 = {baseline['energy']['p75']:.2f} kWh")
        if "co2" in baseline:
            print(f"  CO₂: mean = {baseline['co2']['mean']:.2f} kg/batch, "
                  f"annual est. = {baseline['co2']['annual_estimate_kg']:.0f} kg/year")

        return baseline

    # ──────────────────────────────────────────────
    # Per-Batch Adaptive Target Computation
    # ──────────────────────────────────────────────

    def get_batch_targets(
        self,
        current_batch_number: Optional[int] = None,
        annual_batches: int = 1000,
    ) -> dict:
        """Compute adaptive targets for the next batch.

        Targets are dynamically adjusted based on:
          1. Regulatory cap (hard limit)
          2. Rolling performance (soft target = P75 of recent batches)
          3. Annual reduction commitment (progressively tighter through the year)

        Parameters
        ----------
        current_batch_number : int, optional
            Batch number within the year (for progressive tightening).
        annual_batches : int
            Expected total batches per year.

        Returns
        -------
        dict
            Adaptive targets for energy, CO₂, and quality metrics.
        """
        targets = {}

        # --- Energy Target ---
        regulatory_cap = self.config["max_energy_per_batch_kwh"]
        baseline_energy = self.baseline.get("energy", {}).get("mean", 35.0)

        # Rolling benchmark (if we have history)
        if len(self.rolling_history) >= 10:
            recent_energy = [h.get("energy_kwh", baseline_energy)
                            for h in self.rolling_history[-self.config["rolling_window"]:]]
            rolling_p75 = float(np.percentile(recent_energy, self.config["target_percentile"]))
        else:
            rolling_p75 = self.baseline.get("energy", {}).get("p75", baseline_energy)

        # Progressive reduction through the year
        reduction_pct = self.config["annual_reduction_target_pct"]
        if current_batch_number is not None:
            progress = current_batch_number / max(annual_batches, 1)
            reduction_factor = 1.0 - (reduction_pct / 100) * progress
        else:
            reduction_factor = 1.0 - (reduction_pct / 200)  # Mid-year default

        adaptive_energy = min(
            regulatory_cap,                          # Hard cap
            rolling_p75 * 0.95,                      # 5% below rolling P75
            baseline_energy * reduction_factor,       # Annual reduction path
        )

        targets["energy_kwh"] = {
            "target": round(float(adaptive_energy), 2),
            "regulatory_cap": regulatory_cap,
            "rolling_benchmark": round(float(rolling_p75), 2),
            "reduction_factor": round(float(reduction_factor), 4),
            "baseline": round(float(baseline_energy), 2),
        }

        # --- CO₂ Target ---
        co2_factor = self.config["co2_factor_kg_per_kwh"]
        targets["co2_kg"] = {
            "target": round(float(adaptive_energy * co2_factor), 2),
            "regulatory_cap": round(float(regulatory_cap * co2_factor), 2),
            "baseline": round(float(baseline_energy * co2_factor), 2),
        }

        # --- Quality Targets (maintain minimum while reducing energy) ---
        for metric in ["quality", "yield", "performance"]:
            if metric in self.baseline:
                # Quality targets: at least baseline P75 (stretch goal)
                targets[metric] = {
                    "target": self.baseline[metric]["p75"],
                    "minimum_acceptable": self.baseline[metric]["p25"],
                    "baseline_mean": self.baseline[metric]["mean"],
                }

        return targets

    # ──────────────────────────────────────────────
    # Batch Assessment Against Targets
    # ──────────────────────────────────────────────

    def assess_batch(
        self,
        energy_kwh: float,
        quality_score: Optional[float] = None,
        yield_pct: Optional[float] = None,
        performance_pct: Optional[float] = None,
        batch_number: Optional[int] = None,
    ) -> dict:
        """Assess a completed batch against adaptive targets.

        Parameters
        ----------
        energy_kwh : float
            Actual energy consumption of the batch.
        quality_score, yield_pct, performance_pct : float, optional
            Actual quality metrics.
        batch_number : int, optional
            Batch number in the year.

        Returns
        -------
        dict
            Assessment with compliance status, savings, and recommendations.
        """
        targets = self.get_batch_targets(batch_number)
        co2_factor = self.config["co2_factor_kg_per_kwh"]
        co2_kg = energy_kwh * co2_factor

        # Energy assessment
        energy_target = targets["energy_kwh"]["target"]
        energy_cap = targets["energy_kwh"]["regulatory_cap"]
        energy_saved = energy_target - energy_kwh  # Positive = saved
        co2_saved = energy_saved * co2_factor

        # Budget status
        energy_ratio = energy_kwh / max(energy_target, 0.01)
        thresholds = self.config["carbon_budget_thresholds"]

        if energy_ratio <= thresholds["on_track"]:
            budget_status = "on_track"
            budget_color = "green"
        elif energy_ratio <= thresholds["caution"]:
            budget_status = "caution"
            budget_color = "amber"
        else:
            budget_status = "exceeded"
            budget_color = "red"

        assessment = {
            "energy": {
                "actual_kwh": round(float(energy_kwh), 2),
                "target_kwh": energy_target,
                "regulatory_cap_kwh": energy_cap,
                "savings_kwh": round(float(energy_saved), 2),
                "under_target": energy_kwh <= energy_target,
                "under_regulatory": energy_kwh <= energy_cap,
            },
            "carbon": {
                "actual_kg": round(float(co2_kg), 2),
                "target_kg": targets["co2_kg"]["target"],
                "savings_kg": round(float(co2_saved), 2),
                "budget_status": budget_status,
                "budget_color": budget_color,
            },
            "business_impact": {
                "energy_cost_saved": round(float(energy_saved * self.config["energy_cost_per_kwh"]), 2),
                "carbon_credit_saved": round(float(co2_saved * self.config["carbon_credit_cost_per_kg"]), 2),
                "total_savings_usd": round(float(
                    energy_saved * self.config["energy_cost_per_kwh"] +
                    co2_saved * self.config["carbon_credit_cost_per_kg"]
                ), 2),
            },
            "recommendations": [],
        }

        # Quality assessments
        quality_metrics = {
            "quality": quality_score,
            "yield": yield_pct,
            "performance": performance_pct,
        }
        for metric, value in quality_metrics.items():
            if value is not None and metric in targets:
                target = targets[metric]
                assessment[metric] = {
                    "actual": round(float(value), 2),
                    "target": target["target"],
                    "minimum": target["minimum_acceptable"],
                    "meets_target": value >= target["target"],
                    "meets_minimum": value >= target["minimum_acceptable"],
                }
                if value < target["minimum_acceptable"]:
                    assessment["recommendations"].append(
                        f"{metric.title()} ({value:.1f}) is below minimum "
                        f"({target['minimum_acceptable']:.1f}). Investigate process parameters."
                    )

        # Energy recommendations
        if budget_status == "exceeded":
            assessment["recommendations"].append(
                f"Energy ({energy_kwh:.1f} kWh) exceeds target ({energy_target:.1f} kWh). "
                f"Review conveyor speed and hold time for optimization."
            )
        elif budget_status == "on_track":
            assessment["recommendations"].append(
                f"Energy on track. {abs(energy_saved):.1f} kWh below target. "
                f"Continue current operating parameters."
            )

        # Update rolling history
        self.rolling_history.append({
            "energy_kwh": round(float(energy_kwh), 2),
            "co2_kg": round(float(co2_kg), 2),
            "quality": quality_score,
            "yield": yield_pct,
            "performance": performance_pct,
            "batch_number": batch_number,
            "timestamp": datetime.now().isoformat(),
            "budget_status": budget_status,
        })
        self.save()

        return assessment

    # ──────────────────────────────────────────────
    # Reporting & Analytics
    # ──────────────────────────────────────────────

    def get_performance_report(self) -> dict:
        """Generate a comprehensive performance report from rolling history.

        Returns
        -------
        dict
            Report with trends, compliance rate, cumulative savings.
        """
        if not self.rolling_history:
            return {"error": "No batch history available"}

        history = self.rolling_history
        energy_values = [h["energy_kwh"] for h in history if "energy_kwh" in h]
        co2_values = [h["co2_kg"] for h in history if "co2_kg" in h]

        # Compliance rates
        statuses = [h.get("budget_status", "unknown") for h in history]
        n_on_track = statuses.count("on_track")
        n_caution = statuses.count("caution")
        n_exceeded = statuses.count("exceeded")

        report = {
            "total_batches": len(history),
            "energy_stats": {
                "mean_kwh": round(float(np.mean(energy_values)), 2) if energy_values else 0,
                "median_kwh": round(float(np.median(energy_values)), 2) if energy_values else 0,
                "total_kwh": round(float(sum(energy_values)), 2),
                "trend": self._compute_trend(energy_values),
            },
            "carbon_stats": {
                "total_kg": round(float(sum(co2_values)), 2) if co2_values else 0,
                "mean_kg_per_batch": round(float(np.mean(co2_values)), 2) if co2_values else 0,
            },
            "compliance": {
                "on_track_pct": round(n_on_track / max(len(history), 1) * 100, 1),
                "caution_pct": round(n_caution / max(len(history), 1) * 100, 1),
                "exceeded_pct": round(n_exceeded / max(len(history), 1) * 100, 1),
            },
            "cumulative_savings": {
                "energy_kwh": round(float(
                    sum(self.config["max_energy_per_batch_kwh"] - e for e in energy_values)
                ), 2) if energy_values else 0,
                "co2_kg": round(float(
                    sum(self.config["max_co2_per_batch_kg"] - c for c in co2_values)
                ), 2) if co2_values else 0,
            },
        }

        return report

    def _compute_trend(self, values: list) -> str:
        """Determine if values are trending up, down, or stable."""
        if len(values) < 5:
            return "insufficient_data"

        recent = values[-10:]
        older = values[-20:-10] if len(values) >= 20 else values[:len(values)//2]

        if not older:
            return "insufficient_data"

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        change_pct = (recent_mean - older_mean) / max(abs(older_mean), 0.01) * 100

        if change_pct < -2:
            return "improving"  # Energy decreasing = good
        elif change_pct > 2:
            return "degrading"
        else:
            return "stable"


# ──────────────────────────────────────────────────────────────────
# Convenience Functions
# ──────────────────────────────────────────────────────────────────

def initialize_adaptive_targets(csv_path: Optional[str] = None) -> AdaptiveTargetEngine:
    """Create and initialize an AdaptiveTargetEngine from batch data.

    Parameters
    ----------
    csv_path : str, optional
        Path to batch_data.csv. Defaults to synthetic data.

    Returns
    -------
    AdaptiveTargetEngine
        Initialized engine with baselines computed.
    """
    if csv_path is None:
        csv_path = os.path.join(BACKEND_DIR, "data", "batch_data.csv")

    engine = AdaptiveTargetEngine()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        engine.initialize_from_data(df)
    else:
        print(f"[AdaptiveTarget] Warning: no data found at {csv_path}")

    return engine


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    """CLI: Initialize and demonstrate adaptive target setting."""
    import argparse

    parser = argparse.ArgumentParser(description="PlantIQ — Adaptive Target Setting")
    parser.add_argument("--init", action="store_true", help="Initialize from synthetic data")
    parser.add_argument("--targets", action="store_true", help="Show current adaptive targets")
    parser.add_argument("--assess", type=float, help="Assess a batch with given energy (kWh)")
    parser.add_argument("--report", action="store_true", help="Show performance report")
    args = parser.parse_args()

    engine = initialize_adaptive_targets()

    if args.init or not any([args.targets, args.assess, args.report]):
        print("\n  Adaptive targets initialized.")

    if args.targets:
        targets = engine.get_batch_targets()
        print("\n=== Current Adaptive Targets ===")
        for metric, target in targets.items():
            print(f"\n  {metric}:")
            for k, v in target.items():
                print(f"    {k}: {v}")

    if args.assess:
        result = engine.assess_batch(energy_kwh=args.assess, quality_score=92, yield_pct=93)
        print(f"\n=== Batch Assessment (Energy: {args.assess} kWh) ===")
        print(f"  Budget status: {result['carbon']['budget_status']}")
        print(f"  Energy saved: {result['energy']['savings_kwh']} kWh")
        print(f"  CO₂ saved: {result['carbon']['savings_kg']} kg")
        print(f"  Cost saved: ${result['business_impact']['total_savings_usd']}")
        for rec in result["recommendations"]:
            print(f"  → {rec}")

    if args.report:
        report = engine.get_performance_report()
        print("\n=== Performance Report ===")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
