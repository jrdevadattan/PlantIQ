"""
PlantIQ — Golden Signature Routes
====================================
POST /golden-signature/discover — Discover Pareto-optimal golden signatures
POST /golden-signature/compare  — Compare a batch against the golden signature
POST /golden-signature/update   — Self-improve: update if batch exceeds golden
POST /golden-signature/scenario — Multi-objective scenario analysis
GET  /golden-signature/all      — List all stored golden signatures

Implements the Golden Signature Management system from the hackathon
problem statement (Track B — Optimization Engine):
  "A golden signature is a set of optimised features for the given
   multi-objective target."
"""

from __future__ import annotations

import os
import sys
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    GoldenSignatureDiscoverRequest,
    GoldenSignatureCompareRequest,
    GoldenSignatureUpdateRequest,
    ScenarioRequest,
)

logger = logging.getLogger("plantiq.golden_signature")
router = APIRouter(prefix="/golden-signature", tags=["golden-signature"])

# ── Paths for data sources ────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_CSV = os.path.join(BACKEND_DIR, "data", "batch_data.csv")
HACKATHON_DIR = os.path.join(
    os.path.dirname(BACKEND_DIR),
    "69997ffba83f5_problem_statement",
)


def _get_golden_manager():
    """Lazy-import the global GoldenSignatureManager from main."""
    from main import golden_manager
    if golden_manager is None:
        raise HTTPException(status_code=503, detail="GoldenSignatureManager not loaded.")
    return golden_manager


def _load_data_source(source: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load data based on source type ('synthetic' or 'hackathon').

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        (dataframe, input_cols, target_cols)
    """
    if source == "hackathon":
        from data.hackathon_adapter import (
            HackathonDataAdapter,
            HACKATHON_INPUT_COLS,
            HACKATHON_TARGET_COLS,
        )
        adapter = HackathonDataAdapter()
        df = adapter.load_production_data()
        df = adapter.engineer_features(df)
        input_cols = HACKATHON_INPUT_COLS + [
            "granulation_intensity", "drying_intensity", "compression_speed_ratio",
            "binder_per_time", "drying_efficiency", "force_lubricant_ratio",
            "machine_load_index",
        ]
        target_cols = HACKATHON_TARGET_COLS
        return df, input_cols, target_cols
    else:
        # Synthetic data
        if not os.path.exists(SYNTHETIC_CSV):
            raise HTTPException(status_code=404, detail=f"Synthetic data not found: {SYNTHETIC_CSV}")
        df = pd.read_csv(SYNTHETIC_CSV)
        from preprocessing.normalizer import FEATURE_COLS, TARGET_COLS
        return df, FEATURE_COLS, TARGET_COLS


# ──────────────────────────────────────────────────────────────
# POST /golden-signature/discover
# ──────────────────────────────────────────────────────────────
@router.post("/discover")
async def discover_signatures(request: GoldenSignatureDiscoverRequest):
    """Discover Pareto-optimal golden signatures from historical data.

    Analyzes all batches via multi-objective Pareto dominance to find
    the best trade-off parameter sets. Saves results automatically.
    """
    manager = _get_golden_manager()

    try:
        df, input_cols, target_cols = _load_data_source(request.data_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

    try:
        result = manager.discover_signatures(
            df=df,
            input_cols=input_cols,
            target_cols=target_cols,
            n_top=request.n_top,
        )
    except Exception as e:
        logger.error("Golden signature discovery failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

    return {
        "status": "success",
        "data_source": request.data_source,
        **result,
    }


# ──────────────────────────────────────────────────────────────
# POST /golden-signature/compare
# ──────────────────────────────────────────────────────────────
@router.post("/compare")
async def compare_batch(request: GoldenSignatureCompareRequest):
    """Compare a batch against the golden signature.

    Returns alignment score, parameter deviations, target deviations,
    and actionable recommendations to move the batch closer to golden.
    """
    manager = _get_golden_manager()

    try:
        result = manager.compare_batch(
            batch_params=request.batch_params,
            batch_targets=request.batch_targets,
            scenario_id=request.scenario_id,
        )
    except Exception as e:
        logger.error("Golden signature comparison failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    return {"status": "success", **result}


# ──────────────────────────────────────────────────────────────
# POST /golden-signature/update
# ──────────────────────────────────────────────────────────────
@router.post("/update")
async def update_signature(request: GoldenSignatureUpdateRequest):
    """Self-improvement: update the golden signature if this batch is better.

    Implements continuous learning — compares the new batch's composite
    score against the existing golden signature and blends (EMA 70/30)
    if superior.
    """
    manager = _get_golden_manager()

    try:
        result = manager.update_if_better(
            batch_params=request.batch_params,
            batch_targets=request.batch_targets,
            target_cols=request.target_cols,
        )
    except Exception as e:
        logger.error("Golden signature update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

    return {"status": "success", **result}


# ──────────────────────────────────────────────────────────────
# POST /golden-signature/scenario
# ──────────────────────────────────────────────────────────────
@router.post("/scenario")
async def scenario_analysis(request: ScenarioRequest):
    """Multi-objective scenario analysis.

    Generates golden signature optimized for specific target priorities.
    Examples:
      - primary: [quality_score, yield_pct], secondary: [energy_kwh]
      - primary: [Hardness, Dissolution_Rate], secondary: [Friability]
    """
    manager = _get_golden_manager()

    try:
        df, input_cols, target_cols = _load_data_source(request.data_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

    # Validate that requested targets exist in the data
    all_targets = set(target_cols)
    for t in request.primary_targets:
        if t not in all_targets:
            raise HTTPException(
                status_code=400,
                detail=f"Target '{t}' not found. Available: {sorted(all_targets)}",
            )
    if request.secondary_targets:
        for t in request.secondary_targets:
            if t not in all_targets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Secondary target '{t}' not found. Available: {sorted(all_targets)}",
                )

    try:
        result = manager.get_scenario_recommendations(
            df=df,
            input_cols=input_cols,
            primary_targets=request.primary_targets,
            secondary_targets=request.secondary_targets,
            primary_weight=request.primary_weight,
        )
    except Exception as e:
        logger.error("Scenario analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")

    return {"status": "success", "data_source": request.data_source, **result}


# ──────────────────────────────────────────────────────────────
# GET /golden-signature/all
# ──────────────────────────────────────────────────────────────
@router.get("/all")
async def get_all_signatures():
    """List all stored golden signatures and recent update history."""
    manager = _get_golden_manager()
    return {"status": "success", **manager.get_all_signatures()}
