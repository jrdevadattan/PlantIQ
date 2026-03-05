"""
PlantIQ — Adaptive Target Routes
===================================
POST /targets/initialize — Initialize baseline from historical data
POST /targets/batch      — Get adaptive targets for next batch
POST /targets/assess     — Assess a completed batch against targets
GET  /targets/report     — Get comprehensive performance report

Implements the Adaptive Target Setting Engine from the hackathon
problem statement (Universal Objective #1):
  "Establish dynamic carbon emission targets aligned with regulatory
   and organizational requirements."
"""

from __future__ import annotations

import os
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    AdaptiveTargetInitRequest,
    BatchTargetRequest,
    BatchAssessRequest,
)

logger = logging.getLogger("plantiq.targets")
router = APIRouter(prefix="/targets", tags=["adaptive-targets"])

# ── Paths ────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_CSV = os.path.join(BACKEND_DIR, "data", "batch_data.csv")


def _get_target_engine():
    """Lazy-import the global AdaptiveTargetEngine from main."""
    from main import target_engine
    if target_engine is None:
        raise HTTPException(status_code=503, detail="AdaptiveTargetEngine not loaded.")
    return target_engine


# ──────────────────────────────────────────────────────────────
# POST /targets/initialize
# ──────────────────────────────────────────────────────────────
@router.post("/initialize")
async def initialize_targets(request: AdaptiveTargetInitRequest):
    """Initialize or re-initialize baseline targets from historical data.

    Computes statistical baselines from batch history (mean, median, P75, P90)
    and sets adaptive target thresholds. Can use synthetic or hackathon data.
    """
    engine = _get_target_engine()

    try:
        if request.data_source == "hackathon":
            from data.hackathon_adapter import HackathonDataAdapter
            adapter = HackathonDataAdapter()
            df = adapter.load_production_data()
            # Map hackathon columns to expected format
            # Use Friability as energy proxy (lower is better)
            baseline = engine.initialize_from_data(
                df=df,
                energy_col="Friability",
                quality_col="Content_Uniformity",
                yield_col="Dissolution_Rate",
                performance_col="Hardness",
            )
        else:
            if not os.path.exists(SYNTHETIC_CSV):
                raise HTTPException(
                    status_code=404,
                    detail=f"Synthetic data not found: {SYNTHETIC_CSV}",
                )
            df = pd.read_csv(SYNTHETIC_CSV)
            baseline = engine.initialize_from_data(df=df)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Target initialization failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

    return {
        "status": "success",
        "data_source": request.data_source,
        "baseline": baseline,
    }


# ──────────────────────────────────────────────────────────────
# POST /targets/batch
# ──────────────────────────────────────────────────────────────
@router.post("/batch")
async def get_batch_targets(request: BatchTargetRequest):
    """Compute adaptive targets for the next batch.

    Blends three target-setting strategies:
      1. Regulatory cap (hard limit from compliance)
      2. Rolling benchmark (P75 of recent N batches — stretch goal)
      3. Annual reduction factor (5% year-over-year improvement)

    Returns per-metric targets that operators should aim for.
    """
    engine = _get_target_engine()

    if not engine.baseline:
        raise HTTPException(
            status_code=400,
            detail="Engine not initialized. Call POST /targets/initialize first.",
        )

    try:
        targets = engine.get_batch_targets(
            current_batch_number=request.current_batch_number,
            annual_batches=request.annual_batches,
        )
    except Exception as e:
        logger.error("Target computation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Target computation failed: {str(e)}")

    return {"status": "success", "targets": targets}


# ──────────────────────────────────────────────────────────────
# POST /targets/assess
# ──────────────────────────────────────────────────────────────
@router.post("/assess")
async def assess_batch(request: BatchAssessRequest):
    """Assess a completed batch against adaptive targets.

    Returns:
      - Energy/carbon compliance status (on_track / caution / exceeded)
      - Savings vs. target (kWh, kg CO₂, USD)
      - Quality/yield/performance vs. baselines
      - Actionable recommendations
    """
    engine = _get_target_engine()

    if not engine.baseline:
        raise HTTPException(
            status_code=400,
            detail="Engine not initialized. Call POST /targets/initialize first.",
        )

    try:
        assessment = engine.assess_batch(
            energy_kwh=request.energy_kwh,
            quality_score=request.quality_score,
            yield_pct=request.yield_pct,
            performance_pct=request.performance_pct,
            batch_number=request.batch_number,
        )
    except Exception as e:
        logger.error("Batch assessment failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

    return {"status": "success", **assessment}


# ──────────────────────────────────────────────────────────────
# GET /targets/report
# ──────────────────────────────────────────────────────────────
@router.get("/report")
async def performance_report():
    """Get comprehensive performance report from rolling history.

    Returns aggregate statistics: energy/carbon trends, compliance
    percentages, cumulative savings, and trend direction.
    """
    engine = _get_target_engine()

    try:
        report = engine.get_performance_report()
    except Exception as e:
        logger.error("Performance report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    return {"status": "success", **report}
