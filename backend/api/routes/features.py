"""
PlantIQ — Model Features Route
=================================
GET /model/features — Global feature importance across all 4 targets.

Returns mean |SHAP| values per feature for each target, enabling
the dashboard to show which parameters matter most.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ModelFeaturesResponse

router = APIRouter(tags=["model"])


def _get_predictor():
    """Lazy-import the global predictor from main."""
    from main import predictor
    if predictor is None or not predictor.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is starting up.")
    return predictor


@router.get("/model/features", response_model=ModelFeaturesResponse)
async def get_model_features() -> ModelFeaturesResponse:
    """Return global feature importance for all 4 target models.

    Uses the predictor's get_feature_importance (XGBoost built-in)
    to provide fast importance values without needing SHAP computation.
    """
    predictor = _get_predictor()

    try:
        importance = predictor.get_feature_importance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance retrieval failed: {str(e)}")

    # importance is dict[target_name, dict[feature_name, importance_value]]
    return ModelFeaturesResponse(
        quality=importance.get("quality_score", {}),
        yield_importance=importance.get("yield_pct", {}),
        performance=importance.get("performance_pct", {}),
        energy=importance.get("energy_kwh", {}),
    )
