"""
PlantIQ — Health Route
========================
GET /health — Server status and model readiness check.
"""

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check that the server is running and models are loaded.

    Returns status, model-loaded flag, and API version.
    Used by the frontend Sidebar system-status indicator.
    """
    # Import here to check if models are actually loaded
    from main import predictor, shap_explainer

    models_loaded = (
        predictor is not None
        and predictor.is_fitted
        and shap_explainer is not None
    )

    return HealthResponse(
        status="running",
        models_loaded=models_loaded,
        version="1.0.0",
    )
