"""
PlantIQ — SHAP Explanation Route
===================================
GET /explain/{batch_id}?target=energy — SHAP feature contributions for a batch.

Since batches are predicted on-the-fly (not stored in a DB yet), this route
accepts batch parameters as query params and generates the explanation
in real-time using the ShapExplainer + PlainEnglishConverter.

When a database layer is added, it will resolve batch_id to stored params.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    ExplainResponse,
    FeatureContribution,
    BatchPredictionRequest,
)

router = APIRouter(tags=["explainability"])


def _get_shap_explainer():
    """Lazy-import the global ShapExplainer from main."""
    from main import shap_explainer
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded. Server is starting up.")
    return shap_explainer


def _get_plain_english_converter():
    """Lazy-import the PlainEnglishConverter from main."""
    from main import english_converter
    if english_converter is None:
        raise HTTPException(status_code=503, detail="Plain English converter not loaded.")
    return english_converter


@router.post("/explain/{batch_id}", response_model=ExplainResponse)
async def explain_batch(
    batch_id: str,
    params: BatchPredictionRequest,
    target: str = Query(
        default="energy_kwh",
        description="Target to explain: quality_score, yield_pct, performance_pct, energy_kwh",
    ),
) -> ExplainResponse:
    """Generate SHAP explanation for a batch prediction.

    Accepts batch parameters in the request body (same as /predict/batch)
    and returns per-feature SHAP contributions with plain-English summaries.

    Notes
    -----
    - Uses POST instead of GET because we need batch params in the body
      until a database layer is added.
    - The batch_id is included for response tracking / consistency with
      the README API spec.
    """
    valid_targets = ["quality_score", "yield_pct", "performance_pct", "energy_kwh"]
    if target not in valid_targets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target '{target}'. Must be one of {valid_targets}",
        )

    explainer = _get_shap_explainer()
    converter = _get_plain_english_converter()

    # Convert request to dict
    batch_params = params.model_dump()

    try:
        explanation = explainer.explain_single(batch_params, target=target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

    # Target unit mapping
    TARGET_UNITS = {
        "quality_score": "%",
        "yield_pct": "%",
        "performance_pct": "%",
        "energy_kwh": "kWh",
    }
    target_unit = TARGET_UNITS.get(target, "")

    # Generate plain-English descriptions for each contribution
    contributions: list[FeatureContribution] = []
    for fc in explanation["feature_contributions"]:
        # Generate plain English sentence for this feature
        plain = converter.feature_sentence(
            feature=fc["feature"],
            value=fc["value"],
            contribution=fc["contribution"],
            target=target,
            unit=target_unit,
        )

        contributions.append(FeatureContribution(
            feature=fc["feature"],
            value=fc["value"],
            contribution=fc["contribution"],
            direction=fc["direction"],
            plain_english=plain,
        ))

    # Generate overall summary
    summary = converter.generate_summary(
        target=target,
        prediction=explanation["final_prediction"],
        baseline=explanation["baseline_prediction"],
        unit=explanation.get("unit", ""),
        contributions=[
            {"feature": c.feature, "contribution": c.contribution, "value": c.value}
            for c in contributions
        ],
    )

    return ExplainResponse(
        batch_id=batch_id,
        target=target,
        baseline_prediction=explanation["baseline_prediction"],
        final_prediction=explanation["final_prediction"],
        feature_contributions=contributions,
        summary=summary,
    )
