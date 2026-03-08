"""
PlantIQ — Pydantic Request / Response Schemas
================================================
All API request and response models.  Defines the contract between
frontend and backend, matching the README API Reference exactly.

Every schema uses Pydantic v2 (BaseModel) with Field validators
for domain-specific ranges from the manufacturing context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "running"
    models_loaded: bool = True
    version: str = "1.0.0"


# ══════════════════════════════════════════════════════════════
# Predict — Batch
# ══════════════════════════════════════════════════════════════

class BatchPredictionRequest(BaseModel):
    """POST /predict/batch request — 7 operator-controlled inputs."""
    temperature: float = Field(..., ge=175.0, le=195.0, description="Temperature in °C (175–195)")
    conveyor_speed: float = Field(..., ge=60.0, le=95.0, description="Conveyor speed % (60–95)")
    hold_time: float = Field(..., ge=10.0, le=30.0, description="Hold time in minutes (10–30)")
    batch_size: float = Field(..., ge=300.0, le=700.0, description="Batch size in kg (300–700)")
    material_type: int = Field(..., ge=0, le=2, description="Material type: 0=TypeA, 1=TypeB, 2=TypeC")
    hour_of_day: int = Field(..., ge=6, le=21, description="Hour of day (6–21)")
    operator_exp: int = Field(..., ge=0, le=2, description="Operator experience: 0=junior, 1=mid, 2=senior")


class PredictionValues(BaseModel):
    """Predicted target values."""
    quality_score: float
    yield_pct: float
    performance_pct: float
    energy_kwh: float
    co2_kg: float


class ConfidenceInterval(BaseModel):
    """Confidence interval for a target."""
    lower: float
    upper: float


class CarbonBudget(BaseModel):
    """Carbon budget status for a batch."""
    batch_budget_kg: float
    predicted_usage_kg: float
    status: str  # ON_TRACK, WARNING, OVER_BUDGET
    headroom_kg: float


class BatchPredictionResponse(BaseModel):
    """POST /predict/batch response."""
    batch_id: str
    predictions: PredictionValues
    confidence_intervals: dict[str, ConfidenceInterval]
    carbon_budget: CarbonBudget


# ══════════════════════════════════════════════════════════════
# Predict — Realtime
# ══════════════════════════════════════════════════════════════

class PartialBatchData(BaseModel):
    """Partial data collected mid-batch."""
    elapsed_minutes: float = Field(..., ge=0, le=30, description="Minutes elapsed since batch start")
    energy_so_far: float = Field(..., ge=0, description="kWh consumed so far")
    avg_power_kw: float = Field(..., ge=0, description="Average power draw in kW")
    anomaly_events: int = Field(default=0, ge=0, description="Number of anomaly events detected")


class RealtimePredictionRequest(BaseModel):
    """POST /predict/realtime request."""
    original_params: BatchPredictionRequest
    partial_data: PartialBatchData


class AlertInfo(BaseModel):
    """Alert generated during realtime monitoring."""
    severity: str  # NORMAL, WATCH, WARNING, CRITICAL
    message: str
    recommended_action: Optional[str] = None
    estimated_saving_kwh: Optional[float] = None
    quality_impact_pct: Optional[float] = None


class RealtimePredictionResponse(BaseModel):
    """POST /predict/realtime response."""
    progress_pct: float
    updated_predictions: PredictionValues
    confidence: str
    alert: Optional[AlertInfo] = None


# ══════════════════════════════════════════════════════════════
# Anomaly Detection
# ══════════════════════════════════════════════════════════════

class AnomalyDetectRequest(BaseModel):
    """POST /anomaly/detect request."""
    batch_id: str = Field(..., description="Batch identifier")
    power_readings: list[float] = Field(..., min_length=1, description="Power curve readings (kW)")
    elapsed_seconds: int = Field(..., ge=0, description="Seconds elapsed since batch start")


class DiagnosisInfo(BaseModel):
    """Fault diagnosis details."""
    fault_type: str
    confidence: float
    human_readable: str
    recommended_action: str
    estimated_energy_impact_kwh: Optional[float] = None
    estimated_quality_impact_pct: Optional[float] = None


class AnomalyDetectResponse(BaseModel):
    """POST /anomaly/detect response."""
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    severity: str
    diagnosis: DiagnosisInfo


# ══════════════════════════════════════════════════════════════
# SHAP Explanation
# ══════════════════════════════════════════════════════════════

class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution."""
    feature: str
    value: float
    contribution: float
    direction: str
    plain_english: str


class ExplainResponse(BaseModel):
    """GET /explain/{batch_id} response."""
    batch_id: str
    target: str
    baseline_prediction: float
    final_prediction: float
    feature_contributions: list[FeatureContribution]
    summary: str


# ══════════════════════════════════════════════════════════════
# Model Features (global importance)
# ══════════════════════════════════════════════════════════════

class ModelFeaturesResponse(BaseModel):
    """GET /model/features response."""
    energy: dict[str, float]
    quality: dict[str, float]
    yield_importance: dict[str, float] = Field(alias="yield")
    performance: dict[str, float]

    model_config = {"populate_by_name": True}


# ══════════════════════════════════════════════════════════════
# Golden Signature
# ══════════════════════════════════════════════════════════════

class GoldenSignatureDiscoverRequest(BaseModel):
    """POST /golden-signature/discover request."""
    data_source: str = Field(
        default="synthetic",
        description="Data source: 'synthetic' or 'hackathon'",
    )
    n_top: int = Field(default=5, ge=1, le=20, description="Number of top Pareto-optimal batches")


class GoldenSignatureCompareRequest(BaseModel):
    """POST /golden-signature/compare request."""
    batch_params: dict = Field(..., description="Batch input parameters")
    batch_targets: dict = Field(..., description="Batch target outcomes")
    scenario_id: Optional[str] = Field(default=None, description="Specific scenario to compare against")


class GoldenSignatureUpdateRequest(BaseModel):
    """POST /golden-signature/update request."""
    batch_params: dict = Field(..., description="Batch input parameters")
    batch_targets: dict = Field(..., description="Batch target outcomes")
    target_cols: list[str] = Field(..., description="Target column names")


class ScenarioRequest(BaseModel):
    """POST /golden-signature/scenario request."""
    primary_targets: list[str] = Field(..., min_length=1, description="Primary optimization targets")
    secondary_targets: Optional[list[str]] = Field(default=None, description="Secondary targets")
    primary_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for primary targets")
    data_source: str = Field(default="synthetic", description="Data source: 'synthetic' or 'hackathon'")


# ══════════════════════════════════════════════════════════════
# Adaptive Targets
# ══════════════════════════════════════════════════════════════

class AdaptiveTargetInitRequest(BaseModel):
    """POST /targets/initialize request."""
    data_source: str = Field(
        default="synthetic",
        description="Data source: 'synthetic' or 'hackathon'",
    )


class BatchTargetRequest(BaseModel):
    """POST /targets/batch request."""
    current_batch_number: Optional[int] = Field(default=None, ge=1, description="Current batch number")
    annual_batches: int = Field(default=1000, ge=100, le=10000, description="Estimated annual batch count")


class BatchAssessRequest(BaseModel):
    """POST /targets/assess request."""
    energy_kwh: float = Field(..., ge=0, description="Actual energy consumed (kWh)")
    quality_score: Optional[float] = Field(default=None, ge=0, le=100)
    yield_pct: Optional[float] = Field(default=None, ge=0, le=100)
    performance_pct: Optional[float] = Field(default=None, ge=0, le=100)
    batch_number: Optional[int] = Field(default=None, ge=1)


# ══════════════════════════════════════════════════════════════
# Hackathon Data
# ══════════════════════════════════════════════════════════════

class HackathonTrainRequest(BaseModel):
    """POST /hackathon/train request."""
    verbose: bool = Field(default=True, description="Print training progress")


class HackathonPredictRequest(BaseModel):
    """POST /hackathon/predict request — 7 pharma inputs."""
    granulation_time: float = Field(..., ge=20, le=60, description="Granulation time (min)")
    binder_amount: float = Field(..., ge=3.0, le=8.0, description="Binder amount (kg)")
    drying_temp: float = Field(..., ge=40, le=70, description="Drying temperature (°C)")
    drying_time: float = Field(..., ge=20, le=60, description="Drying time (min)")
    compression_force: float = Field(..., ge=10, le=30, description="Compression force (kN)")
    machine_speed: float = Field(..., ge=20, le=60, description="Machine speed (rpm)")
    lubricant_conc: float = Field(..., ge=0.3, le=1.5, description="Lubricant concentration (%)")


# ══════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════

class DashboardSummaryResponse(BaseModel):
    """GET /dashboard/summary response — KPI card aggregates."""
    total_batches: int
    running_count: int
    avg_energy: float
    avg_quality: float
    avg_yield: float
    avg_performance: float
    anomaly_count: int
    resolved_count: int
    model_accuracy: float
    mape_pct: float
    energy_trend: str = ""       # "up" | "down" | ""
    energy_trend_value: str = "" # e.g. "4.2%"
    quality_trend: str = ""
    quality_trend_value: str = ""
    yield_trend: str = ""
    yield_trend_value: str = ""


class DailyEnergyItem(BaseModel):
    """Single day entry for daily energy bar chart."""
    day: str           # Short day name: Mon, Tue, ...
    kwh: float         # Total energy for the day
    date: str          # ISO date string
    batch_count: int   # Number of batches that day


class DashboardBatchRecord(BaseModel):
    """Batch record shaped for the frontend RecentBatches table."""
    id: str
    timestamp: str
    temperature: float
    conveyorSpeed: float
    holdTime: float
    batchSize: float
    materialType: int
    hourOfDay: int
    qualityScore: float
    yieldPct: float
    performancePct: float
    energyKwh: float
    status: str         # completed | running | scheduled | alert
    anomalyScore: float


class ShiftPerformanceItem(BaseModel):
    """Per-shift aggregated performance metrics."""
    shift: str        # "Morning (6-14)", "Afternoon (14-22)", "Night (22-6)"
    quality: float
    yield_pct: float
    energy: float
    batches: int


class LatestBatchResponse(BaseModel):
    """GET /dashboard/latest-batch response — for performance gauges."""
    batch_id: str
    quality_score: float
    yield_pct: float
    performance_pct: float
    energy_kwh: float
    progress_pct: float
    elapsed_display: str   # "7:48"
    total_display: str     # "30:00"
