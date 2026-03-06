/**
 * PlantIQ — API Service Layer
 * ==============================
 * Central fetch functions for all backend endpoints.
 * Uses Next.js rewrites (/api/* → http://localhost:8000/*) to avoid CORS.
 * Falls back to mock data when the backend is unreachable.
 */

// ── Base URL ────────────────────────────────────────────────
const API_BASE = "/api";

// ── Types ───────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  version: string;
}

export interface PredictionValues {
  quality_score: number;
  yield_pct: number;
  performance_pct: number;
  energy_kwh: number;
  co2_kg: number;
}

export interface ConfidenceInterval {
  lower: number;
  upper: number;
}

export interface CarbonBudget {
  batch_budget_kg: number;
  predicted_usage_kg: number;
  status: string; // ON_TRACK, WARNING, OVER_BUDGET
  headroom_kg: number;
}

export interface BatchPredictionResponse {
  batch_id: string;
  predictions: PredictionValues;
  confidence_intervals: Record<string, ConfidenceInterval>;
  carbon_budget: CarbonBudget;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  plain_english: string;
}

export interface ExplainResponse {
  batch_id: string;
  target: string;
  baseline_prediction: number;
  final_prediction: number;
  feature_contributions: FeatureContribution[];
  summary: string;
}

export interface DiagnosisInfo {
  fault_type: string;
  confidence: number;
  human_readable: string;
  recommended_action: string;
  estimated_energy_impact_kwh?: number;
  estimated_quality_impact_pct?: number;
}

export interface AnomalyDetectResponse {
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  severity: string;
  diagnosis: DiagnosisInfo;
}

export interface ModelFeaturesResponse {
  energy: Record<string, number>;
  quality: Record<string, number>;
  yield: Record<string, number>;
  performance: Record<string, number>;
}

export interface RealtimePredictionResponse {
  progress_pct: number;
  updated_predictions: PredictionValues;
  confidence: string;
  alert?: {
    severity: string;
    message: string;
    recommended_action?: string;
    estimated_saving_kwh?: number;
    quality_impact_pct?: number;
  };
}

// ── Generic Fetch Wrapper ───────────────────────────────────

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
  timeoutMs = 8000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "Unknown error");
      throw new Error(`API ${res.status}: ${detail}`);
    }

    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

// ── Health ──────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

// ── Predictions ─────────────────────────────────────────────

export interface BatchPredictionParams {
  temperature: number;
  conveyor_speed: number;
  hold_time: number;
  batch_size: number;
  material_type: number;
  hour_of_day: number;
  operator_exp: number;
}

export async function predictBatch(
  params: BatchPredictionParams,
): Promise<BatchPredictionResponse> {
  return apiFetch<BatchPredictionResponse>("/predict/batch", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function predictRealtime(
  originalParams: BatchPredictionParams,
  partialData: {
    elapsed_minutes: number;
    energy_so_far: number;
    avg_power_kw: number;
    anomaly_events: number;
  },
): Promise<RealtimePredictionResponse> {
  return apiFetch<RealtimePredictionResponse>("/predict/realtime", {
    method: "POST",
    body: JSON.stringify({
      original_params: originalParams,
      partial_data: partialData,
    }),
  });
}

// ── SHAP Explanation ────────────────────────────────────────

export async function explainBatch(
  batchId: string,
  params: BatchPredictionParams,
  target: string = "energy_kwh",
): Promise<ExplainResponse> {
  return apiFetch<ExplainResponse>(
    `/explain/${batchId}?target=${encodeURIComponent(target)}`,
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  );
}

// ── Anomaly Detection ───────────────────────────────────────

export async function detectAnomaly(
  batchId: string,
  powerReadings: number[],
  elapsedSeconds: number,
): Promise<AnomalyDetectResponse> {
  return apiFetch<AnomalyDetectResponse>("/anomaly/detect", {
    method: "POST",
    body: JSON.stringify({
      batch_id: batchId,
      power_readings: powerReadings,
      elapsed_seconds: elapsedSeconds,
    }),
  });
}

// ── Model Features ──────────────────────────────────────────

export async function fetchModelFeatures(): Promise<ModelFeaturesResponse> {
  return apiFetch<ModelFeaturesResponse>("/model/features");
}

// ── Dashboard ───────────────────────────────────────────────

export interface DashboardSummary {
  total_batches: number;
  running_count: number;
  avg_energy: number;
  avg_quality: number;
  avg_yield: number;
  avg_performance: number;
  anomaly_count: number;
  resolved_count: number;
  model_accuracy: number;
  mape_pct: number;
  energy_trend: string;
  energy_trend_value: string;
  quality_trend: string;
  quality_trend_value: string;
  yield_trend: string;
  yield_trend_value: string;
}

export interface DailyEnergyItem {
  day: string;
  kwh: number;
  date: string;
  batch_count: number;
}

export interface DashboardBatch {
  id: string;
  timestamp: string;
  temperature: number;
  conveyorSpeed: number;
  holdTime: number;
  batchSize: number;
  materialType: number;
  hourOfDay: number;
  qualityScore: number;
  yieldPct: number;
  performancePct: number;
  energyKwh: number;
  status: "completed" | "running" | "scheduled" | "alert";
  anomalyScore: number;
}

export interface ShiftPerformanceData {
  shift: string;
  quality: number;
  yield_pct: number;
  energy: number;
  batches: number;
}

export interface LatestBatch {
  batch_id: string;
  quality_score: number;
  yield_pct: number;
  performance_pct: number;
  energy_kwh: number;
  progress_pct: number;
  elapsed_display: string;
  total_display: string;
}

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  return apiFetch<DashboardSummary>("/dashboard/summary");
}

export async function fetchEnergyDaily(days = 7): Promise<DailyEnergyItem[]> {
  return apiFetch<DailyEnergyItem[]>(`/dashboard/energy-daily?days=${days}`);
}

export async function fetchRecentBatches(limit = 6): Promise<DashboardBatch[]> {
  return apiFetch<DashboardBatch[]>(`/dashboard/recent-batches?limit=${limit}`);
}

export async function fetchShiftPerformance(): Promise<ShiftPerformanceData[]> {
  return apiFetch<ShiftPerformanceData[]>("/dashboard/shift-performance");
}

export async function fetchLatestBatch(): Promise<LatestBatch> {
  return apiFetch<LatestBatch>("/dashboard/latest-batch");
}
