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
  const safeLimit = Math.min(Math.max(limit, 1), 50);
  return apiFetch<DashboardBatch[]>(`/dashboard/recent-batches?limit=${safeLimit}`);
}

export async function fetchShiftPerformance(): Promise<ShiftPerformanceData[]> {
  return apiFetch<ShiftPerformanceData[]>("/dashboard/shift-performance");
}

export async function fetchLatestBatch(): Promise<LatestBatch> {
  return apiFetch<LatestBatch>("/dashboard/latest-batch");
}

// ── Alerts ──────────────────────────────────────────────────

export interface AlertRecord {
  id: string;
  batch_id: string;
  severity: string;
  alert_type: string;
  message: string;
  root_cause: string;
  recommended_action: string;
  estimated_saving_kwh: number;
  timestamp: string;
  acknowledged: boolean;
}

export async function fetchAlerts(limit = 50): Promise<AlertRecord[]> {
  const safeLimit = Math.min(Math.max(limit, 1), 200);
  const raw = await apiFetch<
    AlertRecord[] | { count: number; alerts: Array<Record<string, unknown>> }
  >(`/alerts?limit=${safeLimit}`);

  if (Array.isArray(raw)) {
    return raw;
  }

  const alerts = Array.isArray(raw?.alerts) ? raw.alerts : [];
  return alerts.map((a) => {
    const state = String(a.state ?? "fired");
    return {
      id: String(a.alert_id ?? a.id ?? ""),
      batch_id: String(a.batch_id ?? ""),
      severity: String(a.severity ?? "WATCH"),
      alert_type: String(a.alert_type ?? "anomaly"),
      message: String(a.message ?? ""),
      root_cause: String(a.technical_detail ?? ""),
      recommended_action: String(a.recommended_action ?? ""),
      estimated_saving_kwh: Number(a.estimated_saving_kwh ?? 0),
      timestamp: String(a.fired_at ?? a.timestamp ?? new Date().toISOString()),
      acknowledged: ["acknowledged", "acted_upon", "resolved"].includes(state),
    };
  });
}

// ── Recommendations ─────────────────────────────────────────

export interface RecommendationItem {
  rank: number;
  parameter: string;
  current_value: number;
  recommended_value: number;
  adjustment: number;
  direction: string;
  machine: string;
  control: string;
  instruction: string;
  estimated_energy_saving_kwh: number;
  estimated_quality_impact_pct: number;
  estimated_yield_impact_pct: number;
  response_time_min: number;
  timing_note: string;
  shap_contribution: number;
  shap_direction: string;
  within_safe_range: boolean;
  safety_note: string;
}

export interface RecommendationResponse {
  batch_id: string;
  target: string;
  recommendations: RecommendationItem[];
  summary: string;
  total_estimated_saving_kwh: number;
}

export async function generateRecommendations(
  batchId: string,
  inputParams: BatchPredictionParams,
  shapContributions: Array<{ feature: string; contribution: number; direction: string }>,
  target: string = "energy_kwh",
): Promise<RecommendationResponse> {
  return apiFetch<RecommendationResponse>("/recommend/generate", {
    method: "POST",
    body: JSON.stringify({
      batch_id: batchId,
      input_params: inputParams,
      shap_contributions: shapContributions,
      target,
    }),
  });
}

// ── Cost Translation ────────────────────────────────────────

export interface CostTranslation {
  predicted_cost_inr: number;
  target_cost_inr: number;
  cost_variance_inr: number;
  monthly_projection_inr: number;
  co2_kg: number;
  co2_status: string;
  summary: string;
}

// ── Management Dashboard ────────────────────────────────────

export interface CostConfig {
  tariff_inr_per_kwh: number;
  co2_factor_kg_per_kwh: number;
  energy_target_kwh: number;
  co2_budget_kg: number;
  batches_per_day: number;
  operating_days_per_month: number;
  optimization_headroom_pct: number;
}

export interface ComplianceReport {
  total_batches: number;
  energy_stats: {
    mean_kwh: number;
    median_kwh: number;
    total_kwh: number;
    trend: string;
  };
  carbon_stats: {
    total_kg: number;
    mean_kg_per_batch: number;
  };
  compliance: {
    on_track_pct: number;
    caution_pct: number;
    exceeded_pct: number;
  };
  cumulative_savings: {
    energy_kwh: number;
    co2_kg: number;
  };
}

export async function fetchCostConfig(): Promise<CostConfig> {
  return apiFetch<CostConfig>("/cost/config");
}

export async function fetchComplianceReport(): Promise<ComplianceReport> {
  return apiFetch<ComplianceReport>("/targets/report");
}

// ── Pharma Core Mode (README-aligned) ─────────────────────

export interface PharmaCorePrediction {
  hardness: number;
  friability: number;
  dissolution_rate: number;
  content_uniformity: number;
  energy_kwh: number;
}

export interface PharmaCorePredictionResponse {
  status: string;
  core_predictions: PharmaCorePrediction;
}

export interface PharmaCoreParams {
  granulation_time: number;
  binder_amount: number;
  drying_temp: number;
  drying_time: number;
  compression_force: number;
  machine_speed: number;
  lubricant_conc: number;
}

export async function predictPharmaCore(
  params: PharmaCoreParams,
): Promise<PharmaCorePredictionResponse> {
  return apiFetch<PharmaCorePredictionResponse>("/hackathon/predict-core", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// ── Live WebSocket Channel ────────────────────────────────

export function connectLiveBatch(batchId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  return new WebSocket(`${protocol}://${host}/live/${encodeURIComponent(batchId)}`);
}

