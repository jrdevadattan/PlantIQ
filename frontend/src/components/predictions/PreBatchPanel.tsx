"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  TbSend,
  TbTemperature,
  TbArrowAutofitWidth,
  TbClock,
  TbBox,
  TbCategory,
  TbSunHigh,
  TbCircleCheck,
  TbChartBar,
  TbTrendingUp,
  TbBolt,
  TbGauge,
  TbAlertTriangle,
  TbAdjustments,
  TbArrowUp,
  TbArrowDown,
  TbRefresh,
  TbUser,
  TbLivePhoto,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { useDebounce } from "@/lib/hooks";
import { predictBatch } from "@/lib/api";
import type { BatchPredictionResponse, BatchPredictionParams } from "@/lib/api";
import { CarbonGauge } from "@/components/predictions/CarbonGauge";

interface FormField {
  name: string;
  label: string;
  icon: any;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  unit: string;
}

const fields: FormField[] = [
  { name: "temperature", label: "Temperature", icon: TbTemperature, min: 160, max: 200, step: 1, defaultValue: 183, unit: "C" },
  { name: "conveyorSpeed", label: "Conveyor Speed", icon: TbArrowAutofitWidth, min: 50, max: 100, step: 1, defaultValue: 76, unit: "%" },
  { name: "holdTime", label: "Hold Time", icon: TbClock, min: 5, max: 40, step: 1, defaultValue: 18, unit: "min" },
  { name: "batchSize", label: "Batch Size", icon: TbBox, min: 200, max: 800, step: 10, defaultValue: 500, unit: "kg" },
];

const materialTypes = [
  { value: 0, label: "Type A — Standard" },
  { value: 1, label: "Type B — Dense" },
  { value: 2, label: "Type C — Composite" },
];

const operatorLevels = [
  { value: 0, label: "Junior" },
  { value: 1, label: "Mid-Level" },
  { value: 2, label: "Senior" },
];

/** Default form values — used for reset */
const DEFAULTS = {
  temperature: 183,
  conveyorSpeed: 76,
  holdTime: 18,
  batchSize: 500,
  materialType: 0,
  hourOfDay: Math.max(6, Math.min(21, new Date().getHours())),
  operatorExp: 1,
};

const resultCards = [
  { key: "quality_score", label: "Quality Score", icon: TbChartBar, color: "teal", unit: "%" },
  { key: "yield_pct", label: "Yield", icon: TbTrendingUp, color: "emerald", unit: "%" },
  { key: "performance_pct", label: "Performance", icon: TbGauge, color: "indigo", unit: "%" },
  { key: "energy_kwh", label: "Energy", icon: TbBolt, color: "amber", unit: "kWh" },
];

const colorMap: Record<string, string> = {
  teal: "bg-teal-50 text-teal-600 border-teal-100",
  emerald: "bg-emerald-50 text-emerald-600 border-emerald-100",
  indigo: "bg-indigo-50 text-indigo-600 border-indigo-100",
  amber: "bg-amber-50 text-amber-600 border-amber-100",
};

interface PreBatchPanelProps {
  onPrediction?: (batchId: string, params: BatchPredictionParams) => void;
  onWhatIfToggle?: (enabled: boolean) => void;
}

export function PreBatchPanel({ onPrediction, onWhatIfToggle }: PreBatchPanelProps) {
  const [formData, setFormData] = useState({ ...DEFAULTS });
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [baseline, setBaseline] = useState<BatchPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [whatIfMode, setWhatIfMode] = useState(false);

  /* Debounced form data for What-If auto-predict (400ms) */
  const debouncedForm = useDebounce(formData, 400);
  const hasMounted = useRef(false);

  /* ── What-If auto-predict effect ─────────────────────── */
  useEffect(() => {
    if (!whatIfMode) return;
    // Skip the first render (mounting)
    if (!hasMounted.current) {
      hasMounted.current = true;
      return;
    }
    runPrediction(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedForm, whatIfMode]);

  /* ── Core prediction function ────────────────────────── */
  const runPrediction = useCallback(async (isWhatIf = false) => {
    setLoading(true);
    setError(null);

    const apiParams: BatchPredictionParams = {
      temperature: formData.temperature,
      conveyor_speed: formData.conveyorSpeed,
      hold_time: formData.holdTime,
      batch_size: formData.batchSize,
      material_type: formData.materialType,
      hour_of_day: formData.hourOfDay,
      operator_exp: formData.operatorExp,
    };

    try {
      const response = await predictBatch(apiParams);
      setResult(response);
      /* First prediction becomes the baseline for delta comparison */
      if (!baseline) setBaseline(response);
      onPrediction?.(response.batch_id, apiParams);
    } catch (err: any) {
      console.error("[PreBatchPanel]", err);
      if (!isWhatIf) {
        setError(err.message ?? "Prediction failed. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  }, [formData, baseline, onPrediction]);

  const handleSubmit = async () => {
    const response = await runPredictionAndReturn();
    if (response) setBaseline(response);
  };

  const runPredictionAndReturn = async (): Promise<BatchPredictionResponse | null> => {
    setLoading(true);
    setError(null);

    const apiParams: BatchPredictionParams = {
      temperature: formData.temperature,
      conveyor_speed: formData.conveyorSpeed,
      hold_time: formData.holdTime,
      batch_size: formData.batchSize,
      material_type: formData.materialType,
      hour_of_day: formData.hourOfDay,
      operator_exp: formData.operatorExp,
    };

    try {
      const response = await predictBatch(apiParams);
      setResult(response);
      if (!baseline) setBaseline(response);
      onPrediction?.(response.batch_id, apiParams);
      return response;
    } catch (err: any) {
      console.error("[PreBatchPanel]", err);
      setError(err.message ?? "Prediction failed. Is the backend running?");
      return null;
    } finally {
      setLoading(false);
    }
  };

  const updateField = (name: string, value: number) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const resetDefaults = () => {
    setFormData({ ...DEFAULTS });
  };

  /* ── Delta calculation helper ────────────────────────── */
  const getDelta = (key: string): number | null => {
    if (!whatIfMode || !baseline || !result) return null;
    const curr = result.predictions[key as keyof typeof result.predictions] as number;
    const base = baseline.predictions[key as keyof typeof baseline.predictions] as number;
    if (typeof curr !== "number" || typeof base !== "number") return null;
    const d = curr - base;
    return Math.abs(d) < 0.05 ? null : d;
  };

  return (
    <div className="space-y-5">
      {/* Input Form */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              whatIfMode ? "bg-violet-50" : "bg-teal-50"
            )}>
              {whatIfMode ? (
                <TbAdjustments className="w-[18px] h-[18px] text-violet-600" />
              ) : (
                <TbSend className="w-[18px] h-[18px] text-teal-600" />
              )}
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                {whatIfMode ? "What-If Simulator" : "Batch Parameters"}
                {whatIfMode && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 text-[10px] font-bold uppercase animate-pulse">
                    <TbLivePhoto className="w-3 h-3" />
                    Live
                  </span>
                )}
              </h3>
              <p className="text-[11px] text-slate-400">
                {whatIfMode
                  ? "Drag sliders to see predictions update in real time"
                  : "Enter setup values to predict outcomes before the batch starts"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Reset button */}
            <button
              onClick={resetDefaults}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-all"
              title="Reset to defaults"
            >
              <TbRefresh className="w-4 h-4" />
            </button>

            {/* What-If Mode Toggle */}
            <button
              onClick={() => {
                setWhatIfMode(prev => {
                  const next = !prev;
                  onWhatIfToggle?.(next);
                  return next;
                });
              }}
              className={cn(
                "relative w-11 h-6 rounded-full transition-colors duration-200",
                whatIfMode ? "bg-violet-500" : "bg-slate-200"
              )}
              aria-label="Toggle What-If mode"
            >
              <span
                className={cn(
                  "absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform duration-200",
                  whatIfMode && "translate-x-5"
                )}
              />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {fields.map((field) => {
            const Icon = field.icon;
            const val = formData[field.name as keyof typeof formData] as number;
            return (
              <div key={field.name} className="space-y-2">
                <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
                  <Icon className="w-3.5 h-3.5 text-slate-400" />
                  {field.label}
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={val}
                    onChange={(e) => updateField(field.name, Number(e.target.value))}
                    className="flex-1 h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-teal-500"
                  />
                  <div className="w-20 flex items-center gap-1 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <input
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={val}
                      onChange={(e) => updateField(field.name, Number(e.target.value))}
                      className="w-full text-xs font-bold text-slate-700 bg-transparent outline-none text-center"
                    />
                    <span className="text-[10px] text-slate-400 font-medium">{field.unit}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Material type */}
        <div className="mt-4 space-y-2">
          <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            <TbCategory className="w-3.5 h-3.5 text-slate-400" />
            Material Type
          </label>
          <div className="flex gap-3">
            {materialTypes.map((mt) => (
              <button
                key={mt.value}
                onClick={() => updateField("materialType", mt.value)}
                className={cn(
                  "flex-1 px-4 py-2.5 rounded-lg text-xs font-bold border transition-all",
                  formData.materialType === mt.value
                    ? "bg-teal-50 text-teal-700 border-teal-200"
                    : "bg-white text-slate-500 border-slate-200 hover:bg-slate-50"
                )}
              >
                {mt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Hour of day */}
        <div className="mt-4 space-y-2">
          <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            <TbSunHigh className="w-3.5 h-3.5 text-slate-400" />
            Shift Hour
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={6}
              max={21}
              value={formData.hourOfDay}
              onChange={(e) => updateField("hourOfDay", Number(e.target.value))}
              className="flex-1 h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-teal-500"
            />
            <div className="w-20 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">
              <span className="text-xs font-bold text-slate-700 block text-center">
                {formData.hourOfDay}:00
              </span>
            </div>
          </div>
        </div>

        {/* Operator Experience */}
        <div className="mt-4 space-y-2">
          <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            <TbUser className="w-3.5 h-3.5 text-slate-400" />
            Operator Experience
          </label>
          <div className="flex gap-3">
            {operatorLevels.map((ol) => (
              <button
                key={ol.value}
                onClick={() => updateField("operatorExp", ol.value)}
                className={cn(
                  "flex-1 px-4 py-2.5 rounded-lg text-xs font-bold border transition-all",
                  formData.operatorExp === ol.value
                    ? "bg-teal-50 text-teal-700 border-teal-200"
                    : "bg-white text-slate-500 border-slate-200 hover:bg-slate-50"
                )}
              >
                {ol.label}
              </button>
            ))}
          </div>
        </div>

        {/* Error state */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-100 rounded-lg flex items-start gap-2">
            <TbAlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-red-700 leading-relaxed">{error}</p>
          </div>
        )}

        {/* Submit — visible when What-If mode is OFF */}
        {!whatIfMode && (
          <button
            onClick={handleSubmit}
            disabled={loading}
            className={cn(
              "mt-5 w-full py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
              loading
                ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                : "bg-teal-600 text-white hover:bg-teal-700 active:scale-[0.99] shadow-sm"
            )}
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-300 border-t-slate-500 rounded-full animate-spin" />
                Running Prediction...
              </>
            ) : (
              <>
                <TbSend className="w-4 h-4" />
                Predict Batch Outcomes
              </>
            )}
          </button>
        )}

        {/* What-If mode active indicator */}
        {whatIfMode && (
          <div className="mt-5 flex items-center justify-center gap-2 py-2 text-[11px] text-violet-500 font-medium">
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-violet-200 border-t-violet-500 rounded-full animate-spin" />
                Updating predictions...
              </>
            ) : result ? (
              <>
                <TbCircleCheck className="w-4 h-4 text-violet-500" />
                Predictions update live as you adjust sliders
              </>
            ) : (
              <>
                <TbAdjustments className="w-4 h-4" />
                Adjust any slider to see live predictions
              </>
            )}
          </div>
        )}
      </div>

      {/* Prediction Results */}
      {result && (
        <div className={cn(
          "bg-white rounded-xl border p-5 animate-count",
          whatIfMode ? "border-violet-200" : "border-slate-100"
        )}>
          <div className="flex items-center gap-3 mb-5">
            <div className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              whatIfMode ? "bg-violet-50" : "bg-emerald-50"
            )}>
              {whatIfMode ? (
                <TbAdjustments className="w-[18px] h-[18px] text-violet-600" />
              ) : (
                <TbCircleCheck className="w-[18px] h-[18px] text-emerald-600" />
              )}
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">
                {whatIfMode ? "What-If Results" : "Prediction Results"}
              </h3>
              <p className="text-[11px] text-slate-400">
                {whatIfMode
                  ? "Comparing against baseline — deltas shown"
                  : `Batch ${result.batch_id} — Multi-target prediction with confidence intervals`}
              </p>
            </div>
            {/* Set baseline button in What-If mode */}
            {whatIfMode && baseline && (
              <button
                onClick={() => setBaseline(result)}
                className="ml-auto px-3 py-1.5 rounded-lg text-[10px] font-bold text-violet-600 bg-violet-50 border border-violet-100 hover:bg-violet-100 transition-all"
              >
                Set as Baseline
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {resultCards.map((card) => {
              const Icon = card.icon;
              const val = result.predictions[card.key as keyof typeof result.predictions] as number;
              const ci = result.confidence_intervals[card.key];
              const delta = getDelta(card.key);
              return (
                <div
                  key={card.key}
                  className={cn("p-4 rounded-lg border relative", colorMap[card.color])}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className="w-4 h-4" />
                    <span className="text-[11px] font-bold uppercase tracking-wide">
                      {card.label}
                    </span>
                  </div>
                  <p className={cn(
                    "text-2xl font-bold text-slate-800 transition-opacity",
                    loading && whatIfMode && "opacity-50"
                  )}>
                    {typeof val === "number" ? val.toFixed(1) : val}
                    <span className="text-xs font-semibold text-slate-400 ml-1">
                      {card.unit}
                    </span>
                  </p>
                  {/* Delta indicator — What-If mode only */}
                  {delta !== null && (
                    <div className={cn(
                      "flex items-center gap-1 mt-1.5",
                      delta > 0
                        ? (card.key === "energy_kwh" ? "text-amber-600" : "text-emerald-600")
                        : (card.key === "energy_kwh" ? "text-emerald-600" : "text-amber-600")
                    )}>
                      {delta > 0 ? (
                        <TbArrowUp className="w-3 h-3" />
                      ) : (
                        <TbArrowDown className="w-3 h-3" />
                      )}
                      <span className="text-[10px] font-bold">
                        {delta > 0 ? "+" : ""}{delta.toFixed(1)} {card.unit}
                      </span>
                      <span className="text-[10px] text-slate-400 ml-0.5">vs baseline</span>
                    </div>
                  )}
                  {ci && !delta && (
                    <p className="text-[10px] text-slate-400 mt-1">
                      CI: {ci.lower.toFixed(1)} — {ci.upper.toFixed(1)}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Carbon Budget Gauge */}
          <div className="mt-4">
            <CarbonGauge
              predictedKg={result.carbon_budget.predicted_usage_kg}
              budgetKg={result.carbon_budget.batch_budget_kg}
              status={result.carbon_budget.status}
            />
          </div>
        </div>
      )}
    </div>
  );
}
