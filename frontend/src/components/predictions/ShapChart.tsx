"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { TbBulb, TbArrowBadgeRight, TbLoader2 } from "react-icons/tb";
import { cn } from "@/lib/utils";
import { explainBatch } from "@/lib/api";
import type { BatchPredictionParams, ExplainResponse, FeatureContribution } from "@/lib/api";

/* ── Props ─────────────────────────────────────────────────── */

interface ShapChartProps {
  batchId?: string | null;
  params?: BatchPredictionParams | null;
  target?: string;
  /** When true, uses a lighter loading indicator for rapid updates */
  liveMode?: boolean;
}

/* ── Custom Tooltip ────────────────────────────────────────── */

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as FeatureContribution;
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 max-w-xs">
        <p className="text-xs font-bold text-slate-700 mb-1">{data.feature}</p>
        <p className="text-[11px] text-slate-500 mb-1">
          {data.contribution > 0 ? "Increases" : "Decreases"} energy by{" "}
          <span className="font-bold text-slate-800">
            {Math.abs(data.contribution).toFixed(2)} kWh
          </span>
        </p>
        {data.plain_english && (
          <p className="text-[10px] text-slate-400 italic">{data.plain_english}</p>
        )}
      </div>
    );
  }
  return null;
};

/* ── Component ─────────────────────────────────────────────── */

export function ShapChart({ batchId, params, target = "energy_kwh", liveMode = false }: ShapChartProps) {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* Fetch SHAP explanation whenever batchId / params change */
  const fetchExplanation = useCallback(async () => {
    if (!batchId || !params) return;
    setLoading(true);
    setError(null);

    try {
      const response = await explainBatch(batchId, params, target);
      setData(response);
    } catch (err: any) {
      console.error("[ShapChart]", err);
      setError(err.message ?? "Failed to fetch SHAP explanation");
    } finally {
      setLoading(false);
    }
  }, [batchId, params, target]);

  useEffect(() => {
    fetchExplanation();
  }, [fetchExplanation]);

  /* ── Empty state: no prediction yet ────────────────────── */
  if (!batchId || !params) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
            <TbBulb className="w-[18px] h-[18px] text-violet-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Why This Prediction?</h3>
            <p className="text-[11px] text-slate-400">
              Feature contributions to energy prediction (SHAP values)
            </p>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center h-56 text-center">
          <p className="text-sm text-slate-400">Run a prediction to see SHAP explanations</p>
          <p className="text-[11px] text-slate-300 mt-1">
            Feature contributions will appear here
          </p>
        </div>
      </div>
    );
  }

  /* ── Loading state ─────────────────────────────────────── */
  /* In liveMode, if we already have data, skip the full spinner
     and show a subtle overlay on the existing chart instead */
  if (loading && !(liveMode && data)) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
            <TbBulb className="w-[18px] h-[18px] text-violet-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Why This Prediction?</h3>
            <p className="text-[11px] text-slate-400">Computing SHAP values…</p>
          </div>
        </div>
        <div className="flex flex-col items-center justify-center h-56">
          <TbLoader2 className="w-8 h-8 text-violet-400 animate-spin" />
          <p className="text-xs text-slate-400 mt-3">Analyzing feature contributions…</p>
        </div>
      </div>
    );
  }

  /* ── Error state ───────────────────────────────────────── */
  if (error) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
            <TbBulb className="w-[18px] h-[18px] text-violet-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Why This Prediction?</h3>
            <p className="text-[11px] text-slate-400">Feature contributions to energy prediction</p>
          </div>
        </div>
        <div className="p-3 bg-red-50 border border-red-100 rounded-lg">
          <p className="text-xs text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  /* ── Success state with data ───────────────────────────── */
  if (!data || data.feature_contributions.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
            <TbBulb className="w-[18px] h-[18px] text-violet-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Why This Prediction?</h3>
            <p className="text-[11px] text-slate-400">No feature contributions available</p>
          </div>
        </div>
      </div>
    );
  }

  const sortedData = [...data.feature_contributions].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  );

  const topDriver = sortedData[0];

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-5 relative">
      {/* Live-mode loading overlay — subtle shimmer on existing chart */}
      {liveMode && loading && (
        <div className="absolute inset-0 bg-white/50 backdrop-blur-[1px] rounded-xl z-10 flex items-center justify-center">
          <div className="flex items-center gap-2 text-[11px] text-violet-500 font-medium">
            <div className="w-3.5 h-3.5 border-2 border-violet-200 border-t-violet-500 rounded-full animate-spin" />
            Updating SHAP...
          </div>
        </div>
      )}
      <div className="flex items-center gap-3 mb-1">
        <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
          <TbBulb className="w-[18px] h-[18px] text-violet-600" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-slate-800">
            Why This Prediction?
          </h3>
          <p className="text-[11px] text-slate-400">
            Feature contributions to {target.replace("_", " ")} (SHAP values)
          </p>
        </div>
      </div>

      {/* AI Summary from backend */}
      {data.summary && (
        <div className="mt-4 mb-2 p-3 bg-violet-50 border border-violet-100 rounded-lg">
          <p className="text-[11px] text-violet-800 leading-relaxed">{data.summary}</p>
        </div>
      )}

      {/* Top driver callout */}
      <div className={cn("mb-4 p-3 rounded-lg flex items-start gap-2", data.summary ? "mt-2" : "mt-4", "bg-amber-50 border border-amber-100")}>
        <TbArrowBadgeRight className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
        <p className="text-[11px] text-amber-800 leading-relaxed">
          <span className="font-bold">{topDriver.feature}</span> is the biggest
          driver, contributing{" "}
          <span className="font-bold">
            {topDriver.contribution > 0 ? "+" : ""}{topDriver.contribution.toFixed(2)} kWh
          </span>{" "}
          {topDriver.contribution > 0 ? "above" : "below"} baseline.
          {topDriver.plain_english && (
            <> {topDriver.plain_english}</>
          )}
        </p>
      </div>

      {/* Baseline → Final summary */}
      <div className="flex items-center justify-between mb-3 px-1">
        <span className="text-[10px] text-slate-400">
          Baseline: <span className="font-bold text-slate-600">{data.baseline_prediction.toFixed(1)} kWh</span>
        </span>
        <span className="text-[10px] text-slate-400">
          Final: <span className="font-bold text-slate-700">{data.final_prediction.toFixed(1)} kWh</span>
        </span>
      </div>

      {/* Chart */}
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            layout="vertical"
            margin={{ left: 10, right: 20 }}
          >
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickFormatter={(v) => `${v > 0 ? "+" : ""}${v}`}
            />
            <YAxis
              type="category"
              dataKey="feature"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fontWeight: 600, fill: "#64748b" }}
              width={110}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(0,0,0,0.02)" }} />
            <ReferenceLine x={0} stroke="#e2e8f0" />
            <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={20}>
              {sortedData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.contribution > 0 ? "#f59e0b" : "#14b8a6"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-amber-500" />
          <span className="text-[10px] font-semibold text-slate-400">
            Increases Energy
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-teal-500" />
          <span className="text-[10px] font-semibold text-slate-400">
            Decreases Energy
          </span>
        </div>
      </div>
    </div>
  );
}
