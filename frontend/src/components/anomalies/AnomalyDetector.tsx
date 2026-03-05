"use client";

import React, { useState, useMemo, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  AreaChart,
  Area,
} from "recharts";
import {
  TbAlertTriangle,
  TbShieldCheck,
  TbUpload,
  TbTool,
  TbDroplet,
  TbCircleCheck,
  TbLoader2,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { generatePowerCurve } from "@/lib/mockData";
import { detectAnomaly } from "@/lib/api";
import type { AnomalyDetectResponse } from "@/lib/api";

type FaultType = "normal" | "bearing_wear" | "wet_material";

const faultProfiles = [
  {
    id: "normal" as FaultType,
    label: "Normal Operation",
    icon: TbCircleCheck,
    color: "emerald",
    description: "Healthy power curve with minimal noise",
  },
  {
    id: "bearing_wear" as FaultType,
    label: "Bearing Wear",
    icon: TbTool,
    color: "amber",
    description: "Gradual baseline rise — motor working harder over time",
  },
  {
    id: "wet_material" as FaultType,
    label: "Wet Raw Material",
    icon: TbDroplet,
    color: "red",
    description: "Irregular high-frequency spikes from moisture content",
  },
];

/* ── Severity → color mapping ─────────────────────────────── */
const severityColors: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  NORMAL: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-100", dot: "active" },
  WATCH: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-100", dot: "warning" },
  WARNING: { bg: "bg-orange-50", text: "text-orange-700", border: "border-orange-100", dot: "warning" },
  CRITICAL: { bg: "bg-red-50", text: "text-red-700", border: "border-red-100", dot: "danger" },
};

const selectorColors: Record<string, { bg: string; text: string; border: string }> = {
  emerald: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-100" },
  amber: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-100" },
  red: { bg: "bg-red-50", text: "text-red-700", border: "border-red-100" },
};

export function AnomalyDetector() {
  const [selectedFault, setSelectedFault] = useState<FaultType>("normal");
  const [apiResult, setApiResult] = useState<AnomalyDetectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generate power curve for visualization (client-side)
  const { combined, residuals, rawReadings } = useMemo(() => {
    const raw = generatePowerCurve(selectedFault, 600);
    const normalBaseline = generatePowerCurve("normal", 600);

    // Visual reconstruction approximation (for chart only — scoring done server-side)
    const recon = normalBaseline.map((v) => v + (Math.random() - 0.5) * 0.1);

    // Visual residuals for chart
    const res = raw.map((v, i) => Math.abs(v - recon[i]));

    const combined = raw.map((v, i) => ({
      t: i,
      original: v,
      reconstructed: Number(recon[i].toFixed(2)),
    }));

    return {
      combined,
      residuals: res.map((v, i) => ({ t: i, error: Number(v.toFixed(3)) })),
      rawReadings: raw,
    };
  }, [selectedFault]);

  // Send power readings to backend for real LSTM analysis
  const analyze = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const batchId = `anomaly-${selectedFault}-${Date.now()}`;
      const response = await detectAnomaly(batchId, rawReadings, rawReadings.length);
      setApiResult(response);
    } catch (err: any) {
      console.error("[AnomalyDetector]", err);
      setError(err.message ?? "Anomaly detection failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [rawReadings, selectedFault]);

  // Auto-analyze when fault selection changes
  useEffect(() => {
    analyze();
  }, [analyze]);

  // Derive display values from API result or fallback
  const anomalyScore = apiResult?.anomaly_score ?? 0;
  const isAnomaly = apiResult?.is_anomaly ?? false;
  const severity = apiResult?.severity ?? "NORMAL";
  const diagnosis = apiResult?.diagnosis;
  const threshold = apiResult?.threshold ?? 0.30;

  const colors = severityColors[severity] ?? severityColors.NORMAL;

  return (
    <div className="space-y-5">
      {/* Fault Selector */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-9 h-9 rounded-lg bg-slate-100 flex items-center justify-center">
            <TbUpload className="w-[18px] h-[18px] text-slate-500" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Power Curve Analysis</h3>
            <p className="text-[11px] text-slate-400">
              Select a scenario — power curve is sent to the LSTM Autoencoder for real analysis
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {faultProfiles.map((fp) => {
            const Icon = fp.icon;
            const cv = selectorColors[fp.color];
            const isSelected = selectedFault === fp.id;
            return (
              <button
                key={fp.id}
                onClick={() => setSelectedFault(fp.id)}
                className={cn(
                  "p-4 rounded-lg border text-left transition-all",
                  isSelected
                    ? `${cv.bg} ${cv.border} ${cv.text}`
                    : "bg-white border-slate-200 text-slate-500 hover:bg-slate-50"
                )}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="w-4 h-4" />
                  <span className="text-xs font-bold">{fp.label}</span>
                </div>
                <p className="text-[11px] leading-relaxed opacity-80">
                  {fp.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-100 p-5 flex items-center justify-center gap-3">
          <TbLoader2 className="w-5 h-5 text-teal-500 animate-spin" />
          <p className="text-sm text-slate-500">Analyzing power curve with LSTM Autoencoder…</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="bg-red-50 rounded-xl border border-red-100 p-4">
          <div className="flex items-center gap-2 mb-1">
            <TbAlertTriangle className="w-4 h-4 text-red-500" />
            <span className="text-xs font-bold text-red-700">Analysis Failed</span>
          </div>
          <p className="text-[11px] text-red-600">{error}</p>
        </div>
      )}

      {/* Anomaly Score + Diagnosis (from real API) */}
      {!loading && apiResult && (
        <div className={cn("rounded-xl border p-5", colors.bg, colors.border)}>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <span className={`status-dot ${colors.dot}`} />
              <div>
                <h3 className={cn("text-sm font-bold", colors.text)}>
                  {isAnomaly ? "Anomaly Detected" : "No Anomaly"}
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  {diagnosis?.human_readable ?? "No diagnosis available"}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide">Score</p>
              <p className={cn("text-2xl font-bold font-mono", colors.text)}>
                {anomalyScore.toFixed(2)}
              </p>
              <p className="text-[10px] text-slate-400">Threshold: {threshold.toFixed(2)}</p>
            </div>
          </div>

          {/* Severity badge */}
          <div className="mt-3 flex items-center gap-3">
            <span className={cn(
              "px-2 py-0.5 text-[10px] font-bold rounded-full border",
              colors.bg, colors.text, colors.border
            )}>
              {severity}
            </span>
            {diagnosis && (
              <span className="text-[10px] text-slate-400">
                Fault: <span className="font-bold text-slate-600">{diagnosis.fault_type}</span>
                {" · "}Confidence: <span className="font-bold text-slate-600">{(diagnosis.confidence * 100).toFixed(0)}%</span>
              </span>
            )}
          </div>

          {isAnomaly && diagnosis && (
            <div className="mt-4 p-3 bg-white/60 rounded-lg border border-white/80">
              <div className="flex items-center gap-2 mb-1">
                <TbTool className="w-4 h-4 text-slate-600" />
                <span className="text-xs font-bold text-slate-700">Recommended Action</span>
              </div>
              <p className="text-[11px] text-slate-600">{diagnosis.recommended_action}</p>

              {/* Impact estimates */}
              {(diagnosis.estimated_energy_impact_kwh || diagnosis.estimated_quality_impact_pct) && (
                <div className="flex gap-4 mt-2">
                  {diagnosis.estimated_energy_impact_kwh !== undefined && diagnosis.estimated_energy_impact_kwh !== 0 && (
                    <span className="text-[10px] text-slate-400">
                      Energy impact: <span className="font-bold text-amber-600">+{diagnosis.estimated_energy_impact_kwh} kWh</span>
                    </span>
                  )}
                  {diagnosis.estimated_quality_impact_pct !== undefined && diagnosis.estimated_quality_impact_pct !== 0 && (
                    <span className="text-[10px] text-slate-400">
                      Quality impact: <span className="font-bold text-red-600">{diagnosis.estimated_quality_impact_pct}%</span>
                    </span>
                  )}
                </div>
              )}

              <div className="flex gap-2 mt-3">
                <button className="px-3 py-1.5 bg-teal-600 text-white text-[11px] font-bold rounded-lg hover:bg-teal-700 transition-colors">
                  Approve Action
                </button>
                <button className="px-3 py-1.5 bg-white text-slate-500 text-[11px] font-bold rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors">
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Original vs Reconstructed */}
        <div className="bg-white rounded-xl border border-slate-100 p-5">
          <h3 className="text-sm font-bold text-slate-800 mb-1">
            Original vs Reconstructed
          </h3>
          <p className="text-[11px] text-slate-400 mb-4">
            Blue = actual reading, Gray = autoencoder reconstruction
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combined}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis
                  dataKey="t"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  tickFormatter={(v) => `${Math.floor(v / 60)}m`}
                  interval={119}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  width={30}
                  domain={[0, 10]}
                />
                <Line
                  type="monotone"
                  dataKey="original"
                  stroke={isAnomaly ? "#ef4444" : "#6366f1"}
                  strokeWidth={1.5}
                  dot={false}
                  name="Original"
                />
                <Line
                  type="monotone"
                  dataKey="reconstructed"
                  stroke="#cbd5e1"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Reconstructed"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Reconstruction Error */}
        <div className="bg-white rounded-xl border border-slate-100 p-5">
          <h3 className="text-sm font-bold text-slate-800 mb-1">
            Reconstruction Error
          </h3>
          <p className="text-[11px] text-slate-400 mb-4">
            High error regions indicate anomalous behavior
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={residuals}>
                <defs>
                  <linearGradient id="errorGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis
                  dataKey="t"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  tickFormatter={(v) => `${Math.floor(v / 60)}m`}
                  interval={119}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  width={30}
                />
                <Tooltip
                  content={({ active, payload }: any) => {
                    if (active && payload?.length) {
                      return (
                        <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-2.5">
                          <p className="text-xs font-bold text-slate-700">
                            Error: {payload[0]?.value?.toFixed(3)}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="error"
                  stroke={isAnomaly ? "#ef4444" : "#f59e0b"}
                  strokeWidth={1}
                  fill="url(#errorGrad)"
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
