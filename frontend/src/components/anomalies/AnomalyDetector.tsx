"use client";

import React, { useState, useMemo } from "react";
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
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { generatePowerCurve } from "@/lib/mockData";

type FaultType = "normal" | "bearing_wear" | "wet_material";

const faultProfiles = [
  {
    id: "normal" as FaultType,
    label: "Normal Operation",
    icon: TbCircleCheck,
    color: "emerald",
    description: "Healthy power curve with minimal noise",
    action: "No action needed",
  },
  {
    id: "bearing_wear" as FaultType,
    label: "Bearing Wear",
    icon: TbTool,
    color: "amber",
    description: "Gradual baseline rise — motor working harder over time",
    action: "Schedule maintenance in 5 days",
  },
  {
    id: "wet_material" as FaultType,
    label: "Wet Raw Material",
    icon: TbDroplet,
    color: "red",
    description: "Irregular high-frequency spikes from moisture content",
    action: "Extend drying phase by 3-4 minutes",
  },
];

const colorVariants: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  emerald: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-100", dot: "active" },
  amber: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-100", dot: "warning" },
  red: { bg: "bg-red-50", text: "text-red-700", border: "border-red-100", dot: "danger" },
};

export function AnomalyDetector() {
  const [selectedFault, setSelectedFault] = useState<FaultType>("normal");

  // Generate power curve and reconstruction
  const { combined, residuals, anomalyScore, diagnosis } = useMemo(() => {
    const raw = generatePowerCurve(selectedFault, 600);
    const normalBaseline = generatePowerCurve("normal", 600);

    // Simulate autoencoder: reconstruction ≈ normal baseline (what autoencoder learned)
    const recon = normalBaseline.map((v, i) => v + (Math.random() - 0.5) * 0.1);

    // Residuals = |original - reconstructed|
    const res = raw.map((v, i) => Math.abs(v - recon[i]));

    // Mean anomaly score
    const meanResidual = res.reduce((a, b) => a + b, 0) / res.length;
    const score = Number(Math.min(meanResidual / 2, 1).toFixed(2));

    const diag = selectedFault === "normal"
      ? "No anomaly detected"
      : selectedFault === "bearing_wear"
      ? "Gradual power drift detected — consistent with mechanical wear"
      : "Irregular power spikes detected — consistent with high raw material moisture";

    // Combine original + reconstructed into one array for LineChart
    const combined = raw.map((v, i) => ({
      t: i,
      original: v,
      reconstructed: Number(recon[i].toFixed(2)),
    }));

    return {
      combined,
      residuals: res.map((v, i) => ({ t: i, error: Number(v.toFixed(3)) })),
      anomalyScore: score,
      diagnosis: diag,
    };
  }, [selectedFault]);

  const faultProfile = faultProfiles.find(f => f.id === selectedFault)!;
  const colors = colorVariants[faultProfile.color];
  const isAnomaly = anomalyScore > 0.3;

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
              Select a scenario to see how the LSTM Autoencoder detects anomalies
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {faultProfiles.map((fp) => {
            const Icon = fp.icon;
            const cv = colorVariants[fp.color];
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

      {/* Anomaly Score + Diagnosis */}
      <div className={cn("rounded-xl border p-5", colors.bg, colors.border)}>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <span className={`status-dot ${colors.dot}`} />
            <div>
              <h3 className={cn("text-sm font-bold", colors.text)}>
                {isAnomaly ? "Anomaly Detected" : "No Anomaly"}
              </h3>
              <p className="text-[11px] text-slate-500 mt-0.5">{diagnosis}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide">Score</p>
            <p className={cn("text-2xl font-bold font-mono", colors.text)}>
              {anomalyScore}
            </p>
            <p className="text-[10px] text-slate-400">Threshold: 0.30</p>
          </div>
        </div>

        {isAnomaly && (
          <div className="mt-4 p-3 bg-white/60 rounded-lg border border-white/80">
            <div className="flex items-center gap-2 mb-1">
              <TbTool className="w-4 h-4 text-slate-600" />
              <span className="text-xs font-bold text-slate-700">Recommended Action</span>
            </div>
            <p className="text-[11px] text-slate-600">{faultProfile.action}</p>
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
