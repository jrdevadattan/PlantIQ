"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Area,
  AreaChart,
  ReferenceLine,
} from "recharts";
import {
  TbActivityHeartbeat,
  TbBolt,
  TbPlayerPlay,
  TbPlayerPause,
  TbRefresh,
  TbAlertTriangle,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { detectAnomaly } from "@/lib/api";
import type { AnomalyDetectResponse } from "@/lib/api";
import { CarbonGauge } from "@/components/predictions/CarbonGauge";

// Generate a realistic power reading
function nextPowerReading(t: number, faultType: string = "normal"): number {
  let base = 3 + 4 * (1 - Math.exp(-t / 120)) - 2 * (1 - Math.exp(-(1800 - t) / 120));
  if (faultType === "bearing_wear") base += 0.003 * t;
  else if (faultType === "wet_material") base += 0.8 * Math.sin(t / 30) * (0.5 + Math.random());
  else base += (Math.random() - 0.5) * 0.2;
  return Math.max(0, Number(base.toFixed(2)));
}

const EnergyTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-2.5">
        <p className="text-[10px] font-semibold text-slate-400 mb-1">
          t = {payload[0]?.payload?.time}s
        </p>
        <p className="text-xs font-bold text-slate-800">
          {payload[0]?.value?.toFixed(2)} kW
        </p>
      </div>
    );
  }
  return null;
};

export function LiveMonitor() {
  const [isRunning, setIsRunning] = useState(true);
  const [faultType, setFaultType] = useState<"normal" | "bearing_wear" | "wet_material">("normal");
  const [powerData, setPowerData] = useState<{ time: number; power: number }[]>([]);
  const [energyConsumed, setEnergyConsumed] = useState(0);
  const [anomalyScore, setAnomalyScore] = useState(0.08);
  const [anomalyResult, setAnomalyResult] = useState<AnomalyDetectResponse | null>(null);
  const tickRef = useRef(0);
  const powerAccRef = useRef<number[]>([]);

  // Sliding window forecast data
  const [forecastData, setForecastData] = useState([
    { min: 0, predicted: 38.8, lower: 32.8, upper: 44.8 },
  ]);

  // Send accumulated power readings to backend for real anomaly detection
  const runAnomalyCheck = useCallback(async (readings: number[], elapsed: number) => {
    try {
      const batchId = `live-${faultType}-${Date.now()}`;
      const response = await detectAnomaly(batchId, readings, elapsed);
      setAnomalyScore(response.anomaly_score);
      setAnomalyResult(response);
    } catch (err) {
      console.error("[LiveMonitor] Anomaly check failed:", err);
      // Keep last known score on failure
    }
  }, [faultType]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      tickRef.current += 1;
      const t = tickRef.current;
      const power = nextPowerReading(t, faultType);

      // Accumulate raw readings for anomaly detection
      powerAccRef.current.push(power);

      setPowerData((prev) => {
        const next = [...prev, { time: t, power }];
        return next.length > 300 ? next.slice(-300) : next;
      });

      // Accumulate energy (kW * 1 second / 3600 = kWh)
      setEnergyConsumed((prev) => prev + power / 3600);

      // Send to backend for real anomaly scoring every 30 ticks
      if (t % 30 === 0 && powerAccRef.current.length > 0) {
        runAnomalyCheck([...powerAccRef.current], t);
      }

      // Update forecast every 30 ticks (~30 seconds)
      if (t % 30 === 0) {
        const progress = Math.min(t / 1800, 1);
        const errorRange = 6 * (1 - progress * 0.8);
        const drift = faultType !== "normal" ? 4 + Math.random() * 3 : Math.random() * 2 - 1;
        const predicted = 38.8 + drift;
        setForecastData((prev) => [
          ...prev,
          {
            min: Math.floor(t / 60),
            predicted: Number(predicted.toFixed(1)),
            lower: Number((predicted - errorRange).toFixed(1)),
            upper: Number((predicted + errorRange).toFixed(1)),
          },
        ]);
      }
    }, 100); // 10x speed for demo

    return () => clearInterval(interval);
  }, [isRunning, faultType]);

  const resetSimulation = () => {
    tickRef.current = 0;
    powerAccRef.current = [];
    setPowerData([]);
    setEnergyConsumed(0);
    setAnomalyScore(0.08);
    setAnomalyResult(null);
    setForecastData([{ min: 0, predicted: 38.8, lower: 32.8, upper: 44.8 }]);
  };

  const elapsedMin = Math.floor(tickRef.current / 60);
  const elapsedSec = tickRef.current % 60;
  const progress = Math.min((tickRef.current / 1800) * 100, 100);
  const isAnomaly = anomalyScore > 0.3;

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-teal-50 flex items-center justify-center">
              <TbActivityHeartbeat className="w-[18px] h-[18px] text-teal-600" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">Live Power Monitor</h3>
              <p className="text-[11px] text-slate-400">
                Batch B-2026-0302-001 — Real-time sensor feed (10x speed)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={cn(
                "p-2 rounded-lg border transition-all",
                isRunning
                  ? "bg-amber-50 border-amber-200 text-amber-600"
                  : "bg-emerald-50 border-emerald-200 text-emerald-600"
              )}
            >
              {isRunning ? <TbPlayerPause className="w-4 h-4" /> : <TbPlayerPlay className="w-4 h-4" />}
            </button>
            <button
              onClick={resetSimulation}
              className="p-2 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 transition-all"
            >
              <TbRefresh className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Simulation controls */}
        <div className="flex items-center gap-3 mb-4">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            Simulate:
          </span>
          {(["normal", "wet_material", "bearing_wear"] as const).map((ft) => (
            <button
              key={ft}
              onClick={() => { setFaultType(ft); resetSimulation(); }}
              className={cn(
                "px-3 py-1.5 rounded-lg text-[11px] font-bold border transition-all",
                faultType === ft
                  ? "bg-teal-50 text-teal-700 border-teal-200"
                  : "bg-white text-slate-400 border-slate-200 hover:text-slate-600"
              )}
            >
              {ft === "normal" ? "Normal" : ft === "wet_material" ? "Wet Material" : "Bearing Wear"}
            </button>
          ))}
        </div>

        {/* Status bar */}
        <div className="flex items-center gap-6 p-3 bg-slate-50 rounded-lg border border-slate-100">
          <div>
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide">Elapsed</p>
            <p className="text-sm font-mono font-bold text-slate-700">
              {String(elapsedMin).padStart(2, "0")}:{String(elapsedSec).padStart(2, "0")}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide">Energy Used</p>
            <p className="text-sm font-mono font-bold text-slate-700">
              {energyConsumed.toFixed(2)} kWh
            </p>
          </div>
          <div>
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide">Anomaly Score</p>
            <div className="flex items-center gap-2">
              <span className={`status-dot ${isAnomaly ? "danger" : "active"}`} />
              <p className={cn("text-sm font-mono font-bold", isAnomaly ? "text-red-600" : "text-emerald-600")}>
                {anomalyScore.toFixed(2)}
              </p>
            </div>
          </div>
          <div className="flex-1">
            <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wide mb-1">Progress</p>
            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  isAnomaly ? "bg-red-500" : "bg-teal-500"
                )}
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Anomaly Alert (from backend) */}
      {anomalyResult && anomalyResult.is_anomaly && (
        <div className={cn(
          "rounded-xl border p-4",
          anomalyResult.severity === "CRITICAL" ? "bg-red-50 border-red-100" : "bg-amber-50 border-amber-100"
        )}>
          <div className="flex items-start gap-3">
            <TbAlertTriangle className={cn(
              "w-5 h-5 flex-shrink-0 mt-0.5",
              anomalyResult.severity === "CRITICAL" ? "text-red-500" : "text-amber-500"
            )} />
            <div>
              <p className={cn(
                "text-xs font-bold",
                anomalyResult.severity === "CRITICAL" ? "text-red-700" : "text-amber-700"
              )}>
                {anomalyResult.severity} — {anomalyResult.diagnosis?.fault_type ?? "Unknown"}
              </p>
              <p className="text-[11px] text-slate-500 mt-0.5">
                {anomalyResult.diagnosis?.human_readable ?? "Anomaly detected in power curve"}
              </p>
              {anomalyResult.diagnosis?.recommended_action && (
                <p className="text-[11px] text-slate-600 mt-1 font-medium">
                  Action: {anomalyResult.diagnosis.recommended_action}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Live Carbon Budget Gauge */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <CarbonGauge
          predictedKg={Number((energyConsumed * 0.82 / Math.max(progress / 100, 0.01)).toFixed(1))}
          budgetKg={42.0}
          compact
        />
        <div className="bg-white rounded-xl border border-slate-100 p-4">
          <div className="flex items-center gap-2 mb-3">
            <TbBolt className="w-4 h-4 text-amber-500" />
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">
              Energy Metrics
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-[10px] text-slate-400 uppercase font-bold">Consumed</p>
              <p className="text-lg font-bold font-mono text-slate-700">
                {energyConsumed.toFixed(2)} <span className="text-xs text-slate-400">kWh</span>
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-400 uppercase font-bold">Projected</p>
              <p className="text-lg font-bold font-mono text-slate-700">
                {progress > 0 ? (energyConsumed / (progress / 100)).toFixed(1) : "—"} <span className="text-xs text-slate-400">kWh</span>
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-400 uppercase font-bold">CO₂ So Far</p>
              <p className="text-lg font-bold font-mono text-slate-700">
                {(energyConsumed * 0.82).toFixed(1)} <span className="text-xs text-slate-400">kg</span>
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-400 uppercase font-bold">CO₂ Target</p>
              <p className="text-lg font-bold font-mono text-emerald-600">
                42.0 <span className="text-xs text-slate-400">kg</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Power Curve Chart */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-indigo-50 flex items-center justify-center">
            <TbBolt className="w-[18px] h-[18px] text-indigo-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Power Draw (kW)</h3>
            <p className="text-[11px] text-slate-400">
              Live power curve — {powerData.length} readings captured
            </p>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={powerData}>
              <defs>
                <linearGradient id="powerGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={isAnomaly ? "#ef4444" : "#14b8a6"} stopOpacity={0.15} />
                  <stop offset="100%" stopColor={isAnomaly ? "#ef4444" : "#14b8a6"} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="time"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                tickFormatter={(v) => `${Math.floor(v / 60)}m`}
                interval={59}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                width={30}
                domain={[0, 10]}
              />
              <Tooltip content={<EnergyTooltip />} />
              <ReferenceLine y={7} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "Threshold", fontSize: 10, fill: "#f59e0b" }} />
              <Area
                type="monotone"
                dataKey="power"
                stroke={isAnomaly ? "#ef4444" : "#14b8a6"}
                strokeWidth={1.5}
                fill="url(#powerGrad)"
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sliding Window Forecast */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <h3 className="text-sm font-bold text-slate-800 mb-1">Energy Forecast</h3>
        <p className="text-[11px] text-slate-400 mb-4">
          Predicted final energy consumption — updates every 30 seconds
        </p>

        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="min"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                tickFormatter={(v) => `${v}m`}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                width={35}
                domain={["auto", "auto"]}
              />
              <ReferenceLine y={40} stroke="#22c55e" strokeDasharray="5 5" label={{ value: "Target", fontSize: 10, fill: "#22c55e" }} />
              <Line type="monotone" dataKey="upper" stroke="#e2e8f0" strokeWidth={1} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="lower" stroke="#e2e8f0" strokeWidth={1} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="#6366f1" strokeWidth={2} dot={{ r: 3, fill: "#6366f1" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
