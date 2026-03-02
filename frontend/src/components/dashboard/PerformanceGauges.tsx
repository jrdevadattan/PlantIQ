"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface GaugeProps {
  label: string;
  value: number;
  max?: number;
  unit?: string;
  color: string;
  size?: number;
}

function Gauge({ label, value, max = 100, unit = "%", color, size = 100 }: GaugeProps) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const progress = (value / max) * circumference;
  const offset = circumference - progress;

  const getStatusColor = () => {
    if (value >= 90) return "text-emerald-500";
    if (value >= 80) return "text-amber-500";
    return "text-red-500";
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox="0 0 100 100"
          className="transform -rotate-90"
        >
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="#f1f5f9"
            strokeWidth="8"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-1000 ease-out"
            style={{ animation: "gauge-fill 1.2s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("text-lg font-bold", getStatusColor())}>
            {value.toFixed(1)}
          </span>
          <span className="text-[9px] font-semibold text-slate-400">{unit}</span>
        </div>
      </div>
      <span className="text-[11px] font-semibold text-slate-500">{label}</span>
    </div>
  );
}

export function PerformanceGauges() {
  return (
    <div className="bg-white rounded-xl border border-slate-100 p-5">
      <h3 className="text-sm font-bold text-slate-800 mb-1">
        Current Batch Performance
      </h3>
      <p className="text-[11px] text-slate-400 mb-5">
        B-2026-0302-001 — 26% complete
      </p>

      <div className="grid grid-cols-4 gap-4">
        <Gauge label="Quality" value={90.4} color="#14b8a6" />
        <Gauge label="Yield" value={92.8} color="#22c55e" />
        <Gauge label="Performance" value={93.5} color="#6366f1" />
        <Gauge label="Energy" value={40.1} max={60} unit="kWh" color="#f59e0b" />
      </div>

      {/* Batch progress bar */}
      <div className="mt-5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold text-slate-500">
            Batch Progress
          </span>
          <span className="text-[11px] font-bold text-teal-600">
            7:48 / 30:00
          </span>
        </div>
        <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-teal-500 to-teal-400 rounded-full transition-all duration-1000"
            style={{ width: "26%" }}
          />
        </div>
      </div>
    </div>
  );
}
