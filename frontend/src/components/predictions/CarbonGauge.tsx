"use client";

import React from "react";
import { TbLeaf } from "react-icons/tb";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────
interface CarbonGaugeProps {
  /** Predicted CO₂ usage in kg */
  predictedKg: number;
  /** Batch budget in kg (default 42.0) */
  budgetKg?: number;
  /** Budget status string from API: ON_TRACK | WARNING | OVER_BUDGET */
  status?: string;
  /** Compact mode for inline placement (smaller) */
  compact?: boolean;
}

// ── Constants ────────────────────────────────────────────────
const DEFAULT_BUDGET = 42.0;

// Arc geometry (semicircle from left to right, bottom anchored)
const CX = 100;              // Center X
const CY = 100;              // Center Y
const RADIUS = 80;           // Arc radius
const STROKE_WIDTH = 14;     // Arc thickness
const START_ANGLE = Math.PI;  // 180° (left)
const END_ANGLE = 0;          // 0° (right)

// Zone thresholds (fraction of budget)
const GREEN_END = 0.65;       // 0–65% = green zone
const AMBER_END = 0.85;       // 65–85% = amber zone
// 85–100%+ = red zone

// Colors
const ZONE_COLORS = {
  green: "#10b981",   // emerald-500
  amber: "#f59e0b",   // amber-500
  red: "#ef4444",     // red-500
  track: "#e2e8f0",   // slate-200 (background)
};

const STATUS_CONFIG: Record<string, { bg: string; text: string; border: string; label: string }> = {
  ON_TRACK: {
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    border: "border-emerald-100",
    label: "ON TRACK",
  },
  WARNING: {
    bg: "bg-amber-50",
    text: "text-amber-700",
    border: "border-amber-100",
    label: "WARNING",
  },
  OVER_BUDGET: {
    bg: "bg-red-50",
    text: "text-red-700",
    border: "border-red-100",
    label: "OVER BUDGET",
  },
};

// ── Helpers ──────────────────────────────────────────────────

/** Convert a fraction (0…1+) to an angle on the semicircle */
function fractionToAngle(frac: number): number {
  const clamped = Math.max(0, Math.min(frac, 1.15)); // allow slight overshoot
  return START_ANGLE - clamped * Math.PI;
}

/** Polar to cartesian for SVG arc */
function polarToCartesian(cx: number, cy: number, r: number, angle: number) {
  return {
    x: cx + r * Math.cos(angle),
    y: cy - r * Math.sin(angle),
  };
}

/** Build an SVG arc path from startAngle to endAngle */
function describeArc(
  cx: number, cy: number, r: number,
  startAngle: number, endAngle: number,
): string {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArc = startAngle - endAngle > Math.PI ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

/** Determine status from predicted/budget */
function computeStatus(predicted: number, budget: number): string {
  const ratio = predicted / budget;
  if (ratio <= 0.8) return "ON_TRACK";
  if (ratio <= 1.0) return "WARNING";
  return "OVER_BUDGET";
}

/** Get the fill color for the needle based on fraction */
function needleColor(frac: number): string {
  if (frac <= GREEN_END) return ZONE_COLORS.green;
  if (frac <= AMBER_END) return ZONE_COLORS.amber;
  return ZONE_COLORS.red;
}

// ── Component ────────────────────────────────────────────────

export function CarbonGauge({
  predictedKg,
  budgetKg = DEFAULT_BUDGET,
  status: statusProp,
  compact = false,
}: CarbonGaugeProps) {
  const fraction = predictedKg / Math.max(budgetKg, 0.1);
  const status = statusProp ?? computeStatus(predictedKg, budgetKg);
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.ON_TRACK;
  const headroom = budgetKg - predictedKg;

  // Needle angle
  const needleAngle = fractionToAngle(Math.min(fraction, 1.1));
  const needleTip = polarToCartesian(CX, CY, RADIUS - STROKE_WIDTH / 2 - 4, needleAngle);
  const nColor = needleColor(fraction);

  // Zone arc paths (background)
  const greenArc = describeArc(CX, CY, RADIUS, START_ANGLE, fractionToAngle(GREEN_END));
  const amberArc = describeArc(CX, CY, RADIUS, fractionToAngle(GREEN_END), fractionToAngle(AMBER_END));
  const redArc = describeArc(CX, CY, RADIUS, fractionToAngle(AMBER_END), fractionToAngle(1.0));

  // Progress arc (filled portion)
  const clampedFrac = Math.max(0, Math.min(fraction, 1.1));
  const progressArc = describeArc(CX, CY, RADIUS, START_ANGLE, fractionToAngle(clampedFrac));

  const size = compact ? "w-[160px] h-[100px]" : "w-[200px] h-[120px]";
  const viewBox = "5 15 190 100";

  return (
    <div className={cn(
      "rounded-xl border p-4",
      config.bg, config.border,
      compact ? "p-3" : "p-4"
    )}>
      <div className="flex items-center gap-2 mb-2">
        <TbLeaf className={cn("w-4 h-4", config.text)} />
        <span className={cn("text-[11px] font-bold uppercase tracking-wide", config.text)}>
          Carbon Budget
        </span>
        <span className={cn(
          "ml-auto px-2 py-0.5 rounded-full text-[10px] font-bold",
          status === "ON_TRACK" && "bg-emerald-100 text-emerald-700",
          status === "WARNING" && "bg-amber-100 text-amber-700",
          status === "OVER_BUDGET" && "bg-red-100 text-red-700",
        )}>
          {config.label}
        </span>
      </div>

      {/* SVG Gauge */}
      <div className="flex justify-center">
        <svg
          viewBox={viewBox}
          className={size}
          aria-label={`Carbon budget gauge: ${predictedKg.toFixed(1)} of ${budgetKg} kg CO₂`}
        >
          {/* Background track */}
          <path
            d={describeArc(CX, CY, RADIUS, START_ANGLE, END_ANGLE)}
            fill="none"
            stroke={ZONE_COLORS.track}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
          />

          {/* Zone arcs (subtle colored backgrounds) */}
          <path d={greenArc} fill="none" stroke={ZONE_COLORS.green} strokeWidth={STROKE_WIDTH} strokeLinecap="butt" opacity={0.15} />
          <path d={amberArc} fill="none" stroke={ZONE_COLORS.amber} strokeWidth={STROKE_WIDTH} strokeLinecap="butt" opacity={0.15} />
          <path d={redArc} fill="none" stroke={ZONE_COLORS.red} strokeWidth={STROKE_WIDTH} strokeLinecap="butt" opacity={0.15} />

          {/* Progress arc (filled) */}
          <path
            d={progressArc}
            fill="none"
            stroke={nColor}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            style={{ transition: "d 0.6s ease, stroke 0.4s ease" }}
          />

          {/* Needle */}
          <line
            x1={CX}
            y1={CY}
            x2={needleTip.x}
            y2={needleTip.y}
            stroke={nColor}
            strokeWidth={2.5}
            strokeLinecap="round"
            style={{ transition: "x2 0.6s ease, y2 0.6s ease, stroke 0.4s ease" }}
          />
          <circle cx={CX} cy={CY} r={4} fill={nColor} style={{ transition: "fill 0.4s ease" }} />

          {/* Center value */}
          <text
            x={CX}
            y={CY + 2}
            textAnchor="middle"
            fontSize={compact ? 16 : 18}
            fontWeight="bold"
            fill="#1e293b"
            fontFamily="Inter, sans-serif"
          >
            {predictedKg.toFixed(1)}
          </text>
          <text
            x={CX}
            y={CY + 14}
            textAnchor="middle"
            fontSize={9}
            fill="#94a3b8"
            fontFamily="Inter, sans-serif"
          >
            kg CO₂
          </text>

          {/* Scale labels */}
          <text x={18} y={CY + 3} fontSize={8} fill="#94a3b8" textAnchor="middle">0</text>
          <text x={CX} y={18} fontSize={8} fill="#94a3b8" textAnchor="middle">{(budgetKg / 2).toFixed(0)}</text>
          <text x={182} y={CY + 3} fontSize={8} fill="#94a3b8" textAnchor="middle">{budgetKg.toFixed(0)}</text>
        </svg>
      </div>

      {/* Stats row */}
      <div className={cn(
        "flex items-center justify-between mt-1",
        compact ? "text-[10px]" : "text-[11px]"
      )}>
        <span className="text-slate-500">
          {predictedKg.toFixed(1)} / {budgetKg} kg
        </span>
        <span className={cn(
          "font-bold font-mono",
          headroom >= 0 ? "text-emerald-600" : "text-red-600"
        )}>
          {headroom >= 0 ? "+" : ""}{headroom.toFixed(1)} kg
        </span>
      </div>
    </div>
  );
}
