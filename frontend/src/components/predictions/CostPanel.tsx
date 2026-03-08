"use client";

import { useState } from "react";
import {
  TbCurrencyRupee,
  TbLeaf,
  TbCalendar,
  TbArrowUp,
  TbArrowDown,
  TbMinus,
  TbChevronDown,
  TbChevronUp,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import type { CostTranslation } from "@/lib/api";

/* ── Types ─────────────────────────────────────────────── */

interface CostPanelProps {
  costData: CostTranslation | null;
}

/* ── Component ─────────────────────────────────────────── */

export function CostPanel({ costData }: CostPanelProps) {
  const [expanded, setExpanded] = useState(false);

  if (!costData) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 p-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-slate-50 flex items-center justify-center">
            <TbCurrencyRupee className="w-4 h-4 text-slate-300" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-400">Cost & Emissions</h2>
            <p className="text-[10px] text-slate-300">Run a prediction to see cost analysis</p>
          </div>
        </div>
      </div>
    );
  }

  const isOverBudget = costData.cost_variance_inr > 0;
  const varianceColor = isOverBudget ? "text-red-600" : "text-emerald-600";
  const co2Color =
    costData.co2_status === "ON_TRACK"
      ? "text-emerald-600"
      : costData.co2_status === "WARNING"
        ? "text-amber-600"
        : "text-red-600";

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between"
      >
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 flex items-center justify-center">
            <TbCurrencyRupee className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-left">
            <h2 className="text-sm font-bold text-slate-700">Cost & Emissions</h2>
            <p className="text-[10px] text-slate-400">Energy cost + CO₂ tracking</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-slate-800">
            ₹{costData.predicted_cost_inr.toFixed(0)}
          </span>
          {expanded ? (
            <TbChevronUp className="w-4 h-4 text-slate-300" />
          ) : (
            <TbChevronDown className="w-4 h-4 text-slate-300" />
          )}
        </div>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-slate-100 space-y-3">
          {/* Cost grid */}
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <TbCurrencyRupee className="w-4 h-4 text-slate-400 mx-auto mb-1" />
              <p className="text-lg font-bold text-slate-800">₹{costData.predicted_cost_inr.toFixed(0)}</p>
              <p className="text-[10px] text-slate-400">Predicted</p>
            </div>
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <p className="text-lg font-bold text-slate-800">₹{costData.target_cost_inr.toFixed(0)}</p>
              <p className="text-[10px] text-slate-400">Target</p>
            </div>
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <div className={cn("flex items-center justify-center gap-0.5", varianceColor)}>
                {isOverBudget ? <TbArrowUp className="w-3 h-3" /> : costData.cost_variance_inr < 0 ? <TbArrowDown className="w-3 h-3" /> : <TbMinus className="w-3 h-3" />}
                <p className="text-lg font-bold">₹{Math.abs(costData.cost_variance_inr).toFixed(0)}</p>
              </div>
              <p className="text-[10px] text-slate-400">{isOverBudget ? "Over" : "Under"} target</p>
            </div>
          </div>

          {/* Monthly + CO2 */}
          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2 p-3 bg-amber-50 rounded-lg border border-amber-100">
              <TbCalendar className="w-4 h-4 text-amber-600" />
              <div>
                <p className="text-xs font-bold text-slate-700">
                  ₹{(costData.monthly_projection_inr / 1000).toFixed(1)}K
                </p>
                <p className="text-[10px] text-slate-400">Monthly projection</p>
              </div>
            </div>
            <div className={cn(
              "flex items-center gap-2 p-3 rounded-lg border",
              costData.co2_status === "ON_TRACK" ? "bg-emerald-50 border-emerald-100" :
              costData.co2_status === "WARNING" ? "bg-amber-50 border-amber-100" : "bg-red-50 border-red-100"
            )}>
              <TbLeaf className={cn("w-4 h-4", co2Color)} />
              <div>
                <p className="text-xs font-bold text-slate-700">
                  {costData.co2_kg.toFixed(1)} kg
                </p>
                <p className={cn("text-[10px]", co2Color)}>
                  CO₂ · {costData.co2_status.replace("_", " ")}
                </p>
              </div>
            </div>
          </div>

          {/* Summary */}
          <p className="text-[11px] text-slate-500 leading-relaxed px-1">
            {costData.summary}
          </p>
        </div>
      )}
    </div>
  );
}
