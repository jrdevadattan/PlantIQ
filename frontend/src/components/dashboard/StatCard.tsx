"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { TbInfoCircle } from "react-icons/tb";
import { useState } from "react";
import type { IconType } from "react-icons";

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subValue?: string;
  icon: IconType;
  variant?: "teal" | "amber" | "red" | "slate" | "emerald";
  trend?: "up" | "down" | "flat";
  trendValue?: string;
  tooltip?: string;
}

const variantStyles = {
  teal: {
    icon: "bg-teal-50 text-teal-600",
    border: "border-teal-100",
    trend: "text-teal-600",
  },
  amber: {
    icon: "bg-amber-50 text-amber-600",
    border: "border-amber-100",
    trend: "text-amber-600",
  },
  red: {
    icon: "bg-red-50 text-red-500",
    border: "border-red-100",
    trend: "text-red-500",
  },
  slate: {
    icon: "bg-slate-100 text-slate-500",
    border: "border-slate-100",
    trend: "text-slate-500",
  },
  emerald: {
    icon: "bg-emerald-50 text-emerald-600",
    border: "border-emerald-100",
    trend: "text-emerald-600",
  },
};

export function StatCard({
  label,
  value,
  unit,
  subValue,
  icon: Icon,
  variant = "teal",
  trend,
  trendValue,
  tooltip,
}: StatCardProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const styles = variantStyles[variant];

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover relative">
      <div className="flex items-start justify-between mb-3">
        <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center", styles.icon)}>
          <Icon className="w-[18px] h-[18px]" />
        </div>
        {tooltip && (
          <div className="relative">
            <button
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              className="text-slate-300 hover:text-slate-400 transition-colors"
            >
              <TbInfoCircle className="w-4 h-4" />
            </button>
            {showTooltip && (
              <div className="absolute right-0 top-6 w-48 p-2 bg-slate-800 text-white text-[10px] rounded-lg shadow-lg z-50 leading-relaxed">
                {tooltip}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="space-y-1">
        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">
          {label}
        </p>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold text-slate-800 animate-count">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-semibold text-slate-400">{unit}</span>
          )}
        </div>
        {(subValue || trendValue) && (
          <div className="flex items-center gap-1.5">
            {trendValue && (
              <span
                className={cn(
                  "text-[11px] font-bold",
                  trend === "up" ? "text-emerald-500" : trend === "down" ? "text-red-500" : "text-slate-400"
                )}
              >
                {trend === "up" ? "+" : trend === "down" ? "-" : ""}{trendValue}
              </span>
            )}
            {subValue && (
              <span className="text-[11px] text-slate-400">{subValue}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
