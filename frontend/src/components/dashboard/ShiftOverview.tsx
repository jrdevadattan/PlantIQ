"use client";

import React from "react";
import { shiftPerformance } from "@/lib/mockData";
import { TbSunHigh, TbSunset2, TbMoon } from "react-icons/tb";
import { cn } from "@/lib/utils";

const shiftIcons = [TbSunHigh, TbSunset2, TbMoon];
const shiftColors = [
  "bg-amber-50 text-amber-600 border-amber-100",
  "bg-orange-50 text-orange-600 border-orange-100",
  "bg-indigo-50 text-indigo-600 border-indigo-100",
];

export function ShiftOverview() {
  return (
    <div className="bg-white rounded-xl border border-slate-100 p-5">
      <h3 className="text-sm font-bold text-slate-800 mb-1">Shift Performance</h3>
      <p className="text-[11px] text-slate-400 mb-4">Comparison across shifts today</p>

      <div className="space-y-3">
        {shiftPerformance.map((shift, i) => {
          const Icon = shiftIcons[i];
          return (
            <div
              key={shift.shift}
              className="flex items-center gap-3 p-3 rounded-lg bg-slate-50/50 border border-slate-100"
            >
              <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center border", shiftColors[i])}>
                <Icon className="w-4 h-4" />
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-[11px] font-bold text-slate-700 truncate">{shift.shift}</p>
                <p className="text-[10px] text-slate-400">{shift.batches} batches</p>
              </div>

              <div className="flex items-center gap-4 text-right">
                <div>
                  <p className="text-[10px] text-slate-400">Quality</p>
                  <p className="text-xs font-bold text-slate-700">{shift.quality}%</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-400">Yield</p>
                  <p className="text-xs font-bold text-slate-700">{shift.yield}%</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-400">Energy</p>
                  <p className="text-xs font-bold text-slate-700">{shift.energy} kWh</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
