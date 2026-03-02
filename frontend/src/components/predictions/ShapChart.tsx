"use client";

import React from "react";
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
import { mockShapValues } from "@/lib/mockData";
import { TbBulb, TbArrowBadgeRight } from "react-icons/tb";

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3 max-w-xs">
        <p className="text-xs font-bold text-slate-700 mb-1">{data.feature}</p>
        <p className="text-[11px] text-slate-500">
          {data.contribution > 0 ? "Increases" : "Decreases"} energy by{" "}
          <span className="font-bold text-slate-800">
            {Math.abs(data.contribution).toFixed(1)} kWh
          </span>
        </p>
      </div>
    );
  }
  return null;
};

export function ShapChart() {
  const sortedData = [...mockShapValues].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  );

  const topDriver = sortedData[0];

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-5">
      <div className="flex items-center gap-3 mb-1">
        <div className="w-9 h-9 rounded-lg bg-violet-50 flex items-center justify-center">
          <TbBulb className="w-[18px] h-[18px] text-violet-600" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-slate-800">
            Why This Prediction?
          </h3>
          <p className="text-[11px] text-slate-400">
            Feature contributions to energy prediction (SHAP values)
          </p>
        </div>
      </div>

      {/* Explanation callout */}
      <div className="mt-4 mb-4 p-3 bg-amber-50 border border-amber-100 rounded-lg flex items-start gap-2">
        <TbArrowBadgeRight className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
        <p className="text-[11px] text-amber-800 leading-relaxed">
          <span className="font-bold">{topDriver.feature}</span> is the biggest
          driver, adding{" "}
          <span className="font-bold">
            +{topDriver.contribution.toFixed(1)} kWh
          </span>{" "}
          above the baseline. Consider reducing it to lower energy consumption.
        </p>
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
