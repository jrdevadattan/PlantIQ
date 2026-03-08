"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { DailyEnergyItem } from "@/lib/api";

interface EnergyBarChartProps {
  data: DailyEnergyItem[];
}

export function EnergyBarChart({ data }: EnergyBarChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400 text-xs">
        No energy data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis
          dataKey="day"
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
          width={45}
        />
        <Tooltip
          contentStyle={{
            background: "#1e293b",
            border: "none",
            borderRadius: "8px",
            fontSize: "11px",
            color: "#fff",
          }}
        />
        <Bar dataKey="kwh" fill="#0d9488" radius={[4, 4, 0, 0]} name="Energy (kWh)" />
      </BarChart>
    </ResponsiveContainer>
  );
}
