"use client";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CompliancePieEntry {
  name: string;
  value: number;
  color: string;
}

interface CompliancePieChartProps {
  data: CompliancePieEntry[];
}

export function CompliancePieChart({ data }: CompliancePieChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400 text-xs">
        No compliance data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
        >
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: "#1e293b",
            border: "none",
            borderRadius: "8px",
            fontSize: "11px",
            color: "#fff",
          }}
          formatter={(value: number) => `${value.toFixed(1)}%`}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
