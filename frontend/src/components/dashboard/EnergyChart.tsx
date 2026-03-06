"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { fetchEnergyDaily, type DailyEnergyItem } from "@/lib/api";
import { dailyEnergy as mockDailyEnergy } from "@/lib/mockData";
import { TbBolt, TbArrowUpRight } from "react-icons/tb";

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
        <p className="text-[11px] font-bold text-slate-500 mb-1">{label}</p>
        <p className="text-sm font-bold text-slate-800">
          {payload[0].value} kWh
        </p>
      </div>
    );
  }
  return null;
};

export function EnergyChart() {
  const [chartData, setChartData] = useState<{ day: string; kwh: number }[]>(mockDailyEnergy);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchEnergyDaily(7);
        if (!cancelled && data && data.length > 0) {
          setChartData(data.map((d: DailyEnergyItem) => ({
            day: d.day,
            kwh: d.kwh,
          })));
        }
      } catch (err) {
        console.error("[EnergyChart] API unavailable, using mock:", err);
        // keep mockDailyEnergy as fallback
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const totalKwh = chartData.reduce((sum, d) => sum + d.kwh, 0);
  const avgKwh = chartData.length > 0 ? Math.round(totalKwh / chartData.length) : 0;

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-amber-50 flex items-center justify-center">
            <TbBolt className="w-[18px] h-[18px] text-amber-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Weekly Energy</h3>
            <p className="text-[11px] text-slate-400">
              Avg {avgKwh} kWh/day
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-emerald-500">
          <TbArrowUpRight className="w-4 h-4" />
          <span className="text-[11px] font-bold">8% less</span>
        </div>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barSize={28}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="day"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fontWeight: 600, fill: "#94a3b8" }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(0,0,0,0.03)" }} />
            <Bar dataKey="kwh" fill="#14b8a6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
