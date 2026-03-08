"use client";

import { useState, useEffect, useMemo } from "react";
import {
  TbChartLine,
  TbLeaf,
  TbCurrencyRupee,
  TbTrendingUp,
  TbBolt,
  TbAlertTriangle,
  TbChartBar,
  TbShieldCheck,
} from "react-icons/tb";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";
import {
  fetchDashboardSummary,
  fetchEnergyDaily,
  fetchShiftPerformance,
  fetchComplianceReport,
  fetchCostConfig,
} from "@/lib/api";
import type {
  DashboardSummary,
  DailyEnergyItem,
  ShiftPerformanceData,
  ComplianceReport,
  CostConfig,
} from "@/lib/api";

/* ─── Management Dashboard Page ─────────────────────────────── */
export default function ManagementPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [energy, setEnergy] = useState<DailyEnergyItem[]>([]);
  const [shifts, setShifts] = useState<ShiftPerformanceData[]>([]);
  const [compliance, setCompliance] = useState<ComplianceReport | null>(null);
  const [costConfig, setCostConfig] = useState<CostConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [energyDays, setEnergyDays] = useState(14);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const [s, e, sh, c, cc] = await Promise.all([
          fetchDashboardSummary(),
          fetchEnergyDaily(energyDays),
          fetchShiftPerformance(),
          fetchComplianceReport(),
          fetchCostConfig(),
        ]);
        if (mounted) {
          setSummary(s);
          setEnergy(e);
          setShifts(sh);
          setCompliance(c);
          setCostConfig(cc);
          setLoading(false);
        }
      } catch (err) {
        console.error("[ManagementPage]", err);
        if (mounted) {
          setError("Failed to load management data. Is the backend running?");
          setLoading(false);
        }
      }
    };
    load();
    return () => { mounted = false; };
  }, [energyDays]);

  /* ── Derived metrics ── */
  const costMetrics = useMemo(() => {
    if (!compliance || !costConfig) return null;
    const tariff = costConfig.tariff_inr_per_kwh;
    const totalCost = compliance.energy_stats.total_kwh * tariff;
    const avgCostPerBatch = compliance.energy_stats.mean_kwh * tariff;
    const targetCost = costConfig.energy_target_kwh * tariff * compliance.total_batches;
    const savings = targetCost - totalCost;
    const monthlyProjection = avgCostPerBatch * costConfig.batches_per_day * 30;
    return {
      totalCost: Math.round(totalCost),
      avgCostPerBatch: Math.round(avgCostPerBatch * 100) / 100,
      targetCost: Math.round(targetCost),
      savings: Math.round(savings),
      monthlyProjection: Math.round(monthlyProjection),
      co2Total: Math.round(compliance.carbon_stats.total_kg * 10) / 10,
      co2PerBatch: Math.round(compliance.carbon_stats.mean_kg_per_batch * 10) / 10,
    };
  }, [compliance, costConfig]);

  const compliancePie = useMemo(() => {
    if (!compliance) return [];
    return [
      { name: "On Track", value: compliance.compliance.on_track_pct, color: "#22c55e" },
      { name: "Caution", value: compliance.compliance.caution_pct, color: "#f59e0b" },
      { name: "Exceeded", value: compliance.compliance.exceeded_pct, color: "#ef4444" },
    ].filter((d) => d.value > 0);
  }, [compliance]);

  /* ── Loading / Error states ── */
  if (loading) {
    return (
      <div className="p-5 max-w-[1600px] mx-auto space-y-5">
        <h1 className="text-lg font-bold text-slate-800">Management Dashboard</h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-white rounded-xl border border-slate-100 p-4 h-24 animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-white rounded-xl border border-slate-100 p-4 h-72 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-5 max-w-[1600px] mx-auto">
        <h1 className="text-lg font-bold text-slate-800 mb-4">Management Dashboard</h1>
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
          <TbAlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
          <p className="text-sm font-semibold text-red-700">{error}</p>
          <p className="text-xs text-red-400 mt-1">Ensure the FastAPI server is running at :8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-5 max-w-[1600px] mx-auto space-y-5">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-slate-800">Management Dashboard</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Plant-wide KPIs, cost analysis, and compliance overview
          </p>
        </div>
        <div className="flex items-center gap-2">
          {[7, 14, 30].map((d) => (
            <button
              key={d}
              onClick={() => setEnergyDays(d)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all",
                energyDays === d
                  ? "bg-teal-50 text-teal-700 border border-teal-200"
                  : "text-slate-400 hover:text-slate-600 hover:bg-slate-50"
              )}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {/* ── Row 1: KPI Summary Cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          icon={TbChartBar}
          label="Total Batches"
          value={summary?.total_batches ?? 0}
          variant="slate"
        />
        <KPICard
          icon={TbBolt}
          label="Avg Energy"
          value={`${summary?.avg_energy?.toFixed(1) ?? "—"}`}
          unit="kWh"
          trend={summary?.energy_trend ?? "flat"}
          trendValue={summary?.energy_trend_value ?? ""}
          variant="amber"
        />
        <KPICard
          icon={TbShieldCheck}
          label="Avg Quality"
          value={`${summary?.avg_quality?.toFixed(1) ?? "—"}`}
          unit="%"
          trend={summary?.quality_trend ?? "flat"}
          trendValue={summary?.quality_trend_value ?? ""}
          variant="teal"
        />
        <KPICard
          icon={TbTrendingUp}
          label="Avg Yield"
          value={`${summary?.avg_yield?.toFixed(1) ?? "—"}`}
          unit="%"
          trend={summary?.yield_trend ?? "flat"}
          trendValue={summary?.yield_trend_value ?? ""}
          variant="emerald"
        />
      </div>

      {/* ── Row 2: Cost & CO₂ Cards ── */}
      {costMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KPICard
            icon={TbCurrencyRupee}
            label="Total Energy Cost"
            value={`₹${costMetrics.totalCost.toLocaleString()}`}
            variant="amber"
          />
          <KPICard
            icon={TbCurrencyRupee}
            label="Monthly Projection"
            value={`₹${costMetrics.monthlyProjection.toLocaleString()}`}
            variant="slate"
          />
          <KPICard
            icon={TbLeaf}
            label="Total CO₂"
            value={`${costMetrics.co2Total.toLocaleString()}`}
            unit="kg"
            variant="emerald"
          />
          <KPICard
            icon={TbLeaf}
            label="CO₂ per Batch"
            value={`${costMetrics.co2PerBatch}`}
            unit="kg"
            variant="teal"
          />
        </div>
      )}

      {/* ── Row 3: Charts ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Energy Trend — wide */}
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-100 p-4 card-hover">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-700">Energy Consumption Trend</h2>
              <p className="text-[11px] text-slate-400">Daily total kWh — last {energyDays} days</p>
            </div>
            <TbChartLine className="w-5 h-5 text-slate-300" />
          </div>
          <div className="h-64">
            {energy.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={energy}>
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
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                No energy data available
              </div>
            )}
          </div>
        </div>

        {/* Compliance Pie — narrow */}
        <div className="lg:col-span-4 bg-white rounded-xl border border-slate-100 p-4 card-hover">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-700">CO₂ Compliance</h2>
              <p className="text-[11px] text-slate-400">
                {compliance?.total_batches ?? 0} batches evaluated
              </p>
            </div>
            <TbLeaf className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="h-48">
            {compliancePie.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={compliancePie}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {compliancePie.map((entry, i) => (
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
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                No compliance data
              </div>
            )}
          </div>
          {/* Legend */}
          <div className="flex items-center justify-center gap-4 mt-2">
            {compliancePie.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ background: d.color }}
                />
                <span className="text-[10px] font-semibold text-slate-500">
                  {d.name} ({d.value.toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Row 4: Shift Performance + Savings ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Shift Performance */}
        <div className="lg:col-span-7 bg-white rounded-xl border border-slate-100 p-4 card-hover">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-700">Shift Performance</h2>
              <p className="text-[11px] text-slate-400">Quality, yield, and energy by shift</p>
            </div>
          </div>
          {shifts.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-100">
                    {["Shift", "Batches", "Quality", "Yield", "Energy"].map((h) => (
                      <th key={h} className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pb-2 pr-4">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {shifts.map((s) => (
                    <tr key={s.shift} className="border-b border-slate-50 hover:bg-slate-50/50">
                      <td className="py-2.5 text-xs font-semibold text-slate-700">{s.shift}</td>
                      <td className="py-2.5 text-xs text-slate-600">{s.batches}</td>
                      <td className="py-2.5">
                        <span className={cn(
                          "text-xs font-bold",
                          s.quality >= 90 ? "text-emerald-600" : s.quality >= 80 ? "text-amber-600" : "text-red-500"
                        )}>
                          {s.quality.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5">
                        <span className="text-xs font-bold text-teal-600">{s.yield_pct.toFixed(1)}%</span>
                      </td>
                      <td className="py-2.5">
                        <span className="text-xs font-bold text-slate-700">{s.energy.toFixed(1)} kWh</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="h-32 flex items-center justify-center text-slate-400 text-xs">
              No shift data available
            </div>
          )}
        </div>

        {/* Savings & ROI summary */}
        <div className="lg:col-span-5 bg-white rounded-xl border border-slate-100 p-4 card-hover">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-700">Savings & ROI</h2>
              <p className="text-[11px] text-slate-400">Cumulative optimization impact</p>
            </div>
            <TbCurrencyRupee className="w-5 h-5 text-amber-400" />
          </div>
          {compliance && costConfig ? (
            <div className="space-y-4">
              {/* Energy savings */}
              <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-100">
                <p className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider mb-1">
                  Energy Savings
                </p>
                <p className="text-2xl font-bold text-emerald-700">
                  {compliance.cumulative_savings.energy_kwh.toFixed(0)} kWh
                </p>
                <p className="text-[11px] text-emerald-500 mt-1">
                  ≈ ₹{(compliance.cumulative_savings.energy_kwh * costConfig.tariff_inr_per_kwh).toFixed(0)} saved
                </p>
              </div>

              {/* CO₂ savings */}
              <div className="p-3 bg-teal-50 rounded-lg border border-teal-100">
                <p className="text-[10px] font-bold text-teal-500 uppercase tracking-wider mb-1">
                  Carbon Reduction
                </p>
                <p className="text-2xl font-bold text-teal-700">
                  {compliance.cumulative_savings.co2_kg.toFixed(1)} kg CO₂
                </p>
                <p className="text-[11px] text-teal-500 mt-1">
                  Energy trend: {compliance.energy_stats.trend}
                </p>
              </div>

              {/* Cost config summary */}
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Configuration
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-[10px] text-slate-400">Tariff</p>
                    <p className="text-xs font-bold text-slate-700">₹{costConfig.tariff_inr_per_kwh}/kWh</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-400">CO₂ Factor</p>
                    <p className="text-xs font-bold text-slate-700">{costConfig.co2_factor_kg_per_kwh} kg/kWh</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-400">Energy Budget</p>
                    <p className="text-xs font-bold text-slate-700">{costConfig.energy_target_kwh} kWh/batch</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-400">Optimization</p>
                    <p className="text-xs font-bold text-slate-700">{costConfig.optimization_headroom_pct}%</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center text-slate-400 text-xs">
              No savings data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── KPI Card (local to management page) ────────────────── */
interface KPICardProps {
  icon: React.ElementType;
  label: string;
  value: string | number;
  unit?: string;
  trend?: string;
  trendValue?: string;
  variant?: "teal" | "amber" | "emerald" | "slate";
}

const variantMap = {
  teal: { icon: "bg-teal-50 text-teal-600", trend: "text-teal-600" },
  amber: { icon: "bg-amber-50 text-amber-600", trend: "text-amber-600" },
  emerald: { icon: "bg-emerald-50 text-emerald-600", trend: "text-emerald-600" },
  slate: { icon: "bg-slate-100 text-slate-500", trend: "text-slate-500" },
};

function KPICard({ icon: Icon, label, value, unit, trend, trendValue, variant = "teal" }: KPICardProps) {
  const styles = variantMap[variant];
  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", styles.icon)}>
          <Icon className="w-4 h-4" />
        </div>
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-slate-800">{value}</span>
        {unit && <span className="text-xs font-semibold text-slate-400">{unit}</span>}
      </div>
      {trendValue && (
        <span className={cn(
          "text-[11px] font-bold",
          trend === "down" ? "text-emerald-500" : trend === "up" ? "text-red-500" : "text-slate-400"
        )}>
          {trendValue}
        </span>
      )}
    </div>
  );
}
