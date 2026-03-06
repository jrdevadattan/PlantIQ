"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { StatCard } from "@/components/dashboard/StatCard";
import {
  TbBox,
  TbBolt,
  TbChartBar,
  TbShieldCheck,
  TbAlertTriangle,
  TbTrendingUp,
} from "react-icons/tb";
import {
  fetchDashboardSummary,
  fetchLatestBatch,
  type DashboardSummary,
  type LatestBatch,
} from "@/lib/api";

const PerformanceGauges = dynamic(
  () => import("@/components/dashboard/PerformanceGauges").then(mod => mod.PerformanceGauges),
  { ssr: false }
);
const EnergyChart = dynamic(
  () => import("@/components/dashboard/EnergyChart").then(mod => mod.EnergyChart),
  { ssr: false }
);
const RecentBatches = dynamic(
  () => import("@/components/dashboard/RecentBatches").then(mod => mod.RecentBatches),
  { ssr: false }
);
const ShiftOverview = dynamic(
  () => import("@/components/dashboard/ShiftOverview").then(mod => mod.ShiftOverview),
  { ssr: false }
);

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [latestBatch, setLatestBatch] = useState<LatestBatch | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDashboard = useCallback(async () => {
    try {
      const [summaryData, batchData] = await Promise.all([
        fetchDashboardSummary(),
        fetchLatestBatch(),
      ]);
      setSummary(summaryData);
      setLatestBatch(batchData);
    } catch (err) {
      console.error("[DashboardPage] Failed to load dashboard data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
    // Refresh dashboard data every 30 seconds
    const interval = setInterval(loadDashboard, 30000);
    return () => clearInterval(interval);
  }, [loadDashboard]);

  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-slate-800">Operations Overview</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-Time Plant Floor Summary
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded-lg">
          <span className="text-[11px] font-bold text-slate-500">Today</span>
          <span className="text-[11px] text-slate-300">|</span>
          <span className="text-[11px] text-slate-400">7d</span>
          <span className="text-[11px] text-slate-400">30d</span>
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatCard
          label="Today's Batches"
          value={loading ? "—" : String(summary?.total_batches ?? 0)}
          icon={TbBox}
          variant="teal"
          subValue={loading ? "" : `${summary?.running_count ?? 0} running`}
          tooltip="Total batches started today including scheduled"
        />
        <StatCard
          label="Avg Energy"
          value={loading ? "—" : String(summary?.avg_energy ?? 0)}
          unit="kWh"
          icon={TbBolt}
          variant="amber"
          trend={summary?.energy_trend === "down" ? "down" : summary?.energy_trend === "up" ? "up" : undefined}
          trendValue={summary?.energy_trend_value || undefined}
          tooltip="Average energy consumption per batch vs last week"
        />
        <StatCard
          label="Avg Quality"
          value={loading ? "—" : String(summary?.avg_quality ?? 0)}
          unit="%"
          icon={TbChartBar}
          variant="teal"
          trend={summary?.quality_trend === "up" ? "up" : summary?.quality_trend === "down" ? "down" : undefined}
          trendValue={summary?.quality_trend_value || undefined}
          tooltip="Mean quality score across all completed batches today"
        />
        <StatCard
          label="Avg Yield"
          value={loading ? "—" : String(summary?.avg_yield ?? 0)}
          unit="%"
          icon={TbTrendingUp}
          variant="emerald"
          trend={summary?.yield_trend === "up" ? "up" : summary?.yield_trend === "down" ? "down" : undefined}
          trendValue={summary?.yield_trend_value || undefined}
          tooltip="Average yield percentage today"
        />
        <StatCard
          label="Anomalies"
          value={loading ? "—" : String(summary?.anomaly_count ?? 0)}
          icon={TbAlertTriangle}
          variant="amber"
          subValue={loading ? "" : `${summary?.resolved_count ?? 0} resolved`}
          tooltip="Anomaly events detected across all batches today"
        />
        <StatCard
          label="Model Accuracy"
          value={loading ? "—" : String(summary?.model_accuracy ?? 0)}
          unit="%"
          icon={TbShieldCheck}
          variant="emerald"
          subValue={loading ? "" : `MAPE ${summary?.mape_pct ?? 0}%`}
          tooltip="Multi-target prediction accuracy (1 - MAPE)"
        />
      </div>

      {/* Row 2: Gauges + Energy Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-5">
          <PerformanceGauges latestBatch={latestBatch} loading={loading} />
        </div>
        <div className="lg:col-span-7">
          <EnergyChart />
        </div>
      </div>

      {/* Row 3: Recent Batches + Shift Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-8">
          <RecentBatches />
        </div>
        <div className="lg:col-span-4">
          <ShiftOverview />
        </div>
      </div>
    </div>
  );
}
