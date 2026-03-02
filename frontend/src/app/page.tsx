"use client";

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
  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-slate-800">Operations Overview</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            March 2, 2026 — Plant Floor Summary
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
          value="12"
          icon={TbBox}
          variant="teal"
          subValue="2 running"
          tooltip="Total batches started today including scheduled"
        />
        <StatCard
          label="Avg Energy"
          value="39.4"
          unit="kWh"
          icon={TbBolt}
          variant="amber"
          trend="down"
          trendValue="4.2%"
          tooltip="Average energy consumption per batch vs last week"
        />
        <StatCard
          label="Avg Quality"
          value="91.3"
          unit="%"
          icon={TbChartBar}
          variant="teal"
          trend="up"
          trendValue="1.1%"
          tooltip="Mean quality score across all completed batches today"
        />
        <StatCard
          label="Avg Yield"
          value="93.1"
          unit="%"
          icon={TbTrendingUp}
          variant="emerald"
          trend="up"
          trendValue="0.8%"
          tooltip="Average yield percentage today"
        />
        <StatCard
          label="Anomalies"
          value="2"
          icon={TbAlertTriangle}
          variant="amber"
          subValue="1 resolved"
          tooltip="Anomaly events detected across all batches today"
        />
        <StatCard
          label="Model Accuracy"
          value="95.8"
          unit="%"
          icon={TbShieldCheck}
          variant="emerald"
          subValue="MAPE 4.2%"
          tooltip="Multi-target prediction accuracy (1 - MAPE)"
        />
      </div>

      {/* Row 2: Gauges + Energy Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-5">
          <PerformanceGauges />
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
