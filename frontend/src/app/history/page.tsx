"use client";

import dynamic from "next/dynamic";
import {
  TbChartBar,
  TbBolt,
  TbTrendingUp,
  TbAlertTriangle,
} from "react-icons/tb";
import { StatCard } from "@/components/dashboard/StatCard";
import { recentBatches } from "@/lib/mockData";

const BatchTable = dynamic(
  () => import("@/components/history/BatchTable").then(m => m.BatchTable),
  { ssr: false }
);

export default function HistoryPage() {
  const completedBatches = recentBatches.filter(b => b.status === "completed");
  const avgQuality = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.qualityScore, 0) / completedBatches.length).toFixed(1)
    : "—";
  const avgEnergy = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.energyKwh, 0) / completedBatches.length).toFixed(1)
    : "—";
  const avgYield = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.yieldPct, 0) / completedBatches.length).toFixed(1)
    : "—";
  const anomalyCount = recentBatches.filter(b => b.anomalyScore > 0.3).length;

  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-lg font-bold text-slate-800">Batch History</h1>
        <p className="text-xs text-slate-400 mt-0.5">
          Complete production log with predictions, actuals, and anomaly events
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Avg Quality"
          value={avgQuality}
          unit="%"
          icon={TbChartBar}
          variant="teal"
          tooltip="Average quality across completed batches"
        />
        <StatCard
          label="Avg Energy"
          value={avgEnergy}
          unit="kWh"
          icon={TbBolt}
          variant="amber"
          tooltip="Average energy consumption per batch"
        />
        <StatCard
          label="Avg Yield"
          value={avgYield}
          unit="%"
          icon={TbTrendingUp}
          variant="emerald"
          tooltip="Average yield across completed batches"
        />
        <StatCard
          label="Anomaly Events"
          value={anomalyCount}
          icon={TbAlertTriangle}
          variant="red"
          subValue={`of ${recentBatches.length} batches`}
          tooltip="Batches with anomaly score above threshold"
        />
      </div>

      <BatchTable />
    </div>
  );
}
