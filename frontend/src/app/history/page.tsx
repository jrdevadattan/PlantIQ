"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  TbChartBar,
  TbBolt,
  TbTrendingUp,
  TbAlertTriangle,
} from "react-icons/tb";
import { StatCard } from "@/components/dashboard/StatCard";
import { fetchRecentBatches, type DashboardBatch } from "@/lib/api";
import { recentBatches as mockBatches } from "@/lib/mockData";

const BatchTable = dynamic(
  () => import("@/components/history/BatchTable").then(m => m.BatchTable),
  { ssr: false }
);

export default function HistoryPage() {
  const [batches, setBatches] = useState<DashboardBatch[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function loadBatches() {
      try {
        const data = await fetchRecentBatches(50);
        if (!cancelled) setBatches(data);
      } catch (err) {
        console.error("[HistoryPage] API fallback to mock data:", err);
        if (!cancelled) setBatches(mockBatches as unknown as DashboardBatch[]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadBatches();
    return () => { cancelled = true; };
  }, []);

  const completedBatches = batches.filter(b => b.status === "completed");
  const avgQuality = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.qualityScore, 0) / completedBatches.length).toFixed(1)
    : "—";
  const avgEnergy = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.energyKwh, 0) / completedBatches.length).toFixed(1)
    : "—";
  const avgYield = completedBatches.length
    ? (completedBatches.reduce((s, b) => s + b.yieldPct, 0) / completedBatches.length).toFixed(1)
    : "—";
  const anomalyCount = batches.filter(b => b.anomalyScore > 0.3).length;

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
          value={loading ? "—" : avgQuality}
          unit="%"
          icon={TbChartBar}
          variant="teal"
          tooltip="Average quality across completed batches"
        />
        <StatCard
          label="Avg Energy"
          value={loading ? "—" : avgEnergy}
          unit="kWh"
          icon={TbBolt}
          variant="amber"
          tooltip="Average energy consumption per batch"
        />
        <StatCard
          label="Avg Yield"
          value={loading ? "—" : avgYield}
          unit="%"
          icon={TbTrendingUp}
          variant="emerald"
          tooltip="Average yield across completed batches"
        />
        <StatCard
          label="Anomaly Events"
          value={loading ? "—" : anomalyCount}
          icon={TbAlertTriangle}
          variant="red"
          subValue={loading ? "" : `of ${batches.length} batches`}
          tooltip="Batches with anomaly score above threshold"
        />
      </div>

      <BatchTable />
    </div>
  );
}
