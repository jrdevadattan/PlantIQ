"use client";

import React, { useState, useEffect } from "react";
import { recentBatches as mockBatches } from "@/lib/mockData";
import type { BatchRecord } from "@/lib/mockData";
import { fetchRecentBatches, type DashboardBatch } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  TbCircleCheck,
  TbPlayerPlay,
  TbClock,
  TbAlertTriangle,
  TbChevronRight,
  TbListDetails,
} from "react-icons/tb";
import Link from "next/link";

const statusConfig = {
  completed: {
    label: "Completed",
    icon: TbCircleCheck,
    color: "text-emerald-600 bg-emerald-50 border-emerald-100",
  },
  running: {
    label: "Running",
    icon: TbPlayerPlay,
    color: "text-teal-600 bg-teal-50 border-teal-100",
  },
  scheduled: {
    label: "Scheduled",
    icon: TbClock,
    color: "text-slate-500 bg-slate-50 border-slate-100",
  },
  alert: {
    label: "Alert",
    icon: TbAlertTriangle,
    color: "text-amber-600 bg-amber-50 border-amber-100",
  },
};

export function RecentBatches() {
  const [batches, setBatches] = useState<BatchRecord[]>(mockBatches.slice(0, 6));

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchRecentBatches(6);
        if (!cancelled && data && data.length > 0) {
          // Map DashboardBatch → BatchRecord shape
          const mapped: BatchRecord[] = data.map((b: DashboardBatch) => ({
            id: b.id,
            timestamp: b.timestamp,
            temperature: b.temperature,
            conveyorSpeed: b.conveyorSpeed,
            holdTime: b.holdTime,
            batchSize: b.batchSize,
            materialType: b.materialType,
            hourOfDay: b.hourOfDay,
            qualityScore: b.qualityScore,
            yieldPct: b.yieldPct,
            performancePct: b.performancePct,
            energyKwh: b.energyKwh,
            status: b.status as BatchRecord["status"],
            anomalyScore: b.anomalyScore,
          }));
          setBatches(mapped);
        }
      } catch (err) {
        console.error("[RecentBatches] API unavailable, using mock:", err);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const displayBatches = batches;

  return (
    <div className="bg-white rounded-xl border border-slate-100 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-slate-100 flex items-center justify-center">
            <TbListDetails className="w-[18px] h-[18px] text-slate-500" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Recent Batches</h3>
            <p className="text-[11px] text-slate-400">
              Last {displayBatches.length} production runs
            </p>
          </div>
        </div>
        <Link
          href="/history"
          className="text-[11px] font-bold text-teal-600 hover:text-teal-700 flex items-center gap-1 transition-colors"
        >
          View All <TbChevronRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="bg-slate-50/50 border-b border-slate-100">
              <th className="p-3 pl-5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Batch ID
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Status
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Quality
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Yield
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Energy
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Anomaly
              </th>
              <th className="p-3 pr-5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {displayBatches.map((batch) => {
              const status = statusConfig[batch.status];
              const StatusIcon = status.icon;
              const anomalyLevel =
                batch.anomalyScore > 0.5
                  ? "danger"
                  : batch.anomalyScore > 0.2
                  ? "warning"
                  : "active";
              return (
                <tr
                  key={batch.id}
                  className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors group"
                >
                  <td className="p-3 pl-5">
                    <span className="text-xs font-bold text-slate-700 font-mono">
                      {batch.id}
                    </span>
                  </td>
                  <td className="p-3">
                    <span
                      className={cn(
                        "inline-flex items-center gap-1.5 text-[10px] font-bold px-2 py-1 rounded-md border",
                        status.color
                      )}
                    >
                      <StatusIcon className="w-3 h-3" />
                      {status.label}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-xs font-bold text-slate-700">
                      {batch.qualityScore > 0
                        ? `${batch.qualityScore.toFixed(1)}%`
                        : "—"}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-xs font-bold text-slate-700">
                      {batch.yieldPct > 0
                        ? `${batch.yieldPct.toFixed(1)}%`
                        : "—"}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-xs font-bold text-slate-700">
                      {batch.energyKwh > 0
                        ? `${batch.energyKwh.toFixed(1)} kWh`
                        : "—"}
                    </span>
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span className={`status-dot ${anomalyLevel}`} />
                      <span className="text-[11px] font-mono text-slate-500">
                        {batch.anomalyScore > 0
                          ? batch.anomalyScore.toFixed(2)
                          : "—"}
                      </span>
                    </div>
                  </td>
                  <td className="p-3 pr-5">
                    <span className="text-[11px] text-slate-400">
                      {batch.timestamp.split(" ")[1]}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
