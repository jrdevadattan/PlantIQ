"use client";

import React, { useState } from "react";
import { recentBatches } from "@/lib/mockData";
import { cn } from "@/lib/utils";
import {
  TbCircleCheck,
  TbPlayerPlay,
  TbClock,
  TbAlertTriangle,
  TbSearch,
  TbFilter,
  TbDownload,
  TbSortAscending,
  TbSortDescending,
  TbChevronLeft,
  TbChevronRight,
  TbEye,
} from "react-icons/tb";

const statusConfig = {
  completed: { label: "Completed", icon: TbCircleCheck, color: "text-emerald-600 bg-emerald-50 border-emerald-100" },
  running: { label: "Running", icon: TbPlayerPlay, color: "text-teal-600 bg-teal-50 border-teal-100" },
  scheduled: { label: "Scheduled", icon: TbClock, color: "text-slate-500 bg-slate-50 border-slate-100" },
  alert: { label: "Alert", icon: TbAlertTriangle, color: "text-amber-600 bg-amber-50 border-amber-100" },
};

type SortField = "timestamp" | "qualityScore" | "yieldPct" | "energyKwh" | "anomalyScore";

export function BatchTable() {
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("timestamp");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const filteredBatches = recentBatches
    .filter((b) => {
      if (statusFilter !== "all" && b.status !== statusFilter) return false;
      if (search && !b.id.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      const aVal = a[sortField] as number;
      const bVal = b[sortField] as number;
      if (sortField === "timestamp") {
        return sortDir === "asc"
          ? a.timestamp.localeCompare(b.timestamp)
          : b.timestamp.localeCompare(a.timestamp);
      }
      return sortDir === "asc" ? aVal - bVal : bVal - aVal;
    });

  const SortIcon = sortDir === "asc" ? TbSortAscending : TbSortDescending;

  return (
    <div className="bg-white rounded-xl border border-slate-100 overflow-hidden">
      {/* Toolbar */}
      <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className="relative">
            <TbSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search batch ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 pr-3 py-1.5 text-xs font-bold rounded-lg border border-slate-200 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-teal-500/20 w-52 text-slate-700"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-1.5 text-xs font-bold rounded-lg border border-slate-200 bg-white text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/20 cursor-pointer"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="scheduled">Scheduled</option>
            <option value="alert">Alert</option>
          </select>

          <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-teal-600 text-white text-xs font-bold hover:bg-teal-700 transition-colors shadow-sm">
            <TbDownload className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-50/50 border-b border-slate-100">
              <th className="p-3 pl-5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Batch ID
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Date / Time
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Status
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <button onClick={() => toggleSort("qualityScore")} className="flex items-center gap-1 hover:text-slate-600">
                  Quality {sortField === "qualityScore" && <SortIcon className="w-3 h-3" />}
                </button>
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <button onClick={() => toggleSort("yieldPct")} className="flex items-center gap-1 hover:text-slate-600">
                  Yield {sortField === "yieldPct" && <SortIcon className="w-3 h-3" />}
                </button>
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Performance
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <button onClick={() => toggleSort("energyKwh")} className="flex items-center gap-1 hover:text-slate-600">
                  Energy {sortField === "energyKwh" && <SortIcon className="w-3 h-3" />}
                </button>
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <button onClick={() => toggleSort("anomalyScore")} className="flex items-center gap-1 hover:text-slate-600">
                  Anomaly {sortField === "anomalyScore" && <SortIcon className="w-3 h-3" />}
                </button>
              </th>
              <th className="p-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Params
              </th>
              <th className="p-3 pr-5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredBatches.map((batch) => {
              const status = statusConfig[batch.status];
              const StatusIcon = status.icon;
              const anomalyLevel = batch.anomalyScore > 0.5 ? "danger" : batch.anomalyScore > 0.2 ? "warning" : "active";
              const isNoData = batch.qualityScore === 0;

              return (
                <tr
                  key={batch.id}
                  className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors group"
                >
                  <td className="p-3 pl-5">
                    <span className="text-xs font-bold text-slate-700 font-mono">{batch.id}</span>
                  </td>
                  <td className="p-3">
                    <div>
                      <span className="text-xs font-medium text-slate-600 block">
                        {batch.timestamp.split(" ")[0]}
                      </span>
                      <span className="text-[10px] text-slate-400">
                        {batch.timestamp.split(" ")[1]}
                      </span>
                    </div>
                  </td>
                  <td className="p-3">
                    <span className={cn("inline-flex items-center gap-1.5 text-[10px] font-bold px-2 py-1 rounded-md border", status.color)}>
                      <StatusIcon className="w-3 h-3" />
                      {status.label}
                    </span>
                  </td>
                  <td className="p-3 text-xs font-bold text-slate-700">
                    {isNoData ? "—" : `${batch.qualityScore.toFixed(1)}%`}
                  </td>
                  <td className="p-3 text-xs font-bold text-slate-700">
                    {isNoData ? "—" : `${batch.yieldPct.toFixed(1)}%`}
                  </td>
                  <td className="p-3 text-xs font-bold text-slate-700">
                    {isNoData ? "—" : `${batch.performancePct.toFixed(1)}%`}
                  </td>
                  <td className="p-3 text-xs font-bold text-slate-700">
                    {isNoData ? "—" : `${batch.energyKwh.toFixed(1)}`}
                    {!isNoData && <span className="text-[10px] text-slate-400 ml-0.5">kWh</span>}
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span className={`status-dot ${anomalyLevel}`} />
                      <span className="text-[11px] font-mono text-slate-500">
                        {batch.anomalyScore > 0 ? batch.anomalyScore.toFixed(2) : "—"}
                      </span>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="text-[10px] text-slate-400 leading-tight">
                      <span>{batch.temperature}C</span>
                      <span className="mx-1">|</span>
                      <span>{batch.conveyorSpeed}%</span>
                      <span className="mx-1">|</span>
                      <span>{batch.holdTime}m</span>
                    </div>
                  </td>
                  <td className="p-3 pr-5">
                    <button className="p-1.5 text-slate-400 hover:text-teal-600 hover:bg-teal-50 rounded-lg transition-colors">
                      <TbEye className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-100 flex items-center justify-between">
        <p className="text-[11px] text-slate-400">
          Showing {filteredBatches.length} of {recentBatches.length} batches
        </p>
        <div className="flex items-center gap-1">
          <button className="p-1.5 rounded-lg border border-slate-200 text-slate-400 hover:bg-slate-50">
            <TbChevronLeft className="w-4 h-4" />
          </button>
          <span className="px-3 py-1 text-xs font-bold text-teal-600 bg-teal-50 rounded-lg">1</span>
          <button className="p-1.5 rounded-lg border border-slate-200 text-slate-400 hover:bg-slate-50">
            <TbChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
