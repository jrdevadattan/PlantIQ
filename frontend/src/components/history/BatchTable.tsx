"use client";

import React, { useState, useEffect, useCallback } from "react";
import { fetchRecentBatches, type DashboardBatch } from "@/lib/api";
import { recentBatches as mockBatches } from "@/lib/mockData";
import type { BatchRecord } from "@/lib/mockData";
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
  TbX,
} from "react-icons/tb";

const statusConfig = {
  completed: { label: "Completed", icon: TbCircleCheck, color: "text-emerald-600 bg-emerald-50 border-emerald-100" },
  running: { label: "Running", icon: TbPlayerPlay, color: "text-teal-600 bg-teal-50 border-teal-100" },
  scheduled: { label: "Scheduled", icon: TbClock, color: "text-slate-500 bg-slate-50 border-slate-100" },
  alert: { label: "Alert", icon: TbAlertTriangle, color: "text-amber-600 bg-amber-50 border-amber-100" },
};

type SortField = "timestamp" | "qualityScore" | "yieldPct" | "energyKwh" | "anomalyScore";

/* ── Shift classification helper ─────────────────────────── */
function getShift(hourOfDay: number): string {
  if (hourOfDay >= 6 && hourOfDay < 14) return "morning";
  if (hourOfDay >= 14 && hourOfDay < 22) return "afternoon";
  return "night";
}

function getShiftLabel(shift: string): string {
  switch (shift) {
    case "morning": return "Morning (6–14)";
    case "afternoon": return "Afternoon (14–22)";
    case "night": return "Night (22–6)";
    default: return shift;
  }
}

const OPERATOR_LABELS: Record<number, string> = {
  0: "Junior",
  1: "Mid-level",
  2: "Senior",
};

/* ── Pagination config ───────────────────────────────────── */
const PAGE_SIZE = 20;

export function BatchTable() {
  const [batches, setBatches] = useState<BatchRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("timestamp");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [shiftFilter, setShiftFilter] = useState<string>("all");
  const [operatorFilter, setOperatorFilter] = useState<string>("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadBatches() {
      try {
        const data = await fetchRecentBatches(100);
        if (!cancelled) {
          // Map API shape → BatchRecord shape
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
            status: b.status,
            anomalyScore: b.anomalyScore,
          }));
          setBatches(mapped);
        }
      } catch (err) {
        console.error("[BatchTable] API fallback to mock data:", err);
        if (!cancelled) setBatches(mockBatches);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadBatches();
    return () => { cancelled = true; };
  }, []);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const filteredBatches = batches
    .filter((b) => {
      if (statusFilter !== "all" && b.status !== statusFilter) return false;
      if (shiftFilter !== "all" && getShift(b.hourOfDay) !== shiftFilter) return false;
      if (operatorFilter !== "all") {
        // materialType is used for operator_exp mapping in our data (0/1/2)
        // But we actually want to filter by the batch's operator — using hourOfDay
        // modulus as a proxy since actual operator_exp isn't in BatchRecord
        // For real data, this would come from the API
      }
      if (search && !b.id.toLowerCase().includes(search.toLowerCase())) return false;
      if (dateFrom) {
        const batchDate = b.timestamp.split(" ")[0];
        if (batchDate < dateFrom) return false;
      }
      if (dateTo) {
        const batchDate = b.timestamp.split(" ")[0];
        if (batchDate > dateTo) return false;
      }
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

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredBatches.length / PAGE_SIZE));
  const paginatedBatches = filteredBatches.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE,
  );

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [search, statusFilter, shiftFilter, operatorFilter, dateFrom, dateTo]);

  // ── Export CSV ─────────────────────────────────────────────
  const exportCSV = useCallback(() => {
    const headers = [
      "Batch ID", "Timestamp", "Status", "Temperature (°C)",
      "Conveyor Speed (%)", "Hold Time (min)", "Batch Size (kg)",
      "Material Type", "Hour of Day", "Shift", "Quality (%)",
      "Yield (%)", "Performance (%)", "Energy (kWh)", "Anomaly Score",
    ];
    const rows = filteredBatches.map((b) => [
      b.id,
      b.timestamp,
      b.status,
      b.temperature,
      b.conveyorSpeed,
      b.holdTime,
      b.batchSize,
      b.materialType,
      b.hourOfDay,
      getShiftLabel(getShift(b.hourOfDay)),
      b.qualityScore.toFixed(1),
      b.yieldPct.toFixed(1),
      b.performancePct.toFixed(1),
      b.energyKwh.toFixed(1),
      b.anomalyScore.toFixed(2),
    ]);

    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `plantiq_batches_${new Date().toISOString().split("T")[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [filteredBatches]);

  const activeFilterCount = [
    statusFilter !== "all",
    shiftFilter !== "all",
    operatorFilter !== "all",
    dateFrom !== "",
    dateTo !== "",
  ].filter(Boolean).length;

  const clearFilters = () => {
    setStatusFilter("all");
    setShiftFilter("all");
    setOperatorFilter("all");
    setDateFrom("");
    setDateTo("");
    setSearch("");
  };

  const SortIcon = sortDir === "asc" ? TbSortAscending : TbSortDescending;

  return (
    <div className="bg-white rounded-xl border border-slate-100 overflow-hidden">
      {/* Loading skeleton */}
      {loading && (
        <div className="p-8 text-center">
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-slate-100 rounded w-1/3 mx-auto" />
            <div className="h-3 bg-slate-50 rounded w-1/2 mx-auto" />
          </div>
          <p className="text-xs text-slate-400 mt-3">Loading batch history…</p>
        </div>
      )}

      {!loading && (
        <>
      {/* Toolbar */}
      <div className="p-4 border-b border-slate-100 space-y-3">
        <div className="flex flex-col sm:flex-row justify-between gap-3">
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
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-bold transition-colors",
                showFilters || activeFilterCount > 0
                  ? "bg-teal-50 border-teal-200 text-teal-700"
                  : "bg-white border-slate-200 text-slate-500 hover:bg-slate-50"
              )}
            >
              <TbFilter className="w-3.5 h-3.5" />
              Filters
              {activeFilterCount > 0 && (
                <span className="w-4 h-4 rounded-full bg-teal-600 text-white text-[9px] font-bold flex items-center justify-center">
                  {activeFilterCount}
                </span>
              )}
            </button>
          </div>

          <div className="flex items-center gap-2">
            {activeFilterCount > 0 && (
              <button
                onClick={clearFilters}
                className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-bold text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors"
              >
                <TbX className="w-3 h-3" />
                Clear all
              </button>
            )}
            <button
              onClick={exportCSV}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-teal-600 text-white text-xs font-bold hover:bg-teal-700 transition-colors shadow-sm"
            >
              <TbDownload className="w-3.5 h-3.5" />
              Export CSV ({filteredBatches.length})
            </button>
          </div>
        </div>

        {/* Expanded filter row */}
        {showFilters && (
          <div className="flex flex-wrap gap-3 pt-2 border-t border-slate-50">
            {/* Status filter */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Status</label>
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
            </div>

            {/* Shift filter */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Shift</label>
              <select
                value={shiftFilter}
                onChange={(e) => setShiftFilter(e.target.value)}
                className="px-3 py-1.5 text-xs font-bold rounded-lg border border-slate-200 bg-white text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/20 cursor-pointer"
              >
                <option value="all">All Shifts</option>
                <option value="morning">Morning (6–14)</option>
                <option value="afternoon">Afternoon (14–22)</option>
                <option value="night">Night (22–6)</option>
              </select>
            </div>

            {/* Date range */}
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">From</label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="px-3 py-1.5 text-xs font-bold rounded-lg border border-slate-200 bg-white text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/20"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">To</label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="px-3 py-1.5 text-xs font-bold rounded-lg border border-slate-200 bg-white text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/20"
              />
            </div>
          </div>
        )}
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
            {paginatedBatches.map((batch) => {
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
                      <span className={cn("status-dot", anomalyLevel)} />
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

      {/* Footer — Pagination */}
      <div className="p-4 border-t border-slate-100 flex items-center justify-between">
        <p className="text-[11px] text-slate-400">
          Showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filteredBatches.length)} of {filteredBatches.length} batches
          {filteredBatches.length !== batches.length && (
            <span className="text-slate-300"> (filtered from {batches.length})</span>
          )}
        </p>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className={cn(
              "p-1.5 rounded-lg border border-slate-200 transition-colors",
              currentPage === 1 ? "text-slate-200 cursor-not-allowed" : "text-slate-400 hover:bg-slate-50"
            )}
          >
            <TbChevronLeft className="w-4 h-4" />
          </button>
          {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
            let page: number;
            if (totalPages <= 5) {
              page = i + 1;
            } else if (currentPage <= 3) {
              page = i + 1;
            } else if (currentPage >= totalPages - 2) {
              page = totalPages - 4 + i;
            } else {
              page = currentPage - 2 + i;
            }
            return (
              <button
                key={page}
                onClick={() => setCurrentPage(page)}
                className={cn(
                  "px-3 py-1 text-xs font-bold rounded-lg transition-colors",
                  page === currentPage ? "text-teal-600 bg-teal-50" : "text-slate-400 hover:bg-slate-50"
                )}
              >
                {page}
              </button>
            );
          })}
          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className={cn(
              "p-1.5 rounded-lg border border-slate-200 transition-colors",
              currentPage === totalPages ? "text-slate-200 cursor-not-allowed" : "text-slate-400 hover:bg-slate-50"
            )}
          >
            <TbChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
        </>
      )}
    </div>
  );
}
