"use client";

import { useState, useEffect, useCallback } from "react";
import {
  TbAlertTriangle,
  TbBell,
  TbCheck,
  TbClock,
  TbChevronRight,
  TbBolt,
  TbChartBar,
  TbBug,
  TbRefresh,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { fetchAlerts } from "@/lib/api";
import type { AlertRecord } from "@/lib/api";
import type { IconType } from "react-icons";

/* ── Types ─────────────────────────────────────────────── */

interface AlertPanelProps {
  /** Compact mode shows fewer items (for dashboard sidebar) */
  compact?: boolean;
  /** Maximum alerts to display */
  maxItems?: number;
}

const severityConfig: Record<string, { dot: string; bg: string; text: string; border: string; label: string }> = {
  CRITICAL: { dot: "bg-red-500", bg: "bg-red-50", text: "text-red-700", border: "border-red-100", label: "Critical" },
  WARNING: { dot: "bg-amber-500", bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-100", label: "Warning" },
  WATCH: { dot: "bg-blue-400", bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-100", label: "Watch" },
};

const typeIcons: Record<string, IconType> = {
  energy_overrun: TbBolt,
  quality_risk: TbChartBar,
  anomaly: TbBug,
  drift: TbRefresh,
};

/* ── Component ─────────────────────────────────────────── */

export function AlertPanel({ compact = false, maxItems = 10 }: AlertPanelProps) {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const loadAlerts = useCallback(async () => {
    try {
      const data = await fetchAlerts(50);
      setAlerts(Array.isArray(data) ? data : []);
    } catch {
      // Backend may not be running — show empty state
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 30000);
    return () => clearInterval(interval);
  }, [loadAlerts]);

  const displayAlerts = alerts.slice(0, maxItems);
  const critCount = alerts.filter(a => a.severity === "CRITICAL").length;
  const warnCount = alerts.filter(a => a.severity === "WARNING").length;

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-amber-50 flex items-center justify-center">
            <TbBell className="w-4 h-4 text-amber-600" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-700">Active Alerts</h2>
            {!compact && (
              <p className="text-[10px] text-slate-400">
                {critCount > 0 && <span className="text-red-500 font-bold">{critCount} critical</span>}
                {critCount > 0 && warnCount > 0 && " · "}
                {warnCount > 0 && <span className="text-amber-500 font-bold">{warnCount} warning</span>}
                {critCount === 0 && warnCount === 0 && "All systems normal"}
              </p>
            )}
          </div>
        </div>
        {alerts.length > 0 && (
          <span className={cn(
            "px-2 py-0.5 rounded-full text-[10px] font-bold",
            critCount > 0 ? "bg-red-50 text-red-600" : "bg-amber-50 text-amber-600"
          )}>
            {alerts.length}
          </span>
        )}
      </div>

      {/* Loading state */}
      {loading && (
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 bg-slate-50 rounded-lg animate-pulse" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && alerts.length === 0 && (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 rounded-full bg-emerald-50 flex items-center justify-center mb-3">
            <TbCheck className="w-6 h-6 text-emerald-500" />
          </div>
          <p className="text-sm font-semibold text-slate-600">All Clear</p>
          <p className="text-[11px] text-slate-400 mt-1">No active alerts at this time</p>
        </div>
      )}

      {/* Alert list */}
      {!loading && displayAlerts.length > 0 && (
        <div className="space-y-2">
          {displayAlerts.map((alert) => {
            const sev = severityConfig[alert.severity] ?? severityConfig.WATCH;
            const TypeIcon = typeIcons[alert.alert_type] ?? TbAlertTriangle;
            const isExpanded = expandedId === alert.id;
            const timeAgo = getTimeAgo(alert.timestamp);

            return (
              <div
                key={alert.id}
                className={cn(
                  "rounded-lg border p-3 transition-all cursor-pointer",
                  sev.bg, sev.border,
                  isExpanded && "ring-1 ring-offset-1",
                  isExpanded && alert.severity === "CRITICAL" && "ring-red-200",
                  isExpanded && alert.severity === "WARNING" && "ring-amber-200",
                )}
                onClick={() => setExpandedId(isExpanded ? null : alert.id)}
              >
                <div className="flex items-start gap-2">
                  <div className={cn("w-2 h-2 rounded-full mt-1.5 flex-shrink-0", sev.dot)} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <TypeIcon className={cn("w-3.5 h-3.5 flex-shrink-0", sev.text)} />
                      <span className={cn("text-[11px] font-bold uppercase tracking-wide", sev.text)}>
                        {sev.label}
                      </span>
                      <span className="text-[10px] text-slate-400 ml-auto flex items-center gap-1">
                        <TbClock className="w-3 h-3" />
                        {timeAgo}
                      </span>
                    </div>
                    <p className={cn(
                      "text-xs mt-1 leading-relaxed",
                      compact ? "line-clamp-1" : "line-clamp-2",
                      "text-slate-700"
                    )}>
                      {alert.message}
                    </p>

                    {/* Expanded details */}
                    {isExpanded && !compact && (
                      <div className="mt-3 pt-3 border-t border-slate-200/50 space-y-2">
                        <div className="text-[11px] text-slate-500">
                          <span className="font-semibold">Batch:</span> {alert.batch_id}
                        </div>
                        {alert.root_cause && (
                          <div className="text-[11px] text-slate-500">
                            <span className="font-semibold">Root Cause:</span> {alert.root_cause}
                          </div>
                        )}
                        {alert.recommended_action && (
                          <div className="p-2 bg-white/60 rounded-md border border-slate-100">
                            <span className="text-[10px] font-bold text-teal-600 uppercase tracking-wide">
                              Recommended Action
                            </span>
                            <p className="text-[11px] text-slate-700 mt-0.5">{alert.recommended_action}</p>
                          </div>
                        )}
                        {alert.estimated_saving_kwh !== null && (
                          <div className="text-[10px] text-slate-400">
                            Estimated saving: {alert.estimated_saving_kwh} kWh
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <TbChevronRight className={cn(
                    "w-3.5 h-3.5 text-slate-300 flex-shrink-0 transition-transform mt-0.5",
                    isExpanded && "rotate-90"
                  )} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Helper ────────────────────────────────────────────── */

function getTimeAgo(timestamp: string): string {
  try {
    const diff = Date.now() - new Date(timestamp).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  } catch {
    return "—";
  }
}
