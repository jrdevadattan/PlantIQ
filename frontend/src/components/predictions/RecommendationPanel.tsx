"use client";

import { useState, useEffect } from "react";
import {
  TbWand,
  TbChevronDown,
  TbChevronUp,
  TbArrowDown,
  TbArrowUp,
  TbMinus,
  TbShieldCheck,
  TbClock,
  TbBolt,
  TbLoader,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import {
  generateRecommendations,
  type BatchPredictionParams,
  type RecommendationItem,
  type RecommendationResponse,
} from "@/lib/api";
import type { IconType } from "react-icons";

/* ── Types ─────────────────────────────────────────────── */

interface RecommendationPanelProps {
  batchId: string | null;
  params: BatchPredictionParams | null;
  shapContributions: Array<{ feature: string; contribution: number; direction: string }>;
  target?: string;
}

const directionIcons: Record<string, IconType> = {
  decrease: TbArrowDown,
  increase: TbArrowUp,
  maintain: TbMinus,
};

const directionColors: Record<string, string> = {
  decrease: "text-blue-600 bg-blue-50",
  increase: "text-amber-600 bg-amber-50",
  maintain: "text-slate-500 bg-slate-50",
};

/* ── Component ─────────────────────────────────────────── */

export function RecommendationPanel({
  batchId,
  params,
  shapContributions,
  target = "energy_kwh",
}: RecommendationPanelProps) {
  const [data, setData] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedRank, setExpandedRank] = useState<number | null>(null);

  useEffect(() => {
    if (!params || !shapContributions || shapContributions.length === 0) {
      setData(null);
      return;
    }

    const fetchRecs = async () => {
      setLoading(true);
      try {
        const result = await generateRecommendations(
          batchId ?? "BATCH_UNKNOWN",
          params,
          shapContributions,
          target,
        );
        setData(result);
      } catch (err) {
        console.error("[RecommendationPanel]", err);
      } finally {
        setLoading(false);
      }
    };

    fetchRecs();
  }, [batchId, params, shapContributions, target]);

  if (!params || (!data && !loading)) return null;

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-teal-50 flex items-center justify-center">
            <TbWand className="w-4 h-4 text-teal-600" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-700">Operator Recommendations</h2>
            <p className="text-[10px] text-slate-400">
              Machine-level instructions from SHAP analysis
            </p>
          </div>
        </div>
        {data && (
          <div className="flex items-center gap-1 px-2 py-1 bg-emerald-50 rounded-lg">
            <TbBolt className="w-3 h-3 text-emerald-600" />
            <span className="text-[10px] font-bold text-emerald-700">
              Save ~{data.total_estimated_saving_kwh} kWh
            </span>
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-8 gap-2 text-slate-400">
          <TbLoader className="w-5 h-5 animate-spin" />
          <span className="text-xs">Generating recommendations...</span>
        </div>
      )}

      {/* Summary */}
      {data && !loading && (
        <>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 mb-3">
            <p className="text-[11px] text-slate-600 leading-relaxed">{data.summary}</p>
          </div>

          {/* Recommendation cards */}
          <div className="space-y-2">
            {data.recommendations.map((rec: RecommendationItem) => {
              const DirIcon = directionIcons[rec.direction] ?? TbMinus;
              const dirColor = directionColors[rec.direction] ?? directionColors.maintain;
              const isExpanded = expandedRank === rec.rank;

              return (
                <div
                  key={rec.rank}
                  className="rounded-lg border border-slate-100 overflow-hidden transition-all"
                >
                  {/* Compact row */}
                  <button
                    onClick={() => setExpandedRank(isExpanded ? null : rec.rank)}
                    className="w-full flex items-center gap-3 p-3 hover:bg-slate-50/50 transition-all text-left"
                  >
                    <span className="w-6 h-6 rounded-full bg-teal-600 text-white text-[10px] font-bold flex items-center justify-center flex-shrink-0">
                      {rec.rank}
                    </span>
                    <div className={cn("w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0", dirColor)}>
                      <DirIcon className="w-3.5 h-3.5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-bold text-slate-700 capitalize">
                        {rec.direction} {rec.parameter.replace(/_/g, " ")}
                      </p>
                      <p className="text-[10px] text-slate-400 truncate">
                        {rec.current_value} → {rec.recommended_value} · {rec.machine}
                      </p>
                    </div>
                    {isExpanded ? (
                      <TbChevronUp className="w-4 h-4 text-slate-300" />
                    ) : (
                      <TbChevronDown className="w-4 h-4 text-slate-300" />
                    )}
                  </button>

                  {/* Expanded detail */}
                  {isExpanded && (
                    <div className="px-3 pb-3 pt-0 border-t border-slate-100">
                      <div className="p-3 bg-teal-50/50 rounded-lg mt-2">
                        <p className="text-[11px] text-slate-700 leading-relaxed">
                          {rec.instruction}
                        </p>
                      </div>
                      <div className="grid grid-cols-3 gap-2 mt-2">
                        <div className="text-center p-2 bg-slate-50 rounded-md">
                          <TbBolt className="w-3 h-3 text-amber-500 mx-auto mb-0.5" />
                          <p className="text-[10px] font-bold text-slate-700">{rec.estimated_energy_saving_kwh} kWh</p>
                          <p className="text-[9px] text-slate-400">Energy saved</p>
                        </div>
                        <div className="text-center p-2 bg-slate-50 rounded-md">
                          <TbShieldCheck className="w-3 h-3 text-emerald-500 mx-auto mb-0.5" />
                          <p className="text-[10px] font-bold text-slate-700">{rec.estimated_quality_impact_pct > 0 ? "+" : ""}{rec.estimated_quality_impact_pct}%</p>
                          <p className="text-[9px] text-slate-400">Quality</p>
                        </div>
                        <div className="text-center p-2 bg-slate-50 rounded-md">
                          <TbClock className="w-3 h-3 text-blue-500 mx-auto mb-0.5" />
                          <p className="text-[10px] font-bold text-slate-700">{rec.response_time_min} min</p>
                          <p className="text-[9px] text-slate-400">Response</p>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-2 italic">{rec.safety_note}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
