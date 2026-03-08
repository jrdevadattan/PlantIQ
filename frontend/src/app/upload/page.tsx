"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  TbUpload,
  TbFileSpreadsheet,
  TbCheck,
  TbAlertTriangle,
  TbX,
  TbChevronDown,
  TbChevronUp,
  TbTarget,
  TbFlask,
  TbShieldCheck,
  TbArrowRight,
} from "react-icons/tb";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────

interface QualityCompliance {
  value: number;
  in_spec: boolean;
  spec_range: string;
  optimal: number;
  deviation_pct: number;
}

interface GoldenComparison {
  golden_batch: string;
  alignment_score: number;
  parameter_deviations: Record<string, {
    your_value: number;
    golden_value: number;
    deviation_pct: number;
    in_range: boolean;
  }>;
}

interface BatchPrediction {
  batch_id: string;
  inputs: Record<string, number>;
  predicted_targets: Record<string, number>;
  quality_compliance: Record<string, QualityCompliance>;
  golden_comparison: GoldenComparison | null;
}

interface PredSummaryItem {
  min: number;
  max: number;
  mean: number;
  in_spec_count: number;
  total: number;
  spec: string;
}

interface UploadResponse {
  status: string;
  filename: string;
  rows_uploaded: number;
  columns_detected: string[];
  input_summary: Record<string, { min: number; max: number; mean: number }>;
  prediction_summary: Record<string, PredSummaryItem>;
  predictions: BatchPrediction[];
  warnings: string[];
}

// ── Friendly labels ───────────────────────────────────────

const TARGET_LABELS: Record<string, string> = {
  Moisture_Content: "Moisture Content",
  Tablet_Weight: "Tablet Weight",
  Hardness: "Hardness",
  Friability: "Friability",
  Disintegration_Time: "Disintegration Time",
  Dissolution_Rate: "Dissolution Rate",
  Content_Uniformity: "Content Uniformity",
};

const TARGET_UNITS: Record<string, string> = {
  Moisture_Content: "%",
  Tablet_Weight: "mg",
  Hardness: "N",
  Friability: "%",
  Disintegration_Time: "min",
  Dissolution_Rate: "%",
  Content_Uniformity: "%",
};

const INPUT_LABELS: Record<string, string> = {
  Granulation_Time: "Granulation Time",
  Binder_Amount: "Binder Amount",
  Drying_Temp: "Drying Temp",
  Drying_Time: "Drying Time",
  Compression_Force: "Compression Force",
  Machine_Speed: "Machine Speed",
  Lubricant_Conc: "Lubricant Conc",
};

// ── Upload Page Component ─────────────────────────────────

export default function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedBatch, setExpandedBatch] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        const msg = errData?.detail?.error || errData?.detail || `Upload failed (${res.status})`;
        throw new Error(typeof msg === "object" ? JSON.stringify(msg) : msg);
      }

      const data: UploadResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const onFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  // Calculate overall spec compliance from predictions
  const overallCompliance = result?.predictions
    ? (() => {
        let totalIn = 0;
        let totalChecks = 0;
        result.predictions.forEach((p) => {
          Object.values(p.quality_compliance).forEach((c) => {
            totalChecks++;
            if (c.in_spec) totalIn++;
          });
        });
        return totalChecks > 0 ? Math.round((totalIn / totalChecks) * 100) : 0;
      })()
    : 0;

  const avgAlignment = result?.predictions
    ? (() => {
        const scores = result.predictions
          .map((p) => p.golden_comparison?.alignment_score)
          .filter((s): s is number => s != null);
        return scores.length > 0
          ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
          : 0;
      })()
    : 0;

  return (
    <div className="max-w-[1600px] mx-auto space-y-5">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-slate-800">
            Upload Batch Data
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Upload a CSV or Excel file with pharmaceutical batch parameters to
            get quality predictions
          </p>
        </div>
      </div>

      {/* Upload zone */}
      <div
        className={cn(
          "bg-white rounded-xl border-2 border-dashed p-8 transition-all text-center",
          isDragging
            ? "border-teal-400 bg-teal-50/50"
            : "border-slate-200 hover:border-slate-300",
          isUploading && "opacity-60 pointer-events-none"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={onFileSelect}
        />

        <div className="flex flex-col items-center gap-3">
          {isUploading ? (
            <>
              <div className="w-12 h-12 rounded-full bg-teal-50 flex items-center justify-center animate-pulse">
                <TbFileSpreadsheet className="w-6 h-6 text-teal-600" />
              </div>
              <p className="text-sm font-semibold text-slate-600">
                Processing...
              </p>
              <p className="text-xs text-slate-400">
                Running feature engineering &amp; predictions
              </p>
            </>
          ) : (
            <>
              <div
                className={cn(
                  "w-12 h-12 rounded-full flex items-center justify-center transition-colors",
                  isDragging ? "bg-teal-100" : "bg-slate-100"
                )}
              >
                <TbUpload
                  className={cn(
                    "w-6 h-6",
                    isDragging ? "text-teal-600" : "text-slate-400"
                  )}
                />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">
                  {isDragging
                    ? "Drop file here"
                    : "Drag & drop your batch data file"}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Accepts .csv, .xlsx, or .xls files
                </p>
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 px-4 py-2 bg-teal-600 text-white text-xs font-bold rounded-lg hover:bg-teal-700 transition-colors"
              >
                Browse Files
              </button>
              <p className="text-[10px] text-slate-400 mt-2 max-w-md">
                Required columns: Granulation_Time, Binder_Amount, Drying_Temp,
                Drying_Time, Compression_Force, Machine_Speed, Lubricant_Conc
              </p>
            </>
          )}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
          <TbAlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-700">Upload Failed</p>
            <p className="text-xs text-red-600 mt-1">{error}</p>
          </div>
          <button type="button" onClick={() => setError(null)} className="ml-auto">
            <TbX className="w-4 h-4 text-red-400" />
          </button>
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-5"
          >
            {/* Warnings */}
            {result.warnings.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                <p className="text-xs font-bold text-amber-700 mb-2">
                  ⚠️ Warnings
                </p>
                {result.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-600">
                    {w}
                  </p>
                ))}
              </div>
            )}

            {/* Summary KPI cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <SummaryCard
                label="Batches Uploaded"
                value={result.rows_uploaded}
                icon={<TbFileSpreadsheet className="w-5 h-5" />}
                color="teal"
              />
              <SummaryCard
                label="Targets Predicted"
                value={7}
                icon={<TbTarget className="w-5 h-5" />}
                color="indigo"
              />
              <SummaryCard
                label="Spec Compliance"
                value={`${overallCompliance}%`}
                icon={<TbShieldCheck className="w-5 h-5" />}
                color={overallCompliance >= 80 ? "emerald" : overallCompliance >= 50 ? "amber" : "red"}
              />
              <SummaryCard
                label="Golden Alignment"
                value={`${avgAlignment}%`}
                icon={<TbFlask className="w-5 h-5" />}
                color={avgAlignment >= 80 ? "emerald" : avgAlignment >= 50 ? "amber" : "red"}
              />
            </div>

            {/* Prediction summary table */}
            {Object.keys(result.prediction_summary).length > 0 && (
              <div className="bg-white rounded-xl border border-slate-100 p-4">
                <h2 className="text-sm font-bold text-slate-700 mb-3">
                  Prediction Summary (across all batches)
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-100">
                        <th className="text-left py-2 pr-4 font-bold text-slate-500">
                          Quality Target
                        </th>
                        <th className="text-right py-2 px-3 font-bold text-slate-500">
                          Min
                        </th>
                        <th className="text-right py-2 px-3 font-bold text-slate-500">
                          Mean
                        </th>
                        <th className="text-right py-2 px-3 font-bold text-slate-500">
                          Max
                        </th>
                        <th className="text-right py-2 px-3 font-bold text-slate-500">
                          Spec Range
                        </th>
                        <th className="text-right py-2 pl-3 font-bold text-slate-500">
                          In Spec
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.prediction_summary).map(
                        ([target, s]) => {
                          const allInSpec =
                            s.in_spec_count === s.total;
                          return (
                            <tr
                              key={target}
                              className="border-b border-slate-50 last:border-none"
                            >
                              <td className="py-2 pr-4 font-semibold text-slate-700">
                                {TARGET_LABELS[target] ?? target}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-600 font-mono">
                                {s.min.toFixed(2)}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-800 font-mono font-bold">
                                {s.mean.toFixed(2)}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-600 font-mono">
                                {s.max.toFixed(2)}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-500">
                                {s.spec}
                              </td>
                              <td className="py-2 pl-3 text-right">
                                <span
                                  className={cn(
                                    "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold",
                                    allInSpec
                                      ? "bg-emerald-50 text-emerald-600"
                                      : "bg-amber-50 text-amber-600"
                                  )}
                                >
                                  {allInSpec ? (
                                    <TbCheck className="w-3 h-3" />
                                  ) : (
                                    <TbAlertTriangle className="w-3 h-3" />
                                  )}
                                  {s.in_spec_count}/{s.total}
                                </span>
                              </td>
                            </tr>
                          );
                        }
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Per-batch predictions */}
            <div className="bg-white rounded-xl border border-slate-100 p-4">
              <h2 className="text-sm font-bold text-slate-700 mb-3">
                Per-Batch Predictions
              </h2>
              <div className="space-y-2">
                {result.predictions.map((p) => {
                  const isExpanded = expandedBatch === p.batch_id;
                  const complianceVals = Object.values(p.quality_compliance);
                  const inSpec = complianceVals.filter((c) => c.in_spec).length;
                  const total = complianceVals.length;
                  const alignment =
                    p.golden_comparison?.alignment_score ?? null;

                  return (
                    <div
                      key={p.batch_id}
                      className="border border-slate-100 rounded-lg overflow-hidden"
                    >
                      {/* Batch header row */}
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedBatch(
                            isExpanded ? null : p.batch_id
                          )
                        }
                        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors text-left"
                      >
                        <span className="text-xs font-bold text-slate-700 w-20">
                          {p.batch_id}
                        </span>

                        {/* Quick target values */}
                        <div className="flex-1 flex flex-wrap gap-x-4 gap-y-1">
                          {Object.entries(p.predicted_targets)
                            .slice(0, 4)
                            .map(([key, val]) => (
                              <span
                                key={key}
                                className="text-[10px] text-slate-500"
                              >
                                <span className="font-semibold text-slate-600">
                                  {TARGET_LABELS[key]?.split(" ")[0] ?? key}:
                                </span>{" "}
                                {val.toFixed(2)}
                                {TARGET_UNITS[key] ?? ""}
                              </span>
                            ))}
                        </div>

                        {/* Compliance badge */}
                        <span
                          className={cn(
                            "text-[10px] font-bold px-2 py-0.5 rounded-full",
                            inSpec === total
                              ? "bg-emerald-50 text-emerald-600"
                              : inSpec >= total / 2
                              ? "bg-amber-50 text-amber-600"
                              : "bg-red-50 text-red-600"
                          )}
                        >
                          {inSpec}/{total} spec
                        </span>

                        {/* Golden alignment */}
                        {alignment !== null && (
                          <span
                            className={cn(
                              "text-[10px] font-bold px-2 py-0.5 rounded-full",
                              alignment >= 80
                                ? "bg-teal-50 text-teal-600"
                                : alignment >= 50
                                ? "bg-amber-50 text-amber-600"
                                : "bg-red-50 text-red-600"
                            )}
                          >
                            {alignment}% align
                          </span>
                        )}

                        {isExpanded ? (
                          <TbChevronUp className="w-4 h-4 text-slate-400" />
                        ) : (
                          <TbChevronDown className="w-4 h-4 text-slate-400" />
                        )}
                      </button>

                      {/* Expanded details */}
                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="px-4 pb-4 pt-2 border-t border-slate-100 grid grid-cols-1 lg:grid-cols-2 gap-4">
                              {/* Input parameters */}
                              <div>
                                <h3 className="text-[11px] font-bold text-slate-500 uppercase mb-2">
                                  Input Parameters
                                </h3>
                                <div className="space-y-1">
                                  {Object.entries(p.inputs).map(
                                    ([key, val]) => (
                                      <div
                                        key={key}
                                        className="flex justify-between text-xs"
                                      >
                                        <span className="text-slate-500">
                                          {INPUT_LABELS[key] ?? key}
                                        </span>
                                        <span className="font-mono font-semibold text-slate-700">
                                          {val}
                                        </span>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>

                              {/* Quality compliance */}
                              <div>
                                <h3 className="text-[11px] font-bold text-slate-500 uppercase mb-2">
                                  Quality Compliance
                                </h3>
                                <div className="space-y-1">
                                  {Object.entries(p.quality_compliance).map(
                                    ([key, c]) => (
                                      <div
                                        key={key}
                                        className="flex items-center justify-between text-xs"
                                      >
                                        <span className="text-slate-500">
                                          {TARGET_LABELS[key] ?? key}
                                        </span>
                                        <div className="flex items-center gap-2">
                                          <span className="font-mono font-semibold text-slate-700">
                                            {c.value.toFixed(2)}
                                          </span>
                                          <span
                                            className={cn(
                                              "w-1.5 h-1.5 rounded-full",
                                              c.in_spec
                                                ? "bg-emerald-500"
                                                : "bg-red-500"
                                            )}
                                          />
                                          <span className="text-slate-400 text-[10px]">
                                            {c.spec_range}
                                          </span>
                                        </div>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>

                              {/* Golden comparison */}
                              {p.golden_comparison && (
                                <div className="lg:col-span-2">
                                  <h3 className="text-[11px] font-bold text-slate-500 uppercase mb-2">
                                    vs Golden Signature (T001) — Alignment:{" "}
                                    <span
                                      className={cn(
                                        "font-bold",
                                        p.golden_comparison.alignment_score >= 80
                                          ? "text-emerald-600"
                                          : p.golden_comparison.alignment_score >=
                                            50
                                          ? "text-amber-600"
                                          : "text-red-600"
                                      )}
                                    >
                                      {p.golden_comparison.alignment_score}%
                                    </span>
                                  </h3>
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {Object.entries(
                                      p.golden_comparison
                                        .parameter_deviations
                                    )
                                      .slice(0, 8)
                                      .map(([key, d]) => (
                                        <div
                                          key={key}
                                          className={cn(
                                            "rounded-lg p-2 text-xs",
                                            d.in_range
                                              ? "bg-emerald-50"
                                              : "bg-red-50"
                                          )}
                                        >
                                          <p className="text-slate-500 text-[10px] truncate">
                                            {INPUT_LABELS[key] ?? key}
                                          </p>
                                          <div className="flex items-center gap-1 mt-1">
                                            <span className="font-mono font-bold text-slate-700">
                                              {d.your_value.toFixed(1)}
                                            </span>
                                            <TbArrowRight className="w-3 h-3 text-slate-400" />
                                            <span className="font-mono text-slate-500">
                                              {d.golden_value.toFixed(1)}
                                            </span>
                                          </div>
                                          <p
                                            className={cn(
                                              "text-[10px] mt-0.5",
                                              d.deviation_pct < 10
                                                ? "text-emerald-600"
                                                : d.deviation_pct < 25
                                                ? "text-amber-600"
                                                : "text-red-600"
                                            )}
                                          >
                                            {d.deviation_pct.toFixed(1)}% dev
                                          </p>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions panel — shown when no results */}
      {!result && !isUploading && (
        <div className="bg-white rounded-xl border border-slate-100 p-4">
          <h2 className="text-sm font-bold text-slate-700 mb-3">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StepCard
              step={1}
              title="Upload Data"
              description="Upload a CSV or Excel file with your batch parameters (Granulation_Time, Binder_Amount, Drying_Temp, etc.)"
            />
            <StepCard
              step={2}
              title="AI Predicts Quality"
              description="The XGBoost model (98.13% accuracy) predicts 7 quality targets: Hardness, Dissolution Rate, Content Uniformity, etc."
            />
            <StepCard
              step={3}
              title="Compare to Golden"
              description="Each batch is compared against the T001 golden signature — the optimal reference batch for all spec targets."
            />
          </div>

          <div className="mt-4 p-3 bg-slate-50 rounded-lg">
            <p className="text-[11px] font-bold text-slate-500 mb-1">
              Sample file format (CSV):
            </p>
            <pre className="text-[10px] text-slate-500 font-mono overflow-x-auto">
{`Batch_ID,Granulation_Time,Binder_Amount,Drying_Temp,Drying_Time,Compression_Force,Machine_Speed,Lubricant_Conc
T001,15,8.5,60,25,12.5,150,1.0
T002,20,10.2,55,30,15.0,200,1.5
T003,12,7.0,65,22,10.0,180,0.8`}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────

function SummaryCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    teal: "bg-teal-50 text-teal-600 border-teal-100",
    indigo: "bg-indigo-50 text-indigo-600 border-indigo-100",
    emerald: "bg-emerald-50 text-emerald-600 border-emerald-100",
    amber: "bg-amber-50 text-amber-600 border-amber-100",
    red: "bg-red-50 text-red-600 border-red-100",
  };

  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center",
            colorMap[color] ?? colorMap.teal
          )}
        >
          {icon}
        </div>
        <div>
          <p className="text-2xl font-bold text-slate-800">{value}</p>
          <p className="text-[11px] text-slate-500">{label}</p>
        </div>
      </div>
    </div>
  );
}

function StepCard({
  step,
  title,
  description,
}: {
  step: number;
  title: string;
  description: string;
}) {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-teal-50 flex items-center justify-center flex-shrink-0">
        <span className="text-xs font-bold text-teal-600">{step}</span>
      </div>
      <div>
        <p className="text-xs font-bold text-slate-700">{title}</p>
        <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  );
}
