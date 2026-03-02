"use client";

import React, { useState } from "react";
import {
  TbSend,
  TbTemperature,
  TbArrowAutofitWidth,
  TbClock,
  TbBox,
  TbCategory,
  TbSunHigh,
  TbCircleCheck,
  TbChartBar,
  TbTrendingUp,
  TbBolt,
  TbGauge,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import type { PredictionResult } from "@/lib/mockData";

interface FormField {
  name: string;
  label: string;
  icon: any;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  unit: string;
}

const fields: FormField[] = [
  { name: "temperature", label: "Temperature", icon: TbTemperature, min: 160, max: 200, step: 1, defaultValue: 183, unit: "C" },
  { name: "conveyorSpeed", label: "Conveyor Speed", icon: TbArrowAutofitWidth, min: 50, max: 100, step: 1, defaultValue: 76, unit: "%" },
  { name: "holdTime", label: "Hold Time", icon: TbClock, min: 5, max: 40, step: 1, defaultValue: 18, unit: "min" },
  { name: "batchSize", label: "Batch Size", icon: TbBox, min: 200, max: 800, step: 10, defaultValue: 500, unit: "kg" },
];

const materialTypes = [
  { value: 1, label: "Type A — Standard" },
  { value: 2, label: "Type B — Dense" },
];

const resultCards = [
  { key: "qualityScore", label: "Quality Score", icon: TbChartBar, color: "teal", unit: "%" },
  { key: "yieldPct", label: "Yield", icon: TbTrendingUp, color: "emerald", unit: "%" },
  { key: "performancePct", label: "Performance", icon: TbGauge, color: "indigo", unit: "%" },
  { key: "energyKwh", label: "Energy", icon: TbBolt, color: "amber", unit: "kWh" },
];

const colorMap: Record<string, string> = {
  teal: "bg-teal-50 text-teal-600 border-teal-100",
  emerald: "bg-emerald-50 text-emerald-600 border-emerald-100",
  indigo: "bg-indigo-50 text-indigo-600 border-indigo-100",
  amber: "bg-amber-50 text-amber-600 border-amber-100",
};

export function PreBatchPanel() {
  const [formData, setFormData] = useState({
    temperature: 183,
    conveyorSpeed: 76,
    holdTime: 18,
    batchSize: 500,
    materialType: 1,
    hourOfDay: new Date().getHours(),
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = () => {
    setLoading(true);
    // Simulate API call  
    setTimeout(() => {
      // Deterministic mock prediction based on inputs
      const t = formData.temperature;
      const s = formData.conveyorSpeed;
      const h = formData.holdTime;
      const b = formData.batchSize;
      const m = formData.materialType;

      const energy = 0.3 * t + 0.2 * s + 0.5 * h + 0.01 * b + 3 * (m === 2 ? 1 : 0) - 15;
      const yld = Math.min(100, Math.max(70, 98 - 0.08 * Math.pow(t - 183, 2) - 0.04 * Math.pow(s - 75, 2)));
      const quality = Math.min(100, Math.max(60, 92 + 0.3 * (t - 175) - 0.02 * Math.pow(h, 2) + 5));
      const perf = Math.min(100, Math.max(60, 95 - 0.5 * Math.pow(formData.hourOfDay - 10, 2) / 10));

      setPrediction({
        qualityScore: Number(quality.toFixed(1)),
        yieldPct: Number(yld.toFixed(1)),
        performancePct: Number(perf.toFixed(1)),
        energyKwh: Number(energy.toFixed(1)),
        confidence: {
          quality: [quality - 3, quality + 3],
          yield: [yld - 2.5, yld + 2.5],
          performance: [perf - 4, perf + 4],
          energy: [energy - 3.5, energy + 3.5],
        },
      });
      setLoading(false);
    }, 800);
  };

  const updateField = (name: string, value: number) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  return (
    <div className="space-y-5">
      {/* Input Form */}
      <div className="bg-white rounded-xl border border-slate-100 p-5">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-9 h-9 rounded-lg bg-teal-50 flex items-center justify-center">
            <TbSend className="w-[18px] h-[18px] text-teal-600" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">Batch Parameters</h3>
            <p className="text-[11px] text-slate-400">
              Enter setup values to predict outcomes before the batch starts
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {fields.map((field) => {
            const Icon = field.icon;
            const val = formData[field.name as keyof typeof formData] as number;
            return (
              <div key={field.name} className="space-y-2">
                <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
                  <Icon className="w-3.5 h-3.5 text-slate-400" />
                  {field.label}
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={val}
                    onChange={(e) => updateField(field.name, Number(e.target.value))}
                    className="flex-1 h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-teal-500"
                  />
                  <div className="w-20 flex items-center gap-1 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <input
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={val}
                      onChange={(e) => updateField(field.name, Number(e.target.value))}
                      className="w-full text-xs font-bold text-slate-700 bg-transparent outline-none text-center"
                    />
                    <span className="text-[10px] text-slate-400 font-medium">{field.unit}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Material type */}
        <div className="mt-4 space-y-2">
          <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            <TbCategory className="w-3.5 h-3.5 text-slate-400" />
            Material Type
          </label>
          <div className="flex gap-3">
            {materialTypes.map((mt) => (
              <button
                key={mt.value}
                onClick={() => updateField("materialType", mt.value)}
                className={cn(
                  "flex-1 px-4 py-2.5 rounded-lg text-xs font-bold border transition-all",
                  formData.materialType === mt.value
                    ? "bg-teal-50 text-teal-700 border-teal-200"
                    : "bg-white text-slate-500 border-slate-200 hover:bg-slate-50"
                )}
              >
                {mt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Hour of day */}
        <div className="mt-4 space-y-2">
          <label className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-wide">
            <TbSunHigh className="w-3.5 h-3.5 text-slate-400" />
            Shift Hour
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={23}
              value={formData.hourOfDay}
              onChange={(e) => updateField("hourOfDay", Number(e.target.value))}
              className="flex-1 h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-teal-500"
            />
            <div className="w-20 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">
              <span className="text-xs font-bold text-slate-700 block text-center">
                {formData.hourOfDay}:00
              </span>
            </div>
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className={cn(
            "mt-5 w-full py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
            loading
              ? "bg-slate-100 text-slate-400 cursor-not-allowed"
              : "bg-teal-600 text-white hover:bg-teal-700 active:scale-[0.99] shadow-sm"
          )}
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-slate-300 border-t-slate-500 rounded-full animate-spin" />
              Running Prediction...
            </>
          ) : (
            <>
              <TbSend className="w-4 h-4" />
              Predict Batch Outcomes
            </>
          )}
        </button>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-white rounded-xl border border-slate-100 p-5 animate-count">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-9 h-9 rounded-lg bg-emerald-50 flex items-center justify-center">
              <TbCircleCheck className="w-[18px] h-[18px] text-emerald-600" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">Prediction Results</h3>
              <p className="text-[11px] text-slate-400">
                Multi-target prediction with confidence intervals
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {resultCards.map((card) => {
              const Icon = card.icon;
              const val = prediction[card.key as keyof PredictionResult] as number;
              const confKey = card.key === "qualityScore" ? "quality" : card.key === "yieldPct" ? "yield" : card.key === "performancePct" ? "performance" : "energy";
              const conf = prediction.confidence[confKey as keyof typeof prediction.confidence];
              return (
                <div
                  key={card.key}
                  className={cn("p-4 rounded-lg border", colorMap[card.color])}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className="w-4 h-4" />
                    <span className="text-[11px] font-bold uppercase tracking-wide">
                      {card.label}
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-slate-800">
                    {val}
                    <span className="text-xs font-semibold text-slate-400 ml-1">
                      {card.unit}
                    </span>
                  </p>
                  <p className="text-[10px] text-slate-400 mt-1">
                    CI: {conf[0].toFixed(1)} — {conf[1].toFixed(1)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
