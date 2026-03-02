"use client";

import dynamic from "next/dynamic";

const PreBatchPanel = dynamic(
  () => import("@/components/predictions/PreBatchPanel").then(m => m.PreBatchPanel),
  { ssr: false }
);
const ShapChart = dynamic(
  () => import("@/components/predictions/ShapChart").then(m => m.ShapChart),
  { ssr: false }
);

export default function PredictionsPage() {
  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-lg font-bold text-slate-800">Batch Predictions</h1>
        <p className="text-xs text-slate-400 mt-0.5">
          Multi-target prediction with explainability — predict before you produce
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7">
          <PreBatchPanel />
        </div>
        <div className="lg:col-span-5">
          <ShapChart />
        </div>
      </div>
    </div>
  );
}
