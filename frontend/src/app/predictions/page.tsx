"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import type { BatchPredictionParams, BatchPredictionResponse, CostTranslation } from "@/lib/api";

const PreBatchPanel = dynamic(
  () => import("@/components/predictions/PreBatchPanel").then(m => m.PreBatchPanel),
  { ssr: false }
);
const ShapChart = dynamic(
  () => import("@/components/predictions/ShapChart").then(m => m.ShapChart),
  { ssr: false }
);
const RecommendationPanel = dynamic(
  () => import("@/components/predictions/RecommendationPanel").then(m => m.RecommendationPanel),
  { ssr: false }
);
const CostPanel = dynamic(
  () => import("@/components/predictions/CostPanel").then(m => m.CostPanel),
  { ssr: false }
);

export default function PredictionsPage() {
  const [lastBatchId, setLastBatchId] = useState<string | null>(null);
  const [lastParams, setLastParams] = useState<BatchPredictionParams | null>(null);
  const [liveMode, setLiveMode] = useState(false);
  const [shapContributions, setShapContributions] = useState<Array<{ feature: string; contribution: number; direction: string }>>([]);
  const [costData, setCostData] = useState<CostTranslation | null>(null);

  const handlePrediction = useCallback((batchId: string, params: BatchPredictionParams, response?: BatchPredictionResponse) => {
    setLastBatchId(batchId);
    setLastParams(params);
    // Extract cost_translation from extended response if available
    const extended = response as (BatchPredictionResponse & { cost_translation?: CostTranslation }) | undefined;
    if (extended?.cost_translation) {
      setCostData(extended.cost_translation);
    }
  }, []);

  const handleShapData = useCallback((contributions: Array<{ feature: string; contribution: number; direction: string }>) => {
    setShapContributions(contributions);
  }, []);

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
          <PreBatchPanel
            onPrediction={handlePrediction}
            onWhatIfToggle={setLiveMode}
          />
        </div>
        <div className="lg:col-span-5 space-y-5">
          <ShapChart
            batchId={lastBatchId}
            params={lastParams}
            liveMode={liveMode}
            onShapData={handleShapData}
          />
          <RecommendationPanel
            batchId={lastBatchId}
            params={lastParams}
            shapContributions={shapContributions}
            target="energy_kwh"
          />
          <CostPanel costData={costData} />
        </div>
      </div>
    </div>
  );
}
