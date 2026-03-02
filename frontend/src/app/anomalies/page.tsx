"use client";

import dynamic from "next/dynamic";

const AnomalyDetector = dynamic(
  () => import("@/components/anomalies/AnomalyDetector").then(m => m.AnomalyDetector),
  { ssr: false }
);

export default function AnomaliesPage() {
  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-lg font-bold text-slate-800">Anomaly Detection</h1>
        <p className="text-xs text-slate-400 mt-0.5">
          LSTM Autoencoder power curve analysis with fault classification
        </p>
      </div>

      <AnomalyDetector />
    </div>
  );
}
