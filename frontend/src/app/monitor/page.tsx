"use client";

import dynamic from "next/dynamic";

const LiveMonitor = dynamic(
  () => import("@/components/monitor/LiveMonitor").then(m => m.LiveMonitor),
  { ssr: false }
);

export default function MonitorPage() {
  return (
    <div className="p-5 space-y-5 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-lg font-bold text-slate-800">Live Batch Monitor</h1>
        <p className="text-xs text-slate-400 mt-0.5">
          Real-time power curve analysis with sliding window energy forecast
        </p>
      </div>

      <LiveMonitor />
    </div>
  );
}
