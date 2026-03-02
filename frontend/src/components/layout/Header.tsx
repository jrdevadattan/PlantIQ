"use client";

import { TbBell, TbSearch, TbUser, TbClock } from "react-icons/tb";
import { useState, useEffect } from "react";

export function Header() {
  const [currentTime, setCurrentTime] = useState("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0">
      {/* Left: Search */}
      <div className="relative w-72">
        <TbSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          type="text"
          placeholder="Search batches, parameters..."
          className="w-full pl-10 pr-4 py-2 text-xs font-medium rounded-lg border border-slate-200 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-300 text-slate-700 placeholder:text-slate-400"
        />
      </div>

      {/* Right: Status + Notifications + User */}
      <div className="flex items-center gap-4">
        {/* Clock */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-100">
          <TbClock className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-[11px] font-mono font-bold text-slate-600 w-16 text-center">
            {currentTime}
          </span>
        </div>

        {/* Active batch indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-lg border border-emerald-100">
          <span className="status-dot active" />
          <span className="text-[11px] font-bold text-emerald-700">
            1 Batch Running
          </span>
        </div>

        {/* Notifications */}
        <button className="relative p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-colors">
          <TbBell className="w-5 h-5" />
          <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-amber-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center">
            2
          </span>
        </button>

        {/* User */}
        <div className="flex items-center gap-2 pl-4 border-l border-slate-200">
          <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
            <TbUser className="w-4 h-4 text-slate-500" />
          </div>
          <div className="flex flex-col">
            <span className="text-[11px] font-bold text-slate-700">Operator</span>
            <span className="text-[10px] text-slate-400">Shift A</span>
          </div>
        </div>
      </div>
    </header>
  );
}
