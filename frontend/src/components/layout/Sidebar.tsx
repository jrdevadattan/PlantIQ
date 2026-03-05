"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import {
  TbLayoutDashboard,
  TbTargetArrow,
  TbActivityHeartbeat,
  TbAlertTriangle,
  TbHistory,
  TbSettings,
  TbBook2,
  TbHelp,
  TbChevronRight,
  TbBolt,
} from "react-icons/tb";
import { cn } from "@/lib/utils";
import { fetchHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/api";

const menuItems = [
  { name: "Dashboard", icon: TbLayoutDashboard, href: "/" },
  { name: "Predictions", icon: TbTargetArrow, href: "/predictions" },
  { name: "Live Monitor", icon: TbActivityHeartbeat, href: "/monitor" },
  { name: "Anomaly Detection", icon: TbAlertTriangle, href: "/anomalies" },
  { name: "Batch History", icon: TbHistory, href: "/history" },
];

const footerItems = [
  { name: "Settings", icon: TbSettings, href: "/settings" },
  { name: "Documentation", icon: TbBook2, href: "/docs" },
  { name: "Support", icon: TbHelp, href: "/support" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const data = await fetchHealth();
        if (mounted) setHealth(data);
      } catch {
        if (mounted) setHealth(null);
      }
    };
    poll();
    const interval = setInterval(poll, 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  return (
    <div className="w-64 bg-white border-r border-slate-200 flex flex-col h-screen overflow-hidden">
      {/* Logo area */}
      <div className="h-16 flex items-center px-6 gap-3 flex-shrink-0 border-b border-slate-100">
        <div className="w-9 h-9 rounded-lg bg-teal-600 flex items-center justify-center">
          <TbBolt className="w-5 h-5 text-white" />
        </div>
        <div className="flex flex-col">
          <span className="text-slate-800 font-bold text-sm tracking-tight leading-none">
            PlantIQ
          </span>
          <span className="text-[10px] font-medium text-slate-400 tracking-wide uppercase">
            Manufacturing
          </span>
        </div>
      </div>

      {/* Main navigation */}
      <nav className="flex-1 overflow-y-auto py-4 custom-scrollbar">
        <div className="px-3 mb-1">
          <span className="px-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Operations
          </span>
        </div>
        <div className="px-3 mt-2">
          {menuItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-semibold transition-all mb-0.5 group",
                  isActive
                    ? "bg-teal-50 text-teal-700 border border-teal-100"
                    : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                )}
              >
                <item.icon
                  className={cn(
                    "w-[18px] h-[18px] flex-shrink-0",
                    isActive ? "text-teal-600" : "text-slate-400 group-hover:text-slate-500"
                  )}
                />
                <span className="flex-1">{item.name}</span>
                {isActive && (
                  <TbChevronRight className="w-4 h-4 text-teal-400" />
                )}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Footer navigation */}
      <div className="border-t border-slate-100 py-3 px-3 flex-shrink-0">
        {footerItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-[12px] font-medium transition-all",
                isActive
                  ? "text-teal-600"
                  : "text-slate-400 hover:text-slate-600 hover:bg-slate-50"
              )}
            >
              <item.icon className="w-4 h-4" />
              {item.name}
            </Link>
          );
        })}
      </div>

      {/* System status */}
      <div className="mx-3 mb-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
        <div className="flex items-center gap-2 mb-1">
          <span className={cn("status-dot", health?.models_loaded ? "active" : "danger")} />
          <span className="text-[11px] font-bold text-slate-600">
            {health ? "System Online" : "Backend Offline"}
          </span>
        </div>
        <p className="text-[10px] text-slate-400 leading-relaxed">
          {health
            ? `All models loaded · v${health.version}`
            : "Connect backend at :8000"}
        </p>
      </div>
    </div>
  );
}
