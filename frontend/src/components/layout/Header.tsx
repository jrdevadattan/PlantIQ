"use client";

import { TbBell, TbSearch, TbUser, TbClock, TbMenu2, TbX, TbLayoutDashboard, TbTargetArrow, TbActivityHeartbeat, TbAlertTriangle, TbHistory, TbMaximize, TbMinimize, TbChartBar } from "react-icons/tb";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { usePlantFloorMode } from "@/lib/PlantFloorContext";

const mobileNavItems = [
  { name: "Dashboard", icon: TbLayoutDashboard, href: "/" },
  { name: "Predictions", icon: TbTargetArrow, href: "/predictions" },
  { name: "Live Monitor", icon: TbActivityHeartbeat, href: "/monitor" },
  { name: "Anomalies", icon: TbAlertTriangle, href: "/anomalies" },
  { name: "History", icon: TbHistory, href: "/history" },
  { name: "Management", icon: TbChartBar, href: "/management" },
];

export function Header() {
  const [currentTime, setCurrentTime] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const { isPlantFloorMode, togglePlantFloorMode } = usePlantFloorMode();

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

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  return (
    <>
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-4 lg:px-6 flex-shrink-0">
      {/* Left: Mobile hamburger + Search */}
      <div className="flex items-center gap-3">
        {/* Mobile menu toggle */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="lg:hidden p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-50 rounded-lg transition-colors"
          aria-label="Toggle navigation"
        >
          {mobileMenuOpen ? <TbX className="w-5 h-5" /> : <TbMenu2 className="w-5 h-5" />}
        </button>

        <div className="relative w-48 sm:w-72 hidden sm:block">
          <TbSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search batches, parameters..."
            className="w-full pl-10 pr-4 py-2 text-xs font-medium rounded-lg border border-slate-200 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-300 text-slate-700 placeholder:text-slate-400"
          />
        </div>
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

        {/* Plant Floor Mode toggle */}
        <button
          onClick={togglePlantFloorMode}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all",
            isPlantFloorMode
              ? "bg-teal-50 border-teal-200 text-teal-700"
              : "bg-slate-50 border-slate-100 text-slate-400 hover:text-slate-600 hover:border-slate-200"
          )}
          title={isPlantFloorMode ? "Exit Plant Floor Mode" : "Plant Floor Mode — larger text for shop-floor displays"}
        >
          {isPlantFloorMode ? (
            <TbMinimize className="w-4 h-4" />
          ) : (
            <TbMaximize className="w-4 h-4" />
          )}
          <span className="text-[11px] font-bold hidden md:inline">
            {isPlantFloorMode ? "Standard" : "Floor"}
          </span>
        </button>

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

    {/* Mobile navigation overlay */}
    {mobileMenuOpen && (
      <div className="lg:hidden fixed inset-0 z-50 bg-black/30" onClick={() => setMobileMenuOpen(false)}>
        <nav
          className="absolute top-14 left-0 right-0 bg-white border-b border-slate-200 shadow-lg p-3 space-y-1"
          onClick={(e) => e.stopPropagation()}
        >
          {mobileNavItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all",
                  isActive
                    ? "bg-teal-50 text-teal-700 border border-teal-100"
                    : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                )}
              >
                <item.icon className={cn("w-5 h-5", isActive ? "text-teal-600" : "text-slate-400")} />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>
    )}
    </>
  );
}
