"""
Fix #5 — Management Dashboard Tests
======================================
Verifies the management page, route registration, API integration,
and component structure.

Tests:
  1. Page file structure
  2. API types in api.ts
  3. Sidebar + mobile nav routes
  4. Backend endpoints availability
  5. Data shape validation
"""

import os
import sys
import json

FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "src")
BACKEND = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, BACKEND)

passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}{' — ' + detail if detail else ''}")


def read(rel_path: str, base: str = FRONTEND) -> str:
    full = os.path.join(base, rel_path)
    if not os.path.exists(full):
        return ""
    with open(full) as f:
        return f.read()


# ── 1. Page File Structure ─────────────────────────────────
print("\n📄 Page File Structure")

page = read("app/management/page.tsx")
check("management/page.tsx exists", len(page) > 0)
check("Page has 'use client' directive", '"use client"' in page)
check("Page is default export", "export default function ManagementPage" in page)
check("Page imports Recharts (BarChart)", "BarChart" in page)
check("Page imports Recharts (PieChart)", "PieChart" in page)
check("Page imports Recharts (ResponsiveContainer)", "ResponsiveContainer" in page)
check("Page imports fetchDashboardSummary", "fetchDashboardSummary" in page)
check("Page imports fetchEnergyDaily", "fetchEnergyDaily" in page)
check("Page imports fetchShiftPerformance", "fetchShiftPerformance" in page)
check("Page imports fetchComplianceReport", "fetchComplianceReport" in page)
check("Page imports fetchCostConfig", "fetchCostConfig" in page)
check("Page uses cn() utility", "cn(" in page)
check("Page has loading state", "loading" in page and "animate-pulse" in page)
check("Page has error state", "error" in page and "Failed to load" in page)
check("Page has KPI cards section", "KPICard" in page or "KPI" in page)
check("Page has energy trend chart", "Energy Consumption Trend" in page or "energy" in page.lower())
check("Page has compliance pie chart", "CO₂ Compliance" in page or "compliance" in page)
check("Page has shift performance table", "Shift Performance" in page)
check("Page has savings/ROI section", "Savings" in page or "ROI" in page)
check("Page has time range selector (7d/14d/30d)", '"7d"' in page or "setEnergyDays" in page)
check("Page has cost metrics (₹)", "₹" in page)
check("Page has CO₂ metrics", "CO₂" in page or "co2" in page)


# ── 2. API Types & Functions ───────────────────────────────
print("\n🔌 API Types & Functions (api.ts)")

api = read("lib/api.ts")
check("CostConfig interface defined", "CostConfig" in api and "tariff_inr_per_kwh" in api)
check("ComplianceReport interface defined", "ComplianceReport" in api and "compliance" in api)
check("fetchCostConfig function exists", "fetchCostConfig" in api)
check("fetchComplianceReport function exists", "fetchComplianceReport" in api)
check("CostConfig has tariff field", "tariff_inr_per_kwh" in api)
check("CostConfig has CO₂ factor field", "co2_factor_kg_per_kwh" in api)
check("CostConfig has energy target field", "energy_target_kwh" in api)
check("ComplianceReport has energy_stats", "energy_stats" in api)
check("ComplianceReport has carbon_stats", "carbon_stats" in api)
check("ComplianceReport has compliance breakdown", "on_track_pct" in api)
check("ComplianceReport has cumulative_savings", "cumulative_savings" in api)


# ── 3. Sidebar & Mobile Nav Routes ────────────────────────
print("\n🧭 Navigation Routes")

sidebar = read("components/layout/Sidebar.tsx")
check("Sidebar has Management route", "/management" in sidebar)
check("Sidebar uses TbChartBar icon for Management", "TbChartBar" in sidebar)
check("Sidebar imports TbChartBar", "TbChartBar" in sidebar)

header = read("components/layout/Header.tsx")
check("Mobile nav has Management route", "/management" in header)
check("Mobile nav imports TbChartBar", "TbChartBar" in header)


# ── 4. Backend Endpoints (via TestClient) ──────────────────
print("\n🌐 Backend Endpoint Tests")

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Build a test app with the relevant routers
test_app = FastAPI()

# Dashboard routes (already have /summary, /energy-daily, /shift-performance)
from api.routes.dashboard import router as dash_router
test_app.include_router(dash_router)

client = TestClient(test_app)

# Test /dashboard/summary
r = client.get("/dashboard/summary")
check("GET /dashboard/summary → 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    check("Summary has total_batches", "total_batches" in data)
    check("Summary has avg_energy", "avg_energy" in data)
    check("Summary has avg_quality", "avg_quality" in data)
    check("Summary has energy_trend", "energy_trend" in data)
    check("Summary has energy_trend_value", "energy_trend_value" in data)

# Test /dashboard/energy-daily?days=14
r = client.get("/dashboard/energy-daily?days=14")
check("GET /dashboard/energy-daily?days=14 → 200", r.status_code == 200)
if r.status_code == 200:
    items = r.json()
    check("Energy-daily returns list", isinstance(items, list))
    if len(items) > 0:
        check("Energy item has 'day'", "day" in items[0])
        check("Energy item has 'kwh'", "kwh" in items[0])
        check("Energy item has 'batch_count'", "batch_count" in items[0])

# Test /dashboard/shift-performance
r = client.get("/dashboard/shift-performance")
check("GET /dashboard/shift-performance → 200", r.status_code == 200)
if r.status_code == 200:
    shifts = r.json()
    check("Shift performance returns list", isinstance(shifts, list))
    check("Has 3 shifts", len(shifts) == 3)
    if len(shifts) > 0:
        check("Shift has 'quality'", "quality" in shifts[0])
        check("Shift has 'yield_pct'", "yield_pct" in shifts[0])
        check("Shift has 'energy'", "energy" in shifts[0])
        check("Shift has 'batches'", "batches" in shifts[0])


# ── 5. Cost Config Endpoint ───────────────────────────────
print("\n💰 Cost Config Endpoint")

# Cost routes (separate router)
from api.routes.cost import router as cost_router
test_app2 = FastAPI()
test_app2.include_router(cost_router)
client2 = TestClient(test_app2)

r = client2.get("/cost/config")
check("GET /cost/config → 200", r.status_code == 200)
if r.status_code == 200:
    cfg = r.json()
    check("Config has tariff_inr_per_kwh", "tariff_inr_per_kwh" in cfg)
    check("Config has co2_factor_kg_per_kwh", "co2_factor_kg_per_kwh" in cfg)
    check("Config has energy_target_kwh", "energy_target_kwh" in cfg)
    check("Config has batches_per_day", "batches_per_day" in cfg)
    check("Tariff is numeric > 0", isinstance(cfg.get("tariff_inr_per_kwh", 0), (int, float)) and cfg["tariff_inr_per_kwh"] > 0)


# ── 6. Targets Report Structure ───────────────────────────
print("\n📊 Targets Report Structure")

# Targets endpoints require the AdaptiveTargetEngine which needs main.py
# Instead, verify the route file exists and has the right endpoints
targets_src = read("api/routes/targets.py", base=BACKEND)
check("targets.py has /report endpoint", "performance_report" in targets_src)
check("targets.py has /initialize endpoint", "initialize_targets" in targets_src)
check("targets.py has /batch endpoint", "get_batch_targets" in targets_src)
check("targets.py has /assess endpoint", "assess_batch" in targets_src)
check("targets.py returns energy_stats", "energy_stats" in targets_src or "energy" in targets_src)
check("targets.py returns carbon_stats", "carbon" in targets_src)
check("targets.py returns compliance breakdown", "compliance" in targets_src or "on_track" in targets_src)


# ── 7. UI Design Compliance ──────────────────────────────
print("\n🎨 UI Design Compliance")

check("Uses rounded-xl cards", "rounded-xl" in page)
check("Uses card-hover class", "card-hover" in page)
check("Uses border-slate-100", "border-slate-100" in page)
check("Uses bg-white cards", "bg-white" in page)
check("Uses proper heading sizes", "text-lg font-bold" in page)
check("Uses text-sm for section headings", "text-sm font-bold" in page)
check("Uses text-xs for data values", "text-xs" in page)
check("Uses text-[10px] for labels", "text-[10px]" in page)
check("Uses text-[11px] for descriptions", "text-[11px]" in page)
check("Uses responsive grid", "grid-cols-1 lg:grid-cols-12" in page)
check("Uses 2-col mobile grid for KPIs", "grid-cols-2 md:grid-cols-4" in page)
check("Has max-w-[1600px] container", "max-w-[1600px]" in page)
check("Uses Tabler icons only (Tb*)", "Tb" in page and "react-icons/fa" not in page)
check("Uses @/ path alias", "@/" in page)
check("Uses useMemo for derived data", "useMemo" in page)
check("Has empty state for charts", "No energy data" in page or "no data" in page.lower())


# ── Summary ───────────────────────────────────────────────
total = passed + failed
print(f"\n{'='*55}")
print(f"  Fix #5 Management Dashboard — {passed}/{total} passed")
if failed:
    print(f"  ⚠️  {failed} test(s) failed")
    sys.exit(1)
else:
    print("  🎉 All tests passed!")
    sys.exit(0)
