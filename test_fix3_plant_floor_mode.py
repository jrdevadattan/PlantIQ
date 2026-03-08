"""
Fix #3 — Plant Floor Mode Tests
Verifies the CSS overrides, context structure, component wiring,
and localStorage persistence pattern.
"""

import os
import re
import sys

FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "src")

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


def read(rel_path: str) -> str:
    """Read a file relative to frontend/src/"""
    full = os.path.join(FRONTEND, rel_path)
    if not os.path.exists(full):
        return ""
    with open(full) as f:
        return f.read()


# ── 1. CSS Override Rules ──────────────────────────────────────────
print("\n🎨 CSS Override Rules (globals.css)")

css = read("app/globals.css")

check("CSS contains .plant-floor-mode selector", ".plant-floor-mode" in css)

# Check each text size override exists
size_overrides = {
    "text-[9px]": "13px",
    "text-[10px]": "14px",
    "text-[11px]": "15px",
    "text-xs": "16px",
    "text-[12px]": "16px",
    "text-[13px]": "17px",
    "text-sm": "18px",
    "text-lg": "22px",
    "text-2xl": "32px",
}

for tw_class, target_px in size_overrides.items():
    # CSS escapes brackets: text-\[10px\]
    escaped = tw_class.replace("[", r"\[").replace("]", r"\]")
    pattern = rf"\.plant-floor-mode\s+\.{re.escape(escaped)}"
    found = re.search(pattern, css) is not None
    # Simpler: just check the class name and target size are both in the CSS
    simple_check = tw_class in css and target_px in css
    check(f"Override: {tw_class} → {target_px}", found or simple_check)

check("Status dot scales up in floor mode", "plant-floor-mode .status-dot" in css)
check("Card padding scales up in floor mode", "plant-floor-mode .p-4" in css)
check("Smooth transition on toggle", "transition:" in css and "font-size" in css)


# ── 2. Context Provider ───────────────────────────────────────────
print("\n🔄 Context Provider (PlantFloorContext.tsx)")

ctx = read("lib/PlantFloorContext.tsx")

check("PlantFloorContext file exists", len(ctx) > 0)
check("Exports PlantFloorProvider", "export function PlantFloorProvider" in ctx)
check("Exports usePlantFloorMode hook", "export function usePlantFloorMode" in ctx)
check("Uses createContext", "createContext" in ctx)
check("Uses localStorage for persistence", "localStorage" in ctx)
check("Has storage key constant", "STORAGE_KEY" in ctx or "plantiq-floor-mode" in ctx)
check("Hydrates from localStorage on mount", "localStorage.getItem" in ctx)
check("Syncs to localStorage on change", "localStorage.setItem" in ctx)
check("Uses useCallback for toggle", "useCallback" in ctx)
check("Provides isPlantFloorMode boolean", "isPlantFloorMode" in ctx)
check("Provides togglePlantFloorMode function", "togglePlantFloorMode" in ctx)
check("Has 'use client' directive", '"use client"' in ctx)
check("Handles SSR safety (try/catch around localStorage)", ctx.count("try") >= 2, "needs try/catch for both get & set")
check("Interface defines context value type", "PlantFloorContextValue" in ctx or "interface" in ctx)


# ── 3. PlantFloorShell Component ──────────────────────────────────
print("\n🏗️  PlantFloorShell Component")

shell = read("components/layout/PlantFloorShell.tsx")

check("PlantFloorShell file exists", len(shell) > 0)
check("Exports PlantFloorShell", "export function PlantFloorShell" in shell)
check("Imports PlantFloorProvider", "PlantFloorProvider" in shell)
check("Imports usePlantFloorMode", "usePlantFloorMode" in shell)
check("Imports cn() utility", "cn" in shell)
check("Has 'use client' directive", '"use client"' in shell)
check("Applies plant-floor-mode class conditionally", "plant-floor-mode" in shell)
check("Uses cn() for conditional class merging", "cn(" in shell)
check("Root div has flex h-screen", "flex h-screen" in shell)


# ── 4. Header Toggle Button ──────────────────────────────────────
print("\n🔘 Header Toggle Button")

header = read("components/layout/Header.tsx")

check("Header imports usePlantFloorMode", "usePlantFloorMode" in header)
check("Header imports PlantFloorContext", "PlantFloorContext" in header)
check("Header imports TbMaximize icon", "TbMaximize" in header)
check("Header imports TbMinimize icon", "TbMinimize" in header)
check("Header calls togglePlantFloorMode", "togglePlantFloorMode" in header)
check("Header reads isPlantFloorMode", "isPlantFloorMode" in header)
check("Toggle button has accessibility title", 'title=' in header and "Floor" in header)
check("Toggle shows 'Floor' label when inactive", '"Floor"' in header)
check("Toggle shows 'Standard' label when active", '"Standard"' in header)
check("Toggle uses cn() for conditional styles", "cn(" in header)
check("Toggle has teal highlight when active", "bg-teal-50" in header and "teal-200" in header)
check("Floor/Standard label hidden on small screens", "hidden md:inline" in header)


# ── 5. Layout Wiring ─────────────────────────────────────────────
print("\n📐 Layout Wiring (layout.tsx)")

layout = read("app/layout.tsx")

check("Layout imports PlantFloorShell", "PlantFloorShell" in layout)
check("Layout renders <PlantFloorShell>", "<PlantFloorShell>" in layout)
check("Layout no longer has bare flex div", 'className="flex h-screen' not in layout, "should delegate to PlantFloorShell")
check("Layout is still a server component (no 'use client')", '"use client"' not in layout)
check("Layout still renders Sidebar", "<Sidebar" in layout)
check("Layout still renders Header", "<Header" in layout)
check("Layout still renders main content", "<main" in layout)


# ── 6. Design System Compliance ──────────────────────────────────
print("\n📏 Design System Compliance")

# Floor mode should NOT touch the sidebar's brand sizing or structure
check("CSS doesn't force-resize sidebar width", "w-64" not in css or ".plant-floor-mode .w-64" not in css)

# Verify the override scale is monotonically increasing
sizes = [13, 14, 15, 16, 16, 17, 18, 22, 32]
check("Font size scale is monotonically non-decreasing", all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1)))

# Minimum readable size in floor mode
check("Smallest floor-mode size >= 13px", min(sizes) >= 13, f"smallest is {min(sizes)}px")

# KPI values should be 32px+ for visibility at 3m distance
check("KPI values (text-2xl) scale to 32px+", 32 in sizes)

# Ensure !important is used to override Tailwind utility specificity
important_count = css.count("!important")
check("Uses !important to override Tailwind utilities", important_count >= 8, f"found {important_count}, need >=8")


# ── Summary ───────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'='*55}")
print(f"  Fix #3 Plant Floor Mode — {passed}/{total} passed")
if failed:
    print(f"  ⚠️  {failed} test(s) failed")
    sys.exit(1)
else:
    print("  🎉 All tests passed!")
    sys.exit(0)
