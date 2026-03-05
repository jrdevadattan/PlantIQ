# PlantIQ — Copilot Instructions

> **Read this file BEFORE and AFTER every task.** These are non-negotiable rules for working on PlantIQ.

---

## 1. Project Identity

**PlantIQ** is an AI-Driven Manufacturing Intelligence Platform (Hackathon — Track A: Predictive Modelling).

It predicts Quality, Yield, Performance, and Energy for industrial batches; detects anomalies in power curves via LSTM Autoencoder; explains predictions with SHAP; and provides real-time operator recommendations.

### Architecture Layers

| Layer | Purpose |
|---|---|
| **Data Sources** | IIoT/Smart Meters, MES/ERP, Historical Batches, Regulatory DB |
| **Data Pipeline** | KNN imputation → IQR outlier capping → Feature engineering (7 derived) → Normalization |
| **AI Core** | Multi-Target XGBoost (4 targets), LSTM Autoencoder (anomaly), RandomForest fault classifier, Sliding Window Forecaster, SHAP explainability |
| **Decision Engine** | Adaptive carbon targets, ranked recommendations, 3-severity alert system |
| **Output & UI** | Next.js 14 dashboard with Pre-Batch, Live Monitor, Anomaly Detector, SHAP panels |

---

## 2. Tech Stack — Locked Versions (DO NOT CHANGE)

### Frontend (Current — Active Development)

| Package | Version | Purpose |
|---|---|---|
| Next.js | 14.2.4 | App Router framework |
| React | ^18.3.1 | UI library |
| TypeScript | ^5 | Type system |
| Tailwind CSS | ^3.4.4 | Utility-first styling |
| Recharts | ^2.12.7 | Charts and graphs |
| Framer Motion | ^11.2.10 | Animations |
| react-icons | ^5.2.1 | Icon library (Tabler icons — `Tb*` prefix) |
| clsx | ^2.1.1 | Conditional class strings |
| tailwind-merge | ^2.3.0 | Merge Tailwind classes without conflicts |

### Backend (Planned — Per README Spec)

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Primary language |
| FastAPI | 0.111.0 | REST API + WebSocket |
| XGBoost | 2.0.3 | Multi-target prediction |
| PyTorch | 2.3.0 | LSTM Autoencoder |
| SHAP | 0.45.0 | Explainability |
| scikit-learn | 1.5.0 | Preprocessing, metrics, MultiOutputRegressor |

**MANDATORY:** Never upgrade, downgrade, or swap any package version unless explicitly asked by the user. If a version conflict exists, report it — do not silently resolve it.

---

## 3. Project Structure Map

```
plantiq/
├── .github/
│   └── copilot-instructions.md     ← THIS FILE — read before every task
├── README.md                        ← Comprehensive project spec (1595 lines)
├── frontend/                        ← Next.js 14 App Router
│   ├── src/
│   │   ├── app/                     ← Route segments (App Router)
│   │   │   ├── layout.tsx           ← Root layout (Sidebar + Header shell)
│   │   │   ├── page.tsx             ← Dashboard (/) — dynamic imports, SSR-off for charts
│   │   │   ├── globals.css          ← CSS variables, custom scrollbar, animations
│   │   │   ├── predictions/page.tsx ← /predictions route
│   │   │   ├── monitor/page.tsx     ← /monitor route
│   │   │   ├── anomalies/page.tsx   ← /anomalies route
│   │   │   └── history/page.tsx     ← /history route
│   │   ├── components/
│   │   │   ├── layout/              ← Header.tsx, Sidebar.tsx (persistent shell)
│   │   │   ├── dashboard/           ← StatCard, EnergyChart, PerformanceGauges, RecentBatches, ShiftOverview
│   │   │   ├── predictions/         ← PreBatchPanel.tsx, ShapChart.tsx
│   │   │   ├── monitor/             ← LiveMonitor.tsx
│   │   │   ├── anomalies/           ← AnomalyDetector.tsx
│   │   │   └── history/             ← BatchTable.tsx
│   │   └── lib/
│   │       ├── utils.ts             ← cn() helper (clsx + twMerge)
│   │       └── mockData.ts          ← Typed interfaces + mock data
│   ├── tailwind.config.ts           ← Custom colors: plant-*, iron-*
│   ├── next.config.js
│   ├── tsconfig.json                ← Path alias: @/* → ./src/*
│   └── package.json
└── backend/                         ← (To be built — see README for full spec)
```

---

## 4. Workflow Orchestration

### 4.1 Plan Mode (Default for Non-Trivial Work)

- Enter plan mode for ANY task with 3+ steps or architectural decisions.
- Write a plan to `tasks/todo.md` (or use the todo tracking tool) BEFORE coding.
- If something goes sideways, **STOP and re-plan immediately** — do not keep pushing broken changes.
- Use plan mode for verification steps, not just building.

### 4.2 Task Management

1. **Plan First:** Write a plan with checkable items before starting implementation.
2. **Verify Plan:** Double-check the plan makes sense before writing code.
3. **Track Progress:** Mark items complete as each step finishes.
4. **Explain Changes:** Provide a high-level summary at each step.
5. **Document Results:** Add a review section when the task is done.

### 4.3 Subagent Strategy

- Use subagents liberally to keep the main context window clean.
- Offload research, file exploration, and parallel analysis to subagents.
- One task per subagent for focused execution.

### 4.4 Verification Before Done

- **Never mark a task complete without proving it works.**
- Run type checks (`npx tsc --noEmit`), check for lint errors, verify dev server compiles.
- Ask: "Would a staff engineer approve this?"
- For UI work: confirm the component renders without blank screens or console errors.

### 4.5 Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: reconsider. Implement the clean solution.
- But don't over-engineer simple things — simplicity wins.

---

## 5. Code Conventions — Frontend

### 5.1 File & Naming Rules

| Entity | Convention | Example |
|---|---|---|
| Components | PascalCase, named export | `export function StatCard() {}` |
| Files | PascalCase for components | `StatCard.tsx`, `EnergyChart.tsx` |
| Utilities | camelCase | `utils.ts`, `mockData.ts` |
| Route pages | `page.tsx` (Next.js App Router convention) | `app/predictions/page.tsx` |
| Interfaces | PascalCase, descriptive | `BatchRecord`, `PredictionResult`, `ShapValue` |
| CSS variables | kebab-case with `--color-` prefix | `--color-primary`, `--color-danger` |
| Icon imports | `Tb*` prefix (Tabler icons from react-icons) | `TbBolt`, `TbAlertTriangle` |

### 5.2 Component Patterns

**Client Components:**
```tsx
"use client";
// MUST be the first line in any component that uses hooks, event handlers, or browser APIs
```

**Dynamic Imports (SSR-off for chart components):**
```tsx
const EnergyChart = dynamic(
  () => import("@/components/dashboard/EnergyChart").then(mod => mod.EnergyChart),
  { ssr: false }
);
```
Use `{ ssr: false }` for ALL Recharts-based components and any component accessing `window`/`document`.

**Class Merging:**
```tsx
import { cn } from "@/lib/utils";

className={cn(
  "base-classes here",
  isActive && "conditional-classes",
  variant === "teal" ? "teal-classes" : "other-classes"
)}
```
Always use `cn()` for conditional classes — never string concatenation.

**Variant Pattern (for styled components):**
```tsx
const variantStyles = {
  teal: { icon: "bg-teal-50 text-teal-600", border: "border-teal-100" },
  amber: { icon: "bg-amber-50 text-amber-600", border: "border-amber-100" },
  // ...
};
// Reference: StatCard.tsx, all dashboard components
```

### 5.3 Import Ordering

```tsx
// 1. React / Next.js
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// 2. Third-party libraries
import { motion } from "framer-motion";
import { LineChart, Line } from "recharts";

// 3. Icons (always Tb* from react-icons/tb)
import { TbBolt, TbAlertTriangle } from "react-icons/tb";

// 4. Internal utilities
import { cn } from "@/lib/utils";

// 5. Internal components and data
import { StatCard } from "@/components/dashboard/StatCard";
import { recentBatches } from "@/lib/mockData";

// 6. Types (if separate)
import type { BatchRecord } from "@/lib/mockData";
```

### 5.4 Path Alias

Always use `@/*` for imports — never relative paths like `../../components/...`:
```tsx
// CORRECT
import { cn } from "@/lib/utils";
import { StatCard } from "@/components/dashboard/StatCard";

// WRONG
import { cn } from "../../lib/utils";
```

---

## 6. Styling Rules — Tailwind CSS

### 6.1 Design System Tokens

#### Colors (Use ONLY These)

| Category | Tailwind Classes | CSS Variable | Usage |
|---|---|---|---|
| **Background** | `bg-slate-50`, `bg-slate-50/50` | `--color-bg: #f8fafc` | Page backgrounds |
| **Card surface** | `bg-white` | `--color-card: #ffffff` | All card/panel backgrounds |
| **Border** | `border-slate-100`, `border-slate-200` | `--color-border: #e2e8f0` | Card borders, dividers |
| **Primary** | `text-teal-600`, `bg-teal-50`, `bg-teal-600` | `--color-primary: #0d9488` | Active nav, brand, primary actions |
| **Accent/Warning** | `text-amber-600`, `bg-amber-50`, `bg-amber-500` | `--color-accent: #f59e0b` | Warnings, attention items |
| **Danger** | `text-red-500`, `bg-red-50` | `--color-danger: #ef4444` | Errors, critical alerts |
| **Success** | `text-emerald-600`, `bg-emerald-50`, `text-green-*` | `--color-success: #22c55e` | Success states, on-track |
| **Text primary** | `text-slate-800` | `#1e293b` | Headings, values |
| **Text secondary** | `text-slate-500`, `text-slate-600` | — | Labels, descriptions |
| **Text muted** | `text-slate-400`, `text-slate-300` | — | Hints, timestamps |
| **Custom palette** | `plant-50` to `plant-900`, `iron-50` to `iron-900` | — | Extended palette (see tailwind.config.ts) |

**MANDATORY:** Do not introduce new color values. If a new shade is needed, use existing Tailwind slate/teal/amber/emerald/red scales or the custom `plant-*`/`iron-*` palette.

#### Typography

| Token | Tailwind Class | Usage |
|---|---|---|
| Micro data | `text-[10px]` | Timestamps, muted labels, sub-values |
| Small data | `text-[11px]` | Status labels, nav text, clock, badge text |
| Body small | `text-xs` (12px) | Table data, descriptions |
| Body | `text-sm` (14px) | Form labels, paragraph text |
| Heading | `text-lg` (18px) | Section headings |
| Page title | `text-lg font-bold` | Page-level headings |
| KPI value | `text-2xl font-bold` | Dashboard stat numbers |

**Font families:**
- Body: `Inter` (loaded via Google Fonts in globals.css)
- Mono: `JetBrains Mono` / `Fira Code` (for clocks, code, IDs — use `font-mono`)

#### Spacing

| Context | Value | Example |
|---|---|---|
| Page padding | `p-5` | Main content area |
| Card padding | `p-3` to `p-4` | Interior of cards |
| Grid gap | `gap-4` to `gap-5` | Between cards in grid |
| Section spacing | `space-y-5` | Between dashboard rows |
| Max width | `max-w-[1600px] mx-auto` | Content container |

#### Border Radius

| Component | Class |
|---|---|
| Cards | `rounded-xl` |
| Buttons / pills | `rounded-lg` |
| Avatar / badge | `rounded-full` |
| Input fields | `rounded-lg` |

### 6.2 Card Pattern (Standard)

Every card/panel MUST follow this pattern:
```tsx
<div className="bg-white rounded-xl border border-slate-100 p-4 card-hover">
  {/* Header */}
  <div className="flex items-center justify-between mb-4">
    <h2 className="text-sm font-bold text-slate-700">Card Title</h2>
    {/* Optional: badge, icon, action */}
  </div>
  {/* Content */}
  {/* ... */}
</div>
```

For interactive cards, add `card-hover` class (defined in globals.css — applies `translateY(-1px)` + shadow on hover).

### 6.3 Status Indicators

Use the established status-dot CSS pattern:
```tsx
<span className="status-dot active" />   // Green — normal/online
<span className="status-dot warning" />  // Amber — warning state
<span className="status-dot danger" />   // Red — critical/error
```

For anomaly severity levels:
```
🟢 Normal:   score < 0.15  → emerald/green classes
🟡 Watch:    score 0.15–0.30 → amber/yellow classes
🟠 Warning:  score 0.30–0.60 → orange classes
🔴 Critical: score > 0.60  → red classes
```

### 6.4 Responsive Grid System

```tsx
// KPI row: 2 cols mobile → 3 cols tablet → 6 cols desktop
<div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">

// Two-panel layout with 12-col grid
<div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
  <div className="lg:col-span-5">{/* Narrower panel */}</div>
  <div className="lg:col-span-7">{/* Wider panel */}</div>
</div>
```

---

## 7. Data Layer Rules

### 7.1 Interfaces (Source of Truth: `lib/mockData.ts`)

ALL data shapes are defined in `lib/mockData.ts`. When the backend is connected, these interfaces MUST match the API response exactly.

| Interface | Fields | Used By |
|---|---|---|
| `BatchRecord` | id, timestamp, temperature, conveyorSpeed, holdTime, batchSize, materialType, hourOfDay, qualityScore, yieldPct, performancePct, energyKwh, status, anomalyScore | RecentBatches, BatchTable, LiveMonitor |
| `PredictionResult` | qualityScore, yieldPct, performancePct, energyKwh, confidence (tuples per target) | PreBatchPanel, dashboard |
| `ShapValue` | feature, contribution, direction | ShapChart |
| `AnomalyResult` | score, isAnomaly, threshold, diagnosis, diagnosisConfidence, humanReadable, recommendedAction, estimatedImpact | AnomalyDetector |

### 7.2 API Contract (Future Backend Integration)

When connecting to the FastAPI backend:

| Endpoint | Method | Frontend Consumer |
|---|---|---|
| `/predict/batch` | POST | PreBatchPanel |
| `/predict/realtime` | POST | LiveMonitor (every 30s) |
| `/anomaly/detect` | POST | AnomalyDetector |
| `/explain/{batch_id}` | GET | ShapChart |
| `/model/features` | GET | Dashboard feature importance |
| `/health` | GET | Sidebar system status |
| `/live/{batch_id}` | WebSocket | LiveMonitor real-time feed |

**CRITICAL:** If any API response shape changes, update ALL of the following:
1. The TypeScript interface in `mockData.ts`
2. Every component consuming that interface
3. Any mock data objects matching that interface
4. Any fallback/default objects in components

### 7.3 Defensive Data Access

ALWAYS guard against null/undefined:
```tsx
// CORRECT
const quality = batch?.qualityScore ?? 0;
const label = data?.diagnosis ?? "Unknown";

// WRONG — will crash on undefined
const quality = batch.qualityScore;
```

---

## 8. Domain Knowledge — Manufacturing Context

### Batch Parameters (Operator-Controlled Inputs)

| Parameter | Unit | Range | Optimal |
|---|---|---|---|
| temperature | °C | 175–195 | 183°C |
| conveyor_speed | % | 60–95 | 75% |
| hold_time | minutes | 10–30 | 18 min |
| batch_size | kg | 300–700 | — |
| material_type | enum | 0/1/2 | 0=TypeA, 1=TypeB, 2=TypeC |
| hour_of_day | int | 6–21 | — |
| operator_exp | enum | 0/1/2 | 0=junior, 1=mid, 2=senior |

### Prediction Targets (4 simultaneous outputs)

| Target | Unit | Typical Range |
|---|---|---|
| quality_score | % | 60–100 |
| yield_pct | % | 70–100 |
| performance_pct | % | 60–100 |
| energy_kwh | kWh | 25–55 |
| co2_kg (derived) | kg CO₂e | energy × 0.82 |

### Anomaly Fault Types

| Class | Pattern | Recommended Action |
|---|---|---|
| normal | Smooth power curve | No action |
| bearing_wear | Gradual baseline rise (+0.003 kW/s) | Schedule maintenance within 5 days |
| wet_material | Irregular spikes in first 600s | Extend drying phase by 3–4 minutes |
| calibration_needed | Elevated flat baseline (+0.6 kW) | Machine calibration required |

### Alert Severity Thresholds

| Level | Anomaly Score | Color | Action |
|---|---|---|---|
| Normal | 0.00–0.15 | Green | None |
| Watch | 0.15–0.30 | Amber | Monitor next batches |
| Warning | 0.30–0.60 | Orange | Investigate |
| Critical | 0.60+ | Red | Immediate intervention |

---

## 9. Post-Implementation Review Checklist

### 9.1 Code Quality (0–25 points)

- [ ] **Naming:** All new functions, variables, files follow conventions (PascalCase components, camelCase utilities).
- [ ] **Error handling:** Every async function has try/catch. Decide fail-open or fail-closed intentionally.
- [ ] **Comments:** Every new logic block has a comment explaining *what* and *why*.
- [ ] **Code change policy:** When replacing logic — comment out old code with explanation, add new code below with comment. Never silently delete working code.
- [ ] **Dead code:** Remove or comment out leftover references to replaced functions, models, or variables.
- [ ] **Consistency:** New code matches the style, indentation, and patterns of surrounding code.
- [ ] **DRY:** Extract and reuse duplicated logic. Use existing components before creating new ones.
- [ ] **Magic numbers:** Move hardcoded values to constants, theme tokens, or config.
- [ ] **Types:** All TypeScript params and returns are properly typed. No `any` unless explicitly justified.
- [ ] **`"use client"` directive:** Present on every component using hooks, event handlers, or browser APIs.

### 9.2 Cross-File Impact Analysis (0–25 points)

- [ ] **Trace dependencies:** For every change, check what else depends on it.
- [ ] **Interface sync:** If a data interface changes → update mockData.ts + every consumer component.
- [ ] **API shape sync:** If API response format changes → update interface, mock, every component, every fallback object.
- [ ] **Import check:** After renaming/moving files, update all import paths.
- [ ] **CSS variable check:** If changing globals.css variables → verify all components using them.
- [ ] **Tailwind config check:** If changing theme colors/fonts → verify no component breaks.

### 9.3 UI Consistency Audit (0–20 points)

- [ ] **Card styling:** Uses `bg-white rounded-xl border border-slate-100 p-4 card-hover`.
- [ ] **Text sizes:** Follows the typography scale defined in Section 6.1.
- [ ] **Spacing:** Card padding `p-3`/`p-4`, grid gap `gap-4`/`gap-5`.
- [ ] **Colors:** Only palette colors from tailwind.config.ts and CSS variables.
- [ ] **Empty states:** Show skeleton shimmer or "No data" message — never blank screens.
- [ ] **Responsive:** Must work on standard phone sizes (375px+) and tablets (768px+).
- [ ] **Component reuse:** Use existing components before creating new ones (check `components/` first).
- [ ] **`cn()` usage:** All conditional classes use `cn()` — no manual string concatenation.
- [ ] **Dynamic import:** All Recharts components use `dynamic(() => ..., { ssr: false })`.

### 9.4 Edge Case Analysis (0–20 points)

- [ ] **Data: 0 records** — Tables/lists show empty state, not crash.
- [ ] **Data: Large datasets** — Pagination or virtualization for 1000+ items.
- [ ] **Data: Missing fields** — All optional fields guarded with `?.` and `??`.
- [ ] **Data: Malformed input** — Form validation rejects out-of-range parameters (see domain ranges above).
- [ ] **Network: API down** — Show error state, not blank screen. Graceful degradation to mock data if applicable.
- [ ] **Network: Slow response** — Loading states (skeleton shimmer, spinner).
- [ ] **Network: WebSocket disconnect** — Reconnection logic with exponential backoff.
- [ ] **Security: Role gating** — Operator vs. supervisor access (future consideration).

### 9.5 Confidence Score (0–10 points: Test Readiness)

After all checks, assign a composite score:

| Category | Max Points |
|---|---|
| Code Quality | 25 |
| Cross-File Impact | 25 |
| UI Consistency | 20 |
| Edge Case Coverage | 20 |
| Test Readiness | 10 |
| **Total** | **100** |

**Verdict:**
- **SHIP IT** (90+): Merge-ready.
- **REVIEW NEEDED** (70–89): Address flagged items, then re-score.
- **REWORK** (below 70): Significant issues — re-plan.

---

## 10. Testing Procedure

### 10.1 Syntax & Type Check

```bash
cd frontend
npx tsc --noEmit
```
Must pass with zero errors before any task is marked complete.

### 10.2 Dev Server Smoke Test

```bash
cd frontend
npm run dev
```
Verify:
- All routes load without blank screens: `/`, `/predictions`, `/monitor`, `/anomalies`, `/history`
- No console errors in browser DevTools.
- Charts render (Recharts components).

### 10.3 Build Verification

```bash
cd frontend
npm run build
```
Must complete without errors. This catches SSR issues that `dev` mode misses.

### 10.4 Manual UI Checks

- [ ] All pages load without blank screens.
- [ ] Data tables/lists paginate correctly.
- [ ] Filters and controls work.
- [ ] Modals/panels open and close.
- [ ] Empty states render properly.
- [ ] Sidebar navigation highlights the active route.
- [ ] Header clock ticks every second.
- [ ] Status dots show correct colors.

---

## 11. AI Agent Mandatory Rules (Pitfall Prevention)

### 11.1 Version Consistency

Never change package versions unless asked. This includes:
- `package.json` dependencies and devDependencies
- Python `requirements.txt` versions (when backend exists)
- Node.js / Python runtime versions

### 11.2 API Response Shape ↔ Consumer Sync

When ANY data interface changes:
1. Update the TypeScript interface in `lib/mockData.ts`.
2. Update every mock data object matching that interface.
3. Update every component consuming the interface.
4. Update every fallback/default value.
5. Verify no `undefined` property access at runtime.

### 11.3 Defensive Data Access

```tsx
// ALWAYS use optional chaining + nullish coalescing
const value = data?.predictions?.energyKwh ?? 0;
const status = batch?.status ?? "unknown";
const items = results?.batches ?? [];
```

### 11.4 Fresh Rebuilds

Never copy `node_modules/` or `.next/` cache between tasks. If dependencies change:
```bash
rm -rf node_modules .next
npm install
npm run dev
```

### 11.5 Working Directory Awareness

Always verify `cwd` before running terminal commands. The frontend lives at `frontend/`, not the project root.

### 11.6 No Silent Failures

- Do not suppress errors with empty catch blocks.
- Always log errors: `console.error("[ComponentName]", error)`.
- Show user-visible error states, not blank screens.

### 11.7 Imports After Refactoring

After any rename, move, or delete:
- Grep for the old import path across the entire `src/` directory.
- Update every occurrence.
- Verify with `npx tsc --noEmit`.

---

## 12. Review Personas

When reviewing your own work, simulate these three reviewers:

### Security Architect
- Is user input validated before use?
- Are API calls protected against injection?
- Is sensitive data exposed in client-side code?
- Are auth boundaries respected?

### UX Designer
- Is the visual output consistent with the existing design system?
- Are loading states present for async operations?
- Are error messages user-friendly (not raw error strings)?
- Is the component accessible (semantic HTML, keyboard nav, ARIA labels)?

### QA Engineer
- What happens with empty data? Null data? Malformed data?
- What happens when the network call fails?
- Are boundary conditions handled (min/max values, 0 items, 10000 items)?
- Does the change introduce regression in any existing feature?

---

## 13. Self-Improvement Loop

- After ANY correction from the user: document the pattern in `tasks/lessons.md` (create if needed).
- Review lessons at the start of each session.
- Never make the same mistake twice.

---

## 14. Key Files Quick Reference

| What | Where |
|---|---|
| Root layout (shell) | `frontend/src/app/layout.tsx` |
| Dashboard page | `frontend/src/app/page.tsx` |
| Global styles + CSS vars | `frontend/src/app/globals.css` |
| Tailwind theme (colors) | `frontend/tailwind.config.ts` |
| Class merge utility | `frontend/src/lib/utils.ts` → `cn()` |
| All data interfaces + mocks | `frontend/src/lib/mockData.ts` |
| Sidebar navigation | `frontend/src/components/layout/Sidebar.tsx` |
| Header with clock | `frontend/src/components/layout/Header.tsx` |
| KPI stat cards | `frontend/src/components/dashboard/StatCard.tsx` |
| Energy trend chart | `frontend/src/components/dashboard/EnergyChart.tsx` |
| Performance gauges | `frontend/src/components/dashboard/PerformanceGauges.tsx` |
| Recent batches table | `frontend/src/components/dashboard/RecentBatches.tsx` |
| Shift overview panel | `frontend/src/components/dashboard/ShiftOverview.tsx` |
| Pre-batch prediction | `frontend/src/components/predictions/PreBatchPanel.tsx` |
| SHAP explanation chart | `frontend/src/components/predictions/ShapChart.tsx` |
| Live batch monitor | `frontend/src/components/monitor/LiveMonitor.tsx` |
| Anomaly detection UI | `frontend/src/components/anomalies/AnomalyDetector.tsx` |
| Batch history table | `frontend/src/components/history/BatchTable.tsx` |
| Full project spec | `README.md` (1595 lines — comprehensive) |

---

## 15. Component Creation Checklist

When creating a NEW component:

1. [ ] Check if an existing component can be reused or extended first.
2. [ ] Place in the correct subdirectory under `components/` matching its route/domain.
3. [ ] Add `"use client"` if it uses hooks, events, or browser APIs.
4. [ ] Use named export: `export function MyComponent() {}`.
5. [ ] Define a TypeScript interface for props.
6. [ ] Use `cn()` for all conditional class names.
7. [ ] Follow the card pattern for panel-type components.
8. [ ] Use Tabler icons (`Tb*`) — do not import from other icon sets.
9. [ ] Use `@/*` path alias for all imports.
10. [ ] Add empty state handling if the component displays data.
11. [ ] Add loading state if the component fetches data.
12. [ ] Test with `npx tsc --noEmit` after creation.

---

## 16. Backend Development Rules (When Building Backend)

When the backend is developed:

### File Structure (Per README Spec)

```
backend/
├── main.py                   ← FastAPI entry point
├── requirements.txt
├── data/                     ← Data generation + CSV + power curves
├── preprocessing/            ← 4-stage pipeline (imputer, outlier, feature eng, normalizer)
├── models/                   ← ML models + trained/ artifacts
├── explainability/           ← SHAP + plain English converter
├── api/
│   ├── schemas.py            ← Pydantic request/response models
│   └── routes/               ← predict, anomaly, explain, recommend, health
└── database/                 ← SQLAlchemy + SQLite
```

### Backend Rules

- Every endpoint must have Pydantic request/response validation.
- All async handlers must have try/except with proper error responses.
- Models are loaded ONCE at startup, not per-request.
- CORS must be configured for `http://localhost:3000` (Next.js dev server).
- Every prediction must return confidence intervals.
- Every anomaly detection must return a human-readable explanation.
- SHAP explanations must include plain-English summaries.
- WebSocket endpoints must handle connection drops gracefully.

---

## 17. Git & Commit Rules

- Commit messages follow: `type(scope): description`
  - Types: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`
  - Scope: `dashboard`, `predictions`, `monitor`, `anomalies`, `history`, `layout`, `backend`, `data`
  - Example: `feat(predictions): add pre-batch parameter form with validation`
- Never commit `node_modules/`, `.next/`, `__pycache__/`, `venv/`, `*.pyc`, model artifacts (`*.pkl`, `*.pt`).
- Pull before pushing. Resolve conflicts locally.

---

## 18. Performance Rules

- Lazy-load chart components with `dynamic(() => ..., { ssr: false })`.
- Avoid re-rendering entire pages — isolate state to the smallest component possible.
- For lists > 100 items, implement pagination or virtual scrolling.
- Images: use Next.js `<Image>` component with proper width/height.
- Memoize expensive computations with `useMemo` and callbacks with `useCallback` where measurable benefit exists.
- Do NOT prematurely optimize — measure first, optimize second.

---

*Last updated: March 5, 2026*
*Applies to: PlantIQ Manufacturing Intelligence Platform*
*Stack: Next.js 14 + React 18 + TypeScript + Tailwind CSS 3 (frontend) | FastAPI + XGBoost + PyTorch (backend)*
