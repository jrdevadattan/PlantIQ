"use client";

import { cn } from "@/lib/utils";
import { PlantFloorProvider, usePlantFloorMode } from "@/lib/PlantFloorContext";
import type { ReactNode } from "react";

/** Inner shell that reads context and applies the CSS class */
function ShellInner({ children }: { children: ReactNode }) {
  const { isPlantFloorMode } = usePlantFloorMode();

  return (
    <div
      className={cn(
        "flex h-screen bg-slate-50 overflow-hidden",
        isPlantFloorMode && "plant-floor-mode"
      )}
    >
      {children}
    </div>
  );
}

/** Wraps app with PlantFloorProvider + class-toggling root div */
export function PlantFloorShell({ children }: { children: ReactNode }) {
  return (
    <PlantFloorProvider>
      <ShellInner>{children}</ShellInner>
    </PlantFloorProvider>
  );
}
