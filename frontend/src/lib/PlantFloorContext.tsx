"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import type { ReactNode } from "react";

interface PlantFloorContextValue {
  /** Whether Plant Floor Mode is active (larger fonts for shop-floor displays) */
  isPlantFloorMode: boolean;
  /** Toggle Plant Floor Mode on/off */
  togglePlantFloorMode: () => void;
}

const PlantFloorContext = createContext<PlantFloorContextValue>({
  isPlantFloorMode: false,
  togglePlantFloorMode: () => {},
});

const STORAGE_KEY = "plantiq-floor-mode";

export function PlantFloorProvider({ children }: { children: ReactNode }) {
  const [isPlantFloorMode, setIsPlantFloorMode] = useState(false);

  // Hydrate from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "true") {
        setIsPlantFloorMode(true);
      }
    } catch {
      // SSR or storage unavailable — ignore
    }
  }, []);

  // Sync to localStorage on change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(isPlantFloorMode));
    } catch {
      // Storage unavailable — ignore
    }
  }, [isPlantFloorMode]);

  const togglePlantFloorMode = useCallback(() => {
    setIsPlantFloorMode((prev) => !prev);
  }, []);

  return (
    <PlantFloorContext.Provider value={{ isPlantFloorMode, togglePlantFloorMode }}>
      {children}
    </PlantFloorContext.Provider>
  );
}

export function usePlantFloorMode() {
  return useContext(PlantFloorContext);
}
