/**
 * PlantIQ — Custom React Hooks
 * ================================
 * Reusable hooks for the PlantIQ dashboard.
 */

import { useState, useEffect, useRef } from "react";

/**
 * Debounce a value by `delay` ms.
 * Returns the debounced value that updates only after the caller
 * stops changing the input for `delay` milliseconds.
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debounced;
}

/**
 * Returns a stable reference that is `true` after the component
 * has mounted (useful for skipping the first effect call).
 */
export function useIsMounted(): boolean {
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
  }, []);
  return mounted.current;
}
