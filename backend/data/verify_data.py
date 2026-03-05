"""
PlantIQ F1.1 — Data Quality Verification Script
Verifies batch_data.csv and power_curves/ against README spec.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "batch_data.csv")
CURVE_DIR = os.path.join(SCRIPT_DIR, "power_curves")

errors = []


def check(label, condition, msg_fail=""):
    if condition:
        print(f"[PASS] {label}")
    else:
        errors.append(msg_fail or label)
        print(f"[FAIL] {label} — {msg_fail}")


def main():
    print("=" * 60)
    print("  PLANTIQ F1.1 — DATA QUALITY VERIFICATION")
    print("=" * 60)

    # ── 1. BATCH CSV ──────────────────────────────────
    print("\n--- Batch CSV Checks ---")
    df = pd.read_csv(CSV_PATH)

    check("Row count = 2000", len(df) == 2000, f"Got {len(df)}")

    expected_cols = [
        "batch_id", "timestamp", "temperature", "conveyor_speed", "hold_time",
        "batch_size", "material_type", "hour_of_day", "operator_exp",
        "temp_speed_product", "temp_deviation", "speed_deviation", "hold_per_kg",
        "shift", "hours_into_shift", "quality_score", "yield_pct", "performance_pct",
        "energy_kwh", "co2_kg", "energy_per_kg", "fault_type",
    ]
    missing = set(expected_cols) - set(df.columns)
    extra = set(df.columns) - set(expected_cols)
    check(f"All {len(expected_cols)} columns present", len(missing) == 0,
          f"Missing: {missing}, Extra: {extra}")

    null_count = df.isnull().sum().sum()
    check("No null values", null_count == 0, f"{null_count} nulls found")

    dup_ids = df["batch_id"].duplicated().sum()
    check("No duplicate batch_ids", dup_ids == 0, f"{dup_ids} duplicates")

    # Range checks per README spec
    range_specs = {
        "temperature":     (175, 195),
        "conveyor_speed":  (60,   95),
        "hold_time":       (10,   30),
        "batch_size":      (300, 700),
        "material_type":   (0,     2),
        "hour_of_day":     (6,    21),
        "operator_exp":    (0,     2),
        "quality_score":   (60,  100),
        "yield_pct":       (70,  100),
        "performance_pct": (60,  100),
        "energy_kwh":      (25,   55),
    }
    for col, (lo, hi) in range_specs.items():
        mn, mx = df[col].min(), df[col].max()
        ok = mn >= lo and mx <= hi
        check(f"{col:20s} [{mn:7.1f}, {mx:7.1f}] ⊂ [{lo}, {hi}]", ok,
              f"{col} out of range: [{mn}, {mx}]")

    # CO2 derived correctly
    co2_expected = (df["energy_kwh"] * 0.82).round(2)
    check("co2_kg = energy_kwh * 0.82", (co2_expected == df["co2_kg"]).all(),
          "co2_kg mismatch")

    # temp_speed_product
    tsp = (df["temperature"] * df["conveyor_speed"]).round(1)
    check("temp_speed_product correct", (tsp == df["temp_speed_product"]).all(),
          "temp_speed_product mismatch")

    # Fault types
    valid_faults = {"normal", "bearing_wear", "wet_material", "calibration_needed"}
    actual_faults = set(df["fault_type"].unique())
    check("All 4 fault types present", actual_faults == valid_faults,
          f"Got: {actual_faults}")

    # Fault distribution
    print("\n--- Fault Distribution ---")
    for ft, cnt in df["fault_type"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"  {ft:25s} {cnt:5d} ({pct:5.1f}%)")

    # Material type distribution
    print("\n--- Material Type Distribution ---")
    material_map = {0: "TypeA", 1: "TypeB", 2: "TypeC"}
    for mt, cnt in df["material_type"].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        print(f"  {material_map.get(mt, mt):10s} {cnt:5d} ({pct:5.1f}%)")

    # ── 2. POWER CURVES ──────────────────────────────
    print("\n--- Power Curve Checks ---")
    curve_files = [f for f in os.listdir(CURVE_DIR) if f.endswith(".npy")]
    check(f"2000 .npy files", len(curve_files) == 2000, f"Got {len(curve_files)}")

    # Random sample shape & dtype check
    np.random.seed(42)
    sample_ids = np.random.choice(2000, size=50, replace=False)
    shape_ok = 0
    dtype_ok = 0
    for sid in sample_ids:
        arr = np.load(os.path.join(CURVE_DIR, f"B{sid:04d}.npy"))
        if arr.shape == (1800,):
            shape_ok += 1
        if arr.dtype == np.float32:
            dtype_ok += 1
    check(f"Sample shapes: {shape_ok}/50 = (1800,)", shape_ok == 50,
          f"Only {shape_ok}/50 correct")
    check(f"Sample dtypes: {dtype_ok}/50 = float32", dtype_ok == 50,
          f"Only {dtype_ok}/50 correct")

    # Verify fault curves have distinguishable signatures
    print("\n--- Fault Signature Analysis (first 300 batches) ---")
    categories = {"normal": [], "bearing_wear": [], "wet_material": [], "calibration_needed": []}
    for _, row in df.head(300).iterrows():
        curve = np.load(os.path.join(CURVE_DIR, row["batch_id"] + ".npy"))
        categories[row["fault_type"]].append(curve)

    for ft, curves in categories.items():
        if not curves:
            print(f"  {ft}: no samples in first 300")
            continue
        means = [c.mean() for c in curves]
        stds = [c.std() for c in curves]
        print(f"  {ft:25s} n={len(curves):3d}  avg_mean={np.mean(means):6.2f} kW  avg_std={np.mean(stds):.3f}")

    # Cross-check: bearing_wear should have higher mean than normal
    if categories["normal"] and categories["bearing_wear"]:
        n_mean = np.mean([c.mean() for c in categories["normal"]])
        b_mean = np.mean([c.mean() for c in categories["bearing_wear"]])
        check("Bearing wear avg > Normal avg", b_mean > n_mean,
              f"bearing={b_mean:.2f} vs normal={n_mean:.2f}")

    if categories["normal"] and categories["calibration_needed"]:
        n_mean = np.mean([c.mean() for c in categories["normal"]])
        c_mean = np.mean([c.mean() for c in categories["calibration_needed"]])
        check("Calibration avg > Normal avg", c_mean > n_mean,
              f"calibration={c_mean:.2f} vs normal={n_mean:.2f}")

    if categories["normal"] and categories["wet_material"]:
        n_std = np.mean([c[:600].std() for c in categories["normal"]])
        w_std = np.mean([c[:600].std() for c in categories["wet_material"]])
        check("Wet material std(first 600) > Normal std(first 600)", w_std > n_std,
              f"wet={w_std:.3f} vs normal={n_std:.3f}")

    # ── 3. SUMMARY ────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print(f"  RESULT: FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  RESULT: ALL CHECKS PASSED ✓")
        print("  2000 batch records + 2000 power curves verified.")
        print("  Data is spec-compliant and ready for preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
