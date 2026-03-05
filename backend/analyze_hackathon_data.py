"""Analyze hackathon Excel files to understand data structure."""
import pandas as pd

ps_dir = r"c:\Users\harir\Downloads\ALL\projects\plantiq\69997ffba83f5_problem_statement"

# === Process Data ===
df1 = pd.read_excel(f"{ps_dir}/_h_batch_process_data.xlsx")
print("=" * 80)
print("FILE 1: _h_batch_process_data.xlsx")
print("=" * 80)
print(f"Shape: {df1.shape}")
batch_ids = sorted(df1["Batch_ID"].unique())
print(f"Unique batch IDs ({len(batch_ids)}): {batch_ids}")

for bid in batch_ids[:3]:
    sub = df1[df1["Batch_ID"] == bid]
    print(f"  {bid}: {sub['Time_Minutes'].min()}-{sub['Time_Minutes'].max()} min, {len(sub)} rows")
    print(f"    Phases: {list(sub['Phase'].unique())}")

print(f"\nAll unique phases: {sorted(df1['Phase'].unique())}")
print(f"\nNull counts:\n{df1.isnull().sum()}")

# === Production Data ===
df2 = pd.read_excel(f"{ps_dir}/_h_batch_production_data.xlsx")
print("\n" + "=" * 80)
print("FILE 2: _h_batch_production_data.xlsx")
print("=" * 80)
print(f"Shape: {df2.shape}")
batch_ids2 = sorted(df2["Batch_ID"].unique())
print(f"Unique batch IDs ({len(batch_ids2)}): {batch_ids2[:15]}...")
print(f"\nNull counts:\n{df2.isnull().sum()}")

# === Overlap ===
common = set(batch_ids) & set(batch_ids2)
only_process = set(batch_ids) - set(batch_ids2)
only_prod = set(batch_ids2) - set(batch_ids)
print(f"\n=== BATCH OVERLAP ===")
print(f"Common: {len(common)}")
print(f"Only in process: {len(only_process)} -> {sorted(only_process)[:5]}")
print(f"Only in production: {len(only_prod)} -> {sorted(only_prod)[:5]}")

# === Quality Metrics ===
print("\n=== QUALITY METRICS (Production Data) ===")
quality_cols = ["Moisture_Content", "Tablet_Weight", "Hardness", "Friability",
                "Disintegration_Time", "Dissolution_Rate", "Content_Uniformity"]
for col in quality_cols:
    print(f"  {col}: {df2[col].min():.2f} - {df2[col].max():.2f} (mean: {df2[col].mean():.2f})")

# === Input Parameters ===
print("\n=== INPUT PARAMETERS (Production Data) ===")
input_cols = ["Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
              "Compression_Force", "Machine_Speed", "Lubricant_Conc"]
for col in input_cols:
    print(f"  {col}: {df2[col].min():.2f} - {df2[col].max():.2f} (mean: {df2[col].mean():.2f})")

# === Process Data per Phase ===
print("\n=== PROCESS DATA BY PHASE ===")
for phase in sorted(df1["Phase"].unique()):
    sub = df1[df1["Phase"] == phase]
    print(f"\n  Phase: {phase} ({len(sub)} rows)")
    for col in ["Temperature_C", "Pressure_Bar", "Motor_Speed_RPM", "Power_Consumption_kW", "Vibration_mm_s"]:
        print(f"    {col}: {sub[col].min():.2f} - {sub[col].max():.2f}")

# === Correlation between production inputs and outputs ===
print("\n=== CORRELATION: Inputs vs Outputs (Production Data) ===")
corr = df2[input_cols + quality_cols].corr()
for qcol in quality_cols:
    top_corr = corr[qcol][input_cols].abs().sort_values(ascending=False)
    print(f"  {qcol}: top input = {top_corr.index[0]} (r={corr[qcol][top_corr.index[0]]:.3f})")
