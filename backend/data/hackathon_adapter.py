"""
PlantIQ — Hackathon Data Adapter
==================================
Loads, preprocesses, and adapts the hackathon Excel data files for use
with the PlantIQ ML pipeline. Handles both:

  1. _h_batch_production_data.xlsx — 60 batches, 7 inputs → 7 quality outputs
     (pharmaceutical tablet manufacturing)
  2. _h_batch_process_data.xlsx — Time-series sensor data per batch
     (211 rows for batch T001, 8 manufacturing phases)

This module provides:
  - HackathonDataAdapter: loads, cleans, and prepares hackathon data
  - train_on_hackathon_data(): trains XGBoost on real production data
  - analyze_process_data(): analyses time-series process data for anomalies
  - Feature engineers domain-specific derived features for pharma manufacturing

Domain: Pharmaceutical Tablet Manufacturing
  Inputs: Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time,
          Compression_Force, Machine_Speed, Lubricant_Conc
  Quality Outputs: Moisture_Content, Tablet_Weight, Hardness, Friability,
                   Disintegration_Time, Dissolution_Rate, Content_Uniformity
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional

# Ensure backend is importable
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# ──────────────────────────────────────────────
# Constants — Hackathon Data Schema
# ──────────────────────────────────────────────
HACKATHON_DIR = os.path.join(
    os.path.dirname(BACKEND_DIR),
    "69997ffba83f5_problem_statement"
)

PRODUCTION_FILE = "_h_batch_production_data.xlsx"
PROCESS_FILE = "_h_batch_process_data.xlsx"

# Input features (operator-controlled) from production data
HACKATHON_INPUT_COLS = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
    "Compression_Force", "Machine_Speed", "Lubricant_Conc",
]

# Quality output targets from production data
HACKATHON_TARGET_COLS = [
    "Moisture_Content", "Tablet_Weight", "Hardness", "Friability",
    "Disintegration_Time", "Dissolution_Rate", "Content_Uniformity",
]

# Primary targets (mapped to hackathon): Quality → Content_Uniformity,
# Yield → Dissolution_Rate, Performance → Hardness, Energy → Power_Consumption
PRIMARY_TARGETS = {
    "quality": "Content_Uniformity",      # Quality metric
    "yield": "Dissolution_Rate",          # Output optimization
    "performance": "Hardness",            # Performance metric (higher = better)
    "energy_proxy": "Friability",         # Lower is better (like energy waste)
}

# Process data sensor columns
PROCESS_SENSOR_COLS = [
    "Temperature_C", "Pressure_Bar", "Humidity_Percent", "Motor_Speed_RPM",
    "Compression_Force_kN", "Flow_Rate_LPM", "Power_Consumption_kW",
    "Vibration_mm_s",
]

# Manufacturing phases in order
MANUFACTURING_PHASES = [
    "Preparation", "Granulation", "Drying", "Milling",
    "Blending", "Compression", "Coating", "Quality_Testing",
]

# Optimal ranges for quality (pharmaceutical standards)
QUALITY_SPECS = {
    "Moisture_Content":    {"min": 1.0, "max": 3.0, "unit": "%",     "optimal": 2.0},
    "Tablet_Weight":       {"min": 198, "max": 202, "unit": "mg",    "optimal": 200},
    "Hardness":            {"min": 80,  "max": 120, "unit": "N",     "optimal": 100},
    "Friability":          {"min": 0.0, "max": 1.0, "unit": "%",     "optimal": 0.5},
    "Disintegration_Time": {"min": 5,   "max": 15,  "unit": "min",   "optimal": 10},
    "Dissolution_Rate":    {"min": 85,  "max": 100, "unit": "%",     "optimal": 92},
    "Content_Uniformity":  {"min": 95,  "max": 105, "unit": "%",     "optimal": 100},
}

# CO2 emission factor (kg CO2e per kWh)
CO2_FACTOR = 0.82


class HackathonDataAdapter:
    """Adapter that loads and prepares hackathon Excel data for the PlantIQ pipeline.

    Provides methods to:
      - Load production data (batch-level: inputs → quality outputs)
      - Load process data (time-series sensor readings)
      - Engineer derived features for pharmaceutical domain
      - Compute quality scores and compliance metrics
      - Prepare data for XGBoost multi-target prediction
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize with path to hackathon data directory.

        Parameters
        ----------
        data_dir : str, optional
            Path to the directory containing hackathon Excel files.
            Defaults to the ``69997ffba83f5_problem_statement`` folder.
        """
        self.data_dir = data_dir or HACKATHON_DIR
        self._production_df: Optional[pd.DataFrame] = None
        self._process_df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────

    def load_production_data(self) -> pd.DataFrame:
        """Load and validate the batch production data (60 batches × 15 columns).

        Returns
        -------
        pd.DataFrame
            Production data with Batch_ID as index — 7 input + 7 quality columns.
        """
        path = os.path.join(self.data_dir, PRODUCTION_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Production data not found: {path}\n"
                f"Expected hackathon Excel file: {PRODUCTION_FILE}"
            )

        df = pd.read_excel(path)
        print(f"[HackathonAdapter] Loaded production data: {df.shape[0]} batches × {df.shape[1]} columns")

        # Validate required columns
        missing_inputs = [c for c in HACKATHON_INPUT_COLS if c not in df.columns]
        missing_targets = [c for c in HACKATHON_TARGET_COLS if c not in df.columns]
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        self._production_df = df
        return df

    def load_process_data(self) -> pd.DataFrame:
        """Load the batch process time-series data (211 rows for batch T001).

        Returns
        -------
        pd.DataFrame
            Time-series data with Time_Minutes, Phase, and 8 sensor columns.
        """
        path = os.path.join(self.data_dir, PROCESS_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Process data not found: {path}\n"
                f"Expected hackathon Excel file: {PROCESS_FILE}"
            )

        df = pd.read_excel(path)
        print(f"[HackathonAdapter] Loaded process data: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Batches: {df['Batch_ID'].nunique()}, Phases: {df['Phase'].nunique()}")

        self._process_df = df
        return df

    # ──────────────────────────────────────────────
    # Feature Engineering — Pharmaceutical Domain
    # ──────────────────────────────────────────────

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create derived features for pharmaceutical manufacturing.

        Derived features:
          1. granulation_intensity — Granulation_Time × Binder_Amount
          2. drying_intensity     — Drying_Temp × Drying_Time
          3. compression_speed_ratio — Compression_Force / Machine_Speed
          4. binder_per_time      — Binder_Amount / Granulation_Time
          5. drying_efficiency    — Drying_Temp / Drying_Time
          6. force_lubricant_ratio — Compression_Force / Lubricant_Conc
          7. machine_load_index   — Machine_Speed × Compression_Force

        Parameters
        ----------
        df : pd.DataFrame, optional
            Production data. Uses cached data if not provided.

        Returns
        -------
        pd.DataFrame
            DataFrame with 7 additional derived feature columns.
        """
        if df is None:
            df = self._production_df
        if df is None:
            raise ValueError("No production data loaded. Call load_production_data() first.")

        df = df.copy()
        created = []

        # 1. Granulation intensity — interaction of time and binder
        df["granulation_intensity"] = (df["Granulation_Time"] * df["Binder_Amount"]).round(2)
        created.append("granulation_intensity")

        # 2. Drying intensity — heat × time
        df["drying_intensity"] = (df["Drying_Temp"] * df["Drying_Time"]).round(2)
        created.append("drying_intensity")

        # 3. Compression speed ratio — force per unit speed
        df["compression_speed_ratio"] = (
            df["Compression_Force"] / df["Machine_Speed"].clip(lower=1)
        ).round(4)
        created.append("compression_speed_ratio")

        # 4. Binder per unit time
        df["binder_per_time"] = (
            df["Binder_Amount"] / df["Granulation_Time"].clip(lower=1)
        ).round(4)
        created.append("binder_per_time")

        # 5. Drying efficiency — temp per unit time
        df["drying_efficiency"] = (
            df["Drying_Temp"] / df["Drying_Time"].clip(lower=1)
        ).round(4)
        created.append("drying_efficiency")

        # 6. Force lubricant ratio — compression adjusted for lubricant
        df["force_lubricant_ratio"] = (
            df["Compression_Force"] / df["Lubricant_Conc"].clip(lower=0.01)
        ).round(4)
        created.append("force_lubricant_ratio")

        # 7. Machine load index — speed × force interaction
        df["machine_load_index"] = (df["Machine_Speed"] * df["Compression_Force"]).round(2)
        created.append("machine_load_index")

        print(f"[HackathonAdapter] Engineered {len(created)} derived features: {created}")
        return df

    # ──────────────────────────────────────────────
    # Quality Compliance Scoring
    # ──────────────────────────────────────────────

    def compute_quality_compliance(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute pharmaceutical quality compliance scores per batch.

        For each quality metric, calculates:
          - in_spec: bool — whether value is within pharmaceutical spec limits
          - deviation_pct: float — percentage deviation from optimal value
          - composite_quality_score: float — weighted overall quality (0–100)

        Parameters
        ----------
        df : pd.DataFrame, optional
            Production data with quality columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional compliance columns.
        """
        if df is None:
            df = self._production_df
        if df is None:
            raise ValueError("No production data loaded.")

        df = df.copy()
        compliance_scores = []

        for col, spec in QUALITY_SPECS.items():
            if col not in df.columns:
                continue

            # Check if within spec limits
            df[f"{col}_in_spec"] = (
                (df[col] >= spec["min"]) & (df[col] <= spec["max"])
            )

            # Deviation from optimal (percentage)
            df[f"{col}_deviation_pct"] = (
                abs(df[col] - spec["optimal"]) / spec["optimal"] * 100
            ).round(2)

            # Compliance score (0–1): 1.0 if at optimal, decreasing with deviation
            spec_range = spec["max"] - spec["min"]
            if spec_range > 0:
                score = 1.0 - abs(df[col] - spec["optimal"]) / spec_range
                compliance_scores.append(score.clip(lower=0))

        # Composite quality score (weighted average of all quality metrics)
        if compliance_scores:
            df["composite_quality_score"] = (
                sum(compliance_scores) / len(compliance_scores) * 100
            ).round(2)

        # Count how many specs are met per batch
        spec_cols = [c for c in df.columns if c.endswith("_in_spec")]
        df["specs_met"] = df[spec_cols].sum(axis=1)
        df["total_specs"] = len(spec_cols)

        print(f"[HackathonAdapter] Quality compliance computed: "
              f"avg composite score = {df['composite_quality_score'].mean():.1f}%")
        return df

    # ──────────────────────────────────────────────
    # Data Preparation for ML
    # ──────────────────────────────────────────────

    def prepare_for_training(
        self,
        df: Optional[pd.DataFrame] = None,
        targets: Optional[list[str]] = None,
        test_size: float = 0.2,
    ) -> dict:
        """Prepare production data for XGBoost multi-target training.

        Runs feature engineering, compliance scoring, and train/test split.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Production data. Loads from file if not provided.
        targets : list[str], optional
            Target columns to predict. Defaults to the 4 primary targets.
        test_size : float
            Fraction of data for testing (default 0.2 = 12 batches).

        Returns
        -------
        dict
            Keys: X_train, X_test, y_train, y_test, feature_cols, target_cols,
                  full_df, scaler_info
        """
        if df is None:
            df = self.load_production_data()

        # Engineer features
        df = self.engineer_features(df)
        df = self.compute_quality_compliance(df)

        # Define feature and target columns
        derived_cols = [
            "granulation_intensity", "drying_intensity", "compression_speed_ratio",
            "binder_per_time", "drying_efficiency", "force_lubricant_ratio",
            "machine_load_index",
        ]
        feature_cols = HACKATHON_INPUT_COLS + derived_cols

        if targets is None:
            # Default: predict 4 primary quality metrics
            targets = ["Hardness", "Dissolution_Rate", "Content_Uniformity", "Friability"]

        # Validate all columns exist
        X = df[feature_cols].values
        y = df[targets].values

        # Train/test split (stratified if possible)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Normalize features (StandardScaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"[HackathonAdapter] Prepared for training:")
        print(f"  Features: {len(feature_cols)} ({len(HACKATHON_INPUT_COLS)} raw + {len(derived_cols)} derived)")
        print(f"  Targets: {targets}")
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "X_train_raw": X_train,
            "X_test_raw": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "target_cols": targets,
            "full_df": df,
            "scaler": scaler,
        }

    # ──────────────────────────────────────────────
    # Process (Time-Series) Data Analysis
    # ──────────────────────────────────────────────

    def analyze_process_phases(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Analyse process data by manufacturing phase.

        Computes per-phase statistics for all sensor readings,
        identifies phase transitions, and calculates energy consumption.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Process time-series data.

        Returns
        -------
        dict
            Phase-by-phase statistics, energy breakdown, and transitions.
        """
        if df is None:
            df = self._process_df
        if df is None:
            df = self.load_process_data()

        result = {
            "phases": {},
            "total_energy_kwh": 0.0,
            "phase_energy_breakdown": {},
            "transitions": [],
            "anomaly_indicators": [],
        }

        for phase in MANUFACTURING_PHASES:
            phase_data = df[df["Phase"] == phase]
            if phase_data.empty:
                continue

            stats = {}
            for col in PROCESS_SENSOR_COLS:
                if col in phase_data.columns:
                    stats[col] = {
                        "mean": round(phase_data[col].mean(), 3),
                        "std": round(phase_data[col].std(), 3),
                        "min": round(phase_data[col].min(), 3),
                        "max": round(phase_data[col].max(), 3),
                    }

            # Phase duration (minutes)
            stats["duration_min"] = int(
                phase_data["Time_Minutes"].max() - phase_data["Time_Minutes"].min() + 1
            )

            # Energy for this phase (kWh = kW × hours)
            power_readings = phase_data["Power_Consumption_kW"].values
            duration_hours = stats["duration_min"] / 60.0
            avg_power = phase_data["Power_Consumption_kW"].mean()
            phase_energy = avg_power * duration_hours
            stats["energy_kwh"] = round(phase_energy, 3)

            result["phases"][phase] = stats
            result["total_energy_kwh"] += phase_energy
            result["phase_energy_breakdown"][phase] = round(phase_energy, 3)

        result["total_energy_kwh"] = round(result["total_energy_kwh"], 3)
        result["total_co2_kg"] = round(result["total_energy_kwh"] * CO2_FACTOR, 3)

        # Detect phase transitions (large jumps in sensor readings)
        for i in range(1, len(df)):
            curr_phase = df.iloc[i]["Phase"]
            prev_phase = df.iloc[i - 1]["Phase"]
            if curr_phase != prev_phase:
                result["transitions"].append({
                    "from": prev_phase,
                    "to": curr_phase,
                    "at_minute": int(df.iloc[i]["Time_Minutes"]),
                    "power_jump_kw": round(
                        df.iloc[i]["Power_Consumption_kW"] - df.iloc[i - 1]["Power_Consumption_kW"], 3
                    ),
                })

        # Anomaly indicators (high vibration or power spikes)
        if "Vibration_mm_s" in df.columns:
            vibration_threshold = df["Vibration_mm_s"].mean() + 2 * df["Vibration_mm_s"].std()
            high_vibration = df[df["Vibration_mm_s"] > vibration_threshold]
            for _, row in high_vibration.iterrows():
                result["anomaly_indicators"].append({
                    "type": "high_vibration",
                    "minute": int(row["Time_Minutes"]),
                    "phase": row["Phase"],
                    "value": round(row["Vibration_mm_s"], 3),
                    "threshold": round(vibration_threshold, 3),
                })

        print(f"[HackathonAdapter] Process analysis complete:")
        print(f"  Total energy: {result['total_energy_kwh']:.2f} kWh")
        print(f"  Total CO₂: {result['total_co2_kg']:.2f} kg")
        print(f"  Phases: {len(result['phases'])}")
        print(f"  Transitions: {len(result['transitions'])}")
        print(f"  Anomaly indicators: {len(result['anomaly_indicators'])}")

        return result

    def get_power_curve(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract the Power_Consumption_kW column as a time-series array.

        This provides a power curve similar to our synthetic .npy curves,
        suitable for LSTM Autoencoder anomaly detection.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Process time-series data.

        Returns
        -------
        np.ndarray
            Power consumption values as a 1D array (211 timesteps).
        """
        if df is None:
            df = self._process_df
        if df is None:
            df = self.load_process_data()

        return df["Power_Consumption_kW"].values

    # ──────────────────────────────────────────────
    # Energy Pattern Attribution
    # ──────────────────────────────────────────────

    def attribute_energy_patterns(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Analyse energy consumption patterns and attribute changes to
        asset parameters vs process parameters.

        Per the problem statement: "the change in consumption pattern can be
        attributed to a specific set of asset parameters implying predictive
        maintenance or calibration, or it could be explained due to changes
        to process parameters."

        Parameters
        ----------
        df : pd.DataFrame, optional
            Process time-series data.

        Returns
        -------
        dict
            Energy attribution analysis with:
              - per_phase: energy breakdown by phase
              - asset_indicators: vibration, motor anomalies suggest asset issues
              - process_indicators: temperature, pressure suggest process changes
              - recommendations: maintenance/calibration actions
        """
        if df is None:
            df = self._process_df
        if df is None:
            df = self.load_process_data()

        attribution = {
            "per_phase_energy": {},
            "asset_health_indicators": [],
            "process_deviation_indicators": [],
            "overall_attribution": "normal",
            "recommendations": [],
        }

        for phase in MANUFACTURING_PHASES:
            phase_data = df[df["Phase"] == phase]
            if phase_data.empty:
                continue

            avg_power = phase_data["Power_Consumption_kW"].mean()
            avg_vibration = phase_data["Vibration_mm_s"].mean()
            power_cv = phase_data["Power_Consumption_kW"].std() / max(avg_power, 0.01)

            attribution["per_phase_energy"][phase] = {
                "avg_power_kw": round(avg_power, 2),
                "energy_kwh": round(avg_power * len(phase_data) / 60, 3),
                "vibration_avg": round(avg_vibration, 3),
                "power_variability": round(power_cv, 4),
            }

            # --- Asset Health Indicators ---
            # High vibration suggests bearing wear or mechanical issues
            if avg_vibration > 5.0:
                attribution["asset_health_indicators"].append({
                    "phase": phase,
                    "indicator": "high_vibration",
                    "value": round(avg_vibration, 3),
                    "severity": "warning" if avg_vibration < 8.0 else "critical",
                    "attribution": "Elevated vibration suggests mechanical wear "
                                   "(bearing degradation or alignment issues)",
                    "action": "Schedule predictive maintenance inspection",
                })

            # High power variability suggests calibration needed
            if power_cv > 0.15:
                attribution["asset_health_indicators"].append({
                    "phase": phase,
                    "indicator": "power_instability",
                    "value": round(power_cv, 4),
                    "severity": "watch",
                    "attribution": "Unstable power consumption indicates potential "
                                   "calibration drift or sensor inconsistency",
                    "action": "Machine calibration check recommended",
                })

            # --- Process Deviation Indicators ---
            # Temperature deviations from expected ranges
            if "Temperature_C" in phase_data.columns:
                temp_range = phase_data["Temperature_C"].max() - phase_data["Temperature_C"].min()
                if temp_range > 10:
                    attribution["process_deviation_indicators"].append({
                        "phase": phase,
                        "indicator": "temperature_instability",
                        "value": round(temp_range, 2),
                        "attribution": "Wide temperature swing suggests process "
                                       "control issues (heater or cooling variance)",
                        "action": "Review temperature PID controller settings",
                    })

            # Pressure deviations
            if "Pressure_Bar" in phase_data.columns:
                pressure_cv = phase_data["Pressure_Bar"].std() / max(
                    phase_data["Pressure_Bar"].mean(), 0.01
                )
                if pressure_cv > 0.1:
                    attribution["process_deviation_indicators"].append({
                        "phase": phase,
                        "indicator": "pressure_instability",
                        "value": round(pressure_cv, 4),
                        "attribution": "Pressure fluctuations may indicate raw material "
                                       "moisture variation or valve issues",
                        "action": "Check raw material moisture and pressure valves",
                    })

        # Determine overall attribution
        n_asset = len(attribution["asset_health_indicators"])
        n_process = len(attribution["process_deviation_indicators"])

        if n_asset > n_process and n_asset > 0:
            attribution["overall_attribution"] = "asset_degradation"
            attribution["recommendations"].append(
                "Primary energy pattern changes attributed to ASSET factors. "
                "Schedule predictive maintenance within 5 days."
            )
        elif n_process > n_asset and n_process > 0:
            attribution["overall_attribution"] = "process_deviation"
            attribution["recommendations"].append(
                "Primary energy pattern changes attributed to PROCESS factors. "
                "Review operating parameters and raw material quality."
            )
        elif n_asset > 0 and n_process > 0:
            attribution["overall_attribution"] = "mixed"
            attribution["recommendations"].append(
                "Energy patterns show BOTH asset and process deviations. "
                "Prioritize asset inspection followed by process review."
            )
        else:
            attribution["overall_attribution"] = "normal"
            attribution["recommendations"].append(
                "Energy consumption patterns are within normal operating range. "
                "No immediate action required."
            )

        print(f"[HackathonAdapter] Energy attribution: {attribution['overall_attribution']}")
        print(f"  Asset indicators: {n_asset}, Process indicators: {n_process}")

        return attribution


# ──────────────────────────────────────────────────────────────────
# JSON serialization helper for numpy types
# ──────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ──────────────────────────────────────────────────────────────────
# Standalone Training on Hackathon Data
# ──────────────────────────────────────────────────────────────────

def train_on_hackathon_data(verbose: bool = True) -> dict:
    """Train XGBoost multi-target model on the real hackathon production data.

    Uses a separate model instance so it does not conflict with the
    synthetic-data model used in the main API. The trained model and
    evaluation report are saved to ``models/trained/hackathon_*.pkl``.

    Returns
    -------
    dict
        Training results with per-target metrics (MAE, RMSE, MAPE, R²).
    """
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    adapter = HackathonDataAdapter()
    data = adapter.prepare_for_training(
        targets=HACKATHON_TARGET_COLS,
        test_size=0.2,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    target_cols = data["target_cols"]

    # XGBoost with same hyperparameters as main model
    base_xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model = MultiOutputRegressor(base_xgb)

    if verbose:
        print("\n" + "=" * 60)
        print("  TRAINING ON HACKATHON DATA")
        print("=" * 60)
        print(f"  Features: {len(data['feature_cols'])}")
        print(f"  Targets: {target_cols}")
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate per-target
    results = {"per_target": {}, "overall": {}}

    for i, target in enumerate(target_cols):
        y_true_t = y_test[:, i]
        y_pred_t = y_pred[:, i]

        mae = mean_absolute_error(y_true_t, y_pred_t)
        rmse = np.sqrt(mean_squared_error(y_true_t, y_pred_t))
        r2 = r2_score(y_true_t, y_pred_t)

        # MAPE (avoid division by zero)
        mask = y_true_t != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_t[mask] - y_pred_t[mask]) / y_true_t[mask])) * 100
        else:
            mape = 0.0

        # Accuracy (100% - MAPE, clamped)
        accuracy = max(0, 100 - mape)

        results["per_target"][target] = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape": round(float(mape), 2),
            "r2": round(float(r2), 4),
            "accuracy_pct": round(float(accuracy), 2),
        }

        if verbose:
            spec = QUALITY_SPECS.get(target, {})
            unit = spec.get("unit", "")
            print(f"\n  {target}:")
            print(f"    MAE:  {mae:.4f} {unit}")
            print(f"    RMSE: {rmse:.4f} {unit}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    R²:   {r2:.4f}")
            print(f"    Accuracy: {accuracy:.2f}%")

    # Overall metrics
    avg_accuracy = np.mean([m["accuracy_pct"] for m in results["per_target"].values()])
    avg_r2 = np.mean([m["r2"] for m in results["per_target"].values()])
    results["overall"] = {
        "avg_accuracy_pct": round(float(avg_accuracy), 2),
        "avg_r2": round(float(avg_r2), 4),
        "meets_90pct_target": avg_accuracy >= 90.0,
    }

    if verbose:
        print(f"\n  Overall: Accuracy = {avg_accuracy:.2f}%, R² = {avg_r2:.4f}")
        print(f"  ≥90% Target: {'✅ MET' if avg_accuracy >= 90 else '❌ NOT MET'}")

    # Save model and results
    import joblib
    save_dir = os.path.join(BACKEND_DIR, "models", "trained")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "hackathon_model.pkl")
    joblib.dump(model, model_path)

    scaler_path = os.path.join(save_dir, "hackathon_scaler.pkl")
    joblib.dump(data["scaler"], scaler_path)

    report_path = os.path.join(save_dir, "hackathon_evaluation.json")
    report = {
        "data_source": "hackathon_production_data",
        "n_batches": data["full_df"].shape[0],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "feature_cols": data["feature_cols"],
        "target_cols": target_cols,
        "results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)

    if verbose:
        print(f"\n  Saved: {model_path}")
        print(f"  Saved: {report_path}")

    return results


def analyze_hackathon_process_data(verbose: bool = True) -> dict:
    """Run full analysis on the hackathon process time-series data.

    Returns
    -------
    dict
        Combined analysis: phase stats, energy attribution, power curve.
    """
    adapter = HackathonDataAdapter()

    if verbose:
        print("\n" + "=" * 60)
        print("  HACKATHON PROCESS DATA ANALYSIS")
        print("=" * 60)

    # Phase analysis
    phase_stats = adapter.analyze_process_phases()

    # Energy attribution
    attribution = adapter.attribute_energy_patterns()

    # Power curve for anomaly detection
    power_curve = adapter.get_power_curve()

    return {
        "phase_analysis": phase_stats,
        "energy_attribution": attribution,
        "power_curve": power_curve.tolist(),
        "power_curve_length": len(power_curve),
    }


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    """CLI: train on hackathon data and analyse process data."""
    import argparse

    parser = argparse.ArgumentParser(description="PlantIQ — Hackathon Data Adapter")
    parser.add_argument("--train", action="store_true", help="Train XGBoost on hackathon production data")
    parser.add_argument("--analyze", action="store_true", help="Analyse hackathon process data")
    parser.add_argument("--all", action="store_true", help="Run both training and analysis")
    args = parser.parse_args()

    if args.all or (not args.train and not args.analyze):
        args.train = True
        args.analyze = True

    if args.train:
        train_on_hackathon_data(verbose=True)

    if args.analyze:
        analyze_hackathon_process_data(verbose=True)


if __name__ == "__main__":
    main()
