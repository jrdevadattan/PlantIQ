"""
PlantIQ — Demo Scenario Script (F3.3)
=======================================
A CLI script that drives a 3-minute scripted demonstration
of the PlantIQ Manufacturing Intelligence Platform.

Usage:
  python demo/demo_scenario.py                        # Full auto-demo (normal → wet_material)
  python demo/demo_scenario.py --inject wet_material   # Inject specific fault mid-demo
  python demo/demo_scenario.py --inject bearing_wear   # Bearing wear demo
  python demo/demo_scenario.py --inject calibration_needed  # Calibration fault demo
  python demo/demo_scenario.py --fast                  # Fast-forward mode (skip pauses)

Requires the backend to be running at http://localhost:8000.
"""

import argparse
import json
import sys
import time
import os
import random
import math
from typing import Optional

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

import numpy as np
import requests

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_BASE = "http://localhost:8000"
TIMESTEPS = 1800           # 30-minute batch at 1Hz
CO2_FACTOR = 0.82
BATCH_CO2_BUDGET = 42.0

# Demo parameters (matching README demo script)
DEMO_PARAMS = {
    "temperature": 183.0,
    "conveyor_speed": 76.0,
    "hold_time": 18.0,
    "batch_size": 500.0,
    "material_type": 1,        # Type B — Dense
    "hour_of_day": 9,
    "operator_exp": 1,         # Mid-level
}

# ──────────────────────────────────────────────
# ANSI Color Helpers
# ──────────────────────────────────────────────
class C:
    """ANSI color codes for terminal output."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def banner(text: str, color: str = C.CYAN) -> None:
    """Print a prominent banner."""
    width = max(len(text) + 6, 60)
    line = "═" * width
    print(f"\n{color}{C.BOLD}{line}")
    print(f"   {text}")
    print(f"{line}{C.RESET}\n")


def stage(time_mark: str, title: str) -> None:
    """Print a demo stage header."""
    print(f"\n{C.MAGENTA}{C.BOLD}T={time_mark}{C.RESET}  "
          f"{C.WHITE}{C.BOLD}{title}{C.RESET}")
    print(f"{C.DIM}{'─' * 55}{C.RESET}")


def info(label: str, value: str, indent: int = 2) -> None:
    """Print an info line."""
    pad = " " * indent
    print(f"{pad}{C.DIM}│{C.RESET} {C.CYAN}{label}:{C.RESET} {value}")


def alert(severity: str, message: str) -> None:
    """Print an alert with severity-appropriate color."""
    colors = {
        "NORMAL": C.GREEN,
        "WATCH": C.YELLOW,
        "WARNING": C.YELLOW,
        "CRITICAL": C.RED,
    }
    bg_colors = {
        "NORMAL": "",
        "WATCH": "",
        "WARNING": C.BG_YELLOW,
        "CRITICAL": C.BG_RED,
    }
    color = colors.get(severity, C.WHITE)
    bg = bg_colors.get(severity, "")
    icon = {"NORMAL": "●", "WATCH": "◐", "WARNING": "▲", "CRITICAL": "◆"}.get(severity, "?")
    print(f"  {bg}{color}{C.BOLD}  {icon} [{severity}] {message}  {C.RESET}")


def wait(seconds: float, fast_mode: bool = False, msg: str = "") -> None:
    """Pause for dramatic effect (skipped in fast mode)."""
    if fast_mode:
        return
    if msg:
        print(f"\n  {C.DIM}⏳ {msg} ({seconds}s)...{C.RESET}", end="", flush=True)
    time.sleep(seconds)
    if msg:
        print(f" ✓")


# ──────────────────────────────────────────────
# Power Curve Generation (self-contained)
# ──────────────────────────────────────────────
def generate_normal_curve(
    batch_size: float = 500.0,
    conveyor_speed: float = 76.0,
    temperature: float = 183.0,
) -> np.ndarray:
    """Generate a normal 30-minute power consumption curve."""
    t = np.arange(TIMESTEPS, dtype=np.float64)
    size_factor = 0.002 * (batch_size - 300)
    speed_factor = 0.03 * (conveyor_speed - 60)
    plateau_power = 5.0 + size_factor + speed_factor

    startup_tau = 80.0 - 0.008 * (temperature - 175) * 10
    startup = plateau_power * (1 - np.exp(-t / startup_tau))

    cooldown_start = 1620
    cooldown_tau = 60.0
    cooldown = np.where(
        t > cooldown_start,
        plateau_power * np.exp(-(t - cooldown_start) / cooldown_tau),
        plateau_power,
    )
    curve = np.minimum(startup, cooldown)
    curve += np.random.normal(0, 0.12, TIMESTEPS)
    curve += 0.08 * np.sin(2 * np.pi * t / 120)
    return np.clip(curve, 0.5, 9.5).astype(np.float32)


def inject_wet_material(curve: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """Inject wet-material fault signature starting at a given index."""
    faulted = curve.copy()
    num_spikes = random.randint(15, 40)
    spike_zone_end = min(start_idx + 600, len(faulted))
    spike_positions = np.random.randint(start_idx, spike_zone_end, size=num_spikes)
    for pos in spike_positions:
        amplitude = random.uniform(0.5, 2.5)
        width = random.randint(3, 15)
        end = min(pos + width, len(faulted))
        spike_t = np.arange(end - pos)
        spike_center = width / 2
        spike = amplitude * np.exp(-0.5 * ((spike_t - spike_center) / (width / 4)) ** 2)
        faulted[pos:end] += spike
    faulted[start_idx:spike_zone_end] += np.random.uniform(0.2, 0.8, size=spike_zone_end - start_idx)
    return np.clip(faulted, 0.5, 12.0).astype(np.float32)


def inject_bearing_wear(curve: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """Inject bearing-wear fault signature starting at a given index."""
    faulted = curve.copy()
    t = np.arange(len(curve), dtype=np.float64)
    t_shifted = np.maximum(t - start_idx, 0)
    rise = 0.003 * t_shifted
    acceleration = np.clip(0.0000005 * t_shifted ** 2, 0, 1.5)
    faulted += rise + acceleration
    return np.clip(faulted, 0.5, 15.0).astype(np.float32)


def inject_calibration_needed(curve: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """Inject calibration-needed fault signature starting at a given index."""
    faulted = curve.copy()
    offset = random.uniform(0.4, 0.8)
    faulted[start_idx:] += offset
    smoothing = 0.3
    for i in range(max(start_idx, 1), len(faulted)):
        faulted[i] = smoothing * faulted[i - 1] + (1 - smoothing) * faulted[i]
    return np.clip(faulted, 0.5, 10.5).astype(np.float32)


FAULT_INJECTORS = {
    "wet_material": inject_wet_material,
    "bearing_wear": inject_bearing_wear,
    "calibration_needed": inject_calibration_needed,
}

FAULT_DESCRIPTIONS = {
    "wet_material": "Wet Material — irregular power spikes from high-moisture feedstock",
    "bearing_wear": "Bearing Wear — gradual baseline rise from mechanical degradation",
    "calibration_needed": "Calibration Needed — elevated flat offset from sensor drift",
}


# ──────────────────────────────────────────────
# API Helpers
# ──────────────────────────────────────────────
def api_post(path: str, body: dict) -> Optional[dict]:
    """POST to the backend API. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  {C.RED}✘ API error: {e}{C.RESET}")
        return None


def api_get(path: str) -> Optional[dict]:
    """GET from the backend API."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  {C.RED}✘ API error: {e}{C.RESET}")
        return None


# ──────────────────────────────────────────────
# Demo Stages
# ──────────────────────────────────────────────
def check_backend() -> bool:
    """Verify the backend is reachable."""
    health = api_get("/health")
    if not health:
        print(f"\n{C.RED}{C.BOLD}  ✘ Cannot reach backend at {API_BASE}{C.RESET}")
        print(f"  {C.DIM}Start it with: cd backend && python -m uvicorn main:app --port 8000{C.RESET}\n")
        return False
    info("Status", f"{C.GREEN}{health.get('status', 'unknown')}{C.RESET}")
    info("Models loaded", f"{C.GREEN}{health.get('models_loaded', False)}{C.RESET}")
    info("Version", health.get("version", "?"))
    return True


def stage_pre_batch(fast: bool) -> Optional[str]:
    """Stage 1: Pre-batch prediction."""
    stage("0:00", "Pre-Batch Prediction")
    print(f"  {C.DIM}Sending batch parameters to prediction engine...{C.RESET}")

    result = api_post("/predict/batch", DEMO_PARAMS)
    if not result:
        return None

    preds = result.get("predictions", {})
    carbon = result.get("carbon_budget", {})

    info("Batch ID", f"{C.BOLD}{result.get('batch_id', '?')}{C.RESET}")
    info("Quality Score", f"{C.GREEN}{C.BOLD}{preds.get('quality_score', 0):.1f}%{C.RESET}")
    info("Yield", f"{C.GREEN}{C.BOLD}{preds.get('yield_pct', 0):.1f}%{C.RESET}")
    info("Performance", f"{C.GREEN}{C.BOLD}{preds.get('performance_pct', 0):.1f}%{C.RESET}")
    info("Energy", f"{C.BOLD}{preds.get('energy_kwh', 0):.1f} kWh{C.RESET}")
    info("CO₂ Forecast", f"{preds.get('co2_kg', 0):.1f} kg")
    info("Carbon Budget", f"{carbon.get('predicted_usage_kg', 0):.1f} / {carbon.get('batch_budget_kg', 42):.0f} kg "
         f"— {C.GREEN}{carbon.get('status', '?')}{C.RESET}")

    wait(3, fast, "Presenter speaks about prediction capabilities")
    return result.get("batch_id")


def stage_normal_operation(curve: np.ndarray, fast: bool) -> None:
    """Stage 2: Normal batch operation — first ~45 seconds."""
    stage("0:45", "Normal Operation — Batch Running Smoothly")

    # Send the first 480 timesteps (~8 minutes of batch = 26.7% progress)
    segment = curve[:480].tolist()
    elapsed_s = 480
    elapsed_min = elapsed_s / 60.0
    energy_so_far = float(np.trapezoid(curve[:480]) / 3600)  # kWh
    avg_power = float(np.mean(curve[:480]))

    # Anomaly check
    anomaly = api_post("/anomaly/detect", {
        "batch_id": "DEMO-001",
        "power_readings": segment,
        "elapsed_seconds": elapsed_s,
    })

    if anomaly:
        score = anomaly.get("anomaly_score", 0)
        severity = anomaly.get("severity", "NORMAL")
        info("Anomaly Score", f"{C.GREEN}{score:.3f}{C.RESET}")
        info("Severity", f"{C.GREEN}{severity}{C.RESET}")
        alert(severity, f"Score {score:.3f} — all systems nominal")

    # Realtime prediction
    realtime = api_post("/predict/realtime", {
        "original_params": DEMO_PARAMS,
        "partial_data": {
            "elapsed_minutes": elapsed_min,
            "energy_so_far": energy_so_far,
            "avg_power_kw": avg_power,
            "anomaly_events": 0,
        },
    })
    if realtime:
        updated = realtime.get("updated_predictions", {})
        info("Progress", f"{realtime.get('progress_pct', 0):.0f}%")
        info("Projected Energy", f"{updated.get('energy_kwh', 0):.1f} kWh")

    wait(3, fast, "Presenter highlights normal operation")


def stage_fault_injection(
    curve: np.ndarray,
    fault_type: str,
    injection_point: int = 480,
    fast: bool = False,
) -> np.ndarray:
    """Stage 3: Inject fault mid-batch."""
    stage("1:00", f"⚡ Fault Injection — {FAULT_DESCRIPTIONS.get(fault_type, fault_type)}")
    print(f"  {C.YELLOW}Injecting {fault_type} anomaly at timestep {injection_point} "
          f"(~{injection_point // 60} min into batch)...{C.RESET}")

    injector = FAULT_INJECTORS.get(fault_type)
    if not injector:
        print(f"  {C.RED}Unknown fault type: {fault_type}{C.RESET}")
        return curve

    faulted = injector(curve, start_idx=injection_point)

    # Send extended segment including anomalous data (~18 min = 1080 timesteps)
    segment_end = min(injection_point + 600, TIMESTEPS)
    segment = faulted[:segment_end].tolist()
    elapsed_s = segment_end
    elapsed_min = elapsed_s / 60.0
    energy_so_far = float(np.trapezoid(faulted[:segment_end]) / 3600)
    avg_power = float(np.mean(faulted[:segment_end]))

    # Anomaly detection — should now detect the fault
    print(f"\n  {C.DIM}Running anomaly detection on {len(segment)} readings...{C.RESET}")
    anomaly = api_post("/anomaly/detect", {
        "batch_id": "DEMO-001",
        "power_readings": segment,
        "elapsed_seconds": elapsed_s,
    })

    if anomaly:
        score = anomaly.get("anomaly_score", 0)
        severity = anomaly.get("severity", "NORMAL")
        diag = anomaly.get("diagnosis", {})

        info("Anomaly Score", f"{C.RED}{C.BOLD}{score:.3f}{C.RESET}")
        info("Severity", f"{C.RED}{C.BOLD}{severity}{C.RESET}")
        info("Diagnosis", f"{C.BOLD}{diag.get('fault_type', '?')}{C.RESET} "
             f"({diag.get('confidence', 0) * 100:.0f}% confidence)")
        info("Explanation", f"{diag.get('human_readable', 'N/A')}")

        alert(severity, diag.get("human_readable", "Anomaly detected"))

        if diag.get("recommended_action"):
            print(f"\n  {C.CYAN}{C.BOLD}  💡 Recommendation: {diag['recommended_action']}{C.RESET}")
        if diag.get("estimated_energy_impact_kwh"):
            info("Energy Impact", f"+{diag['estimated_energy_impact_kwh']:.1f} kWh over plan")
        if diag.get("estimated_quality_impact_pct"):
            info("Quality Impact", f"-{diag['estimated_quality_impact_pct']:.1f}% quality risk")

    # Realtime prediction — should show energy overshoot
    realtime = api_post("/predict/realtime", {
        "original_params": DEMO_PARAMS,
        "partial_data": {
            "elapsed_minutes": elapsed_min,
            "energy_so_far": energy_so_far,
            "avg_power_kw": avg_power,
            "anomaly_events": 1,
        },
    })
    if realtime:
        updated = realtime.get("updated_predictions", {})
        rt_alert = realtime.get("alert")
        print()
        info("Progress", f"{realtime.get('progress_pct', 0):.0f}%")
        info("Projected Energy", f"{C.YELLOW}{C.BOLD}{updated.get('energy_kwh', 0):.1f} kWh{C.RESET} "
             f"(was ~{DEMO_PARAMS['hold_time'] * 2.16:.1f} kWh)")
        info("Projected CO₂", f"{updated.get('co2_kg', 0):.1f} kg / {BATCH_CO2_BUDGET:.0f} kg budget")

        if rt_alert:
            alert(rt_alert.get("severity", "WARNING"), rt_alert.get("message", ""))
            if rt_alert.get("recommended_action"):
                print(f"  {C.CYAN}  → {rt_alert['recommended_action']}{C.RESET}")

    wait(5, fast, "Presenter explains anomaly detection + alert system")
    return faulted


def stage_recovery(faulted_curve: np.ndarray, fast: bool) -> None:
    """Stage 4: Apply recommendation + recovery."""
    stage("2:15", "Recovery — Operator Applies Recommendation")

    # Simulate applying the recommendation: re-predict with corrective parameters
    # (e.g., extend hold time from 18 → 22 minutes to account for wet material)
    corrected_params = {**DEMO_PARAMS, "hold_time": 22.0}
    result = api_post("/predict/batch", corrected_params)

    if result:
        preds = result.get("predictions", {})
        carbon = result.get("carbon_budget", {})

        print(f"  {C.GREEN}Applied corrective action: hold_time 18 → 22 min{C.RESET}")
        info("Revised Quality", f"{C.GREEN}{C.BOLD}{preds.get('quality_score', 0):.1f}%{C.RESET}")
        info("Revised Energy", f"{C.GREEN}{C.BOLD}{preds.get('energy_kwh', 0):.1f} kWh{C.RESET}")
        info("Revised CO₂", f"{preds.get('co2_kg', 0):.1f} kg")
        info("Carbon Status", f"{C.GREEN}{carbon.get('status', '?')}{C.RESET} "
             f"({carbon.get('predicted_usage_kg', 0):.1f} / {carbon.get('batch_budget_kg', 42):.0f} kg)")

    wait(3, fast, "Presenter highlights energy savings from early detection")


def stage_what_if(fast: bool) -> None:
    """Stage 5: What-If Simulator demo."""
    stage("2:45", "What-If Simulator — Trade-off Exploration")

    print(f"  {C.DIM}Simulating slider drag: hold_time 18 → 25 minutes...{C.RESET}")

    # Baseline prediction (hold_time = 18)
    base_result = api_post("/predict/batch", DEMO_PARAMS)
    # What-if prediction (hold_time = 25)
    whatif_params = {**DEMO_PARAMS, "hold_time": 25.0}
    whatif_result = api_post("/predict/batch", whatif_params)

    if base_result and whatif_result:
        base_preds = base_result.get("predictions", {})
        whatif_preds = whatif_result.get("predictions", {})

        targets = [
            ("Quality Score", "quality_score", "%", True),
            ("Yield", "yield_pct", "%", True),
            ("Performance", "performance_pct", "%", True),
            ("Energy", "energy_kwh", "kWh", False),
        ]

        print()
        for label, key, unit, higher_is_good in targets:
            base_v = base_preds.get(key, 0)
            what_v = whatif_preds.get(key, 0)
            delta = what_v - base_v
            sign = "+" if delta > 0 else ""
            color = C.GREEN if (delta > 0) == higher_is_good else C.YELLOW
            info(label,
                 f"{base_v:.1f} → {color}{C.BOLD}{what_v:.1f}{C.RESET} "
                 f"({color}{sign}{delta:.1f}{C.RESET}) {unit}")

        base_co2 = base_preds.get("co2_kg", 0)
        whatif_co2 = whatif_preds.get("co2_kg", 0)
        co2_delta = whatif_co2 - base_co2
        info("CO₂ Impact",
             f"{base_co2:.1f} → {whatif_co2:.1f} kg ({'+' if co2_delta > 0 else ''}{co2_delta:.1f} kg)")

        print(f"\n  {C.MAGENTA}{C.BOLD}\"Trade-off visible instantly. The operator can find the right balance "
              f"before starting — not discover it after.\"{C.RESET}")

    wait(3, fast, "Presenter demonstrates slider interaction")


def stage_closing() -> None:
    """Stage 6: Closing statement."""
    stage("3:00", "Closing — PlantIQ Value Proposition")
    print(f"""
  {C.BOLD}{C.WHITE}PlantIQ delivers:{C.RESET}

  {C.GREEN}✓{C.RESET} Predict before you produce        (Multi-target XGBoost)
  {C.GREEN}✓{C.RESET} Monitor during the batch           (Sliding Window Forecaster)
  {C.GREEN}✓{C.RESET} Detect anomalies in real time      (LSTM Autoencoder)
  {C.GREEN}✓{C.RESET} Explain every prediction            (SHAP + plain English)
  {C.GREEN}✓{C.RESET} Act on recommendations immediately  (Adaptive alerts)
  {C.GREEN}✓{C.RESET} Explore trade-offs before commit    (What-If Simulator)

  {C.DIM}Key metrics:{C.RESET}
    • 13% energy overrun caught at minute 8 of a 30-min batch
    • >93% multi-target prediction accuracy
    • <50ms API response time
    • ~18,900 kWh / 15,498 kg CO₂ annual savings potential
""")


# ──────────────────────────────────────────────
# Main Flow
# ──────────────────────────────────────────────
def run_demo(fault_type: str = "wet_material", fast: bool = False) -> None:
    """Execute the full demo scenario."""
    banner(f"PlantIQ — Live Demo Scenario  (fault: {fault_type})")

    # Health check
    print(f"{C.DIM}Checking backend connectivity...{C.RESET}")
    if not check_backend():
        sys.exit(1)

    wait(2, fast, "Ready to begin demo")

    # Generate base power curve
    np.random.seed(42)
    random.seed(42)
    curve = generate_normal_curve(
        batch_size=DEMO_PARAMS["batch_size"],
        conveyor_speed=DEMO_PARAMS["conveyor_speed"],
        temperature=DEMO_PARAMS["temperature"],
    )

    # Stage 1: Pre-batch prediction
    batch_id = stage_pre_batch(fast)
    if not batch_id:
        print(f"\n{C.RED}Demo aborted — prediction endpoint failed.{C.RESET}")
        sys.exit(1)

    # Stage 2: Normal operation
    stage_normal_operation(curve, fast)

    # Stage 3: Fault injection
    faulted = stage_fault_injection(curve, fault_type, injection_point=480, fast=fast)

    # Stage 4: Recovery
    stage_recovery(faulted, fast)

    # Stage 5: What-If Simulator
    stage_what_if(fast)

    # Stage 6: Closing
    stage_closing()

    banner("Demo Complete — Thank You!", C.GREEN)


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="PlantIQ Demo Scenario Script — 3-minute live demo driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fault types:
  wet_material        Irregular power spikes from high-moisture feedstock
  bearing_wear        Gradual baseline rise from mechanical degradation
  calibration_needed  Elevated flat offset from sensor drift

Examples:
  python demo/demo_scenario.py                          # Full auto-demo
  python demo/demo_scenario.py --inject wet_material    # Inject wet material fault
  python demo/demo_scenario.py --inject bearing_wear --fast  # Fast bearing demo
        """,
    )
    parser.add_argument(
        "--inject",
        type=str,
        default="wet_material",
        choices=["wet_material", "bearing_wear", "calibration_needed"],
        help="Fault type to inject mid-batch (default: wet_material)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast-forward mode — skip dramatic pauses",
    )
    args = parser.parse_args()

    run_demo(fault_type=args.inject, fast=args.fast)


if __name__ == "__main__":
    main()
