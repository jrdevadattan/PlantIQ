"""
PlantIQ — Comprehensive API Test Script
==========================================
Tests all new endpoints added in this session:
  1. Golden Signature (discover, compare, scenario, all)
  2. Adaptive Targets (initialize, batch, assess, report)
  3. Hackathon Data (train, predict, production-data, process-analysis,
                     energy-attribution, quality-compliance, power-curve)
  4. Anomaly Detection (with LSTM upgrade)
  5. Existing endpoints (health, predict/batch, predict/realtime, explain)
"""

import json
import sys
import requests

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0
RESULTS = []


def test(name: str, method: str, path: str, body=None, expect_status=200):
    global PASS, FAIL
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=body, timeout=60)

        success = r.status_code == expect_status
        if success:
            PASS += 1
            RESULTS.append(f"  ✅ {name} — {r.status_code}")
            # Print a snippet of the response
            data = r.json()
            snippet = json.dumps(data, indent=2)[:200]
            print(f"  ✅ {name} — {r.status_code}")
            print(f"     {snippet}...")
        else:
            FAIL += 1
            RESULTS.append(f"  ❌ {name} — {r.status_code}: {r.text[:200]}")
            print(f"  ❌ {name} — {r.status_code}: {r.text[:200]}")
    except Exception as e:
        FAIL += 1
        RESULTS.append(f"  ❌ {name} — ERROR: {e}")
        print(f"  ❌ {name} — ERROR: {e}")
    print()


print("=" * 65)
print("  PlantIQ API — Comprehensive Endpoint Tests")
print("=" * 65)
print()

# ── 1. Existing Endpoints ────────────────────────────────────
print("━" * 50)
print("  1. EXISTING ENDPOINTS")
print("━" * 50)

test("Health Check", "GET", "/health")

test("Batch Prediction", "POST", "/predict/batch", {
    "temperature": 183, "conveyor_speed": 75, "hold_time": 18,
    "batch_size": 500, "material_type": 0, "hour_of_day": 10,
    "operator_exp": 2,
})

test("Realtime Prediction", "POST", "/predict/realtime", {
    "original_params": {
        "temperature": 183, "conveyor_speed": 75, "hold_time": 18,
        "batch_size": 500, "material_type": 0, "hour_of_day": 10,
        "operator_exp": 2,
    },
    "partial_data": {
        "elapsed_minutes": 9, "energy_so_far": 15.0,
        "avg_power_kw": 1.67, "anomaly_events": 0,
    },
})

test("SHAP Explanation", "POST", "/explain/BATCH_TEST?target=energy_kwh", {
    "temperature": 183, "conveyor_speed": 75, "hold_time": 18,
    "batch_size": 500, "material_type": 0, "hour_of_day": 10,
    "operator_exp": 2,
})

test("Model Features", "GET", "/model/features")

# ── 2. Anomaly Detection (LSTM) ──────────────────────────────
print("━" * 50)
print("  2. ANOMALY DETECTION (LSTM)")
print("━" * 50)

# Normal curve
import numpy as np
np.random.seed(42)
normal_curve = (5.0 + np.random.normal(0, 0.1, 300)).tolist()
test("Anomaly - Normal Curve", "POST", "/anomaly/detect", {
    "batch_id": "TEST_NORMAL",
    "power_readings": normal_curve,
    "elapsed_seconds": 1800,
})

# Anomalous curve (spiky)
spiky_curve = (5.0 + np.random.normal(0, 0.1, 300)).tolist()
for i in range(0, 100, 5):
    spiky_curve[i] += np.random.uniform(1.0, 3.0)
test("Anomaly - Spiky Curve", "POST", "/anomaly/detect", {
    "batch_id": "TEST_SPIKY",
    "power_readings": spiky_curve,
    "elapsed_seconds": 1800,
})

# ── 3. Golden Signature ──────────────────────────────────────
print("━" * 50)
print("  3. GOLDEN SIGNATURE")
print("━" * 50)

test("Golden - Discover (Synthetic)", "POST", "/golden-signature/discover", {
    "data_source": "synthetic", "n_top": 5,
})

test("Golden - Discover (Hackathon)", "POST", "/golden-signature/discover", {
    "data_source": "hackathon", "n_top": 5,
})

test("Golden - Get All", "GET", "/golden-signature/all")

test("Golden - Compare", "POST", "/golden-signature/compare", {
    "batch_params": {
        "temperature": 183, "conveyor_speed": 75, "hold_time": 18,
        "batch_size": 500, "material_type": 0, "hour_of_day": 10,
        "operator_exp": 2,
    },
    "batch_targets": {
        "quality_score": 92.5, "yield_pct": 94.0,
        "performance_pct": 88.0, "energy_kwh": 32.0,
    },
})

test("Golden - Scenario", "POST", "/golden-signature/scenario", {
    "primary_targets": ["quality_score", "yield_pct"],
    "secondary_targets": ["energy_kwh"],
    "primary_weight": 0.8,
    "data_source": "synthetic",
})

# ── 4. Adaptive Targets ──────────────────────────────────────
print("━" * 50)
print("  4. ADAPTIVE TARGETS")
print("━" * 50)

test("Targets - Initialize (Synthetic)", "POST", "/targets/initialize", {
    "data_source": "synthetic",
})

test("Targets - Get Batch Targets", "POST", "/targets/batch", {
    "current_batch_number": 50, "annual_batches": 1000,
})

test("Targets - Assess Batch", "POST", "/targets/assess", {
    "energy_kwh": 32.5, "quality_score": 92.0,
    "yield_pct": 94.5, "performance_pct": 87.5,
    "batch_number": 51,
})

test("Targets - Performance Report", "GET", "/targets/report")

# ── 5. Hackathon Data ────────────────────────────────────────
print("━" * 50)
print("  5. HACKATHON DATA")
print("━" * 50)

test("Hackathon - Production Data", "GET", "/hackathon/production-data")

test("Hackathon - Quality Compliance", "GET", "/hackathon/quality-compliance")

test("Hackathon - Process Analysis", "GET", "/hackathon/process-analysis")

test("Hackathon - Energy Attribution", "GET", "/hackathon/energy-attribution")

test("Hackathon - Power Curve", "GET", "/hackathon/power-curve")

test("Hackathon - Train Model", "POST", "/hackathon/train", {"verbose": False})

test("Hackathon - Predict", "POST", "/hackathon/predict", {
    "granulation_time": 40, "binder_amount": 5.5, "drying_temp": 55,
    "drying_time": 40, "compression_force": 20, "machine_speed": 40,
    "lubricant_conc": 0.8,
})

# ── Summary ───────────────────────────────────────────────────
print("=" * 65)
print(f"  RESULTS: {PASS} passed, {FAIL} failed / {PASS + FAIL} total")
print("=" * 65)
for r in RESULTS:
    print(r)
print()

if FAIL > 0:
    print(f"  ⚠️ {FAIL} endpoint(s) FAILED — review above")
    sys.exit(1)
else:
    print("  🎉 ALL ENDPOINTS PASSED!")
    sys.exit(0)
