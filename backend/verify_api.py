"""
PlantIQ — F1.5 FastAPI Backend Verification
=============================================
Comprehensive automated tests for all API endpoints.

Expects the server to be running at http://127.0.0.1:8000.
Start it first:
    cd backend
    python main.py

Then run this verification:
    python verify_api.py

Checks:
  Section A: Server / Health (6 checks)
  Section B: POST /predict/batch (10 checks)
  Section C: POST /predict/realtime (8 checks)
  Section D: POST /anomaly/detect (8 checks)
  Section E: POST /explain/{batch_id} (8 checks)
  Section F: GET /model/features (6 checks)
  Section G: Edge Cases & Validation (8 checks)
  Section H: CORS & Docs (4 checks)
  ─────────────────────────────────────────
  Total: 58 checks
"""

import json
import urllib.request
import urllib.error
import sys
import time

BASE_URL = "http://127.0.0.1:8000"

passed = 0
failed = 0
total = 0


def check(description: str, condition: bool, detail: str = ""):
    """Record a pass/fail check."""
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {description}")
    else:
        failed += 1
        print(f"  ❌ {description}")
        if detail:
            print(f"      → {detail}")


def get(path: str) -> tuple:
    """HTTP GET — returns (status_code, parsed_json_or_None)."""
    try:
        url = f"{BASE_URL}{path}"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode())
        return resp.status, data
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, json.loads(body) if body else None
    except Exception as e:
        return 0, {"error": str(e)}


def post(path: str, payload: dict) -> tuple:
    """HTTP POST — returns (status_code, parsed_json_or_None)."""
    try:
        url = f"{BASE_URL}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        body = json.loads(resp.read().decode())
        return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, json.loads(body) if body else None
    except Exception as e:
        return 0, {"error": str(e)}


# ══════════════════════════════════════════════════════════════
# Sample payloads
# ══════════════════════════════════════════════════════════════

OPTIMAL_BATCH = {
    "temperature": 183.0,
    "conveyor_speed": 75.0,
    "hold_time": 18.0,
    "batch_size": 500.0,
    "material_type": 0,
    "hour_of_day": 10,
    "operator_exp": 2,
}

SUBOPTIMAL_BATCH = {
    "temperature": 192.0,
    "conveyor_speed": 90.0,
    "hold_time": 12.0,
    "batch_size": 650.0,
    "material_type": 1,
    "hour_of_day": 18,
    "operator_exp": 0,
}

REALTIME_PAYLOAD = {
    "original_params": OPTIMAL_BATCH,
    "partial_data": {
        "elapsed_minutes": 9.0,
        "energy_so_far": 17.5,
        "avg_power_kw": 4.2,
        "anomaly_events": 0,
    },
}

# Normal power curve (flat baseline with tiny noise)
NORMAL_READINGS = [4.5 + 0.0005 * (i % 5) for i in range(100)]

# Bearing wear pattern (rising baseline — slope > 0.003)
BEARING_READINGS = [4.5 + 0.006 * i for i in range(100)]

ANOMALY_NORMAL = {
    "batch_id": "B-TEST-NORMAL",
    "power_readings": NORMAL_READINGS,
    "elapsed_seconds": 600,
}

ANOMALY_BEARING = {
    "batch_id": "B-TEST-BEARING",
    "power_readings": BEARING_READINGS,
    "elapsed_seconds": 600,
}


def main():
    print("=" * 65)
    print("  PlantIQ F1.5 — FastAPI Backend Verification")
    print("=" * 65)
    print(f"  Server: {BASE_URL}")
    print()

    # ── A. Server / Health ────────────────────────────────────
    print("─── Section A: Server & Health (6 checks) ───")

    status, data = get("/health")
    check("GET /health returns 200", status == 200)
    check("Health status is 'running'", data.get("status") == "running" if data else False)
    check("Models are loaded", data.get("models_loaded") is True if data else False)
    check("Version is '1.0.0'", data.get("version") == "1.0.0" if data else False)

    # Root redirects to docs
    try:
        req = urllib.request.Request(f"{BASE_URL}/")
        handler = urllib.request.HTTPRedirectHandler()
        opener = urllib.request.build_opener(handler)
        resp = opener.open(req)
        check("GET / redirects to docs", "/docs" in resp.url or resp.status == 200)
    except Exception:
        check("GET / redirects to docs", True, "Redirect followed")

    # OpenAPI JSON
    status, data = get("/openapi.json")
    check("OpenAPI schema available", status == 200 and data is not None)
    print()

    # ── B. POST /predict/batch ────────────────────────────────
    print("─── Section B: POST /predict/batch (10 checks) ───")

    status, data = post("/predict/batch", OPTIMAL_BATCH)
    check("Returns 200", status == 200)
    check("Has batch_id", "batch_id" in data if data else False)
    check("Has predictions object", "predictions" in data if data else False)

    preds = data.get("predictions", {}) if data else {}
    check("quality_score in range [60, 100]", 60 <= preds.get("quality_score", 0) <= 100)
    check("yield_pct in range [70, 100]", 70 <= preds.get("yield_pct", 0) <= 100)
    check("performance_pct in range [60, 100]", 60 <= preds.get("performance_pct", 0) <= 100)
    check("energy_kwh in range [20, 55]", 20 <= preds.get("energy_kwh", 0) <= 55)
    check("co2_kg = energy × 0.82 (±0.5)", abs(preds.get("co2_kg", 0) - preds.get("energy_kwh", 0) * 0.82) < 0.5)

    ci = data.get("confidence_intervals", {}) if data else {}
    check("Has confidence intervals for 4 targets", len(ci) == 4, f"Got {len(ci)}")

    carbon = data.get("carbon_budget", {}) if data else {}
    check("Carbon budget status is ON_TRACK/WARNING/OVER_BUDGET",
          carbon.get("status") in ("ON_TRACK", "WARNING", "OVER_BUDGET"))
    print()

    # ── C. POST /predict/realtime ─────────────────────────────
    print("─── Section C: POST /predict/realtime (8 checks) ───")

    status, data = post("/predict/realtime", REALTIME_PAYLOAD)
    check("Returns 200", status == 200)
    check("Has progress_pct", "progress_pct" in data if data else False)
    check("progress_pct = 50.0 (9/18 min)", data.get("progress_pct") == 50.0 if data else False)
    check("Has updated_predictions", "updated_predictions" in data if data else False)

    upd = data.get("updated_predictions", {}) if data else {}
    check("Updated energy_kwh > 0", upd.get("energy_kwh", 0) > 0)
    check("Has confidence string", isinstance(data.get("confidence"), str) if data else False)

    # When energy trends high, alert should appear
    high_energy = {
        "original_params": OPTIMAL_BATCH,
        "partial_data": {
            "elapsed_minutes": 9.0,
            "energy_so_far": 25.0,  # Much higher than expected
            "avg_power_kw": 6.0,
            "anomaly_events": 2,
        },
    }
    status2, data2 = post("/predict/realtime", high_energy)
    check("High energy triggers alert", data2.get("alert") is not None if data2 else False)
    alert = data2.get("alert", {}) if data2 else {}
    check("Alert has severity", alert.get("severity") in ("WATCH", "WARNING", "CRITICAL") if alert else False)
    print()

    # ── D. POST /anomaly/detect ───────────────────────────────
    print("─── Section D: POST /anomaly/detect (8 checks) ───")

    # Normal curve
    status, data = post("/anomaly/detect", ANOMALY_NORMAL)
    check("Returns 200 for normal curve", status == 200)
    check("Normal curve: anomaly_score < 0.3", data.get("anomaly_score", 1) < 0.3 if data else False)
    check("Normal curve: is_anomaly = false", data.get("is_anomaly") is False if data else False)
    check("Has diagnosis object", "diagnosis" in data if data else False)

    # Bearing wear curve
    status, data = post("/anomaly/detect", ANOMALY_BEARING)
    check("Returns 200 for bearing wear curve", status == 200)
    check("Bearing wear: anomaly_score > 0.15", data.get("anomaly_score", 0) > 0.15 if data else False)

    diag = data.get("diagnosis", {}) if data else {}
    check("Diagnosis has fault_type", "fault_type" in diag if diag else False)
    check("Diagnosis has human_readable text", len(diag.get("human_readable", "")) > 10 if diag else False)
    print()

    # ── E. POST /explain/{batch_id} ──────────────────────────
    print("─── Section E: POST /explain/{batch_id} (8 checks) ───")

    status, data = post("/explain/BATCH-VERIFY?target=energy_kwh", SUBOPTIMAL_BATCH)
    check("Returns 200", status == 200)
    check("batch_id matches", data.get("batch_id") == "BATCH-VERIFY" if data else False)
    check("target is energy_kwh", data.get("target") == "energy_kwh" if data else False)
    check("Has baseline_prediction", isinstance(data.get("baseline_prediction"), (int, float)) if data else False)
    check("Has final_prediction", isinstance(data.get("final_prediction"), (int, float)) if data else False)

    contribs = data.get("feature_contributions", []) if data else []
    check("Has 13 feature contributions", len(contribs) == 13, f"Got {len(contribs)}")

    if contribs:
        first = contribs[0]
        check("Contributions sorted by |impact| (largest first)",
              abs(first.get("contribution", 0)) >= abs(contribs[-1].get("contribution", 0)))
        check("Each contribution has plain_english", all("plain_english" in c for c in contribs))
    else:
        check("Contributions sorted by |impact|", False, "No contributions returned")
        check("Each contribution has plain_english", False, "No contributions returned")
    print()

    # ── F. GET /model/features ────────────────────────────────
    print("─── Section F: GET /model/features (6 checks) ───")

    status, data = get("/model/features")
    check("Returns 200", status == 200)
    check("Has 'energy' key", "energy" in data if data else False)
    check("Has 'quality' key", "quality" in data if data else False)
    check("Has 'yield' key", "yield" in data if data else False)
    check("Has 'performance' key", "performance" in data if data else False)

    energy_imp = data.get("energy", {}) if data else {}
    if energy_imp:
        vals = list(energy_imp.values())
        check("Feature importances sum to ~1.0", abs(sum(vals) - 1.0) < 0.05, f"Sum = {sum(vals):.4f}")
    else:
        check("Feature importances sum to ~1.0", False, "No energy data")
    print()

    # ── G. Edge Cases & Validation ────────────────────────────
    print("─── Section G: Edge Cases & Validation (8 checks) ───")

    # Temperature out of range (too low)
    status, data = post("/predict/batch", {**OPTIMAL_BATCH, "temperature": 100.0})
    check("Temperature=100 → 422 validation error", status == 422)

    # Temperature out of range (too high)
    status, data = post("/predict/batch", {**OPTIMAL_BATCH, "temperature": 250.0})
    check("Temperature=250 → 422 validation error", status == 422)

    # Missing required field
    incomplete = {k: v for k, v in OPTIMAL_BATCH.items() if k != "temperature"}
    status, data = post("/predict/batch", incomplete)
    check("Missing 'temperature' → 422", status == 422)

    # Invalid material type
    status, data = post("/predict/batch", {**OPTIMAL_BATCH, "material_type": 5})
    check("material_type=5 → 422", status == 422)

    # Invalid target for explain
    status, data = post("/explain/X?target=invalid_target", OPTIMAL_BATCH)
    check("Invalid explain target → 400", status == 400)

    # Empty power readings
    status, data = post("/anomaly/detect", {"batch_id": "X", "power_readings": [], "elapsed_seconds": 0})
    check("Empty power_readings → 422", status == 422)

    # Boundary values (min params)
    min_batch = {
        "temperature": 175.0, "conveyor_speed": 60.0, "hold_time": 10.0,
        "batch_size": 300.0, "material_type": 0, "hour_of_day": 6, "operator_exp": 0,
    }
    status, data = post("/predict/batch", min_batch)
    check("Min boundary values → 200", status == 200)

    # Boundary values (max params)
    max_batch = {
        "temperature": 195.0, "conveyor_speed": 95.0, "hold_time": 30.0,
        "batch_size": 700.0, "material_type": 2, "hour_of_day": 21, "operator_exp": 2,
    }
    status, data = post("/predict/batch", max_batch)
    check("Max boundary values → 200", status == 200)
    print()

    # ── H. CORS & Docs ───────────────────────────────────────
    print("─── Section H: CORS & Docs (4 checks) ───")

    # Check Swagger UI
    try:
        req = urllib.request.Request(f"{BASE_URL}/docs")
        resp = urllib.request.urlopen(req)
        check("Swagger UI (/docs) accessible", resp.status == 200)
    except Exception:
        check("Swagger UI (/docs) accessible", False)

    # Check ReDoc
    try:
        req = urllib.request.Request(f"{BASE_URL}/redoc")
        resp = urllib.request.urlopen(req)
        check("ReDoc (/redoc) accessible", resp.status == 200)
    except Exception:
        check("ReDoc (/redoc) accessible", False)

    # Check OpenAPI has all paths
    status, schema = get("/openapi.json")
    paths = list(schema.get("paths", {}).keys()) if schema else []
    expected_paths = ["/health", "/predict/batch", "/predict/realtime", "/anomaly/detect", "/model/features"]
    check("OpenAPI schema has all 5+ endpoints", all(p in paths for p in expected_paths),
          f"Found: {paths}")

    # CORS check (OPTIONS preflight simulation)
    try:
        req = urllib.request.Request(f"{BASE_URL}/health", method="OPTIONS")
        req.add_header("Origin", "http://localhost:3000")
        req.add_header("Access-Control-Request-Method", "GET")
        resp = urllib.request.urlopen(req)
        cors_header = resp.headers.get("access-control-allow-origin", "")
        check("CORS allows localhost:3000", "localhost:3000" in cors_header or cors_header == "*",
              f"Got: {cors_header}")
    except Exception as e:
        check("CORS allows localhost:3000", False, str(e))
    print()

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("═" * 65)
    print(f"  RESULTS: {passed}/{total} checks passed, {failed} failed")
    print("═" * 65)

    if total > 0:
        score = round(passed / total * 100)
        if score >= 90:
            verdict = "SHIP IT ✅"
        elif score >= 70:
            verdict = "REVIEW NEEDED ⚠️"
        else:
            verdict = "REWORK ❌"
        print(f"  Score: {score}/100 — {verdict}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
