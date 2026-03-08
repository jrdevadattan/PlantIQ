"""One-shot audit script for PlantIQ backend."""
import sys, os, json

print("=== 1. Import Tests ===")
modules = [
    "api.routes.health", "api.routes.predict", "api.routes.anomaly",
    "api.routes.explain", "api.routes.features", "api.routes.golden_signature",
    "api.routes.targets", "api.routes.hackathon", "api.routes.dashboard",
    "api.routes.cost", "api.routes.audit", "api.routes.recommend"
]
for m in modules:
    try:
        __import__(m)
        print(f"  OK {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")

print("\n=== 2. Model Artifacts ===")
trained_dir = "models/trained"
for f in sorted(os.listdir(trained_dir)):
    path = os.path.join(trained_dir, f)
    size = os.path.getsize(path)
    print(f"  {f} ({size} bytes)")

pkl_files = [f for f in os.listdir(trained_dir) if f.endswith((".pkl", ".pt", ".pth"))]
print(f"\n  Binary models: {pkl_files if pkl_files else 'NONE'}")

print("\n=== 3. Database ===")
db_path = "plantiq.db"
print(f"  {'EXISTS' if os.path.exists(db_path) else 'MISSING'} ({os.path.getsize(db_path) if os.path.exists(db_path) else 0} bytes)")

print("\n=== 4. Data Files ===")
for csv_f in ["data/batch_data.csv", "data/train_processed.csv", "data/test_processed.csv"]:
    if os.path.exists(csv_f):
        with open(csv_f) as fh:
            lines = sum(1 for _ in fh) - 1
        print(f"  {csv_f}: {lines} rows")
    else:
        print(f"  {csv_f}: MISSING")

print("\n=== 5. Power Curves ===")
pc_dir = "data/power_curves"
npy_count = len([f for f in os.listdir(pc_dir) if f.endswith(".npy")])
print(f"  {npy_count} .npy files")

print("\n=== 6. Endpoint Stress Test ===")
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
from api.routes.dashboard import router as dash_r
from api.routes.cost import router as cost_r
from api.routes.health import router as health_r
app.include_router(dash_r)
app.include_router(cost_r)
app.include_router(health_r)
c = TestClient(app)

tests = [
    ("GET", "/health", None),
    ("GET", "/dashboard/summary", None),
    ("GET", "/dashboard/energy-daily?days=7", None),
    ("GET", "/dashboard/energy-daily?days=0", None),
    ("GET", "/dashboard/energy-daily?days=-1", None),
    ("GET", "/dashboard/energy-daily?days=999", None),
    ("GET", "/dashboard/shift-performance", None),
    ("GET", "/dashboard/batches/recent?limit=5", None),
    ("GET", "/dashboard/batches/recent?limit=0", None),
    ("GET", "/dashboard/batches/recent?limit=-1", None),
    ("GET", "/dashboard/batches/recent?limit=99999", None),
    ("GET", "/cost/config", None),
    ("POST", "/cost/translate", {"energy_kwh": 45.0}),
    ("POST", "/cost/translate", {"energy_kwh": 0}),
    ("POST", "/cost/translate", {"energy_kwh": -10}),
    ("POST", "/cost/translate", {}),
    ("POST", "/cost/translate", {"energy_kwh": 999999}),
    ("POST", "/cost/translate-batch", {"batch_id": "TEST", "energy_kwh": 45}),
    ("POST", "/cost/translate-batch", {"batch_id": "", "energy_kwh": 45}),
]

for method, path, body in tests:
    try:
        if method == "GET":
            r = c.get(path)
        else:
            r = c.post(path, json=body)
        status = "OK" if r.status_code < 500 else "FAIL"
        print(f"  {status} {method} {path} -> {r.status_code} (body={body})")
    except Exception as e:
        print(f"  CRASH {method} {path} -> {type(e).__name__}: {e}")

# Edge case: malformed JSON
try:
    r = c.post("/cost/translate", content="not json", headers={"content-type": "application/json"})
    print(f"  {'OK' if r.status_code < 500 else 'FAIL'} POST /cost/translate malformed JSON -> {r.status_code}")
except Exception as e:
    print(f"  CRASH malformed JSON -> {e}")

print("\n=== 7. Hash Determinism Check ===")
# Python hash() is randomized across sessions
v1 = hash("B0001") % 100
v2 = hash("B0001") % 100
print(f"  Same session: hash('B0001')%100 = {v1}, {v2} (match={v1==v2})")
print(f"  WARNING: hash() is randomized across Python sessions (PYTHONHASHSEED)")
print(f"  PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'random')}")
