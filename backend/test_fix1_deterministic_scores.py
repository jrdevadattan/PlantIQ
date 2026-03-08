"""
PlantIQ Fix #1 Test — Deterministic Anomaly Scores
====================================================
Verifies that the /dashboard/recent-batches endpoint returns
STABLE anomaly scores across multiple calls (no random jitter).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the dashboard router
from api.routes.dashboard import router as dashboard_router

app = FastAPI()
app.include_router(dashboard_router)
client = TestClient(app)

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name} — {detail}")


print("=" * 60)
print("Fix #1: Deterministic Anomaly Scores — Test Suite")
print("=" * 60)

# ── Test 1: Scores are stable across 3 consecutive calls ────
print("\n🔁 Test 1: Score stability across 3 calls")
responses = []
for i in range(3):
    r = client.get("/dashboard/recent-batches?limit=20")
    assert r.status_code == 200, f"Request {i+1} failed: {r.status_code}"
    responses.append(r.json())

# Compare anomaly scores across all 3 calls
scores_by_batch = {}
for call_idx, batches in enumerate(responses):
    for batch in batches:
        bid = batch["id"]
        score = batch["anomalyScore"]
        if bid not in scores_by_batch:
            scores_by_batch[bid] = []
        scores_by_batch[bid].append(score)

all_stable = True
unstable_batches = []
for bid, scores in scores_by_batch.items():
    if len(set(scores)) > 1:
        all_stable = False
        unstable_batches.append(f"{bid}: {scores}")

check(
    "All anomaly scores identical across 3 calls",
    all_stable,
    f"Unstable batches: {unstable_batches[:3]}"
)

# ── Test 2: Scores are within expected ranges per fault type ─
print("\n📊 Test 2: Scores within fault-type ranges")
batches = responses[0]

# We can't know fault types from the API response, but we can
# verify scores fall in valid ranges (0.02 – 0.90)
for batch in batches:
    score = batch["anomalyScore"]
    check(
        f"  {batch['id']}: score {score} in range [0.02, 0.90]",
        0.02 <= score <= 0.90,
        f"Score {score} is out of expected range"
    )

# ── Test 3: Normal batches (completed status) have low scores ─
print("\n🟢 Test 3: Status-score correlation")
completed = [b for b in batches if b["status"] == "completed"]
alert = [b for b in batches if b["status"] == "alert"]

if completed:
    # Most completed batches should have low anomaly scores (normal faults)
    avg_completed = sum(b["anomalyScore"] for b in completed) / len(completed)
    check(
        f"Completed batches avg score ({avg_completed:.2f}) < 0.40",
        avg_completed < 0.40,
        f"Completed batches have unexpectedly high avg score: {avg_completed:.2f}"
    )

if alert:
    # Alert batches should have higher anomaly scores
    avg_alert = sum(b["anomalyScore"] for b in alert) / len(alert)
    check(
        f"Alert batches avg score ({avg_alert:.2f}) > 0.30",
        avg_alert > 0.30,
        f"Alert batches have unexpectedly low avg score: {avg_alert:.2f}"
    )

# ── Test 4: Scores vary between different batches ────────────
print("\n🎯 Test 4: Score variety (not all the same)")
unique_scores = set(b["anomalyScore"] for b in batches)
check(
    f"At least 3 distinct scores among {len(batches)} batches",
    len(unique_scores) >= 3,
    f"Only {len(unique_scores)} distinct scores — too uniform"
)

# ── Test 5: Large request also stable ────────────────────────
print("\n📦 Test 5: Stability with limit=50")
r1 = client.get("/dashboard/recent-batches?limit=50")
r2 = client.get("/dashboard/recent-batches?limit=50")
assert r1.status_code == 200 and r2.status_code == 200

scores1 = {b["id"]: b["anomalyScore"] for b in r1.json()}
scores2 = {b["id"]: b["anomalyScore"] for b in r2.json()}

mismatches = [bid for bid in scores1 if scores1.get(bid) != scores2.get(bid)]
check(
    f"50-batch request stable (0 mismatches out of {len(scores1)})",
    len(mismatches) == 0,
    f"{len(mismatches)} mismatches: {mismatches[:3]}"
)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Results: {PASSED}/{PASSED + FAILED} passed, {FAILED} failed")
if FAILED == 0:
    print("🎉 ALL TESTS PASSED — Anomaly scores are deterministic!")
else:
    print(f"⚠️  {FAILED} test(s) failed")
print("=" * 60)

sys.exit(0 if FAILED == 0 else 1)
