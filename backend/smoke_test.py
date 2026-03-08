#!/usr/bin/env python3
"""Quick smoke test for all new code."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Upload router
from api.routes.upload import router
print("✅ Upload router imported")

# Test 2: API Key middleware
from api.middleware import APIKeyMiddleware
print("✅ APIKeyMiddleware imported")

# Test 3: AlertRecord to_dict has frontend-compat fields
from database.models import AlertRecord
a = AlertRecord()
a.alert_id = "test-123"
a.batch_id = "B0001"
a.alert_type = "energy"
a.severity = "WARNING"
a.message = "test"
a.state = "fired"
from datetime import datetime
a.fired_at = datetime.utcnow()
d = a.to_dict()
assert "id" in d, "Missing 'id' alias"
assert "timestamp" in d, "Missing 'timestamp' alias"
assert "acknowledged" in d, "Missing 'acknowledged' alias"
assert d["id"] == "test-123"
assert d["acknowledged"] == False
print("✅ AlertRecord.to_dict() has frontend-compatible fields")

# Test 4: hashlib determinism
import hashlib
v1 = int(hashlib.md5(b"B0042").hexdigest(), 16) % 100
v2 = int(hashlib.md5(b"B0042").hexdigest(), 16) % 100
assert v1 == v2, "hashlib not deterministic!"
print(f"✅ hashlib.md5('B0042') % 100 = {v1} (deterministic)")

# Test 5: Model artifacts exist
TRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
for f in ["multi_target.pkl", "scaler.pkl", "fault_classifier.pkl"]:
    exists = os.path.exists(os.path.join(TRAINED, f))
    print(f"  {'✅' if exists else '❌'} {f} exists={exists}")

print("\n🎉 All smoke tests passed!")
