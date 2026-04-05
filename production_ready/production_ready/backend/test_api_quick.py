"""Quick API smoke test — runs against running uvicorn."""
import sys
import requests

BASE = "http://localhost:8000"

def ok(label, resp, expect=200):
    if resp.status_code == expect:
        print(f"  [PASS] {label} -> {resp.status_code}")
        return resp.json()
    else:
        print(f"  [FAIL] {label} -> {resp.status_code}: {resp.text[:200]}")
        return {}

print("=" * 55)
print("API SMOKE TEST")
print("=" * 55)

# 1. Root
print("\n[1] Root endpoint")
r = requests.get(f"{BASE}/")
ok("GET /", r)

# 2. Health
print("\n[2] Health check")
r = requests.get(f"{BASE}/v1/health")
d = ok("GET /v1/health", r)
print(f"       status={d.get('status')}")

# 3. Docs
print("\n[3] Swagger docs")
r = requests.get(f"{BASE}/docs")
if r.status_code == 200:
    print(f"  [PASS] GET /docs -> {r.status_code} (HTML, len={len(r.text)})")
else:
    print(f"  [FAIL] GET /docs -> {r.status_code}")

# 4. Login
print("\n[4] Auth — login")
r = requests.post(f"{BASE}/v1/auth/login", json={"username": "admin", "password": "admin123"})
d = ok("POST /v1/auth/login", r)
token = d.get("access_token", "")
if not token:
    print("  ERROR: No token received — cannot proceed")
    sys.exit(1)
print(f"       token (first 40 chars): {token[:40]}...")
headers = {"Authorization": f"Bearer {token}"}

# 5. Predict
print("\n[5] Prediction")
payload = {
    "student_id": None,
    "sgpa": 8.5,
    "jee_score": 250,
    "marks_12": 88.0,
    "attendance": 92.0,
    "gender": "female"
}
r = requests.post(f"{BASE}/v1/predict", json=payload, headers=headers)
d = ok("POST /v1/predict", r)
print(f"       eligible={d.get('eligible')} | prob={d.get('probability')} | confidence={d.get('confidence')}")

# 6. Predictions list
print("\n[6] Predictions list")
r = requests.get(f"{BASE}/v1/predictions", headers=headers)
d = ok("GET /v1/predictions", r)
print(f"       returned {len(d) if isinstance(d, list) else '?'} records")

# 7. Fairness
print("\n[7] Fairness report")
r = requests.get(f"{BASE}/v1/fairness-report", headers=headers)
d = ok("GET /v1/fairness-report", r)
print(f"       overall_fair={d.get('overall_fair')}")

# 8. SHAP explanation (GET /v1/explain/{student_id} — skip if no student exists)
print("\n[8] SHAP explanation")
r = requests.get(f"{BASE}/v1/explain/1", headers=headers)
if r.status_code == 404:
    print("  [SKIP] GET /v1/explain/1 -> 404 (no student id=1 in DB — expected on fresh DB)")
else:
    d = ok("GET /v1/explain/1", r)
    print(f"       prob={d.get('probability')} | top features: {list((d.get('feature_contributions') or {}).keys())[:3]}")

# 9. Students list
print("\n[9] Students")
r = requests.get(f"{BASE}/v1/students", headers=headers)
ok("GET /v1/students", r)

print("\n" + "=" * 55)
print("SMOKE TESTS COMPLETE")
print("=" * 55)
