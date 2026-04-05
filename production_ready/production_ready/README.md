# 🚀 Scholarship Dashboard - Quick Setup & Test

This README provides direct, copy-pasteable terminal commands to get the system running and verified on this machine.

---

### 📦 1. Fast Setup (Install Requirements)

Since `loguru` was missing from your current environment, run this first to install all necessary ML and API libraries:

```powershell
# From the current backend folder
cd C:\project\production_ready\production_ready\backend
python -m pip install -r requirements.txt
```

---

### 🧪 2. Direct ML Validation (Run the Test Script)

Verify the classification model, hot-reloading, and feature scaling with this one command:

```powershell
# From the backend folder
python test_predictions_e2e.py
```

---

### 📡 3. Terminal API Test (Manual Validation)

If your server is already running (use the uvicorn command below if not), run these to perform a manual test using PowerShell:

**Step A: Get Authentication Token (Admin)**
```powershell
$body = '{"username":"admin","password":"admin123","role":"admin"}'
$reg = Invoke-RestMethod -Uri "http://localhost:8000/v1/auth/register" -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue
$login = Invoke-RestMethod -Uri "http://localhost:8000/v1/auth/login" -Method Post -Body $body -ContentType "application/json"
$headers = @{Authorization = "Bearer $($login.access_token)"}
```

**Step B: Test Prediction (Eligibility Result)**
```powershell
# Test high-performing student
$payload = '{"sgpa":9.5,"jee_score":310,"marks_12":98.0,"attendance":99.0,"gender":"female"}'
Invoke-RestMethod -Uri "http://localhost:8000/v1/predict" -Method Post -Body $payload -ContentType "application/json" -Headers $headers
```

---

### 🛠️ 4. Server Execution

To start the actual API server from scratch with hot-reload enabled:

```powershell
# From the backend folder
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 📂 Directory Reference

- **Backend Logic & ML**: `C:\project\production_ready\production_ready\backend\app`
- **Frontend Dashboard**: `C:\project\production_ready\production_ready\frontend\index.html`
- **Models**: `C:\project\production_ready\production_ready\backend\storage\models`
