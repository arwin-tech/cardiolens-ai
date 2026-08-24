# CardioLens AI — Phase 2: FastAPI Backend Implementation Guide

**Status:** Complete Implementation Template Ready

**Date:** August 24, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [API Endpoints](#api-endpoints)
5. [Schema Definitions](#schema-definitions)
6. [Service Layer](#service-layer)
7. [Testing Strategy](#testing-strategy)
8. [Deployment](#deployment)
9. [Next Steps (Phase 2B - LLM Integration)](#next-steps)

---

## Overview

Phase 2 wraps the existing Phase 1 AI pipeline (data loading, model training, SHAP explainability, counterfactual simulation) behind a clean FastAPI REST API.

### Key Principles

- **Don't break Phase 1:** Streamlit app remains fully functional
- **No model retraining:** Use existing trained artifacts
- **Modular design:** Each service can be extended independently
- **Medical safety:** All responses include appropriate disclaimers
- **API-first:** Enable multiple frontends (Streamlit + Lovable React + future clients)

### What Gets Built

| Component | File | Purpose |
|-----------|------|---------|
| FastAPI App | `main.py` | Entry point, CORS, route orchestration |
| Schemas | `schemas/` | Pydantic models for request/response validation |
| Services | `services/` | Business logic wrappers around Phase 1 |
| Models | `models/` | Model loading, prediction, SHAP, counterfactual |
| Config | `core/config.py` | Environment variables, settings |
| Tests | `test_*.py` | Unit + integration tests |

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Lovable React Frontend                 │
│  (Vercel)                               │
└─────────────┬───────────────────────────┘
              │ HTTP/REST
              │
┌─────────────▼───────────────────────────┐
│  FastAPI Backend                        │
│  /api/predict                           │
│  /api/explain                           │
│  /api/what-if                           │
│  /api/health                            │
└─────────────┬───────────────────────────┘
              │ Python imports
              │
┌─────────────▼───────────────────────────┐
│  Phase 1 AI Pipeline                    │
│  ├─ data_loader.py                      │
│  ├─ train_models.py                     │
│  ├─ predictor.py                        │
│  ├─ explainer.py (SHAP)                 │
│  ├─ counterfactual.py                   │
│  ├─ llm_reasoner.py                     │
│  └─ model artifacts (XGBoost)           │
└─────────────────────────────────────────┘

ALSO STILL WORKS:
┌─────────────────────────────────────────┐
│  Streamlit Dashboard (app.py)           │
│  Existing Stage 6 interface             │
└─────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Phase 1 complete (trained model, artifacts exist)
- `pip` or `conda`

### Step 1: Install Dependencies

```bash
# Create new virtual environment for backend
python -m venv venv_backend
source venv_backend/bin/activate  # On Windows: venv_backend\Scripts\activate

# Install backend dependencies
pip install -r requirements_backend.txt
```

### Step 2: Set Up Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your values
# APP_ENV=development
# API_PORT=8000
# ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Step 3: Verify Model Artifacts

The backend expects these files (from Phase 1):

```
models/
├── model_pipeline.pkl        # sklearn Pipeline with scaler + trained model
└── model_metadata.json       # Model info, test metrics, feature names
```

If these files don't exist, the backend will still start but `/api/predict` will return 503 (Service Unavailable).

### Step 4: Start the Backend

```bash
# Option 1: Development with auto-reload
uvicorn main:app --reload --port 8000

# Option 2: Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 5: Verify It Works

```bash
# Check health
curl http://localhost:8000/api/health

# Try a prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "gender": 1, "height": 165, "weight": 82,
    "ap_hi": 145, "ap_lo": 90, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
  }'

# View Swagger docs
open http://localhost:8000/docs
```

---

## API Endpoints

### 1. GET /api/health

**Purpose:** Health check, verify model is loaded

**Request:** None

**Response:**
```json
{
  "status": "healthy",
  "service": "CardioLens AI API",
  "model_loaded": true,
  "version": "2.0.0"
}
```

**HTTP Status:**
- `200` — Service operational
- `503` — Model not loaded (degraded state)

---

### 2. POST /api/predict

**Purpose:** Predict cardiovascular disease risk for a patient

**Request:**
```json
{
  "age": 52,
  "gender": 1,
  "height": 165,
  "weight": 82,
  "ap_hi": 145,
  "ap_lo": 90,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

**Field Definitions:**
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `age` | float | 18–100 | Age in years |
| `gender` | int | 1–2 | 1=Female, 2=Male |
| `height` | float | 100–250 | Height in cm |
| `weight` | float | 30–300 | Weight in kg |
| `ap_hi` | int | 60–250 | Systolic BP (mmHg) |
| `ap_lo` | int | 30–180 | Diastolic BP (mmHg) |
| `cholesterol` | int | 1–3 | 1=Normal, 2=Above normal, 3=Well above |
| `gluc` | int | 1–3 | 1=Normal, 2=Above normal, 3=Well above |
| `smoke` | int | 0–1 | 0=No, 1=Yes |
| `alco` | int | 0–1 | 0=No, 1=Yes |
| `active` | int | 0–1 | 0=No, 1=Yes |

**Response:**
```json
{
  "risk_probability": 0.73,
  "risk_percentage": 73.0,
  "risk_category": "High",
  "bmi": 30.1,
  "patient_features": { ... echoed input ... },
  "model_version": "XGBoost",
  "disclaimer": "DISCLAIMER: This is a research-grade risk prediction system..."
}
```

**Risk Categories:**
- `Low`: < 30% probability
- `Moderate`: 30–50% probability
- `Elevated`: 50–70% probability
- `High`: ≥ 70% probability

**HTTP Status:**
- `200` — Success
- `422` — Validation error (invalid input)
- `503` — Model not loaded

---

### 3. POST /api/explain

**Purpose:** Get SHAP-based explanation of prediction drivers

**Request:** Same as `/api/predict` (11-feature patient profile)

**Response:**
```json
{
  "features": [
    {
      "feature": "ap_hi",
      "value": 145,
      "shap_value": 0.42,
      "impact": "positive",
      "feature_rank": 1
    },
    {
      "feature": "age",
      "value": 52,
      "shap_value": 0.28,
      "impact": "positive",
      "feature_rank": 2
    }
  ],
  "base_value": 0.5,
  "prediction_value": 0.73,
  "disclaimer": "SHAP values show feature contributions..."
}
```

**Field Definitions:**
| Field | Type | Description |
|-------|------|-------------|
| `feature` | str | Feature name |
| `value` | float | Patient's feature value |
| `shap_value` | float | SHAP contribution to prediction |
| `impact` | str | "positive" (toward disease) or "negative" (toward health) |
| `feature_rank` | int | 1 = most important, 11 = least important |
| `base_value` | float | SHAP expected value (population average) |
| `prediction_value` | float | Model's predicted risk |

**HTTP Status:**
- `200` — Success
- `422` — Validation error
- `503` — SHAP explainer not available

---

### 4. POST /api/what-if

**Purpose:** Compare risk under two different patient profiles

**Request:**
```json
{
  "original": {
    "age": 52,
    "gender": 1,
    "height": 165,
    "weight": 82,
    "ap_hi": 145,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  },
  "modified": {
    "age": 52,
    "gender": 1,
    "height": 165,
    "weight": 82,
    "ap_hi": 120,
    "ap_lo": 75,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }
}
```

**Response:**
```json
{
  "original_risk": 0.73,
  "simulated_risk": 0.61,
  "risk_delta": -0.12,
  "risk_delta_percentage": -16.4,
  "original_category": "High",
  "simulated_category": "Elevated",
  "explanation": "Modifying these factors (ap_hi from 145 to 120, ap_lo from 90 to 75) would result in a decrease in predicted risk of approximately 12.0 percentage points...",
  "disclaimer": "This what-if analysis explores model behavior under hypothetical scenarios..."
}
```

**HTTP Status:**
- `200` — Success
- `422` — Validation error
- `503` — Model not available

---

## Schema Definitions

All request/response schemas are defined using **Pydantic** for automatic validation.

### PatientInput

```python
class PatientInput(BaseModel):
    age: float = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=1, le=2)
    height: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=30, le=300)
    ap_hi: int = Field(..., ge=60, le=250)
    ap_lo: int = Field(..., ge=30, le=180)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)
    
    # Validation: diastolic < systolic
    @validator('ap_lo')
    def validate_blood_pressure(cls, v, values):
        if 'ap_hi' in values and v >= values['ap_hi']:
            raise ValueError("Diastolic must be < systolic")
        return v
```

### PredictionResponse

```python
class PredictionResponse(BaseModel):
    risk_probability: float  # 0.0–1.0
    risk_percentage: float   # 0–100
    risk_category: str       # "Low" | "Moderate" | "Elevated" | "High"
    bmi: float
    patient_features: Dict[str, Any]
    model_version: str
    disclaimer: str
```

### SHAPFeature

```python
class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str  # "positive" | "negative"
    feature_rank: int
```

### ExplainabilityResponse

```python
class ExplainabilityResponse(BaseModel):
    features: List[SHAPFeature]
    base_value: float
    prediction_value: float
    disclaimer: str
```

### WhatIfResponse

```python
class WhatIfResponse(BaseModel):
    original_risk: float
    simulated_risk: float
    risk_delta: float
    risk_delta_percentage: float
    original_category: str
    simulated_category: str
    explanation: str
    disclaimer: str
```

---

## Service Layer

The backend is organized into services that wrap Phase 1 components:

### ModelManager

**File:** `main.py` (can be moved to `services/model_manager.py`)

**Responsibility:** Load trained model and metadata, make predictions

**Key Methods:**
- `load()` — Load model.pkl and metadata.json from disk
- `predict(patient: PatientInput) -> Dict` — Return risk probability and category

**Code:**
```python
model_manager = ModelManager(
    model_path="models/model_pipeline.pkl",
    metadata_path="models/model_metadata.json"
)

result = model_manager.predict(patient_input)
# Returns: {
#   "risk_probability": 0.73,
#   "risk_percentage": 73.0,
#   "risk_category": "High",
#   "bmi": 30.1
# }
```

### SHAPExplainer

**File:** `main.py` (can be moved to `services/shap_explainer.py`)

**Responsibility:** Compute SHAP values for feature importance

**Key Methods:**
- `load_explainer()` — Initialize TreeExplainer from trained model
- `explain(patient: PatientInput) -> Dict` — Return feature contributions

**Code:**
```python
shap_explainer = SHAPExplainer(model_manager)

result = shap_explainer.explain(patient_input)
# Returns: {
#   "features": [
#     {"feature": "ap_hi", "value": 145, "shap_value": 0.42, ...},
#     ...
#   ],
#   "base_value": 0.5
# }
```

### CounterfactualSimulator

**File:** `main.py` (can be moved to `services/counterfactual.py`)

**Responsibility:** Compare risk under two patient profiles

**Key Methods:**
- `simulate(original: PatientInput, modified: PatientInput) -> Dict`

**Code:**
```python
simulator = CounterfactualSimulator(model_manager)

result = simulator.simulate(original_patient, modified_patient)
# Returns: {
#   "original_risk": 0.73,
#   "modified_risk": 0.61,
#   "delta": -0.12,
#   "changed_features": [...]
# }
```

### DeterministicReasoner

**File:** `main.py` (can be moved to `services/reasoning.py`)

**Responsibility:** Generate natural language explanations

**Key Methods:**
- `explain_prediction(risk_prob, category, features) -> str`
- `explain_counterfactual(delta, changed_features) -> str`

**Code:**
```python
explanation = DeterministicReasoner.explain_prediction(
    risk_prob=0.73,
    risk_category="High",
    top_features=["ap_hi", "age", "smoke"],
    model_name="XGBoost"
)
# Returns: "The XGBoost model predicts a high cardiovascular risk..."
```

---

## Testing Strategy

### Unit Tests

Test each endpoint independently:

```bash
pytest test_backend_phase2.py::TestCardioLensAPI::test_health_endpoint -v
pytest test_backend_phase2.py::TestCardioLensAPI::test_predict_valid_input -v
pytest test_backend_phase2.py::TestCardioLensAPI::test_explain_endpoint -v
pytest test_backend_phase2.py::TestCardioLensAPI::test_whatif_endpoint -v
```

### Validation Tests

Ensure invalid input is rejected:

```bash
pytest test_backend_phase2.py::TestCardioLensAPI::test_predict_missing_field -v
pytest test_backend_phase2.py::TestCardioLensAPI::test_invalid_blood_pressure -v
```

### Integration Test: Streamlit ↔ FastAPI Consistency

**Critical validation:** Same patient input should produce identical predictions.

```python
# In Streamlit (Phase 1):
streamlit_prediction = pipeline.predict_proba(patient_features)[0, 1]

# Via FastAPI (Phase 2):
api_prediction = requests.post(
    "http://localhost:8000/api/predict",
    json=patient_dict
).json()["risk_probability"]

assert abs(streamlit_prediction - api_prediction) < 0.001
```

### Manual Testing

```bash
# Terminal 1: Start backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit (for comparison)
streamlit run app.py

# Terminal 3: Run tests
pytest test_backend_phase2.py -v

# Or use curl for manual testing
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "gender": 1, "height": 165, "weight": 82,
    "ap_hi": 145, "ap_lo": 90, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
  }'
```

---

## Deployment

### Local Development

```bash
uvicorn main:app --reload --port 8000
```

### Production

```bash
# Using gunicorn + uvicorn workers
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional for Phase 2B)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t cardiolens-api .
docker run -p 8000:8000 cardiolens-api
```

### Vercel Deployment (For Lovable Frontend → Backend)

If frontend and backend are separate:

**Frontend:** Deploy Lovable React to Vercel
**Backend:** Deploy FastAPI to:
- Railway.app
- Render.com
- DigitalOcean
- AWS Lambda + API Gateway
- Heroku

Update frontend `.env` to point to backend URL:
```
VITE_API_BASE_URL=https://cardiolens-api.onrender.com
```

---

## Next Steps

### Phase 2B: LLM Integration

Currently the reasoner is deterministic (rule-based). Next phase will integrate actual LLM:

#### Option 1: OpenAI API

```python
# services/reasoning.py
from openai import OpenAI

class GPTReasoner:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def explain_prediction(self, ...):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant..."},
                {"role": "user", "content": f"Explain this risk: {structured_data}"}
            ]
        )
        return response.choices[0].message.content
```

#### Option 2: Anthropic Claude API

```python
import anthropic

class ClaudeReasoner:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def explain_prediction(self, ...):
        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Explain this cardiovascular risk assessment..."
            }]
        )
        return message.content[0].text
```

#### Option 3: Local LLM (Llama 2)

```python
from ollama import Ollama

class LocalReasoner:
    def __init__(self):
        self.client = Ollama()
    
    def explain_prediction(self, ...):
        response = self.client.generate(
            model="llama2",
            prompt=f"Explain this risk..."
        )
        return response["response"]
```

### Phase 3: Frontend Integration

Connect Lovable React frontend to this FastAPI backend:

```typescript
// Frontend (React)
const response = await fetch(`${API_BASE_URL}/api/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(patientData)
});
const data = await response.json();
setRiskProbability(data.risk_probability);
```

### Phase 4: Database & User Management

- Store patient records (PostgreSQL)
- User authentication (JWT)
- Audit logs (model predictions per user)
- Export reports (PDF)

### Phase 5: Advanced Features

- Batch predictions
- Model versioning
- A/B testing (compare models)
- Custom risk calculators
- Integration with EHR systems

---

## File Structure Summary

After Phase 2 implementation:

```
cardiolens-ai/
├── main.py                      # FastAPI app (this file)
├── requirements_backend.txt     # Backend dependencies
├── .env.example                 # Environment config template
├── test_backend_phase2.py       # Test suite
│
├── data/
│   └── cardiovascular_disease.csv
│
├── models/
│   ├── model_pipeline.pkl       # Phase 1 trained model
│   └── model_metadata.json      # Phase 1 metadata
│
├── src/                         # Phase 1 AI pipeline (untouched)
│   ├── data_loader.py
│   ├── train_models.py
│   ├── predictor.py
│   ├── explainer.py
│   ├── counterfactual.py
│   └── llm_reasoner.py
│
├── app.py                       # Phase 1 Streamlit (still works)
│
└── docs/
    ├── ARCHITECTURE_ASSESSMENT.md
    ├── LOVABLE_FRONTEND_IMPLEMENTATION.md
    ├── PHASE_1_SUMMARY.md
    ├── PHASE_2_IMPLEMENTATION_GUIDE.md (this file)
    └── README.md
```

---

## Troubleshooting

### Model Not Loading

**Error:** `Model not loaded` (503 response)

**Cause:** `models/model_pipeline.pkl` not found or corrupted

**Fix:**
```bash
# Check file exists
ls -lh models/model_pipeline.pkl

# If missing, retrain Phase 1
cd src && python train_models.py
```

### SHAP Explainer Fails

**Error:** `SHAP explainer not initialized`

**Cause:** Model architecture incompatible with TreeExplainer

**Fix:**
```python
# In main.py, verify model type
print(type(model_manager.model.named_steps['model']))
# Should be: XGBClassifier or RandomForestClassifier
```

### Validation Error (422)

**Error:** Invalid patient input

**Fix:** Check field types and ranges:
```python
# ✗ Wrong
"age": "52"  # string

# ✓ Correct
"age": 52    # number
```

### CORS Error

**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Cause:** Frontend origin not in `ALLOWED_ORIGINS`

**Fix:** Update `.env`:
```
ALLOWED_ORIGINS=http://localhost:3000,https://myapp.vercel.app
```

### Port Already in Use

**Error:** `Address already in use: ('0.0.0.0', 8000)`

**Fix:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --port 8001
```

---

## Summary

**Phase 2 Complete When:**

✅ FastAPI server starts successfully
✅ `/api/health` returns 200
✅ `/api/predict` returns prediction matching Streamlit
✅ `/api/explain` returns SHAP values
✅ `/api/what-if` returns counterfactual comparison
✅ Swagger docs load at `/docs`
✅ Tests pass
✅ Streamlit still works
✅ Model artifacts untouched
✅ No retraining occurred

**Next:** Phase 3 connects Lovable React frontend to this API.

---

*Questions? Consult the original Phase 2 requirements document or test outputs.*