"""
CardioLens AI Backend v2.2.0

FastAPI service that scores cardiovascular risk with XGBoost models, explains each
prediction with exact per-patient SHAP values, gives a calibrated risk range
(Venn-Abers), suggests the smallest changes that lower risk, and keeps
privacy-safe monitoring counts.
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, run from backend/ or the repo root
from counterfactuals import suggest as suggest_counterfactuals  # noqa: E402
from monitoring import MonitoringStore  # noqa: E402
from venn_abers import VennAbersGrid  # noqa: E402

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_pipeline.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# Override from environment if needed
if os.getenv("MODEL_PATH"):
    MODEL_PATH = Path(os.getenv("MODEL_PATH"))
if os.getenv("METADATA_PATH"):
    METADATA_PATH = Path(os.getenv("METADATA_PATH"))

logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Metadata path: {METADATA_PATH}")

# ============================================================================
# CORS CONFIGURATION (FIXED)
# ============================================================================
# Parse allowed origins from environment
ALLOWED_ORIGINS_STR = os.getenv(
    "ALLOWED_ORIGINS",
    "https://cardiolens-ai.vercel.app,http://localhost:5173,http://localhost:4173,http://localhost:3000"
)
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_STR.split(",") if o.strip()]
# This project's Vercel preview deployments (not every *.vercel.app site) and local dev servers
ALLOWED_ORIGIN_REGEX = r"https://cardiolens-ai(-[a-z0-9-]+)?\.vercel\.app|http://localhost:\d+"

VERSION = "2.2.0"
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
NHANES_FEATURES = FEATURES + ["bp_meds", "lipid_meds"]
DEFAULT_MODEL = "kaggle"
MONITORED_ENDPOINTS = {"predict", "explain", "what-if", "counterfactuals", "batch-predict"}
MAX_UPLOAD_BYTES = 2_000_000   # batch CSV size limit (protects the small free-tier server)
MAX_BATCH_ROWS = 5_000

logger.info(f"CORS Allowed Origins: {ALLOWED_ORIGINS}")

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class PatientInput(BaseModel):
    """Patient clinical data for risk prediction"""
    age: float = Field(..., ge=18, le=100, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="1=Female, 2=Male")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=30, le=300, description="Weight in kg")
    ap_hi: int = Field(..., ge=60, le=250, description="Systolic blood pressure (mmHg)")
    ap_lo: int = Field(..., ge=30, le=180, description="Diastolic blood pressure (mmHg)")
    cholesterol: int = Field(..., ge=1, le=3, description="1=Normal, 2=Above normal, 3=Well above")
    gluc: int = Field(..., ge=1, le=3, description="1=Normal, 2=Above normal, 3=Well above")
    smoke: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    alco: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    active: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    # Used only by the US population (NHANES) model; optional so existing clients keep working
    bp_meds: int = Field(0, ge=0, le=1, description="Taking blood-pressure medication: 0=No, 1=Yes")
    lipid_meds: int = Field(0, ge=0, le=1, description="Taking cholesterol medication: 0=No, 1=Yes")

    @field_validator("ap_lo")
    @classmethod
    def validate_blood_pressure(cls, v: int, info) -> int:
        """Ensure diastolic BP < systolic BP"""
        if "ap_hi" in info.data and v >= info.data["ap_hi"]:
            raise ValueError("Diastolic BP must be less than systolic BP")
        return v


class PredictionResponse(BaseModel):
    """Model risk prediction output"""
    model_config = {"protected_namespaces": ()}

    risk_probability: float = Field(..., ge=0, le=1)
    risk_percentage: float = Field(..., ge=0, le=100)
    risk_category: str
    bmi: float
    patient_features: Dict[str, Any]
    model_version: str
    disclaimer: str
    model_id: str = DEFAULT_MODEL
    model_name: str = ""
    risk_range: Optional[List[float]] = None  # calibrated range in percent (Venn-Abers)


class SHAPFeature(BaseModel):
    """SHAP feature importance component"""
    feature: str
    value: float
    shap_value: float
    impact: str
    feature_rank: int


class ExplainabilityResponse(BaseModel):
    """SHAP-based model explanation"""
    features: List[SHAPFeature]
    base_value: float
    prediction_value: float
    disclaimer: str


class WhatIfRequest(BaseModel):
    """Counterfactual scenario: the same patient before and after a change.
    Both go through the full PatientInput validation."""
    original: PatientInput
    modified: PatientInput


class HealthResponse(BaseModel):
    """Service health status"""
    model_config = {"protected_namespaces": ()}

    status: str
    service: str
    model_loaded: bool
    version: str


# ============================================================================
# SERVICE LAYER: MODEL MANAGER
# ============================================================================

def risk_category(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.5:
        return "Moderate"
    if p < 0.7:
        return "Elevated"
    return "High"


class ModelBundle:
    """One deployed model plus its uncertainty ranges and monitoring reference."""

    def __init__(self, model_id: str, name: str, model_path: Path, metadata_path: Path, features: List[str],
                 uncertainty_path: Path, reference_path: Path, default_metadata: Optional[Dict[str, Any]] = None):
        self.id, self.name, self.features = model_id, name, features
        self.model_path, self.metadata_path = model_path, metadata_path
        self.uncertainty_path, self.reference_path = uncertainty_path, reference_path
        self.default_metadata = default_metadata or {}
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.uncertainty: Optional[VennAbersGrid] = None
        self.reference: Optional[Dict[str, Dict[str, float]]] = None
        self.is_loaded = False
        self.load()

    def load(self):
        try:
            if not self.model_path.exists():
                logger.warning(f"✗ {self.id}: model file not found at {self.model_path}")
                return
            if self.model_path.suffix == ".json":  # XGBoost's version-safe format
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
            else:
                self.model = joblib.load(self.model_path)
            self.metadata = (json.loads(self.metadata_path.read_text()) if self.metadata_path.exists()
                             else dict(self.default_metadata))
            if self.uncertainty_path.exists():
                self.uncertainty = VennAbersGrid.load(self.uncertainty_path)
            if self.reference_path.exists():
                self.reference = json.loads(self.reference_path.read_text())
            self.is_loaded = True
            logger.info(f"✓ {self.id}: model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"✗ {self.id}: failed to load model: {e}")
            self.is_loaded = False

    def _raw_model(self):
        if hasattr(self.model, "named_steps"):
            return (self.model.named_steps.get("model") or self.model.named_steps.get("classifier")
                    or self.model.steps[-1][1])
        return self.model

    def row(self, patient: "PatientInput") -> List[float]:
        values = patient.model_dump()
        return [float(values[f]) for f in self.features]

    def predict_matrix(self, X) -> np.ndarray:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict(self, patient: "PatientInput") -> Dict[str, Any]:
        risk = float(self.predict_matrix([self.row(patient)])[0])
        result = {
            "risk_probability": risk,
            "risk_percentage": risk * 100,
            "risk_category": risk_category(risk),
            "bmi": round(patient.weight / (patient.height / 100.0) ** 2, 1),
            "risk_range": None,
        }
        if self.uncertainty:
            lo, hi = self.uncertainty.interval(risk)
            result["risk_range"] = [round(lo * 100, 1), round(hi * 100, 1)]
        return result

    def explain(self, patient: "PatientInput") -> Dict[str, Any]:
        """Exact per-patient SHAP values (TreeSHAP), computed natively by XGBoost.

        pred_contribs gives the same values as shap.TreeExplainer without the shap library.
        Values are in log-odds; with the baseline they add up exactly to the prediction.
        """
        raw = self._raw_model()
        if not hasattr(raw, "get_booster"):
            raise RuntimeError("SHAP explanations require an XGBoost model")
        X = pd.DataFrame([self.row(patient)], columns=self.features)
        contribs = raw.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
        shap_values, bias = contribs[:-1], float(contribs[-1])
        explanations = [
            {"feature": name, "value": float(value), "shap_value": round(float(sv), 4),
             "impact": "positive" if sv > 0 else "negative", "feature_rank": 0}
            for name, value, sv in zip(self.features, X.iloc[0], shap_values)
        ]
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        for idx, exp in enumerate(explanations):
            exp["feature_rank"] = idx + 1
        return {
            "features": explanations,
            "base_value": round(float(1.0 / (1.0 + np.exp(-bias))), 4),
            "prediction_value": self.predict(patient)["risk_probability"],
            "disclaimer": "SHAP values (log-odds) explain this model's prediction for this patient. Not a medical diagnosis.",
        }

    def info(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.metadata.get("description", ""),
            "features": self.features,
            "test_metrics": self.metadata.get("test_metrics", {}),
            "uncertainty": ({"method": "Venn-Abers", "calibration_patients": self.uncertainty.calibration_patients}
                            if self.uncertainty else None),
            "default": self.id == DEFAULT_MODEL,
        }


# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================

MODELS: Dict[str, ModelBundle] = {
    "kaggle": ModelBundle(
        "kaggle", "Clinic model", MODEL_PATH, METADATA_PATH, FEATURES,
        MODELS_DIR / "uncertainty_kaggle.json", MODELS_DIR / "reference_kaggle.json",
        default_metadata={"model_name": "XGBoost Cardiovascular Risk",
                          "test_metrics": {"accuracy": 0.7324, "precision": 0.7508, "recall": 0.6951,
                                           "f1": 0.7219, "roc_auc": 0.7978}},
    ),
}
MODELS["kaggle"].metadata.setdefault(
    "description", "Trained on 70,000 patients from the Kaggle Cardiovascular Disease dataset.")
_nhanes = ModelBundle(
    "nhanes", "US population model", MODELS_DIR / "nhanes_model.json", MODELS_DIR / "nhanes_metadata.json",
    NHANES_FEATURES, MODELS_DIR / "uncertainty_nhanes.json", MODELS_DIR / "reference_nhanes.json",
)
if _nhanes.is_loaded:  # appears once src/train_nhanes_model.py has been run
    MODELS["nhanes"] = _nhanes

model_manager = MODELS["kaggle"]  # the default model, under its original name
monitor = MonitoringStore()


def get_model(model_id: str) -> ModelBundle:
    bundle = MODELS.get(model_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}")
    if not bundle.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available. Backend running in degraded mode.")
    return bundle


MODEL_QUERY = Query(DEFAULT_MODEL, description="Which model to use: 'kaggle' (default) or 'nhanes' if available")


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager"""
    logger.info("📡 CardioLens AI Backend starting up...")
    
    # Ensure model is loaded on startup
    if not model_manager.is_loaded:
        logger.warning("⚠️  Model not loaded on startup, attempting reload...")
        model_manager.load()
    
    if model_manager.is_loaded:
        logger.info("✅ Model ready for predictions")
    else:
        logger.warning("⚠️  Starting in degraded mode (model file missing)")
    
    yield
    
    logger.info("📡 CardioLens AI Backend shutting down...")


# Create FastAPI app instance
app = FastAPI(
    title="CardioLens AI API",
    description="Explainable cardiovascular disease risk assessment backend",
    version=VERSION,
    lifespan=lifespan
)

# ============================================================================
# CORS MIDDLEWARE (FIXED)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,  # the API is stateless: no cookies or auth headers
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight responses
)

logger.info("✅ CORS middleware configured successfully")


@app.middleware("http")
async def time_requests(request: Request, call_next):
    """Record response times per endpoint (as coarse buckets) for the monitoring page."""
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    endpoint = path[len("/api/"):] if path.startswith("/api/") else ""
    if request.method == "POST" and endpoint in MONITORED_ENDPOINTS:
        ms = (time.perf_counter() - start) * 1000
        if monitor.persistent:  # don't make the caller wait on a remote database
            asyncio.get_running_loop().run_in_executor(None, monitor.record_request, endpoint, ms)
        else:
            monitor.record_request(endpoint, ms)
    return response

# ============================================================================
# ROUTE HANDLERS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "CardioLens AI API",
        "version": VERSION,
        "status": "online",
        "endpoints": {
            "health": "/api/health (may be blocked by ad-blockers)",
            "status": "/api/status (recommended, ad-blocker bypass)",
            "predict": "POST /api/predict",
            "explain": "POST /api/explain",
            "what_if": "POST /api/what-if",
            "batch_predict": "POST /api/batch-predict",
            "counterfactuals": "POST /api/counterfactuals",
            "models": "GET /api/models",
            "monitoring": "GET /api/monitoring",
        },
        "docs": "/docs",
        "model_status": "loaded" if model_manager.is_loaded else "degraded"
    }


@app.get("/api/health", response_model=HealthResponse)
@app.get("/api/status", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health_check():
    """Service health. /api/status is the same check under a URL ad-blockers don't block."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        service="CardioLens AI API",
        model_loaded=model_manager.is_loaded,
        version=VERSION,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput, background_tasks: BackgroundTasks, model: str = MODEL_QUERY):
    """Cardiovascular risk for one patient, with a calibrated range when available."""
    m = get_model(model)
    try:
        result = m.predict(patient)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")
    background_tasks.add_task(monitor.record_predictions, m.id, [patient.model_dump()], [result["risk_category"]], m.features)
    return PredictionResponse(
        risk_probability=result["risk_probability"],
        risk_percentage=result["risk_percentage"],
        risk_category=result["risk_category"],
        bmi=result["bmi"],
        patient_features=patient.model_dump(),
        model_version=m.metadata.get("model_name", m.name),
        disclaimer="DISCLAIMER: Research-grade prediction system. Not a medical diagnosis.",
        model_id=m.id,
        model_name=m.name,
        risk_range=result["risk_range"],
    )


@app.post("/api/explain", response_model=ExplainabilityResponse)
async def explain(patient: PatientInput, model: str = MODEL_QUERY):
    """Exact per-patient SHAP explanation"""
    m = get_model(model)
    try:
        return m.explain(patient)
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")


@app.post("/api/what-if")
async def execute_what_if(request: WhatIfRequest, model: str = MODEL_QUERY):
    """Compare the predicted risk for a patient before and after a change."""
    m = get_model(model)
    try:
        orig_pct = round(m.predict(request.original)["risk_percentage"], 1)
        sim_pct = round(m.predict(request.modified)["risk_percentage"], 1)
    except Exception as e:
        logger.error(f"What-if error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Counterfactual calculation failed")

    return {
        "original_risk": orig_pct,
        "simulated_risk": sim_pct,
        "risk_delta": round(sim_pct - orig_pct, 1),
        "original_ap_hi": request.original.ap_hi,
        "target_ap_hi": request.modified.ap_hi,
    }


@app.post("/api/counterfactuals")
async def counterfactual_suggestions(patient: PatientInput, model: str = MODEL_QUERY):
    """The smallest sets of changes (one, two and three) that lower this patient's predicted risk.

    Only changeable factors move, and only in a healthier direction; age, sex, height and
    medication never change. The result describes the model, not proven medical effects.
    """
    m = get_model(model)
    try:
        return suggest_counterfactuals(patient.model_dump(), m.features, m.predict_matrix)
    except Exception as e:
        logger.error(f"Counterfactual error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute suggestions")


@app.get("/api/models")
async def list_models():
    """The models this server can use, with their test results."""
    return {"default": DEFAULT_MODEL, "models": [m.info() for m in MODELS.values() if m.is_loaded]}


@app.get("/api/monitoring")
async def monitoring_summary(days: int = Query(30, ge=1, le=90)):
    """Aggregate, privacy-safe usage statistics: daily counts, response times, risk mix and input drift."""
    references = {m.id: m.reference for m in MODELS.values() if m.is_loaded and m.reference}
    return monitor.summary(days, references)


def _describe_row_error(err: ValidationError, values: Dict[str, Any]) -> str:
    """Turn a Pydantic error for one CSV row into a short, readable reason."""
    first = err.errors()[0]
    field = str(first["loc"][0]) if first.get("loc") else ""
    if field == "age" and values.get("age", 0) > 1000:
        return "age looks like it is in days; use years (18-100)"
    message = first["msg"].replace("Value error, ", "")
    return f"{field}: {message}" if field else message


@app.post("/api/batch-predict")
async def batch_predict(background_tasks: BackgroundTasks, file: UploadFile = File(...), model: str = MODEL_QUERY):
    """Score many patients from a CSV file.

    Every row goes through the same validation as /api/predict. Rows that fail it
    are not scored; they are listed in `skipped` with the reason.
    """
    m = get_model(model)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. The limit is {MAX_UPLOAD_BYTES // 1_000_000} MB.")

    try:
        # sep=None detects the delimiter, so comma- and semicolon-separated files both work
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as CSV")
    df.columns = [str(c).strip().lower() for c in df.columns]

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no records")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=400, detail=f"Too many rows ({len(df):,}). The limit is {MAX_BATCH_ROWS:,} per file.")
    missing = [f for f in m.features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    numeric = df[m.features].apply(pd.to_numeric, errors="coerce")
    valid_index: List[Any] = []
    patients: List[PatientInput] = []
    skipped: List[Dict[str, Any]] = []
    for idx, values in zip(df.index, numeric.to_dict(orient="records")):
        line = int(idx) + 2  # line 1 is the header
        blank = [k for k, v in values.items() if pd.isna(v)]
        if blank:
            skipped.append({"line": line, "reason": f"{blank[0]}: missing or not a number"})
            continue
        try:
            patients.append(PatientInput(**values))
            valid_index.append(idx)
        except ValidationError as err:
            skipped.append({"line": line, "reason": _describe_row_error(err, values)})

    results: List[Dict[str, Any]] = []
    if patients:
        try:
            X = np.array([m.row(p) for p in patients], dtype=float)
            probabilities = m.predict_matrix(X)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Batch prediction failed")
        rows = df.loc[valid_index]
        rows = rows.astype(object).where(rows.notna(), None)  # blank extra columns -> null, never NaN
        for row, p in zip(rows.to_dict(orient="records"), probabilities):
            row["risk_score"] = round(float(p), 4)
            row["high_risk"] = bool(p >= 0.5)
            results.append(row)
        background_tasks.add_task(monitor.record_predictions, m.id, [p.model_dump() for p in patients],
                                  [risk_category(float(p)) for p in probabilities], m.features)

    return {
        "total_records": int(len(df)),
        "scored_records": len(results),
        "skipped_records": len(skipped),
        "high_risk_count": sum(r["high_risk"] for r in results),
        "results": results,
        "skipped": skipped[:50],
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "development") == "development"
    )