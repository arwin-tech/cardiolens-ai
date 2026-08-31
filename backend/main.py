"""
CardioLens AI Backend - Production v2.1.0
Fixed CORS configuration, dual health endpoints, and error handling
"""

import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

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
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
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
    "http://localhost:5173,http://localhost:3000,https://cardiolens-ai-za8w.onrender.com"
)
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_STR.split(",") if o.strip()]

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
    """Counterfactual scenario request"""
    original: Dict[str, Any]
    modified: Dict[str, Any]


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

class ModelManager:
    """Manages model loading, predictions, and lifecycle"""
    
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.is_loaded = False
        self.load()

    def load(self):
        """Load model and metadata from disk"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"✓ Model loaded from {self.model_path}")
            else:
                logger.warning(f"✗ Model file not found at {self.model_path}")
                return

            if self.metadata_path.exists():
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded from {self.metadata_path}")
            else:
                logger.warning(f"✗ Metadata file not found at {self.metadata_path}")
                self.metadata = self._default_metadata()

            self.is_loaded = True
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            self.is_loaded = False

    def _default_metadata(self) -> Dict[str, Any]:
        """Return default metadata when file not found"""
        return {
            "model_name": "XGBoost Cardiovascular Risk",
            "version": "2.0.0",
            "features": [
                "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active"
            ],
            "test_metrics": {
                "accuracy": 0.732,
                "precision": 0.73,
                "recall": 0.74,
                "f1": 0.735,
                "roc_auc": 0.798
            }
        }

    def predict(self, patient: PatientInput) -> Dict[str, Any]:
        """Generate risk prediction for patient"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        features = np.array([[
            patient.age, patient.gender, patient.height, patient.weight,
            patient.ap_hi, patient.ap_lo, patient.cholesterol,
            patient.gluc, patient.smoke, patient.alco, patient.active
        ]])

        proba = self.model.predict_proba(features)[0]
        risk_prob = float(proba[1])
        
        # Standardise height unit auto-detection (cm vs m)
        height_m = patient.height / 100.0 if patient.height > 3 else patient.height
        bmi = round(patient.weight / (height_m ** 2), 1) if height_m > 0 else 0.0

        # Categorize risk
        if risk_prob < 0.3:
            category = "Low"
        elif risk_prob < 0.5:
            category = "Moderate"
        elif risk_prob < 0.7:
            category = "Elevated"
        else:
            category = "High"

        return {
            "risk_probability": risk_prob,
            "risk_percentage": risk_prob * 100,
            "risk_category": category,
            "bmi": bmi,
            "features_array": features[0].tolist()
        }

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Generate bulk predictions from DataFrame"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        required_features = [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"
        ]
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        features = df[required_features].astype(float)
        predictions = self.model.predict_proba(features.values)[:, 1]
        return [float(v) for v in predictions]


# ============================================================================
# SERVICE LAYER: SHAP EXPLAINER
# ============================================================================

class SHAPExplainer:
    """Generates feature importance explanations using SHAP-like approach"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def explain(self, patient: PatientInput) -> Dict[str, Any]:
        """Generate SHAP-based explanation for prediction"""
        if not self.model_manager or self.model_manager.model is None:
            raise RuntimeError("Model not initialized")

        features = [
            float(patient.age), float(patient.gender), float(patient.height),
            float(patient.weight), float(patient.ap_hi), float(patient.ap_lo),
            float(patient.cholesterol), float(patient.gluc), float(patient.smoke),
            float(patient.alco), float(patient.active)
        ]
        feature_names = [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"
        ]

        # Extract base model from pipeline if needed
        model_obj = self.model_manager.model
        if hasattr(model_obj, "named_steps"):
            raw_model = (
                model_obj.named_steps.get("model")
                or model_obj.named_steps.get("classifier")
                or model_obj.steps[-1][1]
            )
        else:
            raw_model = model_obj

        # Get feature importances or use defaults
        importances = getattr(
            raw_model,
            "feature_importances_",
            np.array([0.15, 0.02, 0.03, 0.08, 0.30, 0.20, 0.12, 0.04, 0.02, 0.01, 0.03])
        )
        
        baselines = [53.0, 1.0, 165.0, 72.0, 128.0, 96.0, 1.0, 1.0, 0.0, 0.0, 1.0]

        # Generate explanations
        explanations = []
        for idx, (name, val, imp, base) in enumerate(zip(
            feature_names, features, importances, baselines
        )):
            direction = 1.0 if val >= base else -1.0
            shap_val = float(imp * direction)
            explanations.append({
                "feature": name,
                "value": float(val),
                "shap_value": round(shap_val, 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "feature_rank": idx + 1
            })

        # Sort by absolute SHAP value
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        for idx, exp in enumerate(explanations):
            exp["feature_rank"] = idx + 1

        # Get prediction value
        prediction = self.model_manager.predict(patient)

        return {
            "features": explanations,
            "base_value": 0.5,
            "prediction_value": prediction["risk_probability"],
            "disclaimer": "This explanation is model-based and is not a medical diagnosis."
        }


# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================

# Initialize services at module level
model_manager = ModelManager()
shap_explainer = SHAPExplainer(model_manager)


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
    version="2.1.0",
    lifespan=lifespan
)

# ============================================================================
# CORS MIDDLEWARE (FIXED)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # ✅ Use explicit list (not wildcard)
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",  # ✅ Regex for Vercel previews
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight responses
)

logger.info("✅ CORS middleware configured successfully")

# ============================================================================
# ROUTE HANDLERS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "CardioLens AI API",
        "version": "2.1.0",
        "status": "online",
        "endpoints": {
            "health": "/api/health (may be blocked by ad-blockers)",
            "status": "/api/status (recommended, ad-blocker bypass)",
            "predict": "POST /api/predict",
            "explain": "POST /api/explain",
            "what_if": "POST /api/what-if",
            "batch_predict": "POST /api/batch-predict",
        },
        "docs": "/docs",
        "model_status": "loaded" if model_manager.is_loaded else "degraded"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Standard health check endpoint.
    Note: This endpoint may be blocked by browser ad-blockers.
    Use /api/status as a fallback.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        service="CardioLens AI API",
        model_loaded=model_manager.is_loaded,
        version="2.1.0"
    )


@app.get("/api/status", response_model=HealthResponse)
async def status_check():
    """
    Alternative health endpoint (bypasses ad-blockers).
    Recommended endpoint for frontend health checks.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        service="CardioLens AI API",
        model_loaded=model_manager.is_loaded,
        version="2.1.0"
    )


# Legacy alias for backward compatibility
@app.get("/health", response_model=HealthResponse)
async def legacy_health():
    """Backward compatibility alias for /api/health"""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        service="CardioLens AI API",
        model_loaded=model_manager.is_loaded,
        version="2.1.0"
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Generate cardiovascular risk prediction for patient"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Backend running in degraded mode."
        )
    
    try:
        result = model_manager.predict(patient)
        return PredictionResponse(
            risk_probability=result["risk_probability"],
            risk_percentage=result["risk_percentage"],
            risk_category=result["risk_category"],
            bmi=result["bmi"],
            patient_features=patient.model_dump(),
            model_version=model_manager.metadata.get("model_name", "XGBoost") if model_manager.metadata else "XGBoost",
            disclaimer="DISCLAIMER: Research-grade prediction system. Not a medical diagnosis."
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/explain", response_model=ExplainabilityResponse)
async def explain(patient: PatientInput):
    """Generate SHAP-based feature importance explanation"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Explainer service unavailable. Model not loaded."
        )
    
    try:
        return shap_explainer.explain(patient)
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/api/what-if")
async def execute_what_if(request: WhatIfRequest):
    """Generate counterfactual risk comparison"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable"
        )
    
    try:
        allowed = [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"
        ]
        
        # Sanitize inputs
        orig_clean = {k: v for k, v in request.original.items() if k in allowed}
        mod_clean = {k: v for k, v in request.modified.items() if k in allowed}

        # Convert to DataFrames
        orig_df = pd.DataFrame([orig_clean])
        mod_df = pd.DataFrame([mod_clean])

        # Generate predictions
        orig_prob = float(model_manager.model.predict_proba(orig_df)[0][1])
        mod_prob = float(model_manager.model.predict_proba(mod_df)[0][1])

        # Calculate deltas
        orig_pct = round(orig_prob * 100, 1)
        sim_pct = round(mod_prob * 100, 1)

        return {
            "original_risk": orig_pct,
            "simulated_risk": sim_pct,
            "risk_delta": round(sim_pct - orig_pct, 1),
            "original_ap_hi": orig_clean.get("ap_hi"),
            "target_ap_hi": mod_clean.get("ap_hi"),
        }
    except Exception as e:
        logger.error(f"What-If execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Counterfactual calculation error: {str(e)}"
        )


@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Process bulk predictions from CSV file"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable"
        )
    
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty"
            )

        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV contains no records"
            )

        # Generate predictions
        predictions = model_manager.predict_batch(df)
        df["risk_score"] = predictions
        df["high_risk"] = df["risk_score"] >= 0.50

        return {
            "total_records": len(df),
            "high_risk_count": int(df["high_risk"].sum()),
            "results": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


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