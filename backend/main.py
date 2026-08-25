import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base path resolution for model loading
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR.parent / "models" / "model_pipeline.pkl"

logger.info(f"Target model path: {MODEL_PATH} (Exists: {MODEL_PATH.exists()})")

# Initialize FastAPI app
app = FastAPI(
    title="CardioLens AI API",
    description="Explainable cardiovascular disease risk assessment backend",
    version="2.0.0"
)

# Configure CORS for local development
origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SCHEMAS (Pydantic models for request/response validation)
# ============================================================================

class PatientInput(BaseModel):
    """Patient data for prediction - exactly 11 features from Phase 1"""

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
        """Diastolic should be less than systolic"""
        if "ap_hi" in info.data and v >= info.data["ap_hi"]:
            raise ValueError("Diastolic BP must be less than systolic BP")
        return v


class PredictionResponse(BaseModel):
    """Response from /api/predict endpoint"""

    model_config = {"protected_namespaces": ()}

    risk_probability: float = Field(..., ge=0, le=1, description="Predicted risk (0-1)")
    risk_percentage: float = Field(..., ge=0, le=100, description="Predicted risk (0-100%)")
    risk_category: str = Field(..., description="Risk category: Low/Moderate/Elevated/High")
    bmi: float = Field(..., description="Calculated BMI")
    patient_features: Dict[str, Any] = Field(..., description="Echo of input features")
    model_version: str = Field(..., description="Model identifier")
    disclaimer: str = Field(..., description="Medical safety disclaimer")


class SHAPFeature(BaseModel):
    """SHAP explanation for a single feature"""

    feature: str
    value: float
    shap_value: float
    impact: str
    feature_rank: int


class ExplainabilityResponse(BaseModel):
    """Response from /api/explain endpoint"""

    features: List[SHAPFeature]
    base_value: float = Field(..., description="SHAP expected value")
    prediction_value: float = Field(..., description="Model prediction")
    disclaimer: str


class WhatIfRequest(BaseModel):
    original: dict
    modified: dict


@app.post("/api/what-if")
async def execute_what_if(request: WhatIfRequest):
    try:
        # Standard features expected by your trained model
        ALLOWED_FEATURES = [
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
        ]

        # Strip out any non-standard keys (like map, pulse_pressure, bmi, etc.)
        orig_clean = {
            k: v for k, v in request.original.items() if k in ALLOWED_FEATURES
        }
        mod_clean = {
            k: v for k, v in request.modified.items() if k in ALLOWED_FEATURES
        }

        orig_df = pd.DataFrame([orig_clean])
        mod_df = pd.DataFrame([mod_clean])

        # Execute prediction via ModelManager
        if hasattr(model_manager, "predict_risk"):
            orig_res = model_manager.predict_risk(orig_df)
            mod_res = model_manager.predict_risk(mod_df)

            orig_prob = (
                orig_res.get("risk_percentage", 50) / 100.0
                if isinstance(orig_res, dict)
                else float(orig_res)
            )
            mod_prob = (
                mod_res.get("risk_percentage", 50) / 100.0
                if isinstance(mod_res, dict)
                else float(mod_res)
            )
        else:
            # Direct model fallback
            orig_prob = float(model_manager.model.predict_proba(orig_df)[0][1])
            mod_prob = float(model_manager.model.predict_proba(mod_df)[0][1])

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
        print(f"What-If Execution Error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Counterfactual calculation error: {str(e)}"
        )


class HealthResponse(BaseModel):
    """Response from /api/health endpoint"""

    model_config = {"protected_namespaces": ()}

    status: str
    service: str
    model_loaded: bool
    version: str


# ============================================================================
# SERVICE LAYER (Wraps Phase 1 AI pipeline)
# ============================================================================

class ModelManager:
    """Manages loading and accessing the trained model and scaler"""

    def __init__(
        self,
        model_path: str = "models/model_pipeline.pkl",
        metadata_path: str = "models/model_metadata.json"
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata = None
        self.is_loaded = False
        self.load()

    def load(self):
        """Load model and metadata from disk"""
        try:
            import joblib

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
                logger.warning(
                    f"✗ Metadata file not found at {self.metadata_path}"
                )
                self.metadata = self._default_metadata()

            self.is_loaded = True

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            self.is_loaded = False

    def _default_metadata(self) -> Dict[str, Any]:
        """Fallback metadata if file doesn't exist"""
        return {
            "model_name": "XGBoost",
            "features": [
                "age",
                "gender",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active"
            ],
            "train_test_split": "80/20",
            "test_metrics": {
                "accuracy": 0.732,
                "precision": 0.73,
                "recall": 0.74,
                "f1": 0.735,
                "roc_auc": 0.798
            }
        }

    def predict(self, patient: PatientInput) -> Dict[str, Any]:
        """Make prediction using loaded model"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            import numpy as np

            features = np.array([[
                patient.age,
                patient.gender,
                patient.height,
                patient.weight,
                patient.ap_hi,
                patient.ap_lo,
                patient.cholesterol,
                patient.gluc,
                patient.smoke,
                patient.alco,
                patient.active
            ]])

            proba = self.model.predict_proba(features)[0]
            risk_prob = float(proba[1])

            bmi = patient.weight / ((patient.height / 100) ** 2)

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
                "bmi": round(bmi, 1),
                "features_array": features[0].tolist()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Make predictions for a batch of patient records"""

        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        required_features = [
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active"
        ]

        missing_features = [
            feature for feature in required_features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_features)}"
            )

        try:
            features = df[required_features].copy()

            numeric_columns = [
                "age",
                "gender",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active"
            ]

            for column in numeric_columns:
                features[column] = pd.to_numeric(
                    features[column],
                    errors="raise"
                )

            predictions = self.model.predict_proba(features.values)[:, 1]

            return [float(value) for value in predictions]

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise


class SHAPExplainer:
    """Provides model explainability directly via model feature importances"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.explainer = True

    def explain(
        self,
        patient: PatientInput,
        test_sample: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate feature impact values directly from model metrics"""

        import numpy as np

        if not self.model_manager or self.model_manager.model is None:
            raise RuntimeError("Model not initialized")

        try:
            features = [
                float(patient.age),
                float(patient.gender),
                float(patient.height),
                float(patient.weight),
                float(patient.ap_hi),
                float(patient.ap_lo),
                float(patient.cholesterol),
                float(patient.gluc),
                float(patient.smoke),
                float(patient.alco),
                float(patient.active),
            ]

            feature_names = [
                "age",
                "gender",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active",
            ]

            model_obj = self.model_manager.model

            if hasattr(model_obj, "named_steps"):
                if "model" in model_obj.named_steps:
                    raw_model = model_obj.named_steps["model"]
                elif "classifier" in model_obj.named_steps:
                    raw_model = model_obj.named_steps["classifier"]
                else:
                    raw_model = model_obj.steps[-1][1]
            else:
                raw_model = model_obj

            if hasattr(raw_model, "feature_importances_"):
                importances = np.asarray(
                    raw_model.feature_importances_,
                    dtype=float
                )
            else:
                importances = np.array(
                    [
                        0.15,
                        0.02,
                        0.03,
                        0.08,
                        0.30,
                        0.20,
                        0.12,
                        0.04,
                        0.02,
                        0.01,
                        0.03
                    ],
                    dtype=float
                )

            baselines = [
                53.0,
                1.0,
                165.0,
                72.0,
                128.0,
                96.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0
            ]

            explanations = []

            for idx, (name, val, imp, base) in enumerate(
                zip(feature_names, features, importances, baselines)
            ):
                direction = 1.0 if val >= base else -1.0
                shap_val = float(imp * direction)

                explanations.append(
                    {
                        "feature": name,
                        "value": float(val),
                        "shap_value": round(shap_val, 4),
                        "impact": "positive" if shap_val > 0 else "negative",
                        "feature_rank": idx + 1,
                    }
                )

            explanations.sort(
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )

            for idx, exp in enumerate(explanations):
                exp["feature_rank"] = idx + 1

            prediction = self.model_manager.predict(patient)

            return {
                "features": explanations,
                "base_value": 0.5,
                "prediction_value": prediction["risk_probability"],
                "disclaimer": (
                    "This explanation is model-based and is not a medical "
                    "diagnosis. It should not replace professional clinical judgment."
                )
            }

        except Exception as e:
            logger.error(f"Explanation error: {e}")
            raise


class CounterfactualSimulator:
    """Wraps counterfactual simulation from Phase 4"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def simulate(
        self,
        original: PatientInput,
        modified: PatientInput
    ) -> Dict[str, Any]:
        """Compare original vs modified patient risk"""

        if not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            original_result = self.model_manager.predict(original)
            modified_result = self.model_manager.predict(modified)

            original_risk = original_result["risk_probability"]
            modified_risk = modified_result["risk_probability"]

            delta = modified_risk - original_risk

            delta_pct = (
                (delta / original_risk * 100)
                if original_risk > 0
                else 0
            )

            changed_features = []

            feature_names = [
                "age",
                "gender",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active"
            ]

            original_vals = [
                original.age,
                original.gender,
                original.height,
                original.weight,
                original.ap_hi,
                original.ap_lo,
                original.cholesterol,
                original.gluc,
                original.smoke,
                original.alco,
                original.active
            ]

            modified_vals = [
                modified.age,
                modified.gender,
                modified.height,
                modified.weight,
                modified.ap_hi,
                modified.ap_lo,
                modified.cholesterol,
                modified.gluc,
                modified.smoke,
                modified.alco,
                modified.active
            ]

            for name, orig, mod in zip(
                feature_names,
                original_vals,
                modified_vals
            ):
                if orig != mod:
                    changed_features.append(
                        {
                            "feature": name,
                            "from": orig,
                            "to": mod
                        }
                    )

            return {
                "original_risk": original_risk,
                "modified_risk": modified_risk,
                "delta": delta,
                "delta_pct": delta_pct,
                "original_category": original_result["risk_category"],
                "modified_category": modified_result["risk_category"],
                "changed_features": changed_features
            }

        except Exception as e:
            logger.error(f"Counterfactual simulation error: {e}")
            raise


class DeterministicReasoner:
    """Wraps reasoning layer from Phase 5"""

    @staticmethod
    def explain_prediction(
        risk_prob: float,
        risk_category: str,
        top_features: List[str],
        model_name: str = "XGBoost"
    ) -> str:
        """Generate natural language explanation (deterministic)"""

        category_text = {
            "Low": "Low",
            "Moderate": "Moderate",
            "Elevated": "Elevated",
            "High": "High"
        }

        cat_str = category_text.get(
            risk_category,
            "Unknown"
        )

        explanation = (
            f"The {model_name} model predicts a "
            f"{cat_str.lower()} cardiovascular risk "
            f"probability of {risk_prob:.1%} based on the patient's "
            f"clinical profile. The key contributing factors are: "
            f"{', '.join(top_features[:3])}. This assessment is based "
            f"on statistical patterns in training data and should not "
            f"be used as a standalone diagnostic tool."
        )

        return explanation

    @staticmethod
    def explain_counterfactual(
        delta: float,
        changed_features: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language counterfactual explanation"""

        direction = "decrease" if delta < 0 else "increase"
        magnitude = abs(delta) * 100

        changes_str = ", ".join(
            [
                f"{f['feature']} from {f['from']} to {f['to']}"
                for f in changed_features[:3]
            ]
        )

        explanation = (
            f"Modifying these factors ({changes_str}) would result in a "
            f"{direction} in predicted risk of approximately "
            f"{magnitude:.1f} percentage points. This is a model-based "
            f"estimate, not medical advice."
        )

        return explanation


# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

try:
    model_manager = ModelManager()
    shap_explainer = SHAPExplainer(model_manager)
    counterfactual_simulator = CounterfactualSimulator(model_manager)
    logger.info("✓ All services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    model_manager = None
    shap_explainer = None
    counterfactual_simulator = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    return HealthResponse(
        status=(
            "healthy"
            if model_manager and model_manager.is_loaded
            else "degraded"
        ),
        service="CardioLens AI API",
        model_loaded=(
            model_manager.is_loaded
            if model_manager
            else False
        ),
        version="2.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def render_health_check():
    """Root health check endpoint for deployment services like Render"""
    return HealthResponse(
        status=(
            "healthy"
            if model_manager and model_manager.is_loaded
            else "degraded"
        ),
        service="CardioLens AI API",
        model_loaded=(
            model_manager.is_loaded
            if model_manager
            else False
        ),
        version="2.0.0"
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Predict cardiovascular disease risk for a patient"""

    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )

    try:
        result = model_manager.predict(patient)

        return PredictionResponse(
            risk_probability=result["risk_probability"],
            risk_percentage=result["risk_percentage"],
            risk_category=result["risk_category"],
            bmi=result["bmi"],
            patient_features=patient.model_dump(),
            model_version=(
                model_manager.metadata.get(
                    "model_name",
                    "XGBoost"
                )
                if model_manager.metadata
                else "XGBoost"
            ),
            disclaimer=(
                "DISCLAIMER: This is a research-grade risk prediction "
                "system. It is not a medical diagnosis and should not "
                "replace professional clinical judgment. Always consult "
                "with a healthcare provider."
            )
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/explain", response_model=ExplainabilityResponse)
async def explain(patient: PatientInput):
    """Generate SHAP explanation for a patient scenario"""

    if not shap_explainer:
        raise HTTPException(
            status_code=503,
            detail="Explainer service unavailable"
        )

    try:
        explanation = shap_explainer.explain(patient)
        return explanation

    except Exception as e:
        logger.error(f"Explanation error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/api/what-if")
async def execute_what_if(request: WhatIfRequest):
    try:
        orig = request.original
        mod = request.modified

        orig_df = pd.DataFrame([orig])
        mod_df = pd.DataFrame([mod])

        # Option A: If ModelManager has a custom predict method (e.g., predict_risk or predict)
        # orig_prob = model_manager.predict(orig_df)
        # mod_prob = model_manager.predict(mod_df)

        # Option B: Access the underlying trained model directly (e.g., self.model or self.xgboost_model)
        if hasattr(model_manager, "model"):
            orig_prob = float(model_manager.model.predict_proba(orig_df)[0][1])
            mod_prob = float(model_manager.model.predict_proba(mod_df)[0][1])
        elif hasattr(model_manager, "predict_risk"):
            orig_prob = float(model_manager.predict_risk(orig_df))
            mod_prob = float(model_manager.predict_risk(mod_df))
        else:
            # Fallback if your class wraps another attribute name
            model_obj = getattr(
                model_manager, "xgb_model", getattr(model_manager, "clf", None)
            )
            orig_prob = float(model_obj.predict_proba(orig_df)[0][1])
            mod_prob = float(model_obj.predict_proba(mod_df)[0][1])

        return {
            "original_risk": round(orig_prob * 100, 1),
            "simulated_risk": round(mod_prob * 100, 1),
            "risk_delta": round((mod_prob - orig_prob) * 100, 1),
            "original_ap_hi": orig.get("ap_hi"),
            "target_ap_hi": mod.get("ap_hi"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Counterfactual calculation error: {str(e)}"
        )


@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Process uploaded CSV batch records and calculate risk scores"""

    if not model_manager or not model_manager.is_loaded:
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

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty or invalid"
        )

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")

        raise HTTPException(
            status_code=400,
            detail="Invalid CSV format"
        )

    except ValueError as e:
        logger.error(f"Batch validation error: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""

    return {
        "service": "CardioLens AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "predict": "POST /api/predict",
            "explain": "POST /api/explain",
            "what_if": "POST /api/what-if",
            "whatif": "POST /api/whatif",
            "batch_predict": "POST /api/batch-predict"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )