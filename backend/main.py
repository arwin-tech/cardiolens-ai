from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CardioLens AI API",
    description="Explainable cardiovascular disease risk assessment backend",
    version="2.0.0"
)

# Configure CORS for local development
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

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
    
    @field_validator('ap_lo')
    @classmethod
    def validate_blood_pressure(cls, v: int, info) -> int:
        """Diastolic should be less than systolic"""
        if 'ap_hi' in info.data and v >= info.data['ap_hi']:
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
    impact: str  # "positive" or "negative"
    feature_rank: int


class ExplainabilityResponse(BaseModel):
    """Response from /api/explain endpoint"""
    
    features: List[SHAPFeature]
    base_value: float = Field(..., description="SHAP expected value")
    prediction_value: float = Field(..., description="Model prediction")
    disclaimer: str


class WhatIfResponse(BaseModel):
    """Response from /api/what-if endpoint"""
    
    original_risk: float
    simulated_risk: float
    risk_delta: float
    risk_delta_percentage: float
    original_category: str
    simulated_category: str
    explanation: str
    disclaimer: str


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
    
    def __init__(self, model_path: str = "models/model_pipeline.pkl", 
                 metadata_path: str = "models/model_metadata.json"):
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
                with open(self.metadata_path, 'r') as f:
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
        """Fallback metadata if file doesn't exist"""
        return {
            "model_name": "XGBoost",
            "features": ["age", "gender", "height", "weight", "ap_hi", "ap_lo", 
                        "cholesterol", "gluc", "smoke", "alco", "active"],
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


class SHAPExplainer:
    """Wraps SHAP explainability safely handling raw binary trees"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.explainer = None
        self.load_explainer()

    def load_explainer(self):
        """Initialize SHAP explainer without triggering binary decoding errors"""
        try:
            import shap

            if not self.model_manager or self.model_manager.model is None:
                logger.warning("Cannot initialize SHAP - model not loaded")
                return

            pipeline = self.model_manager.model

            # Extract underlying estimator step if inside a Pipeline
            if hasattr(pipeline, "named_steps"):
                model_obj = pipeline.named_steps.get("model", pipeline.steps[-1][1])
            else:
                model_obj = pipeline

            # Primary attempt: Use raw booster for XGBoost to avoid C-extension binary string decoding
            if hasattr(model_obj, "get_booster"):
                booster = model_obj.get_booster()
                self.explainer = shap.TreeExplainer(booster)
            else:
                self.explainer = shap.TreeExplainer(model_obj)

            logger.info("✓ SHAP TreeExplainer initialized successfully")

        except Exception as e:
            try:
                import shap

                pipeline = self.model_manager.model
                model_obj = (
                    pipeline.named_steps.get("model", pipeline.steps[-1][1])
                    if hasattr(pipeline, "named_steps")
                    else pipeline
                )

                # Robust fallback using model's prediction function directly
                self.explainer = shap.Explainer(
                    model_obj.predict,
                    feature_names=[
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
                    ],
                )
                logger.info("✓ SHAP Explainer initialized via prediction fallback")
            except Exception as fallback_error:
                logger.error(f"SHAP initialization error: {fallback_error}")
                self.explainer = None

    def explain(
        self, patient: PatientInput, test_sample: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate SHAP explanation for patient prediction"""
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized")

        try:
            import numpy as np

            features = np.array(
                [
                    [
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
                        patient.active,
                    ]
                ]
            )

            # Generate SHAP values (handles Explanation objects and raw arrays)
            shap_output = self.explainer(features)

            if hasattr(shap_output, "values"):
                sv = shap_output.values[0]
                if sv.ndim > 1:
                    sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]
            elif isinstance(shap_output, list):
                sv = shap_output[1][0]
            else:
                sv = shap_output[0]

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
            feature_values = features[0].tolist()

            explanations = []
            for idx, (name, shap_val, feat_val) in enumerate(
                zip(feature_names, sv, feature_values)
            ):
                explanations.append(
                    {
                        "feature": name,
                        "value": float(feat_val),
                        "shap_value": float(shap_val),
                        "impact": "positive" if shap_val > 0 else "negative",
                        "feature_rank": idx,
                    }
                )

            explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            for idx, exp in enumerate(explanations):
                exp["feature_rank"] = idx + 1

            # Determine expected base value across different SHAP versions
            if hasattr(self.explainer, "expected_value"):
                expected_val = self.explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)):
                    base_val = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
                else:
                    base_val = float(expected_val)
            elif hasattr(shap_output, "base_values"):
                bv = shap_output.base_values[0]
                base_val = float(bv[1] if bv.ndim > 0 and bv.shape[0] > 1 else bv)
            else:
                base_val = 0.5

            return {"features": explanations, "base_value": base_val}

        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            raise


class CounterfactualSimulator:
    """Wraps counterfactual simulation from Phase 4"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def simulate(self, original: PatientInput, modified: PatientInput) -> Dict[str, Any]:
        """Compare original vs modified patient risk"""
        if not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            original_result = self.model_manager.predict(original)
            modified_result = self.model_manager.predict(modified)
            
            original_risk = original_result["risk_probability"]
            modified_risk = modified_result["risk_probability"]
            delta = modified_risk - original_risk
            delta_pct = (delta / original_risk * 100) if original_risk > 0 else 0
            
            changed_features = []
            feature_names = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                             "cholesterol", "gluc", "smoke", "alco", "active"]
            
            original_vals = [
                original.age, original.gender, original.height, original.weight,
                original.ap_hi, original.ap_lo, original.cholesterol, 
                original.gluc, original.smoke, original.alco, original.active
            ]
            modified_vals = [
                modified.age, modified.gender, modified.height, modified.weight,
                modified.ap_hi, modified.ap_lo, modified.cholesterol,
                modified.gluc, modified.smoke, modified.alco, modified.active
            ]
            
            for name, orig, mod in zip(feature_names, original_vals, modified_vals):
                if orig != mod:
                    changed_features.append({
                        "feature": name,
                        "from": orig,
                        "to": mod
                    })
            
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
    def explain_prediction(risk_prob: float, risk_category: str, 
                           top_features: List[str], model_name: str = "XGBoost") -> str:
        """Generate natural language explanation (deterministic)"""
        
        category_text = {
            "Low": "Low",
            "Moderate": "Moderate",
            "Elevated": "Elevated",
            "High": "High"
        }
        
        cat_str = category_text.get(risk_category, "Unknown")
        
        explanation = (
            f"The {model_name} model predicts a {cat_str.lower()} cardiovascular risk "
            f"probability of {risk_prob:.1%} based on the patient's clinical profile. "
            f"The key contributing factors are: {', '.join(top_features[:3])}. "
            f"This assessment is based on statistical patterns in training data and should "
            f"not be used as a standalone diagnostic tool."
        )
        
        return explanation
    
    @staticmethod
    def explain_counterfactual(delta: float, changed_features: List[Dict[str, Any]]) -> str:
        """Generate natural language counterfactual explanation"""
        
        direction = "decrease" if delta < 0 else "increase"
        magnitude = abs(delta) * 100
        
        changes_str = ", ".join([
            f"{f['feature']} from {f['from']} to {f['to']}"
            for f in changed_features[:3]
        ])
        
        explanation = (
            f"Modifying these factors ({changes_str}) would result in a "
            f"{direction} in predicted risk of approximately {magnitude:.1f} percentage points. "
            f"This is a model-based estimate, not medical advice."
        )
        
        return explanation


# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

try:
    model_manager = ModelManager()
    shap_explainer = SHAPExplainer(model_manager)
    logger.info("✓ All services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    model_manager = None
    shap_explainer = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager and model_manager.is_loaded else "degraded",
        service="CardioLens AI API",
        model_loaded=model_manager.is_loaded if model_manager else False,
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
            model_version=model_manager.metadata.get("model_name", "XGBoost"),
            disclaimer=(
                "DISCLAIMER: This is a research-grade risk prediction system. "
                "It is not a medical diagnosis and should not replace professional "
                "clinical judgment. Always consult with a healthcare provider."
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
    """Get SHAP-based explanation of prediction"""
    
    if not shap_explainer or shap_explainer.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available"
        )
    
    try:
        shap_result = shap_explainer.explain(patient)
        prediction = model_manager.predict(patient)
        
        features = [
            SHAPFeature(
                feature=f["feature"],
                value=f["value"],
                shap_value=f["shap_value"],
                impact=f["impact"],
                feature_rank=f["feature_rank"]
            )
            for f in shap_result["features"]
        ]
        
        return ExplainabilityResponse(
            features=features,
            base_value=shap_result.get("base_value", 0.5),
            prediction_value=prediction["risk_probability"],
            disclaimer=(
                "SHAP values show feature contributions to the model prediction. "
                "This is exploratory analysis, not medical advice."
            )
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/what-if", response_model=WhatIfResponse)
async def what_if(original: PatientInput, modified: PatientInput):
    """Run counterfactual what-if scenario"""
    
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    try:
        sim_result = CounterfactualSimulator(model_manager).simulate(original, modified)
        
        explanation = DeterministicReasoner.explain_counterfactual(
            sim_result["delta"],
            sim_result["changed_features"]
        )
        
        return WhatIfResponse(
            original_risk=sim_result["original_risk"],
            simulated_risk=sim_result["modified_risk"],
            risk_delta=sim_result["delta"],
            risk_delta_percentage=sim_result["delta_pct"],
            original_category=sim_result["original_category"],
            simulated_category=sim_result["modified_category"],
            explanation=explanation,
            disclaimer=(
                "This what-if analysis explores model behavior under hypothetical scenarios. "
                "It does not imply causality or medical recommendations."
            )
        )
    
    except Exception as e:
        logger.error(f"What-if error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
            "what_if": "POST /api/what-if"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)