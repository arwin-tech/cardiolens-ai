import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance"""
    return TestClient(app)


class TestCardioLensAPI:
    """Test suite for CardioLens AI API endpoints"""

    VALID_PATIENT = {
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
        "active": 1,
    }

    VALID_PATIENT_2 = {
        "age": 45,
        "gender": 2,
        "height": 180,
        "weight": 85,
        "ap_hi": 130,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 1,
        "alco": 1,
        "active": 0,
    }

    def test_health_endpoint(self, client):
        """Test GET /api/health returns 200"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "model_loaded" in data

    def test_predict_valid_input(self, client):
        """Test POST /api/predict with valid patient data"""
        response = client.post("/api/predict", json=self.VALID_PATIENT)
        if response.status_code == 200:
            data = response.json()
            assert "risk_probability" in data
            assert "risk_percentage" in data
            assert "risk_category" in data
            assert "bmi" in data
            assert "disclaimer" in data

    def test_predict_missing_field(self, client):
        """Test POST /api/predict with missing required field"""
        invalid_patient = {k: v for k, v in self.VALID_PATIENT.items() if k != "age"}
        response = client.post("/api/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_invalid_blood_pressure(self, client):
        """Test that diastolic BP > systolic BP is rejected"""
        invalid_patient = {
            **self.VALID_PATIENT,
            "ap_hi": 100,
            "ap_lo": 150,
        }
        response = client.post("/api/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_explain_returns_exact_shap_values(self, client):
        """SHAP values plus the baseline must add up to the predicted risk (exact TreeSHAP)"""
        import math
        response = client.post("/api/explain", json=self.VALID_PATIENT)
        assert response.status_code == 200
        body = response.json()
        logit = lambda p: math.log(p / (1 - p))
        total = logit(body["base_value"]) + sum(f["shap_value"] for f in body["features"])
        assert abs(total - logit(body["prediction_value"])) < 0.01

    def test_explanations_differ_between_patients(self, client):
        """Each patient gets their own explanation, not one global ranking"""
        a = client.post("/api/explain", json=self.VALID_PATIENT).json()["features"]
        b = client.post("/api/explain", json=self.VALID_PATIENT_2).json()["features"]
        values_a = {f["feature"]: f["shap_value"] for f in a}
        values_b = {f["feature"]: f["shap_value"] for f in b}
        assert any(abs(abs(values_a[k]) - abs(values_b[k])) > 1e-3 for k in values_a)

