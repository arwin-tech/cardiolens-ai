import io

import pytest
from fastapi.testclient import TestClient
from backend.main import MAX_BATCH_ROWS, MAX_UPLOAD_BYTES, app


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
        assert response.status_code == 200  # fails if the model didn't load
        data = response.json()
        assert 0 <= data["risk_probability"] <= 1
        assert data["risk_category"] in {"Low", "Moderate", "Elevated", "High"}
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

    # ------------------------------------------------------------------ what-if
    def test_what_if_valid_change(self, client):
        """Lowering blood pressure returns both risks and their difference"""
        lowered = {**self.VALID_PATIENT, "ap_hi": 120, "ap_lo": 80}
        response = client.post("/api/what-if", json={"original": self.VALID_PATIENT, "modified": lowered})
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["simulated_risk"] <= 100
        assert body["risk_delta"] == pytest.approx(body["simulated_risk"] - body["original_risk"], abs=0.11)
        assert body["target_ap_hi"] == 120

    def test_what_if_matches_predict(self, client):
        """The what-if baseline equals the normal prediction for the same patient"""
        predicted = client.post("/api/predict", json=self.VALID_PATIENT).json()["risk_percentage"]
        body = client.post("/api/what-if", json={"original": self.VALID_PATIENT, "modified": self.VALID_PATIENT}).json()
        assert body["original_risk"] == round(predicted, 1)

    @pytest.mark.parametrize(
        "modified",
        [
            {},                                                   # empty patient (used to score 98.9%)
            {"ap_hi": 100, "ap_lo": 150},                         # diastolic above systolic
            {"age": 500},                                         # impossible age
            {"weight": -20},                                      # impossible weight
            {"age": "old"},                                       # not a number (used to crash with 500)
        ],
    )
    def test_what_if_rejects_invalid_patients(self, client, modified):
        patient = {} if modified == {} else {**self.VALID_PATIENT, **modified}
        response = client.post("/api/what-if", json={"original": self.VALID_PATIENT, "modified": patient})
        assert response.status_code == 422

    # -------------------------------------------------------------------- batch
    HEADER = "age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active\n"

    @staticmethod
    def _upload(client, text, name="patients.csv"):
        return client.post("/api/batch-predict", files={"file": (name, io.BytesIO(text.encode()), "text/csv")})

    def test_batch_scores_valid_rows_and_reports_bad_ones(self, client):
        csv = (
            self.HEADER
            + "52,1,165,82,145,90,2,1,0,0,1\n"      # line 2: valid
            + "52,1,165,,145,90,2,1,0,0,1\n"        # line 3: blank weight (used to crash with 500)
            + "18393,2,168,62,110,80,1,1,0,0,1\n"   # line 4: age in days, as in the raw Kaggle file
            + "52,1,165,82,100,150,2,1,0,0,1\n"     # line 5: diastolic above systolic
            + "52,1,165,82,16020,90,2,1,0,0,1\n"    # line 6: impossible systolic
        )
        response = self._upload(client, csv)
        assert response.status_code == 200
        body = response.json()
        assert body["total_records"] == 5
        assert body["scored_records"] == 1
        assert body["skipped_records"] == 4
        reasons = {s["line"]: s["reason"] for s in body["skipped"]}
        assert "missing" in reasons[3]
        assert "days" in reasons[4]
        assert "Diastolic" in reasons[5]
        assert reasons[6].startswith("ap_hi")

    def test_batch_matches_single_prediction(self, client):
        """A patient scored in a batch gets the same risk as through /api/predict"""
        single = client.post("/api/predict", json=self.VALID_PATIENT).json()["risk_probability"]
        row = ",".join(str(self.VALID_PATIENT[k]) for k in self.HEADER.strip().split(","))
        body = self._upload(client, self.HEADER + row + "\n").json()
        assert body["results"][0]["risk_score"] == pytest.approx(single, abs=1e-4)

    def test_batch_accepts_semicolon_files(self, client):
        csv = self.HEADER.replace(",", ";") + "52;1;165;82;145;90;2;1;0;0;1\n"
        assert self._upload(client, csv).json()["scored_records"] == 1

    def test_batch_rejects_missing_columns(self, client):
        response = self._upload(client, "age,gender\n52,1\n")
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]

    def test_batch_rejects_too_many_rows(self, client):
        row = "52,1,165,82,145,90,2,1,0,0,1\n"
        response = self._upload(client, self.HEADER + row * (MAX_BATCH_ROWS + 1))
        assert response.status_code == 400

    def test_batch_rejects_oversized_files(self, client):
        response = self._upload(client, self.HEADER + "x" * (MAX_UPLOAD_BYTES + 10))
        assert response.status_code == 413

    def test_batch_rejects_non_csv(self, client):
        assert self._upload(client, "hello", name="notes.txt").status_code == 400

    # --------------------------------------------------------------------- CORS
    @pytest.mark.parametrize(
        "origin,allowed",
        [
            ("https://cardiolens-ai.vercel.app", True),
            ("https://cardiolens-ai-git-main-arwin-tech.vercel.app", True),   # Vercel preview
            ("http://localhost:5173", True),
            ("https://some-other-app.vercel.app", False),                     # used to be allowed
            ("https://evil.example.com", False),
        ],
    )
    def test_cors_allows_only_this_frontend(self, client, origin, allowed):
        response = client.options(
            "/api/predict",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
        )
        assert (response.headers.get("access-control-allow-origin") == origin) is allowed
        assert "access-control-allow-credentials" not in response.headers



# ============================================================================
# Uncertainty, counterfactuals, model registry and monitoring
# ============================================================================
from backend.main import MODELS, monitor  # noqa: E402

MODERATE = dict(age=53, gender=2, height=176, weight=84, ap_hi=135, ap_lo=85, cholesterol=1, gluc=1,
                smoke=1, alco=0, active=1)
HIGH = dict(age=61, gender=2, height=172, weight=98, ap_hi=165, ap_lo=100, cholesterol=3, gluc=2,
            smoke=1, alco=0, active=0)


@pytest.fixture
def api():
    return TestClient(app)


def test_prediction_includes_calibrated_range(api):
    body = api.post("/api/predict", json=MODERATE).json()
    lo, hi = body["risk_range"]
    assert 0 <= lo <= hi <= 100
    assert hi - lo < 15
    assert body["model_id"] == "kaggle"


def test_counterfactuals_lower_risk_and_never_touch_fixed_traits(api):
    body = api.post("/api/counterfactuals", json=HIGH).json()
    assert body["plans"], "a high-risk patient should get suggestions"
    previous = body["baseline_risk"]
    for k, plan in enumerate(body["plans"], start=1):
        assert len(plan["changes"]) <= 3
        assert plan["risk"] < previous  # each bigger plan must do better than the smaller one
        previous = plan["risk"]
        changed = {c["feature"] for c in plan["changes"]}
        assert not changed & {"age", "gender", "height", "bp_meds", "lipid_meds"}


def test_counterfactuals_empty_when_nothing_can_improve(api):
    healthy = dict(age=34, gender=1, height=168, weight=60, ap_hi=115, ap_lo=75, cholesterol=1, gluc=1,
                   smoke=0, alco=0, active=1)
    body = api.post("/api/counterfactuals", json=healthy).json()
    assert body["plans"] == []
    assert "not proven medical effects" in body["note"]


def test_counterfactuals_validate_input(api):
    assert api.post("/api/counterfactuals", json={}).status_code == 422


def test_unknown_model_is_rejected(api):
    assert api.post("/api/predict?model=nope", json=MODERATE).status_code == 404


def test_models_endpoint_lists_default(api):
    body = api.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    assert body["default"] == "kaggle" and "kaggle" in ids
    kaggle = next(m for m in body["models"] if m["id"] == "kaggle")
    assert kaggle["uncertainty"]["method"] == "Venn-Abers"


def test_monitoring_counts_and_hides_small_numbers(api):
    before = api.get("/api/monitoring").json()["predictions_total"]
    api.post("/api/predict", json=MODERATE)
    after = api.get("/api/monitoring").json()
    assert after["predictions_total"] == before + 1
    for counts in after["risk_categories"].values():
        assert all(v is None or v >= 5 for v in counts.values())


def test_monitoring_never_stores_raw_inputs(api):
    odd = dict(MODERATE, weight=83.7, height=177.3, ap_hi=137, ap_lo=86)
    api.post("/api/predict", json=odd)
    stored = {str(r["bucket"]) for r in monitor.raw_rows()} | {str(r["metric"]) for r in monitor.raw_rows()}
    for exact in ("83.7", "177.3", "137", "86"):
        assert exact not in stored


def test_monitoring_reports_drift_after_enough_predictions(api):
    for _ in range(30):
        api.post("/api/predict", json=MODERATE)
    drift = api.get("/api/monitoring").json()["drift"]["kaggle"]
    assert drift["status"] == "ok"
    assert drift["features"]["ap_hi"]["psi"] > 0.25  # 30 identical patients look nothing like the training mix


@pytest.mark.skipif("nhanes" not in MODELS, reason="run src/train_nhanes_model.py to create the US population model")
def test_nhanes_model_uses_medication(api):
    base = dict(MODERATE, bp_meds=0, lipid_meds=0)
    treated = dict(base, bp_meds=1, lipid_meds=1)
    r0 = api.post("/api/predict?model=nhanes", json=base).json()
    r1 = api.post("/api/predict?model=nhanes", json=treated).json()
    assert r0["model_id"] == "nhanes" and r0["risk_range"] is not None
    assert r0["risk_probability"] != r1["risk_probability"]
    assert api.post("/api/counterfactuals?model=nhanes", json=base).status_code == 200
