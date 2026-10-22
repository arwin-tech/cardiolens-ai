# CardioLens AI

[![CI](https://github.com/arwin-tech/cardiolens-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/arwin-tech/cardiolens-ai/actions/workflows/ci.yml)

**Explainable cardiovascular disease risk prediction**

CardioLens AI predicts a person's risk of cardiovascular disease from 11 basic health details, like age, blood pressure, cholesterol and smoking, and shows which factors raised or lowered that risk.

**Live demo:** [cardiolens-ai.vercel.app](https://cardiolens-ai.vercel.app) · **API docs:** [cardiolens-ai-za8w.onrender.com/docs](https://cardiolens-ai-za8w.onrender.com/docs)

> [!IMPORTANT]
> This is a research and learning project, not a medical tool. Do not use it to diagnose or treat anyone.

The backend runs on Render's free tier, which sleeps when idle, so the first request can take up to a minute. The app shows a "waking up" message while it starts, and one-click example patients let you try it straight away.

## Features

- **Risk prediction:** enter patient details to get a risk percentage, a risk level (Low, Moderate, Elevated or High) and BMI
- **Factor breakdown:** exact SHAP values for each patient (TreeSHAP, computed by XGBoost), showing which factors pushed their risk up or down
- **What-if simulator:** change the blood pressure and see how the risk changes
- **Report download:** save a plain-text summary of the assessment
- **Batch scoring:** upload a CSV of up to 5,000 patients and get a risk score for each. Every row is checked with the same rules as the form, and rows that fail are skipped with the reason

## How It Works

The model is an XGBoost classifier trained on 70,000 patient records from the Kaggle Cardiovascular Disease dataset. It beat Logistic Regression and Random Forest, and on 14,000 test patients it reaches **73.2% accuracy** and a **ROC-AUC of 0.798**.

SHAP analysis shows that systolic blood pressure is by far the biggest factor in the model's predictions, followed by age and cholesterol.

<p align="center">
  <img src="explainability/shap_bar.png" alt="SHAP feature importance" width="60%">
</p>

## Calibration

The risk percentages are well calibrated. On the 14,000 test patients, the expected calibration error is 1.5% and the Brier score is 0.182, so a predicted 30% risk means roughly 30% of similar patients had the disease. The one weak spot is the very top: patients predicted at around 92% had an 85% disease rate, so very high scores are best read as "high" rather than exact.

Platt scaling was tested on a held-out half of the test set and only reduced the calibration error from 1.8% to 1.2%, so it isn't deployed. To reproduce: `python src/evaluate_calibration.py`.

<p align="center">
  <img src="models/calibration_curve.png" alt="Calibration curve" width="45%">
</p>

## Fairness Check

Does the model work equally well for everyone? On the same 14,000 test patients:

| Group | Patients | ROC-AUC | Recall at the 50% cut-off | Calibration error |
|---|---|---|---|---|
| Women | 9,080 | 0.793 | 68.6% | 1.6% |
| Men | 4,920 | 0.806 | 71.1% | 1.4% |
| Under 45 | 2,033 | 0.837 | 57.4% | 1.7% |
| 45–49 | 1,793 | 0.786 | 61.1% | 3.8% |
| 50–54 | 3,732 | 0.786 | 60.9% | 1.9% |
| 55–59 | 3,254 | 0.750 | 67.5% | 1.5% |
| 60 and over | 3,188 | 0.715 | 85.2% | 2.5% |

Women and men get almost identical performance. Age is where the model differs: the risk percentages stay well calibrated in every age band, but at the app's 50% cut-off it catches 57% of disease cases in patients under 45 versus 85% in patients 60 and over, because younger patients get lower scores overall. Separating sick from healthy patients also gets harder with age (ROC-AUC 0.84 down to 0.72). A real screening tool would need to choose its cut-off for the population it serves; this demo keeps 50%. To reproduce: `python src/fairness_check.py`.

## SQL Data Layer

The `sql/` folder rebuilds the data preparation in SQL with DuckDB: a data-quality profile (it flags 1,321 physiologically impossible blood-pressure readings), cleaning and feature engineering, and disease rates by age band and blood-pressure category. The deployed model is unchanged. To run it: `pip install duckdb`, then `python train_from_sql.py`.

## Tech Stack

- **ML:** Python, XGBoost, scikit-learn, SHAP, SQL (DuckDB)
- **Backend:** FastAPI, hosted on Render
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Recharts, hosted on Vercel
- **Engineering:** pytest (25 API tests), GitHub Actions CI on every push, Docker image for the API

## Run Locally

You need Python 3.10+ and Node.js 18+.

**1. Start the backend**

```bash
git clone https://github.com/arwin-tech/cardiolens-ai.git
cd cardiolens-ai
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload
```

The API runs at http://127.0.0.1:8000, with docs at http://127.0.0.1:8000/docs.

**2. Start the frontend** (in a new terminal, from the project folder)

```bash
cd frontend
cp .env.example .env      # Windows: copy .env.example .env
npm install
npm run dev
```

Then open http://localhost:5173.

**Or run the API with Docker**

```bash
docker build -t cardiolens-api backend
docker run -p 8000:8000 cardiolens-api
```

**Run the tests**

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m pytest backend -v
```

## Project Structure

```text
backend/          FastAPI server and the trained model
frontend/         React web app
src/              Training and explainability scripts
data/             Dataset
explainability/   SHAP charts
sql/              Data profiling and feature engineering in SQL (DuckDB)
```
