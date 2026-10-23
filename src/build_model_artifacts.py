"""Build the clinic (Kaggle) model's uncertainty ranges and monitoring reference.

The 14,000 held-out test patients are split in half (stratified, seed 42): 7,000
calibrate the Venn-Abers ranges and the other 7,000 check them, so the reported
quality of the ranges is measured on patients they never saw.

Writes:
  backend/models/uncertainty_kaggle.json   precomputed Venn-Abers ranges
  backend/models/reference_kaggle.json     training data's input mix (for drift monitoring)
  models/uncertainty_report.json           how good the ranges are
  frontend/src/data/model_reports.json     the numbers shown on the About page

Run from the repo root:  python src/build_model_artifacts.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from monitoring import reference_distribution  # noqa: E402
from src.data_loader import load_and_preprocess_data  # noqa: E402
from venn_abers import VennAbersGrid, build_grid  # noqa: E402

warnings.filterwarnings("ignore")
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def update_frontend_report(key, value):
    path = ROOT / "frontend" / "src" / "data" / "model_reports.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    X_train, X_test, _, y_test = load_and_preprocess_data(str(ROOT / "data" / "cardiovascular_disease.csv"))[:4]
    y_test = np.asarray(y_test)
    model = joblib.load(ROOT / "backend" / "models" / "model_pipeline.pkl")
    scores = model.predict_proba(X_test[FEATURES].to_numpy(dtype=float))[:, 1]

    idx_cal, idx_eval = train_test_split(np.arange(len(y_test)), test_size=0.5, stratify=y_test, random_state=42)
    print(f"Calibrating Venn-Abers on {len(idx_cal):,} patients (takes about a minute)...")
    grid = build_grid(scores[idx_cal], y_test[idx_cal])
    va = VennAbersGrid(grid)

    s_eval, y_eval = scores[idx_eval], y_test[idx_eval]
    ranges = np.array([va.interval(s) for s in s_eval])
    widths = (ranges[:, 1] - ranges[:, 0]) * 100
    merged = np.array([va.merged(s) for s in s_eval])
    report = {
        "method": "Venn-Abers (inductive), precomputed on a 1,001-point score grid",
        "calibration_patients": int(len(idx_cal)),
        "evaluation_patients": int(len(idx_eval)),
        "range_width_points": {
            "median": round(float(np.median(widths)), 2),
            "p90": round(float(np.percentile(widths, 90)), 2),
            "max": round(float(widths.max()), 2),
        },
        "calibration_error": {
            "model_as_is": round(ece(y_eval, s_eval), 4),
            "venn_abers_estimate": round(ece(y_eval, merged), 4),
        },
        "brier": {
            "model_as_is": round(float(brier_score_loss(y_eval, s_eval)), 4),
            "venn_abers_estimate": round(float(brier_score_loss(y_eval, merged)), 4),
        },
    }

    out = ROOT / "backend" / "models"
    (out / "uncertainty_kaggle.json").write_text(json.dumps(grid))
    reference = reference_distribution(X_train[FEATURES].to_dict(orient="records"), FEATURES)
    (out / "reference_kaggle.json").write_text(json.dumps(reference, indent=1))
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "uncertainty_report.json").write_text(json.dumps(report, indent=2) + "\n")
    update_frontend_report("kaggle", {"uncertainty": report})

    w = report["range_width_points"]
    print(f"Range width (percentage points): median {w['median']}, 90th percentile {w['p90']}, widest {w['max']}")
    c = report["calibration_error"]
    print(f"Calibration error on the other {len(idx_eval):,}: model {c['model_as_is']:.1%} -> Venn-Abers {c['venn_abers_estimate']:.1%}")
    print(f"Reference distribution saved for {len(FEATURES)} inputs from {len(X_train):,} training patients")


if __name__ == "__main__":
    main()
