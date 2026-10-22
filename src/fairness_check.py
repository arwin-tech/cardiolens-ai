"""Does CardioLens work equally well for different groups of patients?

Compares the deployed model's discrimination (ROC-AUC), sensitivity (recall at the
0.5 cut-off) and calibration across sex and age bands, on the same 14,000 held-out
test patients used for the headline metrics.

Run from the repo root:  python src/fairness_check.py
Writes models/fairness_report.json.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_and_preprocess_data  # noqa: E402

warnings.filterwarnings("ignore")

THRESHOLD = 0.5  # the cut-off the app uses for "high risk"
AGE_BANDS = [(0, 44, "under 45"), (45, 49, "45-49"), (50, 54, "50-54"), (55, 59, "55-59"), (60, 200, "60 and over")]


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted risk and observed disease rate, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def group_metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)
    return {
        "patients": int(len(y)),
        "disease_rate": round(float(y.mean()), 4),
        "mean_predicted_risk": round(float(p.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def compare(groups):
    """Largest gap between groups for each headline metric."""
    keys = ["roc_auc", "recall", "ece"]
    return {k: round(max(g[k] for g in groups.values()) - min(g[k] for g in groups.values()), 4) for k in keys}


def main():
    _, X_test, _, y_test = load_and_preprocess_data("data/cardiovascular_disease.csv")[:4]
    y = np.asarray(y_test)
    model = joblib.load("backend/models/model_pipeline.pkl")
    p = model.predict_proba(X_test)[:, 1]

    sex = pd.Series(np.where(X_test["gender"].to_numpy() == 2, "male", "female"))
    age = X_test["age"].to_numpy()
    band = pd.Series([next(label for lo, hi, label in AGE_BANDS if lo <= a <= hi) for a in age])

    by_sex = {s: group_metrics(y[sex == s], p[sex == s]) for s in ["female", "male"]}
    by_age = {label: group_metrics(y[band == label], p[band == label]) for _, _, label in AGE_BANDS}

    report = {
        "test_patients": int(len(y)),
        "threshold": THRESHOLD,
        "overall": group_metrics(y, p),
        "by_sex": by_sex,
        "by_age_band": by_age,
        "largest_gap": {"sex": compare(by_sex), "age_band": compare(by_age)},
    }

    Path("models").mkdir(exist_ok=True)
    with open("models/fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'group':<14}{'patients':>9}{'disease':>9}{'AUC':>8}{'recall':>8}{'precision':>11}{'ECE':>7}")
    rows = [("overall", report["overall"])] + list(by_sex.items()) + list(by_age.items())
    for name, m in rows:
        print(f"{name:<14}{m['patients']:>9,}{m['disease_rate']:>9.1%}{m['roc_auc']:>8.3f}"
              f"{m['recall']:>8.1%}{m['precision']:>11.1%}{m['ece']:>7.1%}")
    print("\nLargest gaps:", json.dumps(report["largest_gap"]))


if __name__ == "__main__":
    main()
