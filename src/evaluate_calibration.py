"""Check how trustworthy CardioLens's risk percentages are (probability calibration).

Run from the repo root:  python src/evaluate_calibration.py
Writes models/calibration_curve.png and models/calibration_report.json.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_and_preprocess_data  # noqa: E402

warnings.filterwarnings("ignore")


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted risk and observed disease rate, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": round(brier_score_loss(y, p), 4), "ece": round(expected_calibration_error(y, p), 4),
            "log_loss": round(log_loss(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4)}


def main():
    _, X_test, _, y_test = load_and_preprocess_data("data/cardiovascular_disease.csv")[:4]
    y_test = np.asarray(y_test)
    model = joblib.load("backend/models/model_pipeline.pkl")
    p = model.predict_proba(X_test)[:, 1]
    report = {"test_patients": int(len(y_test)), "deployed_model": metrics(y_test, p)}

    # Would recalibration help? Fit on one half of the held-out set, judge on the other half.
    pc, pe, yc, ye = train_test_split(p, y_test, test_size=0.5, random_state=42, stratify=y_test)
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression().fit(logit(pc).reshape(-1, 1), yc)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)
    report["recalibration_check_on_held_out_half"] = {
        "raw_model": metrics(ye, pe),
        "platt_scaling": metrics(ye, platt.predict_proba(logit(pe).reshape(-1, 1))[:, 1]),
        "isotonic_regression": metrics(ye, iso.predict(pe)),
    }

    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, "o-", color="#8B2D2D", label="CardioLens (XGBoost)")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed disease rate")
    plt.title(f"Calibration on {len(y_test):,} test patients\n"
              f"ECE {report['deployed_model']['ece']:.1%} | Brier {report['deployed_model']['brier']:.3f}")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    Path("models").mkdir(exist_ok=True)
    plt.savefig("models/calibration_curve.png", dpi=150, bbox_inches="tight")
    with open("models/calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
