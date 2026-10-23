"""Train CardioLens's US population model on NHANES and test it on later, unseen data.

Why: the clinic model (trained on the Kaggle data) dropped to ROC-AUC 0.63 on NHANES,
partly because most people already diagnosed take blood-pressure or cholesterol
medication, which lowers the readings the model relies on. This model learns from US
survey data directly and knows about medication.

Design:
  train     NHANES 2009-2010, 2011-2012, 2013-2014, 2015-2016 (80% fit, 20% calibration)
  test      NHANES 2017-March 2020, a later survey with a new blood-pressure device,
            never touched during training or tuning (temporal validation)
  inputs    the app's 11 inputs + taking blood-pressure medication + taking cholesterol medication
  outcome   a doctor ever said: coronary heart disease, angina, heart attack, heart failure or stroke
  ablation  the same model without the two medication inputs, to measure what they add

Run from the repo root (downloads about 75 MB from CDC once, cached in data/external/):
    python src/train_nhanes_model.py

Writes backend/models/nhanes_model.json (XGBoost's version-safe format) with its
metadata, Venn-Abers ranges and monitoring reference; models/nhanes_report.json;
the About page data in frontend/src/data/model_reports.json; and the README section.
"""
import argparse
import json
import sys
import time
import urllib.request
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from monitoring import reference_distribution  # noqa: E402
from venn_abers import VennAbersGrid, build_grid  # noqa: E402

warnings.filterwarnings("ignore")

BASE_FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
FEATURES = BASE_FEATURES + ["bp_meds", "lipid_meds"]
SEED = 42
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"

# (label, CDC folder year, file-name pattern)
TRAIN_CYCLES = [("2009-2010", "2009", "{c}_F"), ("2011-2012", "2011", "{c}_G"),
                ("2013-2014", "2013", "{c}_H"), ("2015-2016", "2015", "{c}_I")]
TEST_CYCLE = ("2017-2020", "2017", "P_{c}")

COMPONENTS = {
    "DEMO": ["RIAGENDR", "RIDAGEYR", "RIDEXPRG"],
    "BMX": ["BMXHT", "BMXWT"],
    "BPX": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "TCHOL": ["LBXTC"],
    "GHB": ["LBXGH"],
    "SMQ": ["SMQ020", "SMQ040"],
    "ALQ": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U"],
    "PAQ": ["PAQ605", "PAQ620", "PAQ650", "PAQ665"],
    "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "DIQ": ["DIQ010"],
    "BPQ": ["BPQ050A", "BPQ100D"],
}
# 2017-2020 changes: oscillometric blood pressure (3 readings) and a redesigned alcohol questionnaire
P_OVERRIDES = {
    "BPX": ("BPXO", ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]),
    "ALQ": ("ALQ", ["ALQ111", "ALQ121"]),
}


# ----------------------------------------------------------------------------- data
def _fetch(year: str, name: str, cache: Path, offline: bool) -> Path:
    path = cache / f"{name}.xpt"
    if path.exists():
        return path
    if offline:
        raise SystemExit(f"{path} is missing and --offline was given.")
    cache.mkdir(parents=True, exist_ok=True)
    url = URL.format(year=year, name=name)
    for attempt in range(3):
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CardioLens training)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                path.write_bytes(r.read())
            return path
        except Exception as e:  # network hiccup: retry, keep what's already cached
            if attempt == 2:
                raise SystemExit(f"Could not download {url} ({e}). Re-run the script to resume.")
            time.sleep(3)
    return path


def load_cycle(label: str, year: str, pattern: str, cache_root: Path, offline: bool) -> pd.DataFrame:
    is_p = pattern.startswith("P_")
    merged = None
    for comp, cols in COMPONENTS.items():
        file_comp, file_cols = (P_OVERRIDES[comp] if is_p and comp in P_OVERRIDES else (comp, cols))
        name = pattern.format(c=file_comp)
        df = pd.read_sas(_fetch(year, name, cache_root / label, offline), format="xport")
        df.columns = [c.upper() for c in df.columns]
        missing = [c for c in ["SEQN"] + file_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"{name}.xpt ({label}) has no column(s) {missing}. CDC may have renamed them; "
                             f"please report this message.")
        df = df[["SEQN"] + file_cols]
        if is_p and comp == "BPX":  # rename the oscillometric readings to the classic names
            df = df.rename(columns={f"BPXOSY{i}": f"BPXSY{i}" for i in (1, 2, 3)})
            df = df.rename(columns={f"BPXODI{i}": f"BPXDI{i}" for i in (1, 2, 3)})
        merged = df if merged is None else merged.merge(df, on="SEQN", how="left")
    merged["cycle"] = label
    return merged


def harmonise(d: pd.DataFrame) -> pd.DataFrame:
    """Map NHANES variables to the app's inputs (same rules as src/external_validation.py)."""
    d = d.copy().round(6)  # SAS transport files store some zeros as ~5e-79
    out = pd.DataFrame(index=d.index)
    out["cycle"] = d["cycle"]
    out["age"] = d["RIDAGEYR"]
    out["gender"] = d["RIAGENDR"].map({1: 2, 2: 1})
    out["height"] = d["BMXHT"]
    out["weight"] = d["BMXWT"]
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in d]
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in d]
    out["ap_hi"] = d[sys_cols].mean(axis=1).round()
    out["ap_lo"] = d[dia_cols].where(lambda x: x > 0).mean(axis=1).round()
    out["cholesterol"] = pd.cut(d["LBXTC"], [-np.inf, 199.999, 239.999, np.inf], labels=[1, 2, 3]).astype(float)
    gluc = pd.cut(d["LBXGH"], [-np.inf, 5.699, 6.499, np.inf], labels=[1, 2, 3]).astype(float)
    gluc[d["DIQ010"] == 1] = 3
    out["gluc"] = gluc

    smoke = pd.Series(np.nan, index=d.index)
    smoke[(d["SMQ020"] == 2) | (d["SMQ040"] == 3)] = 0
    smoke[d["SMQ040"].isin([1, 2])] = 1
    out["smoke"] = smoke

    alco = pd.Series(np.nan, index=d.index)
    if "ALQ121" in d:  # 2017+: categories, 1-3 = every day / nearly every day / 3-4 times a week
        alco[d["ALQ121"].isin([0, 4, 5, 6, 7, 8, 9, 10]) | (d["ALQ111"] == 2)] = 0
        alco[d["ALQ121"].isin([1, 2, 3])] = 1
    else:
        weeks = d["ALQ120U"].map({1: 1.0, 2: 52 / 12, 3: 52.0})
        per_week = (d["ALQ120Q"] / weeks).where(d["ALQ120Q"] <= 365)
        alco[per_week.notna()] = (per_week[per_week.notna()] >= 3).astype(float)
        alco[d["ALQ120Q"] == 0] = 0
        alco[alco.isna() & ((d["ALQ101"] == 2) | (d["ALQ110"] == 2))] = 0
    out["alco"] = alco

    pa = d[["PAQ605", "PAQ620", "PAQ650", "PAQ665"]]
    active = pd.Series(np.nan, index=d.index)
    active[(pa == 2).all(axis=1)] = 0
    active[(pa == 1).any(axis=1)] = 1
    out["active"] = active

    # only asked of people told to take medication; everyone else isn't taking any
    out["bp_meds"] = (d["BPQ050A"] == 1).astype(float)
    out["lipid_meds"] = (d["BPQ100D"] == 1).astype(float)

    mcq = d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    cvd = pd.Series(np.nan, index=d.index)
    cvd[(mcq == 2).all(axis=1)] = 0
    cvd[(mcq == 1).any(axis=1)] = 1
    out["cvd"] = cvd
    out["pregnant"] = d["RIDEXPRG"] == 1
    return out


def cohort(h: pd.DataFrame) -> pd.DataFrame:
    h = h[(h["age"] >= 20) & (h["age"] <= 80) & ~h["pregnant"]].dropna(subset=FEATURES + ["cvd"])
    ok = (h["height"].between(100, 250) & h["weight"].between(30, 300) & h["ap_hi"].between(60, 250)
          & h["ap_lo"].between(30, 180) & (h["ap_lo"] < h["ap_hi"]))
    return h[ok]


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def auc_ci(y, s, n_boot=1000):
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(roc_auc_score(y[i], s[i]))
    return round(float(roc_auc_score(y, s)), 4), [round(float(v), 4) for v in np.percentile(stats, [2.5, 97.5])]


def choose_depth(X, y) -> Dict:
    """Pick tree depth and number of trees by 5-fold cross-validation on the training data only."""
    best = None
    for depth in (2, 3, 4):
        params = {"objective": "binary:logistic", "eval_metric": "auc", "eta": 0.05, "max_depth": depth,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0, "seed": SEED}
        cv = xgb.cv(params, xgb.DMatrix(X, label=y), num_boost_round=1500, nfold=5, stratified=True,
                    early_stopping_rounds=50, seed=SEED)
        score, rounds = float(cv["test-auc-mean"].iloc[-1]), len(cv)
        print(f"  depth {depth}: cross-validated ROC-AUC {score:.4f} with {rounds} trees")
        if best is None or score > best["cv_auc"]:
            best = {"max_depth": depth, "n_estimators": rounds, "cv_auc": round(score, 4)}
    return best


def make_model(hp: Dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                             eval_metric="auc", random_state=SEED, n_jobs=-1)


# ----------------------------------------------------------------------------- outputs
def write_readme(report: Dict) -> None:
    path = ROOT / "README.md"
    start, end = "<!-- nhanes-model:start -->", "<!-- nhanes-model:end -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    t, c, abl = report["test"], report["comparison_age_30_65"], report["ablation_without_medication"]
    verdict = ("beats" if t["roc_auc"] > t["roc_auc_age_alone"] else "does not beat")
    body = f"""{start}
Trained on {report['train']['patients']:,} US adults from NHANES {', '.join(report['train']['cycles'])} and tested on {t['patients']:,} adults from NHANES {report['test_cycle']}, a later survey the model never saw (with a new blood-pressure device). Inputs: the 11 app inputs plus whether the person takes blood-pressure or cholesterol medication.

| On the unseen 2017–2020 survey | ROC-AUC |
|---|---|
| US population model | {t['roc_auc']:.3f} (95% CI {t['roc_auc_95ci'][0]:.3f}–{t['roc_auc_95ci'][1]:.3f}) |
| Same model without the medication inputs | {abl['roc_auc']:.3f} |
| Ranking by age alone | {t['roc_auc_age_alone']:.3f} |
| Logistic regression, same inputs | {t['roc_auc_logistic']:.3f} |
| Ages 30–65: US population model vs clinic model | {c['us_population_model']:.3f} vs {c['clinic_model']:.3f} |

It {verdict} age alone. Calibration on the unseen survey: average predicted risk {t['mean_predicted_risk']:.1%} against {t['cvd_rate']:.1%} observed (calibration error {t['ece']:.1%}). Caveat: medication partly reflects treatment after a diagnosis, so this model recognises existing heart disease; it does not forecast future risk. Reproduce with `python src/train_nhanes_model.py`.
{end}"""
    path.write_text(text[:text.index(start)] + body + text[text.index(end) + len(end):], encoding="utf-8")


def update_frontend_report(value: Dict) -> None:
    path = ROOT / "frontend" / "src" / "data" / "model_reports.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data["nhanes"] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train the US population (NHANES) model")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "external" / "nhanes"))
    parser.add_argument("--offline", action="store_true", help="use cached files only")
    parser.add_argument("--out-dir", default=str(ROOT / "backend" / "models"))
    args = parser.parse_args(argv)
    cache = Path(args.cache_dir)

    print("Loading NHANES surveys:")
    train_raw = pd.concat([load_cycle(*c, cache, args.offline) for c in TRAIN_CYCLES], ignore_index=True)
    test_raw = load_cycle(*TEST_CYCLE, cache, args.offline)
    train_all, test = cohort(harmonise(train_raw)), cohort(harmonise(test_raw))
    for label, n in train_all["cycle"].value_counts().sort_index().items():
        print(f"  {label}: {n:,} adults with complete data")
    print(f"  {TEST_CYCLE[0]} (test): {len(test):,} adults")

    X_all, y_all = train_all[FEATURES].astype(float), train_all["cvd"].astype(int).to_numpy()
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    print("Choosing model size by 5-fold cross-validation on the training surveys:")
    hp = choose_depth(X_fit, y_fit)
    model = make_model(hp).fit(X_fit, y_fit)
    no_meds = make_model(hp).fit(X_fit[BASE_FEATURES], y_fit)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_fit, y_fit)

    print(f"Calibrating Venn-Abers ranges on {len(y_cal):,} held-out training adults...")
    va_grid = build_grid(model.predict_proba(X_cal)[:, 1], y_cal)
    va = VennAbersGrid(va_grid)

    X_te, y_te = test[FEATURES].astype(float), test["cvd"].astype(int).to_numpy()
    p_te = model.predict_proba(X_te)[:, 1]
    auc, ci = auc_ci(y_te, p_te)
    age_auc, age_ci = auc_ci(y_te, X_te["age"].to_numpy())
    widths = np.array([np.subtract(*va.interval(s)[::-1]) for s in p_te]) * 100
    untreated = (X_te["bp_meds"] == 0) & (X_te["lipid_meds"] == 0)

    # The clinic model only ever saw ages 30-65, so compare there
    import joblib
    clinic = joblib.load(ROOT / "backend" / "models" / "model_pipeline.pkl")
    band = X_te["age"].between(30, 65).to_numpy()
    p_clinic = clinic.predict_proba(X_te.loc[band, BASE_FEATURES].to_numpy(dtype=float))[:, 1]

    sex = np.where(X_te["gender"].to_numpy() == 2, "men", "women")
    report = {
        "trained_at": date.today().isoformat(),
        "xgboost_version": xgb.__version__,
        "features": FEATURES,
        "hyperparameters": hp,
        "train": {"cycles": [c[0] for c in TRAIN_CYCLES], "patients": int(len(y_all)), "fit": int(len(y_fit)),
                  "calibration": int(len(y_cal)), "cvd_rate": round(float(y_all.mean()), 4)},
        "test_cycle": TEST_CYCLE[0],
        "test": {
            "patients": int(len(y_te)), "cvd_cases": int(y_te.sum()), "cvd_rate": round(float(y_te.mean()), 4),
            "roc_auc": auc, "roc_auc_95ci": ci, "roc_auc_age_alone": age_auc, "roc_auc_age_alone_95ci": age_ci,
            "roc_auc_logistic": round(float(roc_auc_score(y_te, logistic.predict_proba(X_te)[:, 1])), 4),
            "roc_auc_by_sex": {s: round(float(roc_auc_score(y_te[sex == s], p_te[sex == s])), 4) for s in ("women", "men")},
            "roc_auc_untreated": round(float(roc_auc_score(y_te[untreated], p_te[untreated])), 4),
            "roc_auc_untreated_age_alone": round(float(roc_auc_score(y_te[untreated], X_te["age"][untreated])), 4),
            "mean_predicted_risk": round(float(p_te.mean()), 4), "ece": round(ece(y_te, p_te), 4),
            "brier": round(float(brier_score_loss(y_te, p_te)), 4),
            "venn_abers_width_points": {"median": round(float(np.median(widths)), 2),
                                        "p90": round(float(np.percentile(widths, 90)), 2)},
        },
        "ablation_without_medication": {
            "roc_auc": round(float(roc_auc_score(y_te, no_meds.predict_proba(X_te[BASE_FEATURES])[:, 1])), 4)},
        "comparison_age_30_65": {
            "patients": int(band.sum()),
            "us_population_model": round(float(roc_auc_score(y_te[band], p_te[band])), 4),
            "clinic_model": round(float(roc_auc_score(y_te[band], p_clinic)), 4),
            "age_alone": round(float(roc_auc_score(y_te[band], X_te["age"].to_numpy()[band])), 4),
        },
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "nhanes_model.json")
    (out / "uncertainty_nhanes.json").write_text(json.dumps(va_grid))
    (out / "reference_nhanes.json").write_text(json.dumps(reference_distribution(X_fit.to_dict(orient="records"), FEATURES), indent=1))
    t = report["test"]
    metadata = {
        "model_name": "XGBoost (NHANES)",
        "description": (f"Trained on {len(y_all):,} US adults from NHANES {TRAIN_CYCLES[0][0][:4]}–{TRAIN_CYCLES[-1][0][-4:]}; "
                        f"tested on {t['patients']:,} adults from the later {TEST_CYCLE[0]} survey."),
        "features": FEATURES,
        "test_metrics": {"roc_auc": t["roc_auc"], "roc_auc_age_alone": t["roc_auc_age_alone"], "ece": t["ece"],
                         "brier": t["brier"]},
        "trained_at": report["trained_at"],
        "xgboost_version": report["xgboost_version"],
    }
    (out / "nhanes_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if out == ROOT / "backend" / "models":
        (ROOT / "models").mkdir(exist_ok=True)
        (ROOT / "models" / "nhanes_report.json").write_text(json.dumps(report, indent=2) + "\n")
        update_frontend_report(report)
        write_readme(report)

    c, a = report["comparison_age_30_65"], report["ablation_without_medication"]
    print(f"\nTest on {TEST_CYCLE[0]} ({t['patients']:,} adults, {t['cvd_rate']:.1%} with heart disease):")
    print(f"  US population model  ROC-AUC {t['roc_auc']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})")
    print(f"  without medication   ROC-AUC {a['roc_auc']:.3f}")
    print(f"  age alone            ROC-AUC {age_auc:.3f} (95% CI {age_ci[0]:.3f}-{age_ci[1]:.3f})")
    print(f"  logistic regression  ROC-AUC {t['roc_auc_logistic']:.3f}")
    print(f"  ages 30-65: US population {c['us_population_model']:.3f} vs clinic model {c['clinic_model']:.3f} "
          f"vs age alone {c['age_alone']:.3f}")
    print(f"  calibration: predicts {t['mean_predicted_risk']:.1%} on average vs {t['cvd_rate']:.1%} observed "
          f"(error {t['ece']:.1%}); range width median {t['venn_abers_width_points']['median']} points")
    print(f"Saved the model to {out / 'nhanes_model.json'}")


if __name__ == "__main__":
    main()
