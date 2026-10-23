"""External validation: does CardioLens hold up on data it has never seen?

Scores the deployed model, unchanged, on adults from the US National Health and
Nutrition Examination Survey (NHANES) 2013-2014, a different country, population
and measurement process from the Kaggle training data. NHANES is public domain
data from the US Centers for Disease Control and Prevention (CDC).

Run from the repo root:   python src/external_validation.py
It downloads the NHANES files from CDC once (about 10 MB, cached in
data/external/) and writes models/external_validation_report.json.

Mapping from NHANES to the model's 11 inputs (documented in the report too):
  age             RIDAGEYR, restricted to 30-65, the training data's age range
  gender          RIAGENDR (1 male, 2 female) -> model coding (2 male, 1 female)
  height, weight  BMXHT (cm), BMXWT (kg)
  ap_hi, ap_lo    mean of the measured readings BPXSY1-4 / BPXDI1-4
  cholesterol     LBXTC mg/dL: <200 normal (1), 200-239 above normal (2), >=240 well above (3)
  gluc            HbA1c LBXGH: <5.7% normal (1), 5.7-6.4% above normal (2),
                  >=6.5% or diagnosed diabetes (DIQ010) well above (3)
  smoke           smokes now, every day or some days (SMQ040)
  alco            drinks alcohol on 3 or more days a week (ALQ120Q/ALQ120U)
  active          any moderate or vigorous work or leisure activity (PAQ605/620/650/665)
Outcome: a doctor ever said the person had coronary heart disease, angina, a heart
attack, heart failure or a stroke (MCQ160B-F).
"""
import argparse
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

CDC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/{name}.xpt"
CDC_FILES = {
    "DEMO_H": ["RIAGENDR", "RIDAGEYR", "RIDEXPRG"],
    "BMX_H": ["BMXHT", "BMXWT"],
    "BPX_H": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "TCHOL_H": ["LBXTC"],
    "GHB_H": ["LBXGH"],
    "SMQ_H": ["SMQ020", "SMQ040"],
    "ALQ_H": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U"],
    "PAQ_H": ["PAQ605", "PAQ620", "PAQ650", "PAQ665"],
    "MCQ_H": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "DIQ_H": ["DIQ010"],
    "BPQ_H": ["BPQ050A", "BPQ100D"],  # now taking blood-pressure / cholesterol medication
}
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
CACHE = Path("data/external/nhanes_2013_2014")
MODEL_PATH = "backend/models/model_pipeline.pkl"
THRESHOLD = 0.5
SEED = 42


# ----------------------------------------------------------------------------- loading
def _download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.xpt"
    if not path.exists():
        url = CDC_BASE.format(name=name)
        print(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CardioLens external validation)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def load_from_cdc() -> pd.DataFrame:
    merged = None
    for name, cols in CDC_FILES.items():
        df = pd.read_sas(_download(name), format="xport")
        df.columns = [c.upper() for c in df.columns]
        df = df[["SEQN"] + cols]
        merged = df if merged is None else merged.merge(df, on="SEQN", how="left")
    return merged


def load_from_csv_mirror(folder: str) -> pd.DataFrame:
    """The same CDC files as merged CSVs (Kaggle "National Health and Nutrition Examination Survey")."""
    wanted = {c for cols in CDC_FILES.values() for c in cols} | {"SEQN"}
    merged = None
    for name in ["demographic", "examination", "labs", "questionnaire"]:
        df = pd.read_csv(Path(folder) / f"{name}.csv", usecols=lambda c: c.upper() in wanted)
        df.columns = [c.upper() for c in df.columns]
        merged = df if merged is None else merged.merge(df, on="SEQN", how="left")
    return merged


# ----------------------------------------------------------------------------- mapping
def to_model_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy().round(6)  # SAS transport files store some zeros as ~5e-79
    out = pd.DataFrame(index=d.index)
    out["age"] = d["RIDAGEYR"]
    out["gender"] = d["RIAGENDR"].map({1: 2, 2: 1})
    out["height"] = d["BMXHT"]
    out["weight"] = d["BMXWT"]
    out["ap_hi"] = d[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1).round()
    # a diastolic reading of 0 means sounds were heard down to zero: not a usable value
    out["ap_lo"] = d[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].where(lambda x: x > 0).mean(axis=1).round()

    out["cholesterol"] = pd.cut(d["LBXTC"], [-np.inf, 199.999, 239.999, np.inf], labels=[1, 2, 3]).astype(float)
    gluc = pd.cut(d["LBXGH"], [-np.inf, 5.699, 6.499, np.inf], labels=[1, 2, 3]).astype(float)
    gluc[d["DIQ010"] == 1] = 3
    out["gluc"] = gluc

    smoke = pd.Series(np.nan, index=d.index)
    smoke[(d["SMQ020"] == 2) | (d["SMQ040"] == 3)] = 0
    smoke[d["SMQ040"].isin([1, 2])] = 1
    out["smoke"] = smoke

    per_week = d["ALQ120Q"] / d["ALQ120U"].map({1: 1.0, 2: 52 / 12 / 1.0, 3: 52.0}).where(lambda x: x > 0)
    per_week = per_week.where(d["ALQ120Q"] <= 365)  # 777/999 = refused / don't know
    alco = (per_week >= 3).astype(float).where(per_week.notna())
    alco[(d["ALQ120Q"] == 0)] = 0
    alco[alco.isna() & ((d["ALQ101"] == 2) | (d["ALQ110"] == 2))] = 0  # never a regular drinker
    out["alco"] = alco

    pa = d[["PAQ605", "PAQ620", "PAQ650", "PAQ665"]]
    active = pd.Series(np.nan, index=d.index)
    active[(pa == 2).all(axis=1)] = 0
    active[(pa == 1).any(axis=1)] = 1
    out["active"] = active

    mcq = d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    cvd = pd.Series(np.nan, index=d.index)
    cvd[(mcq == 2).all(axis=1)] = 0
    cvd[(mcq == 1).any(axis=1)] = 1
    out["cvd"] = cvd

    out["pregnant"] = d["RIDEXPRG"] == 1
    # only asked of people told to take medication; everyone else is not taking any
    out["on_bp_or_lipid_meds"] = (d["BPQ050A"] == 1) | (d["BPQ100D"] == 1)
    return out


def select_cohort(m: pd.DataFrame):
    steps = {"NHANES 2013-2014 participants": len(m)}
    m = m[(m["age"] >= 30) & (m["age"] <= 65)]
    steps["aged 30-65 (training age range)"] = len(m)
    m = m[~m["pregnant"]]
    steps["not pregnant"] = len(m)
    m = m.dropna(subset=FEATURES + ["cvd"])
    steps["all 11 inputs and the outcome recorded"] = len(m)
    # the same range checks the API applies to every patient
    ok = (m["height"].between(100, 250) & m["weight"].between(30, 300) & m["ap_hi"].between(60, 250)
          & m["ap_lo"].between(30, 180) & (m["ap_lo"] < m["ap_hi"]))
    m = m[ok]
    steps["passed the API's input checks"] = len(m)
    return m, steps


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def auc_with_ci(y, score, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n, stats = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            stats.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(roc_auc_score(y, score)), 4), [round(float(lo), 4), round(float(hi), 4)]


def calibration(y, p):
    return {"mean_predicted_risk": round(float(p.mean()), 4), "observed_rate": round(float(y.mean()), 4),
            "ece": round(ece(y, p), 4), "brier": round(float(brier_score_loss(y, p)), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv-dir", help="use the Kaggle CSV copy of the same CDC files instead of downloading")
    args = parser.parse_args()

    raw = load_from_csv_mirror(args.csv_dir) if args.csv_dir else load_from_cdc()
    cohort, steps = select_cohort(to_model_inputs(raw))
    X = cohort[FEATURES].to_numpy(dtype=float)
    y = cohort["cvd"].to_numpy(dtype=int)

    model = joblib.load(MODEL_PATH)
    p = model.predict_proba(X)[:, 1]
    flagged = (p >= THRESHOLD).astype(int)

    auc, auc_ci = auc_with_ci(y, p)
    age_auc, age_ci = auc_with_ci(y, cohort["age"].to_numpy())

    # Percentages from a model trained where half the patients were sick won't match a
    # population where few are. Refit only the scale (logistic recalibration) on one half
    # and judge it on the other half, so the check is honest.
    rng = np.random.default_rng(SEED)
    half = rng.permutation(len(y)) < len(y) // 2
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    recal = LogisticRegression().fit(logit[half], y[half])
    p_recal = recal.predict_proba(logit[~half])[:, 1]

    # Sensitivity check: the alcohol mapping is the loosest one, so try "nobody drinks"
    X_no_alco = X.copy()
    X_no_alco[:, FEATURES.index("alco")] = 0
    auc_no_alco = round(float(roc_auc_score(y, model.predict_proba(X_no_alco)[:, 1])), 4)

    sex = np.where(cohort["gender"].to_numpy() == 2, "male", "female")

    # Why the drop? Systolic blood pressure is the model's strongest input. Compare how well it
    # separates sick from healthy here versus in the training data, and look at medication,
    # which lowers measured blood pressure and cholesterol in people already diagnosed.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_and_preprocess_data
    _, X_int, _, y_int = load_and_preprocess_data("data/cardiovascular_disease.csv")[:4]
    sbp_auc_internal = round(float(roc_auc_score(y_int, X_int["ap_hi"].clip(60, 250))), 4)
    sbp_auc_nhanes = round(float(roc_auc_score(y, cohort["ap_hi"])), 4)
    treated = cohort["on_bp_or_lipid_meds"].to_numpy()
    age = cohort["age"].to_numpy()
    by_treatment = {}
    for name, mask in [("untreated", ~treated), ("on_medication", treated)]:
        by_treatment[name] = {
            "patients": int(mask.sum()), "cvd_cases": int(y[mask].sum()),
            "roc_auc": round(float(roc_auc_score(y[mask], p[mask])), 4),
            "roc_auc_age_alone": round(float(roc_auc_score(y[mask], age[mask])), 4),
        }
    report = {
        "dataset": "NHANES 2013-2014 (US CDC, public domain)",
        "cohort_steps": steps,
        "patients": int(len(y)),
        "cvd_cases": int(y.sum()),
        "cvd_rate": round(float(y.mean()), 4),
        "roc_auc": auc,
        "roc_auc_95ci": auc_ci,
        "roc_auc_age_alone": age_auc,
        "roc_auc_age_alone_95ci": age_ci,
        "roc_auc_by_sex": {s: round(float(roc_auc_score(y[sex == s], p[sex == s])), 4) for s in ["female", "male"]},
        "at_threshold_0_5": {
            "flagged_share": round(float(flagged.mean()), 4),
            "recall": round(float(recall_score(y, flagged)), 4),
            "precision": round(float(precision_score(y, flagged, zero_division=0)), 4),
        },
        "calibration_as_deployed": calibration(y, p),
        "calibration_after_recalibration_on_other_half": calibration(y[~half], p_recal),
        "sensitivity_alcohol_set_to_no": {"roc_auc": auc_no_alco},
        "why": {
            "systolic_bp_alone_roc_auc": {"training_data_test_set": sbp_auc_internal, "nhanes": sbp_auc_nhanes},
            "share_on_bp_or_lipid_medication": {
                "everyone": round(float(treated.mean()), 4),
                "people_with_cvd": round(float(treated[y == 1].mean()), 4),
            },
            "by_medication": by_treatment,
        },
        "for_comparison_internal_test_set": {"roc_auc": 0.7978, "cvd_rate": 0.5},
    }

    Path("models").mkdir(exist_ok=True)
    with open("models/external_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Cohort:")
    for k, v in steps.items():
        print(f"  {v:>6,}  {k}")
    print(f"\nPatients {len(y):,}, with heart disease {y.sum():,} ({y.mean():.1%})")
    print(f"ROC-AUC  {auc:.3f} (95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})   internal test set: 0.798")
    print(f"Age alone {age_auc:.3f} (95% CI {age_ci[0]:.3f}-{age_ci[1]:.3f})")
    print(f"By sex: {report['roc_auc_by_sex']}")
    print(f"At 0.5: flags {flagged.mean():.1%} of people, catches {report['at_threshold_0_5']['recall']:.1%} "
          f"of cases, precision {report['at_threshold_0_5']['precision']:.1%}")
    c, r = report["calibration_as_deployed"], report["calibration_after_recalibration_on_other_half"]
    print(f"Calibration as deployed: predicts {c['mean_predicted_risk']:.1%} on average vs {c['observed_rate']:.1%} "
          f"observed (ECE {c['ece']:.1%})")
    print(f"After recalibration (held-out half): predicts {r['mean_predicted_risk']:.1%} vs {r['observed_rate']:.1%} "
          f"(ECE {r['ece']:.1%})")
    print(f"Sensitivity, alcohol set to 'no' for everyone: ROC-AUC {auc_no_alco:.3f}")
    print(f"\nWhy: systolic BP alone scores ROC-AUC {sbp_auc_internal:.3f} in the training data but {sbp_auc_nhanes:.3f} here.")
    print(f"     {treated[y == 1].mean():.0%} of people with heart disease take BP or cholesterol medication "
          f"(vs {treated.mean():.0%} of everyone).")
    for name, r in by_treatment.items():
        print(f"     {name:13s} n={r['patients']:,} cases={r['cvd_cases']}  model {r['roc_auc']:.3f}  "
              f"age alone {r['roc_auc_age_alone']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
