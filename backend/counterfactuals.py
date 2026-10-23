"""Counterfactual suggestions: the smallest sets of changes that lower predicted risk.

Searches every combination of realistic, healthier changes to the things a person
can change (blood pressure, weight, smoking, alcohol, activity, cholesterol and
glucose levels), scores them all in one batch, and returns the best plan using one,
two and three changes. Age, sex, height and medication are never changed.

These are statements about the *model*, not proven medical effects.
"""
from itertools import product
from typing import Callable, Dict, List

import numpy as np

LEVELS = {1: "normal", 2: "above normal", 3: "well above normal"}


def _bp_options(ap_hi: int, ap_lo: int):
    out = [(0, ap_hi, ap_lo)]
    for drop in (10, 20, 30):
        new_hi = ap_hi - drop
        if new_hi < 110:
            break
        new_lo = max(60, ap_lo - round(drop / 2))
        new_lo = min(new_lo, new_hi - 20)
        out.append((drop, new_hi, new_lo))
    return out


def _weight_options(weight: float, height_cm: float):
    floor = 22.0 * (height_cm / 100.0) ** 2  # don't suggest going below a BMI of 22
    return [0.0] + [d for d in (5.0, 10.0) if weight - d >= floor]


def suggest(patient: Dict, features: List[str], predict: Callable[[np.ndarray], np.ndarray], max_changes: int = 3) -> Dict:
    base_risk = float(predict(np.array([[patient[f] for f in features]], dtype=float))[0])

    bp = _bp_options(int(patient["ap_hi"]), int(patient["ap_lo"]))
    weight = _weight_options(float(patient["weight"]), float(patient["height"]))
    smoke = [patient["smoke"]] + ([0] if patient["smoke"] == 1 else [])
    alco = [patient["alco"]] + ([0] if patient["alco"] == 1 else [])
    active = [patient["active"]] + ([1] if patient["active"] == 0 else [])
    chol = list(range(int(patient["cholesterol"]), 0, -1))
    gluc = list(range(int(patient["gluc"]), 0, -1))

    combos, rows = [], []
    for (drop, hi, lo), w, s, a, act, c, g in product(bp, weight, smoke, alco, active, chol, gluc):
        changed = {
            "bp": drop > 0, "weight": w > 0, "smoke": s != patient["smoke"], "alco": a != patient["alco"],
            "active": act != patient["active"], "cholesterol": c != patient["cholesterol"], "gluc": g != patient["gluc"],
        }
        n = sum(changed.values())
        if n == 0 or n > max_changes:
            continue
        candidate = dict(patient, ap_hi=hi, ap_lo=lo, weight=patient["weight"] - w, smoke=s, alco=a,
                         active=act, cholesterol=c, gluc=g)
        effort = drop / 10 + w / 5 + (patient["cholesterol"] - c) + (patient["gluc"] - g) + sum(
            changed[k] for k in ("smoke", "alco", "active"))
        combos.append((n, effort, candidate, changed))
        rows.append([candidate[f] for f in features])

    plans = []
    if rows:
        risks = predict(np.array(rows, dtype=float))
        best_so_far = base_risk
        for k in range(1, max_changes + 1):
            options = [(risks[i], combos[i][1], i) for i in range(len(combos)) if combos[i][0] == k]
            if not options:
                continue
            risk, _, i = min(options)  # lowest risk, then least effort
            if risk >= best_so_far - 0.005:  # must beat the smaller plan by at least half a point
                continue
            best_so_far = risk
            _, _, cand, changed = combos[i]
            plans.append({
                "changes": _describe(patient, cand, changed),
                "risk": round(float(risk) * 100, 1),
                "delta": round(float(risk - base_risk) * 100, 1),
            })

    return {
        "baseline_risk": round(base_risk * 100, 1),
        "plans": plans,
        "note": "These show which changes lower the model's prediction. They are not proven medical effects; "
                "decisions about treatment belong with a doctor.",
    }


def _describe(before: Dict, after: Dict, changed: Dict) -> List[Dict]:
    out = []
    if changed["bp"]:
        out.append({"feature": "blood_pressure", "label": f"Lower blood pressure to {after['ap_hi']}/{after['ap_lo']} mmHg",
                    "from": f"{before['ap_hi']}/{before['ap_lo']}", "to": f"{after['ap_hi']}/{after['ap_lo']}"})
    if changed["weight"]:
        lost = before["weight"] - after["weight"]
        out.append({"feature": "weight", "label": f"Lose {lost:g} kg (to {after['weight']:g} kg)",
                    "from": before["weight"], "to": after["weight"]})
    if changed["smoke"]:
        out.append({"feature": "smoke", "label": "Stop smoking", "from": 1, "to": 0})
    if changed["alco"]:
        out.append({"feature": "alco", "label": "Stop regular drinking", "from": 1, "to": 0})
    if changed["active"]:
        out.append({"feature": "active", "label": "Become physically active", "from": 0, "to": 1})
    if changed["cholesterol"]:
        out.append({"feature": "cholesterol", "label": f"Bring cholesterol to {LEVELS[after['cholesterol']]}",
                    "from": before["cholesterol"], "to": after["cholesterol"]})
    if changed["gluc"]:
        out.append({"feature": "gluc", "label": f"Bring glucose to {LEVELS[after['gluc']]}",
                    "from": before["gluc"], "to": after["gluc"]})
    return out
