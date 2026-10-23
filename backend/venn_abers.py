"""Venn-Abers predictors: calibrated probability *ranges* (a conformal prediction method).

For a model score s, Venn-Abers fits isotonic regression on a held-out calibration
set twice, once pretending the new patient had the disease (label 1) and once not
(label 0). The two fitted values p0 <= p1 bracket the calibrated probability, with
validity guarantees from conformal prediction theory. Narrow ranges mean plenty of
similar calibration patients; wide ranges flag scores the model has rarely seen.

Refitting per request is slow, so the ranges are precomputed on a fine grid of
scores offline (build_grid) and interpolated at request time (VennAbersGrid).
"""
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def build_grid(cal_scores: Sequence[float], cal_labels: Sequence[int], points: int = 1001) -> Dict[str, List[float]]:
    """Offline: compute (p0, p1) for evenly spaced scores in [0, 1]."""
    from sklearn.isotonic import IsotonicRegression  # only needed offline

    scores = np.asarray(cal_scores, dtype=float)
    labels = np.asarray(cal_labels, dtype=float)
    grid = np.linspace(0.0, 1.0, points)
    p0, p1 = [], []
    for s in grid:
        xs = np.append(scores, s)
        for label, out in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(xs, np.append(labels, label))
            out.append(float(iso.predict([s])[0]))
    return {"grid": grid.round(6).tolist(), "p0": np.round(p0, 6).tolist(), "p1": np.round(p1, 6).tolist(),
            "calibration_patients": int(len(scores))}


class VennAbersGrid:
    def __init__(self, data: Dict[str, List[float]]):
        self.grid = np.asarray(data["grid"])
        self.p0 = np.asarray(data["p0"])
        self.p1 = np.asarray(data["p1"])
        self.calibration_patients = int(data.get("calibration_patients", 0))

    @classmethod
    def load(cls, path: Path) -> "VennAbersGrid":
        with open(path) as f:
            return cls(json.load(f))

    def interval(self, score: float) -> Tuple[float, float]:
        lo = float(np.interp(score, self.grid, self.p0))
        hi = float(np.interp(score, self.grid, self.p1))
        return min(lo, hi), max(lo, hi)

    def merged(self, score: float) -> float:
        """Single calibrated estimate from the range (Vovk & Petej, 2014)."""
        p0, p1 = self.interval(score)
        return p1 / (1.0 - p0 + p1)
