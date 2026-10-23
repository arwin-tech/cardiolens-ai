"""Privacy-safe model monitoring.

What is stored: daily counts in coarse buckets, for example "systolic 130-139: 14
today" or "response time 50-100 ms: 9 today". Never patient inputs, exact values,
timestamps finer than a day, or IP addresses.

What is shown: the public summary hides any count below K_MIN and only reports
input drift once a model has DRIFT_MIN predictions in the window. On a quiet day
with a single prediction, even bucketed counts would otherwise describe that one
person.

Storage: set DATABASE_URL (for example a free Supabase or Neon Postgres) to keep
counts across restarts. Without it, counts live in memory and reset whenever the
server restarts or Render's free tier puts it to sleep.
"""
import logging
import math
import os
import threading
from collections import Counter
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool
except ImportError:  # the training scripts only need bucket() and reference_distribution()
    create_engine = text = StaticPool = None

logger = logging.getLogger(__name__)

K_MIN = 5
DRIFT_MIN = 30
LATENCY_BUCKETS = [(50, "<50 ms"), (100, "50-100 ms"), (250, "100-250 ms"), (500, "250-500 ms"),
                   (1000, "500-1000 ms"), (math.inf, "1000+ ms")]
RANGE_FEATURES = {"age": 5, "height": 10, "weight": 10, "ap_hi": 10, "ap_lo": 10}


def bucket(feature: str, value) -> str:
    """Coarse bucket for one input value. Shared with the training scripts so live
    inputs and the training reference use identical buckets."""
    v = float(value)
    width = RANGE_FEATURES.get(feature)
    if width:
        lo = int(v // width * width)
        return f"{lo}-{lo + width - 1}"
    return str(int(round(v)))


def reference_distribution(rows: Iterable[Dict], features: List[str]) -> Dict[str, Dict[str, float]]:
    """Share of training patients in each bucket, per feature."""
    counts = {f: Counter() for f in features}
    n = 0
    for row in rows:
        n += 1
        for f in features:
            counts[f][bucket(f, row[f])] += 1
    return {f: {b: round(c / n, 6) for b, c in sorted(counts[f].items())} for f in features}


def psi(live: Dict[str, int], reference: Dict[str, float]) -> float:
    """Population stability index between live bucket counts and reference shares."""
    total = sum(live.values())
    eps = 1e-4
    value = 0.0
    for b in set(live) | set(reference):
        p = max(live.get(b, 0) / total, eps)
        q = max(reference.get(b, 0.0), eps)
        value += (p - q) * math.log(p / q)
    return value


def _latency_label(ms: float) -> str:
    return next(label for limit, label in LATENCY_BUCKETS if ms < limit)


def _normalise_url(url: str) -> str:
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class MonitoringStore:
    def __init__(self, url: Optional[str] = None):
        url = url if url is not None else os.getenv("DATABASE_URL", "")
        self.persistent = bool(url)
        if url:
            self.engine = create_engine(_normalise_url(url), pool_pre_ping=True, pool_size=2, max_overflow=2)
        else:
            self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                        connect_args={"check_same_thread": False})
        self._lock = threading.Lock()
        with self.engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS monitoring_daily ("
                " day VARCHAR(10) NOT NULL, metric VARCHAR(80) NOT NULL, bucket VARCHAR(60) NOT NULL,"
                " n BIGINT NOT NULL, PRIMARY KEY (day, metric, bucket))"
            ))

    # ------------------------------------------------------------------ writes
    def _increment(self, counts: Counter) -> None:
        if not counts:
            return
        day = date.today().isoformat()
        stmt = text(
            "INSERT INTO monitoring_daily (day, metric, bucket, n) VALUES (:day, :metric, :bucket, :n) "
            "ON CONFLICT (day, metric, bucket) DO UPDATE SET n = monitoring_daily.n + excluded.n"
        )
        params = [{"day": day, "metric": m, "bucket": b, "n": n} for (m, b), n in counts.items()]
        try:
            with self._lock, self.engine.begin() as conn:
                conn.execute(stmt, params)
        except Exception as e:  # monitoring must never break predictions
            logger.warning(f"Monitoring write failed: {e}")

    def record_predictions(self, model_id: str, patients: List[Dict], categories: List[str], features: List[str]) -> None:
        counts = Counter()
        counts[("predictions", model_id)] += len(patients)
        for c in categories:
            counts[("category", f"{model_id}|{c}")] += 1
        for p in patients:
            for f in features:
                counts[(f"input|{model_id}|{f}", bucket(f, p[f]))] += 1
        self._increment(counts)

    def record_request(self, endpoint: str, ms: float) -> None:
        self._increment(Counter({("requests", endpoint): 1, ("latency", f"{endpoint}|{_latency_label(ms)}"): 1}))

    # ------------------------------------------------------------------- reads
    def raw_rows(self) -> List[Dict]:
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(text("SELECT * FROM monitoring_daily"))]

    def summary(self, days: int, references: Dict[str, Dict[str, Dict[str, float]]]) -> Dict:
        days = max(1, min(int(days), 90))
        today = date.today()
        start = (today - timedelta(days=days - 1)).isoformat()
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT day, metric, bucket, n FROM monitoring_daily WHERE day >= :s"),
                                {"s": start}).all()

        per_day, per_model, requests = Counter(), Counter(), Counter()
        categories, latency, inputs = {}, {}, {}
        for day, metric, b, n in rows:
            n = int(n)
            if metric == "predictions":
                per_day[day] += n
                per_model[b] += n
            elif metric == "requests":
                requests[b] += n
            elif metric == "category":
                model_id, cat = b.split("|", 1)
                categories.setdefault(model_id, Counter())[cat] += n
            elif metric == "latency":
                endpoint, label = b.split("|", 1)
                latency.setdefault(endpoint, Counter())[label] += n
            elif metric.startswith("input|"):
                _, model_id, feature = metric.split("|", 2)
                inputs.setdefault(model_id, {}).setdefault(feature, Counter())[b] += n

        def shown(n: int):
            return n if n >= K_MIN else None

        drift = {}
        for model_id, ref in references.items():
            have = per_model.get(model_id, 0)
            if have < DRIFT_MIN:
                drift[model_id] = {"status": "not_enough_data", "have": have, "needed": DRIFT_MIN, "features": {}}
                continue
            feats = {}
            for feature, ref_dist in ref.items():
                live = inputs.get(model_id, {}).get(feature)
                if not live:
                    continue
                value = psi(live, ref_dist)
                feats[feature] = {"psi": round(value, 3),
                                  "status": "stable" if value < 0.1 else "moderate shift" if value < 0.25 else "major shift"}
            drift[model_id] = {"status": "ok", "have": have, "needed": DRIFT_MIN, "features": feats}

        latency_out = {}
        for endpoint, hist in latency.items():
            total = sum(hist.values())
            ordered = [(label, hist.get(label, 0)) for _, label in LATENCY_BUCKETS]
            latency_out[endpoint] = {
                "requests": total,
                "histogram": [{"bucket": label, "count": shown(c)} for label, c in ordered],
                "median": _quantile_bucket(ordered, 0.5),
                "p95": _quantile_bucket(ordered, 0.95),
            }

        return {
            "window_days": days,
            "storage": "persistent" if self.persistent else "in_memory",
            "predictions_total": sum(per_model.values()),
            "predictions_by_model": dict(per_model),
            "predictions_per_day": [
                {"day": (today - timedelta(days=i)).isoformat(), "predictions": per_day.get((today - timedelta(days=i)).isoformat(), 0)}
                for i in range(days - 1, -1, -1)
            ],
            "requests_by_endpoint": dict(requests),
            "risk_categories": {m: {cat: shown(n) for cat, n in sorted(c.items())} for m, c in categories.items()},
            "response_times": latency_out,
            "drift": drift,
            "privacy": {
                "stored": "daily counts per coarse bucket only; no patient inputs, exact values, times or IP addresses",
                "smallest_count_shown": K_MIN,
                "drift_needs_predictions": DRIFT_MIN,
            },
        }


def _quantile_bucket(ordered: List, q: float) -> Optional[str]:
    total = sum(c for _, c in ordered)
    if total < K_MIN:
        return None
    running = 0
    for label, c in ordered:
        running += c
        if running >= q * total:
            return label
    return ordered[-1][0]
