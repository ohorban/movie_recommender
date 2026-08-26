"""The personalized ranking model.

A few hundred ratings is a small-data problem, and treating it like a big-data
one is the classic way to build a recommender that confidently suggests
nonsense. Three defences:

1. **Regularised linear models first.** Ridge on standardised features is hard
   to overfit and its coefficients are readable, which makes the Insights tab
   honest rather than decorative. Gradient boosting is fitted too but only wins
   if it beats ridge in cross-validation.
2. **Cross-validated model selection**, scored with Spearman rank correlation -
   the ranking is what matters, not the absolute predicted rating.
3. **Blending with a hand-tuned heuristic**, weighted by how well the learned
   model actually did. If cross-validation says the model has learned nothing,
   its influence goes to zero automatically.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..db import fetch_all, transaction, utcnow
from ..logging_utils import get_logger
from .features import FEATURE_NAMES, HEURISTIC_WEIGHTS, FeatureMatrix

log = get_logger("recommend.ranker")

MIN_TRAINING_ROWS = 25
CONFIDENCE_FULL_TRUST = 0.45  # Spearman at which the learned model is trusted outright


@dataclass
class RankerMetrics:
    n_train: int = 0
    spearman: float = 0.0
    mae: float = 0.0
    ndcg_at_10: float = 0.0
    model_kind: str = "heuristic"
    blend_weight: float = 0.0
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "spearman": round(self.spearman, 4),
            "mae": round(self.mae, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "model_kind": self.model_kind,
            "blend_weight": round(self.blend_weight, 4),
            "top_features": [(n, round(v, 4)) for n, v in self.top_features],
        }


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    Plain ``argsort(argsort(x))`` breaks ties arbitrarily but monotonically, so
    a constant vector comes out as 0,1,2,3... and correlates perfectly with
    anything. That would report a flawless model whenever every candidate
    scored the same, which is exactly when the model knows nothing.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    gains = np.clip(y_true - y_true.min(), 0, None)
    if gains.sum() <= 0:
        return 0.0
    k = min(k, y_true.size)
    order = np.argsort(-y_score)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains[order] * discounts).sum())
    ideal = float((np.sort(gains)[::-1][:k] * discounts).sum())
    return dcg / ideal if ideal > 0 else 0.0


def heuristic_scores(fm: FeatureMatrix) -> np.ndarray:
    """Hand-weighted linear score over the same features."""
    w = np.array([HEURISTIC_WEIGHTS.get(n, 0.0) for n in fm.names], dtype=np.float32)
    return fm.matrix @ w


@dataclass
class TasteRanker:
    """Fitted ranking model, or a heuristic-only stand-in."""

    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    metrics: RankerMetrics = field(default_factory=RankerMetrics)
    _model: Any = None
    _heuristic_mean: float = 0.0
    _heuristic_std: float = 1.0
    _learned_mean: float = 0.0
    _learned_std: float = 1.0

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, fm: FeatureMatrix, targets: np.ndarray) -> TasteRanker:
        ranker = cls(feature_names=list(fm.names))
        n = fm.matrix.shape[0]
        heur = heuristic_scores(fm)
        ranker._heuristic_mean = float(heur.mean())
        ranker._heuristic_std = float(heur.std()) or 1.0

        if n < MIN_TRAINING_ROWS:
            ranker.metrics = RankerMetrics(n_train=n, model_kind="heuristic", blend_weight=0.0)
            log.info("only %d training rows; using the heuristic ranker", n)
            return ranker

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = fm.matrix.astype(np.float64), targets.astype(np.float64)
        candidates: dict[str, Any] = {
            "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 25))),
        }
        if n >= 80:
            candidates["gbdt"] = GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                min_samples_leaf=8,
                random_state=17,
            )

        folds = KFold(n_splits=min(5, max(3, n // 20)), shuffle=True, random_state=17)
        best_name, best_score, best_oof = "heuristic", _spearman(y, heur), heur
        for name, model in candidates.items():
            oof = np.zeros(n)
            try:
                for train_idx, test_idx in folds.split(X):
                    import copy

                    fold_model = copy.deepcopy(model)
                    fold_model.fit(X[train_idx], y[train_idx])
                    oof[test_idx] = fold_model.predict(X[test_idx])
            except Exception as exc:
                log.warning("cross-validation failed for %s: %s", name, exc)
                continue
            score = _spearman(y, oof)
            log.info("cv %-9s spearman=%.3f", name, score)
            if score > best_score:
                best_name, best_score, best_oof = name, score, oof

        ranker.metrics = RankerMetrics(
            n_train=n,
            spearman=best_score,
            mae=float(np.abs(y - best_oof).mean())
            if best_name != "heuristic"
            else float(np.abs(y - y.mean()).mean()),
            ndcg_at_10=_ndcg_at_k(y, best_oof, 10),
            model_kind=best_name,
        )

        if best_name != "heuristic":
            model = candidates[best_name]
            model.fit(X, y)
            ranker._model = model
            preds = model.predict(X)
            ranker._learned_mean = float(preds.mean())
            ranker._learned_std = float(preds.std()) or 1.0
            ranker.metrics.top_features = ranker._explain(model, X, y)

        # Trust the model in proportion to demonstrated skill.
        ranker.metrics.blend_weight = (
            float(np.clip(best_score / CONFIDENCE_FULL_TRUST, 0.0, 1.0))
            if best_name != "heuristic"
            else 0.0
        )
        log.info(
            "ranker: %s, spearman=%.3f, ndcg@10=%.3f, blend=%.2f",
            best_name,
            best_score,
            ranker.metrics.ndcg_at_10,
            ranker.metrics.blend_weight,
        )
        return ranker

    def _explain(self, model: Any, X: np.ndarray, y: np.ndarray) -> list[tuple[str, float]]:
        """Readable feature importances, standardised so they compare."""
        try:
            if hasattr(model, "named_steps"):
                coefs = model.named_steps["ridgecv"].coef_
                pairs = list(zip(self.feature_names, [float(c) for c in coefs]))
            elif hasattr(model, "feature_importances_"):
                pairs = list(
                    zip(self.feature_names, [float(v) for v in model.feature_importances_])
                )
            else:
                return []
        except Exception:
            return []
        return sorted(pairs, key=lambda t: -abs(t[1]))[:10]

    # -------------------------------------------------------------- predict
    def score(self, fm: FeatureMatrix) -> np.ndarray:
        """Blended, standardised preference score for each candidate."""
        heur = heuristic_scores(fm)
        heur_z = (heur - self._heuristic_mean) / self._heuristic_std
        if self._model is None or self.metrics.blend_weight <= 0:
            return heur_z
        try:
            learned = np.asarray(
                self._model.predict(fm.matrix.astype(np.float64)), dtype=np.float64
            )
        except Exception as exc:
            log.warning("learned model failed at predict time: %s", exc)
            return heur_z
        learned_z = (learned - self._learned_mean) / self._learned_std
        w = self.metrics.blend_weight
        return w * learned_z + (1.0 - w) * heur_z

    # ---------------------------------------------------------- persistence
    def save(self, conn: sqlite3.Connection) -> None:
        import base64
        import pickle

        blob = (
            base64.b64encode(pickle.dumps(self._model)).decode("ascii")
            if self._model is not None
            else ""
        )
        payload = {
            "feature_names": self.feature_names,
            "heuristic_mean": self._heuristic_mean,
            "heuristic_std": self._heuristic_std,
            "learned_mean": self._learned_mean,
            "learned_std": self._learned_std,
            "model_pickle_b64": blob,
        }
        version = int(
            fetch_all(
                conn,
                "SELECT COALESCE(MAX(version),0)+1 AS v FROM model_artifacts WHERE name='ranker'",
            )[0]["v"]
        )
        with transaction(conn):
            conn.execute("UPDATE model_artifacts SET is_active = 0 WHERE name = 'ranker'")
            conn.execute(
                "INSERT INTO model_artifacts (name, version, payload_json, metrics_json, created_at, is_active) "
                "VALUES ('ranker', ?, ?, ?, ?, 1)",
                (version, json.dumps(payload), json.dumps(self.metrics.to_json()), utcnow()),
            )

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> TasteRanker | None:
        rows = fetch_all(
            conn,
            "SELECT payload_json, metrics_json FROM model_artifacts WHERE name='ranker' AND is_active=1 ORDER BY version DESC LIMIT 1",
        )
        if not rows:
            return None
        payload = json.loads(rows[0]["payload_json"])
        metrics_raw = json.loads(rows[0]["metrics_json"] or "{}")
        ranker = cls(feature_names=payload.get("feature_names", list(FEATURE_NAMES)))
        ranker._heuristic_mean = payload.get("heuristic_mean", 0.0)
        ranker._heuristic_std = payload.get("heuristic_std", 1.0) or 1.0
        ranker._learned_mean = payload.get("learned_mean", 0.0)
        ranker._learned_std = payload.get("learned_std", 1.0) or 1.0
        blob = payload.get("model_pickle_b64") or ""
        if blob:
            import base64
            import pickle

            try:
                ranker._model = pickle.loads(base64.b64decode(blob))
            except Exception as exc:
                log.warning(
                    "could not restore the learned model (%s); falling back to heuristic", exc
                )
                ranker._model = None
        ranker.metrics = RankerMetrics(
            n_train=metrics_raw.get("n_train", 0),
            spearman=metrics_raw.get("spearman", 0.0),
            mae=metrics_raw.get("mae", 0.0),
            ndcg_at_10=metrics_raw.get("ndcg_at_10", 0.0),
            model_kind=metrics_raw.get("model_kind", "heuristic"),
            blend_weight=metrics_raw.get("blend_weight", 0.0) if ranker._model is not None else 0.0,
            top_features=[tuple(t) for t in metrics_raw.get("top_features", [])],
        )
        return ranker
