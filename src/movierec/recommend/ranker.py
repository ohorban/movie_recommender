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
from .features import FEATURE_NAMES, FEATURE_VERSION, HEURISTIC_WEIGHTS, FeatureMatrix

log = get_logger("recommend.ranker")

MIN_TRAINING_ROWS = 25
# Only used when no held-out blend search is available (a very small history).
CONFIDENCE_FULL_TRUST = 0.45  # Spearman at which the learned model is trusted outright

# A learned model replaces the hand-tuned prior only if it wins by more than
# this many standard errors of the paired per-split difference. With ~160
# ratings the two are usually within noise of each other: ridge once beat the
# prior by 0.0015 Spearman (paired p = 0.91) and, because the old rule divided
# by CONFIDENCE_FULL_TRUST and clipped, that coin flip did not merely tilt the
# blend — it set the weight to 1.0 and switched the prior off entirely.
SELECTION_MARGIN_SE = 1.0

# And it has to be worth something in absolute terms. Beating the prior is not
# the same as being any good: on random ratings the prior scores about zero, so
# a model that merely clears it can still be ranking noise. This is the floor
# below which nothing learned is trusted, whatever the comparison says.
MIN_LEARNED_SKILL = 0.20

# Half a point of Spearman. Differences smaller than this across the blend grid
# are not worth trading away the prior for.
BLEND_TOLERANCE = 0.005


def _select_model(eligible: dict[str, dict[str, Any]]) -> str:
    """Pick a model, keeping the prior unless a learned one clearly beats it."""
    baseline = eligible.get("heuristic")
    learned = {n: v for n, v in eligible.items() if n != "heuristic"}
    if not learned:
        return "heuristic"
    best = max(learned, key=lambda n: learned[n]["spearman"])
    if learned[best]["spearman"] < MIN_LEARNED_SKILL:
        log.info(
            "%s scored %.3f held out, below the %.2f needed to be trusted at all",
            best,
            learned[best]["spearman"],
            MIN_LEARNED_SKILL,
        )
        return "heuristic"
    if baseline is None:
        return best

    theirs = np.asarray(learned[best].get("per_split") or [], dtype=np.float64)
    ours = np.asarray(baseline.get("per_split") or [], dtype=np.float64)
    if theirs.size != ours.size or theirs.size < 2:
        # No paired samples to compare; fall back to requiring a visible gap.
        return best if learned[best]["spearman"] > baseline["spearman"] + 0.02 else "heuristic"

    diff = theirs - ours
    stderr = float(diff.std(ddof=1)) / np.sqrt(diff.size)
    if stderr <= 1e-9:
        return best if diff.mean() > 0 else "heuristic"
    if diff.mean() > SELECTION_MARGIN_SE * stderr:
        return best
    log.info(
        "%s beat the prior by %.4f (%.2f standard errors) — not enough to replace it",
        best,
        float(diff.mean()),
        float(diff.mean() / stderr),
    )
    return "heuristic"


def _better_by_a_margin(theirs: np.ndarray, ours: np.ndarray) -> bool:
    """Is the first sample better than the second by more than sampling noise?

    A paired comparison across splits, because the two are scored on exactly the
    same folds. Comparing two averages instead let a 0.0015 difference - paired
    p = 0.91 - decide which model shipped.
    """
    if theirs.size != ours.size or theirs.size < 2:
        return bool(theirs.mean() > ours.mean() + 0.02)
    diff = theirs - ours
    stderr = float(diff.std(ddof=1)) / np.sqrt(diff.size)
    if stderr <= 1e-9:
        return bool(diff.mean() > 0)
    return bool(diff.mean() > SELECTION_MARGIN_SE * stderr)


def _best_blend(
    blend_scores: dict[str, dict[float, list[float]]], eligible: dict[str, dict[str, Any]]
) -> tuple[str, float, float]:
    """The scorer that actually ranked best on the held-out folds.

    Searches over both *which* learned model and *how much* of it to mix into
    the prior, because neither question answers the other: on real data ridge
    lost to the prior on its own while a mixture of the two beat both. Weight
    zero is in the grid, so "use the prior alone" competes on the same footing.

    The smallest weight within :data:`BLEND_TOLERANCE` of the best score wins,
    not the argmax. The peak is flat - on real data w=0.2 and w=0.4 differ by
    0.0007 - and taking the maximum of eleven correlated estimates flatters
    itself, so the tie-break leans toward the hand-tuned prior.
    """
    best = ("heuristic", 0.0, float("-inf"))
    baseline: np.ndarray | None = None

    for name, grid in (blend_scores or {}).items():
        if name not in eligible:
            continue
        means = {float(w): float(np.mean([m[0] for m in v])) for w, v in grid.items() if v}
        if not means:
            continue
        if baseline is None and 0.0 in grid:
            baseline = np.asarray([m[0] for m in grid[0.0]], dtype=np.float64)
        if eligible[name]["spearman"] < MIN_LEARNED_SKILL:
            log.info(
                "%s scored %.3f held out, below the %.2f needed to be trusted at all",
                name,
                eligible[name]["spearman"],
                MIN_LEARNED_SKILL,
            )
            continue
        ceiling = max(means.values())
        for w in sorted(means):
            if w > 0.0 and means[w] >= ceiling - BLEND_TOLERANCE:
                if means[w] > best[2] or best[0] == "heuristic":
                    best = (name, w, means[w])
                break

    if best[1] == 0.0 or baseline is None:
        return "heuristic", 0.0, float(np.mean(baseline)) if baseline is not None else 0.0

    name, weight, score = best
    candidate = np.asarray([m[0] for m in blend_scores[name][weight]], dtype=np.float64)
    # Two hurdles, and both matter: the gain has to be big enough to be worth
    # having at all, and consistent enough across splits not to be noise.
    clears_floor = score > float(baseline.mean()) + BLEND_TOLERANCE
    if not clears_floor or not _better_by_a_margin(candidate, baseline):
        log.info(
            "the best mixture (%s at %.1f, %.3f) did not beat the prior alone (%.3f) by enough",
            name,
            weight,
            score,
            float(baseline.mean()),
        )
        return "heuristic", 0.0, float(baseline.mean())
    return name, weight, score


@dataclass
class RankerMetrics:
    n_train: int = 0
    spearman: float = 0.0
    mae: float = 0.0
    ndcg_at_10: float = 0.0
    model_kind: str = "heuristic"
    blend_weight: float = 0.0
    # Spread of the held-out score across repeated fold splits. On ~160 ratings
    # a single split swings by roughly 0.07, so the headline figure means
    # little without it.
    spearman_sd: float = 0.0
    cv_repeats: int = 1
    # Held-out score of the blended scorer that actually ships, when one was
    # searched for. `spearman` is set to this too, so the headline figure always
    # describes the thing being deployed rather than one component of it.
    blend_spearman: float = 0.0
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "spearman": round(self.spearman, 4),
            "mae": round(self.mae, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "model_kind": self.model_kind,
            "blend_weight": round(self.blend_weight, 4),
            "spearman_sd": round(self.spearman_sd, 4),
            "cv_repeats": self.cv_repeats,
            "blend_spearman": round(self.blend_spearman, 4),
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
    @staticmethod
    def candidate_models(n_rows: int) -> dict[str, Any]:
        """The models considered, given how much training data there is."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        candidates: dict[str, Any] = {
            "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 25))),
        }
        if n_rows >= 80:
            candidates["gbdt"] = GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                min_samples_leaf=8,
                random_state=17,
            )
        return candidates

    @classmethod
    def fit(
        cls,
        fm: FeatureMatrix,
        targets: np.ndarray,
        *,
        oof: dict[str, np.ndarray] | None = None,
        oof_scores: dict[str, dict[str, Any]] | None = None,
        blend_scores: dict[str, dict[float, list[float]]] | None = None,
    ) -> TasteRanker:
        """Fit the ranker.

        ``oof`` supplies externally computed out-of-fold predictions per model.
        Pass it whenever the features are themselves fitted quantities - see
        :func:`movierec.taste.training.train_ranker` - because internal
        cross-validation cannot see that leakage and will report a score that is
        far too optimistic.
        """
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

        best_name, best_score, best_oof = "heuristic", _spearman(y, heur), heur

        if oof_scores:
            # Repeated cross-validation: each entry is already the mean of the
            # per-fold metrics. Averaging the *metrics* estimates the model we
            # actually deploy; averaging the predictions instead would measure
            # an ensemble of fold models and read several points too high —
            # enough, on this data, to pick a learned model that does not in
            # fact beat the prior.
            eligible = {n: v for n, v in oof_scores.items() if n == "heuristic" or n in candidates}
            for name, value in sorted(eligible.items(), key=lambda t: -t[1]["spearman"]):
                log.info(
                    "held-out %-9s spearman=%.3f (sd %.3f)",
                    name,
                    value["spearman"],
                    value.get("sd", 0.0),
                )
            if blend_scores:
                best_name, chosen_weight, chosen_score = _best_blend(blend_scores, eligible)
            else:
                best_name, chosen_weight, chosen_score = _select_model(eligible), 0.0, 0.0
            chosen = eligible[best_name]
            ranker.metrics = RankerMetrics(
                n_train=n,
                spearman=chosen["spearman"],
                mae=chosen.get("mae", 0.0),
                ndcg_at_10=chosen.get("ndcg_at_10", 0.0),
                model_kind=best_name,
                spearman_sd=chosen.get("sd", 0.0),
                cv_repeats=int(chosen.get("repeats", 1)),
            )
        elif oof is not None:
            # Prefer the held-out heuristic score as the baseline: comparing a
            # learned model's out-of-fold score against an in-sample baseline
            # would tilt the choice toward the learned model for free.
            if "heuristic" in oof:
                best_oof = np.asarray(oof["heuristic"], dtype=np.float64)
                best_score = _spearman(y, best_oof)
            for name, predictions in oof.items():
                score = _spearman(y, np.asarray(predictions, dtype=np.float64))
                log.info("held-out %-9s spearman=%.3f", name, score)
                if name in candidates and score > best_score:
                    best_name, best_score, best_oof = (
                        name,
                        score,
                        np.asarray(predictions, dtype=np.float64),
                    )
        else:
            folds = KFold(n_splits=min(5, max(3, n // 20)), shuffle=True, random_state=17)
            for name, model in candidates.items():
                fold_oof = np.zeros(n)
                try:
                    for train_idx, test_idx in folds.split(X):
                        import copy

                        fold_model = copy.deepcopy(model)
                        fold_model.fit(X[train_idx], y[train_idx])
                        fold_oof[test_idx] = fold_model.predict(X[test_idx])
                except Exception as exc:
                    log.warning("cross-validation failed for %s: %s", name, exc)
                    continue
                score = _spearman(y, fold_oof)
                log.info("cv %-9s spearman=%.3f", name, score)
                if score > best_score:
                    best_name, best_score, best_oof = name, score, fold_oof

        if not oof_scores:
            ranker.metrics = RankerMetrics(
                n_train=n,
                spearman=best_score,
                mae=float(np.abs(y - best_oof).mean())
                if best_name != "heuristic"
                else float(np.abs(y - y.mean()).mean()),
                ndcg_at_10=_ndcg_at_k(y, best_oof, 10),
                model_kind=best_name,
            )
        best_score = ranker.metrics.spearman

        if best_name != "heuristic":
            model = candidates[best_name]
            model.fit(X, y)
            ranker._model = model
            preds = model.predict(X)
            ranker._learned_mean = float(preds.mean())
            ranker._learned_std = float(preds.std()) or 1.0
            ranker.metrics.top_features = ranker._explain(model, X, y)
        else:
            # Nothing learned beat the prior, so report the prior's own weights
            # rather than leaving the diagnostics blank.
            ranker.metrics.top_features = sorted(
                ((n, HEURISTIC_WEIGHTS.get(n, 0.0)) for n in fm.names),
                key=lambda t: -abs(t[1]),
            )[:10]

        # How much to trust the learned model: measured, not assumed.
        if best_name == "heuristic":
            ranker.metrics.blend_weight = 0.0
        elif blend_scores:
            ranker.metrics.blend_weight = chosen_weight
            # Report the scorer that actually ships. Quoting the learned model's
            # solo score while deploying a mixture described something nobody ran.
            ranker.metrics.spearman = chosen_score
            ranker.metrics.blend_spearman = chosen_score
            # ndcg and mae have to describe the mixture too, not the component.
            mixed = blend_scores[best_name][chosen_weight]
            ranker.metrics.ndcg_at_10 = float(np.mean([m[1] for m in mixed]))
            ranker.metrics.mae = float(np.mean([m[2] for m in mixed]))
            ranker.metrics.spearman_sd = float(np.std([m[0] for m in mixed]))
            log.info(
                "blend: %.0f%% %s + %.0f%% prior, held-out spearman %.3f",
                100 * chosen_weight,
                best_name,
                100 * (1 - chosen_weight),
                chosen_score,
            )
        else:
            ranker.metrics.blend_weight = float(
                np.clip(best_score / CONFIDENCE_FULL_TRUST, 0.0, 1.0)
            )
        log.info(
            "ranker: %s, spearman=%.3f, ndcg@10=%.3f, blend=%.2f",
            ranker.metrics.model_kind,
            ranker.metrics.spearman,
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
            "feature_version": FEATURE_VERSION,
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
        stale = payload.get("feature_version", 1) != FEATURE_VERSION or payload.get(
            "feature_names", list(FEATURE_NAMES)
        ) != list(FEATURE_NAMES)

        blob = payload.get("model_pickle_b64") or ""
        if blob and not stale:
            import base64
            import pickle

            try:
                ranker._model = pickle.loads(base64.b64decode(blob))
            except Exception as exc:
                log.warning(
                    "could not restore the learned model (%s); falling back to heuristic", exc
                )
                ranker._model = None
        elif blob and stale:
            log.info(
                "stored ranker was fitted on a different feature set; using the prior "
                "until the next training run"
            )

        ranker.metrics = RankerMetrics(
            n_train=metrics_raw.get("n_train", 0),
            spearman=metrics_raw.get("spearman", 0.0),
            mae=metrics_raw.get("mae", 0.0),
            ndcg_at_10=metrics_raw.get("ndcg_at_10", 0.0),
            model_kind=metrics_raw.get("model_kind", "heuristic"),
            spearman_sd=metrics_raw.get("spearman_sd", 0.0),
            cv_repeats=metrics_raw.get("cv_repeats", 1),
            blend_spearman=metrics_raw.get("blend_spearman", 0.0),
            blend_weight=metrics_raw.get("blend_weight", 0.0) if ranker._model is not None else 0.0,
            top_features=[tuple(t) for t in metrics_raw.get("top_features", [])],
        )
        if ranker._model is None and ranker.metrics.model_kind != "heuristic":
            # The artifact claims a learned model but the heuristic is doing all
            # the ranking. Reporting "ridge, spearman 0.51" here described a
            # model that was not running.
            log.warning(
                "the stored %s model is not usable; reporting the prior that is actually serving",
                ranker.metrics.model_kind,
            )
            ranker.metrics.model_kind = "heuristic"
            ranker.metrics.spearman = 0.0
            ranker.metrics.spearman_sd = 0.0
            ranker.metrics.ndcg_at_10 = 0.0
            ranker.metrics.mae = 0.0
            ranker.metrics.blend_spearman = 0.0
            ranker.metrics.top_features = sorted(
                ((name, HEURISTIC_WEIGHTS.get(name, 0.0)) for name in ranker.feature_names),
                key=lambda t: -abs(t[1]),
            )[:10]
        return ranker
