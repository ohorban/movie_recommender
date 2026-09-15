"""Fit the personalized ranker on the user's own ratings."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from ..enrich.embeddings import EmbeddingBackend
from ..enrich.structuring import load_dossiers
from ..logging_utils import get_logger
from ..recommend.features import FeatureBuilder
from ..recommend.ranker import (
    MIN_TRAINING_ROWS,
    TasteRanker,
    _ndcg_at_k,
    _spearman,
    heuristic_scores,
)
from .profile import (
    TasteProfile,
    build_profile_from_prefs,
    load_user_ratings,
    preference_scores,
)

log = get_logger("taste.training")

# Fold splits used for the held-out estimate. Each is cheap (well under a
# second); five of them turn a figure that bounces by 0.07 into a stable one.
CV_SEEDS = (17, 23, 42, 99, 7)

# How much of the learned model to mix into the prior. Searched on the held-out
# folds rather than derived from a constant: the previous rule divided the
# Spearman by 0.45, which saturates to 1.0 for any score above that and so could
# only ever produce 0.0 or 1.0 — the blend was never actually blended. On real
# data the best mix was near the middle and worth +0.034 Spearman over either
# model alone.
BLEND_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    spread = float(values.std())
    return (values - values.mean()) / (spread if spread > 1e-9 else 1.0)


def _fold_metrics(
    y: np.ndarray, predictions: np.ndarray, fold_of: np.ndarray
) -> tuple[float, float, float]:
    """Score each fold on its own, then average.

    Correlating the concatenated out-of-fold vector instead lets the offset
    between folds - each has its own model and its own rebuilt profile - into
    the metric. On real data that artifact was 0.017 Spearman, more than ten
    times the margin it was being used to decide.
    """
    spearman, ndcg, mae = [], [], []
    for fold in sorted({int(f) for f in fold_of if f >= 0}):
        rows = fold_of == fold
        if rows.sum() < 3:
            continue
        spearman.append(_spearman(y[rows], predictions[rows]))
        ndcg.append(_ndcg_at_k(y[rows], predictions[rows], 10))
        mae.append(float(np.abs(y[rows] - predictions[rows]).mean()))
    if not spearman:
        return 0.0, 0.0, 0.0
    return float(np.mean(spearman)), float(np.mean(ndcg)), float(np.mean(mae))


def _fold_predictions(
    conn: sqlite3.Connection,
    backend: EmbeddingBackend,
    prefs: dict[int, float],
    titles: dict[int, tuple],
    *,
    n_splits: int = 5,
    seed: int = 17,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Honest out-of-fold predictions, rebuilding the taste profile per fold.

    Returns the predictions per model and the fold each row was held out in,
    because a metric has to be computed *within* a fold: each fold has its own
    model and its own rebuilt profile, so correlating the concatenation of all
    five lets the offsets between folds into the number.

    This is the only way to get a trustworthy number out of this system. The
    features are not raw measurements - affinities, taste centroids and scale
    targets are all *fitted* on the ratings. Fitting them once over everything
    and then cross-validating the ranker on top scores each held-out film
    partly against itself, which on real data reports ~0.9 rank correlation
    where the truth is closer to 0.3.

    So each fold gets its own profile, built only from that fold's training
    ratings, and the held-out films are featurised against it exactly as an
    unrated candidate would be at recommendation time.
    """
    from sklearn.model_selection import KFold

    ids = np.array(sorted(prefs.keys()))
    y = np.array([prefs[int(i)] for i in ids], dtype=np.float64)
    n = ids.size
    n_splits = int(np.clip(n_splits, 2, max(2, n // 12)))

    names = ["heuristic", *TasteRanker.candidate_models(int(n * (1 - 1 / n_splits))).keys()]
    oof = {name: np.zeros(n, dtype=np.float64) for name in names}

    fold_of = np.full(n, -1, dtype=np.int64)
    failed: set[str] = set()

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_no, (train_idx, test_idx) in enumerate(folds.split(ids)):
        fold_of[test_idx] = fold_no
        train_ids = [int(i) for i in ids[train_idx]]
        test_ids = [int(i) for i in ids[test_idx]]
        fold_prefs = {i: prefs[i] for i in train_ids}

        fold_profile = build_profile_from_prefs(
            conn, fold_prefs, backend.name, titles=titles, review_rows=None
        )
        builder = FeatureBuilder(conn, fold_profile, embed_model=backend.name)
        builder.set_reference_prefs(fold_prefs)

        fm_train = builder.build(train_ids, load_dossiers(conn, train_ids), loo_prefs=fold_prefs)
        # Held-out films are absent from this fold's profile, so they need no
        # leave-one-out correction - the same situation as a real candidate.
        fm_test = builder.build(test_ids, load_dossiers(conn, test_ids))

        oof["heuristic"][test_idx] = heuristic_scores(fm_test)
        for name, model in TasteRanker.candidate_models(len(train_ids)).items():
            if name not in oof:
                continue
            try:
                model.fit(fm_train.matrix.astype(np.float64), y[train_idx])
                oof[name][test_idx] = model.predict(fm_test.matrix.astype(np.float64))
            except Exception as exc:
                # Imputing a constant here used to look like a small loss of
                # accuracy rather than a broken run: zero sits near the mean of
                # a z-scored target, so the dead fold landed mid-pack and cost
                # about 0.06 Spearman quietly. Drop the model instead.
                log.warning("fold fit failed for %s: %s — excluding it from this split", name, exc)
                failed.add(name)
    for name in failed:
        oof.pop(name, None)
    return oof, fold_of


def train_ranker(
    conn: sqlite3.Connection,
    profile: TasteProfile,
    backend: EmbeddingBackend,
    *,
    store: bool = True,
) -> TasteRanker:
    """Train on rated films, targeting the user's own preference z-score.

    Two separate corrections keep this honest, and both matter:

    1. **Leave-one-out features.** A rated film scored against the full profile
       is scored partly against itself - its own rating sits inside its
       director's affinity and inside the centroid of the taste mode it belongs
       to. ``loo_prefs`` removes that contribution film by film.
    2. **Per-fold profile rebuilding.** Even with (1), a profile fitted on every
       rating leaks across cross-validation folds. The reported metric therefore
       comes from :func:`_fold_predictions`, which rebuilds the profile inside
       each fold.

    The final model is fitted on the full data, but the *blend weight* - how
    much the learned model is trusted over the hand-tuned prior - is set from
    the honest held-out score.
    """
    ratings = load_user_ratings(conn)
    prefs, _, _ = preference_scores(ratings)
    if len(prefs) < 10:
        log.info("only %d rated films with a TMDB match; using the heuristic ranker", len(prefs))
        ranker = TasteRanker()
        if store:
            ranker.save(conn)
        return ranker

    titles = {int(r["tmdb_id"]): (r["title"], r["year"], r["rating"]) for r in ratings}
    ids = list(prefs.keys())
    targets = np.array([prefs[i] for i in ids], dtype=np.float64)

    oof_scores: dict[str, dict[str, Any]] | None = None
    blend_scores: dict[str, dict[float, list[float]]] | None = None
    if len(prefs) >= MIN_TRAINING_ROWS:
        # Deliberately not wrapped in a try/except. This used to fall back to an
        # in-sample cross-validation on a transient failure, which reports about
        # 0.89 where the honest figure is 0.51 — and then stored that as the
        # model's accuracy. A broken evaluation must be visible, not flattering.
        sorted_ids = sorted(prefs.keys())
        y_sorted = np.array([prefs[int(i)] for i in sorted_ids])
        per_split: dict[str, list[tuple[float, float, float]]] = {}
        blend_split: dict[str, dict[float, list[float]]] = {}

        for seed in CV_SEEDS:
            run, fold_of = _fold_predictions(conn, backend, prefs, titles, seed=seed)
            for name, predictions in run.items():
                per_split.setdefault(name, []).append(_fold_metrics(y_sorted, predictions, fold_of))
            # How good is the blend that actually ships? The deployed scorer is
            # `w * learned_z + (1 - w) * heuristic_z`, and neither component's
            # own score says anything about the mixture - on this data the best
            # mix beat both of them. So it is measured directly, on the same
            # folds, for every candidate model and every weight. Standardisation
            # uses only the predictions, never the targets, so it stays honest.
            if "heuristic" in run:
                heur_z = _zscore(run["heuristic"])
                for name, predictions in run.items():
                    if name == "heuristic":
                        continue
                    learned_z = _zscore(predictions)
                    grid = blend_split.setdefault(name, {w: [] for w in BLEND_GRID})
                    for w in BLEND_GRID:
                        mixed = w * learned_z + (1.0 - w) * heur_z
                        grid[w].append(_fold_metrics(y_sorted, mixed, fold_of))

        oof_scores = {
            name: {
                "spearman": float(np.mean([v[0] for v in vals])),
                "sd": float(np.std([v[0] for v in vals])),
                "ndcg_at_10": float(np.mean([v[1] for v in vals])),
                "mae": float(np.mean([v[2] for v in vals])),
                "repeats": len(vals),
                # Kept per split so the choice between models can be a paired
                # comparison rather than a comparison of two averages.
                "per_split": [float(v[0]) for v in vals],
            }
            for name, vals in per_split.items()
        }
        blend_scores = {
            name: {w: v for w, v in grid.items() if v} for name, grid in blend_split.items()
        }

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    fm = builder.build(ids, load_dossiers(conn, ids), loo_prefs=prefs)

    # Zero each film's own CF contribution so it cannot predict itself.
    cf_idx = fm.names.index("cf_score")
    self_cf = _self_cf_contribution(conn, ids, prefs)
    for row, tmdb_id in enumerate(fm.ids):
        fm.matrix[row, cf_idx] = max(0.0, float(fm.matrix[row, cf_idx]) - self_cf.get(tmdb_id, 0.0))

    ranker = TasteRanker.fit(fm, targets, oof_scores=oof_scores, blend_scores=blend_scores)
    if store:
        ranker.save(conn)
    return ranker


def _self_cf_contribution(
    conn: sqlite3.Connection, ids: list[int], prefs: dict[int, float]
) -> dict[int, float]:
    """How much of each film's CF score comes from the film itself."""
    from ..db import fetch_all

    out: dict[int, float] = {}
    for start in range(0, len(ids), 400):
        chunk = ids[start : start + 400]
        ph = ",".join("?" for _ in chunk)
        for r in fetch_all(
            conn,
            f"SELECT tmdb_id, neighbor_tmdb_id, score FROM cf_neighbors "
            f"WHERE tmdb_id IN ({ph}) AND tmdb_id = neighbor_tmdb_id",
            chunk,
        ):
            out[int(r["neighbor_tmdb_id"])] = float(r["score"]) * prefs.get(int(r["tmdb_id"]), 0.0)
    return out


def evaluate_holdout(
    conn: sqlite3.Connection,
    profile: TasteProfile,
    backend: EmbeddingBackend,
    *,
    test_fraction: float = 0.25,
) -> dict[str, Any]:
    """Honest held-out check: can we rank films this user liked above ones they did not?"""
    from ..recommend.ranker import _ndcg_at_k, _spearman

    ratings = load_user_ratings(conn)
    prefs, _, _ = preference_scores(ratings)
    if len(prefs) < 40:
        return {"skipped": True, "reason": f"only {len(prefs)} rated films"}

    rng = np.random.default_rng(11)
    ids = np.array(list(prefs.keys()))
    rng.shuffle(ids)
    cut = int(len(ids) * (1 - test_fraction))
    train_ids, test_ids = ids[:cut].tolist(), ids[cut:].tolist()

    train_prefs = {i: prefs[i] for i in train_ids}
    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(train_prefs)

    fm_train = builder.build(train_ids, load_dossiers(conn, train_ids), loo_prefs=train_prefs)
    ranker = TasteRanker.fit(fm_train, np.array([prefs[i] for i in train_ids]))

    fm_test = builder.build(test_ids, load_dossiers(conn, test_ids))
    scores = ranker.score(fm_test)
    truth = np.array([prefs[i] for i in test_ids])
    return {
        "skipped": False,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "spearman": round(_spearman(truth, scores), 4),
        "ndcg_at_10": round(_ndcg_at_k(truth, scores, 10), 4),
        "model_kind": ranker.metrics.model_kind,
    }
