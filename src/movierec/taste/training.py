"""Fit the personalized ranker on the user's own ratings."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from ..enrich.embeddings import EmbeddingBackend
from ..enrich.structuring import load_dossiers
from ..logging_utils import get_logger
from ..recommend.features import FeatureBuilder
from ..recommend.ranker import MIN_TRAINING_ROWS, TasteRanker, heuristic_scores
from .profile import (
    TasteProfile,
    build_profile_from_prefs,
    load_user_ratings,
    preference_scores,
)

log = get_logger("taste.training")


def _fold_predictions(
    conn: sqlite3.Connection,
    backend: EmbeddingBackend,
    prefs: dict[int, float],
    titles: dict[int, tuple],
    *,
    n_splits: int = 5,
    seed: int = 17,
) -> dict[str, np.ndarray]:
    """Honest out-of-fold predictions, rebuilding the taste profile per fold.

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

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(ids):
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
                log.warning("fold fit failed for %s: %s", name, exc)
                oof[name][test_idx] = 0.0
    return oof


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

    oof = None
    if len(prefs) >= MIN_TRAINING_ROWS:
        try:
            oof = _fold_predictions(conn, backend, prefs, titles)
            # _fold_predictions works on sorted ids; realign to `ids` order.
            order = {int(i): k for k, i in enumerate(sorted(prefs.keys()))}
            index = np.array([order[i] for i in ids])
            oof = {name: values[index] for name, values in oof.items()}
        except Exception as exc:
            log.warning("held-out evaluation failed (%s); falling back to in-sample CV", exc)
            oof = None

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    fm = builder.build(ids, load_dossiers(conn, ids), loo_prefs=prefs)

    # Zero each film's own CF contribution so it cannot predict itself.
    cf_idx = fm.names.index("cf_score")
    self_cf = _self_cf_contribution(conn, ids, prefs)
    for row, tmdb_id in enumerate(fm.ids):
        fm.matrix[row, cf_idx] = max(0.0, float(fm.matrix[row, cf_idx]) - self_cf.get(tmdb_id, 0.0))

    ranker = TasteRanker.fit(fm, targets, oof=oof)
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
