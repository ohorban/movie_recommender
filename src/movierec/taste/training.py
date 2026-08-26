"""Fit the personalized ranker on the user's own ratings."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from ..enrich.embeddings import EmbeddingBackend
from ..enrich.structuring import load_dossiers
from ..logging_utils import get_logger
from ..recommend.features import FeatureBuilder
from ..recommend.ranker import TasteRanker
from .profile import TasteProfile, load_user_ratings, preference_scores

log = get_logger("taste.training")


def train_ranker(
    conn: sqlite3.Connection,
    profile: TasteProfile,
    backend: EmbeddingBackend,
    *,
    store: bool = True,
) -> TasteRanker:
    """Train on rated films, targeting the user's own preference z-score.

    Training features are built with the *same* builder used at inference, so a
    feature that is unavailable at recommendation time is unavailable here too.
    The one asymmetry we must avoid is leakage: a film cannot be allowed to
    contribute to the taste centroids it is then scored against, so the CF
    reference set excludes the film being scored.
    """
    ratings = load_user_ratings(conn)
    prefs, _, _ = preference_scores(ratings)
    if len(prefs) < 10:
        log.info("only %d rated films with a TMDB match; using the heuristic ranker", len(prefs))
        ranker = TasteRanker()
        if store:
            ranker.save(conn)
        return ranker

    ids = list(prefs.keys())
    targets = np.array([prefs[i] for i in ids], dtype=np.float64)

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    fm = builder.build(ids, load_dossiers(conn, ids))

    # Zero each film's own CF contribution so it cannot predict itself.
    cf_idx = fm.names.index("cf_score")
    self_cf = _self_cf_contribution(conn, ids, prefs)
    for row, tmdb_id in enumerate(fm.ids):
        fm.matrix[row, cf_idx] = max(0.0, float(fm.matrix[row, cf_idx]) - self_cf.get(tmdb_id, 0.0))

    ranker = TasteRanker.fit(fm, targets)
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

    fm_train = builder.build(train_ids, load_dossiers(conn, train_ids))
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
