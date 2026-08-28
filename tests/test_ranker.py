"""The ranker must learn real signal, and must refuse to trust noise."""

from __future__ import annotations

import numpy as np
import pytest

from movierec.recommend.features import FEATURE_NAMES, FeatureMatrix
from movierec.recommend.ranker import (
    MIN_TRAINING_ROWS,
    TasteRanker,
    _ndcg_at_k,
    _spearman,
    heuristic_scores,
)


def make_fm(n: int, seed: int = 3) -> tuple[FeatureMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    y = 1.4 * X[:, 0] + 0.8 * X[:, 4] - 0.6 * X[:, 3] + rng.normal(scale=0.6, size=n)
    return FeatureMatrix(list(range(n)), X, list(FEATURE_NAMES)), y


def test_spearman_edges():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, -a) == pytest.approx(-1.0)
    assert _spearman(a, np.ones(4)) == 0.0, "constant scores correlate with nothing"
    assert _spearman(
        np.array([1.0, 1.0, 2.0, 2.0]), np.array([1.0, 1.0, 2.0, 2.0])
    ) == pytest.approx(1.0)
    assert _spearman(np.array([1.0]), np.array([1.0])) == 0.0


def test_ndcg_bounds():
    y = np.linspace(5.0, 1.0, 40)
    assert _ndcg_at_k(y, y, 10) == pytest.approx(1.0)
    assert _ndcg_at_k(y, -y, 10) < _ndcg_at_k(y, y, 10) / 2
    assert _ndcg_at_k(np.zeros(5), np.arange(5.0), 5) == 0.0, "no gain to distribute"


def test_learns_a_real_signal():
    fm, y = make_fm(160)
    ranker = TasteRanker.fit(fm, y)
    assert ranker.metrics.spearman > 0.7
    assert ranker.metrics.blend_weight > 0.9
    assert _spearman(y, ranker.score(fm)) > 0.7


def test_identifies_the_predictive_features():
    fm, y = make_fm(200)
    ranker = TasteRanker.fit(fm, y)
    top = {name for name, _ in ranker.metrics.top_features[:4]}
    assert {"sim_mode_best", "aff_genre"} <= top


def test_refuses_to_trust_noise():
    fm, _ = make_fm(160)
    noise = np.random.default_rng(9).normal(size=160)
    ranker = TasteRanker.fit(fm, noise)
    assert ranker.metrics.blend_weight < 0.3, "a model that learned nothing must not be trusted"


def test_falls_back_to_heuristic_on_small_data():
    fm, y = make_fm(MIN_TRAINING_ROWS - 5)
    ranker = TasteRanker.fit(fm, y)
    assert ranker.metrics.model_kind == "heuristic"
    assert ranker.metrics.blend_weight == 0.0
    assert ranker.score(fm).shape == (MIN_TRAINING_ROWS - 5,)


def test_heuristic_scores_are_finite_and_ordered():
    fm, _ = make_fm(40)
    scores = heuristic_scores(fm)
    assert np.isfinite(scores).all()
    assert scores.std() > 0


def test_save_and_load_roundtrip(conn):
    fm, y = make_fm(160)
    original = TasteRanker.fit(fm, y)
    original.save(conn)

    restored = TasteRanker.load(conn)
    assert restored is not None
    assert restored.metrics.model_kind == original.metrics.model_kind
    assert np.allclose(restored.score(fm), original.score(fm), atol=1e-5)


def test_load_returns_none_without_an_artifact(conn):
    assert TasteRanker.load(conn) is None


def test_only_one_active_artifact(conn):
    from movierec.db import fetch_all

    fm, y = make_fm(160)
    for _ in range(3):
        TasteRanker.fit(fm, y).save(conn)
    active = fetch_all(
        conn, "SELECT COUNT(*) c FROM model_artifacts WHERE name='ranker' AND is_active=1"
    )
    assert active[0]["c"] == 1
    versions = fetch_all(conn, "SELECT COUNT(*) c FROM model_artifacts WHERE name='ranker'")
    assert versions[0]["c"] == 3, "old versions are kept for auditability"


# --------------------------------------------------------------------------- #
# Target leakage
#
# Every taste feature is derived from the ratings the model is trying to
# predict. Without leave-one-out construction the model reads the label back
# out of its own features and reports near-perfect accuracy while recommending
# badly. These tests pin that down.
# --------------------------------------------------------------------------- #
def test_affinity_leaks_the_label_without_leave_one_out():
    """A facet value seen exactly once is a pure function of that film's rating."""
    from movierec.taste.profile import affinity_value, compute_affinity_stats

    prefs = {1: 2.0, 2: -1.5, 3: 0.5}
    facets = {i: {"director": [f"Director {i}"]} for i in prefs}  # each seen once
    stats = compute_affinity_stats(prefs, facets)

    leaked = [affinity_value(stats, "director", f"Director {i}") for i in prefs]
    assert _spearman(np.array(list(prefs.values())), np.array(leaked)) == pytest.approx(1.0), (
        "without LOO the affinity ranks the films exactly by their own rating"
    )

    clean = [
        affinity_value(stats, "director", f"Director {i}", exclude_pref=p) for i, p in prefs.items()
    ]
    assert clean == [0.0, 0.0, 0.0], "with LOO a one-off facet carries no information"


def test_leave_one_out_shrinks_a_shared_facet_correctly():
    from movierec.taste.profile import affinity_value, compute_affinity_stats

    prefs = {1: 2.0, 2: 0.0, 3: 1.0}
    facets = {i: {"genre": ["Drama"]} for i in prefs}
    stats = compute_affinity_stats(prefs, facets)

    # Excluding film 1 leaves mean(0.0, 1.0) = 0.5, shrunk by 2 / (2 + k_genre).
    expected = 0.5 * (2 / (2 + 4.0))
    assert affinity_value(stats, "genre", "Drama", exclude_pref=2.0) == pytest.approx(expected)
    # The full affinity is higher because film 1's own high rating is included.
    assert affinity_value(stats, "genre", "Drama") > expected
