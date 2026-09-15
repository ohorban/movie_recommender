"""The ranker must learn real signal, and must refuse to trust noise."""

from __future__ import annotations

import json

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


# --------------------------------------------------------------------------- #
# Repeated cross-validation
#
# A single fold split of ~160 ratings gives a score that moves by about 0.07
# run to run on identical data. The fix is to repeat the split and average —
# but average the *metrics*, not the predictions. Averaging predictions scores
# an ensemble of fold models rather than the single model deployed, which on
# real data read high enough to pick a learned model that loses to the prior.
# --------------------------------------------------------------------------- #
def test_averaged_scores_drive_model_selection():
    fm, y = make_fm(160)
    scores = {
        "heuristic": {"spearman": 0.52, "sd": 0.02, "ndcg_at_10": 0.80, "mae": 0.7, "repeats": 5},
        "ridge": {"spearman": 0.48, "sd": 0.02, "ndcg_at_10": 0.77, "mae": 0.7, "repeats": 5},
    }
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind == "heuristic", "the prior won on held-out data"
    assert ranker.metrics.spearman == pytest.approx(0.52)
    assert ranker.metrics.spearman_sd == pytest.approx(0.02)
    assert ranker.metrics.cv_repeats == 5
    assert ranker.metrics.blend_weight == 0.0, "a losing learned model must be ignored"


def test_a_learned_model_that_wins_is_used():
    fm, y = make_fm(160)
    scores = {
        "heuristic": {"spearman": 0.20, "sd": 0.03, "ndcg_at_10": 0.60, "mae": 0.9, "repeats": 5},
        "ridge": {"spearman": 0.46, "sd": 0.02, "ndcg_at_10": 0.78, "mae": 0.7, "repeats": 5},
    }
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind == "ridge"
    assert ranker.metrics.blend_weight > 0.9
    assert ranker.metrics.top_features, "a fitted model should explain itself"


def test_unavailable_models_are_not_selected():
    """gbdt needs 80+ rows; a score for it must not win below that."""
    fm, y = make_fm(40)
    scores = {
        "heuristic": {"spearman": 0.30, "sd": 0.02, "ndcg_at_10": 0.7, "mae": 0.8, "repeats": 5},
        "gbdt": {"spearman": 0.90, "sd": 0.01, "ndcg_at_10": 0.95, "mae": 0.3, "repeats": 5},
    }
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind != "gbdt"


def test_the_spread_survives_a_save_and_load(conn):
    fm, y = make_fm(160)
    scores = {
        "heuristic": {"spearman": 0.5, "sd": 0.031, "ndcg_at_10": 0.8, "mae": 0.7, "repeats": 5}
    }
    TasteRanker.fit(fm, y, oof_scores=scores).save(conn)
    restored = TasteRanker.load(conn)
    assert restored is not None
    assert restored.metrics.spearman_sd == pytest.approx(0.031)
    assert restored.metrics.cv_repeats == 5


# --------------------------------------------------------------------------- #
# Choosing what to deploy
# --------------------------------------------------------------------------- #
# With ~160 ratings the learned models and the hand-tuned prior are usually
# within noise of each other. Ridge once won by 0.0015 Spearman (paired
# p = 0.91) and, because the weight was derived by dividing that score by a
# constant and clipping, the coin flip did not tilt the blend — it set the
# weight to 1.0 and switched the prior off entirely.
def _split_scores(name_to_splits, **extra):
    return {
        name: {
            "spearman": float(np.mean(splits)),
            "sd": float(np.std(splits)),
            "ndcg_at_10": 0.8,
            "mae": 0.7,
            "repeats": len(splits),
            "per_split": list(splits),
            **extra,
        }
        for name, splits in name_to_splits.items()
    }


def test_a_learned_model_that_wins_by_noise_does_not_replace_the_prior():
    fm, y = make_fm(160)
    # Ridge is ahead on average but loses on two of the five splits: the paired
    # difference is well inside its own standard error.
    scores = _split_scores(
        {
            "heuristic": [0.521, 0.515, 0.525, 0.502, 0.483],
            "ridge": [0.522, 0.476, 0.560, 0.513, 0.484],
        }
    )
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind == "heuristic"
    assert ranker.metrics.blend_weight == 0.0


def test_a_consistent_winner_still_replaces_the_prior():
    """The guard must not be so strict that nothing can ever be learned."""
    fm, y = make_fm(160)
    scores = _split_scores(
        {
            "heuristic": [0.30, 0.31, 0.29, 0.30, 0.32],
            "ridge": [0.48, 0.49, 0.47, 0.50, 0.48],
        }
    )
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind == "ridge"


def test_beating_a_useless_prior_is_not_evidence_of_skill():
    """On random ratings the prior scores about zero, so clearing it proves
    nothing. A learned model has to be worth something in absolute terms."""
    fm, y = make_fm(160)
    scores = _split_scores(
        {
            "heuristic": [-0.06, -0.04, -0.07, -0.05, -0.05],
            "ridge": [0.13, 0.12, 0.14, 0.13, 0.13],
        }
    )
    ranker = TasteRanker.fit(fm, y, oof_scores=scores)
    assert ranker.metrics.model_kind == "heuristic"
    assert ranker.metrics.blend_weight == 0.0


# --------------------------------------------------------------------------- #
# The blend weight
# --------------------------------------------------------------------------- #
def _blend_grid(by_weight):
    return {"ridge": {w: [(s, 0.8, 0.7)] * 5 for w, s in by_weight.items()}}


def test_the_blend_weight_comes_from_the_held_out_search():
    """Neither component's own score says anything about the mixture.

    On real data ridge lost to the prior alone and a mix of the two beat both,
    which the old rule — divide the winner's Spearman by 0.45 and clip — could
    not express: above 0.45 everything saturated, so the only reachable weights
    were 0.0 and 1.0 and the blend was never actually blended.
    """
    fm, y = make_fm(160)
    scores = _split_scores(
        {"heuristic": [0.548] * 5, "ridge": [0.511] * 5},
    )
    grid = _blend_grid({0.0: 0.548, 0.2: 0.585, 0.4: 0.580, 0.6: 0.560, 0.8: 0.535, 1.0: 0.511})
    ranker = TasteRanker.fit(fm, y, oof_scores=scores, blend_scores=grid)
    assert ranker.metrics.model_kind == "ridge"
    assert ranker.metrics.blend_weight == pytest.approx(0.2)
    assert ranker.metrics.spearman == pytest.approx(0.585), (
        "the reported score must describe the mixture that ships, not one component"
    )
    assert ranker.metrics.blend_spearman == pytest.approx(0.585)


def test_a_flat_peak_keeps_more_of_the_prior():
    """Taking the argmax of eleven correlated estimates flatters itself."""
    fm, y = make_fm(160)
    scores = _split_scores({"heuristic": [0.548] * 5, "ridge": [0.511] * 5})
    grid = _blend_grid({0.0: 0.548, 0.2: 0.5600, 0.4: 0.5607, 0.6: 0.552, 1.0: 0.511})
    ranker = TasteRanker.fit(fm, y, oof_scores=scores, blend_scores=grid)
    assert ranker.metrics.blend_weight == pytest.approx(0.2), (
        "0.0007 is not a reason to trade away the prior"
    )


def test_a_mixture_that_does_not_beat_the_prior_is_refused():
    fm, y = make_fm(160)
    scores = _split_scores({"heuristic": [0.548] * 5, "ridge": [0.511] * 5})
    grid = _blend_grid({0.0: 0.548, 0.2: 0.5495, 0.5: 0.545, 1.0: 0.511})
    ranker = TasteRanker.fit(fm, y, oof_scores=scores, blend_scores=grid)
    assert ranker.metrics.model_kind == "heuristic"
    assert ranker.metrics.blend_weight == 0.0


# --------------------------------------------------------------------------- #
# Reporting what is actually running
# --------------------------------------------------------------------------- #
def test_a_model_fitted_on_a_different_feature_set_is_refused(conn):
    """Stale coefficients must never be applied to redefined features.

    `scale_fit` changed meaning without changing name, so matching on names
    alone would have kept applying weights fitted for the old definition.
    """
    from movierec.db import fetch_all

    fm, y = make_fm(160)
    scores = _split_scores({"heuristic": [0.2] * 5, "ridge": [0.6] * 5})
    TasteRanker.fit(fm, y, oof_scores=scores).save(conn)
    assert TasteRanker.load(conn)._model is not None, "the fixture needs a learned model"

    row = fetch_all(conn, "SELECT payload_json FROM model_artifacts WHERE is_active=1")[0]
    payload = json.loads(row["payload_json"])
    payload["feature_version"] = 999
    conn.execute(
        "UPDATE model_artifacts SET payload_json = ? WHERE is_active = 1", (json.dumps(payload),)
    )

    restored = TasteRanker.load(conn)
    assert restored._model is None
    assert restored.metrics.model_kind == "heuristic"
    assert restored.metrics.blend_weight == 0.0


def test_an_unusable_model_reports_the_prior_that_is_actually_serving(conn):
    """It used to keep reporting "ridge, spearman 0.51" while the heuristic did
    all the ranking — a number for a model that was not running."""
    from movierec.db import fetch_all

    fm, y = make_fm(160)
    scores = _split_scores({"heuristic": [0.2] * 5, "ridge": [0.6] * 5})
    TasteRanker.fit(fm, y, oof_scores=scores).save(conn)

    row = fetch_all(conn, "SELECT payload_json FROM model_artifacts WHERE is_active=1")[0]
    payload = json.loads(row["payload_json"])
    payload["model_pickle_b64"] = "bm90IGEgcGlja2xl"  # decodes, does not unpickle
    conn.execute(
        "UPDATE model_artifacts SET payload_json = ? WHERE is_active = 1", (json.dumps(payload),)
    )

    restored = TasteRanker.load(conn)
    assert restored._model is None
    assert restored.metrics.model_kind == "heuristic"
    assert restored.metrics.spearman == 0.0, "do not quote a score for a model that cannot run"
    assert restored.metrics.top_features, "the prior's own weights should be shown instead"
