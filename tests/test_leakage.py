"""Regression tests for target leakage in the ranker.

Every taste feature is *fitted* on the ratings we are trying to predict, which
makes this system unusually easy to fool. On real data the naive setup reported
a cross-validated rank correlation of 0.96 where the honest figure was 0.53.

The test below reproduces that in miniature: films whose ratings are pure
noise, each with a director nobody else has. Any evaluation that still finds
signal there is reading the label out of its own features.
"""

from __future__ import annotations

import numpy as np
import pytest

from movierec.db import insert_ignore, upsert
from movierec.enrich.embeddings import HashBackend, embed_movies
from movierec.recommend.features import FeatureBuilder
from movierec.recommend.ranker import TasteRanker, _spearman
from movierec.taste.profile import build_profile, preference_scores
from movierec.taste.training import train_ranker

N_FILMS = 90


def _corr(column: np.ndarray, y: np.ndarray) -> float:
    """Correlation, treating a constant column as carrying no information."""
    if float(np.std(column)) < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(column, y)[0, 1]))


@pytest.fixture
def noisy_library(conn):
    """Films with unique directors and ratings drawn at random.

    There is nothing to learn here. Any reported skill is leakage.
    """
    rng = np.random.default_rng(4)
    upsert(
        conn,
        "movies",
        [
            {
                "tmdb_id": i,
                "title": f"Film {i}",
                "year": 2000 + (i % 25),
                "overview": f"A story about subject {i % 11} and situation {i % 7}.",
                "runtime": 90 + (i % 5) * 12,
                "original_language": "en",
                "in_catalog": 1,
                "detail_level": 2,
                "tmdb_vote_count": 500 + i,
                "tmdb_vote_average": 6.5,
            }
            for i in range(1, N_FILMS + 1)
        ],
        key=["tmdb_id"],
    )
    insert_ignore(conn, "genres", [{"genre_id": 18, "name": "Drama"}])
    insert_ignore(
        conn, "movie_genres", [{"tmdb_id": i, "genre_id": 18} for i in range(1, N_FILMS + 1)]
    )
    # One director per film: with no leave-one-out correction, that director's
    # affinity is a direct function of the film's own rating.
    insert_ignore(
        conn,
        "people",
        [{"person_id": 1000 + i, "name": f"Director {i}"} for i in range(1, N_FILMS + 1)],
    )
    insert_ignore(
        conn,
        "movie_credits",
        [
            {
                "tmdb_id": i,
                "person_id": 1000 + i,
                "role": "crew",
                "job": "Director",
                "character": None,
                "cast_order": None,
            }
            for i in range(1, N_FILMS + 1)
        ],
    )

    upsert(
        conn,
        "user_films",
        [
            {
                "film_key": f"film-{i}|{2000 + (i % 25)}",
                "title": f"Film {i}",
                "year": 2000 + (i % 25),
                "tmdb_id": i,
            }
            for i in range(1, N_FILMS + 1)
        ],
        key=["film_key"],
    )
    ratings = rng.uniform(0.5, 5.0, size=N_FILMS)
    upsert(
        conn,
        "user_ratings",
        [
            {"film_key": f"film-{i}|{2000 + (i % 25)}", "rating": float(ratings[i - 1])}
            for i in range(1, N_FILMS + 1)
        ],
        key=["film_key"],
    )

    backend = HashBackend(dim=96)
    embed_movies(conn, backend)
    return conn, backend


def test_naive_features_leak_the_label(noisy_library):
    """Without leave-one-out the features alone all but reproduce the ratings."""
    conn, backend = noisy_library
    profile = build_profile(conn, backend, store=False)
    prefs, _, _ = preference_scores(
        [
            dict(r)
            for r in conn.execute(
                "SELECT f.tmdb_id, r.rating, NULL AS liked FROM user_ratings r "
                "JOIN user_films f USING(film_key)"
            )
        ]
    )
    ids = list(prefs)
    y = np.array([prefs[i] for i in ids])

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    leaky = builder.build(ids)
    idx = leaky.names.index("aff_director")
    assert _corr(leaky.matrix[:, idx], y) > 0.85, (
        "a one-off director should trivially reproduce the label without LOO"
    )


def test_leave_one_out_removes_the_leak(noisy_library):
    conn, backend = noisy_library
    profile = build_profile(conn, backend, store=False)
    prefs, _, _ = preference_scores(
        [
            dict(r)
            for r in conn.execute(
                "SELECT f.tmdb_id, r.rating, NULL AS liked FROM user_ratings r "
                "JOIN user_films f USING(film_key)"
            )
        ]
    )
    ids = list(prefs)
    y = np.array([prefs[i] for i in ids])

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    clean = builder.build(ids, loo_prefs=prefs)
    idx = clean.names.index("aff_director")
    # Each director is seen exactly once, so leaving that film out leaves no
    # evidence at all — the feature collapses to a constant zero.
    assert _corr(clean.matrix[:, idx], y) < 0.2, "LOO must break the identity"
    assert float(np.std(clean.matrix[:, idx])) < 1e-6


def test_honest_evaluation_finds_no_signal_in_noise(noisy_library):
    """The end-to-end check: random ratings must not produce a confident model."""
    conn, backend = noisy_library
    profile = build_profile(conn, backend, store=False)
    ranker = train_ranker(conn, profile, backend, store=False)

    assert ranker.metrics.spearman < 0.35, (
        f"reported {ranker.metrics.spearman:.3f} rank correlation on pure noise"
    )
    assert ranker.metrics.blend_weight < 0.8, "a model this weak must not be fully trusted"


def test_naive_in_sample_cv_would_have_been_fooled(noisy_library):
    """Documents the failure mode the honest evaluation replaces."""
    conn, backend = noisy_library
    profile = build_profile(conn, backend, store=False)
    prefs, _, _ = preference_scores(
        [
            dict(r)
            for r in conn.execute(
                "SELECT f.tmdb_id, r.rating, NULL AS liked FROM user_ratings r "
                "JOIN user_films f USING(film_key)"
            )
        ]
    )
    ids = list(prefs)
    y = np.array([prefs[i] for i in ids])

    builder = FeatureBuilder(conn, profile, embed_model=backend.name)
    builder.set_reference_prefs(prefs)
    naive = TasteRanker.fit(builder.build(ids), y)  # no LOO, internal CV
    honest = train_ranker(conn, profile, backend, store=False)

    assert naive.metrics.spearman > honest.metrics.spearman + 0.3, (
        "the naive path should look dramatically better than it is — that is the bug"
    )
    assert _spearman(y, np.asarray(y)) == pytest.approx(1.0)
