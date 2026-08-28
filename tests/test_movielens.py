"""Tag genome and collaborative filtering, against a synthetic MovieLens."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movierec.db import fetch_all, scalar, upsert
from movierec.ingest.movielens import (
    GENOME_MIN_RELEVANCE,
    GENOME_TOP_K,
    build_cf,
    ingest_genome,
    load_crosswalk,
)

N_MOVIES, N_TAGS, N_USERS = 60, 50, 600
CLUSTERS = {0: range(1, 21), 1: range(21, 41), 2: range(41, 61)}


@pytest.fixture
def ml_dir(tmp_path):
    """A MovieLens-shaped dataset with three planted taste clusters."""
    d = tmp_path / "ml-25m"
    d.mkdir()
    rng = np.random.default_rng(5)

    pd.DataFrame(
        {
            "movieId": range(1, N_MOVIES + 1),
            "imdbId": [f"{i:07d}" for i in range(1, N_MOVIES + 1)],
            "tmdbId": [1000 + i for i in range(1, N_MOVIES + 1)],
        }
    ).to_csv(d / "links.csv", index=False)

    pd.DataFrame(
        {"tagId": range(1, N_TAGS + 1), "tag": [f"tag{i}" for i in range(1, N_TAGS + 1)]}
    ).to_csv(d / "genome-tags.csv", index=False)

    rows = []
    for m in range(1, N_MOVIES + 1):
        for t in range(1, N_TAGS + 1):
            # Movie 1 gets a clean descending ordering so top-K is checkable.
            rel = max(0.0, 1.0 - 0.02 * t) if m == 1 else float(rng.random())
            rows.append((m, t, round(rel, 4)))
    pd.DataFrame(rows, columns=["movieId", "tagId", "relevance"]).to_csv(
        d / "genome-scores.csv", index=False
    )

    ratings = []
    for u in range(1, N_USERS + 1):
        cluster = u % 3
        for m in rng.choice(list(CLUSTERS[cluster]), size=12, replace=False):
            ratings.append((u, int(m), 5.0))
        for m in rng.choice(list(CLUSTERS[(cluster + 1) % 3]), size=3, replace=False):
            ratings.append((u, int(m), 1.5))  # disliked: must be excluded
    pd.DataFrame(ratings, columns=["userId", "movieId", "rating"]).to_csv(
        d / "ratings.csv", index=False
    )
    return d


@pytest.fixture
def seeded(conn):
    upsert(
        conn,
        "movies",
        [
            {"tmdb_id": 1000 + i, "title": f"M{i}", "in_catalog": 1, "detail_level": 2}
            for i in range(1, N_MOVIES + 1)
        ],
        key=["tmdb_id"],
    )
    return conn


def test_crosswalk_restricted_to_catalog(seeded, ml_dir):
    seeded.execute("UPDATE movies SET in_catalog = 0 WHERE tmdb_id = 1001")
    assert len(load_crosswalk(seeded, ml_dir)) == N_MOVIES - 1


def test_genome_respects_relevance_floor_and_top_k(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    ingest_genome(seeded, ml_dir, cw)
    assert (
        scalar(
            seeded, "SELECT COUNT(*) FROM movie_tags WHERE relevance < ?", (GENOME_MIN_RELEVANCE,)
        )
        == 0
    )
    worst = scalar(
        seeded, "SELECT MAX(c) FROM (SELECT COUNT(*) c FROM movie_tags GROUP BY tmdb_id)"
    )
    assert worst <= GENOME_TOP_K


def test_genome_keeps_the_most_relevant_tags(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    ingest_genome(seeded, ml_dir, cw)
    top = [
        r["tag"]
        for r in fetch_all(
            seeded, "SELECT tag FROM movie_tags WHERE tmdb_id=1001 ORDER BY relevance DESC LIMIT 3"
        )
    ]
    assert top == ["tag1", "tag2", "tag3"]


def test_genome_is_replaced_not_duplicated(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    first = ingest_genome(seeded, ml_dir, cw)["rows"]
    second = ingest_genome(seeded, ml_dir, cw)["rows"]
    assert first == second
    assert scalar(seeded, "SELECT COUNT(*) FROM movie_tags") == first


def test_cf_recovers_the_planted_clusters(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    result = build_cf(seeded, ml_dir, cw, block_size=16)
    assert result["items"] > 0 and result["edges"] > 0

    correct = total = 0
    for movie_id in (3, 25, 50):
        cluster = next(c for c, members in CLUSTERS.items() if movie_id in members)
        expected = {1000 + m for m in CLUSTERS[cluster]}
        rows = fetch_all(
            seeded,
            "SELECT neighbor_tmdb_id FROM cf_neighbors WHERE tmdb_id=? ORDER BY score DESC LIMIT 8",
            (1000 + movie_id,),
        )
        for r in rows:
            total += 1
            correct += r["neighbor_tmdb_id"] in expected
    assert correct / max(total, 1) > 0.85, f"only {correct}/{total} neighbours in the right cluster"


def test_cf_has_no_self_edges(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    build_cf(seeded, ml_dir, cw, block_size=16)
    assert scalar(seeded, "SELECT COUNT(*) FROM cf_neighbors WHERE tmdb_id = neighbor_tmdb_id") == 0


def test_cf_is_replaced_on_rerun(seeded, ml_dir):
    cw = load_crosswalk(seeded, ml_dir)
    first = build_cf(seeded, ml_dir, cw, block_size=16)["edges"]
    assert scalar(seeded, "SELECT COUNT(*) FROM cf_neighbors") == first
    build_cf(seeded, ml_dir, cw, block_size=16)
    assert scalar(seeded, "SELECT COUNT(*) FROM cf_neighbors") == first


def test_cf_handles_an_empty_crosswalk(seeded, ml_dir):
    assert build_cf(seeded, ml_dir, {}, block_size=16) == {"items": 0, "edges": 0}
