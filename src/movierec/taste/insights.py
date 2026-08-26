"""Derived statistics for the Insights tab.

Everything here is a plain query or a small numpy computation, returned as
dataframes and dicts so the UI layer stays presentation-only.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from ..db import fetch_all, scalar
from .profile import TasteProfile


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows])


def headline_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "films_watched": scalar(conn, "SELECT COUNT(*) FROM user_watched", default=0),
        "films_rated": scalar(conn, "SELECT COUNT(*) FROM user_ratings", default=0),
        "reviews_written": scalar(conn, "SELECT COUNT(*) FROM user_reviews", default=0),
        "watchlist": scalar(conn, "SELECT COUNT(*) FROM user_watchlist", default=0),
        "mean_rating": scalar(conn, "SELECT ROUND(AVG(rating), 2) FROM user_ratings", default=0.0),
        "catalog_size": scalar(conn, "SELECT COUNT(*) FROM movies WHERE in_catalog = 1", default=0),
        "catalog_detailed": scalar(
            conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2", default=0
        ),
        "with_plot": scalar(
            conn, "SELECT COUNT(*) FROM movie_texts WHERE source='wikipedia_plot'", default=0
        ),
        "dossiers": scalar(conn, "SELECT COUNT(*) FROM movie_dossiers", default=0),
        "unmatched": scalar(
            conn, "SELECT COUNT(*) FROM user_films WHERE tmdb_id IS NULL", default=0
        ),
        "needs_review": scalar(
            conn, "SELECT COUNT(*) FROM user_films WHERE needs_review = 1", default=0
        ),
    }


def rating_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        fetch_all(
            conn,
            "SELECT rating, COUNT(*) AS films FROM user_ratings GROUP BY rating ORDER BY rating",
        )
    )


def affinity_table(
    profile: TasteProfile, facet: str, top: int = 14, min_abs: float = 0.02
) -> pd.DataFrame:
    data = profile.affinities.get(facet, {})
    rows = [{"name": k, "affinity": v} for k, v in data.items() if abs(v) >= min_abs]
    if not rows:
        return pd.DataFrame(columns=["name", "affinity"])
    df = pd.DataFrame(rows).sort_values("affinity", ascending=False)
    return pd.concat([df.head(top), df.tail(min(top, max(0, len(df) - top)))]).drop_duplicates(
        "name"
    )


def genre_coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """What fraction of each genre's well-known films the user has seen."""
    rows = fetch_all(
        conn,
        """
        SELECT g.name AS genre,
               COUNT(DISTINCT CASE WHEN uf.tmdb_id IS NOT NULL THEN mg.tmdb_id END) AS watched,
               COUNT(DISTINCT CASE WHEN m.tmdb_vote_count > 1000 THEN mg.tmdb_id END) AS well_known,
               ROUND(AVG(CASE WHEN ur.rating IS NOT NULL THEN ur.rating END), 2) AS mean_rating
        FROM genres g
        JOIN movie_genres mg USING(genre_id)
        JOIN movies m USING(tmdb_id)
        LEFT JOIN user_films uf ON uf.tmdb_id = mg.tmdb_id
        LEFT JOIN user_ratings ur ON ur.film_key = uf.film_key
        WHERE m.in_catalog = 1
        GROUP BY g.genre_id
        HAVING well_known > 50
        ORDER BY watched DESC
        """,
    )
    df = _df(rows)
    if not df.empty:
        df["coverage_pct"] = (100.0 * df["watched"] / df["well_known"].clip(lower=1)).round(2)
    return df


def watching_timeline(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        fetch_all(
            conn,
            """SELECT substr(watched_date, 1, 7) AS month, COUNT(*) AS films, ROUND(AVG(rating), 2) AS mean_rating
           FROM user_diary WHERE watched_date IS NOT NULL AND watched_date != ''
           GROUP BY month ORDER BY month""",
        )
    )


def decade_profile(conn: sqlite3.Connection) -> pd.DataFrame:
    return _df(
        fetch_all(
            conn,
            """SELECT (m.year / 10) * 10 AS decade, COUNT(*) AS films, ROUND(AVG(r.rating), 2) AS mean_rating
           FROM user_ratings r JOIN user_films f USING(film_key) JOIN movies m ON m.tmdb_id = f.tmdb_id
           WHERE m.year IS NOT NULL GROUP BY decade ORDER BY decade""",
        )
    )


def crowd_disagreement(conn: sqlite3.Connection, top: int = 12) -> dict[str, pd.DataFrame]:
    """Where the user's rating departs most from the consensus."""
    rows = fetch_all(
        conn,
        """SELECT f.title, f.year, r.rating AS yours,
                  ROUND(m.tmdb_vote_average / 2.0, 2) AS crowd,
                  ROUND(r.rating - m.tmdb_vote_average / 2.0, 2) AS delta
           FROM user_ratings r JOIN user_films f USING(film_key) JOIN movies m ON m.tmdb_id = f.tmdb_id
           WHERE m.tmdb_vote_count > 500 AND m.tmdb_vote_average > 0""",
    )
    df = _df(rows)
    if df.empty:
        return {"overrated": df, "underrated": df}
    df = df.sort_values("delta")
    return {
        "overrated": df.head(top).reset_index(drop=True),
        "underrated": df.tail(top).iloc[::-1].reset_index(drop=True),
    }


def aspect_table(profile: TasteProfile) -> pd.DataFrame:
    rows = [{"aspect": k, "affinity": v} for k, v in profile.aspect_affinity.items()]
    if not rows:
        return pd.DataFrame(columns=["aspect", "affinity"])
    return pd.DataFrame(rows).sort_values("affinity", ascending=False).reset_index(drop=True)


def scale_table(profile: TasteProfile) -> pd.DataFrame:
    rows = [
        {"scale": k, "your_sweet_spot": v, "how_much_it_matters": profile.scale_weights.get(k, 0.0)}
        for k, v in profile.scale_targets.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["scale", "your_sweet_spot", "how_much_it_matters"])
    return (
        pd.DataFrame(rows)
        .sort_values("how_much_it_matters", ascending=False)
        .reset_index(drop=True)
    )


def most_similar_pairs(
    conn: sqlite3.Connection, profile: TasteProfile, embed_model: str, top: int = 8
) -> pd.DataFrame:
    """Films they rated very differently despite being close in embedding space.

    A useful diagnostic: these are the cases where content similarity alone
    would mislead the recommender.
    """
    from ..db import blob_to_vector

    rows = fetch_all(
        conn,
        """SELECT f.tmdb_id, f.title, f.year, r.rating FROM user_ratings r
           JOIN user_films f USING(film_key) WHERE f.tmdb_id IS NOT NULL""",
    )
    if len(rows) < 6:
        return pd.DataFrame(columns=["film_a", "film_b", "similarity", "rating_gap"])

    ids = [int(r["tmdb_id"]) for r in rows]
    ph = ",".join("?" for _ in ids)
    vec_rows = fetch_all(
        conn,
        f"SELECT entity_id, vector FROM embeddings WHERE entity_type='movie' AND model=? AND entity_id IN ({ph})",
        [embed_model, *[str(i) for i in ids]],
    )
    vectors = {int(r["entity_id"]): blob_to_vector(r["vector"]) for r in vec_rows}
    have = [r for r in rows if int(r["tmdb_id"]) in vectors]
    if len(have) < 6:
        return pd.DataFrame(columns=["film_a", "film_b", "similarity", "rating_gap"])

    mat = np.vstack([vectors[int(r["tmdb_id"])] for r in have])
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)
    ratings = np.array([float(r["rating"]) for r in have])
    gaps = np.abs(ratings[:, None] - ratings[None, :])

    interest = sims * (gaps / 5.0)
    out = []
    iu = np.triu_indices(len(have), k=1)
    order = np.argsort(-interest[iu])[: top * 3]
    seen: set[int] = set()
    for pos in order:
        i, j = int(iu[0][pos]), int(iu[1][pos])
        if sims[i, j] < 0.5 or gaps[i, j] < 1.5 or i in seen:
            continue
        seen.add(i)
        out.append(
            {
                "film_a": f"{have[i]['title']} ({have[i]['rating']}★)",
                "film_b": f"{have[j]['title']} ({have[j]['rating']}★)",
                "similarity": round(float(sims[i, j]), 3),
                "rating_gap": round(float(gaps[i, j]), 1),
            }
        )
        if len(out) >= top:
            break
    return pd.DataFrame(out)


def ingest_history(conn: sqlite3.Connection, limit: int = 40) -> pd.DataFrame:
    return _df(
        fetch_all(
            conn,
            """SELECT run_id, kind, stage, status, started_at, finished_at, stats_json, error
           FROM ingest_runs ORDER BY run_id DESC LIMIT ?""",
            (limit,),
        )
    )
