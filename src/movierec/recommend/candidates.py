"""Multi-source candidate generation.

Retrieval and ranking are separated on purpose. Each source has a different
failure mode - embeddings drift toward description-similar films, CF drifts
toward popular ones, facet rules drift toward whatever you already watch - so
the union is far more robust than any one of them, and the ranker sorts it out.

Every candidate remembers which sources proposed it, which is what makes the
"why am I seeing this" line in the UI truthful rather than reconstructed.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..db import fetch_all
from ..enrich.embeddings import MOVIE, EmbeddingBackend, VectorStore
from ..logging_utils import get_logger
from ..taste.profile import TasteProfile

log = get_logger("recommend.candidates")


@dataclass
class CandidatePool:
    sources: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    retrieval_score: dict[int, float] = field(default_factory=dict)

    def add(self, tmdb_id: int, source: str, score: float = 0.0) -> None:
        tmdb_id = int(tmdb_id)
        self.sources[tmdb_id].add(source)
        if score > self.retrieval_score.get(tmdb_id, -1e9):
            self.retrieval_score[tmdb_id] = score

    @property
    def ids(self) -> list[int]:
        return list(self.sources.keys())

    def __len__(self) -> int:
        return len(self.sources)


def seen_tmdb_ids(conn: sqlite3.Connection) -> set[int]:
    """Everything the user has already watched or logged."""
    rows = fetch_all(
        conn,
        """
        SELECT DISTINCT f.tmdb_id FROM user_films f
        WHERE f.tmdb_id IS NOT NULL AND (
              EXISTS (SELECT 1 FROM user_watched w WHERE w.film_key = f.film_key)
           OR EXISTS (SELECT 1 FROM user_ratings r WHERE r.film_key = f.film_key)
           OR EXISTS (SELECT 1 FROM user_diary d WHERE d.film_key = f.film_key))
        """,
    )
    return {int(r["tmdb_id"]) for r in rows}


def watchlist_tmdb_ids(conn: sqlite3.Connection) -> set[int]:
    rows = fetch_all(
        conn,
        "SELECT f.tmdb_id FROM user_watchlist w JOIN user_films f USING(film_key) WHERE f.tmdb_id IS NOT NULL",
    )
    return {int(r["tmdb_id"]) for r in rows}


def dismissed_tmdb_ids(conn: sqlite3.Connection, *, days: int = 60) -> set[int]:
    rows = fetch_all(
        conn,
        "SELECT DISTINCT tmdb_id FROM feedback WHERE action IN ('dislike','dismiss') "
        "AND tmdb_id IS NOT NULL AND created_at >= datetime('now', ?)",
        (f"-{int(days)} days",),
    )
    return {int(r["tmdb_id"]) for r in rows}


def generate(
    conn: sqlite3.Connection,
    cfg: Config,
    profile: TasteProfile,
    store: VectorStore,
    *,
    query_vector: np.ndarray | None = None,
    exclude: set[int] | None = None,
    per_source: int | None = None,
    include_watchlist: bool = True,
    hard_filter_ids: set[int] | None = None,
) -> CandidatePool:
    """Union candidates from every retrieval strategy.

    ``hard_filter_ids`` restricts every source to a pre-filtered set (used when
    the natural-language layer imposes genre, year or runtime constraints).
    """
    per_source = per_source or cfg.candidates_per_source
    exclude = set(exclude or set())
    pool = CandidatePool()

    allowed: set[str] | None = (
        {str(i) for i in hard_filter_ids} if hard_filter_ids is not None else None
    )
    excluded_keys = {str(i) for i in exclude}

    def _search(vec: np.ndarray, k: int, source: str, weight: float = 1.0) -> None:
        if not len(store):
            return
        hits = store.search(vec, k=k * (3 if allowed is not None else 1), exclude=excluded_keys)
        taken = 0
        for entity_id, score in hits:
            if allowed is not None and entity_id not in allowed:
                continue
            pool.add(int(entity_id), source, score * weight)
            taken += 1
            if taken >= k:
                break

    # 1. Each mode of taste retrieves its own neighbourhood.
    for mode in profile.modes:
        k = max(40, int(per_source * mode.weight))
        _search(mode.centroid, k, f"taste:{mode.label[:28]}", 1.0)

    # 2. Nearest neighbours of individual favourites - narrower and more literal.
    for row in fetch_all(
        conn,
        """SELECT f.tmdb_id, f.title, r.rating FROM user_ratings r JOIN user_films f USING(film_key)
           WHERE f.tmdb_id IS NOT NULL AND r.rating >= 4.0 ORDER BY r.rating DESC, r.rated_date DESC LIMIT 12""",
    ):
        vec = store.vector(str(row["tmdb_id"]))
        if vec is not None:
            _search(vec, 30, f"similar-to:{row['title']}", 0.95)

    # 3. Collaborative filtering: taste-adjacent rather than description-adjacent.
    liked_rows = fetch_all(
        conn,
        """SELECT f.tmdb_id, r.rating FROM user_ratings r JOIN user_films f USING(film_key)
           WHERE f.tmdb_id IS NOT NULL AND r.rating >= 3.5""",
    )
    liked = {int(r["tmdb_id"]): float(r["rating"]) for r in liked_rows}
    if liked:
        cf_scores: dict[int, float] = defaultdict(float)
        ids = list(liked.keys())
        for start in range(0, len(ids), 400):
            chunk = ids[start : start + 400]
            ph = ",".join("?" for _ in chunk)
            for r in fetch_all(
                conn,
                f"SELECT tmdb_id, neighbor_tmdb_id, score FROM cf_neighbors WHERE tmdb_id IN ({ph})",
                chunk,
            ):
                nb = int(r["neighbor_tmdb_id"])
                if nb in exclude:
                    continue
                if allowed is not None and str(nb) not in allowed:
                    continue
                cf_scores[nb] += float(r["score"]) * (liked[int(r["tmdb_id"])] - 2.5)
        for tmdb_id, score in sorted(cf_scores.items(), key=lambda t: -t[1])[:per_source]:
            pool.add(tmdb_id, "viewers-like-you", min(1.0, score / 10.0))

    # 4. Facet rules: strong genre/tag affinity plus a quality floor.
    top_genres = [
        g
        for g, v in sorted(profile.affinities.get("genre", {}).items(), key=lambda t: -t[1])[:4]
        if v > 0.05
    ]
    if top_genres:
        ph = ",".join("?" for _ in top_genres)
        for r in fetch_all(
            conn,
            f"""SELECT m.tmdb_id, m.tmdb_vote_average, m.tmdb_vote_count
                FROM movies m JOIN movie_genres mg USING(tmdb_id) JOIN genres g USING(genre_id)
                WHERE g.name IN ({ph}) AND m.in_catalog = 1 AND m.detail_level = 2
                  AND m.tmdb_vote_count >= 300 AND m.tmdb_vote_average >= 7.0
                ORDER BY m.tmdb_vote_average * (m.tmdb_vote_count / (m.tmdb_vote_count + 500.0)) DESC LIMIT ?""",
            [*top_genres, per_source],
        ):
            tmdb_id = int(r["tmdb_id"])
            if tmdb_id in exclude or (allowed is not None and str(tmdb_id) not in allowed):
                continue
            pool.add(tmdb_id, "well-reviewed-in-your-genres", 0.4)

    # 5. The watchlist is an explicit statement of intent; always eligible.
    if include_watchlist:
        for tmdb_id in watchlist_tmdb_ids(conn):
            if tmdb_id in exclude or (allowed is not None and str(tmdb_id) not in allowed):
                continue
            pool.add(tmdb_id, "your-watchlist", 0.5)

    # 6. Deliberate exploration: acclaimed films far from every taste centroid.
    if cfg.exploration_ratio > 0 and profile.modes and len(store):
        sims = np.max(np.vstack([store.similarity(m.centroid) for m in profile.modes]), axis=0)
        quality = _quality_lookup(conn, store.ids)
        far = np.argsort(sims)  # least similar first
        taken = 0
        budget = max(20, int(per_source * cfg.exploration_ratio))
        for idx in far:
            entity_id = str(store.ids[idx])
            tmdb_id = int(entity_id)
            if tmdb_id in exclude or quality.get(tmdb_id, 0.0) < 7.2:
                continue
            if allowed is not None and entity_id not in allowed:
                continue
            pool.add(tmdb_id, "outside-your-usual", 0.25)
            taken += 1
            if taken >= budget:
                break

    # 7. Whatever the request itself asked for.
    if query_vector is not None:
        _search(query_vector, per_source, "matches-your-request", 1.0)

    log.info(
        "candidate pool: %d films from %d sources",
        len(pool),
        len({s for ss in pool.sources.values() for s in ss}),
    )
    return pool


def _quality_lookup(conn: sqlite3.Connection, ids: np.ndarray) -> dict[int, float]:
    rows = fetch_all(
        conn,
        "SELECT tmdb_id, tmdb_vote_average FROM movies WHERE in_catalog = 1 AND tmdb_vote_count >= 500",
    )
    return {int(r["tmdb_id"]): float(r["tmdb_vote_average"] or 0.0) for r in rows}


def load_movie_store(conn: sqlite3.Connection, backend: EmbeddingBackend) -> VectorStore:
    return VectorStore.load(conn, MOVIE, backend.name)
