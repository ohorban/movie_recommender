"""Feature construction for the personalized ranker.

Every feature is designed to be meaningful on its own, because with a few
hundred training examples the model has no capacity to discover interactions.
The heuristic fallback in :mod:`movierec.recommend.ranker` uses the same
features with hand-set weights, so a cold-start user still gets sane results.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..db import blob_to_vector, fetch_all
from ..enrich.embeddings import MOVIE, PLOT
from ..enrich.structuring import DOSSIER_SCALES
from ..logging_utils import get_logger
from ..taste.profile import TasteProfile, load_movie_facets, runtime_bucket

log = get_logger("recommend.features")

FEATURE_NAMES = [
    "sim_mode_best",
    "sim_mode_weighted",
    "sim_plot_best",
    "sim_dislike",
    "aff_genre",
    "aff_keyword",
    "aff_tag",
    "aff_director",
    "aff_actor",
    "aff_decade",
    "aff_language",
    "aff_runtime",
    "cf_score",
    "quality",
    "popularity",
    "scale_fit",
    "recency",
]

# Hand-tuned weights used before the learned model earns its keep.
HEURISTIC_WEIGHTS = {
    "sim_mode_best": 1.15,
    "sim_mode_weighted": 0.55,
    "sim_plot_best": 0.45,
    "sim_dislike": -0.70,
    "aff_genre": 0.55,
    "aff_keyword": 0.40,
    "aff_tag": 0.45,
    "aff_director": 0.35,
    "aff_actor": 0.20,
    "aff_decade": 0.20,
    "aff_language": 0.10,
    "aff_runtime": 0.10,
    "cf_score": 0.60,
    "quality": 0.45,
    "popularity": 0.05,
    "scale_fit": 0.50,
    "recency": 0.05,
}

# Bayesian prior for blending TMDB and IMDb scores.
PRIOR_VOTES = 300.0
PRIOR_SCORE = 6.4


@dataclass
class FeatureMatrix:
    ids: list[int]
    matrix: np.ndarray
    names: list[str]

    def as_dict(self, tmdb_id: int) -> dict[str, float]:
        idx = self.ids.index(tmdb_id)
        return dict(zip(self.names, self.matrix[idx].tolist()))


def _mean_top(values: list[float], k: int = 5) -> float:
    if not values:
        return 0.0
    top = sorted(values, key=abs, reverse=True)[:k]
    return float(np.mean(top))


class FeatureBuilder:
    """Pre-loads everything needed to featurise a candidate set in bulk."""

    def __init__(
        self, conn: sqlite3.Connection, profile: TasteProfile, *, embed_model: str
    ) -> None:
        self.conn = conn
        self.profile = profile
        self.embed_model = embed_model
        self.this_year = 2026

        row = fetch_all(conn, "SELECT CAST(strftime('%Y','now') AS INTEGER) AS y")
        if row:
            self.this_year = int(row[0]["y"])

        self.mode_matrix = (
            np.vstack([m.centroid for m in profile.modes]).astype(np.float32)
            if profile.modes
            else np.zeros((0, 1), np.float32)
        )
        self.mode_weights = (
            np.array([m.weight for m in profile.modes], dtype=np.float32)
            if profile.modes
            else np.zeros(0, np.float32)
        )
        self.dislike = profile.dislike_centroid

        # User's liked films drive the CF lookup.
        self._liked: dict[int, float] = {}

    def set_reference_prefs(self, prefs: dict[int, float]) -> None:
        self._liked = {i: p for i, p in prefs.items() if p > 0.2}

    # ------------------------------------------------------------------ bulk
    def _vectors(self, tmdb_ids: list[int], entity_type: str) -> dict[int, np.ndarray]:
        if not tmdb_ids:
            return {}
        out: dict[int, np.ndarray] = {}
        for chunk_start in range(0, len(tmdb_ids), 900):
            chunk = tmdb_ids[chunk_start : chunk_start + 900]
            ph = ",".join("?" for _ in chunk)
            rows = fetch_all(
                self.conn,
                f"SELECT entity_id, vector FROM embeddings WHERE entity_type = ? AND model = ? AND entity_id IN ({ph})",
                [entity_type, self.embed_model, *[str(i) for i in chunk]],
            )
            out.update({int(r["entity_id"]): blob_to_vector(r["vector"]) for r in rows})
        return out

    def _cf_scores(self, tmdb_ids: list[int]) -> dict[int, float]:
        """Weighted CF evidence: how strongly the user's liked films point here."""
        if not self._liked or not tmdb_ids:
            return {}
        candidates = set(tmdb_ids)
        scores: dict[int, float] = defaultdict(float)
        liked_ids = list(self._liked.keys())
        for start in range(0, len(liked_ids), 400):
            chunk = liked_ids[start : start + 400]
            ph = ",".join("?" for _ in chunk)
            for r in fetch_all(
                self.conn,
                f"SELECT tmdb_id, neighbor_tmdb_id, score FROM cf_neighbors WHERE tmdb_id IN ({ph})",
                chunk,
            ):
                nb = r["neighbor_tmdb_id"]
                if nb in candidates:
                    scores[nb] += float(r["score"]) * self._liked[r["tmdb_id"]]
        if not scores:
            return {}
        # Squash into 0-1 so a film adjacent to many favourites cannot dominate.
        peak = max(scores.values()) or 1.0
        return {k: math.tanh(2.0 * v / peak) for k, v in scores.items()}

    def _quality(self, row: dict[str, Any]) -> float:
        tmdb_score, tmdb_votes = (
            row.get("tmdb_vote_average") or 0.0,
            row.get("tmdb_vote_count") or 0,
        )
        imdb_score, imdb_votes = row.get("imdb_rating") or 0.0, row.get("imdb_votes") or 0
        num = PRIOR_SCORE * PRIOR_VOTES
        den = PRIOR_VOTES
        if tmdb_votes:
            num += tmdb_score * tmdb_votes
            den += tmdb_votes
        if imdb_votes:
            num += imdb_score * min(imdb_votes, 200_000)
            den += min(imdb_votes, 200_000)
        # Map a 0-10 scale onto roughly -1..1 centred on a typical film.
        return float((num / den - PRIOR_SCORE) / 1.6)

    def _scale_fit(self, dossier: dict[str, Any] | None) -> float:
        if not dossier or not self.profile.scale_targets:
            return 0.0
        scales = dossier.get("scales") or {}
        num = den = 0.0
        for name in DOSSIER_SCALES:
            target = self.profile.scale_targets.get(name)
            weight = self.profile.scale_weights.get(name, 0.0)
            value = scales.get(name)
            if target is None or value is None or weight <= 0.01:
                continue
            num += weight * (1.0 - 2.0 * abs(float(value) - float(target)))
            den += weight
        return float(num / den) if den > 0 else 0.0

    # ------------------------------------------------------------------ main
    def build(
        self, tmdb_ids: list[int], dossiers: dict[int, dict[str, Any]] | None = None
    ) -> FeatureMatrix:
        ids = [int(i) for i in tmdb_ids]
        if not ids:
            return FeatureMatrix(
                [], np.zeros((0, len(FEATURE_NAMES)), np.float32), list(FEATURE_NAMES)
            )

        facets = load_movie_facets(self.conn, ids)
        movie_vecs = self._vectors(ids, MOVIE)
        plot_vecs = self._vectors(ids, PLOT)
        cf = self._cf_scores(ids)
        dossiers = dossiers or {}

        rows: dict[int, dict[str, Any]] = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start : start + 900]
            ph = ",".join("?" for _ in chunk)
            for r in fetch_all(
                self.conn,
                f"""SELECT tmdb_id, year, runtime, original_language, tmdb_vote_average, tmdb_vote_count,
                           tmdb_popularity, imdb_rating, imdb_votes
                    FROM movies WHERE tmdb_id IN ({ph})""",
                chunk,
            ):
                rows[r["tmdb_id"]] = dict(r)

        aff = self.profile.affinities
        out = np.zeros((len(ids), len(FEATURE_NAMES)), dtype=np.float32)

        for i, tmdb_id in enumerate(ids):
            f = facets.get(tmdb_id, {})
            row = rows.get(tmdb_id, {})
            vec = movie_vecs.get(tmdb_id)

            if vec is not None and self.mode_matrix.shape[0]:
                sims = self.mode_matrix @ vec
                sim_best = float(sims.max())
                sim_weighted = float((sims * self.mode_weights).sum())
            else:
                sim_best = sim_weighted = 0.0

            # Films without a Wikipedia synopsis have no plot vector. Falling
            # back to the profile similarity keeps the feature comparable rather
            # than penalising every film that lacks an article.
            pvec = plot_vecs.get(tmdb_id)
            if pvec is not None and self.mode_matrix.shape[0]:
                sim_plot = float((self.mode_matrix @ pvec).max())
            else:
                sim_plot = sim_best
            sim_dislike = (
                float(self.dislike @ vec) if (self.dislike is not None and vec is not None) else 0.0
            )

            values = {
                "sim_mode_best": sim_best,
                "sim_mode_weighted": sim_weighted,
                "sim_plot_best": sim_plot,
                "sim_dislike": sim_dislike,
                "aff_genre": _mean_top(
                    [aff.get("genre", {}).get(g, 0.0) for g in f.get("genre", [])], 3
                ),
                "aff_keyword": _mean_top(
                    [aff.get("keyword", {}).get(k, 0.0) for k in f.get("keyword", [])], 6
                ),
                "aff_tag": _mean_top([aff.get("tag", {}).get(t, 0.0) for t in f.get("tag", [])], 6),
                "aff_director": _mean_top(
                    [aff.get("director", {}).get(d, 0.0) for d in f.get("director", [])], 1
                ),
                "aff_actor": _mean_top(
                    [aff.get("actor", {}).get(a, 0.0) for a in f.get("actor", [])], 3
                ),
                "aff_decade": aff.get("decade", {}).get(
                    f"{((row.get('year') or 0) // 10) * 10}s", 0.0
                ),
                "aff_language": aff.get("language", {}).get(
                    row.get("original_language") or "", 0.0
                ),
                "aff_runtime": aff.get("runtime", {}).get(runtime_bucket(row.get("runtime")), 0.0),
                "cf_score": cf.get(tmdb_id, 0.0),
                "quality": self._quality(row),
                "popularity": float(math.log1p(row.get("tmdb_popularity") or 0.0) / 6.0),
                "scale_fit": self._scale_fit(dossiers.get(tmdb_id)),
                "recency": float(np.clip(((row.get("year") or 1990) - 1990) / 40.0, -1.0, 1.0)),
            }
            for j, name in enumerate(FEATURE_NAMES):
                out[i, j] = values[name]

        return FeatureMatrix(ids, out, list(FEATURE_NAMES))
