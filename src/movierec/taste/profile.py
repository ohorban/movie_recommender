"""Build a model of the user's taste from ratings, reviews and film metadata.

Three deliberate choices shape this module.

**Preference, not rating.** A 3/5 from someone whose mean is 2.9 is mild
approval, not a middling score. Everything downstream works in units of
standard deviations from the user's own mean.

**Multi-modal, not a single centroid.** Averaging every liked film produces a
vector that describes nobody's taste - the midpoint of a war film and an
animated musical is neither. Liked films are clustered instead, and a candidate
only has to match one cluster well.

**Shrunk affinities.** With a couple of hundred ratings, a genre seen twice
will otherwise look like the strongest signal in the dataset. Every affinity is
pulled toward zero in proportion to how little evidence supports it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..db import blob_to_vector, fetch_all, transaction, utcnow
from ..enrich.embeddings import MOVIE, EmbeddingBackend
from ..enrich.structuring import DOSSIER_SCALES, load_dossiers, load_review_facts
from ..logging_utils import get_logger

log = get_logger("taste.profile")

# Evidence needed before an affinity is trusted at full strength.
SHRINK_K = {
    "genre": 4.0,
    "keyword": 2.5,
    "tag": 3.0,
    "decade": 4.0,
    "director": 1.5,
    "actor": 2.0,
    "language": 3.0,
    "runtime": 5.0,
    "collection": 1.0,
}
LIKED_THRESHOLD = 0.35  # in preference units (z-scores)
DISLIKED_THRESHOLD = -0.6
MIN_MODE_SIZE = 3


@dataclass
class TasteMode:
    mode_id: int
    label: str
    weight: float
    size: int
    centroid: np.ndarray
    exemplars: list[dict[str, Any]] = field(default_factory=list)
    top_tags: list[str] = field(default_factory=list)
    top_genres: list[str] = field(default_factory=list)
    # Which films are in this mode. Needed to rebuild the centroid without any
    # one of them, which is what keeps the training features leakage-free.
    member_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "label": self.label,
            "weight": round(self.weight, 4),
            "size": self.size,
            "exemplars": self.exemplars,
            "top_tags": self.top_tags,
            "top_genres": self.top_genres,
            "member_ids": self.member_ids,
        }


@dataclass
class TasteProfile:
    n_rated: int = 0
    n_reviewed: int = 0
    rating_mean: float = 3.0
    rating_std: float = 1.0
    modes: list[TasteMode] = field(default_factory=list)
    dislike_centroid: np.ndarray | None = None
    affinities: dict[str, dict[str, float]] = field(default_factory=dict)
    aspect_affinity: dict[str, float] = field(default_factory=dict)
    scale_targets: dict[str, float] = field(default_factory=dict)
    scale_weights: dict[str, float] = field(default_factory=dict)
    taste_signals: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    embed_model: str = ""
    dislike_member_ids: list[int] = field(default_factory=list)
    # Raw statistics behind `affinities`, used only to build leakage-free
    # training features. Not serialised: training always follows a fresh build.
    affinity_stats: AffinityStats = field(default_factory=dict, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_rated": self.n_rated,
            "n_reviewed": self.n_reviewed,
            "rating_mean": round(self.rating_mean, 4),
            "rating_std": round(self.rating_std, 4),
            "modes": [m.to_json() for m in self.modes],
            "affinities": {
                k: {n: round(v, 4) for n, v in sorted(d.items(), key=lambda t: -abs(t[1]))[:80]}
                for k, d in self.affinities.items()
            },
            "aspect_affinity": {k: round(v, 4) for k, v in self.aspect_affinity.items()},
            "scale_targets": {k: round(v, 4) for k, v in self.scale_targets.items()},
            "scale_weights": {k: round(v, 4) for k, v in self.scale_weights.items()},
            "taste_signals": self.taste_signals,
            "summary": self.summary,
            "embed_model": self.embed_model,
            "dislike_member_ids": self.dislike_member_ids,
        }

    def affinity(self, facet: str, name: str) -> float:
        return self.affinities.get(facet, {}).get(name, 0.0)


# --------------------------------------------------------------------------- #
# Ratings
# --------------------------------------------------------------------------- #
def load_user_ratings(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Rated films joined to their resolved TMDB entry."""
    rows = fetch_all(
        conn,
        """
        SELECT f.film_key, f.tmdb_id, f.title, f.year, r.rating, r.rated_date,
               (SELECT 1 FROM user_likes l WHERE l.film_key = f.film_key) AS liked
        FROM user_ratings r
        JOIN user_films f USING(film_key)
        WHERE f.tmdb_id IS NOT NULL
        """,
    )
    return [dict(r) for r in rows]


def preference_scores(ratings: list[dict[str, Any]]) -> tuple[dict[int, float], float, float]:
    """Convert raw stars into z-scored preference, keyed by tmdb_id."""
    values = np.array([r["rating"] for r in ratings], dtype=np.float64)
    if values.size == 0:
        return {}, 3.0, 1.0
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    prefs: dict[int, float] = {}
    for row in ratings:
        z = (row["rating"] - mean) / std
        # An explicit Letterboxd "like" is a second, independent endorsement.
        if row.get("liked"):
            z += 0.25
        prefs[int(row["tmdb_id"])] = float(np.clip(z, -2.5, 2.5))
    return prefs, mean, std


# --------------------------------------------------------------------------- #
# Facets
# --------------------------------------------------------------------------- #
def load_movie_facets(
    conn: sqlite3.Connection, tmdb_ids: list[int]
) -> dict[int, dict[str, list[str]]]:
    """Every categorical descriptor we have for a set of films."""
    if not tmdb_ids:
        return {}
    ph = ",".join("?" for _ in tmdb_ids)
    facets: dict[int, dict[str, list[str]]] = {i: defaultdict(list) for i in tmdb_ids}

    def add(rows, facet, col="name"):
        for r in rows:
            facets[r["tmdb_id"]][facet].append(str(r[col]))

    add(
        fetch_all(
            conn,
            f"SELECT mg.tmdb_id, g.name FROM movie_genres mg JOIN genres g USING(genre_id) WHERE mg.tmdb_id IN ({ph})",
            tmdb_ids,
        ),
        "genre",
    )
    add(
        fetch_all(
            conn,
            f"SELECT mk.tmdb_id, k.name FROM movie_keywords mk JOIN keywords k USING(keyword_id) WHERE mk.tmdb_id IN ({ph})",
            tmdb_ids,
        ),
        "keyword",
    )
    add(
        fetch_all(
            conn,
            f"SELECT tmdb_id, tag AS name FROM movie_tags WHERE tmdb_id IN ({ph}) AND relevance >= 0.6",
            tmdb_ids,
        ),
        "tag",
    )
    add(
        fetch_all(
            conn,
            f"SELECT mc.tmdb_id, p.name FROM movie_credits mc JOIN people p USING(person_id) WHERE mc.tmdb_id IN ({ph}) AND mc.job = 'Director'",
            tmdb_ids,
        ),
        "director",
    )
    add(
        fetch_all(
            conn,
            f"SELECT mc.tmdb_id, p.name FROM movie_credits mc JOIN people p USING(person_id) WHERE mc.tmdb_id IN ({ph}) AND mc.role = 'cast' AND mc.cast_order < 4",
            tmdb_ids,
        ),
        "actor",
    )

    for r in fetch_all(
        conn,
        f"SELECT tmdb_id, year, original_language, runtime, collection_name FROM movies WHERE tmdb_id IN ({ph})",
        tmdb_ids,
    ):
        f = facets[r["tmdb_id"]]
        if r["year"]:
            f["decade"].append(f"{(r['year'] // 10) * 10}s")
        if r["original_language"]:
            f["language"].append(r["original_language"])
        if r["runtime"]:
            f["runtime"].append(runtime_bucket(r["runtime"]))
        if r["collection_name"]:
            f["collection"].append(r["collection_name"])
    return {k: dict(v) for k, v in facets.items()}


def runtime_bucket(minutes: int | None) -> str:
    if not minutes:
        return "unknown"
    if minutes < 85:
        return "under-85"
    if minutes < 105:
        return "85-105"
    if minutes < 130:
        return "105-130"
    if minutes < 160:
        return "130-160"
    return "over-160"


AffinityStats = dict[str, dict[str, tuple[float, int]]]


def compute_affinity_stats(
    prefs: dict[int, float], facets: dict[int, dict[str, list[str]]]
) -> AffinityStats:
    """Per facet value, the (sum of preference, count) that back its affinity.

    Kept separately from the affinities themselves so that training can
    subtract a film's own contribution - see :func:`affinity_value`.
    """
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for tmdb_id, pref in prefs.items():
        for facet, values in facets.get(tmdb_id, {}).items():
            for value in set(values):
                cell = acc[facet][value]
                cell[0] += pref
                cell[1] += 1
    return {f: {v: (c[0], int(c[1])) for v, c in d.items()} for f, d in acc.items()}


def affinity_value(
    stats: AffinityStats, facet: str, value: str, *, exclude_pref: float | None = None
) -> float:
    """Shrunk mean preference for one facet value.

    ``affinity = mean_preference * n / (n + k)`` - an empirical-Bayes shrink
    toward zero, so one lucky hit never outranks a consistent pattern.

    ``exclude_pref`` removes a single film's own contribution. This is what
    makes the training features honest: without it, a director seen exactly
    once has an affinity that is a direct function of that film's rating, and
    the model "predicts" the label by reading it back out of the feature.
    """
    total, n = stats.get(facet, {}).get(value, (0.0, 0))
    if exclude_pref is not None:
        total -= exclude_pref
        n -= 1
    if n <= 0:
        return 0.0
    k = SHRINK_K.get(facet, 3.0)
    return float((total / n) * (n / (n + k)))


def affinities_from_stats(stats: AffinityStats) -> dict[str, dict[str, float]]:
    return {f: {v: affinity_value(stats, f, v) for v in d} for f, d in stats.items()}


def compute_affinities(
    prefs: dict[int, float], facets: dict[int, dict[str, list[str]]]
) -> dict[str, dict[str, float]]:
    """Shrunk mean preference per facet value, over every rated film."""
    return affinities_from_stats(compute_affinity_stats(prefs, facets))


# --------------------------------------------------------------------------- #
# Taste modes
# --------------------------------------------------------------------------- #
def _load_movie_vectors(
    conn: sqlite3.Connection, tmdb_ids: list[int], model: str
) -> dict[int, np.ndarray]:
    if not tmdb_ids:
        return {}
    ph = ",".join("?" for _ in tmdb_ids)
    rows = fetch_all(
        conn,
        f"SELECT entity_id, vector FROM embeddings WHERE entity_type = ? AND model = ? AND entity_id IN ({ph})",
        [MOVIE, model, *[str(i) for i in tmdb_ids]],
    )
    return {int(r["entity_id"]): blob_to_vector(r["vector"]) for r in rows}


def _label_mode(tags: list[str], genres: list[str]) -> str:
    bits = list(tags[:3]) or genres[:2]
    return ", ".join(bits) if bits else "unlabelled"


def build_taste_modes(
    conn: sqlite3.Connection,
    prefs: dict[int, float],
    vectors: dict[int, np.ndarray],
    facets: dict[int, dict[str, list[str]]],
    titles: dict[int, tuple[str, int | None, float]],
) -> list[TasteMode]:
    """Cluster the films the user liked into distinct modes of taste."""
    liked = [(i, p) for i, p in prefs.items() if p >= LIKED_THRESHOLD and i in vectors]
    if len(liked) < MIN_MODE_SIZE * 2:
        # Not enough evidence to split: one mode over everything positive.
        pool = liked or [(i, p) for i, p in prefs.items() if i in vectors]
        if not pool:
            return []
        mat = np.vstack([vectors[i] for i, _ in pool])
        w = np.array([max(0.05, p) for _, p in pool], dtype=np.float32)
        centroid = _normalize((mat * w[:, None]).sum(axis=0))
        return [_make_mode(0, [i for i, _ in pool], centroid, 1.0, facets, titles)]

    from sklearn.cluster import KMeans

    ids = [i for i, _ in liked]
    mat = np.vstack([vectors[i] for i in ids]).astype(np.float32)
    weights = np.array([p for _, p in liked], dtype=np.float64)

    k = int(np.clip(round(len(ids) / 12), 2, 5))
    k = min(k, max(2, len(ids) // MIN_MODE_SIZE))
    km = KMeans(n_clusters=k, n_init=10, random_state=17).fit(mat, sample_weight=weights)

    modes: list[TasteMode] = []
    total = float(weights.sum()) or 1.0
    for c in range(k):
        member_pos = np.flatnonzero(km.labels_ == c)
        if member_pos.size < MIN_MODE_SIZE:
            continue
        member_ids = [ids[p] for p in member_pos]
        w = weights[member_pos]
        centroid = _normalize((mat[member_pos] * w[:, None]).sum(axis=0))
        modes.append(
            _make_mode(len(modes), member_ids, centroid, float(w.sum() / total), facets, titles)
        )

    if not modes:  # every cluster was too small
        centroid = _normalize((mat * weights[:, None]).sum(axis=0))
        modes = [_make_mode(0, ids, centroid, 1.0, facets, titles)]

    # Renormalise weights over the modes we kept.
    kept = sum(m.weight for m in modes) or 1.0
    for m in modes:
        m.weight /= kept
    modes.sort(key=lambda m: -m.weight)
    for idx, m in enumerate(modes):
        m.mode_id = idx
    return modes


def _make_mode(mode_id, member_ids, centroid, weight, facets, titles) -> TasteMode:
    tag_counts: dict[str, int] = defaultdict(int)
    genre_counts: dict[str, int] = defaultdict(int)
    for i in member_ids:
        for t in set(facets.get(i, {}).get("tag", [])):
            tag_counts[t] += 1
        for g in set(facets.get(i, {}).get("genre", [])):
            genre_counts[g] += 1
    top_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])[:6]]
    top_genres = [g for g, _ in sorted(genre_counts.items(), key=lambda x: -x[1])[:4]]
    exemplars = sorted(
        (
            {
                "tmdb_id": i,
                "title": titles.get(i, ("?", None, 0))[0],
                "year": titles.get(i, ("?", None, 0))[1],
                "rating": titles.get(i, ("?", None, 0))[2],
            }
            for i in member_ids
        ),
        key=lambda d: -(d["rating"] or 0),
    )[:5]
    return TasteMode(
        mode_id,
        _label_mode(top_tags, top_genres),
        weight,
        len(member_ids),
        centroid,
        exemplars,
        top_tags,
        top_genres,
        [int(i) for i in member_ids],
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


# --------------------------------------------------------------------------- #
# Review-derived signals
# --------------------------------------------------------------------------- #
def aggregate_review_facts(facts_rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[str]]:
    """Roll structured reviews up into per-category affinity and taste signals."""
    scores: dict[str, list[float]] = defaultdict(list)
    signals: list[tuple[float, str]] = []

    for row in facts_rows:
        facts = row.get("facts") or {}
        weight = float(facts.get("signal_strength") or 0.5)
        for entry in facts.get("liked") or []:
            cat = str(entry.get("category") or "other")
            scores[cat].append(float(entry.get("strength") or 0.5) * weight)
        for entry in facts.get("disliked") or []:
            cat = str(entry.get("category") or "other")
            scores[cat].append(-float(entry.get("strength") or 0.5) * weight)
        for sig in facts.get("taste_signals") or []:
            if isinstance(sig, str) and len(sig) > 12:
                signals.append((weight, sig.strip()))

    affinity = {}
    for cat, points in scores.items():
        n = len(points)
        affinity[cat] = float(np.mean(points) * (n / (n + 2.0)))

    signals.sort(key=lambda t: -t[0])
    seen: set[str] = set()
    unique_signals = []
    for _, sig in signals:
        norm = sig.lower().rstrip(".")
        if norm not in seen:
            seen.add(norm)
            unique_signals.append(sig)
    return affinity, unique_signals[:60]


def scale_preferences(
    prefs: dict[int, float], dossiers: dict[int, dict[str, Any]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Preferred value and importance for each dossier scale.

    The target is the preference-weighted mean of the scale among films the user
    rated; the weight is how strongly the scale correlates with preference,
    which is what tells the ranker whether the user actually cares about it.
    """
    targets: dict[str, float] = {}
    weights: dict[str, float] = {}
    for scale in DOSSIER_SCALES:
        xs, ys = [], []
        for tmdb_id, pref in prefs.items():
            d = dossiers.get(tmdb_id)
            if not d:
                continue
            value = (d.get("scales") or {}).get(scale)
            if value is None:
                continue
            xs.append(float(value))
            ys.append(float(pref))
        if len(xs) < 8:
            continue
        x = np.array(xs)
        y = np.array(ys)
        # Preference-weighted mean, using softmax-ish positive weights.
        w = np.exp(np.clip(y, -2, 2))
        targets[scale] = float((x * w).sum() / w.sum())
        if x.std() > 1e-6 and y.std() > 1e-6:
            r = float(np.corrcoef(x, y)[0, 1])
            weights[scale] = 0.0 if math.isnan(r) else abs(r) * min(1.0, len(xs) / 40.0)
        else:
            weights[scale] = 0.0
    return targets, weights


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_profile_from_prefs(
    conn: sqlite3.Connection,
    prefs: dict[int, float],
    embed_model: str,
    *,
    titles: dict[int, tuple[str, int | None, float]] | None = None,
    review_rows: list[dict[str, Any]] | None = None,
    n_rated: int | None = None,
) -> TasteProfile:
    """Build a taste profile from an explicit set of preferences.

    Factored out of :func:`build_profile` so that cross-validation can rebuild
    the profile from a subset of the ratings. That matters more than it looks:
    the affinities, the taste centroids and the scale targets are all *fitted*
    quantities. Computing them once over every rating and then cross-validating
    the ranker on top inflates the score badly, because each fold's held-out
    films helped build the very signals they are then scored against.
    """
    tmdb_ids = list(prefs.keys())
    facets = load_movie_facets(conn, tmdb_ids)
    vectors = _load_movie_vectors(conn, tmdb_ids, embed_model)
    dossiers = load_dossiers(conn, tmdb_ids)
    titles = titles or {}

    profile = TasteProfile(
        n_rated=n_rated if n_rated is not None else len(prefs),
        n_reviewed=len(review_rows or []),
        embed_model=embed_model,
    )
    profile.affinity_stats = compute_affinity_stats(prefs, facets)
    profile.affinities = affinities_from_stats(profile.affinity_stats)
    profile.modes = build_taste_modes(conn, prefs, vectors, facets, titles)
    profile.scale_targets, profile.scale_weights = scale_preferences(prefs, dossiers)

    if review_rows is not None:
        profile.aspect_affinity, profile.taste_signals = aggregate_review_facts(review_rows)

    dislike_ids = [i for i, p in prefs.items() if p <= DISLIKED_THRESHOLD and i in vectors]
    if len(dislike_ids) >= 3:
        profile.dislike_member_ids = [int(i) for i in dislike_ids]
        profile.dislike_centroid = _normalize(
            np.mean(np.vstack([vectors[i] for i in dislike_ids]), axis=0)
        )
    return profile


def build_profile(
    conn: sqlite3.Connection, backend: EmbeddingBackend, *, store: bool = True
) -> TasteProfile:
    """Compute the full taste profile and persist it as a model artifact."""
    ratings = load_user_ratings(conn)
    prefs, mean, std = preference_scores(ratings)
    titles = {int(r["tmdb_id"]): (r["title"], r["year"], r["rating"]) for r in ratings}

    profile = build_profile_from_prefs(
        conn,
        prefs,
        backend.name,
        titles=titles,
        review_rows=load_review_facts(conn),
        n_rated=len(ratings),
    )
    profile.rating_mean = mean
    profile.rating_std = std

    if store:
        save_profile(conn, profile)
    log.info(
        "taste profile: %d rated, %d modes, %d review signals, %d scales",
        profile.n_rated,
        len(profile.modes),
        len(profile.taste_signals),
        len(profile.scale_targets),
    )
    return profile


def save_profile(conn: sqlite3.Connection, profile: TasteProfile) -> None:
    """Persist the profile plus its mode centroids."""
    from ..db import upsert, vector_to_blob

    payload = profile.to_json()
    row = fetch_all(
        conn,
        "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM model_artifacts WHERE name = 'taste_profile'",
    )
    version = int(row[0]["v"])
    with transaction(conn):
        conn.execute("UPDATE model_artifacts SET is_active = 0 WHERE name = 'taste_profile'")
        conn.execute(
            "INSERT INTO model_artifacts (name, version, payload_json, created_at, is_active) VALUES (?, ?, ?, ?, 1)",
            ("taste_profile", version, json.dumps(payload), utcnow()),
        )
        vecs = [
            {
                "entity_type": "taste",
                "entity_id": f"mode:{m.mode_id}",
                "model": profile.embed_model,
                "dim": int(m.centroid.shape[0]),
                "content_hash": f"v{version}",
                "vector": vector_to_blob(m.centroid),
                "updated_at": utcnow(),
            }
            for m in profile.modes
        ]
        if profile.dislike_centroid is not None:
            vecs.append(
                {
                    "entity_type": "taste",
                    "entity_id": "dislike",
                    "model": profile.embed_model,
                    "dim": int(profile.dislike_centroid.shape[0]),
                    "content_hash": f"v{version}",
                    "vector": vector_to_blob(profile.dislike_centroid),
                    "updated_at": utcnow(),
                }
            )
        upsert(conn, "embeddings", vecs, key=["entity_type", "entity_id", "model"])


def load_profile(conn: sqlite3.Connection, backend_name: str | None = None) -> TasteProfile | None:
    """Load the active taste profile, rehydrating its centroids."""
    rows = fetch_all(
        conn,
        "SELECT payload_json FROM model_artifacts WHERE name = 'taste_profile' AND is_active = 1 ORDER BY version DESC LIMIT 1",
    )
    if not rows:
        return None
    payload = json.loads(rows[0]["payload_json"])
    model = backend_name or payload.get("embed_model") or ""

    profile = TasteProfile(
        n_rated=payload.get("n_rated", 0),
        n_reviewed=payload.get("n_reviewed", 0),
        rating_mean=payload.get("rating_mean", 3.0),
        rating_std=payload.get("rating_std", 1.0),
        affinities=payload.get("affinities", {}),
        aspect_affinity=payload.get("aspect_affinity", {}),
        scale_targets=payload.get("scale_targets", {}),
        scale_weights=payload.get("scale_weights", {}),
        taste_signals=payload.get("taste_signals", []),
        summary=payload.get("summary"),
        embed_model=payload.get("embed_model", ""),
    )
    vec_rows = fetch_all(
        conn,
        "SELECT entity_id, vector FROM embeddings WHERE entity_type = 'taste' AND model = ?",
        (model,),
    )
    centroids = {r["entity_id"]: blob_to_vector(r["vector"]) for r in vec_rows}
    for m in payload.get("modes", []):
        centroid = centroids.get(f"mode:{m['mode_id']}")
        if centroid is None:
            continue
        profile.modes.append(
            TasteMode(
                mode_id=m["mode_id"],
                label=m["label"],
                weight=m["weight"],
                size=m["size"],
                centroid=centroid,
                exemplars=m.get("exemplars", []),
                top_tags=m.get("top_tags", []),
                top_genres=m.get("top_genres", []),
            )
        )
    # Weights are stored rounded, so renormalise: downstream code takes weighted
    # sums over the modes and relies on them summing to exactly one.
    total = sum(m.weight for m in profile.modes)
    if total > 0:
        for m in profile.modes:
            m.weight /= total

    profile.dislike_centroid = centroids.get("dislike")
    return profile
