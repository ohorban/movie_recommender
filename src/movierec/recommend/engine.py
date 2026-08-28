"""The recommendation engine: retrieve, rank, diversify, explain.

This is the stage that turns a candidate pool into an answer. Ordering is
personalized by the learned ranker, then blended with semantic match to the
request, then diversified so the list is not five variations of one film, then
explained with reference to the user's own history.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ..config import Config
from ..db import fetch_all
from ..enrich.coerce import normalize_pitches
from ..enrich.embeddings import EmbeddingBackend, VectorStore, make_backend
from ..enrich.llm import ClaudeClient
from ..enrich.schemas import PITCH_SCHEMA
from ..enrich.structuring import DOSSIER_SCALES, generate_dossiers, load_dossiers
from ..logging_utils import get_logger
from ..taste.profile import TasteProfile, load_profile, load_user_ratings, preference_scores
from . import candidates as cand
from .features import FeatureBuilder
from .intent import Intent, apply_hard_filters, parse_intent
from .ranker import TasteRanker

log = get_logger("recommend.engine")

MMR_LAMBDA = 0.72  # relevance vs diversity in the final selection
DOSSIER_BUDGET = 24  # how many finalists earn an on-demand Claude dossier

PITCH_SYSTEM = """You write short, honest recommendation notes for one specific person.

You are given their taste profile, films they have rated with their own words, and a shortlist \
the recommender produced. For each film write a hook, a because and a caveat.

Rules:
- The `because` must cite something real from their history by name. "You gave Top Gun: Maverick \
5 and wrote about the problem-solving" is good. "This matches your love of thrillers" is not.
- Never claim they have seen something that is not in the material given to you.
- If a film is a genuine stretch, say so in `because` rather than manufacturing a connection. A \
recommender that admits uncertainty is more useful than one that does not.
- The `caveat` is for real friction - pacing, length, bleakness, subtitles, a divisive ending. \
Leave it empty rather than inventing a flaw.
- No marketing language. No "a masterclass in", no "tour de force", no "you won't believe"."""


@dataclass
class Recommendation:
    tmdb_id: int
    title: str
    year: int | None
    score: float
    rank: int = 0
    sources: list[str] = field(default_factory=list)
    overview: str = ""
    tagline: str = ""
    runtime: int | None = None
    genres: list[str] = field(default_factory=list)
    poster_path: str | None = None
    language: str | None = None
    tmdb_rating: float | None = None
    imdb_rating: float | None = None
    on_watchlist: bool = False
    dossier: dict[str, Any] | None = None
    features: dict[str, float] = field(default_factory=dict)
    hook: str = ""
    because: str = ""
    caveat: str = ""

    @property
    def poster_url(self) -> str | None:
        return f"https://image.tmdb.org/t/p/w342{self.poster_path}" if self.poster_path else None

    @property
    def tmdb_url(self) -> str:
        return f"https://www.themoviedb.org/movie/{self.tmdb_id}"


@dataclass
class RecommendationResult:
    items: list[Recommendation]
    intent: Intent
    pool_size: int = 0
    ranker_kind: str = "heuristic"
    notes: list[str] = field(default_factory=list)


class RecommendationEngine:
    """Holds the loaded model state and answers recommendation requests."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        cfg: Config,
        *,
        backend: EmbeddingBackend | None = None,
        client: ClaudeClient | None = None,
    ) -> None:
        self.conn = conn
        self.cfg = cfg
        self._backend = backend
        self._client = client

    # ------------------------------------------------------------ lazy state
    @cached_property
    def backend(self) -> EmbeddingBackend:
        return self._backend or make_backend(self.cfg)

    @cached_property
    def client(self) -> ClaudeClient | None:
        if self._client is not None:
            return self._client
        if not self.cfg.anthropic_api_key:
            return None
        try:
            return ClaudeClient(self.cfg)
        except Exception as exc:
            log.warning("Claude unavailable: %s", exc)
            return None

    @cached_property
    def profile(self) -> TasteProfile:
        profile = load_profile(self.conn, self.backend.name)
        if profile is None:
            raise RuntimeError(
                "No taste profile has been built yet. Run the update pipeline first."
            )
        return profile

    @cached_property
    def ranker(self) -> TasteRanker:
        return TasteRanker.load(self.conn) or TasteRanker()

    @cached_property
    def store(self) -> VectorStore:
        return cand.load_movie_store(self.conn, self.backend)

    @cached_property
    def _prefs(self) -> dict[int, float]:
        prefs, _, _ = preference_scores(load_user_ratings(self.conn))
        return prefs

    def readiness(self) -> tuple[bool, str]:
        """Whether the engine can answer, and a specific reason when it cannot.

        The model-mismatch case is checked first because it is the one that
        misleads. Embeddings and taste centroids are both keyed by model name,
        so changing MOVIEREC_EMBED_MODEL orphans all of them at once and the
        symptom looks exactly like a database that was never built.
        """
        try:
            profile = self.profile
        except Exception as exc:
            return False, f"No taste profile has been built yet ({exc})."

        if len(self.store) and profile.modes:
            return True, ""

        stored = {
            r["model"]
            for r in fetch_all(
                self.conn, "SELECT DISTINCT model FROM embeddings WHERE entity_type = 'movie'"
            )
        }
        if stored and self.backend.name not in stored:
            return False, (
                f"No embeddings for model '{self.backend.name}'. This database was built with "
                f"{' / '.join(sorted(stored))}. Either set MOVIEREC_EMBED_MODEL back to that, or "
                f"run `movierec update` to re-embed the catalog with the new model."
            )
        if not stored:
            return False, "The catalog has no embeddings yet. Run `movierec setup`."
        if not profile.modes:
            return False, "The taste profile has no modes yet — you may not have enough ratings."
        return False, "The catalog has no embeddings yet. Run `movierec setup`."

    def is_ready(self) -> bool:
        return self.readiness()[0]

    # ------------------------------------------------------------- main flow
    def recommend(
        self,
        text: str = "",
        *,
        n: int = 8,
        exclude: set[int] | None = None,
        explain: bool = True,
        diversify: bool = True,
        allow_seen: bool = False,
        use_llm: bool = True,
    ) -> RecommendationResult:
        notes: list[str] = []
        client = self.client if use_llm else None
        intent = parse_intent(client, self.profile, text) if text else Intent.default("")
        if text and client is None:
            notes.append(
                "Claude is unavailable, so the request was matched literally rather than interpreted."
            )

        blocked = set(exclude or set())
        if not (allow_seen or intent.allow_rewatch):
            blocked |= cand.seen_tmdb_ids(self.conn)
        blocked |= cand.dismissed_tmdb_ids(self.conn)

        hard_ids = apply_hard_filters(self.conn, intent) if intent.has_filters() else None

        query_vec = None
        if intent.semantic_query:
            query_vec = self.backend.encode([intent.semantic_query], is_query=True)[0]

        pool = cand.generate(
            self.conn,
            self.cfg,
            self.profile,
            self.store,
            query_vector=query_vec,
            exclude=blocked,
            hard_filter_ids=hard_ids,
        )
        if not len(pool):
            return RecommendationResult(
                [],
                intent,
                0,
                self.ranker.metrics.model_kind,
                [*notes, "No candidates survived the filters."],
            )

        ids = pool.ids
        dossiers = load_dossiers(self.conn, ids)

        builder = FeatureBuilder(self.conn, self.profile, embed_model=self.backend.name)
        builder.set_reference_prefs(self._prefs)
        fm = builder.build(ids, dossiers)
        taste_scores = self.ranker.score(fm)

        final = self._blend(ids, taste_scores, intent, query_vec, pool, dossiers)

        order = np.argsort(-final)
        shortlist_n = min(len(ids), max(n * 6, 40))
        shortlist = [ids[i] for i in order[:shortlist_n]]
        shortlist_scores = {ids[i]: float(final[i]) for i in order[:shortlist_n]}

        chosen = self._diversify(shortlist, shortlist_scores, n) if diversify else shortlist[:n]

        # Dossiers are generated on demand, so the shortlist gets richer over time.
        if client is not None and use_llm:
            missing = [i for i in chosen if i not in dossiers][:DOSSIER_BUDGET]
            if missing:
                try:
                    generate_dossiers(self.conn, client, missing)
                    dossiers.update(load_dossiers(self.conn, missing))
                except Exception as exc:
                    log.warning("dossier generation failed: %s", exc)

        items = self._hydrate(chosen, shortlist_scores, pool, dossiers, fm)
        if explain and client is not None and items:
            try:
                self._add_pitches(client, items)
            except Exception as exc:
                log.warning("pitch generation failed: %s", exc)
                notes.append("Could not generate explanations for this batch.")

        for rank, item in enumerate(items, 1):
            item.rank = rank
        return RecommendationResult(items, intent, len(pool), self.ranker.metrics.model_kind, notes)

    # ------------------------------------------------------------- internals
    def _blend(
        self,
        ids: list[int],
        taste_scores: np.ndarray,
        intent: Intent,
        query_vec: np.ndarray | None,
        pool: cand.CandidatePool,
        dossiers: dict[int, dict[str, Any]],
    ) -> np.ndarray:
        """Combine personalized score, request match and explicit boosts."""
        taste_z = _zscore(taste_scores)
        w = float(np.clip(intent.taste_weight, 0.0, 1.0))

        if query_vec is not None:
            keys, vecs = self.store.vectors_for([str(i) for i in ids])
            sem = np.zeros(len(ids), dtype=np.float32)
            if vecs.size:
                lookup = dict(zip(keys, (vecs @ _unit(query_vec)).tolist()))
                sem = np.array([lookup.get(str(i), 0.0) for i in ids], dtype=np.float32)
            final = w * taste_z + (1.0 - w) * _zscore(sem)
        else:
            final = taste_z

        # Requested scales (e.g. "nothing bleak") nudge the ordering.
        if intent.target_scales:
            bonus = np.zeros(len(ids), dtype=np.float32)
            for idx, tmdb_id in enumerate(ids):
                d = dossiers.get(tmdb_id)
                if not d:
                    continue
                scales = d.get("scales") or {}
                hits = [
                    1.0 - 2.0 * abs(float(scales[k]) - float(v))
                    for k, v in intent.target_scales.items()
                    if k in DOSSIER_SCALES and scales.get(k) is not None
                ]
                if hits:
                    bonus[idx] = float(np.mean(hits))
            final = final + 0.45 * bonus

        watchlist = cand.watchlist_tmdb_ids(self.conn)
        boost = np.array(
            [self.cfg.watchlist_boost if i in watchlist else 0.0 for i in ids], dtype=np.float32
        )

        # A film proposed by several independent retrieval strategies is a
        # stronger bet than one that only a single source liked.
        agreement = np.array(
            [min(3, len(pool.sources.get(i, ()))) * 0.035 for i in ids], dtype=np.float32
        )
        return final + boost + agreement

    def _diversify(self, ids: list[int], scores: dict[int, float], n: int) -> list[int]:
        """Maximal-marginal-relevance selection over the embedding space."""
        if len(ids) <= n:
            return ids
        keys, vecs = self.store.vectors_for([str(i) for i in ids])
        if not keys:
            return ids[:n]
        index = {int(k): vec for k, vec in zip(keys, vecs)}
        raw = np.array([scores[i] for i in ids], dtype=np.float32)
        norm = {i: float(v) for i, v in zip(ids, _zscore(raw))}

        chosen: list[int] = []
        remaining = list(ids)
        while remaining and len(chosen) < n:
            best_id, best_val = None, -1e9
            for tmdb_id in remaining:
                vec = index.get(tmdb_id)
                penalty = 0.0
                if vec is not None and chosen:
                    sims = [float(vec @ index[c]) for c in chosen if c in index]
                    penalty = max(sims) if sims else 0.0
                value = MMR_LAMBDA * norm[tmdb_id] - (1.0 - MMR_LAMBDA) * penalty * 3.0
                if value > best_val:
                    best_id, best_val = tmdb_id, value
            chosen.append(best_id)  # type: ignore[arg-type]
            remaining.remove(best_id)  # type: ignore[arg-type]
        return chosen

    def _hydrate(
        self,
        ids: list[int],
        scores: dict[int, float],
        pool: cand.CandidatePool,
        dossiers: dict[int, dict[str, Any]],
        fm: Any,
    ) -> list[Recommendation]:
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        rows = {
            r["tmdb_id"]: dict(r)
            for r in fetch_all(
                self.conn,
                f"""SELECT tmdb_id, title, year, overview, tagline, runtime, poster_path,
                           original_language, tmdb_vote_average, imdb_rating
                    FROM movies WHERE tmdb_id IN ({ph})""",
                ids,
            )
        }
        genres: dict[int, list[str]] = {}
        for r in fetch_all(
            self.conn,
            f"SELECT mg.tmdb_id, g.name FROM movie_genres mg JOIN genres g USING(genre_id) WHERE mg.tmdb_id IN ({ph})",
            ids,
        ):
            genres.setdefault(r["tmdb_id"], []).append(r["name"])
        watchlist = cand.watchlist_tmdb_ids(self.conn)

        out = []
        for tmdb_id in ids:
            row = rows.get(tmdb_id)
            if not row:
                continue
            out.append(
                Recommendation(
                    tmdb_id=tmdb_id,
                    title=row["title"],
                    year=row["year"],
                    score=scores.get(tmdb_id, 0.0),
                    sources=sorted(pool.sources.get(tmdb_id, set())),
                    overview=row["overview"] or "",
                    tagline=row["tagline"] or "",
                    runtime=row["runtime"],
                    genres=genres.get(tmdb_id, []),
                    poster_path=row["poster_path"],
                    language=row["original_language"],
                    tmdb_rating=row["tmdb_vote_average"],
                    imdb_rating=row["imdb_rating"],
                    on_watchlist=tmdb_id in watchlist,
                    dossier=dossiers.get(tmdb_id),
                    features=fm.as_dict(tmdb_id) if tmdb_id in fm.ids else {},
                )
            )
        return out

    def _add_pitches(self, client: ClaudeClient, items: list[Recommendation]) -> None:
        """One batched Claude call writes the notes for the whole shortlist."""
        evidence = self._history_evidence()
        lines = []
        for it in items:
            bits = [f"tmdb_id {it.tmdb_id}: {it.title} ({it.year})"]
            if it.genres:
                bits.append(f"  genres: {', '.join(it.genres)}")
            if it.dossier:
                d = it.dossier
                bits.append(f"  logline: {d.get('logline', '')}")
                bits.append(
                    f"  tone: {', '.join(d.get('tone', []))} | themes: {', '.join(d.get('themes', []))}"
                )
                bits.append(f"  who it's for: {d.get('who_its_for', '')}")
                if d.get("avoid_if"):
                    bits.append(f"  avoid if: {d.get('avoid_if')}")
            elif it.overview:
                bits.append(f"  overview: {it.overview[:400]}")
            if it.on_watchlist:
                bits.append("  NOTE: already on their watchlist")
            bits.append(f"  proposed by: {', '.join(it.sources[:4])}")
            lines.append("\n".join(bits))

        payload = client.structured(
            kind="pitches",
            system=PITCH_SYSTEM,
            user=(
                f"THEIR TASTE\n{self._taste_brief()}\n\n"
                f"THEIR OWN WORDS ON FILMS THEY HAVE SEEN\n{evidence}\n\n"
                f"SHORTLIST\n" + "\n\n".join(lines)
            ),
            schema=PITCH_SCHEMA,
            tool_name="write_recommendation_notes",
            tool_description="Write a hook, a because and a caveat for each shortlisted film.",
            max_tokens=400 * max(1, len(items)),
            temperature=0.55,
            use_cache=False,
        )
        by_id = normalize_pitches(payload)
        for it in items:
            pitch = by_id.get(it.tmdb_id) or {}
            it.hook = pitch.get("hook", "")
            it.because = pitch.get("because", "")
            it.caveat = pitch.get("caveat", "")

    def _history_evidence(self, limit: int = 26) -> str:
        """Their strongest opinions, in their own words, for grounding pitches."""
        rows = fetch_all(
            self.conn,
            """
            SELECT f.title, f.year, r.rating, rv.review_text
            FROM user_ratings r
            JOIN user_films f USING(film_key)
            LEFT JOIN user_reviews rv USING(film_key)
            WHERE r.rating >= 4.0 OR r.rating <= 2.0
            ORDER BY ABS(r.rating - 3) DESC, rv.review_text IS NULL
            LIMIT ?
            """,
            (limit,),
        )
        out = []
        for r in rows:
            line = f"- {r['title']} ({r['year']}) — {r['rating']}/5"
            if r["review_text"]:
                line += f': "{r["review_text"][:260]}"'
            out.append(line)
        return "\n".join(out)

    def _taste_brief(self) -> str:
        from .intent import _taste_brief

        return _taste_brief(self.profile)

    # ------------------------------------------------------------- feedback
    def record_feedback(
        self, tmdb_id: int, action: str, surface: str = "", context: dict | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO feedback (tmdb_id, action, surface, context_json) VALUES (?, ?, ?, ?)",
            (int(tmdb_id), action, surface, json.dumps(context or {})),
        )


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    return (values - float(values.mean())) / (std if std > 1e-6 else 1.0)


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec
