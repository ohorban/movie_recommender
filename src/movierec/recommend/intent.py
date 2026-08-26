"""The natural-language intent layer.

A free-text request is turned into a structured query: a rich semantic
paragraph to embed, hard filters to apply, and a weight saying how much the
user's general taste should override the literal request. That last number is
what separates "recommend me something" (lean on taste) from "a Japanese film
about grief under two hours" (lean on the request).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..db import fetch_all
from ..enrich.llm import ClaudeClient
from ..enrich.schemas import INTENT_SCHEMA
from ..logging_utils import get_logger
from ..taste.profile import TasteProfile

log = get_logger("recommend.intent")

INTENT_SYSTEM = """You translate a person's request for a film into a structured search query.

The `semantic_query` field is the important one. It is embedded and matched against film \
descriptions, so write it as a vivid paragraph describing the ideal film - its premise, tone, \
texture and emotional shape - as though describing a real movie. Do not write it as a search \
string or a list of attributes.

Use hard filters (genres, years, runtime, language) only when the user actually constrained those \
things. Over-filtering is the main way this system fails: a filter that removes the right answer \
cannot be recovered later in the pipeline. When in doubt, express the preference in \
`semantic_query` and `target_scales` instead of as a filter.

Set `taste_weight` by how specific the request is:
- "something to watch tonight" -> 0.9
- "something funny" -> 0.7
- "a slow character study set in rural Japan" -> 0.25
- "the 1974 Coppola one about surveillance" -> 0.05

You will be given a summary of this viewer's taste. Use it to interpret vague requests \
("something cosy") in their terms, not in general terms. Never use it to override an explicit request."""


@dataclass
class Intent:
    raw_text: str = ""
    semantic_query: str = ""
    interpretation: str = ""
    include_genres: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    runtime_max: int | None = None
    languages: list[str] = field(default_factory=list)
    novelty: str = "balanced"
    taste_weight: float = 0.7
    target_scales: dict[str, float] = field(default_factory=dict)
    allow_rewatch: bool = False

    @classmethod
    def from_payload(cls, raw: str, payload: dict[str, Any]) -> Intent:
        scales = {
            k: float(v) for k, v in (payload.get("target_scales") or {}).items() if v is not None
        }
        return cls(
            raw_text=raw,
            semantic_query=str(payload.get("semantic_query") or raw),
            interpretation=str(payload.get("interpretation") or ""),
            include_genres=[str(g) for g in payload.get("include_genres") or []],
            exclude_genres=[str(g) for g in payload.get("exclude_genres") or []],
            keywords=[str(k) for k in payload.get("keywords") or []],
            people=[str(p) for p in payload.get("people") or []],
            year_min=payload.get("year_min"),
            year_max=payload.get("year_max"),
            runtime_max=payload.get("runtime_max"),
            languages=[str(x) for x in payload.get("languages") or []],
            novelty=str(payload.get("novelty") or "balanced"),
            taste_weight=float(payload.get("taste_weight", 0.7)),
            target_scales=scales,
            allow_rewatch=bool(payload.get("allow_rewatch", False)),
        )

    @classmethod
    def default(cls, raw: str = "") -> Intent:
        return cls(raw_text=raw, semantic_query=raw, taste_weight=0.95 if not raw else 0.6)

    def has_filters(self) -> bool:
        return bool(
            self.include_genres
            or self.exclude_genres
            or self.year_min
            or self.year_max
            or self.runtime_max
            or self.languages
            or self.people
            or self.keywords
        )


def _taste_brief(profile: TasteProfile) -> str:
    """A compact description of the viewer for the intent prompt."""
    lines = []
    if profile.summary:
        lines.append(profile.summary.get("headline", ""))
        loves = profile.summary.get("loves") or []
        dislikes = profile.summary.get("dislikes") or []
        if loves:
            lines.append("Reliably works for them: " + "; ".join(loves[:6]))
        if dislikes:
            lines.append("Reliably does not: " + "; ".join(dislikes[:5]))
    else:
        for mode in profile.modes[:4]:
            titles = ", ".join(e["title"] for e in mode.exemplars[:3])
            lines.append(f"- A cluster of their favourites around {mode.label}: {titles}")
        if profile.taste_signals:
            lines.append("From their own reviews: " + "; ".join(profile.taste_signals[:8]))
    top_genres = sorted(profile.affinities.get("genre", {}).items(), key=lambda t: -t[1])[:5]
    if top_genres:
        lines.append(
            "Strongest genre affinities: " + ", ".join(f"{g} ({v:+.2f})" for g, v in top_genres)
        )
    return "\n".join(x for x in lines if x)


def parse_intent(
    client: ClaudeClient | None, profile: TasteProfile, text: str, *, use_cache: bool = True
) -> Intent:
    """Parse a free-text request. Falls back to a literal query without an LLM."""
    text = (text or "").strip()
    if not text:
        return Intent.default("")
    if client is None:
        return Intent.default(text)
    try:
        payload = client.structured(
            kind="intent",
            system=INTENT_SYSTEM,
            user=f'THIS VIEWER\'S TASTE\n{_taste_brief(profile)}\n\nTHEIR REQUEST\n"""\n{text}\n"""',
            schema=INTENT_SCHEMA,
            tool_name="build_search_query",
            tool_description="Turn the request into a structured film search query.",
            max_tokens=1400,
            temperature=0.2,
            use_cache=use_cache,
        )
    except Exception as exc:
        log.warning("intent parsing failed, falling back to literal query: %s", exc)
        return Intent.default(text)
    return Intent.from_payload(text, payload)


def apply_hard_filters(conn: sqlite3.Connection, intent: Intent) -> set[int] | None:
    """Resolve an intent's hard constraints into a set of eligible tmdb_ids.

    Returns ``None`` when the intent imposes no constraints, meaning "no
    restriction" rather than "nothing matched".
    """
    clauses = ["m.in_catalog = 1", "m.detail_level = 2", "m.adult = 0"]
    params: list[Any] = []

    if intent.year_min:
        clauses.append("m.year >= ?")
        params.append(int(intent.year_min))
    if intent.year_max:
        clauses.append("m.year <= ?")
        params.append(int(intent.year_max))
    if intent.runtime_max:
        clauses.append("(m.runtime IS NULL OR m.runtime <= ?)")
        params.append(int(intent.runtime_max))
    if intent.languages:
        ph = ",".join("?" for _ in intent.languages)
        clauses.append(f"m.original_language IN ({ph})")
        params.extend([x.lower() for x in intent.languages])
    if intent.include_genres:
        ph = ",".join("?" for _ in intent.include_genres)
        clauses.append(
            f"EXISTS (SELECT 1 FROM movie_genres mg JOIN genres g USING(genre_id) "
            f"WHERE mg.tmdb_id = m.tmdb_id AND g.name IN ({ph}) COLLATE NOCASE)"
        )
        params.extend(intent.include_genres)
    if intent.exclude_genres:
        ph = ",".join("?" for _ in intent.exclude_genres)
        clauses.append(
            f"NOT EXISTS (SELECT 1 FROM movie_genres mg JOIN genres g USING(genre_id) "
            f"WHERE mg.tmdb_id = m.tmdb_id AND g.name IN ({ph}) COLLATE NOCASE)"
        )
        params.extend(intent.exclude_genres)
    if intent.people:
        ph = ",".join("?" for _ in intent.people)
        clauses.append(
            f"EXISTS (SELECT 1 FROM movie_credits mc JOIN people p USING(person_id) "
            f"WHERE mc.tmdb_id = m.tmdb_id AND p.name IN ({ph}) COLLATE NOCASE)"
        )
        params.extend(intent.people)

    if intent.novelty == "obscure":
        clauses.append("COALESCE(m.tmdb_vote_count, 0) < 4000")
    elif intent.novelty == "familiar":
        clauses.append("COALESCE(m.tmdb_vote_count, 0) >= 1500")

    if len(clauses) <= 3 and not intent.people:
        return None

    rows = fetch_all(conn, f"SELECT m.tmdb_id FROM movies m WHERE {' AND '.join(clauses)}", params)
    ids = {int(r["tmdb_id"]) for r in rows}
    log.info("hard filters matched %d films", len(ids))
    # A filter set that eliminates almost everything is more likely to be an
    # over-eager parse than a genuine constraint, so ignore it.
    if len(ids) < 15:
        log.info("filter set too small (%d); ignoring hard filters", len(ids))
        return None
    return ids
