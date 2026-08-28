"""Turn natural language into structured data with Claude.

Two directions:

* **Reviews → facts.** Each of the user's written reviews becomes a record of
  what they liked, what they objected to, how engaged they were and - most
  usefully - generalisable statements about their taste that transfer to films
  they have never seen.
* **Films → dossiers.** A film's plot, keywords and genome tags become
  calibrated scales (intellectual demand, darkness, feel-good, tension...)
  that both the ranker and the natural-language layer can reason over.

Dossiers are generated lazily: only films that actually reach the ranking stage
earn one, so the cost tracks use rather than catalog size.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

from ..db import content_hash, fetch_all, transaction, upsert, utcnow
from ..logging_utils import get_logger
from ..text_utils import truncate
from .coerce import normalize_dossier, normalize_review_facts
from .documents import load_movie_documents
from .llm import ClaudeClient
from .schemas import DOSSIER_SCHEMA, REVIEW_FACTS_SCHEMA

log = get_logger("enrich.structuring")

DOSSIER_SCALES = [
    "intellectual_demand",
    "emotional_intensity",
    "originality",
    "feel_good",
    "darkness",
    "spectacle",
    "realism",
    "humor",
    "tension",
]

ProgressFn = Callable[[str, float], None]

REVIEW_SYSTEM = """You extract structured preference data from one person's film reviews.

You are building a model of a single viewer's taste, so read for what the review reveals about \
THE REVIEWER, not about the film. A review is evidence about a person.

Rules:
- Work only from what the review says or clearly implies. Never infer from the film's reputation.
- A short review yields little: set signal_strength low rather than inventing detail.
- taste_signals must generalise to films the reviewer has not seen. "Liked the twist" is not a \
taste signal; "rewards films that earn their twists structurally" is.
- Respect the reviewer's own emphasis. If they spend three words on the plot and a sentence on \
how bored they were, pacing is the dominant signal.
- Numeric ratings are on a 0.5-5 scale. Use the rating to calibrate sentiment, but the prose wins \
when the two disagree - a 3/5 with an enthusiastic review means they enjoyed it more than the number suggests."""

DOSSIER_SYSTEM = """You write compact, calibrated profiles of films for a recommendation engine.

Your output is consumed by software, so consistency across films matters more than nuance in any \
one of them. Calibrate the 0-1 scales against the whole of cinema, not against the film's genre:
- intellectual_demand 0.9 is Primer or Last Year at Marienbad; 0.5 is Inception; 0.1 is a broad comedy.
- darkness 0.9 is Come and See; 0.5 is The Dark Knight; 0.1 is Paddington.
- feel_good 0.9 is Paddington 2; 0.5 is a bittersweet drama; 0.1 is Requiem for a Dream.
- originality rates how unusual the premise or execution is, not how good the film is.

Base everything on the supplied material. Do not rely on outside knowledge of the film's \
reputation, awards or box office. Be concrete and avoid marketing language."""


# --------------------------------------------------------------------------- #
# Reviews -> structured facts
# --------------------------------------------------------------------------- #
def pending_reviews(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Reviews with no facts yet, or whose text changed since they were analysed."""
    rows = fetch_all(
        conn,
        """
        SELECT r.review_uri, r.review_text, r.text_hash, r.rating, r.rewatch,
               f.title, f.year, rf.text_hash AS done_hash
        FROM user_reviews r
        JOIN user_films f USING(film_key)
        LEFT JOIN review_facts rf ON rf.review_uri = r.review_uri
        WHERE rf.review_uri IS NULL OR rf.text_hash != r.text_hash
        ORDER BY r.review_date DESC
        """,
    )
    return [dict(r) for r in rows]


def _review_prompt(row: dict[str, Any]) -> str:
    header = f"Film: {row['title']}"
    if row.get("year"):
        header += f" ({row['year']})"
    if row.get("rating") is not None:
        header += f"\nTheir rating: {row['rating']}/5"
    if row.get("rewatch"):
        header += "\nThis was a rewatch."
    return f'{header}\n\nTheir review:\n"""\n{row["review_text"]}\n"""'


def structure_reviews(
    conn: sqlite3.Connection,
    client: ClaudeClient,
    *,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Extract structured taste facts from every new or edited review."""
    todo = pending_reviews(conn)
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"processed": 0, "skipped": 0}

    log.info("structuring %d reviews with %s", len(todo), client.model)
    jobs = [
        {
            "kind": "review_facts",
            "system": REVIEW_SYSTEM,
            "user": _review_prompt(row),
            "schema": REVIEW_FACTS_SCHEMA,
            "tool_name": "record_review_facts",
            "tool_description": "Record the structured preference data extracted from this review.",
            "max_tokens": 1500,
        }
        for row in todo
    ]
    results = client.map_structured(
        jobs, progress=progress, progress_span=progress_span, label="Reading your reviews"
    )

    rows = [
        {
            "review_uri": row["review_uri"],
            "text_hash": row["text_hash"],
            "payload_json": json.dumps(normalize_review_facts(payload)),
            "model": client.model,
            "generated_at": utcnow(),
        }
        for row, payload in zip(todo, results)
        if payload is not None
    ]
    with transaction(conn):
        upsert(conn, "review_facts", rows, key=["review_uri"])
    failed = len(todo) - len(rows)
    log.info("review facts: %d stored, %d failed", len(rows), failed)
    return {"processed": len(rows), "failed": failed, "usage": client.usage_summary()}


def load_review_facts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All structured review facts joined to their film and rating."""
    rows = fetch_all(
        conn,
        """
        SELECT rf.payload_json, r.rating, r.review_text, r.review_date, r.rewatch,
               f.title, f.year, f.film_key, f.tmdb_id
        FROM review_facts rf
        JOIN user_reviews r USING(review_uri)
        JOIN user_films f USING(film_key)
        """,
    )
    out = []
    for r in rows:
        try:
            facts = normalize_review_facts(json.loads(r["payload_json"]))
        except json.JSONDecodeError:
            continue
        fields = {k: r[k] for k in r.keys() if k != "payload_json"}  # noqa: SIM118 (sqlite3.Row)
        out.append({**fields, "facts": facts})
    return out


# --------------------------------------------------------------------------- #
# Films -> dossiers
# --------------------------------------------------------------------------- #
def _dossier_prompt(doc_profile: str, doc_plot: str, meta: dict[str, Any]) -> str:
    parts = [f"FILM PROFILE\n{doc_profile}"]
    if doc_plot and meta.get("has_plot"):
        parts.append(f"PLOT SYNOPSIS\n{truncate(doc_plot, 3500)}")
    return "\n\n".join(parts)


def generate_dossiers(
    conn: sqlite3.Connection,
    client: ClaudeClient,
    tmdb_ids: Sequence[int],
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
    force: bool = False,
) -> dict[str, Any]:
    """Generate dossiers for films that do not have a current one."""
    ids = [int(i) for i in tmdb_ids]
    if not ids:
        return {"generated": 0, "skipped": 0}

    docs = load_movie_documents(conn, ids)
    ph = ",".join("?" for _ in ids)
    have = {
        r["tmdb_id"]: r["input_hash"]
        for r in fetch_all(
            conn, f"SELECT tmdb_id, input_hash FROM movie_dossiers WHERE tmdb_id IN ({ph})", ids
        )
    }

    todo: list[tuple[int, str, str]] = []
    for tmdb_id, doc in docs.items():
        if not doc.profile:
            continue
        prompt = _dossier_prompt(doc.profile, doc.plot, doc.meta)
        ihash = content_hash(prompt, client.model)
        if not force and have.get(tmdb_id) == ihash:
            continue
        todo.append((tmdb_id, ihash, prompt))

    if not todo:
        return {"generated": 0, "skipped": len(ids)}

    jobs = [
        {
            "kind": "dossier",
            "system": DOSSIER_SYSTEM,
            "user": prompt,
            "schema": DOSSIER_SCHEMA,
            "tool_name": "record_film_profile",
            "tool_description": "Record the calibrated profile of this film.",
            "max_tokens": 1200,
        }
        for _, _, prompt in todo
    ]
    results = client.map_structured(
        jobs, progress=progress, progress_span=progress_span, label="Profiling films"
    )

    rows = [
        {
            "tmdb_id": tmdb_id,
            "payload_json": json.dumps(normalize_dossier(payload, DOSSIER_SCALES)),
            "model": client.model,
            "input_hash": ihash,
            "generated_at": utcnow(),
        }
        for (tmdb_id, ihash, _), payload in zip(todo, results)
        if payload is not None
    ]
    with transaction(conn):
        upsert(conn, "movie_dossiers", rows, key=["tmdb_id"])
    log.info("dossiers: %d generated, %d already current", len(rows), len(ids) - len(todo))
    return {
        "generated": len(rows),
        "skipped": len(ids) - len(todo),
        "usage": client.usage_summary(),
    }


def load_dossiers(
    conn: sqlite3.Connection, tmdb_ids: Sequence[int] | None = None
) -> dict[int, dict[str, Any]]:
    if tmdb_ids is not None:
        if not tmdb_ids:
            return {}
        ph = ",".join("?" for _ in tmdb_ids)
        rows = fetch_all(
            conn,
            f"SELECT tmdb_id, payload_json FROM movie_dossiers WHERE tmdb_id IN ({ph})",
            list(tmdb_ids),
        )
    else:
        rows = fetch_all(conn, "SELECT tmdb_id, payload_json FROM movie_dossiers")
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        try:
            out[r["tmdb_id"]] = normalize_dossier(json.loads(r["payload_json"]), DOSSIER_SCALES)
        except json.JSONDecodeError:
            continue
    return out
