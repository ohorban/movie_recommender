"""Claude's prose reading of the taste profile, used by the Insights tab."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..db import fetch_all, transaction, utcnow
from ..enrich.llm import ClaudeClient
from ..enrich.schemas import TASTE_SUMMARY_SCHEMA
from ..logging_utils import get_logger
from .profile import TasteProfile

log = get_logger("taste.summary")

SUMMARY_SYSTEM = """You characterise one person's film taste from their rating history and their own reviews.

Be specific and non-generic. "Likes good movies" is useless; "trusts a film that commits to a \
strange premise and distrusts one that coasts on style" is the level to aim for.

Ground every claim in the evidence given. Quote or paraphrase their own words where you can. If \
the evidence is thin on something, leave it out rather than padding.

Name real tensions in `contradictions` if they exist - people are not consistent, and the \
inconsistencies are usually the interesting part. If their taste is coherent, return an empty array \
rather than inventing conflict.

For `blind_spots`, use the genre and decade coverage data supplied: name territory that is \
well-regarded and adjacent to what they already like but that they have barely touched. Do not \
guess at territory you were not given data about."""


def _evidence(conn: sqlite3.Connection, profile: TasteProfile) -> str:
    parts: list[str] = []

    parts.append(
        f"They have rated {profile.n_rated} films and written {profile.n_reviewed} reviews."
    )
    parts.append(f"Their mean rating is {profile.rating_mean:.2f}/5 (sd {profile.rating_std:.2f}).")

    dist = fetch_all(
        conn, "SELECT rating, COUNT(*) c FROM user_ratings GROUP BY rating ORDER BY rating"
    )
    parts.append("Rating distribution: " + ", ".join(f"{r['rating']}★×{r['c']}" for r in dist))

    parts.append("\nHIGHEST RATED, WITH THEIR REVIEWS")
    for r in fetch_all(
        conn,
        """SELECT f.title, f.year, r.rating, rv.review_text FROM user_ratings r
           JOIN user_films f USING(film_key) LEFT JOIN user_reviews rv USING(film_key)
           WHERE r.rating >= 4.0 ORDER BY r.rating DESC LIMIT 30""",
    ):
        line = f"- {r['title']} ({r['year']}) {r['rating']}★"
        if r["review_text"]:
            line += f': "{r["review_text"][:300]}"'
        parts.append(line)

    parts.append("\nLOWEST RATED, WITH THEIR REVIEWS")
    for r in fetch_all(
        conn,
        """SELECT f.title, f.year, r.rating, rv.review_text FROM user_ratings r
           JOIN user_films f USING(film_key) LEFT JOIN user_reviews rv USING(film_key)
           WHERE r.rating <= 2.0 ORDER BY r.rating ASC LIMIT 25""",
    ):
        line = f"- {r['title']} ({r['year']}) {r['rating']}★"
        if r["review_text"]:
            line += f': "{r["review_text"][:300]}"'
        parts.append(line)

    if profile.taste_signals:
        parts.append("\nPREFERENCE SIGNALS EXTRACTED FROM THEIR REVIEWS")
        parts.extend(f"- {s}" for s in profile.taste_signals[:35])

    if profile.modes:
        parts.append("\nCLUSTERS FOUND IN THEIR FAVOURITES")
        for m in profile.modes:
            titles = ", ".join(e["title"] for e in m.exemplars[:4])
            parts.append(
                f"- {m.size} films ({m.weight:.0%}), themes {', '.join(m.top_tags[:5])}: {titles}"
            )

    genre_aff = sorted(profile.affinities.get("genre", {}).items(), key=lambda t: -t[1])
    if genre_aff:
        parts.append("\nGENRE AFFINITY (shrunk mean preference, +/- in sd units)")
        parts.append(", ".join(f"{g} {v:+.2f}" for g, v in genre_aff[:14]))

    parts.append("\nGENRE COVERAGE (how much they have watched vs what the catalog holds)")
    for r in fetch_all(
        conn,
        """SELECT g.name,
                  COUNT(DISTINCT uf.tmdb_id) AS watched,
                  (SELECT COUNT(*) FROM movie_genres mg2 JOIN movies m2 USING(tmdb_id)
                   WHERE mg2.genre_id = g.genre_id AND m2.in_catalog = 1 AND m2.tmdb_vote_count > 1000) AS catalog
           FROM genres g
           LEFT JOIN movie_genres mg ON mg.genre_id = g.genre_id
           LEFT JOIN user_films uf ON uf.tmdb_id = mg.tmdb_id
           GROUP BY g.genre_id ORDER BY catalog DESC""",
    ):
        parts.append(
            f"- {r['name']}: watched {r['watched'] or 0} of {r['catalog'] or 0} well-known"
        )

    if profile.aspect_affinity:
        top = sorted(profile.aspect_affinity.items(), key=lambda t: -abs(t[1]))[:12]
        parts.append("\nWHAT THEIR REVIEWS PRAISE OR CRITICISE (by aspect, +/-)")
        parts.append(", ".join(f"{k} {v:+.2f}" for k, v in top))

    return "\n".join(parts)


def generate_summary(
    conn: sqlite3.Connection, client: ClaudeClient, profile: TasteProfile, *, force: bool = False
) -> dict[str, Any] | None:
    """Produce and persist the human-readable taste summary."""
    if profile.summary and not force:
        return profile.summary
    try:
        payload = client.structured(
            kind="taste_summary",
            system=SUMMARY_SYSTEM,
            user=_evidence(conn, profile),
            schema=TASTE_SUMMARY_SCHEMA,
            tool_name="record_taste_summary",
            tool_description="Record a characterisation of this viewer's film taste.",
            max_tokens=2000,
            temperature=0.3,
            use_cache=not force,
        )
    except Exception as exc:
        log.warning("taste summary failed: %s", exc)
        return None

    profile.summary = payload
    rows = fetch_all(
        conn,
        "SELECT version, payload_json FROM model_artifacts WHERE name='taste_profile' AND is_active=1 ORDER BY version DESC LIMIT 1",
    )
    if rows:
        stored = json.loads(rows[0]["payload_json"])
        stored["summary"] = payload
        with transaction(conn):
            conn.execute(
                "UPDATE model_artifacts SET payload_json = ?, created_at = ? WHERE name='taste_profile' AND version = ?",
                (json.dumps(stored), utcnow(), rows[0]["version"]),
            )
    log.info("taste summary generated")
    return payload
