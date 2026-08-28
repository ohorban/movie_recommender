"""Compose the natural-language document that represents a film.

Embedding quality is mostly a function of what you feed the encoder. A TMDB
overview alone flattens very different films into the same region of the space,
so each document is assembled from every source we have - credits, genre,
genome tags, keywords, the Wikipedia plot and real audience prose - under an
explicit character budget, because small encoders truncate hard at ~512 tokens.

Two documents are produced per film:

``profile``  identity and texture: what kind of film this is.
``plot``     narrative: what actually happens in it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..db import fetch_all
from ..text_utils import clean_ws, truncate

# Character budgets, tuned for a 512-token encoder (~4 chars/token).
PROFILE_BUDGET = 1800
PLOT_BUDGET = 1800
MAX_TAGS = 14
MAX_KEYWORDS = 14
MAX_CAST = 5


@dataclass
class MovieDocument:
    tmdb_id: int
    profile: str
    plot: str
    meta: dict[str, Any] = field(default_factory=dict)


def _fmt_people(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    directors = [r["name"] for r in rows if r["role"] == "crew" and r["job"] == "Director"]
    cast = [r["name"] for r in rows if r["role"] == "cast"][:MAX_CAST]
    return ", ".join(directors), cast


def load_movie_documents(conn: sqlite3.Connection, tmdb_ids: list[int]) -> dict[int, MovieDocument]:
    """Build documents for the given films in a handful of bulk queries."""
    if not tmdb_ids:
        return {}
    ph = ",".join("?" for _ in tmdb_ids)

    movies = {
        r["tmdb_id"]: dict(r)
        for r in fetch_all(
            conn,
            f"""SELECT tmdb_id, title, original_title, year, runtime, original_language, overview,
                       tagline, production_countries, collection_name
                FROM movies WHERE tmdb_id IN ({ph})""",
            tmdb_ids,
        )
    }
    genres: dict[int, list[str]] = {}
    for r in fetch_all(
        conn,
        f"SELECT mg.tmdb_id, g.name FROM movie_genres mg JOIN genres g USING(genre_id) WHERE mg.tmdb_id IN ({ph})",
        tmdb_ids,
    ):
        genres.setdefault(r["tmdb_id"], []).append(r["name"])

    keywords: dict[int, list[str]] = {}
    for r in fetch_all(
        conn,
        f"SELECT mk.tmdb_id, k.name FROM movie_keywords mk JOIN keywords k USING(keyword_id) WHERE mk.tmdb_id IN ({ph})",
        tmdb_ids,
    ):
        keywords.setdefault(r["tmdb_id"], []).append(r["name"])

    credits: dict[int, list[dict[str, Any]]] = {}
    for r in fetch_all(
        conn,
        f"""SELECT mc.tmdb_id, p.name, mc.role, mc.job, mc.cast_order
            FROM movie_credits mc JOIN people p USING(person_id)
            WHERE mc.tmdb_id IN ({ph}) ORDER BY mc.cast_order IS NULL, mc.cast_order""",
        tmdb_ids,
    ):
        credits.setdefault(r["tmdb_id"], []).append(dict(r))

    tags: dict[int, list[str]] = {}
    for r in fetch_all(
        conn,
        f"""SELECT tmdb_id, tag FROM (
              SELECT tmdb_id, tag, relevance,
                     ROW_NUMBER() OVER (PARTITION BY tmdb_id ORDER BY relevance DESC) rn
              FROM movie_tags WHERE tmdb_id IN ({ph})
            ) WHERE rn <= {MAX_TAGS}""",
        tmdb_ids,
    ):
        tags.setdefault(r["tmdb_id"], []).append(r["tag"])

    texts: dict[int, dict[str, str]] = {}
    for r in fetch_all(
        conn, f"SELECT tmdb_id, source, text FROM movie_texts WHERE tmdb_id IN ({ph})", tmdb_ids
    ):
        texts.setdefault(r["tmdb_id"], {})[r["source"]] = r["text"]

    out: dict[int, MovieDocument] = {}
    for tmdb_id, m in movies.items():
        g = genres.get(tmdb_id, [])
        kws = keywords.get(tmdb_id, [])[:MAX_KEYWORDS]
        tg = tags.get(tmdb_id, [])
        director, cast = _fmt_people(credits.get(tmdb_id, []))
        blobs = texts.get(tmdb_id, {})

        parts: list[str] = []
        title = m["title"] or m["original_title"] or ""
        header = f"{title} ({m['year']})" if m["year"] else title
        if m["original_title"] and m["original_title"] != title:
            header += f" [{m['original_title']}]"
        parts.append(header)
        if director:
            parts.append(f"Directed by {director}.")
        if cast:
            parts.append(f"Starring {', '.join(cast)}.")
        if g:
            parts.append(f"Genre: {', '.join(g)}.")
        facts = []
        if m["runtime"]:
            facts.append(f"{m['runtime']} minutes")
        if m["original_language"] and m["original_language"] != "en":
            facts.append(f"in {m['original_language']}")
        if m["collection_name"]:
            facts.append(f"part of {m['collection_name']}")
        if facts:
            parts.append(f"({'; '.join(facts)}).")
        if m["tagline"]:
            parts.append(f'Tagline: "{clean_ws(m["tagline"])}"')
        if tg:
            parts.append(f"Feels like: {', '.join(tg)}.")
        if kws:
            parts.append(f"Themes and subjects: {', '.join(kws)}.")
        if m["overview"]:
            parts.append(clean_ws(m["overview"]))

        # Audience prose is the most opinionated signal we have; give it the tail
        # of the budget so it survives truncation only when there is room.
        profile = " ".join(parts)
        reviews = blobs.get("tmdb_reviews")
        if reviews and len(profile) < PROFILE_BUDGET - 200:
            profile = f"{profile} What viewers say: {truncate(reviews, PROFILE_BUDGET - len(profile) - 40)}"
        profile = truncate(profile, PROFILE_BUDGET)

        # Only a genuine synopsis earns its own vector. Falling back to the
        # overview here would just re-embed text the profile already contains.
        plot_source = blobs.get("wikipedia_plot") or ""
        plot = truncate(f"{header}. {clean_ws(plot_source)}", PLOT_BUDGET) if plot_source else ""

        out[tmdb_id] = MovieDocument(
            tmdb_id=tmdb_id,
            profile=profile,
            plot=plot,
            meta={
                "title": title,
                "year": m["year"],
                "genres": g,
                "tags": tg,
                "keywords": kws,
                "director": director,
                "cast": cast,
                "has_plot": bool(blobs.get("wikipedia_plot")),
            },
        )
    return out


def review_document(title: str, year: int | None, rating: float | None, text: str) -> str:
    """Frame a user review so the encoder knows it is an opinion about a film."""
    head = f"Review of {title}" + (f" ({year})" if year else "")
    if rating is not None:
        head += f" — rated {rating}/5"
    return truncate(f"{head}. {clean_ws(text)}", 1600)
