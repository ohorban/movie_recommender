"""Resolve Letterboxd films to TMDB ids.

Letterboxd exports give a title and a year but no external id, so every film
has to be matched by search. Matches are scored rather than guessed at, and
anything below the confidence bar is flagged for review in the UI instead of
being silently attached to the wrong movie.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from ..db import fetch_all, transaction, utcnow
from ..logging_utils import get_logger
from ..media import split_id
from ..text_utils import match_variants, normalize_title, parse_year
from .tmdb import TMDBClient

log = get_logger("ingest.resolve")

# Above AUTO we trust the match outright; between REVIEW and AUTO we accept it
# but flag it; below REVIEW we refuse to guess.
AUTO_ACCEPT = 0.88
REVIEW_FLOOR = 0.62

# How far the two sites may disagree about a release year before an otherwise
# exact title match stops being treated as certainly the same work.
MAX_TITLE_EXACT_DRIFT = 2

# A film TMDB has no record of will not appear because we asked again tomorrow.
# Failed matches are retried, but only after this cooldown, so a routine update
# does not burn search calls re-confirming the same handful of misses.
RETRY_AFTER_DAYS = 14

ProgressFn = Callable[[str, float], None]


@dataclass
class Match:
    tmdb_id: int | None
    confidence: float
    method: str
    candidate_title: str = ""
    candidate_year: int | None = None


def _year_score(want: int | None, got: int | None) -> float:
    if want is None or got is None:
        return 0.55
    delta = abs(want - got)
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.94  # release-year disagreements between sites are routine
    if delta == 2:
        return 0.72  # festival premiere one year, wide release two years later
    return max(0.0, 0.3 - 0.05 * delta)


def _title_score(query: str, candidate: dict[str, Any]) -> tuple[float, bool]:
    """Similarity of a candidate's titles to the query, and whether one is exact.

    The exactness flag is separate because the two say different things. A
    fuzzy score of 0.9 can mean "the same film spelled differently" or "a
    longer title that happens to contain this one" - *Baby Reindeer* scores
    0.9 against *A Baby Reindeer's First Christmas*. Only an exact match on a
    normalised title is evidence of identity.

    Exactness is judged against the *whole* title only. The looser variants
    strip articles and subtitles, and "Top Gun: Maverick" reduced to "Top Gun"
    matches the 1986 film exactly without being it.
    """
    targets = [
        normalize_title(candidate.get("title") or ""),
        normalize_title(candidate.get("original_title") or ""),
    ]
    targets = [t for t in targets if t]
    if not targets:
        return 0.0, False
    whole = normalize_title(query)
    if whole and whole in targets:
        return 1.0, True
    best = 0.0
    for v in match_variants(query):
        for t in targets:
            best = max(best, 1.0 if v == t else fuzz.WRatio(v, t) / 100.0)
    return best, False


def _length_ratio(query: str, candidate: dict[str, Any]) -> float:
    """How much of the candidate's title the query accounts for.

    `fuzz.WRatio` contains a partial-ratio arm, so a short title buried in a
    much longer one scores nearly as well as a true match. Comparing lengths
    is what separates the two.
    """
    q = len(normalize_title(query))
    lengths = [len(normalize_title(candidate.get(k) or "")) for k in ("title", "original_title")]
    lengths = [n for n in lengths if n]
    if not q or not lengths:
        return 1.0
    return min(1.0, q / max(min(lengths), 1))


def score_candidate(title: str, year: int | None, candidate: dict[str, Any]) -> float:
    """Blend title similarity, year proximity and a light popularity prior."""
    t, exact = _title_score(title, candidate)
    y = _year_score(year, parse_year(candidate.get("release_date")))
    votes = candidate.get("vote_count") or 0
    # Popularity only breaks ties between otherwise equal matches.
    pop = min(1.0, (votes / 500.0)) * 0.06

    cand_year = parse_year(candidate.get("release_date"))
    drift = abs(year - cand_year) if (year and cand_year) else 0
    if exact and drift <= MAX_TITLE_EXACT_DRIFT:
        # The title is the same string and the years are close. A one- or
        # two-year disagreement is the two sites dating the same release
        # differently - Letterboxd has Obsession at 2025, TMDB at 2026 - so the
        # year should not be able to sink an otherwise certain match.
        #
        # A larger gap gets no such benefit: same title, decades apart, is what
        # a remake looks like, and those must still be reviewed by hand.
        score = 0.86 * t + 0.11 * y + pop
    else:
        score = 0.68 * t + 0.26 * y + pop
        # A near-match on a title much longer than the query is usually a
        # different work that merely contains these words.
        ratio = _length_ratio(title, candidate)
        if ratio < 0.75:
            score *= 0.55 + 0.45 * ratio
    return round(min(1.0, score), 4)


def _gather(client: TMDBClient, title: str, year: int | None) -> tuple[list[dict[str, Any]], str]:
    """Every plausible candidate for a title, across film and television.

    Letterboxd logs both under one list and gives no hint which is which, so
    both TMDB indexes are searched every time and the candidates compete on one
    scale. Searching film only is what previously matched *Baby Reindeer* to
    *A Baby Reindeer's First Christmas* and left *Chernobyl* on a documentary.
    """
    results = [*client.search(title, year), *client.search_tv(title, year)]
    method = "search+year"
    if not results and year:
        results = [*client.search(title, None), *client.search_tv(title, None)]
        method = "search"
    # Last resort: search the part before a colon (subtitle drift).
    if not results and ":" in title:
        head = title.split(":", 1)[0].strip()
        results = [*client.search(head, year), *client.search_tv(head, year)]
        method = "search-prefix"
    return results, method


def best_match(client: TMDBClient, title: str, year: int | None) -> Match:
    """Search TMDB and pick the best-scoring candidate."""
    try:
        results, method = _gather(client, title, year)
        if not results:
            return Match(None, 0.0, "no-results")
    except Exception as exc:
        log.warning("search failed for %r (%s): %s", title, year, exc)
        return Match(None, 0.0, "error")

    # Candidates are pre-sorted by vote count so that, when scores tie, the
    # better-known title wins rather than whichever index answered first.
    ordered = sorted(results, key=lambda c: -(c.get("vote_count") or 0))[:24]
    scored = sorted(
        ((score_candidate(title, year, c), c) for c in ordered),
        key=lambda t: t[0],
        reverse=True,
    )
    top_score, top = scored[0]
    # A close runner-up means genuine ambiguity: dock confidence so it gets
    # reviewed. Searching two indexes makes this more likely, not less - a film
    # and a series can share a title - so the rule applies across both.
    runner_score = next((s for s, c in scored[1:] if c["id"] != top["id"]), None)
    if runner_score is not None and runner_score > top_score - 0.03:
        top_score -= 0.15
    return Match(
        tmdb_id=int(top["id"]),
        confidence=round(max(0.0, top_score), 4),
        method=f"{method}:{top.get('media_type') or 'movie'}",
        candidate_title=top.get("title") or "",
        candidate_year=parse_year(top.get("release_date")),
    )


def resolve_user_films(
    conn: sqlite3.Connection,
    client: TMDBClient,
    *,
    only_unresolved: bool = True,
    retry_failed: bool = False,
    workers: int = 6,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.6, 0.75),
) -> dict[str, Any]:
    """Attach a tmdb_id to every user film that lacks one."""
    # Manual overrides always win and are applied first.
    overrides = fetch_all(
        conn, "SELECT film_key, tmdb_id FROM title_overrides WHERE tmdb_id IS NOT NULL"
    )
    if overrides:
        conn.executemany(
            "INSERT INTO movies (tmdb_id, title, origin, in_catalog, detail_level, media_type,"
            " source_tmdb_id) VALUES (?, '', 'manual', 1, 0, ?, ?) "
            "ON CONFLICT (tmdb_id) DO UPDATE SET in_catalog = 1",
            [(o["tmdb_id"], *split_id(o["tmdb_id"])) for o in overrides],
        )
        conn.executemany(
            "UPDATE user_films SET tmdb_id = ?, match_confidence = 1.0, match_method = 'override',"
            " needs_review = 0, resolved_at = ? WHERE film_key = ?",
            [(o["tmdb_id"], utcnow(), o["film_key"]) for o in overrides],
        )

    if retry_failed:
        conn.execute("UPDATE user_films SET resolved_at = NULL WHERE tmdb_id IS NULL")
    if only_unresolved:
        where = (
            "WHERE tmdb_id IS NULL AND ("
            "  resolved_at IS NULL"
            f"  OR resolved_at < datetime('now', '-{RETRY_AFTER_DAYS} days'))"
        )
    else:
        where = "WHERE match_method IS NULL OR match_method != 'override'"
    films = fetch_all(
        conn, f"SELECT film_key, title, year FROM user_films {where} ORDER BY film_key"
    )
    if not films:
        return {"attempted": 0, "matched": 0, "flagged": 0, "unmatched": 0}

    log.info("resolving %d films against TMDB", len(films))
    lo, hi = progress_span
    matched = flagged = unmatched = 0
    results: list[tuple[str, Match]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            (f["film_key"], pool.submit(best_match, client, f["title"], f["year"])) for f in films
        ]
        for idx, (key, fut) in enumerate(futures):
            results.append((key, fut.result()))
            if progress and idx % 25 == 0:
                progress(
                    f"Matching your films to TMDB · {idx}/{len(films)}",
                    lo + (hi - lo) * idx / len(films),
                )

    updates = []
    for key, m in results:
        if m.tmdb_id is None or m.confidence < REVIEW_FLOOR:
            unmatched += 1
            updates.append((None, m.confidence, m.method, 1, utcnow(), key))
            continue
        needs_review = 0 if m.confidence >= AUTO_ACCEPT else 1
        if needs_review:
            flagged += 1
        matched += 1
        updates.append((m.tmdb_id, m.confidence, m.method, needs_review, utcnow(), key))

    # user_films.tmdb_id is a foreign key into movies, and a freshly matched id
    # may not be in the catalog yet (an obscure film the user watched). Create a
    # stub first; the detail pass fills it in immediately afterwards.
    new_ids = sorted({u[0] for u in updates if u[0] is not None})
    with transaction(conn):
        if new_ids:
            conn.executemany(
                "INSERT INTO movies (tmdb_id, title, origin, in_catalog, detail_level, media_type,"
                " source_tmdb_id) VALUES (?, '', 'user', 1, 0, ?, ?) "
                "ON CONFLICT (tmdb_id) DO UPDATE SET in_catalog = 1",
                [(i, *split_id(i)) for i in new_ids],
            )
        conn.executemany(
            "UPDATE user_films SET tmdb_id = ?, match_confidence = ?, match_method = ?,"
            " needs_review = ?, resolved_at = ? WHERE film_key = ?",
            updates,
        )
    log.info("resolution: %d matched (%d flagged), %d unmatched", matched, flagged, unmatched)
    return {"attempted": len(films), "matched": matched, "flagged": flagged, "unmatched": unmatched}


def set_override(
    conn: sqlite3.Connection, film_key: str, tmdb_id: int | None, note: str = ""
) -> None:
    """Record a manual correction and apply it immediately."""
    conn.execute(
        "INSERT INTO title_overrides (film_key, tmdb_id, note) VALUES (?, ?, ?) "
        "ON CONFLICT (film_key) DO UPDATE SET tmdb_id = excluded.tmdb_id, note = excluded.note,"
        " created_at = datetime('now')",
        (film_key, tmdb_id, note),
    )
    if tmdb_id:
        conn.execute(
            "INSERT INTO movies (tmdb_id, title, origin, in_catalog, detail_level, media_type,"
            " source_tmdb_id) VALUES (?, '', 'manual', 1, 0, ?, ?) "
            "ON CONFLICT (tmdb_id) DO UPDATE SET in_catalog = 1",
            (tmdb_id, *split_id(tmdb_id)),
        )
    conn.execute(
        "UPDATE user_films SET tmdb_id = ?, match_confidence = 1.0, match_method = 'override',"
        " needs_review = 0, resolved_at = ? WHERE film_key = ?",
        (tmdb_id, utcnow(), film_key),
    )


def unresolved_report(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Films the UI should ask the user to confirm or correct."""
    rows = fetch_all(
        conn,
        """
        SELECT uf.film_key, uf.title, uf.year, uf.tmdb_id, uf.match_confidence, uf.match_method,
               m.title AS matched_title, m.year AS matched_year,
               m.media_type AS matched_media_type,
               (SELECT rating FROM user_ratings r WHERE r.film_key = uf.film_key) AS rating
        FROM user_films uf
        LEFT JOIN movies m ON m.tmdb_id = uf.tmdb_id
        WHERE uf.needs_review = 1 OR uf.tmdb_id IS NULL
        ORDER BY (uf.tmdb_id IS NULL) DESC, uf.match_confidence ASC
        """,
    )
    return [dict(r) for r in rows]
