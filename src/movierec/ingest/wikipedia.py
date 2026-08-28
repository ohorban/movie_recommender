"""Wikipedia plot synopses - the deepest natural-language source in the system.

A TMDB overview is a two-sentence marketing blurb. A Wikipedia plot section is
several hundred words of actual narrative: who does what, how it escalates and
how it ends. That difference is what lets the embeddings distinguish films that
share a genre but not a shape.

Fetching is tiered - the user's own films and the strongest part of the catalog
are fetched eagerly, everything else on demand - because one request per film
across a 30k catalog is otherwise an hour of politeness.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rapidfuzz import fuzz

from ..db import content_hash, fetch_all, scalar, transaction, upsert, utcnow
from ..http_cache import CachedSession
from ..logging_utils import get_logger
from ..text_utils import clean_ws, normalize_title, truncate

log = get_logger("ingest.wikipedia")

API = "https://en.wikipedia.org/w/api.php"
MAX_PLOT_CHARS = 6000

# Section headings that hold the actual story, in order of preference.
_PLOT_HEADINGS = ("plot", "plot summary", "synopsis", "story", "premise", "plot synopsis")
_SECTION_RE = re.compile(r"^==+\s*(?P<title>[^=]+?)\s*==+\s*$", re.MULTILINE)

ProgressFn = Callable[[str, float], None]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Break a plain-text Wikipedia extract into (heading, body) pairs."""
    sections: list[tuple[str, str]] = []
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group("title").strip().lower(), text[m.end() : end]))
    return sections


def extract_plot(extract: str) -> str:
    """Pull the plot section out of a full-article plain-text extract."""
    if not extract:
        return ""
    sections = _split_sections(extract)
    lookup = dict(sections)
    for heading in _PLOT_HEADINGS:
        body = lookup.get(heading)
        if body and len(body.strip()) > 200:
            return truncate(clean_ws(body), MAX_PLOT_CHARS)
    # Some articles use "Plot (season 1)" style headings.
    for name, body in sections:
        if name.startswith("plot") and len(body.strip()) > 200:
            return truncate(clean_ws(body), MAX_PLOT_CHARS)
    # Fall back to the lead paragraphs, which still beat a TMDB blurb.
    lead = lookup.get("", "")
    return truncate(clean_ws(lead), MAX_PLOT_CHARS) if len(lead.strip()) > 300 else ""


def _plausible_article(title: str, year: int | None, page_title: str, extract: str) -> bool:
    """Guard against the search returning an unrelated article."""
    page_norm = normalize_title(re.sub(r"\s*\([^)]*\)\s*$", "", page_title))
    want = normalize_title(title)
    if fuzz.WRatio(want, page_norm) < 82:
        return False
    head = extract[:1200].lower()
    if "film" not in head and "movie" not in head and "anime" not in head:
        return False
    # The year should appear near the top of a film article; allow ±1 drift.
    if not year:
        return True
    lead = extract[:2000]
    return any(str(y) in lead or str(y) in page_title for y in (year - 1, year, year + 1))


def fetch_plot(session: CachedSession, title: str, year: int | None) -> tuple[str, str] | None:
    """Return ``(plot_text, article_url)`` for a film, or None."""
    query = f'"{title}" {year} film' if year else f'"{title}" film'
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "gsrnamespace": 0,
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": 1,
        "redirects": 1,
    }
    try:
        payload = session.get_json(API, params, max_age_days=180)
    except Exception as exc:
        log.debug("wikipedia lookup failed for %r: %s", title, exc)
        return None
    pages = ((payload or {}).get("query") or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    extract = page.get("extract") or ""
    page_title = page.get("title") or ""
    if not _plausible_article(title, year, page_title, extract):
        return None
    plot = extract_plot(extract)
    if len(plot) < 200:
        return None
    url = "https://en.wikipedia.org/wiki/" + page_title.replace(" ", "_")
    return plot, url


# How long before a film with no confident article is reconsidered. Matches the
# HTTP cache TTL: retrying sooner just replays the same cached response.
MISS_RETRY_DAYS = 180


def plot_coverage(conn: sqlite3.Connection) -> int:
    """How many catalog films already have a stored synopsis."""
    return int(
        scalar(conn, "SELECT COUNT(*) FROM movie_texts WHERE source = 'wikipedia_plot'", default=0)
    )


def pending_plot_ids(conn: sqlite3.Connection, *, target: int) -> list[int]:
    """Films to fetch a synopsis for next.

    ``target`` is a **coverage target**, not a per-run batch size: once this
    many films have a synopsis, the catalog sweep stops. Treating it as a batch
    size instead means every update fetches another full batch and slowly walks
    the entire catalog, hours at a time, which is not what the setting says.

    The user's own films are exempt from the budget - there are only a few
    hundred of them and they are the ones that actually matter - so a newly
    logged film always gets a synopsis even when the catalog target is met.
    """
    exhausted = (
        "AND NOT EXISTS (SELECT 1 FROM enrichment_attempts a "
        " WHERE a.tmdb_id = m.tmdb_id AND a.source = 'wikipedia_plot' "
        f"   AND a.last_attempt >= datetime('now', '-{MISS_RETRY_DAYS} days'))"
    )
    base = (
        "FROM movies m "
        "LEFT JOIN movie_texts t ON t.tmdb_id = m.tmdb_id AND t.source = 'wikipedia_plot' "
        "WHERE m.in_catalog = 1 AND m.detail_level = 2 AND t.tmdb_id IS NULL "
        f"{exhausted}"
    )

    mine = [
        r["tmdb_id"]
        for r in fetch_all(
            conn,
            f"SELECT m.tmdb_id {base} AND EXISTS "
            "(SELECT 1 FROM user_films uf WHERE uf.tmdb_id = m.tmdb_id) "
            "ORDER BY COALESCE(m.tmdb_vote_count, 0) DESC",
        )
    ]

    budget = max(0, int(target) - plot_coverage(conn))
    catalog: list[int] = []
    if budget:
        catalog = [
            r["tmdb_id"]
            for r in fetch_all(
                conn,
                f"SELECT m.tmdb_id {base} AND NOT EXISTS "
                "(SELECT 1 FROM user_films uf WHERE uf.tmdb_id = m.tmdb_id) "
                "ORDER BY COALESCE(m.tmdb_vote_count, 0) DESC LIMIT ?",
                (budget,),
            )
        ]

    log.info(
        "plots: %d stored, target %d -> %d of yours + %d from the catalog",
        plot_coverage(conn),
        target,
        len(mine),
        len(catalog),
    )
    return mine + catalog


def record_miss(conn: sqlite3.Connection, tmdb_id: int, source: str = "wikipedia_plot") -> None:
    conn.execute(
        "INSERT INTO enrichment_attempts (tmdb_id, source, outcome, attempts, last_attempt) "
        "VALUES (?, ?, 'miss', 1, ?) "
        "ON CONFLICT (tmdb_id, source) DO UPDATE SET attempts = attempts + 1, last_attempt = excluded.last_attempt",
        (int(tmdb_id), source, utcnow()),
    )


def fetch_plots(
    conn: sqlite3.Connection,
    session: CachedSession,
    tmdb_ids: Sequence[int],
    *,
    workers: int = 6,
    batch_size: int = 100,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.75, 0.85),
) -> dict[str, Any]:
    """Fetch and store plot synopses for the given films."""
    ids = list(tmdb_ids)
    if not ids:
        return {"fetched": 0, "missed": 0}

    placeholders = ",".join("?" for _ in ids)
    meta = {
        r["tmdb_id"]: (r["title"], r["year"])
        for r in fetch_all(
            conn, f"SELECT tmdb_id, title, year FROM movies WHERE tmdb_id IN ({placeholders})", ids
        )
    }
    fetched = missed = 0
    lo, hi = progress_span

    def _one(tmdb_id: int):
        title, year = meta.get(tmdb_id, ("", None))
        if not title:
            return tmdb_id, None
        return tmdb_id, fetch_plot(session, title, year)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            rows = []
            misses: list[int] = []
            for tmdb_id, result in pool.map(_one, batch):
                if result is None:
                    missed += 1
                    misses.append(tmdb_id)
                    continue
                plot, url = result
                rows.append(
                    {
                        "tmdb_id": tmdb_id,
                        "source": "wikipedia_plot",
                        "text": plot,
                        "lang": "en",
                        "url": url,
                        "content_hash": content_hash(plot),
                        "fetched_at": utcnow(),
                    }
                )
                fetched += 1
            if rows or misses:
                with transaction(conn):
                    if rows:
                        upsert(conn, "movie_texts", rows, key=["tmdb_id", "source"])
                    # Remember the misses so they do not consume the budget on
                    # every future update.
                    for tmdb_id in misses:
                        record_miss(conn, tmdb_id)
            if progress:
                frac = (start + len(batch)) / len(ids)
                progress(
                    f"Fetching plot synopses · {start + len(batch):,}/{len(ids):,}",
                    lo + (hi - lo) * frac,
                )

    log.info("wikipedia: %d plots stored, %d without a confident article", fetched, missed)
    return {"fetched": fetched, "missed": missed}
