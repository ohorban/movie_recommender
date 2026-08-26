"""TMDB ingestion: catalog discovery, full detail fetch and title search.

TMDB is the backbone of the catalog. Discovery walks year by year to build the
universe of candidate films; the detail pass then enriches each one with
keywords, credits and audience reviews, all of which feed the embedding text.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from ..config import Config
from ..db import content_hash, fetch_all, insert_ignore, scalar, transaction, upsert, utcnow
from ..http_cache import CachedSession
from ..logging_utils import get_logger
from ..text_utils import clean_ws, parse_year, truncate

log = get_logger("ingest.tmdb")

API = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"

# How many top-billed cast members and how much audience-review text to keep.
MAX_CAST = 12
MAX_REVIEWS = 3
MAX_REVIEW_CHARS = 4000

ProgressFn = Callable[[str, float], None]


class TMDBError(RuntimeError):
    pass


class TMDBClient:
    def __init__(self, api_key: str, session: CachedSession) -> None:
        if not api_key:
            raise TMDBError("TMDB_API_KEY is not set. Add it to your .env file.")
        self.api_key = api_key
        self.session = session

    def _get(self, path: str, params: dict[str, Any] | None = None, **kw: Any) -> Any:
        params = dict(params or {})
        # Keep the key out of the cache key so a rotated key does not invalidate the cache.
        cache_params = dict(params)
        cache_params["__path"] = path
        params["api_key"] = self.api_key
        return self.session.get_json(f"{API}{path}", params, cache_key_params=cache_params, **kw)

    # ------------------------------------------------------------------ api
    def configuration(self) -> dict[str, Any]:
        return self._get("/configuration", max_age_days=30) or {}

    def discover(
        self, page: int, *, year: int | None = None, min_votes: int = 50
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": page,
            "sort_by": "vote_count.desc",
            "include_adult": "false",
            "include_video": "false",
            "vote_count.gte": min_votes,
        }
        if year is not None:
            params["primary_release_year"] = year
        return self._get("/discover/movie", params, max_age_days=7) or {}

    def movie_detail(self, tmdb_id: int) -> dict[str, Any] | None:
        return self._get(
            f"/movie/{tmdb_id}",
            {"append_to_response": "keywords,credits,reviews,external_ids", "language": "en-US"},
        )

    def search(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": title, "include_adult": "false"}
        if year:
            params["year"] = year
        payload = self._get("/search/movie", params, max_age_days=90) or {}
        return payload.get("results", []) or []

    def find_by_imdb(self, imdb_id: str) -> list[dict[str, Any]]:
        payload = (
            self._get(f"/find/{imdb_id}", {"external_source": "imdb_id"}, max_age_days=90) or {}
        )
        return payload.get("movie_results", []) or []


# --------------------------------------------------------------------------- #
# Row shaping
# --------------------------------------------------------------------------- #
def _summary_row(item: dict[str, Any], origin: str) -> dict[str, Any]:
    return {
        "tmdb_id": item["id"],
        "title": clean_ws(item.get("title") or item.get("original_title") or ""),
        "original_title": clean_ws(item.get("original_title")),
        "year": parse_year(item.get("release_date")),
        "release_date": item.get("release_date") or None,
        "original_language": item.get("original_language"),
        "overview": clean_ws(item.get("overview")) or None,
        "poster_path": item.get("poster_path"),
        "backdrop_path": item.get("backdrop_path"),
        "adult": 1 if item.get("adult") else 0,
        "tmdb_popularity": item.get("popularity"),
        "tmdb_vote_average": item.get("vote_average"),
        "tmdb_vote_count": item.get("vote_count"),
        "origin": origin,
        "detail_level": 1,
        "in_catalog": 1,
        "updated_at": utcnow(),
    }


def _detail_row(d: dict[str, Any]) -> dict[str, Any]:
    collection = d.get("belongs_to_collection") or {}
    countries = ",".join(c.get("iso_3166_1", "") for c in (d.get("production_countries") or []))
    return {
        "tmdb_id": d["id"],
        "imdb_id": (d.get("imdb_id") or (d.get("external_ids") or {}).get("imdb_id")) or None,
        "title": clean_ws(d.get("title") or d.get("original_title") or ""),
        "original_title": clean_ws(d.get("original_title")),
        "year": parse_year(d.get("release_date")),
        "release_date": d.get("release_date") or None,
        "runtime": d.get("runtime") or None,
        "original_language": d.get("original_language"),
        "production_countries": countries or None,
        "overview": clean_ws(d.get("overview")) or None,
        "tagline": clean_ws(d.get("tagline")) or None,
        "poster_path": d.get("poster_path"),
        "backdrop_path": d.get("backdrop_path"),
        "homepage": d.get("homepage") or None,
        "adult": 1 if d.get("adult") else 0,
        "status": d.get("status"),
        "budget": d.get("budget") or None,
        "revenue": d.get("revenue") or None,
        "tmdb_popularity": d.get("popularity"),
        "tmdb_vote_average": d.get("vote_average"),
        "tmdb_vote_count": d.get("vote_count"),
        "collection_id": collection.get("id"),
        "collection_name": clean_ws(collection.get("name")) or None,
        "detail_level": 2,
        "detail_fetched_at": utcnow(),
        "updated_at": utcnow(),
    }


def _write_detail(conn: sqlite3.Connection, detail: dict[str, Any]) -> None:
    """Persist one full TMDB detail payload across every related table."""
    tmdb_id = detail["id"]
    upsert(conn, "movies", [_detail_row(detail)], key=["tmdb_id"])

    genres = detail.get("genres") or []
    insert_ignore(conn, "genres", [{"genre_id": g["id"], "name": g["name"]} for g in genres])
    conn.execute("DELETE FROM movie_genres WHERE tmdb_id = ?", (tmdb_id,))
    insert_ignore(conn, "movie_genres", [{"tmdb_id": tmdb_id, "genre_id": g["id"]} for g in genres])

    keywords = ((detail.get("keywords") or {}).get("keywords")) or []
    insert_ignore(conn, "keywords", [{"keyword_id": k["id"], "name": k["name"]} for k in keywords])
    conn.execute("DELETE FROM movie_keywords WHERE tmdb_id = ?", (tmdb_id,))
    insert_ignore(
        conn, "movie_keywords", [{"tmdb_id": tmdb_id, "keyword_id": k["id"]} for k in keywords]
    )

    credits = detail.get("credits") or {}
    cast = (credits.get("cast") or [])[:MAX_CAST]
    crew = [
        c
        for c in (credits.get("crew") or [])
        if c.get("job")
        in {
            "Director",
            "Screenplay",
            "Writer",
            "Story",
            "Original Music Composer",
            "Director of Photography",
        }
    ]
    people = {p["id"]: p for p in [*cast, *crew]}
    insert_ignore(
        conn,
        "people",
        [
            {"person_id": pid, "name": clean_ws(p.get("name")), "popularity": p.get("popularity")}
            for pid, p in people.items()
        ],
    )
    conn.execute("DELETE FROM movie_credits WHERE tmdb_id = ?", (tmdb_id,))
    credit_rows = [
        {
            "tmdb_id": tmdb_id,
            "person_id": c["id"],
            "role": "cast",
            "job": "",
            "character": clean_ws(c.get("character")),
            "cast_order": c.get("order"),
        }
        for c in cast
    ] + [
        {
            "tmdb_id": tmdb_id,
            "person_id": c["id"],
            "role": "crew",
            "job": c.get("job") or "",
            "character": None,
            "cast_order": None,
        }
        for c in crew
    ]
    insert_ignore(conn, "movie_credits", credit_rows)

    # Audience reviews: real critical prose, a genuinely different signal from
    # the marketing-flavoured overview.
    reviews = ((detail.get("reviews") or {}).get("results")) or []
    chunks = []
    for r in reviews[:MAX_REVIEWS]:
        body = clean_ws(r.get("content"))
        if len(body) < 120:
            continue
        chunks.append(truncate(body, MAX_REVIEW_CHARS // max(1, MAX_REVIEWS)))
    if chunks:
        blob = "\n\n---\n\n".join(chunks)
        upsert(
            conn,
            "movie_texts",
            [
                {
                    "tmdb_id": tmdb_id,
                    "source": "tmdb_reviews",
                    "text": blob,
                    "lang": "en",
                    "url": None,
                    "content_hash": content_hash(blob),
                    "fetched_at": utcnow(),
                }
            ],
            key=["tmdb_id", "source"],
        )

    imdb_id = detail.get("imdb_id")
    if imdb_id:
        insert_ignore(
            conn,
            "external_ids",
            [{"namespace": "imdb", "external_id": imdb_id, "tmdb_id": tmdb_id}],
        )


# --------------------------------------------------------------------------- #
# Stage 1: build the candidate universe
# --------------------------------------------------------------------------- #
def build_catalog(
    conn: sqlite3.Connection,
    client: TMDBClient,
    cfg: Config,
    *,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Walk TMDB year by year, newest first, collecting well-attested films."""
    this_year = date.today().year
    years = list(range(this_year, cfg.min_year - 1, -1))
    collected: dict[int, dict[str, Any]] = {}
    pages_fetched = 0

    for idx, year in enumerate(years):
        if len(collected) >= cfg.catalog_size:
            log.info("catalog target reached at year %d", year)
            break
        page = 1
        while page <= 500:
            payload = client.discover(page, year=year, min_votes=cfg.min_votes)
            pages_fetched += 1
            results = payload.get("results") or []
            if not results:
                break
            for item in results:
                if item.get("adult") or not item.get("id"):
                    continue
                if (item.get("vote_count") or 0) < cfg.min_votes:
                    continue
                collected[item["id"]] = _summary_row(item, "discover")
            total_pages = min(int(payload.get("total_pages") or 1), 500)
            if page >= total_pages:
                break
            # results are vote_count-descending, so once we drop below the
            # threshold every later page is below it too.
            if (results[-1].get("vote_count") or 0) < cfg.min_votes:
                break
            page += 1
        if progress and idx % 5 == 0:
            progress(
                f"Discovering catalog · {year} · {len(collected):,} films",
                0.05 + 0.25 * idx / len(years),
            )

    # Keep the strongest `catalog_size` films by vote count.
    ranked = sorted(collected.values(), key=lambda r: r["tmdb_vote_count"] or 0, reverse=True)
    keep = ranked[: cfg.catalog_size]

    with transaction(conn):
        upsert(
            conn,
            "movies",
            keep,
            key=["tmdb_id"],
            update=[
                "title",
                "original_title",
                "year",
                "release_date",
                "original_language",
                "overview",
                "poster_path",
                "backdrop_path",
                "adult",
                "tmdb_popularity",
                "tmdb_vote_average",
                "tmdb_vote_count",
                "in_catalog",
                "updated_at",
            ],
        )
    log.info("catalog: %d discovered, %d kept, %d pages", len(collected), len(keep), pages_fetched)
    return {"discovered": len(collected), "kept": len(keep), "pages": pages_fetched}


# --------------------------------------------------------------------------- #
# Stage 2: full detail
# --------------------------------------------------------------------------- #
def pending_detail_ids(conn: sqlite3.Connection, *, limit: int | None = None) -> list[int]:
    """Catalog films that still lack a full detail fetch, best first."""
    sql = (
        "SELECT tmdb_id FROM movies WHERE in_catalog = 1 AND detail_level < 2 "
        "ORDER BY COALESCE(tmdb_vote_count, 0) DESC"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r["tmdb_id"] for r in fetch_all(conn, sql)]


def fetch_details(
    conn: sqlite3.Connection,
    client: TMDBClient,
    ids: Sequence[int],
    *,
    workers: int = 8,
    batch_size: int = 200,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.3, 0.55),
) -> dict[str, Any]:
    """Fetch full detail for ``ids`` concurrently, writing in ordered batches."""
    ids = list(ids)
    if not ids:
        return {"fetched": 0, "missing": 0}

    fetched = missing = 0
    lo, hi = progress_span
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            details = list(pool.map(_safe_detail(client), batch))
            with transaction(conn):
                for tmdb_id, detail in zip(batch, details):
                    if detail is None:
                        missing += 1
                        conn.execute(
                            "UPDATE movies SET in_catalog = 0, detail_level = 3, updated_at = ? WHERE tmdb_id = ?",
                            (utcnow(), tmdb_id),
                        )
                        continue
                    _write_detail(conn, detail)
                    fetched += 1
            if progress:
                frac = (start + len(batch)) / len(ids)
                progress(
                    f"Fetching film details · {start + len(batch):,}/{len(ids):,}",
                    lo + (hi - lo) * frac,
                )
    log.info("detail fetch: %d ok, %d missing", fetched, missing)
    return {"fetched": fetched, "missing": missing}


def _safe_detail(client: TMDBClient) -> Callable[[int], dict[str, Any] | None]:
    def _inner(tmdb_id: int) -> dict[str, Any] | None:
        try:
            return client.movie_detail(tmdb_id)
        except Exception as exc:  # network hiccup on one film must not kill the run
            log.warning("detail fetch failed for %s: %s", tmdb_id, exc)
            return None

    return _inner


def ensure_movies(
    conn: sqlite3.Connection,
    client: TMDBClient,
    tmdb_ids: Iterable[int],
    *,
    progress: ProgressFn | None = None,
) -> int:
    """Guarantee that specific films exist with full detail (e.g. user history)."""
    wanted = [int(i) for i in tmdb_ids if i]
    if not wanted:
        return 0
    placeholders = ",".join("?" for _ in wanted)
    have = {
        r["tmdb_id"]
        for r in fetch_all(
            conn,
            f"SELECT tmdb_id FROM movies WHERE detail_level = 2 AND tmdb_id IN ({placeholders})",
            wanted,
        )
    }
    todo = [i for i in wanted if i not in have]
    if todo:
        # A film the user actually watched belongs in the catalog whatever its vote count.
        conn.executemany(
            "INSERT INTO movies (tmdb_id, title, origin, in_catalog) VALUES (?, '', 'user', 1) "
            "ON CONFLICT (tmdb_id) DO UPDATE SET in_catalog = 1",
            [(i,) for i in todo],
        )
        fetch_details(conn, client, todo, progress=progress, progress_span=(0.55, 0.6))
    return len(todo)


def catalog_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "movies": scalar(conn, "SELECT COUNT(*) FROM movies WHERE in_catalog = 1", default=0),
        "with_detail": scalar(
            conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2", default=0
        ),
        "keywords": scalar(conn, "SELECT COUNT(*) FROM movie_keywords", default=0),
        "credits": scalar(conn, "SELECT COUNT(*) FROM movie_credits", default=0),
    }
