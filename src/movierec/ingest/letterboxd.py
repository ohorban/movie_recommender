"""Parse a Letterboxd export folder and merge it into the database.

The user replaces the whole export folder each time, so every run sees the full
history. To keep updates cheap, each row is compared against what is already
stored and only genuine changes are written - which in turn means only new or
edited reviews get re-sent to the LLM and re-embedded downstream.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..db import content_hash, fetch_all, record_source_file, upsert, utcnow
from ..logging_utils import get_logger
from ..text_utils import clean_ws, film_key, parse_year

log = get_logger("ingest.letterboxd")

EXPORT_DIR_RE = re.compile(
    r"^letterboxd-(?P<user>[^-]+)-(?P<stamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2})-utc$"
)

# Sub-folders Letterboxd fills with tombstones for removed content.
IGNORED_SUBDIRS = {"deleted", "orphaned"}


@dataclass
class IngestStats:
    """Per-table counts, surfaced in the Data tab so updates are auditable."""

    export_dir: str = ""
    files_seen: int = 0
    files_changed: int = 0
    films_new: int = 0
    films_total: int = 0
    ratings_new: int = 0
    ratings_changed: int = 0
    watched_new: int = 0
    diary_new: int = 0
    reviews_new: int = 0
    reviews_changed: int = 0
    watchlist_new: int = 0
    watchlist_removed: int = 0
    likes_new: int = 0
    lists_new: int = 0
    comments_new: int = 0
    unchanged: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Locating the export
# --------------------------------------------------------------------------- #
def find_export_dirs(data_dir: Path) -> list[Path]:
    """All Letterboxd export folders under ``data_dir``, newest first."""
    if not data_dir.exists():
        return []
    matches: list[tuple[str, Path]] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        m = EXPORT_DIR_RE.match(child.name)
        if m:
            matches.append((m.group("stamp"), child))
        elif (child / "watched.csv").exists() and (child / "ratings.csv").exists():
            matches.append(("0000-00-00-00-00", child))  # unnamed but valid
    matches.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in matches]


def latest_export_dir(data_dir: Path) -> Path | None:
    dirs = find_export_dirs(data_dir)
    return dirs[0] if dirs else None


# --------------------------------------------------------------------------- #
# CSV reading
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [
            {(k or "").strip(): (v or "") for k, v in row.items()} for row in csv.DictReader(fh)
        ]


def _read_list_csv(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse Letterboxd's two-block list format.

    Block 1 is list metadata, block 2 (after a blank line) is the films. A
    leading ``Letterboxd list export vN`` banner line may precede both.
    """
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw:
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        if line.lower().startswith("letterboxd list export"):
            continue
        current.append(line)
    if current:
        blocks.append(current)

    meta: dict[str, str] = {}
    films: list[dict[str, str]] = []
    for block in blocks:
        rows = list(csv.DictReader(block))
        if not rows:
            continue
        headers = {h.strip() for h in rows[0]}
        if "Position" in headers or "Year" in headers:
            films.extend({(k or "").strip(): (v or "") for k, v in r.items()} for r in rows)
        elif not meta:
            meta = {(k or "").strip(): (v or "") for k, v in rows[0].items()}
    return meta, films


def _iter_export_files(export_dir: Path) -> Iterator[Path]:
    for path in sorted(export_dir.rglob("*.csv")):
        if any(part in IGNORED_SUBDIRS for part in path.relative_to(export_dir).parts):
            continue
        yield path


# --------------------------------------------------------------------------- #
# Film registry
# --------------------------------------------------------------------------- #
class FilmRegistry:
    """Accumulates every (title, year) seen anywhere in the export."""

    def __init__(self) -> None:
        self.films: dict[str, dict[str, Any]] = {}

    def add(self, title: str, year: object, uri: str | None = None) -> str | None:
        title = clean_ws(title)
        if not title:
            return None
        key = film_key(title, year)
        entry = self.films.setdefault(
            key, {"film_key": key, "title": title, "year": parse_year(year), "film_uri": None}
        )
        # Only film URIs (short slugs) are useful; entry URIs are per-log.
        if uri and is_film_uri(uri) and not entry["film_uri"]:
            entry["film_uri"] = uri.strip()
        if entry["year"] is None:
            entry["year"] = parse_year(year)
        return key


def is_film_uri(uri: str) -> bool:
    """Film URIs use short slugs; diary/review entry URIs are noticeably longer."""
    slug = uri.strip().rstrip("/").rsplit("/", 1)[-1]
    return 0 < len(slug) <= 5


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def ingest_export(
    conn: sqlite3.Connection,
    export_dir: Path,
    *,
    data_root: Path | None = None,
    force: bool = False,
) -> IngestStats:
    """Merge one export folder into the database and report what changed."""
    stats = IngestStats(export_dir=export_dir.name)

    files = list(_iter_export_files(export_dir))
    stats.files_seen = len(files)
    changed_files = [p for p in files if record_source_file(conn, p, root=data_root)]
    stats.files_changed = len(changed_files)
    if not changed_files and not force:
        stats.unchanged = True
        stats.notes.append("No export file changed since the last run; nothing to do.")
        log.info("export unchanged, skipping")
        return stats

    registry = FilmRegistry()

    ratings = _read_csv(export_dir / "ratings.csv")
    watched = _read_csv(export_dir / "watched.csv")
    reviews = _read_csv(export_dir / "reviews.csv")
    diary = _read_csv(export_dir / "diary.csv")
    watchlist = _read_csv(export_dir / "watchlist.csv")
    likes = _read_csv(export_dir / "likes" / "films.csv")
    comments = _read_csv(export_dir / "comments.csv")
    profile = _read_csv(export_dir / "profile.csv")

    # ---- pass 1: register every film mentioned anywhere -------------------
    for rows in (watched, ratings, watchlist, likes, reviews, diary):
        for row in rows:
            registry.add(row.get("Name", ""), row.get("Year"), row.get("Letterboxd URI"))

    list_files = (
        sorted((export_dir / "lists").glob("*.csv")) if (export_dir / "lists").exists() else []
    )
    parsed_lists: list[tuple[dict[str, str], list[dict[str, str]]]] = []
    for lf in list_files:
        meta, films = _read_list_csv(lf)
        meta.setdefault("Name", lf.stem)
        parsed_lists.append((meta, films))
        for row in films:
            registry.add(
                row.get("Name", ""), row.get("Year"), row.get("URL") or row.get("Letterboxd URI")
            )

    existing_keys = {r["film_key"] for r in fetch_all(conn, "SELECT film_key FROM user_films")}
    film_rows = list(registry.films.values())
    stats.films_new = sum(1 for f in film_rows if f["film_key"] not in existing_keys)
    stats.films_total = len(film_rows)

    upsert(
        conn,
        "user_films",
        [
            {
                "film_key": f["film_key"],
                "film_uri": f["film_uri"],
                "title": f["title"],
                "year": f["year"],
                "last_seen": utcnow(),
            }
            for f in film_rows
        ],
        key=["film_key"],
        # Never clobber a resolved tmdb_id or a URI we already know.
        update=["title", "year", "last_seen"],
    )
    # Fill in film_uri only where it is currently missing.
    conn.executemany(
        "UPDATE user_films SET film_uri = ? WHERE film_key = ? AND film_uri IS NULL",
        [(f["film_uri"], f["film_key"]) for f in film_rows if f["film_uri"]],
    )

    # ---- ratings ----------------------------------------------------------
    prev_ratings = {
        r["film_key"]: r["rating"]
        for r in fetch_all(conn, "SELECT film_key, rating FROM user_ratings")
    }
    rating_rows = []
    for row in ratings:
        key = film_key(row.get("Name", ""), row.get("Year"))
        value = _to_float(row.get("Rating"))
        if value is None:
            continue
        rating_rows.append(
            {
                "film_key": key,
                "rating": value,
                "rated_date": row.get("Date") or None,
                "updated_at": utcnow(),
            }
        )
        if key not in prev_ratings:
            stats.ratings_new += 1
        elif abs((prev_ratings[key] or 0) - value) > 1e-9:
            stats.ratings_changed += 1
    upsert(conn, "user_ratings", rating_rows, key=["film_key"])

    # ---- watched ----------------------------------------------------------
    prev_watched = {r["film_key"] for r in fetch_all(conn, "SELECT film_key FROM user_watched")}
    watched_rows = []
    for row in watched:
        key = film_key(row.get("Name", ""), row.get("Year"))
        watched_rows.append({"film_key": key, "watched_date": row.get("Date") or None})
        if key not in prev_watched:
            stats.watched_new += 1
    upsert(conn, "user_watched", watched_rows, key=["film_key"])

    # ---- diary ------------------------------------------------------------
    prev_diary = {r["entry_uri"] for r in fetch_all(conn, "SELECT entry_uri FROM user_diary")}
    diary_rows = []
    for row in diary:
        key = film_key(row.get("Name", ""), row.get("Year"))
        uri = (row.get("Letterboxd URI") or "").strip()
        entry_uri = (
            uri or f"synthetic:{content_hash(key, row.get('Watched Date'), row.get('Date'))}"
        )
        diary_rows.append(
            {
                "entry_uri": entry_uri,
                "film_key": key,
                "rating": _to_float(row.get("Rating")),
                "rewatch": 1
                if (row.get("Rewatch") or "").strip().lower() in {"yes", "true", "1"}
                else 0,
                "watched_date": row.get("Watched Date") or None,
                "logged_date": row.get("Date") or None,
                "tags": row.get("Tags") or None,
            }
        )
        if entry_uri not in prev_diary:
            stats.diary_new += 1
    upsert(conn, "user_diary", diary_rows, key=["entry_uri"])

    # ---- reviews ----------------------------------------------------------
    prev_reviews = {
        r["review_uri"]: r["text_hash"]
        for r in fetch_all(conn, "SELECT review_uri, text_hash FROM user_reviews")
    }
    review_rows = []
    for row in reviews:
        text = clean_ws(row.get("Review"))
        if not text:
            continue
        key = film_key(row.get("Name", ""), row.get("Year"))
        uri = (row.get("Letterboxd URI") or "").strip()
        review_uri = uri or f"synthetic:{content_hash(key, text)}"
        thash = content_hash(text)
        review_rows.append(
            {
                "review_uri": review_uri,
                "film_key": key,
                "review_text": text,
                "text_hash": thash,
                "rating": _to_float(row.get("Rating")),
                "rewatch": 1
                if (row.get("Rewatch") or "").strip().lower() in {"yes", "true", "1"}
                else 0,
                "watched_date": row.get("Watched Date") or None,
                "review_date": row.get("Date") or None,
                "tags": row.get("Tags") or None,
                "updated_at": utcnow(),
            }
        )
        if review_uri not in prev_reviews:
            stats.reviews_new += 1
        elif prev_reviews[review_uri] != thash:
            stats.reviews_changed += 1
    upsert(conn, "user_reviews", review_rows, key=["review_uri"])

    # ---- watchlist (a full replace: removals are meaningful) --------------
    prev_wl = {r["film_key"] for r in fetch_all(conn, "SELECT film_key FROM user_watchlist")}
    wl_rows = []
    seen_wl = set()
    for row in watchlist:
        key = film_key(row.get("Name", ""), row.get("Year"))
        seen_wl.add(key)
        wl_rows.append({"film_key": key, "added_date": row.get("Date") or None})
        if key not in prev_wl:
            stats.watchlist_new += 1
    if watchlist or force:
        removed = prev_wl - seen_wl
        stats.watchlist_removed = len(removed)
        if removed:
            conn.executemany(
                "DELETE FROM user_watchlist WHERE film_key = ?", [(k,) for k in removed]
            )
    upsert(conn, "user_watchlist", wl_rows, key=["film_key"])

    # ---- likes ------------------------------------------------------------
    prev_likes = {r["film_key"] for r in fetch_all(conn, "SELECT film_key FROM user_likes")}
    like_rows = []
    for row in likes:
        key = film_key(row.get("Name", ""), row.get("Year"))
        like_rows.append({"film_key": key, "liked_date": row.get("Date") or None})
        if key not in prev_likes:
            stats.likes_new += 1
    upsert(conn, "user_likes", like_rows, key=["film_key"])

    # ---- lists ------------------------------------------------------------
    list_rows = []
    for meta, films in parsed_lists:
        name = clean_ws(meta.get("Name") or "untitled")
        for row in films:
            key = film_key(row.get("Name", ""), row.get("Year"))
            list_rows.append(
                {
                    "list_name": name,
                    "film_key": key,
                    "position": _to_int(row.get("Position")),
                    "notes": row.get("Description") or None,
                    "list_uri": meta.get("URL") or None,
                    "list_date": meta.get("Date") or None,
                }
            )
    prev_lists = {
        (r["list_name"], r["film_key"])
        for r in fetch_all(conn, "SELECT list_name, film_key FROM user_lists")
    }
    stats.lists_new = sum(1 for r in list_rows if (r["list_name"], r["film_key"]) not in prev_lists)
    upsert(conn, "user_lists", list_rows, key=["list_name", "film_key"])

    # ---- comments ---------------------------------------------------------
    prev_comments = {
        r["comment_hash"] for r in fetch_all(conn, "SELECT comment_hash FROM user_comments")
    }
    comment_rows = []
    for row in comments:
        text = clean_ws(row.get("Comment"))
        if not text:
            continue
        chash = content_hash(text)
        comment_rows.append(
            {
                "comment_hash": chash,
                "target_uri": (row.get("Content") or "").strip() or None,
                "comment_text": text,
                "comment_date": row.get("Date") or None,
            }
        )
        if chash not in prev_comments:
            stats.comments_new += 1
    upsert(conn, "user_comments", comment_rows, key=["comment_hash"])

    # ---- profile ----------------------------------------------------------
    if profile:
        upsert(
            conn,
            "user_profile",
            [{"key": k, "value": v, "updated_at": utcnow()} for k, v in profile[0].items() if v],
            key=["key"],
        )

    # Any film whose title changed loses its match and gets re-resolved later.
    log.info(
        "ingested %s: %d films (%d new), %d ratings, %d reviews (%d new/%d edited)",
        export_dir.name,
        stats.films_total,
        stats.films_new,
        len(rating_rows),
        len(review_rows),
        stats.reviews_new,
        stats.reviews_changed,
    )
    return stats


def _to_float(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _to_int(value: object) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None
