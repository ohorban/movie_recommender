"""TMDB's two id namespaces, reconciled into one key.

TMDB numbers movies and television independently: ``/movie/1399`` and
``/tv/1399`` are unrelated titles. Every table in this database is keyed on
``movies.tmdb_id``, so a show is stored under its TMDB id plus
:data:`TV_ID_OFFSET` and the original id is kept alongside it. That keeps a
single integer key across a dozen tables instead of threading a composite
``(id, media_type)`` key through all of them, and TMDB movie ids are six
figures today, so the offset has decades of headroom.

Nothing outside this module should do the arithmetic by hand.
"""

from __future__ import annotations

import re

MOVIE = "movie"
TV = "tv"

TV_ID_OFFSET = 10_000_000

_TMDB_URL = re.compile(r"themoviedb\.org/(movie|tv)/(\d+)", re.I)


def internal_id(media_type: str, source_id: int | str) -> int:
    """The key this database uses for a TMDB id of the given kind."""
    value = int(source_id)
    return value + TV_ID_OFFSET if media_type == TV else value


def split_id(key: int | str) -> tuple[str, int]:
    """Inverse of :func:`internal_id`: ``(media_type, tmdb_id)``."""
    value = int(key)
    return (TV, value - TV_ID_OFFSET) if value >= TV_ID_OFFSET else (MOVIE, value)


def is_tv(key: int | str) -> bool:
    return int(key) >= TV_ID_OFFSET


def tmdb_url(key: int | str) -> str:
    media_type, source = split_id(key)
    return f"https://www.themoviedb.org/{media_type}/{source}"


def parse_tmdb_reference(text: str) -> tuple[str, int] | None:
    """Read a TMDB id out of whatever the user pasted.

    Accepts a full URL (``https://www.themoviedb.org/tv/259265-some-slug``), a
    bare id, or an id prefixed with its kind (``tv 259265``, ``tv/259265``).
    A bare number is read as a movie, which is what TMDB's own URLs imply.
    Returns ``None`` when there is no id in the string at all.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    url = _TMDB_URL.search(raw)
    if url:
        return url.group(1).lower(), int(url.group(2))

    prefixed = re.fullmatch(r"(movie|tv)\s*[/:\-]?\s*(\d+)", raw, re.I)
    if prefixed:
        return prefixed.group(1).lower(), int(prefixed.group(2))

    if re.fullmatch(r"\d+", raw):
        return MOVIE, int(raw)
    return None
