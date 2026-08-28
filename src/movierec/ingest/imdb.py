"""IMDb bulk datasets: an independent quality and popularity prior.

TMDB vote counts skew toward recent and English-language releases. IMDb's
rating file is a useful second opinion, and it joins cleanly because TMDB hands
us the IMDb id for every film.
"""

from __future__ import annotations

import gzip
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ..db import fetch_all, transaction
from ..logging_utils import get_logger
from .download import download

log = get_logger("ingest.imdb")

RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
ProgressFn = Callable[[str, float], None]


def ingest_ratings(
    conn: sqlite3.Connection,
    external_dir: Path,
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
    refresh: bool = False,
) -> dict[str, Any]:
    """Attach IMDb rating and vote count to every film we have an IMDb id for."""
    dest = external_dir / "title.ratings.tsv.gz"
    if refresh and dest.exists():
        dest.unlink()
    lo, hi = progress_span
    download(
        RATINGS_URL,
        dest,
        label="IMDb ratings",
        progress=progress,
        progress_span=(lo, lo + (hi - lo) * 0.6),
    )

    known = {
        r["imdb_id"]: r["tmdb_id"]
        for r in fetch_all(conn, "SELECT imdb_id, tmdb_id FROM movies WHERE imdb_id IS NOT NULL")
    }
    if not known:
        return {"matched": 0, "note": "no IMDb ids in catalog yet"}

    if progress:
        progress("Joining IMDb ratings", lo + (hi - lo) * 0.7)

    updates: list[tuple[float, int, int]] = []
    with gzip.open(dest, "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(
            fh, sep="\t", chunksize=250_000, na_values="\\N", dtype={"tconst": str}
        ):
            hit = chunk[chunk["tconst"].isin(known.keys())]
            for row in hit.itertuples(index=False):
                tmdb_id = known.get(row.tconst)
                if tmdb_id is None:
                    continue
                updates.append((float(row.averageRating), int(row.numVotes), tmdb_id))

    with transaction(conn):
        conn.executemany(
            "UPDATE movies SET imdb_rating = ?, imdb_votes = ? WHERE tmdb_id = ?", updates
        )
    log.info("imdb: attached ratings to %d films", len(updates))
    if progress:
        progress(f"IMDb ratings attached to {len(updates):,} films", hi)
    return {"matched": len(updates), "catalog_with_imdb_id": len(known)}
