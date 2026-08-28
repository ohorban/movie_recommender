"""MovieLens ingestion: the tag genome and item-item collaborative filtering.

Two things come from here that nothing else in the pipeline provides:

* The **tag genome** - 1,128 human-authored descriptors ("atmospheric",
  "thought-provoking", "visually stunning", "dystopia") scored 0-1 for every
  well-known film. These are interpretable in a way embeddings are not, so they
  drive both ranking features and the Insights tab.
* **Item-item CF** - "people who liked what you liked also liked X". With only
  a couple of hundred ratings of your own, content similarity alone gets
  narrow fast; CF neighbours pull in films that are genuinely adjacent in
  taste-space but not in description-space.
"""

from __future__ import annotations

import sqlite3
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..db import fetch_all, insert_ignore, transaction
from ..logging_utils import get_logger
from .download import download

log = get_logger("ingest.movielens")

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
NEEDED = ("links.csv", "genome-scores.csv", "genome-tags.csv", "ratings.csv")

# Genome tuning: keep the descriptive tags, drop the long tail of near-zero relevance.
GENOME_MIN_RELEVANCE = 0.40
GENOME_TOP_K = 40

# CF tuning.
LIKE_THRESHOLD = 4.0  # a "like" in MovieLens' 0.5-5 scale
MIN_ITEM_SUPPORT = 25  # ignore films too thinly rated to say anything
NEIGHBORS_PER_ITEM = 40
SHRINKAGE = 20.0  # damps spurious similarity from tiny co-occurrence counts

ProgressFn = Callable[[str, float], None]


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def ensure_dataset(
    external_dir: Path,
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 0.4),
) -> Path:
    """Download and extract ml-25m, keeping only the files we use."""
    target = external_dir / "ml-25m"
    if all((target / name).exists() for name in NEEDED):
        return target

    archive = external_dir / "ml-25m.zip"
    lo, hi = progress_span
    download(
        DATASET_URL,
        archive,
        label="MovieLens 25M",
        progress=progress,
        progress_span=(lo, lo + (hi - lo) * 0.8),
    )

    if progress:
        progress("Extracting MovieLens", lo + (hi - lo) * 0.85)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in NEEDED and not member.endswith("/"):
                out = target / name
                if not out.exists():
                    with zf.open(member) as src, open(out, "wb") as dst:
                        while block := src.read(1 << 20):
                            dst.write(block)
    return target


# --------------------------------------------------------------------------- #
# Crosswalk
# --------------------------------------------------------------------------- #
def load_crosswalk(conn: sqlite3.Connection, ml_dir: Path) -> dict[int, int]:
    """Map MovieLens movieId -> tmdb_id, restricted to films in our catalog."""
    links = pd.read_csv(ml_dir / "links.csv", dtype={"imdbId": str})
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype("int64")

    catalog = {
        r["tmdb_id"] for r in fetch_all(conn, "SELECT tmdb_id FROM movies WHERE in_catalog = 1")
    }
    mapping = {
        int(row.movieId): int(row.tmdbId)
        for row in links.itertuples(index=False)
        if int(row.tmdbId) in catalog
    }
    with transaction(conn):
        insert_ignore(
            conn,
            "external_ids",
            [
                {"namespace": "movielens", "external_id": str(ml_id), "tmdb_id": tmdb_id}
                for ml_id, tmdb_id in mapping.items()
            ],
        )
    log.info("movielens crosswalk: %d of %d links land in the catalog", len(mapping), len(links))
    return mapping


# --------------------------------------------------------------------------- #
# Tag genome
# --------------------------------------------------------------------------- #
def ingest_genome(
    conn: sqlite3.Connection,
    ml_dir: Path,
    crosswalk: dict[int, int],
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.4, 0.6),
) -> dict[str, Any]:
    """Store the top genome tags per film."""
    tags = pd.read_csv(ml_dir / "genome-tags.csv")
    tag_names = dict(zip(tags["tagId"].astype(int), tags["tag"].astype(str)))

    wanted = np.array(sorted(crosswalk.keys()), dtype=np.int64)
    if wanted.size == 0:
        return {"rows": 0}

    lo, hi = progress_span
    mids: list[np.ndarray] = []
    tids: list[np.ndarray] = []
    rels: list[np.ndarray] = []
    seen = 0
    for chunk in pd.read_csv(ml_dir / "genome-scores.csv", chunksize=2_000_000):
        seen += len(chunk)
        m = chunk["movieId"].to_numpy(np.int64)
        keep = np.isin(m, wanted) & (
            chunk["relevance"].to_numpy(np.float32) >= GENOME_MIN_RELEVANCE
        )
        if keep.any():
            mids.append(m[keep])
            tids.append(chunk["tagId"].to_numpy(np.int32)[keep])
            rels.append(chunk["relevance"].to_numpy(np.float32)[keep])
        if progress:
            progress(
                f"Reading tag genome · {seen / 1e6:.1f}M rows",
                lo + (hi - lo) * min(0.85, seen / 16e6),
            )

    if not mids:
        return {"rows": 0}
    movie_ids = np.concatenate(mids)
    tag_ids = np.concatenate(tids)
    relevance = np.concatenate(rels)

    # Top-K per film: sort by (movie asc, relevance desc) then take a per-group prefix.
    order = np.lexsort((-relevance, movie_ids))
    movie_ids, tag_ids, relevance = movie_ids[order], tag_ids[order], relevance[order]
    starts = np.flatnonzero(np.r_[True, movie_ids[1:] != movie_ids[:-1]])
    rank_within = np.arange(movie_ids.size) - np.repeat(
        starts, np.diff(np.r_[starts, movie_ids.size])
    )
    keep = rank_within < GENOME_TOP_K

    rows = [
        {
            "tmdb_id": crosswalk[int(m)],
            "tag": tag_names.get(int(t), str(t)),
            "relevance": round(float(r), 4),
            "source": "movielens_genome",
        }
        for m, t, r in zip(movie_ids[keep], tag_ids[keep], relevance[keep])
    ]
    with transaction(conn):
        conn.execute("DELETE FROM movie_tags WHERE source = 'movielens_genome'")
        insert_ignore(conn, "movie_tags", rows)
    log.info("genome: %d tag rows across %d films", len(rows), len(set(movie_ids[keep].tolist())))
    if progress:
        progress(f"Tag genome stored ({len(rows):,} rows)", hi)
    return {"rows": len(rows), "films": int(np.unique(movie_ids[keep]).size)}


# --------------------------------------------------------------------------- #
# Item-item collaborative filtering
# --------------------------------------------------------------------------- #
def build_cf(
    conn: sqlite3.Connection,
    ml_dir: Path,
    crosswalk: dict[int, int],
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.6, 1.0),
    block_size: int = 512,
) -> dict[str, Any]:
    """Compute shrunk cosine item-item neighbours from MovieLens 'likes'."""
    wanted = np.array(sorted(crosswalk.keys()), dtype=np.int64)
    if wanted.size == 0:
        return {"items": 0, "edges": 0}

    lo, hi = progress_span
    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    seen = 0
    for chunk in pd.read_csv(
        ml_dir / "ratings.csv",
        chunksize=4_000_000,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int64, "rating": np.float32},
    ):
        seen += len(chunk)
        m = chunk["movieId"].to_numpy(np.int64)
        keep = (chunk["rating"].to_numpy(np.float32) >= LIKE_THRESHOLD) & np.isin(m, wanted)
        if keep.any():
            users.append(chunk["userId"].to_numpy(np.int32)[keep])
            items.append(m[keep])
        if progress:
            progress(
                f"Reading MovieLens ratings · {seen / 1e6:.0f}M",
                lo + (hi - lo) * min(0.45, seen / 25e6 * 0.45),
            )

    if not users:
        return {"items": 0, "edges": 0}
    u = np.concatenate(users)
    i = np.concatenate(items)
    del users, items

    # Reindex to contiguous ids.
    uniq_items, item_idx = np.unique(i, return_inverse=True)
    uniq_users, user_idx = np.unique(u, return_inverse=True)
    del u, i

    support = np.bincount(item_idx, minlength=uniq_items.size)
    strong = support >= MIN_ITEM_SUPPORT
    if strong.sum() < 2:
        return {"items": 0, "edges": 0}

    keep_mask = strong[item_idx]
    item_idx, user_idx = item_idx[keep_mask], user_idx[keep_mask]
    remap = -np.ones(uniq_items.size, dtype=np.int64)
    remap[np.flatnonzero(strong)] = np.arange(int(strong.sum()))
    item_idx = remap[item_idx]
    kept_movie_ids = uniq_items[strong]
    n_items = kept_movie_ids.size

    if progress:
        progress(f"Building item matrix · {n_items:,} films", lo + (hi - lo) * 0.5)

    # items x users, binary
    mat = sparse.csr_matrix(
        (np.ones(item_idx.size, dtype=np.float32), (item_idx, user_idx)),
        shape=(n_items, uniq_users.size),
    )
    mat.sum_duplicates()
    mat.data[:] = 1.0
    norms = np.sqrt(np.asarray(mat.sum(axis=1)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0

    edges: list[tuple[int, int, float]] = []
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        # Co-occurrence counts for this block against every item.
        co = (mat[start:stop] @ mat.T).toarray().astype(np.float32)
        cosine = co / (norms[start:stop, None] * norms[None, :])
        # Shrink toward zero when the overlap is small.
        cosine *= co / (co + SHRINKAGE)
        for local in range(stop - start):
            row = cosine[local]
            row[start + local] = 0.0
            k = min(NEIGHBORS_PER_ITEM, row.size - 1)
            top = np.argpartition(-row, k)[:k]
            top = top[row[top] > 0.01]
            src = int(crosswalk[int(kept_movie_ids[start + local])])
            for j in top:
                edges.append((src, int(crosswalk[int(kept_movie_ids[j])]), round(float(row[j]), 5)))
        if progress:
            progress(
                f"Computing taste neighbours · {stop:,}/{n_items:,}",
                lo + (hi - lo) * (0.55 + 0.45 * stop / n_items),
            )

    with transaction(conn):
        conn.execute("DELETE FROM cf_neighbors")
        conn.executemany(
            "INSERT OR REPLACE INTO cf_neighbors (tmdb_id, neighbor_tmdb_id, score) VALUES (?, ?, ?)",
            edges,
        )
    log.info("cf: %d items, %d neighbour edges", n_items, len(edges))
    return {"items": n_items, "edges": len(edges), "users": int(uniq_users.size)}


def ingest_all(
    conn: sqlite3.Connection,
    external_dir: Path,
    *,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Download MovieLens and run both the genome and CF stages."""
    lo, hi = progress_span

    def sub(a: float, b: float) -> tuple[float, float]:
        return (lo + (hi - lo) * a, lo + (hi - lo) * b)

    ml_dir = ensure_dataset(external_dir, progress=progress, progress_span=sub(0.0, 0.35))
    crosswalk = load_crosswalk(conn, ml_dir)
    genome = ingest_genome(
        conn, ml_dir, crosswalk, progress=progress, progress_span=sub(0.35, 0.55)
    )
    cf = build_cf(conn, ml_dir, crosswalk, progress=progress, progress_span=sub(0.55, 1.0))
    return {"crosswalk": len(crosswalk), "genome": genome, "cf": cf}
