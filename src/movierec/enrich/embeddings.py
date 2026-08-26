"""Embedding backends, incremental embedding and the in-memory vector store.

Anthropic does not serve an embeddings endpoint, so semantic vectors come from
a local sentence-transformer by default: free, fast on Apple silicon and cheap
enough to recompute the whole catalog whenever the model changes. The backend
is an interface so an API encoder can be swapped in without touching callers.

Everything is content-hashed. Re-running the pipeline only re-encodes films and
reviews whose text actually changed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..config import Config
from ..db import (
    blob_to_vector,
    content_hash,
    fetch_all,
    transaction,
    upsert,
    utcnow,
    vector_to_blob,
)
from ..logging_utils import get_logger
from .documents import load_movie_documents, review_document

log = get_logger("enrich.embeddings")

ProgressFn = Callable[[str, float], None]

# Entity namespaces inside the `embeddings` table.
MOVIE = "movie"
PLOT = "movie_plot"
REVIEW = "review"
TASTE = "taste"

CHUNK_CHARS = 1500


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class EmbeddingBackend(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per input."""

    def encode_long(self, text: str) -> np.ndarray:
        """Encode text longer than the context window by mean-pooling chunks."""
        chunks = _split_chunks(text, CHUNK_CHARS)
        if len(chunks) == 1:
            return self.encode(chunks)[0]
        vecs = self.encode(chunks)
        pooled = vecs.mean(axis=0)
        return _normalize(pooled)


class SentenceTransformerBackend(EmbeddingBackend):
    """Local sentence-transformers encoder (the default)."""

    # BGE models want an instruction prefix on the query side only.
    _QUERY_PREFIXES: ClassVar[dict[str, str]] = {
        "bge": "Represent this sentence for searching relevant passages: "
    }

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Run:  uv pip install -e '.[embed]'\n"
                "Or set MOVIEREC_EMBED_BACKEND=hash to run without it (much lower quality)."
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._prefix = next(
            (p for k, p in self._QUERY_PREFIXES.items() if k in model_name.lower()), ""
        )

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = (
            [f"{self._prefix}{t}" for t in texts] if (is_query and self._prefix) else list(texts)
        )
        vecs = self.model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class HashBackend(EmbeddingBackend):
    """Deterministic offline encoder used by tests and as an emergency fallback.

    Hashed character n-grams projected into a fixed space. It has no semantic
    understanding whatsoever - identical text matches, related text does not -
    but it keeps the whole pipeline runnable with no model download.
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = (text or "").lower().split()
            grams = tokens + [" ".join(tokens[j : j + 2]) for j in range(len(tokens) - 1)]
            for gram in grams:
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"
                )
                out[i, h % self.dim] += 1.0 if h % 2 else -1.0
        return np.vstack([_normalize(row) for row in out]) if len(texts) else out


def make_backend(cfg: Config) -> EmbeddingBackend:
    kind = (cfg.embed_backend or "").strip().lower()
    if kind in {"hash", "test", "offline"}:
        return HashBackend()
    return SentenceTransformerBackend(cfg.embed_model, batch_size=cfg.embed_batch)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _split_chunks(text: str, size: int) -> list[str]:
    text = (text or "").strip()
    if len(text) <= size:
        return [text]
    chunks, current, length = [], [], 0
    for sentence in text.replace("\n", " ").split(". "):
        piece = sentence + ". "
        if length + len(piece) > size and current:
            chunks.append("".join(current).strip())
            current, length = [], 0
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c] or [text[:size]]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _existing_hashes(conn: sqlite3.Connection, entity_type: str, model: str) -> dict[str, str]:
    return {
        r["entity_id"]: r["content_hash"]
        for r in fetch_all(
            conn,
            "SELECT entity_id, content_hash FROM embeddings WHERE entity_type = ? AND model = ?",
            (entity_type, model),
        )
    }


def store_vectors(
    conn: sqlite3.Connection,
    entity_type: str,
    model: str,
    items: list[tuple[str, str, np.ndarray]],
) -> int:
    """Persist ``(entity_id, content_hash, vector)`` triples."""
    if not items:
        return 0
    rows = [
        {
            "entity_type": entity_type,
            "entity_id": eid,
            "model": model,
            "dim": int(vec.shape[0]),
            "content_hash": chash,
            "vector": vector_to_blob(vec),
            "updated_at": utcnow(),
        }
        for eid, chash, vec in items
    ]
    with transaction(conn):
        upsert(conn, "embeddings", rows, key=["entity_type", "entity_id", "model"])
    return len(rows)


def embed_movies(
    conn: sqlite3.Connection,
    backend: EmbeddingBackend,
    *,
    tmdb_ids: Sequence[int] | None = None,
    batch_size: int = 256,
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Embed the profile and plot documents of every catalog film that changed."""
    if tmdb_ids is None:
        ids = [
            r["tmdb_id"]
            for r in fetch_all(
                conn,
                "SELECT tmdb_id FROM movies WHERE in_catalog = 1 AND detail_level = 2 ORDER BY COALESCE(tmdb_vote_count,0) DESC",
            )
        ]
    else:
        ids = list(tmdb_ids)
    if not ids:
        return {"movies": 0, "plots": 0, "skipped": 0}

    have_profile = _existing_hashes(conn, MOVIE, backend.name)
    have_plot = _existing_hashes(conn, PLOT, backend.name)
    lo, hi = progress_span
    n_profile = n_plot = skipped = 0

    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        docs = load_movie_documents(conn, batch)

        profile_todo: list[tuple[str, str, str]] = []  # (id, hash, text)
        plot_todo: list[tuple[str, str, str]] = []
        for tmdb_id, doc in docs.items():
            key = str(tmdb_id)
            if doc.profile:
                h = content_hash(doc.profile)
                if have_profile.get(key) != h:
                    profile_todo.append((key, h, doc.profile))
                else:
                    skipped += 1
            if doc.plot:
                h = content_hash(doc.plot)
                if have_plot.get(key) != h:
                    plot_todo.append((key, h, doc.plot))

        if profile_todo:
            vecs = backend.encode([t for _, _, t in profile_todo])
            n_profile += store_vectors(
                conn, MOVIE, backend.name, [(i, h, v) for (i, h, _), v in zip(profile_todo, vecs)]
            )
        if plot_todo:
            # Plots routinely exceed the context window, so pool over chunks.
            vecs = [backend.encode_long(t) for _, _, t in plot_todo]
            n_plot += store_vectors(
                conn, PLOT, backend.name, [(i, h, v) for (i, h, _), v in zip(plot_todo, vecs)]
            )

        if progress:
            frac = (start + len(batch)) / len(ids)
            progress(
                f"Embedding films · {start + len(batch):,}/{len(ids):,}", lo + (hi - lo) * frac
            )

    log.info("embedded %d profiles, %d plots (%d unchanged)", n_profile, n_plot, skipped)
    return {"movies": n_profile, "plots": n_plot, "skipped": skipped}


def embed_reviews(
    conn: sqlite3.Connection,
    backend: EmbeddingBackend,
    *,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Embed the user's own reviews - the most direct statement of taste we have."""
    rows = fetch_all(
        conn,
        """SELECT r.review_uri, r.review_text, r.text_hash, r.rating, f.title, f.year
           FROM user_reviews r JOIN user_films f USING(film_key)""",
    )
    if not rows:
        return {"reviews": 0}
    have = _existing_hashes(conn, REVIEW, backend.name)
    todo = [r for r in rows if have.get(r["review_uri"]) != r["text_hash"]]
    if not todo:
        return {"reviews": 0, "skipped": len(rows)}

    texts = [review_document(r["title"], r["year"], r["rating"], r["review_text"]) for r in todo]
    vecs = backend.encode(texts)
    n = store_vectors(
        conn,
        REVIEW,
        backend.name,
        [(r["review_uri"], r["text_hash"], v) for r, v in zip(todo, vecs)],
    )
    if progress:
        progress(f"Embedded {n} reviews", 1.0)
    log.info("embedded %d reviews (%d unchanged)", n, len(rows) - len(todo))
    return {"reviews": n, "skipped": len(rows) - len(todo)}


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
@dataclass
class VectorStore:
    """Dense matrix of unit vectors with an id index. Brute-force cosine kNN.

    A 30k x 384 matrix is 46 MB and a full similarity sweep is a single matmul
    of a few milliseconds, so an ANN index would add dependencies and failure
    modes for no measurable gain at this scale.
    """

    ids: np.ndarray
    matrix: np.ndarray
    model: str
    entity_type: str

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @classmethod
    def load(cls, conn: sqlite3.Connection, entity_type: str, model: str) -> VectorStore:
        rows = fetch_all(
            conn,
            "SELECT entity_id, vector FROM embeddings WHERE entity_type = ? AND model = ? ORDER BY entity_id",
            (entity_type, model),
        )
        if not rows:
            return cls(
                np.array([], dtype=object), np.zeros((0, 0), dtype=np.float32), model, entity_type
            )
        ids = np.array([r["entity_id"] for r in rows], dtype=object)
        matrix = np.vstack([blob_to_vector(r["vector"]) for r in rows]).astype(np.float32)
        return cls(ids, matrix, model, entity_type)

    def index_of(self, entity_id: str) -> int | None:
        hits = np.flatnonzero(self.ids == entity_id)
        return int(hits[0]) if hits.size else None

    def vector(self, entity_id: str) -> np.ndarray | None:
        idx = self.index_of(entity_id)
        return self.matrix[idx] if idx is not None else None

    def vectors_for(self, entity_ids: Sequence[str]) -> tuple[list[str], np.ndarray]:
        lookup = {eid: i for i, eid in enumerate(self.ids)}
        idx = [(e, lookup[e]) for e in entity_ids if e in lookup]
        if not idx:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        return [e for e, _ in idx], self.matrix[[i for _, i in idx]]

    def similarity(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query vector against every stored vector."""
        if not len(self):
            return np.zeros(0, dtype=np.float32)
        return self.matrix @ _normalize(query)

    def search(
        self,
        query: np.ndarray,
        k: int = 50,
        *,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not len(self):
            return []
        scores = self.similarity(query)
        if exclude:
            mask = np.array([eid in exclude for eid in self.ids])
            scores = np.where(mask, -np.inf, scores)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(str(self.ids[i]), float(scores[i])) for i in top]
