"""Embedding backends, document composition, incremental re-embedding, kNN."""

from __future__ import annotations

import numpy as np

from movierec.db import insert_ignore, upsert
from movierec.enrich.documents import load_movie_documents, review_document
from movierec.enrich.embeddings import (
    MOVIE,
    VectorStore,
    _split_chunks,
    embed_movies,
    embed_reviews,
)
from movierec.ingest.letterboxd import ingest_export


def seed_movie(conn, tmdb_id, title="A Film", year=2010, overview="Something happens.", plot=None):
    upsert(
        conn,
        "movies",
        [
            {
                "tmdb_id": tmdb_id,
                "title": title,
                "year": year,
                "overview": overview,
                "runtime": 100,
                "original_language": "en",
                "tagline": "A tagline.",
                "in_catalog": 1,
                "detail_level": 2,
                "tmdb_vote_count": 500,
            }
        ],
        key=["tmdb_id"],
    )
    insert_ignore(conn, "genres", [{"genre_id": 18, "name": "Drama"}])
    insert_ignore(conn, "movie_genres", [{"tmdb_id": tmdb_id, "genre_id": 18}])
    insert_ignore(conn, "keywords", [{"keyword_id": 1, "name": "memory"}])
    insert_ignore(conn, "movie_keywords", [{"tmdb_id": tmdb_id, "keyword_id": 1}])
    insert_ignore(conn, "people", [{"person_id": 9, "name": "Jane Director"}])
    insert_ignore(
        conn,
        "movie_credits",
        [
            {
                "tmdb_id": tmdb_id,
                "person_id": 9,
                "role": "crew",
                "job": "Director",
                "character": None,
                "cast_order": None,
            }
        ],
    )
    insert_ignore(
        conn,
        "movie_tags",
        [
            {
                "tmdb_id": tmdb_id,
                "tag": "atmospheric",
                "relevance": 0.9,
                "source": "movielens_genome",
            }
        ],
    )
    if plot:
        insert_ignore(
            conn,
            "movie_texts",
            [
                {
                    "tmdb_id": tmdb_id,
                    "source": "wikipedia_plot",
                    "text": plot,
                    "lang": "en",
                    "url": None,
                    "content_hash": "h1",
                }
            ],
        )


def test_hash_backend_is_deterministic_and_normalised(backend):
    a = backend.encode(["a tense thriller about memory"])
    b = backend.encode(["a tense thriller about memory"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)
    assert backend.encode([]).shape == (0, backend.dim)


def test_hash_backend_separates_different_text(backend):
    vecs = backend.encode(["a tense thriller about memory", "a sunny romantic comedy in paris"])
    assert float(vecs[0] @ vecs[1]) < 0.5


def test_split_chunks_covers_the_text():
    text = ". ".join(f"Sentence number {i} carries some words" for i in range(200))
    chunks = _split_chunks(text, 400)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks)


def test_encode_long_pools_and_normalises(backend):
    long_text = ". ".join(f"Scene {i} happens in the story" for i in range(400))
    vec = backend.encode_long(long_text)
    assert vec.shape == (backend.dim,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_document_includes_every_source(conn):
    seed_movie(conn, 1, title="Deep Cut", plot="The protagonist descends into the archive. " * 20)
    doc = load_movie_documents(conn, [1])[1]
    for fragment in [
        "Deep Cut (2010)",
        "Jane Director",
        "Drama",
        "atmospheric",
        "memory",
        "A tagline",
    ]:
        assert fragment in doc.profile, fragment
    assert doc.meta["has_plot"] is True
    assert "archive" in doc.plot


def test_no_plot_vector_without_a_real_synopsis(conn):
    """A film with only a TMDB blurb gets no separate plot document.

    Re-embedding the overview as a "plot" would double the storage and add no
    signal, since the profile document already contains it.
    """
    seed_movie(conn, 2, title="Thin Data", overview="A short blurb about nothing much.")
    doc = load_movie_documents(conn, [2])[2]
    assert doc.meta["has_plot"] is False
    assert doc.plot == ""
    assert "short blurb" in doc.profile


def test_review_document_frames_the_opinion():
    out = review_document("Weapons", 2025, 3.5, "Very random and unique.")
    assert "Weapons" in out and "3.5/5" in out and "unique" in out


def test_embed_movies_is_incremental(conn, backend):
    seed_movie(conn, 1, plot="A long plot. " * 40)
    seed_movie(conn, 2)
    first = embed_movies(conn, backend)
    assert first["movies"] == 2
    assert first["plots"] == 1, "only the film with a real synopsis gets a plot vector"

    second = embed_movies(conn, backend)
    assert second["movies"] == 0, "unchanged documents must not be re-embedded"
    assert second["skipped"] == 2

    conn.execute("UPDATE movies SET overview='Completely different text now.' WHERE tmdb_id=1")
    third = embed_movies(conn, backend)
    assert third["movies"] == 1, "a changed document must be re-embedded"


def test_embed_reviews_is_incremental(conn, backend, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert embed_reviews(conn, backend)["reviews"] == 2
    assert embed_reviews(conn, backend)["reviews"] == 0


def test_vector_store_search_and_exclusion(conn, backend):
    for i, title in enumerate(["Space Opera", "Kitchen Drama", "Space Station"], start=1):
        seed_movie(conn, i, title=title, overview=f"A film called {title}.")
    embed_movies(conn, backend)
    store = VectorStore.load(conn, MOVIE, backend.name)
    assert len(store) == 3
    assert store.dim == backend.dim

    query = backend.encode(["Space Opera"], is_query=True)[0]
    hits = store.search(query, k=3)
    assert hits[0][0] == "1"

    excluded = store.search(query, k=3, exclude={"1"})
    assert all(eid != "1" for eid, _ in excluded)


def test_vector_store_handles_empty(conn, backend):
    store = VectorStore.load(conn, MOVIE, backend.name)
    assert len(store) == 0
    assert store.search(np.zeros(8, dtype=np.float32), k=5) == []


def test_vectors_for_skips_unknown_ids(conn, backend):
    seed_movie(conn, 1)
    embed_movies(conn, backend)
    store = VectorStore.load(conn, MOVIE, backend.name)
    keys, mat = store.vectors_for(["1", "does-not-exist"])
    assert keys == ["1"] and mat.shape[0] == 1
