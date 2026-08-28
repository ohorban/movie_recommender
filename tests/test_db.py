"""Schema, migrations and the upsert helpers."""

from __future__ import annotations

import numpy as np
import pytest

from movierec.db import (
    SCHEMA_VERSION,
    blob_to_vector,
    content_hash,
    current_version,
    init_db,
    kv_get,
    kv_set,
    migrate,
    record_source_file,
    scalar,
    split_statements,
    transaction,
    upsert,
    vector_to_blob,
)


def test_migrations_apply_and_are_idempotent(tmp_path):
    db = tmp_path / "a.db"
    conn = init_db(db)
    assert current_version(conn) == SCHEMA_VERSION
    assert migrate(conn) == 0
    conn.close()
    assert migrate(init_db(db)) == 0


def test_expected_tables_exist(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in [
        "movies",
        "user_films",
        "user_ratings",
        "user_reviews",
        "review_facts",
        "embeddings",
        "cf_neighbors",
        "movie_tags",
        "movie_dossiers",
        "ingest_runs",
        "source_files",
        "title_overrides",
        "feedback",
    ]:
        assert table in names, table


def test_foreign_keys_are_enforced(conn):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO user_ratings (film_key, rating) VALUES ('nope', 4.0)")


def test_split_statements_respects_strings_and_triggers():
    sql = "CREATE TABLE t (a TEXT DEFAULT 'x;y'); CREATE INDEX i ON t(a);"
    assert len(split_statements(sql)) == 2


def test_upsert_inserts_then_updates(conn):
    rows = [{"tmdb_id": 1, "title": "First", "year": 2000}]
    upsert(conn, "movies", rows, key=["tmdb_id"])
    assert scalar(conn, "SELECT title FROM movies WHERE tmdb_id=1") == "First"

    upsert(conn, "movies", [{"tmdb_id": 1, "title": "Second", "year": 2001}], key=["tmdb_id"])
    assert scalar(conn, "SELECT title FROM movies WHERE tmdb_id=1") == "Second"
    assert scalar(conn, "SELECT COUNT(*) FROM movies") == 1


def test_upsert_can_protect_columns(conn):
    upsert(conn, "movies", [{"tmdb_id": 2, "title": "Keep", "year": 1990}], key=["tmdb_id"])
    upsert(
        conn,
        "movies",
        [{"tmdb_id": 2, "title": "Ignored", "year": 1991}],
        key=["tmdb_id"],
        update=["year"],
    )
    assert scalar(conn, "SELECT title FROM movies WHERE tmdb_id=2") == "Keep"
    assert scalar(conn, "SELECT year FROM movies WHERE tmdb_id=2") == 1991


def test_transaction_rolls_back(conn):
    upsert(conn, "movies", [{"tmdb_id": 3, "title": "Before"}], key=["tmdb_id"])
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("UPDATE movies SET title='After' WHERE tmdb_id=3")
        raise RuntimeError("boom")
    assert scalar(conn, "SELECT title FROM movies WHERE tmdb_id=3") == "Before"


def test_content_hash_is_stable_and_order_independent_for_dicts():
    assert content_hash("a", 1) == content_hash("a", 1)
    assert content_hash("a", 1) != content_hash("a", 2)
    assert content_hash({"x": 1, "y": 2}) == content_hash({"y": 2, "x": 1})


def test_vector_roundtrip():
    vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(blob_to_vector(vector_to_blob(vec)), vec)


def test_kv_roundtrip(conn):
    kv_set(conn, "k", {"a": [1, 2]})
    assert kv_get(conn, "k") == {"a": [1, 2]}
    assert kv_get(conn, "missing", "fallback") == "fallback"


def test_record_source_file_detects_change(tmp_path, conn):
    f = tmp_path / "x.csv"
    f.write_text("a\n")
    assert record_source_file(conn, f) is True
    assert record_source_file(conn, f) is False
    f.write_text("b\n")
    assert record_source_file(conn, f) is True
