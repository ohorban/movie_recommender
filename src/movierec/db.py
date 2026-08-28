"""SQLite access layer: connections, migrations and small upsert helpers.

The database is the single source of truth for the whole system - catalog,
user history, embeddings and trained artefacts all live here, which keeps
incremental updates honest and makes the whole thing trivially backup-able.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import get_logger
from .migrations import MIGRATIONS

log = get_logger("db")

SCHEMA_VERSION = max(v for v, _, _ in MIGRATIONS)


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #
def connect(
    db_path: str | Path, *, read_only: bool = False, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open a tuned connection. Creates parent directories as needed.

    ``check_same_thread=False`` is needed by Streamlit, which caches the
    connection across script reruns that may land on different threads.
    """
    db_path = Path(db_path)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path), timeout=30.0, isolation_level=None, check_same_thread=check_same_thread
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-64000")  # ~64 MB page cache
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction with rollback on error (isolation_level is None)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
def current_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations").fetchone()
    return int(row["v"])


def split_statements(sql: str) -> list[str]:
    """Split a migration script into individual statements.

    ``sqlite3.executescript`` implicitly commits, which would break the
    surrounding transaction, so migrations are executed statement by statement.
    String literals and ``BEGIN ... END`` trigger bodies are respected.
    """
    statements: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    in_line_comment = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_string:
            buf.append(ch)
            if ch == "'":
                if nxt == "'":  # escaped quote
                    buf.append(nxt)
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
            i += 1
            continue

        # Track trigger bodies so their inner semicolons do not split.
        upper_tail = "".join(buf[-6:]).upper()
        if ch in " \t\n(" and upper_tail.endswith("BEGIN"):
            depth += 1
        elif ch in " \t\n;" and upper_tail.endswith("END"):
            depth = max(0, depth - 1)

        if ch == ";" and depth == 0:
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every pending migration. Returns the number applied."""
    have = current_version(conn)
    applied = 0
    for version, name, sql in sorted(MIGRATIONS):
        if version <= have:
            continue
        log.info("applying migration %03d %s", version, name)
        with transaction(conn):
            for statement in split_statements(sql):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)", (version, name)
            )
        applied += 1
    if applied:
        conn.execute("ANALYZE")
    return applied


def init_db(db_path: str | Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open and bring a database fully up to date."""
    conn = connect(db_path, check_same_thread=check_same_thread)
    migrate(conn)
    return conn


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def content_hash(*parts: Any) -> str:
    """Stable short hash of arbitrary content, used for change detection."""
    h = hashlib.sha256()
    for part in parts:
        if part is None:
            h.update(b"\x00")
        elif isinstance(part, (dict, list, tuple)):
            h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def upsert(
    conn: sqlite3.Connection,
    table: str,
    rows: Sequence[dict[str, Any]],
    *,
    key: Sequence[str],
    update: Sequence[str] | None = None,
) -> int:
    """Bulk INSERT ... ON CONFLICT DO UPDATE. Returns the row count written.

    ``update`` names the columns refreshed on conflict; omit it to refresh
    every non-key column. Pass an empty sequence for insert-or-ignore.
    """
    if not rows:
        return 0
    columns = list(rows[0].keys())
    if update is None:
        update = [c for c in columns if c not in key]

    placeholders = ", ".join("?" for _ in columns)
    collist = ", ".join(f'"{c}"' for c in columns)
    conflict = ", ".join(f'"{c}"' for c in key)

    if update:
        setters = ", ".join(f'"{c}"=excluded."{c}"' for c in update)
        sql = (
            f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {setters}"
        )
    else:
        sql = f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO NOTHING"

    payload = [tuple(r.get(c) for c in columns) for r in rows]
    conn.executemany(sql, payload)
    return len(payload)


def insert_ignore(conn: sqlite3.Connection, table: str, rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    sql = (
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    conn.executemany(sql, [tuple(r.get(c) for c in columns) for r in rows])
    return len(rows)


def fetch_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def fetch_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def scalar(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = (), default: Any = None
) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return default
    value = row[0]
    return default if value is None else value


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# Key/value state
# --------------------------------------------------------------------------- #
def kv_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = fetch_one(conn, "SELECT value FROM kv WHERE key = ?", (key,))
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def kv_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(value, default=str), utcnow()),
    )


# --------------------------------------------------------------------------- #
# Vector (de)serialisation - float32, little-endian, stored as BLOB
# --------------------------------------------------------------------------- #
def vector_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


# --------------------------------------------------------------------------- #
# Ingest run bookkeeping
# --------------------------------------------------------------------------- #
class RunLogger:
    """Records each pipeline stage into ``ingest_runs`` for the Data tab."""

    def __init__(self, conn: sqlite3.Connection, kind: str) -> None:
        self.conn = conn
        self.kind = kind

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        cur = self.conn.execute(
            "INSERT INTO ingest_runs (kind, stage, status) VALUES (?, ?, 'running')",
            (self.kind, name),
        )
        run_id = cur.lastrowid
        stats: dict[str, Any] = {}
        try:
            yield stats
        except Exception as exc:
            self.conn.execute(
                "UPDATE ingest_runs SET status='error', finished_at=?, error=?, stats_json=? WHERE run_id=?",
                (
                    utcnow(),
                    f"{type(exc).__name__}: {exc}"[:2000],
                    json.dumps(stats, default=str),
                    run_id,
                ),
            )
            raise
        else:
            self.conn.execute(
                "UPDATE ingest_runs SET status=?, finished_at=?, stats_json=? WHERE run_id=?",
                (
                    stats.pop("_status", "ok"),
                    utcnow(),
                    json.dumps(stats, default=str),
                    run_id,
                ),
            )


def record_source_file(conn: sqlite3.Connection, path: Path, *, root: Path | None = None) -> bool:
    """Fingerprint a file. Returns True when it is new or has changed."""
    rel = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
    digest = file_sha256(path)
    stat = path.stat()
    prev = fetch_one(conn, "SELECT sha256 FROM source_files WHERE path = ?", (rel,))
    changed = prev is None or prev["sha256"] != digest
    conn.execute(
        "INSERT INTO source_files (path, sha256, size_bytes, mtime) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (path) DO UPDATE SET sha256=excluded.sha256, size_bytes=excluded.size_bytes,"
        " mtime=excluded.mtime, last_seen=datetime('now')",
        (rel, digest, stat.st_size, stat.st_mtime),
    )
    return changed
