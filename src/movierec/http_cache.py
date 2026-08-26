"""A small on-disk HTTP cache plus a polite rate-limited session.

Every external response is cached in its own SQLite file, which means a rebuild
of the catalog costs no API calls and an interrupted download resumes for free.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .db import content_hash, utcnow
from .logging_utils import get_logger

log = get_logger("http")

# Distinguishes "cached 404" from "not in cache" so missing ids are not refetched.
_NULL_SENTINEL = {"__movierec_null__": True}

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key        TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    status     INTEGER NOT NULL,
    body       BLOB NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


class ResponseCache:
    """Thread-safe gzip-compressed response cache keyed by URL + params."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_CACHE_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def make_key(url: str, params: dict[str, Any] | None) -> str:
        return content_hash(url, params or {})

    def get(self, key: str, *, max_age_days: float | None = None) -> Any | None:
        row = (
            self._conn()
            .execute("SELECT body, fetched_at FROM responses WHERE key = ?", (key,))
            .fetchone()
        )
        if row is None:
            return None
        if max_age_days is not None:
            try:
                age = time.time() - time.mktime(time.strptime(row[1], "%Y-%m-%d %H:%M:%S"))
                if age > max_age_days * 86400:
                    return None
            except ValueError:
                pass
        return json.loads(gzip.decompress(row[0]).decode("utf-8"))

    def put(self, key: str, url: str, status: int, payload: Any) -> None:
        blob = gzip.compress(json.dumps(payload).encode("utf-8"), compresslevel=6)
        self._conn().execute(
            "INSERT INTO responses (key, url, status, body, fetched_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET body=excluded.body, status=excluded.status, fetched_at=excluded.fetched_at",
            (key, url, status, blob, utcnow()),
        )
        self._conn().commit()

    def stats(self) -> dict[str, Any]:
        row = (
            self._conn()
            .execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(body)), 0) FROM responses")
            .fetchone()
        )
        return {"entries": row[0], "bytes": row[1]}


class RateLimiter:
    """Simple thread-safe token bucket."""

    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = 1.0 / max(rate_per_sec, 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class RateLimitError(RuntimeError):
    """Raised on HTTP 429 so tenacity can back off."""


class CachedSession:
    """`requests` wrapper with caching, rate limiting and exponential backoff."""

    def __init__(
        self,
        cache: ResponseCache,
        *,
        rate_per_sec: float = 20.0,
        timeout: float = 20.0,
        user_agent: str = "movierec/0.1 (personal use)",
    ) -> None:
        self.cache = cache
        self.limiter = RateLimiter(rate_per_sec)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.hits = 0
        self.misses = 0

    @retry(
        retry=retry_if_exception_type((RateLimitError, requests.RequestException)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch(
        self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> tuple[int, Any]:
        self.limiter.wait()
        resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "2"))
            time.sleep(min(retry_after, 30))
            raise RateLimitError(url)
        if resp.status_code >= 500:
            raise requests.RequestException(f"{resp.status_code} from {url}")
        if resp.status_code == 404:
            return 404, None
        resp.raise_for_status()
        return resp.status_code, resp.json()

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        cache_key_params: dict[str, Any] | None = None,
        use_cache: bool = True,
        max_age_days: float | None = None,
    ) -> Any | None:
        """GET a JSON document, preferring the cache.

        ``cache_key_params`` lets callers keep secrets (API keys) out of the
        cache key while still varying it by the meaningful query parameters.
        """
        key = self.cache.make_key(url, cache_key_params if cache_key_params is not None else params)
        if use_cache:
            cached = self.cache.get(key, max_age_days=max_age_days)
            if cached is not None:
                self.hits += 1
                return None if cached == _NULL_SENTINEL else cached
        status, payload = self._fetch(url, params, headers)
        self.misses += 1
        if status == 404:
            self.cache.put(key, url, 404, _NULL_SENTINEL)
            return None
        self.cache.put(key, url, status, payload)
        return payload
