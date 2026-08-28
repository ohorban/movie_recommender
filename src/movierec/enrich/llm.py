"""Claude client: schema-enforced JSON, response caching and bounded concurrency.

Every call goes through a forced tool-use call so the model returns a validated
object rather than prose we have to parse. Responses are cached in the database
by content hash, which makes re-runs free and keeps the review-structuring pass
a one-time cost even when the rest of the pipeline is rebuilt.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import Config
from ..db import content_hash, utcnow
from ..logging_utils import get_logger

log = get_logger("enrich.llm")

ProgressFn = Callable[[str, float], None]

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key          TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    model        TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


class LLMError(RuntimeError):
    pass


class ClaudeClient:
    """Thin, thread-safe wrapper around the Anthropic Messages API."""

    def __init__(self, cfg: Config, cache_path: str | None = None) -> None:
        if not cfg.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The `anthropic` package is not installed.") from exc

        self.cfg = cfg
        self.model = cfg.llm_model
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, max_retries=4)
        self._cache_path = cache_path or str(cfg.cache_dir / "llm_cache.db")
        self._local = threading.local()
        self._lock = threading.Lock()
        self._create_params = self._supported_create_params()
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._init_cache()

    def _supported_create_params(self) -> set[str]:
        """Which keyword arguments this installed SDK's `messages.create` accepts.

        The SDK is a moving target: anthropic 1.x dropped `temperature` and
        `top_p` from Messages.create entirely. Rather than pin to one version,
        introspect once and drop anything unsupported, so the same code runs on
        0.x and 1.x alike.
        """
        try:
            return set(inspect.signature(self.client.messages.create).parameters)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return set()

    def _call_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Drop keyword arguments the installed SDK does not accept."""
        if not self._create_params:
            return kwargs
        supported, dropped = {}, []
        for key, value in kwargs.items():
            if key in self._create_params:
                supported[key] = value
            else:
                dropped.append(key)
        if dropped and not getattr(self, "_warned_dropped", False):
            log.info(
                "anthropic %s does not accept %s; continuing without it",
                getattr(self._anthropic, "__version__", "?"),
                ", ".join(sorted(dropped)),
            )
            self._warned_dropped = True
        return supported

    # ----------------------------------------------------------------- cache
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            from pathlib import Path

            Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._cache_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_cache(self) -> None:
        self._conn().executescript(_CACHE_SCHEMA)

    def _cache_get(self, key: str) -> Any | None:
        row = (
            self._conn()
            .execute("SELECT payload_json FROM llm_cache WHERE key = ?", (key,))
            .fetchone()
        )
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, kind: str, payload: Any) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO llm_cache (key, kind, model, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, kind, self.model, json.dumps(payload), utcnow()),
        )
        self._conn().commit()

    # ------------------------------------------------------------------ call
    def structured(
        self,
        *,
        kind: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        tool_name: str,
        tool_description: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """One schema-enforced call. Returns the tool input as a dict."""
        key = content_hash(self.model, kind, system, user, schema)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                with self._lock:
                    self.cache_hits += 1
                return cached

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {"name": tool_name, "description": tool_description, "input_schema": schema}
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as exc:
            if "model" in str(exc).lower() and "not_found" in str(exc).lower():
                raise LLMError(
                    f"Model {self.model!r} was rejected by the API. Set MOVIEREC_LLM_MODEL in .env "
                    f"to an available model id. Known current ids include 'claude-sonnet-5', "
                    f"'claude-opus-5' and 'claude-haiku-4-5-20251001'."
                ) from exc
            raise

        with self._lock:
            self.calls += 1
            usage = getattr(resp, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.output_tokens += getattr(usage, "output_tokens", 0) or 0

        payload: dict[str, Any] | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                payload = dict(block.input)
                break
        if payload is None:
            raise LLMError(f"Claude returned no structured output for {kind!r}")

        if use_cache:
            self._cache_put(key, kind, payload)
        return payload

    def map_structured(
        self,
        jobs: Sequence[dict[str, Any]],
        *,
        progress: ProgressFn | None = None,
        progress_span: tuple[float, float] = (0.0, 1.0),
        label: str = "Thinking",
    ) -> list[dict[str, Any] | None]:
        """Run many `structured` calls concurrently, preserving input order."""
        if not jobs:
            return []
        lo, hi = progress_span
        done = 0
        lock = threading.Lock()

        def run(job: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal done
            try:
                return self.structured(**job)
            except Exception as exc:
                log.warning("LLM call failed (%s): %s", job.get("kind"), exc)
                return None
            finally:
                with lock:
                    done += 1
                    if progress and done % 5 == 0:
                        progress(f"{label} · {done}/{len(jobs)}", lo + (hi - lo) * done / len(jobs))

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.llm_max_concurrency)) as pool:
            return list(pool.map(run, jobs))

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 600,
        temperature: float = 0.4,
        use_cache: bool = False,
    ) -> str:
        """Plain prose response, used for recommendation pitches."""
        key = content_hash(self.model, "text", system, user, temperature)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached.get("text", "")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        with self._lock:
            self.calls += 1
        out = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if use_cache:
            self._cache_put(key, "text", {"text": out})
        return out

    def usage_summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
