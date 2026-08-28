"""Central configuration, resolved once from the environment and `.env`.

Every tunable in the system lands here so that the pipeline, the CLI and the
Streamlit app all agree on paths, model names and thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_a, **_k):  # type: ignore[misc]
        return False


def _project_root() -> Path:
    """Walk upwards from this file to the repository root.

    src/movierec/config.py -> src/movierec -> src -> <root>
    """
    return Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of every runtime setting."""

    root: Path

    # --- credentials -------------------------------------------------------
    tmdb_api_key: str = ""
    anthropic_api_key: str = ""

    # --- models ------------------------------------------------------------
    llm_model: str = "claude-sonnet-5"
    llm_max_concurrency: int = 4
    embed_backend: str = "sentence_transformers"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch: int = 64

    # --- catalog scope -----------------------------------------------------
    catalog_size: int = 30_000
    min_votes: int = 50
    min_year: int = 1930

    # --- optional sources --------------------------------------------------
    enable_movielens: bool = True
    enable_imdb: bool = True
    enable_wikipedia: bool = True

    # --- enrichment scope --------------------------------------------------
    wikipedia_limit: int = 8_000  # films fetched eagerly; the rest lazily
    dossier_seed_limit: int = 400  # user films profiled by Claude during setup

    # --- recommendation tuning --------------------------------------------
    candidates_per_source: int = 400
    exploration_ratio: float = 0.15
    watchlist_boost: float = 0.06

    # --- paths (derived) ---------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path("db/movierec.db"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self) -> None:
        """Anchor relative paths to ``root``.

        :func:`load_config` already passes absolute paths, but a directly
        constructed ``Config`` would otherwise resolve `db/` and `data/`
        against the current working directory — quietly reading and writing
        somebody else's files depending on where the process was started.
        """
        for field_name in ("db_path", "data_dir"):
            value = getattr(self, field_name)
            if not Path(value).is_absolute():
                object.__setattr__(self, field_name, (self.root / value).resolve())

    # ---------------------------------------------------------------- paths
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def vectors_dir(self) -> Path:
        return self.db_path.parent / "vectors"

    @property
    def state_dir(self) -> Path:
        return self.root / "var"

    def ensure_dirs(self) -> None:
        for path in (
            self.db_path.parent,
            self.cache_dir,
            self.external_dir,
            self.vectors_dir,
            self.state_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- validity
    def missing_credentials(self, *, need_llm: bool = True, need_tmdb: bool = True) -> list[str]:
        missing = []
        if need_tmdb and not self.tmdb_api_key:
            missing.append("TMDB_API_KEY")
        if need_llm and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


def load_config(root: Path | None = None, *, reload_env: bool = True) -> Config:
    """Build a :class:`Config` from `.env` plus the process environment."""
    root = Path(root).resolve() if root else _project_root()
    if reload_env:
        load_dotenv(root / ".env", override=False)

    env = os.environ.get

    def _path(key: str, fallback: str) -> Path:
        raw = env(key) or fallback
        p = Path(raw)
        return p if p.is_absolute() else root / p

    return Config(
        root=root,
        tmdb_api_key=(env("TMDB_API_KEY") or "").strip(),
        anthropic_api_key=(env("ANTHROPIC_API_KEY") or "").strip(),
        llm_model=env("MOVIEREC_LLM_MODEL") or "claude-sonnet-5",
        llm_max_concurrency=_as_int(env("MOVIEREC_LLM_MAX_CONCURRENCY"), 4),
        embed_backend=(env("MOVIEREC_EMBED_BACKEND") or "sentence_transformers").strip(),
        embed_model=env("MOVIEREC_EMBED_MODEL") or "BAAI/bge-small-en-v1.5",
        embed_batch=_as_int(env("MOVIEREC_EMBED_BATCH"), 64),
        catalog_size=_as_int(env("MOVIEREC_CATALOG_SIZE"), 30_000),
        min_votes=_as_int(env("MOVIEREC_MIN_VOTES"), 50),
        min_year=_as_int(env("MOVIEREC_MIN_YEAR"), 1930),
        enable_movielens=_as_bool(env("MOVIEREC_ENABLE_MOVIELENS"), True),
        enable_imdb=_as_bool(env("MOVIEREC_ENABLE_IMDB"), True),
        enable_wikipedia=_as_bool(env("MOVIEREC_ENABLE_WIKIPEDIA"), True),
        wikipedia_limit=_as_int(env("MOVIEREC_WIKIPEDIA_LIMIT"), 8_000),
        dossier_seed_limit=_as_int(env("MOVIEREC_DOSSIER_SEED_LIMIT"), 400),
        candidates_per_source=_as_int(env("MOVIEREC_CANDIDATES_PER_SOURCE"), 400),
        exploration_ratio=_as_float(env("MOVIEREC_EXPLORATION_RATIO"), 0.15),
        watchlist_boost=_as_float(env("MOVIEREC_WATCHLIST_BOOST"), 0.06),
        db_path=_path("MOVIEREC_DB_PATH", "db/movierec.db"),
        data_dir=_path("MOVIEREC_DATA_DIR", "data"),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached config. Call :func:`load_config` for a fresh one."""
    return load_config()
