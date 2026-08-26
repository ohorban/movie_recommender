"""Pipeline orchestration: the `setup` and `update` runs.

`setup` builds everything from nothing. `update` re-runs the same stages but
every one of them is incremental - unchanged export files are skipped, films
that already have detail are not refetched, reviews are only re-analysed when
their text changed, and embeddings are only recomputed when their document
changed. In practice an update after adding a handful of new logs costs a few
seconds and a couple of API calls.

Both are driven through the same function so there is only one code path to
keep correct.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import Config
from .db import RunLogger, fetch_all, init_db, kv_get, kv_set, scalar, utcnow
from .enrich.embeddings import EmbeddingBackend, embed_movies, embed_reviews, make_backend
from .enrich.llm import ClaudeClient
from .enrich.structuring import generate_dossiers, structure_reviews
from .http_cache import CachedSession, ResponseCache
from .ingest import imdb as imdb_ingest
from .ingest import movielens as ml_ingest
from .ingest import wikipedia as wiki_ingest
from .ingest.letterboxd import ingest_export, latest_export_dir
from .ingest.resolve import resolve_user_films
from .ingest.tmdb import TMDBClient, build_catalog, ensure_movies, fetch_details, pending_detail_ids
from .logging_utils import ProgressFn, get_logger
from .taste.profile import build_profile
from .taste.summary import generate_summary
from .taste.training import train_ranker

log = get_logger("pipeline")


@dataclass
class PipelineReport:
    kind: str
    started_at: str = field(default_factory=utcnow)
    finished_at: str = ""
    stages: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True

    def add(self, name: str, stats: Any) -> None:
        self.stages[name] = stats

    def warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)


def _noop(_msg: str, _frac: float) -> None:
    return None


def run(
    cfg: Config,
    *,
    kind: str = "update",
    progress: ProgressFn | None = None,
    conn: sqlite3.Connection | None = None,
    skip_llm: bool = False,
    force_export: bool = False,
    tmdb_client: Any = None,
    llm_client: Any = None,
    session: Any = None,
    backend: EmbeddingBackend | None = None,
) -> PipelineReport:
    """Run the whole pipeline. ``kind`` is 'setup', 'update' or 'rebuild'.

    The client arguments exist so the tests can drive the real pipeline with
    stand-ins instead of network calls - the code path under test is then the
    same one that runs in production.
    """
    progress = progress or _noop
    cfg.ensure_dirs()
    report = PipelineReport(kind=kind)

    owns_conn = conn is None
    conn = conn or init_db(cfg.db_path)
    runner = RunLogger(conn, kind)

    try:
        backend = backend or _stage_backend(cfg, report, progress)

        # ---- 1. Letterboxd -------------------------------------------------
        progress("Reading your Letterboxd export", 0.01)
        with runner.stage("letterboxd") as stats:
            export_dir = latest_export_dir(cfg.data_dir)
            if export_dir is None:
                raise FileNotFoundError(
                    f"No Letterboxd export found in {cfg.data_dir}. Drop the unzipped export folder there."
                )
            result = ingest_export(
                conn, export_dir, data_root=cfg.data_dir, force=force_export or kind != "update"
            )
            stats.update(result.as_dict())
            report.add("letterboxd", result.as_dict())
            kv_set(conn, "last_export_dir", export_dir.name)

        # ---- 2. External catalog -------------------------------------------
        session = session or CachedSession(
            ResponseCache(cfg.cache_dir / "http_cache.db"), rate_per_sec=22.0
        )
        tmdb = tmdb_client or (TMDBClient(cfg.tmdb_api_key, session) if cfg.tmdb_api_key else None)
        if tmdb is None:
            report.warn(
                "TMDB_API_KEY is not set — skipping all catalog work. Recommendations cannot run without it."
            )
        else:
            _run_catalog_stages(conn, cfg, tmdb, session, runner, report, progress, kind)

        # ---- 3. Embeddings -------------------------------------------------
        progress("Embedding films", 0.76)
        with runner.stage("embeddings") as stats:
            movie_stats = embed_movies(conn, backend, progress=progress, progress_span=(0.76, 0.86))
            review_stats = embed_reviews(conn, backend)
            # Both return a `skipped` key, so namespace the review side rather
            # than letting it silently overwrite the movie counts.
            stats.update(movie_stats)
            stats["reviews"] = review_stats.get("reviews", 0)
            stats["reviews_skipped"] = review_stats.get("skipped", 0)
            report.add("embeddings", dict(stats))

        # ---- 4. Claude enrichment ------------------------------------------
        client = llm_client
        if client is None and not skip_llm and cfg.anthropic_api_key:
            try:
                client = ClaudeClient(cfg)
            except Exception as exc:
                report.warn(f"Claude unavailable ({exc}); continuing without language enrichment.")
        elif client is None and not skip_llm:
            report.warn(
                "ANTHROPIC_API_KEY is not set — review structuring and explanations are disabled."
            )

        if client is not None:
            with runner.stage("review_structuring") as stats:
                progress("Reading your reviews", 0.87)
                stats.update(
                    structure_reviews(conn, client, progress=progress, progress_span=(0.87, 0.90))
                )
                report.add("review_structuring", dict(stats))

            with runner.stage("dossiers") as stats:
                seed = _dossier_seed_ids(conn, cfg.dossier_seed_limit)
                progress(f"Profiling {len(seed)} of your films", 0.90)
                stats.update(
                    generate_dossiers(
                        conn, client, seed, progress=progress, progress_span=(0.90, 0.94)
                    )
                )
                report.add("dossiers", dict(stats))

        # ---- 5. Taste model -------------------------------------------------
        progress("Building your taste profile", 0.95)
        with runner.stage("taste_profile") as stats:
            profile = build_profile(conn, backend)
            stats.update(
                {
                    "modes": len(profile.modes),
                    "rated": profile.n_rated,
                    "signals": len(profile.taste_signals),
                    "scales": len(profile.scale_targets),
                }
            )
            report.add("taste_profile", dict(stats))

        if client is not None:
            with runner.stage("taste_summary") as stats:
                progress("Summarising your taste", 0.97)
                summary = generate_summary(conn, client, profile, force=(kind != "update"))
                stats["generated"] = summary is not None
                report.add("taste_summary", dict(stats))

        # ---- 6. Ranker -------------------------------------------------------
        progress("Training the ranking model", 0.98)
        with runner.stage("ranker") as stats:
            ranker = train_ranker(conn, profile, backend)
            stats.update(ranker.metrics.to_json())
            report.add("ranker", ranker.metrics.to_json())

        kv_set(conn, "last_run", {"kind": kind, "at": utcnow()})
        progress("Done", 1.0)
    except Exception as exc:
        report.ok = False
        report.warn(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.finished_at = utcnow()
        if owns_conn:
            conn.commit()
    return report


def _stage_backend(cfg: Config, report: PipelineReport, progress: ProgressFn) -> EmbeddingBackend:
    progress("Loading the embedding model", 0.005)
    try:
        return make_backend(cfg)
    except Exception as exc:
        from .enrich.embeddings import HashBackend

        report.warn(
            f"Could not load the embedding model ({exc}). Falling back to the offline hash backend, "
            "which has no semantic understanding — install with `uv pip install -e '.[embed]'` for real results."
        )
        return HashBackend()


def _run_catalog_stages(conn, cfg, tmdb, session, runner, report, progress, kind) -> None:
    catalog_size = scalar(conn, "SELECT COUNT(*) FROM movies WHERE in_catalog = 1", default=0)

    # Discovery: full sweep on first build, recent years only on update.
    with runner.stage("tmdb_discover") as stats:
        if catalog_size < cfg.catalog_size * 0.6 or kind != "update":
            progress("Discovering the film catalog", 0.05)
            stats.update(build_catalog(conn, tmdb, cfg, progress=progress))
        else:
            progress("Checking for new releases", 0.05)
            stats.update(_refresh_recent(conn, tmdb, cfg))
        report.add("tmdb_discover", dict(stats))

    with runner.stage("tmdb_detail") as stats:
        todo = pending_detail_ids(conn)
        progress(f"Fetching details for {len(todo):,} films", 0.30)
        stats.update(fetch_details(conn, tmdb, todo, progress=progress, progress_span=(0.30, 0.52)))
        report.add("tmdb_detail", dict(stats))

    with runner.stage("resolve") as stats:
        progress("Matching your films to the catalog", 0.55)
        stats.update(resolve_user_films(conn, tmdb, progress=progress, progress_span=(0.55, 0.60)))
        user_ids = [
            r["tmdb_id"]
            for r in fetch_all(
                conn, "SELECT DISTINCT tmdb_id FROM user_films WHERE tmdb_id IS NOT NULL"
            )
        ]
        stats["ensured"] = ensure_movies(conn, tmdb, user_ids, progress=progress)
        report.add("resolve", dict(stats))

    if cfg.enable_imdb:
        with runner.stage("imdb") as stats:
            progress("Adding IMDb ratings", 0.62)
            try:
                stats.update(
                    imdb_ingest.ingest_ratings(
                        conn, cfg.external_dir, progress=progress, progress_span=(0.62, 0.65)
                    )
                )
            except Exception as exc:
                stats["_status"] = "error"
                report.warn(f"IMDb ingest failed ({exc}); continuing without IMDb ratings.")
            report.add("imdb", dict(stats))

    if cfg.enable_movielens:
        with runner.stage("movielens") as stats:
            progress("Adding MovieLens tags and taste neighbours", 0.66)
            try:
                stats.update(
                    ml_ingest.ingest_all(
                        conn, cfg.external_dir, progress=progress, progress_span=(0.66, 0.72)
                    )
                )
            except Exception as exc:
                stats["_status"] = "error"
                report.warn(
                    f"MovieLens ingest failed ({exc}); continuing without tag genome or CF."
                )
            report.add("movielens", dict(stats))

    if cfg.enable_wikipedia:
        with runner.stage("wikipedia") as stats:
            todo = wiki_ingest.pending_plot_ids(conn, limit=cfg.wikipedia_limit)
            progress(f"Fetching {len(todo):,} plot synopses", 0.73)
            try:
                stats.update(
                    wiki_ingest.fetch_plots(
                        conn, session, todo, progress=progress, progress_span=(0.73, 0.76)
                    )
                )
            except Exception as exc:
                stats["_status"] = "error"
                report.warn(f"Wikipedia ingest failed ({exc}); continuing with TMDB text only.")
            report.add("wikipedia", dict(stats))


def _refresh_recent(conn, tmdb, cfg) -> dict[str, Any]:
    """Pull in films released in the last two years that we do not have yet."""
    from .db import transaction, upsert
    from .ingest.tmdb import _summary_row

    added = 0
    for year in (date.today().year, date.today().year - 1):
        page = 1
        while page <= 15:
            payload = tmdb.discover(page, year=year, min_votes=cfg.min_votes)
            results = payload.get("results") or []
            if not results:
                break
            rows = [
                _summary_row(item, "discover")
                for item in results
                if item.get("id") and not item.get("adult")
            ]
            with transaction(conn):
                added += upsert(
                    conn,
                    "movies",
                    rows,
                    key=["tmdb_id"],
                    update=[
                        "tmdb_popularity",
                        "tmdb_vote_average",
                        "tmdb_vote_count",
                        "updated_at",
                    ],
                )
            if page >= min(int(payload.get("total_pages") or 1), 15):
                break
            page += 1
    return {"refreshed": added}


def _dossier_seed_ids(conn: sqlite3.Connection, limit: int) -> list[int]:
    """The user's own films first - their dossiers calibrate the taste scales."""
    rows = fetch_all(
        conn,
        """
        SELECT DISTINCT f.tmdb_id
        FROM user_films f
        LEFT JOIN user_ratings r USING(film_key)
        WHERE f.tmdb_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM movies m WHERE m.tmdb_id = f.tmdb_id AND m.detail_level = 2)
        ORDER BY (r.rating IS NULL), ABS(COALESCE(r.rating, 3) - 3) DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [int(r["tmdb_id"]) for r in rows]


def rebuild(
    cfg: Config, *, progress: ProgressFn | None = None, keep_overrides: bool = True
) -> PipelineReport:
    """Delete the database and rebuild it, preserving manual title corrections."""
    overrides: list[tuple[Any, ...]] = []
    if keep_overrides and Path(cfg.db_path).exists():
        conn = init_db(cfg.db_path)
        overrides = [
            tuple(r)
            for r in fetch_all(
                conn, "SELECT film_key, tmdb_id, note, created_at FROM title_overrides"
            )
        ]
        conn.close()

    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(cfg.db_path) + suffix)
        if candidate.exists():
            candidate.unlink()

    conn = init_db(cfg.db_path)
    if overrides:
        conn.executemany(
            "INSERT OR REPLACE INTO title_overrides (film_key, tmdb_id, note, created_at) VALUES (?, ?, ?, ?)",
            overrides,
        )
        log.info("restored %d manual title corrections", len(overrides))
    return run(cfg, kind="rebuild", progress=progress, conn=conn)


def status(cfg: Config) -> dict[str, Any]:
    """A quick health check for the CLI and the Data tab."""
    if not Path(cfg.db_path).exists():
        return {"initialised": False}
    conn = init_db(cfg.db_path)
    from .taste.insights import headline_stats

    out = {"initialised": True, **headline_stats(conn)}
    out["last_run"] = kv_get(conn, "last_run")
    out["last_export"] = kv_get(conn, "last_export_dir")
    out["embeddings"] = scalar(
        conn, "SELECT COUNT(*) FROM embeddings WHERE entity_type='movie'", default=0
    )
    out["cf_edges"] = scalar(conn, "SELECT COUNT(*) FROM cf_neighbors", default=0)
    out["tag_rows"] = scalar(conn, "SELECT COUNT(*) FROM movie_tags", default=0)
    conn.close()
    return out
