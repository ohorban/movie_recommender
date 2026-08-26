"""Command-line interface: `movierec <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .config import load_config
from .db import init_db
from .logging_utils import setup_logging


def _progress(message: str, fraction: float) -> None:
    bar_len = 28
    filled = int(bar_len * max(0.0, min(1.0, fraction)))
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(f"\r  {bar} {fraction * 100:5.1f}%  {message[:64]:<64}")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")


def _print_report(report: Any) -> None:
    print(f"\n{report.kind} finished at {report.finished_at}\n")
    for stage, stats in report.stages.items():
        summary = ", ".join(
            f"{k}={v}"
            for k, v in stats.items()
            if not str(k).startswith("_") and v not in (0, "", [], None)
        )
        print(f"  {stage:22s} {summary or 'nothing to do'}")
    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    ! {w}")
    print()


def cmd_setup(args: argparse.Namespace) -> int:
    from .pipeline import run

    cfg = load_config()
    missing = cfg.missing_credentials()
    if missing:
        print(f"Missing required settings in .env: {', '.join(missing)}", file=sys.stderr)
        if "TMDB_API_KEY" in missing:
            return 2
    report = run(cfg, kind="setup", progress=_progress, skip_llm=args.no_llm)
    _print_report(report)
    return 0 if report.ok else 1


def cmd_update(args: argparse.Namespace) -> int:
    from .pipeline import run

    cfg = load_config()
    report = run(
        cfg, kind="update", progress=_progress, skip_llm=args.no_llm, force_export=args.force
    )
    _print_report(report)
    return 0 if report.ok else 1


def cmd_rebuild(args: argparse.Namespace) -> int:
    from .pipeline import rebuild

    cfg = load_config()
    if not args.yes:
        answer = input(f"This deletes {cfg.db_path} and rebuilds from scratch. Continue? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    report = rebuild(cfg, progress=_progress)
    _print_report(report)
    return 0 if report.ok else 1


def cmd_status(_args: argparse.Namespace) -> int:
    from .pipeline import status

    print(json.dumps(status(load_config()), indent=2, default=str))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from .recommend.engine import RecommendationEngine

    cfg = load_config()
    conn = init_db(cfg.db_path)
    engine = RecommendationEngine(conn, cfg)
    if not engine.is_ready():
        print("The model is not built yet. Run `movierec setup` first.", file=sys.stderr)
        return 2

    result = engine.recommend(
        args.query or "", n=args.n, explain=not args.no_llm, use_llm=not args.no_llm
    )
    if result.intent.interpretation:
        print(f"\nReading your request as: {result.intent.interpretation}")
    print(
        f"\n{len(result.items)} picks from {result.pool_size:,} candidates (ranker: {result.ranker_kind})\n"
    )
    for item in result.items:
        flag = " ★watchlist" if item.on_watchlist else ""
        print(f"{item.rank:2d}. {item.title} ({item.year}){flag}")
        if item.hook:
            print(f"    {item.hook}")
        if item.because:
            print(f"    Why you: {item.because}")
        if item.caveat:
            print(f"    Heads up: {item.caveat}")
        print(
            f"    {', '.join(item.genres)} · score {item.score:+.2f} · {', '.join(item.sources[:3])}"
        )
        print()
    for note in result.notes:
        print(f"  ! {note}")
    return 0


def cmd_unmatched(_args: argparse.Namespace) -> int:
    from .ingest.resolve import unresolved_report

    cfg = load_config()
    conn = init_db(cfg.db_path)
    rows = unresolved_report(conn)
    if not rows:
        print("Every film in your history is matched.")
        return 0
    print(f"{len(rows)} films need attention:\n")
    for r in rows:
        matched = f"→ {r['matched_title']} ({r['matched_year']})" if r["tmdb_id"] else "→ no match"
        conf = f"{r['match_confidence']:.2f}" if r["match_confidence"] is not None else " -- "
        print(f"  [{conf}] {r['title']} ({r['year']})  {matched}")
        print(f"          film_key: {r['film_key']}")
    print("\nFix one with:  movierec fix-match <film_key> <tmdb_id>")
    return 0


def cmd_fix_match(args: argparse.Namespace) -> int:
    from .ingest.resolve import set_override

    cfg = load_config()
    conn = init_db(cfg.db_path)
    set_override(conn, args.film_key, args.tmdb_id, note="set from CLI")
    conn.commit()
    print(
        f"{args.film_key} is now pinned to TMDB id {args.tmdb_id}. Run `movierec update` to refresh."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="movierec",
        description="A personal movie recommender built on your Letterboxd history.",
    )
    parser.add_argument("--version", action="version", version=f"movierec {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="First-time build of the whole database")
    p.add_argument("--no-llm", action="store_true", help="Skip every Claude call")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("update", help="Incremental update from the newest export")
    p.add_argument("--no-llm", action="store_true", help="Skip every Claude call")
    p.add_argument("--force", action="store_true", help="Reprocess the export even if unchanged")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("rebuild", help="Delete and rebuild the database")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("status", help="Show what is in the database")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("recommend", help="Get recommendations from the terminal")
    p.add_argument("query", nargs="?", default="", help="Optional natural-language request")
    p.add_argument("-n", type=int, default=6, help="How many to return")
    p.add_argument(
        "--no-llm", action="store_true", help="Skip Claude interpretation and explanations"
    )
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("unmatched", help="List films that need a manual TMDB match")
    p.set_defaults(func=cmd_unmatched)

    p = sub.add_parser("fix-match", help="Pin a film to a specific TMDB id")
    p.add_argument("film_key")
    p.add_argument("tmdb_id", type=int)
    p.set_defaults(func=cmd_fix_match)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
