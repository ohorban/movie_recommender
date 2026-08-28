"""End-to-end: the real pipeline, driven with stand-in clients.

This is the test that matters most. It runs the production code path from a
Letterboxd export all the way to ranked recommendations, so a break anywhere in
the chain - schema, ingest, resolution, embedding, taste model, ranker,
retrieval - shows up here.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeClaudeClient, FakeTMDBClient
from movierec.db import scalar
from movierec.enrich.embeddings import HashBackend
from movierec.pipeline import run, status
from movierec.recommend.candidates import seen_tmdb_ids
from movierec.recommend.engine import RecommendationEngine


@pytest.fixture
def built(tmp_config, conn, library_export):
    """A fully built database, from the user's actual export."""
    shutil.copytree(library_export, tmp_config.data_dir / library_export.name)
    tmdb = FakeTMDBClient(n_movies=600, min_year=2000, max_year=2026)
    claude = FakeClaudeClient()
    backend = HashBackend(dim=128)
    report = run(
        tmp_config,
        kind="setup",
        conn=conn,
        tmdb_client=tmdb,
        llm_client=claude,
        backend=backend,
    )
    return {
        "cfg": tmp_config,
        "conn": conn,
        "report": report,
        "tmdb": tmdb,
        "claude": claude,
        "backend": backend,
    }


def test_pipeline_completes(built):
    report = built["report"]
    assert report.ok, report.warnings
    for stage in [
        "letterboxd",
        "tmdb_discover",
        "tmdb_detail",
        "resolve",
        "embeddings",
        "review_structuring",
        "dossiers",
        "taste_profile",
        "ranker",
    ]:
        assert stage in report.stages, f"stage {stage} did not run"


def test_catalog_and_user_history_are_populated(built):
    conn = built["conn"]
    assert scalar(conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2") > 100
    assert scalar(conn, "SELECT COUNT(*) FROM user_films") > 100
    assert scalar(conn, "SELECT COUNT(*) FROM user_ratings") > 50
    assert scalar(conn, "SELECT COUNT(*) FROM movie_genres") > 100
    assert scalar(conn, "SELECT COUNT(*) FROM movie_keywords") > 100


def test_every_user_film_resolves_or_is_flagged(built):
    conn = built["conn"]
    unresolved = scalar(
        conn, "SELECT COUNT(*) FROM user_films WHERE tmdb_id IS NULL AND needs_review = 0"
    )
    assert unresolved == 0, "an unmatched film must always be flagged for review"


def test_no_dangling_foreign_keys(built):
    conn = built["conn"]
    assert (
        scalar(
            conn,
            "SELECT COUNT(*) FROM user_films uf LEFT JOIN movies m USING(tmdb_id) "
            "WHERE uf.tmdb_id IS NOT NULL AND m.tmdb_id IS NULL",
        )
        == 0
    )
    assert list(conn.execute("PRAGMA foreign_key_check")) == []


def test_embeddings_cover_the_catalog_and_reviews(built):
    conn, backend = built["conn"], built["backend"]
    movies = scalar(
        conn,
        "SELECT COUNT(*) FROM embeddings WHERE entity_type='movie' AND model=?",
        (backend.name,),
    )
    reviews = scalar(
        conn,
        "SELECT COUNT(*) FROM embeddings WHERE entity_type='review' AND model=?",
        (backend.name,),
    )
    detailed = scalar(conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2 AND in_catalog = 1")
    assert movies == detailed
    assert reviews > 50
    assert scalar(conn, "SELECT COUNT(DISTINCT dim) FROM embeddings") == 1, (
        "all vectors share one dimension"
    )


def test_reviews_become_structured_facts(built):
    conn = built["conn"]
    reviews = scalar(conn, "SELECT COUNT(*) FROM user_reviews")
    facts = scalar(conn, "SELECT COUNT(*) FROM review_facts")
    assert facts == reviews > 50
    payload = scalar(conn, "SELECT payload_json FROM review_facts LIMIT 1")
    assert "taste_signals" in payload


def test_taste_profile_has_modes_and_signals(built):
    from movierec.taste.profile import load_profile

    profile = load_profile(built["conn"], built["backend"].name)
    assert profile is not None
    assert profile.n_rated > 50
    assert len(profile.modes) >= 1
    assert all(m.centroid.shape[0] == built["backend"].dim for m in profile.modes)
    assert abs(sum(m.weight for m in profile.modes) - 1.0) < 1e-6
    assert profile.affinities.get("genre")
    assert profile.taste_signals
    assert profile.summary and profile.summary.get("headline")


def test_ranker_is_trained_and_reloadable(built):
    from movierec.recommend.ranker import TasteRanker

    ranker = TasteRanker.load(built["conn"])
    assert ranker is not None
    assert ranker.metrics.n_train > 50


def test_recommendations_are_produced_and_unseen(built):
    engine = RecommendationEngine(
        built["conn"], built["cfg"], backend=built["backend"], client=built["claude"]
    )
    assert engine.is_ready()
    result = engine.recommend("", n=6)
    assert len(result.items) == 6
    assert result.pool_size > 20

    already_seen = seen_tmdb_ids(built["conn"])
    assert not ({i.tmdb_id for i in result.items} & already_seen), (
        "never recommend a film they have seen"
    )
    assert len({i.tmdb_id for i in result.items}) == 6, "no duplicates"
    assert all(i.title for i in result.items)
    assert all(i.rank == n for n, i in enumerate(result.items, 1))


def test_natural_language_request_runs_end_to_end(built):
    engine = RecommendationEngine(
        built["conn"], built["cfg"], backend=built["backend"], client=built["claude"]
    )
    result = engine.recommend("something tense but not bleak, under two hours", n=5)
    assert len(result.items) == 5
    assert result.intent.interpretation
    assert result.intent.semantic_query
    assert all(i.hook for i in result.items), "every pick should carry an explanation"


def test_dismissed_films_do_not_come_back(built):
    engine = RecommendationEngine(
        built["conn"], built["cfg"], backend=built["backend"], client=built["claude"]
    )
    first = engine.recommend("", n=5)
    victim = first.items[0].tmdb_id
    engine.record_feedback(victim, "dislike", "test")
    second = engine.recommend("", n=8)
    assert victim not in {i.tmdb_id for i in second.items}


def test_exclusion_is_respected(built):
    engine = RecommendationEngine(
        built["conn"], built["cfg"], backend=built["backend"], client=built["claude"]
    )
    first = engine.recommend("", n=4)
    blocked = {i.tmdb_id for i in first.items}
    second = engine.recommend("", n=4, exclude=blocked)
    assert not (blocked & {i.tmdb_id for i in second.items})


def test_rerunning_the_pipeline_does_almost_nothing(built):
    """The whole point of the incremental design: a no-change update is cheap.

    An update still picks up newly released films, so the invariant is not
    "zero work" but "work proportional to what actually changed": no existing
    film is re-embedded, and nothing at all is re-sent to the LLM.
    """
    cfg, conn, tmdb, claude, backend = (
        built[k] for k in ("cfg", "conn", "tmdb", "claude", "backend")
    )
    detail_calls_before = tmdb.detail_calls
    llm_calls_before = len(claude.calls)
    movies_before = scalar(conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2")

    report = run(
        cfg, kind="update", conn=conn, tmdb_client=tmdb, llm_client=claude, backend=backend
    )
    movies_after = scalar(conn, "SELECT COUNT(*) FROM movies WHERE detail_level = 2")

    assert report.ok
    assert report.stages["letterboxd"]["unchanged"] is True

    # Exactly the newly-detailed films were embedded, and not one more.
    newly_detailed = movies_after - movies_before
    assert report.stages["embeddings"]["movies"] == newly_detailed
    assert report.stages["embeddings"]["skipped"] == movies_before, "every existing film was reused"

    # The user's side changed not at all, so none of it is recomputed.
    assert report.stages["embeddings"]["reviews"] == 0
    assert report.stages["review_structuring"]["processed"] == 0, (
        "reviews must not be re-sent to the LLM"
    )
    assert report.stages["dossiers"]["generated"] == 0
    assert tmdb.detail_calls - detail_calls_before == newly_detailed
    assert len(claude.calls) - llm_calls_before < 5, "only the taste summary may be regenerated"


def test_resolution_does_not_re_search_recent_failures(built):
    """Films TMDB has no record of are not re-queried on every single update."""
    cfg, conn, tmdb, claude, backend = (
        built[k] for k in ("cfg", "conn", "tmdb", "claude", "backend")
    )
    conn.execute(
        "UPDATE user_films SET tmdb_id = NULL, resolved_at = datetime('now') WHERE film_key = "
        "(SELECT film_key FROM user_films LIMIT 1)"
    )
    searches_before = tmdb.search_calls
    run(cfg, kind="update", conn=conn, tmdb_client=tmdb, llm_client=claude, backend=backend)
    assert tmdb.search_calls == searches_before, "a fresh failure must wait out its cooldown"


def test_new_ratings_flow_through_an_update(built):
    """Add a rating to the export, update, and confirm only the delta is processed."""
    cfg, conn, tmdb, claude, backend = (
        built[k] for k in ("cfg", "conn", "tmdb", "claude", "backend")
    )
    export = sorted(cfg.data_dir.glob("letterboxd-*"))[-1]

    before_films = scalar(conn, "SELECT COUNT(*) FROM user_films")
    before_ratings = scalar(conn, "SELECT COUNT(*) FROM user_ratings")

    for name, line in [
        ("ratings.csv", "2026-08-27,A Brand New Film,2026,https://boxd.it/newf,4.5\n"),
        ("watched.csv", "2026-08-27,A Brand New Film,2026,https://boxd.it/newf\n"),
    ]:
        path = export / name
        path.write_text(path.read_text() + line, encoding="utf-8")

    report = run(
        cfg, kind="update", conn=conn, tmdb_client=tmdb, llm_client=claude, backend=backend
    )

    assert report.stages["letterboxd"]["films_new"] == 1
    assert report.stages["letterboxd"]["ratings_new"] == 1
    assert scalar(conn, "SELECT COUNT(*) FROM user_films") == before_films + 1
    assert scalar(conn, "SELECT COUNT(*) FROM user_ratings") == before_ratings + 1
    assert scalar(conn, "SELECT tmdb_id FROM user_films WHERE title='A Brand New Film'") is not None


def test_status_reports_a_built_database(built):
    out = status(built["cfg"])
    assert out["initialised"] is True
    assert out["films_rated"] > 50
    assert out["catalog_size"] > 100
    assert out["embeddings"] > 100


def test_pipeline_survives_a_missing_llm(tmp_config, conn, library_export):
    """Without Claude the system must still build and recommend, just with less insight."""
    shutil.copytree(library_export, tmp_config.data_dir / library_export.name)
    report = run(
        tmp_config,
        kind="setup",
        conn=conn,
        tmdb_client=FakeTMDBClient(n_movies=400),
        backend=HashBackend(dim=128),
        skip_llm=True,
    )
    assert report.ok
    assert scalar(conn, "SELECT COUNT(*) FROM review_facts") == 0

    engine = RecommendationEngine(conn, tmp_config, backend=HashBackend(dim=128), client=None)
    result = engine.recommend("", n=4, explain=False, use_llm=False)
    assert len(result.items) == 4


def test_missing_export_raises_a_clear_error(tmp_config, conn):
    with pytest.raises(FileNotFoundError, match="No Letterboxd export"):
        run(
            tmp_config,
            kind="setup",
            conn=conn,
            tmdb_client=FakeTMDBClient(n_movies=20),
            backend=HashBackend(dim=128),
            skip_llm=True,
        )


def test_readiness_explains_an_embedding_model_mismatch(built):
    """Changing the embedding model orphans every vector; say so plainly."""
    other = HashBackend(dim=64)  # a different model name entirely
    engine = RecommendationEngine(built["conn"], built["cfg"], backend=other, client=None)
    ready, reason = engine.readiness()
    assert ready is False
    assert "No embeddings for model" in reason
    assert built["backend"].name in reason, (
        "the reason should name what the database was built with"
    )


def test_readiness_is_clean_when_built(built):
    engine = RecommendationEngine(
        built["conn"], built["cfg"], backend=built["backend"], client=built["claude"]
    )
    assert engine.readiness() == (True, "")
