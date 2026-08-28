"""Streamlit interface tests that actually click things.

Rendering the app proves almost nothing. The bug that prompted these tests —
`st.session_state["ask_query"] = ...` after the widget owning that key had been
created — raises only when the button is pressed, so a suite that merely called
`AppTest.run()` reported a healthy app while a whole tab was broken.

`test_every_button_survives_a_click` is the general guard: it presses every
button on every tab, one per fresh run, and fails on any exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")

pytest.importorskip("streamlit", reason="the app extra is not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fakes import FakeClaudeClient, FakeTMDBClient  # noqa: E402

# Pressing this runs the whole ingestion pipeline; not something to fire blindly.
DESTRUCTIVE = {"🔄 Update database"}


@pytest.fixture(scope="module")
def built_app_db(tmp_path_factory):
    """A small but complete database for the UI to read. Built once."""
    import shutil

    from conftest import _find_real_export, generate_export
    from movierec.config import Config
    from movierec.db import init_db
    from movierec.enrich.embeddings import make_backend
    from movierec.pipeline import run

    root = tmp_path_factory.mktemp("appdb")
    cfg = Config(
        root=root,
        tmdb_api_key="test-key",
        anthropic_api_key="",
        embed_backend="hash",
        catalog_size=400,
        min_votes=50,
        min_year=2000,
        enable_movielens=False,
        enable_imdb=False,
        enable_wikipedia=False,
        candidates_per_source=60,
        db_path=root / "db" / "app.db",
        data_dir=root / "data",
    )
    cfg.ensure_dirs()

    export = _find_real_export() or generate_export(root / "_export_source")
    shutil.copytree(export, cfg.data_dir / export.name)

    run(
        cfg,
        kind="setup",
        conn=init_db(cfg.db_path),
        tmdb_client=FakeTMDBClient(n_movies=500),
        llm_client=FakeClaudeClient(),
        # Must be the backend the app itself resolves: embeddings are keyed by
        # model name, so a different dimension here reads as "not built yet".
        backend=make_backend(cfg),
    )
    return cfg


@pytest.fixture
def app(built_app_db, monkeypatch):
    """A freshly run app pointed at that database, with caches cleared."""
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("MOVIEREC_DB_PATH", str(built_app_db.db_path))
    monkeypatch.setenv("MOVIEREC_DATA_DIR", str(built_app_db.data_dir))
    monkeypatch.setenv("MOVIEREC_EMBED_BACKEND", "hash")
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # no network from the UI tests

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _no_exceptions(at: AppTest, context: str = "") -> None:
    if at.exception:
        raise AssertionError(f"{context}: {at.exception[0].value}")


# --------------------------------------------------------------------------- #
# The regression
# --------------------------------------------------------------------------- #
def test_clicking_an_example_chip_prefills_the_box(app):
    """`st.session_state[widget_key] = ...` after the widget exists is illegal.

    This raised StreamlitAPIException on every example chip in the Ask tab.
    """
    chips = [b for b in app.button if b.key and b.key.startswith("ex_")]
    assert chips, "the Ask tab should offer example queries"

    after = chips[0].click().run()
    _no_exceptions(after, "clicking an example chip")
    assert after.session_state["ask_query"], "the chip should fill the query box"


def test_every_example_chip_works(app):
    for chip in [b for b in app.button if b.key and b.key.startswith("ex_")]:
        after = chip.click().run()
        _no_exceptions(after, f"chip {chip.label!r}")


# --------------------------------------------------------------------------- #
# The general guard
# --------------------------------------------------------------------------- #
def test_every_button_survives_a_click(app):
    """Press every button, one per fresh run, and fail on any exception."""
    labels = [(b.key, b.label) for b in app.button if b.label not in DESTRUCTIVE]
    assert len(labels) > 5, "expected a populated UI to test against"

    failures: list[str] = []
    for key, label in labels:
        run = AppTest.from_file(APP, default_timeout=180)
        run.run()
        target = next((b for b in run.button if b.key == key), None)
        if target is None:
            continue  # the widget tree shifted; covered by another case
        after = target.click().run()
        if after.exception:
            failures.append(f"{label!r} ({key}): {after.exception[0].value}")
    assert not failures, "buttons raised:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------- #
# Tabs and core interactions
# --------------------------------------------------------------------------- #
def test_app_renders_without_error(app):
    _no_exceptions(app, "initial render")
    # `at.tabs` is flat and includes nested tabs, so check the labels we own.
    labels = {t.label for t in app.tabs}
    assert {"Tonight", "Ask", "Insights", "Data"} <= labels
    assert app.title[0].value == "What should I watch?"


def test_tonight_shows_a_recommendation(app):
    cards = [m for m in app.markdown if "rec-title" in str(m.value)]
    assert len(cards) >= 2, "expected a hero pick plus alternatives"


def test_reroll_produces_different_picks(app):
    before = {m.value for m in app.markdown if "rec-title" in str(m.value)}
    reroll = next(b for b in app.button if "Reroll" in (b.label or ""))
    after = reroll.click().run()
    _no_exceptions(after, "reroll")
    now = {m.value for m in after.markdown if "rec-title" in str(m.value)}
    assert now and now != before, "reroll should surface something new"


def test_thumbs_down_records_feedback_and_hides_the_film(app, built_app_db):
    from movierec.db import init_db, scalar

    down = next(b for b in app.button if b.key and b.key.startswith("t0dn"))
    tmdb_id = int(down.key.replace("t0dn", ""))
    after = down.click().run()
    _no_exceptions(after, "thumbs down")

    conn = init_db(built_app_db.db_path)
    assert (
        scalar(
            conn, "SELECT COUNT(*) FROM feedback WHERE tmdb_id=? AND action='dislike'", (tmdb_id,)
        )
        == 1
    )
    conn.close()


def test_insights_tab_is_populated(app):
    labels = {m.label for m in app.metric}
    assert {"Films watched", "Rated", "Reviews"} <= labels
    assert "Rank correlation" in labels, "model diagnostics should be shown"


def test_data_tab_lists_the_export(app):
    assert any("Current export" in str(i.value) for i in app.info)


def test_no_destructive_button_fires_on_load(app, built_app_db):
    """Rendering must not trigger an update; that would run the pipeline."""
    from movierec.db import init_db, scalar

    conn = init_db(built_app_db.db_path)
    runs = scalar(conn, "SELECT COUNT(*) FROM ingest_runs")
    conn.close()
    app.run()
    conn = init_db(built_app_db.db_path)
    assert scalar(conn, "SELECT COUNT(*) FROM ingest_runs") == runs
    conn.close()
