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
from movierec.media import internal_id  # noqa: E402

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


# --------------------------------------------------------------------------- #
# The recommendation card
# --------------------------------------------------------------------------- #
def test_there_is_no_thumbs_up(app):
    """A button whose effect on the recommendations is not legible is worse
    than no button. Ratings and reviews are what the taste model learns from."""
    labels = {(b.label or "") for b in app.button}
    keys = {(b.key or "") for b in app.button}
    assert "👍" not in labels
    assert not any(k.endswith("up") or k[-2:] == "up" for k in keys if k.startswith(("t0", "ta")))


def test_the_snooze_button_says_what_it_does(app):
    """It was a bare 👎, which reads as "this is bad" rather than "hide it"."""
    snooze = [b for b in app.button if b.key and b.key.startswith("t0dn")]
    assert snooze, "the hero card should offer a way to dismiss a film"
    assert snooze[0].label == "Not interested"


def test_the_card_names_the_director(app):
    cards = [str(m.value) for m in app.markdown if "rec-meta" in str(m.value)]
    assert cards, "a recommendation card should render its metadata line"
    assert any("dir. " in c for c in cards), "the director belongs on the card"


def test_the_card_links_out_to_every_site(app):
    """TMDB alone is not where this viewer keeps their film life."""
    urls = " ".join(str(getattr(b, "proto", b)) for b in app.get("link_button"))
    assert "themoviedb.org" in urls
    assert "imdb.com" in urls
    assert "letterboxd.com" in urls


# --------------------------------------------------------------------------- #
# Correcting a bad match
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_with_unmatched(built_app_db, monkeypatch):
    """The app, with one film deliberately left unresolved to correct."""
    import streamlit as st

    from movierec.db import init_db

    conn = init_db(built_app_db.db_path)
    conn.execute(
        "INSERT OR REPLACE INTO user_films (film_key, title, year, tmdb_id, match_confidence,"
        " match_method, needs_review) VALUES ('needs-a-fix|2026', 'Needs A Fix', 2026, NULL,"
        " 0.0, 'no-results', 1)"
    )
    conn.commit()
    conn.close()

    st.cache_resource.clear()
    for key, value in {
        "MOVIEREC_DB_PATH": str(built_app_db.db_path),
        "MOVIEREC_DATA_DIR": str(built_app_db.data_dir),
        "MOVIEREC_EMBED_BACKEND": "hash",
        "TMDB_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "",
    }.items():
        monkeypatch.setenv(key, value)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _fix_form(at):
    key = "needs-a-fix|2026"
    box = next(t for t in at.text_input if t.key == f"fx{key}")
    submit = next(b for b in at.button if b.key and b.key.endswith(f"form{key}-Pin this id"))
    return box, submit


def test_pasting_a_tv_url_pins_the_show(app_with_unmatched, built_app_db):
    """The reported bug: pasting an id moved the confidence to 1.00 and left
    the film on "no match found" forever.

    Two causes. The value never reached the handler, because a plain widget
    beside a plain button hands over the click before the typed text commits;
    and the handler then wrote a *null* override, which pinned the film as
    unmatched permanently and was re-applied on every subsequent run.
    """
    from movierec.db import fetch_all, init_db

    box, submit = _fix_form(app_with_unmatched)
    box.set_value("https://www.themoviedb.org/tv/259265-something-very-bad?language=en-US")
    after = submit.click().run()
    _no_exceptions(after, "pinning a TMDB id")

    conn = init_db(built_app_db.db_path)
    row = fetch_all(
        conn,
        "SELECT tmdb_id, match_method FROM user_films WHERE film_key = 'needs-a-fix|2026'",
    )[0]
    assert row["tmdb_id"] == internal_id("tv", 259265)
    assert row["match_method"] == "override"
    stored = fetch_all(
        conn, "SELECT media_type, source_tmdb_id FROM movies WHERE tmdb_id = ?", (row["tmdb_id"],)
    )[0]
    assert stored["media_type"] == "tv"
    assert stored["source_tmdb_id"] == 259265
    conn.close()


def test_an_unreadable_paste_never_clears_the_match(app_with_unmatched, built_app_db):
    from movierec.db import fetch_all, init_db, scalar

    box, submit = _fix_form(app_with_unmatched)
    box.set_value("no idea")
    after = submit.click().run()
    _no_exceptions(after, "pinning nonsense")
    assert any("No TMDB id" in str(e.value) for e in after.error), "the user must be told"

    conn = init_db(built_app_db.db_path)
    assert scalar(conn, "SELECT COUNT(*) FROM title_overrides WHERE tmdb_id IS NULL") == 0
    row = fetch_all(
        conn, "SELECT match_method FROM user_films WHERE film_key = 'needs-a-fix|2026'"
    )[0]
    assert row["match_method"] != "override", "nonsense must not be recorded as a correction"
    conn.close()


# --------------------------------------------------------------------------- #
# Tonight is computed once, not on every page load
# --------------------------------------------------------------------------- #
def _count_recommend_calls(monkeypatch):
    from movierec.recommend.engine import RecommendationEngine

    calls: list[str] = []
    original = RecommendationEngine.recommend

    def counted(self, text="", **kw):
        calls.append(text)
        return original(self, text, **kw)

    monkeypatch.setattr(RecommendationEngine, "recommend", counted)
    return calls


def _titles(at):
    return [str(m.value) for m in at.markdown if "rec-title" in str(m.value)]


def _forget_stored_pick(built_app_db):
    """The fixture database is shared across the module, and the pick is stored
    in it on purpose - so a test about *when* it is computed has to start from
    a known state."""
    from movierec.db import init_db

    conn = init_db(built_app_db.db_path)
    conn.execute("DELETE FROM kv WHERE key = 'tonight'")
    conn.commit()
    conn.close()


def test_reloading_the_page_does_not_recompute_tonight(built_app_db, monkeypatch):
    """Ranking 30k films and writing the notes is seconds of work and a paid
    Claude call, and none of it is random - so a refresh used to buy an
    identical answer twice."""
    import streamlit as st

    for key, value in {
        "MOVIEREC_DB_PATH": str(built_app_db.db_path),
        "MOVIEREC_DATA_DIR": str(built_app_db.data_dir),
        "MOVIEREC_EMBED_BACKEND": "hash",
        "TMDB_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "",
    }.items():
        monkeypatch.setenv(key, value)
    _forget_stored_pick(built_app_db)
    calls = _count_recommend_calls(monkeypatch)

    st.cache_resource.clear()
    first = AppTest.from_file(APP, default_timeout=180)
    first.run()
    _no_exceptions(first, "first load")
    assert len(calls) == 1, "the first load has to actually compute something"

    # A brand-new session, as a browser refresh or a restarted server would be.
    st.cache_resource.clear()
    second = AppTest.from_file(APP, default_timeout=180)
    second.run()
    _no_exceptions(second, "second load")

    assert len(calls) == 1, f"a reload must reuse the stored pick, got {len(calls)} computations"
    assert _titles(second) == _titles(first)


def test_reroll_still_recomputes(built_app_db, monkeypatch):
    """The cache must not turn Reroll into a no-op."""
    import streamlit as st

    for key, value in {
        "MOVIEREC_DB_PATH": str(built_app_db.db_path),
        "MOVIEREC_DATA_DIR": str(built_app_db.data_dir),
        "MOVIEREC_EMBED_BACKEND": "hash",
        "TMDB_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "",
    }.items():
        monkeypatch.setenv(key, value)
    _forget_stored_pick(built_app_db)
    calls = _count_recommend_calls(monkeypatch)

    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    before = _titles(at)
    reroll = next(b for b in at.button if "Reroll" in (b.label or ""))
    after = reroll.click().run()
    _no_exceptions(after, "reroll")
    assert len(calls) == 2
    assert _titles(after) != before
