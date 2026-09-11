"""Matching Letterboxd titles to TMDB - and refusing to guess when unsure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeTMDBClient, make_show
from movierec.db import fetch_all, scalar, upsert
from movierec.ingest.letterboxd import ingest_export
from movierec.ingest.resolve import (
    AUTO_ACCEPT,
    REVIEW_FLOOR,
    best_match,
    resolve_user_films,
    score_candidate,
    set_override,
    unresolved_report,
)
from movierec.ingest.tmdb import _tv_to_movie_shape
from movierec.media import internal_id, parse_tmdb_reference, split_id


def c(title, date, votes=1000, original=None):
    return {
        "id": 1,
        "title": title,
        "original_title": original or title,
        "release_date": date,
        "vote_count": votes,
    }


def test_exact_title_and_year_scores_top():
    assert score_candidate("Weapons", 2025, c("Weapons", "2025-08-08")) >= AUTO_ACCEPT


def test_accent_difference_still_matches():
    assert score_candidate("Amelie", 2001, c("Amélie", "2001-04-25", 11000)) >= AUTO_ACCEPT


def test_original_title_can_carry_the_match():
    score = score_candidate(
        "The Summit of the Gods",
        2021,
        c("The Summit of the Gods", "2021-09-29", 300, "Le Sommet des Dieux"),
    )
    assert score >= AUTO_ACCEPT


def test_one_year_drift_is_tolerated():
    assert score_candidate("Parasite", 2019, c("Parasite", "2020-01-01", 9000)) >= AUTO_ACCEPT


def test_wrong_year_is_rejected():
    assert score_candidate("Weapons", 2025, c("Weapons", "1989-01-01", 30)) < AUTO_ACCEPT


def test_sequel_does_not_match_the_original():
    assert (
        score_candidate("Top Gun: Maverick", 2022, c("Top Gun", "1986-05-16", 6000)) < AUTO_ACCEPT
    )


def test_unrelated_title_falls_below_the_review_floor():
    assert score_candidate("Interstellar", 2014, c("Zoolander", "2001-09-28", 4000)) < REVIEW_FLOOR


def test_ambiguous_runner_up_docks_confidence():
    client = FakeTMDBClient(n_movies=5)
    client.details[1] = {
        "id": 1,
        "title": "Echo",
        "original_title": "Echo",
        "release_date": "2010-01-01",
        "vote_count": 900,
        "popularity": 3,
    }
    client.details[2] = {
        "id": 2,
        "title": "Echo",
        "original_title": "Echo",
        "release_date": "2010-06-01",
        "vote_count": 880,
        "popularity": 3,
    }
    client.search = lambda title, year=None: [client.details[1], client.details[2]]  # type: ignore
    match = best_match(client, "Echo", 2010)
    assert match.confidence < AUTO_ACCEPT, "two identical candidates must not auto-accept"


def test_resolution_flags_low_confidence_for_review(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    client = FakeTMDBClient(n_movies=40)
    stats = resolve_user_films(conn, client)
    assert stats["attempted"] == 6
    assert scalar(conn, "SELECT COUNT(*) FROM user_films WHERE tmdb_id IS NOT NULL") > 0


def test_override_wins_and_survives_re_resolution(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    client = FakeTMDBClient(n_movies=40)
    resolve_user_films(conn, client)

    key = fetch_all(conn, "SELECT film_key FROM user_films ORDER BY film_key LIMIT 1")[0][
        "film_key"
    ]
    upsert(conn, "movies", [{"tmdb_id": 777, "title": "Pinned"}], key=["tmdb_id"])
    set_override(conn, key, 777, "test")

    assert scalar(conn, "SELECT tmdb_id FROM user_films WHERE film_key=?", (key,)) == 777
    resolve_user_films(conn, client, only_unresolved=False)
    assert scalar(conn, "SELECT tmdb_id FROM user_films WHERE film_key=?", (key,)) == 777
    assert scalar(conn, "SELECT needs_review FROM user_films WHERE film_key=?", (key,)) == 0


def test_unresolved_report_lists_what_needs_attention(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    conn.execute(
        "UPDATE user_films SET needs_review = 1, match_confidence = 0.7 WHERE title='Alpha'"
    )
    rows = unresolved_report(conn)
    assert any(r["title"] == "Alpha" for r in rows)


# --------------------------------------------------------------------------- #
# Year drift between sites
# --------------------------------------------------------------------------- #
# Letterboxd and TMDB routinely disagree about which year a film belongs to -
# festival premiere vs wide release, or a December release dated to the
# following year. Every one of these was flagged for manual review in the real
# database despite being the correct match.
def test_a_one_year_disagreement_does_not_block_an_identical_title():
    assert score_candidate("Obsession", 2025, c("Obsession", "2026-02-01", 300)) >= AUTO_ACCEPT
    assert score_candidate("Enemy", 2013, c("Enemy", "2014-03-14", 4000)) >= AUTO_ACCEPT


def test_a_two_year_disagreement_still_auto_accepts_an_identical_title():
    assert score_candidate("Am I OK?", 2022, c("Am I OK?", "2024-06-06", 200)) >= AUTO_ACCEPT


def test_a_remake_decades_later_is_not_auto_accepted():
    """The same title far apart in time is what a remake looks like.

    This is the counterweight to the tolerance above: it must buy leniency for
    a dating disagreement without waving through a different film.
    """
    score = score_candidate("Psycho", 1960, c("Psycho", "1998-12-04", 2000))
    assert REVIEW_FLOOR <= score < AUTO_ACCEPT


# --------------------------------------------------------------------------- #
# Titles that merely contain the query
# --------------------------------------------------------------------------- #
# `fuzz.WRatio` has a partial-ratio arm, so a short title inside a much longer
# one scores near 0.9. Each of these pairs was a real wrong match.
def test_a_longer_title_containing_the_query_is_not_a_match():
    cases = [
        ("Baby Reindeer", 2024, "A Baby Reindeer's First Christmas", "2020-12-24", 1),
        ("Midnight Mass", 2021, "The Manson Brothers: Midnight Zombie Massacre", "2021-01-01", 40),
        ("Adolescence", 2025, "The Real Adolescence: Our Killer Kids", "2025-01-01", 10),
        ("Chernobyl", 2019, "Chernobyl: Zone of Exclusion", "2019-01-01", 60),
        ("Maid", 2021, "Snow Maiden Against Everyone", "2021-01-01", 30),
    ]
    for title, year, cand_title, date, votes in cases:
        score = score_candidate(title, year, c(cand_title, date, votes))
        assert score < REVIEW_FLOOR, f"{title!r} should not match {cand_title!r} (scored {score})"


def test_the_length_guard_does_not_punish_a_genuine_subtitle():
    """A real title is allowed to be longer than the query when it is the work.

    Letterboxd frequently drops a subtitle TMDB keeps, so the guard has to
    leave those alone.
    """
    score = score_candidate("Dune: Part Two", 2024, c("Dune: Part Two", "2024-02-27", 5000))
    assert score >= AUTO_ACCEPT


# --------------------------------------------------------------------------- #
# Television
# --------------------------------------------------------------------------- #
def test_a_show_is_matched_against_the_tv_index():
    """Letterboxd logs shows; TMDB's movie index does not contain them.

    Without a TV search the only candidates are films that share a word with
    the title, which is how Baby Reindeer ended up as a Christmas short.
    """
    client = FakeTMDBClient(n_movies=5)
    show = client.add_show(259265, "Baby Reindeer", year=2024)
    client.search = lambda title, year=None: [  # type: ignore[method-assign]
        {
            "id": 780411,
            "title": "A Baby Reindeer's First Christmas",
            "original_title": "A Baby Reindeer's First Christmas",
            "release_date": "2020-12-24",
            "vote_count": 1,
            "media_type": "movie",
        }
    ]
    match = best_match(client, "Baby Reindeer", 2024)
    assert match.tmdb_id == internal_id("tv", show["id"])
    assert match.confidence >= AUTO_ACCEPT
    assert match.method.endswith(":tv")


def test_a_tv_id_does_not_collide_with_a_movie_id():
    """`/movie/1399` and `/tv/1399` are different titles.

    Every table is keyed on one integer, so the two namespaces have to be kept
    apart before anything is written.
    """
    assert internal_id("movie", 1399) != internal_id("tv", 1399)
    assert split_id(internal_id("tv", 1399)) == ("tv", 1399)
    assert split_id(internal_id("movie", 1399)) == ("movie", 1399)


def test_a_show_payload_is_reshaped_into_the_film_fields():
    """The rest of the pipeline never learns that television exists."""
    shaped = _tv_to_movie_shape(make_show(259265, name="Baby Reindeer", year=2024))
    assert shaped["title"] == "Baby Reindeer"
    assert shaped["release_date"].startswith("2024")
    assert shaped["runtime"] in (28, 42, 55)
    assert shaped["id"] == internal_id("tv", 259265)
    assert shaped["source_tmdb_id"] == 259265
    # A show's creator is its closest analogue to a director, and the director
    # affinity is one of the ranker's features.
    directors = [p for p in shaped["credits"]["crew"] if p["job"] == "Director"]
    assert directors and directors[0]["name"].startswith("Creator")
    assert shaped["credits"]["cast"][0]["character"] == "Role 0"
    # /movie puts keywords under "keywords", /tv under "results". The writer
    # reads one of those, so the shim has to reconcile them or every show
    # silently loses its keywords - one of the ranker's affinity features.
    assert shaped["keywords"]["keywords"], "a show must keep its keywords"


def test_a_show_resolves_and_is_stored_with_its_namespace(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    upsert(
        conn,
        "user_films",
        [{"film_key": "baby-reindeer|2024", "title": "Baby Reindeer", "year": 2024}],
        key=["film_key"],
    )
    client = FakeTMDBClient(n_movies=10)
    client.add_show(259265, "Baby Reindeer", year=2024)
    client.search = lambda title, year=None: []  # type: ignore[method-assign]
    resolve_user_films(conn, client)
    row = fetch_all(
        conn,
        "SELECT uf.tmdb_id, m.media_type, m.source_tmdb_id FROM user_films uf "
        "JOIN movies m ON m.tmdb_id = uf.tmdb_id WHERE uf.film_key = 'baby-reindeer|2024'",
    )
    assert row and row[0]["media_type"] == "tv"
    assert row[0]["source_tmdb_id"] == 259265
    assert row[0]["tmdb_id"] == internal_id("tv", 259265)


# --------------------------------------------------------------------------- #
# Pasted references
# --------------------------------------------------------------------------- #
def test_a_pasted_tmdb_url_carries_its_media_type():
    url = "https://www.themoviedb.org/tv/259265-something-very-bad?language=en-US"
    assert parse_tmdb_reference(url) == ("tv", 259265)
    assert parse_tmdb_reference("https://www.themoviedb.org/movie/1124") == ("movie", 1124)
    assert parse_tmdb_reference("1124") == ("movie", 1124)
    assert parse_tmdb_reference("tv/259265") == ("tv", 259265)
    assert parse_tmdb_reference("") is None
    assert parse_tmdb_reference("not an id") is None
