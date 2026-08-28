"""Matching Letterboxd titles to TMDB - and refusing to guess when unsure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeTMDBClient
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
