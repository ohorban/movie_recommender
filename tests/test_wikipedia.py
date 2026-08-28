"""Plot extraction and the guard against fetching the wrong article."""

from __future__ import annotations

import pytest

from movierec.db import insert_ignore, upsert
from movierec.ingest.wikipedia import (
    _plausible_article,
    _split_sections,
    extract_plot,
    pending_plot_ids,
    plot_coverage,
    record_miss,
)

BODY = (
    "Cooper, a widowed former NASA pilot, runs a farm with his family. His daughter Murph "
    "believes her bedroom is haunted. They discover the anomaly is gravitational and it gives "
    "coordinates to a secret NASA facility planning a mission through a wormhole. "
) * 3
ARTICLE = (
    f"Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.\n\n"
    f"== Plot ==\n{BODY}\n\n== Cast ==\nMatthew McConaughey as Cooper\n\n"
    f"== Reception ==\nThe film grossed a lot.\n"
)


def test_split_sections_finds_headings():
    names = [name for name, _ in _split_sections(ARTICLE)]
    assert "plot" in names and "cast" in names and "reception" in names


def test_extract_plot_takes_only_the_plot():
    plot = extract_plot(ARTICLE)
    assert "Cooper" in plot
    assert "McConaughey" not in plot
    assert "grossed" not in plot


def test_extract_plot_accepts_synopsis_heading():
    article = "A film.\n\n== Synopsis ==\n" + BODY
    assert "Cooper" in extract_plot(article)


def test_extract_plot_returns_empty_for_a_stub():
    assert extract_plot("Tiny is a film.\n\n== Plot ==\nShort.\n") == ""


def test_extract_plot_handles_no_headings():
    assert extract_plot("") == ""


def test_plausible_article_accepts_the_right_film():
    assert _plausible_article("Interstellar", 2014, "Interstellar (film)", ARTICLE)


def test_plausible_article_rejects_a_different_title():
    assert not _plausible_article("Tenet", 2020, "Interstellar (film)", ARTICLE)


def test_plausible_article_rejects_a_wrong_year():
    assert not _plausible_article("Interstellar", 1998, "Interstellar (film)", ARTICLE)


def test_plausible_article_rejects_a_non_film():
    band = "Interstellar is a Norwegian progressive rock band formed in 2003. " * 30
    assert not _plausible_article("Interstellar", 2014, "Interstellar (band)", band)


def test_plausible_article_tolerates_one_year_drift():
    assert _plausible_article("Interstellar", 2015, "Interstellar (film)", ARTICLE)


# --------------------------------------------------------------------------- #
# Fetch budgeting
#
# `wikipedia_limit` is a coverage target, not a per-run batch size. Treating it
# as a batch size means every update fetches another full batch and walks the
# whole catalog, hours at a time.
# --------------------------------------------------------------------------- #


@pytest.fixture
def catalog(conn):
    upsert(
        conn,
        "movies",
        [
            {
                "tmdb_id": i,
                "title": f"Film {i}",
                "year": 2000,
                "in_catalog": 1,
                "detail_level": 2,
                "tmdb_vote_count": 10_000 - i,
            }
            for i in range(1, 101)
        ],
        key=["tmdb_id"],
    )
    return conn


def _store_plot(conn, tmdb_id):
    insert_ignore(
        conn,
        "movie_texts",
        [
            {
                "tmdb_id": tmdb_id,
                "source": "wikipedia_plot",
                "text": "A plot. " * 40,
                "lang": "en",
                "url": None,
                "content_hash": f"h{tmdb_id}",
            }
        ],
    )


def test_target_is_a_coverage_target_not_a_batch_size(catalog):
    assert len(pending_plot_ids(catalog, target=30)) == 30
    for i in range(1, 31):
        _store_plot(catalog, i)
    assert plot_coverage(catalog) == 30
    # Coverage is met: a second run must not fetch another 30.
    assert pending_plot_ids(catalog, target=30) == []


def test_budget_is_the_remaining_shortfall(catalog):
    for i in range(1, 21):
        _store_plot(catalog, i)
    assert len(pending_plot_ids(catalog, target=30)) == 10


def test_misses_stop_consuming_the_budget(catalog):
    first = pending_plot_ids(catalog, target=5)
    assert len(first) == 5
    for tmdb_id in first:
        record_miss(catalog, tmdb_id)
    # Those five found no article. They must not be retried on the next run,
    # and must not block the budget either.
    second = pending_plot_ids(catalog, target=5)
    assert not set(second) & set(first)
    assert len(second) == 5


def test_your_own_films_are_exempt_from_the_budget(catalog):
    upsert(
        catalog,
        "user_films",
        [{"film_key": "mine|2000", "title": "Film 90", "year": 2000, "tmdb_id": 90}],
        key=["film_key"],
    )
    for i in range(1, 41):
        _store_plot(catalog, i)
    # Coverage target already met, but a film the user logged still gets fetched.
    assert pending_plot_ids(catalog, target=20) == [90]


def test_your_own_films_come_first(catalog):
    upsert(
        catalog,
        "user_films",
        [{"film_key": "mine|2000", "title": "Film 99", "year": 2000, "tmdb_id": 99}],
        key=["film_key"],
    )
    assert pending_plot_ids(catalog, target=5)[0] == 99, "low vote count, but it is theirs"
