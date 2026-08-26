"""Plot extraction and the guard against fetching the wrong article."""

from __future__ import annotations

from movierec.ingest.wikipedia import _plausible_article, _split_sections, extract_plot

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
