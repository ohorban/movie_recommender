"""Title normalisation is the join key for the whole user side of the database."""

from __future__ import annotations

import pytest

from movierec.text_utils import (
    clean_ws,
    film_key,
    match_variants,
    normalize_title,
    parse_year,
    truncate,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Amélie", "amelie"),
        ("WALL·E", "wall e"),
        ("Se7en", "se7en"),
        ("The Lord of the Rings: The Two Towers", "the lord of the rings the two towers"),
        ("Fast & Furious", "fast and furious"),
        ("  Extra   Spaces  ", "extra spaces"),
        ("Læther", "laether"),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def test_film_key_is_stable_across_punctuation_and_accents():
    assert film_key("Amélie", 2001) == film_key("Amelie", "2001")
    assert film_key("Spider-Man: No Way Home", 2021) == film_key("Spider Man  No Way Home", 2021)


def test_film_key_separates_remakes():
    assert film_key("Dune", 1984) != film_key("Dune", 2021)


def test_film_key_handles_missing_year():
    assert film_key("Untitled", None) == "untitled"
    assert film_key("Untitled", "") == "untitled"


def test_match_variants_drops_articles_and_expands_numerals():
    variants = match_variants("The Godfather Part II")
    assert "the godfather part ii" in variants
    assert "godfather part ii" in variants
    assert "the godfather part 2" in variants


def test_match_variants_includes_prefix_before_colon():
    assert "top gun" in match_variants("Top Gun: Maverick")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2014", 2014),
        ("2014-06-01", 2014),
        ("", None),
        (None, None),
        ("nope", None),
        ("1500", None),
    ],
)
def test_parse_year(value, expected):
    assert parse_year(value) == expected


def test_truncate_breaks_on_a_word_boundary():
    out = truncate("the quick brown fox jumps over the lazy dog", 20)
    assert len(out) <= 21 and not out.rstrip("…").endswith(" ")


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_clean_ws_collapses_nbsp():
    assert clean_ws("a  b\n c") == "a b c"
