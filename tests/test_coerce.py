"""Normalising LLM payloads.

Every case here was observed in a single real run of 127 reviews and 413
dossiers. A tool-use schema constrains what the model is *asked* for, not what
it returns.
"""

from __future__ import annotations

import pytest

from movierec.enrich.coerce import (
    as_aspects,
    as_obj,
    as_scales,
    as_str,
    as_str_list,
    as_unit,
    normalize_dossier,
    normalize_pitches,
    normalize_review_facts,
    normalize_summary,
)
from movierec.enrich.structuring import DOSSIER_SCALES


# --------------------------------------------------------------------------- #
# The three deviations seen in production
# --------------------------------------------------------------------------- #
def test_a_json_string_inside_an_array_of_objects():
    """The crash: one `liked` entry arrived as a JSON string, not an object."""
    payload = {
        "liked": [
            {"aspect": "engaging quality", "category": "pacing", "strength": 0.6},
            '{"aspect": "comforting despite subject matter", "category": "emotional_impact", "strength": 0.7}',
        ]
    }
    aspects = normalize_review_facts(payload)["liked"]
    assert len(aspects) == 2
    assert aspects[1]["aspect"] == "comforting despite subject matter"
    assert aspects[1]["category"] == "emotional_impact"
    assert aspects[1]["strength"] == pytest.approx(0.7)


def test_a_bare_string_where_a_list_was_declared():
    """The silent one: `tone` as a string was iterated character by character."""
    dossier = normalize_dossier({"tone": "tense", "themes": "grief"}, DOSSIER_SCALES)
    assert dossier["tone"] == ["tense"], "must not become ['t','e','n','s','e']"
    assert dossier["themes"] == ["grief"]


def test_null_for_a_required_array():
    facts = normalize_review_facts(
        {"liked": None, "disliked": None, "taste_signals": None, "engagement": None}
    )
    assert facts["liked"] == [] and facts["disliked"] == []
    assert facts["taste_signals"] == []
    assert set(facts["engagement"]) == {"intellectual", "emotional", "visceral"}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (["a", "b"], ["a", "b"]),
        ("single", ["single"]),
        (None, []),
        ([], []),
        (["a", "", "  ", "b"], ["a", "b"]),
        ([{"name": "from-dict"}], ["from-dict"]),
        ([1, 2], ["1", "2"]),
    ],
)
def test_as_str_list(value, expected):
    assert as_str_list(value) == expected


def test_as_obj_parses_a_json_object_string():
    assert as_obj('{"a": 1}') == {"a": 1}
    assert as_obj("not json") == {}
    assert as_obj('["a"]') == {}, "an array is not an object"
    assert as_obj(None) == {}


@pytest.mark.parametrize(
    "value,expected", [(0.5, 0.5), ("0.5", 0.5), (2.0, 1.0), (-1.0, 0.0), (None, 0.5), ("x", 0.5)]
)
def test_as_unit_clamps(value, expected):
    assert as_unit(value) == pytest.approx(expected)


def test_as_str_rejects_structures():
    assert as_str({"a": 1}) == ""
    assert as_str(["a"]) == ""
    assert as_str("  padded  ") == "padded"


def test_as_aspects_accepts_a_bare_phrase():
    out = as_aspects(["great ending"])
    assert out == [{"aspect": "great ending", "category": "other", "strength": 0.5}]


def test_as_aspects_drops_unusable_entries():
    assert as_aspects([{"category": "pacing"}, None, 7, ""]) == []


def test_as_scales_keeps_only_known_keys():
    scales = as_scales({"darkness": 0.8, "made_up": 0.5, "humor": "0.2"}, DOSSIER_SCALES)
    assert scales == {"darkness": pytest.approx(0.8), "humor": pytest.approx(0.2)}


# --------------------------------------------------------------------------- #
# Payload normalisers
# --------------------------------------------------------------------------- #
def test_review_facts_constrains_the_enum_and_ranges():
    facts = normalize_review_facts({"verdict": "ADORED", "sentiment": 9.0, "signal_strength": -3})
    assert facts["verdict"] == "mixed"
    assert facts["sentiment"] == pytest.approx(1.0)
    assert facts["signal_strength"] == pytest.approx(0.0)


def test_review_facts_drops_trivial_taste_signals():
    facts = normalize_review_facts({"taste_signals": ["short", "a properly generalisable signal"]})
    assert facts["taste_signals"] == ["a properly generalisable signal"]


def test_dossier_constrains_pacing():
    assert normalize_dossier({"pacing": "breakneck"}, DOSSIER_SCALES)["pacing"] == "measured"
    assert normalize_dossier({"pacing": "Brisk"}, DOSSIER_SCALES)["pacing"] == "brisk"


def test_normalize_pitches_indexes_by_id_and_tolerates_junk():
    payload = {
        "pitches": [
            {"tmdb_id": 27205, "hook": "h", "because": "b", "caveat": ""},
            {"tmdb_id": "550", "hook": "h2"},
            {"hook": "no id"},
            "a bare string",
        ]
    }
    out = normalize_pitches(payload)
    assert set(out) == {27205, 550}
    assert out[550]["because"] == ""


def test_normalize_summary_handles_strings_for_lists():
    summary = normalize_summary({"headline": "H", "loves": "originality", "dislikes": None})
    assert summary["loves"] == ["originality"]
    assert summary["dislikes"] == []


def test_everything_survives_total_garbage():
    for bad in (None, "", [], 42, "just prose"):
        assert normalize_review_facts(bad)["liked"] == []
        assert normalize_dossier(bad, DOSSIER_SCALES)["tone"] == []
        assert normalize_pitches(bad) == {}
        assert normalize_summary(bad)["loves"] == []
