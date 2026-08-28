"""Preference scoring, affinity shrinkage and taste-mode clustering."""

from __future__ import annotations

from movierec.taste.profile import (
    aggregate_review_facts,
    compute_affinities,
    preference_scores,
    runtime_bucket,
    scale_preferences,
)


def rating_rows(pairs, liked=()):
    return [{"tmdb_id": i, "rating": r, "liked": 1 if i in liked else None} for i, r in pairs]


def test_preference_is_centred_on_the_users_own_mean():
    prefs, mean, _std = preference_scores(rating_rows([(1, 1.0), (2, 3.0), (3, 5.0)]))
    assert mean == 3.0
    assert prefs[2] == 0.0  # their average film scores zero
    assert prefs[3] > 0 > prefs[1]


def test_a_harsh_rater_still_has_positive_preferences():
    # Someone whose mean is 2.5 genuinely likes a 3.5.
    prefs, mean, _ = preference_scores(
        rating_rows([(i, r) for i, r in enumerate([1, 2, 2, 2.5, 3, 3.5], 1)])
    )
    assert mean < 3.0
    assert prefs[6] > 0.5


def test_an_explicit_like_adds_signal():
    plain, _, _ = preference_scores(rating_rows([(1, 4.0), (2, 3.0), (3, 2.0)]))
    with_like, _, _ = preference_scores(rating_rows([(1, 4.0), (2, 3.0), (3, 2.0)], liked={1}))
    assert with_like[1] > plain[1]


def test_preference_scores_handles_empty():
    prefs, mean, std = preference_scores([])
    assert prefs == {}
    assert mean == 3.0
    assert std == 1.0


def test_affinity_shrinks_toward_zero_with_little_evidence():
    # One film with 'Rare', eight with 'Common', identical preference.
    prefs = dict.fromkeys(range(1, 10), 2.0)
    facets = {1: {"genre": ["Rare"]}, **{i: {"genre": ["Common"]} for i in range(2, 10)}}
    aff = compute_affinities(prefs, facets)["genre"]
    assert aff["Common"] > aff["Rare"], "a pattern seen once must not outrank one seen eight times"
    assert aff["Rare"] < 2.0


def test_affinity_captures_direction():
    prefs = {1: 1.5, 2: 1.5, 3: 1.5, 4: -1.5, 5: -1.5, 6: -1.5}
    facets = {
        **{i: {"genre": ["Good"]} for i in (1, 2, 3)},
        **{i: {"genre": ["Bad"]} for i in (4, 5, 6)},
    }
    aff = compute_affinities(prefs, facets)["genre"]
    assert aff["Good"] > 0 > aff["Bad"]


def test_runtime_buckets():
    assert runtime_bucket(80) == "under-85"
    assert runtime_bucket(100) == "85-105"
    assert runtime_bucket(180) == "over-160"
    assert runtime_bucket(None) == "unknown"


def test_review_aggregation_signs_and_signals():
    rows = [
        {
            "facts": {
                "signal_strength": 1.0,
                "liked": [{"aspect": "premise", "category": "originality", "strength": 0.9}],
                "disliked": [{"aspect": "drag", "category": "pacing", "strength": 0.8}],
                "taste_signals": ["values originality of premise over execution polish"],
            }
        },
        {
            "facts": {
                "signal_strength": 0.8,
                "liked": [{"aspect": "idea", "category": "originality", "strength": 0.7}],
                "disliked": [],
                "taste_signals": ["values originality of premise over execution polish"],
            }
        },
    ]
    aff, signals = aggregate_review_facts(rows)
    assert aff["originality"] > 0 > aff["pacing"]
    assert len(signals) == 1, "duplicate signals must be de-duplicated"


def test_review_aggregation_handles_empty():
    aff, signals = aggregate_review_facts([])
    assert aff == {} and signals == []


def test_scale_preferences_needs_enough_evidence():
    prefs = dict.fromkeys(range(5), 0.5)
    dossiers = {i: {"scales": {"darkness": 0.5}} for i in range(5)}
    assert scale_preferences(prefs, dossiers) == ({}, {})


def test_scale_preferences_finds_a_correlated_dimension():
    # Preference tracks darkness exactly: the weight should be high.
    prefs = {i: (i / 20.0) - 0.5 for i in range(20)}
    dossiers = {i: {"scales": {"darkness": i / 20.0, "humor": 0.5}} for i in range(20)}
    targets, weights = scale_preferences(prefs, dossiers)
    assert weights["darkness"] > weights.get("humor", 0.0)
    assert targets["darkness"] > 0.5, "they prefer the darker end, so the target should sit high"
