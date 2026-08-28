"""Normalise LLM payloads into the shapes the rest of the system expects.

A tool-use `input_schema` tells the model what to produce; it does not
guarantee it. Observed deviations on a single 127-review / 413-dossier run:

* a required array field returned as ``null`` (11 of 127 reviews)
* an array of objects containing one entry that was a *JSON string* rather
  than an object — the crash that took the pipeline down
* ``tone`` and ``themes`` returned as a bare string instead of a list of
  strings (96 of 413 dossiers), which never crashed: it silently iterated the
  string character by character and produced "t, e, n, s, e" in the UI

So every payload is funnelled through here, on the way in *and* on the way out
of the database. Normalising on read matters as much as on write: it repairs
records already stored without paying to regenerate them.
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_ASPECT_CATEGORY = "other"
DEFAULT_STRENGTH = 0.5


def as_obj(value: Any) -> dict[str, Any]:
    """A dict, parsing a JSON-object string if that is what we were handed."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
    return {}


def as_list(value: Any) -> list[Any]:
    """A list. A bare scalar becomes a one-element list; None becomes empty."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_str_list(value: Any) -> list[str]:
    """A list of non-empty strings.

    A bare string is wrapped, not iterated - the bug that turned "tense" into
    five single-character tags.
    """
    out: list[str] = []
    for item in as_list(value):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("name") or item.get("value") or item.get("tag") or "").strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def as_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return str(value)
    return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def as_unit(value: Any, default: float = DEFAULT_STRENGTH) -> float:
    """A float clamped to 0-1."""
    number = as_float(value, default)
    if number is None:
        number = default
    return max(0.0, min(1.0, float(number)))


def as_aspects(value: Any) -> list[dict[str, Any]]:
    """A list of ``{aspect, category, strength}`` records.

    Tolerates entries that arrive as JSON strings or as bare phrases.
    """
    out: list[dict[str, Any]] = []
    for item in as_list(value):
        record = as_obj(item)
        if not record and isinstance(item, str) and item.strip():
            record = {"aspect": item.strip()}
        if not record:
            continue
        aspect = as_str(record.get("aspect"))
        if not aspect:
            continue
        out.append(
            {
                "aspect": aspect,
                "category": as_str(record.get("category"), DEFAULT_ASPECT_CATEGORY)
                or DEFAULT_ASPECT_CATEGORY,
                "strength": as_unit(record.get("strength")),
            }
        )
    return out


def as_scales(value: Any, keys: list[str]) -> dict[str, float]:
    """A mapping of scale name to a 0-1 float, keeping only recognised keys."""
    record = as_obj(value)
    out: dict[str, float] = {}
    for key in keys:
        number = as_float(record.get(key))
        if number is not None:
            out[key] = max(0.0, min(1.0, number))
    return out


# --------------------------------------------------------------------------- #
# Payload-level normalisers
# --------------------------------------------------------------------------- #
VERDICTS = {"loved", "liked", "mixed", "disliked", "hated"}
PACINGS = {"slow", "measured", "brisk", "relentless"}


def normalize_review_facts(payload: Any) -> dict[str, Any]:
    facts = as_obj(payload)
    engagement = as_obj(facts.get("engagement"))
    verdict = as_str(facts.get("verdict"), "mixed").lower()
    sentiment = as_float(facts.get("sentiment"), 0.0) or 0.0
    return {
        "verdict": verdict if verdict in VERDICTS else "mixed",
        "sentiment": max(-1.0, min(1.0, sentiment)),
        "signal_strength": as_unit(facts.get("signal_strength")),
        "liked": as_aspects(facts.get("liked")),
        "disliked": as_aspects(facts.get("disliked")),
        "themes": as_str_list(facts.get("themes")),
        "tone_words": as_str_list(facts.get("tone_words")),
        "engagement": {
            "intellectual": as_unit(engagement.get("intellectual")),
            "emotional": as_unit(engagement.get("emotional")),
            "visceral": as_unit(engagement.get("visceral")),
        },
        "taste_signals": [s for s in as_str_list(facts.get("taste_signals")) if len(s) > 12],
    }


def normalize_dossier(payload: Any, scale_keys: list[str]) -> dict[str, Any]:
    dossier = as_obj(payload)
    pacing = as_str(dossier.get("pacing"), "measured").lower()
    return {
        "logline": as_str(dossier.get("logline")),
        "tone": as_str_list(dossier.get("tone")),
        "themes": as_str_list(dossier.get("themes")),
        "pacing": pacing if pacing in PACINGS else "measured",
        "scales": as_scales(dossier.get("scales"), scale_keys),
        "who_its_for": as_str(dossier.get("who_its_for")),
        "avoid_if": as_str(dossier.get("avoid_if")),
    }


def normalize_pitches(payload: Any) -> dict[int, dict[str, str]]:
    """Map tmdb_id -> ``{hook, because, caveat}``."""
    out: dict[int, dict[str, str]] = {}
    for item in as_list(as_obj(payload).get("pitches")):
        record = as_obj(item)
        tmdb_id = as_float(record.get("tmdb_id"))
        if tmdb_id is None:
            continue
        out[int(tmdb_id)] = {
            "hook": as_str(record.get("hook")),
            "because": as_str(record.get("because")),
            "caveat": as_str(record.get("caveat")),
        }
    return out


def normalize_summary(payload: Any) -> dict[str, Any]:
    summary = as_obj(payload)
    return {
        "headline": as_str(summary.get("headline")),
        "loves": as_str_list(summary.get("loves")),
        "dislikes": as_str_list(summary.get("dislikes")),
        "contradictions": as_str_list(summary.get("contradictions")),
        "blind_spots": as_str_list(summary.get("blind_spots")),
        "rating_style": as_str(summary.get("rating_style")),
    }
