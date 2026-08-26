"""JSON schemas for every structured Claude call.

Kept in one place because these schemas are the contract between the
natural-language layer and the numeric parts of the recommender.
"""

from __future__ import annotations

from typing import Any

ASPECT_CATEGORIES = [
    "concept",
    "story",
    "ending",
    "characters",
    "acting",
    "dialogue",
    "visuals",
    "cinematography",
    "animation",
    "music",
    "sound",
    "pacing",
    "tone",
    "humor",
    "emotional_impact",
    "tension",
    "realism",
    "themes",
    "worldbuilding",
    "originality",
    "craft",
    "style",
    "representation",
    "other",
]

_ASPECT_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspect": {
            "type": "string",
            "description": "Short phrase in the reviewer's own framing, e.g. 'unique concept', 'superficial romance'.",
        },
        "category": {"type": "string", "enum": ASPECT_CATEGORIES},
        "strength": {
            "type": "number",
            "description": "0-1: how emphatic the reviewer was about this point.",
        },
    },
    "required": ["aspect", "category", "strength"],
}

REVIEW_FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["loved", "liked", "mixed", "disliked", "hated"]},
        "sentiment": {"type": "number", "description": "-1 (hated) to 1 (loved)."},
        "signal_strength": {
            "type": "number",
            "description": "0-1: how much usable preference information this review contains. A one-word review is near 0.",
        },
        "liked": {
            "type": "array",
            "items": _ASPECT_ITEM,
            "description": "What the reviewer responded well to. Empty if nothing.",
        },
        "disliked": {
            "type": "array",
            "items": _ASPECT_ITEM,
            "description": "What the reviewer objected to. Empty if nothing.",
        },
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Subject matter or ideas the reviewer engaged with.",
        },
        "tone_words": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Adjectives describing the film's feel as the reviewer experienced it.",
        },
        "engagement": {
            "type": "object",
            "properties": {
                "intellectual": {
                    "type": "number",
                    "description": "0-1: how much the reviewer engaged with ideas.",
                },
                "emotional": {"type": "number", "description": "0-1: how much they were moved."},
                "visceral": {
                    "type": "number",
                    "description": "0-1: thrill, tension, edge-of-seat.",
                },
            },
            "required": ["intellectual", "emotional", "visceral"],
        },
        "taste_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Generalisable statements about this viewer's preferences that this review supports, "
                "phrased to transfer to other films. Good: 'values originality of premise over execution polish'. "
                "Bad: 'liked this movie'. Return an empty array if the review supports none."
            ),
        },
    },
    "required": [
        "verdict",
        "sentiment",
        "signal_strength",
        "liked",
        "disliked",
        "themes",
        "tone_words",
        "engagement",
        "taste_signals",
    ],
}


DOSSIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "logline": {
            "type": "string",
            "description": "One sentence on what the film actually is, no marketing voice.",
        },
        "tone": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 adjectives for how it feels to watch.",
        },
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 ideas the film is genuinely about.",
        },
        "pacing": {"type": "string", "enum": ["slow", "measured", "brisk", "relentless"]},
        "scales": {
            "type": "object",
            "properties": {
                "intellectual_demand": {
                    "type": "number",
                    "description": "0-1: how much thinking it asks of the viewer.",
                },
                "emotional_intensity": {
                    "type": "number",
                    "description": "0-1: how heavy the emotional load is.",
                },
                "originality": {
                    "type": "number",
                    "description": "0-1: how unusual its premise or execution is.",
                },
                "feel_good": {
                    "type": "number",
                    "description": "0-1: how uplifting you feel afterwards.",
                },
                "darkness": {"type": "number", "description": "0-1: bleakness, cruelty, despair."},
                "spectacle": {
                    "type": "number",
                    "description": "0-1: visual scale and craft as an attraction in itself.",
                },
                "realism": {
                    "type": "number",
                    "description": "0-1: grounded and plausible vs stylised or fantastical.",
                },
                "humor": {"type": "number", "description": "0-1: how funny it means to be."},
                "tension": {
                    "type": "number",
                    "description": "0-1: suspense and edge-of-seat pressure.",
                },
            },
            "required": [
                "intellectual_demand",
                "emotional_intensity",
                "originality",
                "feel_good",
                "darkness",
                "spectacle",
                "realism",
                "humor",
                "tension",
            ],
        },
        "who_its_for": {
            "type": "string",
            "description": "One sentence: the viewer who will love this.",
        },
        "avoid_if": {"type": "string", "description": "One sentence: who should skip it and why."},
    },
    "required": ["logline", "tone", "themes", "pacing", "scales", "who_its_for", "avoid_if"],
}


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_query": {
            "type": "string",
            "description": "A rich descriptive paragraph of the ideal film for this request, written as if describing an actual movie. This is embedded and matched against film descriptions, so be concrete and evocative rather than abstract.",
        },
        "interpretation": {
            "type": "string",
            "description": "One sentence back to the user on how you read their request.",
        },
        "include_genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "TMDB genre names that must be present. Use sparingly.",
        },
        "exclude_genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "TMDB genre names to rule out.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete subjects or motifs requested, e.g. 'heist', 'time loop'.",
        },
        "people": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Named directors or actors requested.",
        },
        "year_min": {"type": ["integer", "null"]},
        "year_max": {"type": ["integer", "null"]},
        "runtime_max": {
            "type": ["integer", "null"],
            "description": "Minutes, if the user constrained length.",
        },
        "languages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ISO-639-1 codes if a language was requested.",
        },
        "novelty": {
            "type": "string",
            "enum": ["familiar", "balanced", "obscure"],
            "description": "How well-known the result should be.",
        },
        "taste_weight": {
            "type": "number",
            "description": "0-1. How much the user's general taste profile should influence this, versus matching the literal request. A specific request like 'a Japanese film about grief' is low (0.2); a vague 'something good tonight' is high (0.9).",
        },
        "target_scales": {
            "type": "object",
            "description": "Desired values 0-1 on any dossier scale the request implies. Omit any the user did not imply.",
            "properties": {
                "intellectual_demand": {"type": ["number", "null"]},
                "emotional_intensity": {"type": ["number", "null"]},
                "feel_good": {"type": ["number", "null"]},
                "darkness": {"type": ["number", "null"]},
                "humor": {"type": ["number", "null"]},
                "tension": {"type": ["number", "null"]},
                "spectacle": {"type": ["number", "null"]},
                "realism": {"type": ["number", "null"]},
            },
        },
        "allow_rewatch": {
            "type": "boolean",
            "description": "True only if the user asked for something they have already seen.",
        },
    },
    "required": ["semantic_query", "interpretation", "novelty", "taste_weight", "allow_rewatch"],
}


TASTE_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One vivid sentence characterising this viewer's taste.",
        },
        "loves": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5-8 specific things that reliably work for them.",
        },
        "dislikes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "4-6 specific things that reliably do not.",
        },
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tensions in their taste worth naming, if any.",
        },
        "blind_spots": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Well-regarded territory they have barely explored.",
        },
        "rating_style": {
            "type": "string",
            "description": "One sentence on how they use the 5-point scale.",
        },
    },
    "required": ["headline", "loves", "dislikes", "contradictions", "blind_spots", "rating_style"],
}


PITCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pitches": {
            "type": "array",
            "description": "One entry per film, in the order given.",
            "items": {
                "type": "object",
                "properties": {
                    "tmdb_id": {"type": "integer"},
                    "hook": {
                        "type": "string",
                        "description": "One sentence on what the film is, written to make them want to watch it. No spoilers, no marketing cliche.",
                    },
                    "because": {
                        "type": "string",
                        "description": "One sentence connecting it to this viewer's own history, citing a specific film they rated or something they wrote in a review. Be concrete: name the film. If nothing in their history genuinely supports it, say what makes it a stretch instead of inventing a connection.",
                    },
                    "caveat": {
                        "type": "string",
                        "description": "A short honest warning if there is a real reason they might bounce off it. Empty string if there is not.",
                    },
                },
                "required": ["tmdb_id", "hook", "because", "caveat"],
            },
        }
    },
    "required": ["pitches"],
}
