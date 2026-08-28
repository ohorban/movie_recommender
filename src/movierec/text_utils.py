"""Title normalisation and small text helpers.

The `film_key` produced here is the join key that holds the whole user side of
the database together, so it has to be stable across exports and forgiving of
the punctuation and accent noise in Letterboxd titles.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_ARTICLES = ("the ", "a ", "an ")

# Roman numeral sequels are common enough to be worth normalising.
_ROMAN = {
    " ii": " 2",
    " iii": " 3",
    " iv": " 4",
    " v": " 5",
    " vi": " 6",
    " vii": " 7",
    " viii": " 8",
    " ix": " 9",
    " x": " 10",
}


# Letters that are distinct characters rather than accented forms, so NFKD
# leaves them intact. They turn up in Scandinavian, Icelandic and Polish titles.
_LIGATURES = str.maketrans(
    {
        "æ": "ae",
        "Æ": "AE",
        "ø": "o",
        "Ø": "O",
        "œ": "oe",
        "Œ": "OE",
        "ß": "ss",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "ł": "l",
        "Ł": "L",
        "ı": "i",
        "İ": "I",
        "ŋ": "ng",
    }
)


def strip_accents(text: str) -> str:
    """Fold a title to plain ASCII letters where a sensible folding exists."""
    text = str(text or "").translate(_LIGATURES)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_title(title: str) -> str:
    """Lowercase, de-accent, drop punctuation and collapse whitespace."""
    text = strip_accents(str(title or "")).lower().strip()
    text = text.replace("&", " and ")
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text


def title_slug(title: str) -> str:
    return normalize_title(title).replace(" ", "-")


def film_key(title: str, year: int | str | None) -> str:
    """Canonical key for a film as it appears in a Letterboxd export."""
    slug = title_slug(title)
    try:
        y = int(str(year).strip()[:4]) if year not in (None, "", "-") else 0
    except (ValueError, TypeError):
        y = 0
    return f"{slug}|{y}" if y else slug


def match_variants(title: str) -> list[str]:
    """Alternative normalised forms used when matching against an external catalog."""
    base = normalize_title(title)
    out = {base}
    for article in _ARTICLES:
        if base.startswith(article):
            out.add(base[len(article) :])
    if ":" in title:
        out.add(normalize_title(title.split(":", 1)[0]))
    for roman, arabic in _ROMAN.items():
        if base.endswith(roman):
            out.add(base[: -len(roman)] + arabic)
    return [v for v in out if v]


def parse_year(value: object) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        year = int(str(value).strip()[:4])
    except (ValueError, TypeError):
        return None
    return year if 1870 <= year <= 2100 else None


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut[-40:]:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip(" .,;:") + "…"


def clean_ws(text: str | None) -> str:
    return _SPACE.sub(" ", (text or "").replace(" ", " ")).strip()
