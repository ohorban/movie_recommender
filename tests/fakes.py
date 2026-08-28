"""Stand-ins for the network and LLM clients, so tests exercise real code paths."""

from __future__ import annotations

import random
import threading
from typing import Any

GENRES = [
    (28, "Action"),
    (12, "Adventure"),
    (16, "Animation"),
    (35, "Comedy"),
    (80, "Crime"),
    (18, "Drama"),
    (14, "Fantasy"),
    (27, "Horror"),
    (878, "Science Fiction"),
    (53, "Thriller"),
]
KEYWORDS = [
    (1, "time loop"),
    (2, "heist"),
    (3, "dystopia"),
    (4, "coming of age"),
    (5, "revenge"),
    (6, "space travel"),
    (7, "friendship"),
    (8, "war"),
    (9, "artificial intelligence"),
    (10, "survival"),
]
TAGS = ["atmospheric", "thought-provoking", "visually stunning", "dark", "quirky", "emotional"]


def make_movie(
    tmdb_id: int, *, year: int = 2015, title: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    """A TMDB-shaped detail payload."""
    rng = random.Random(seed if seed is not None else tmdb_id)
    genres = rng.sample(GENRES, k=rng.randint(1, 3))
    keywords = rng.sample(KEYWORDS, k=rng.randint(2, 5))
    votes = rng.randint(60, 12000)
    return {
        "id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title or f"Test Film {tmdb_id}",
        "original_title": title or f"Test Film {tmdb_id}",
        "release_date": f"{year}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
        "runtime": rng.choice([88, 96, 104, 118, 132, 148]),
        "original_language": rng.choice(["en", "en", "en", "ja", "fr", "ko"]),
        "overview": f"A {genres[0][1].lower()} film about {keywords[0][1]} and {keywords[-1][1]}. "
        f"It follows a protagonist through escalating trouble.",
        "tagline": f"Nothing is what it seems, {tmdb_id}.",
        "poster_path": f"/p{tmdb_id}.jpg",
        "backdrop_path": None,
        "homepage": "",
        "adult": False,
        "status": "Released",
        "budget": rng.randint(1, 200) * 1_000_000,
        "revenue": rng.randint(1, 900) * 1_000_000,
        "popularity": round(rng.uniform(1, 120), 3),
        "vote_average": round(rng.uniform(4.5, 8.8), 1),
        "vote_count": votes,
        "genres": [{"id": g, "name": n} for g, n in genres],
        "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
        "belongs_to_collection": None,
        "keywords": {"keywords": [{"id": k, "name": n} for k, n in keywords]},
        "credits": {
            "cast": [
                {
                    "id": 1000 + i,
                    "name": f"Actor {1000 + i}",
                    "character": f"Role {i}",
                    "order": i,
                    "popularity": 5.0,
                }
                for i in range(rng.randint(3, 8))
            ],
            "crew": [
                {
                    "id": 2000 + (tmdb_id % 25),
                    "name": f"Director {2000 + (tmdb_id % 25)}",
                    "job": "Director",
                    "department": "Directing",
                    "popularity": 3.0,
                }
            ],
        },
        "reviews": {
            "results": [
                {"author": "someone", "content": "A genuinely absorbing film. " * 12},
            ]
        },
        "external_ids": {"imdb_id": f"tt{tmdb_id:07d}"},
    }


class FakeTMDBClient:
    """Implements the surface of TMDBClient that the pipeline actually uses."""

    def __init__(self, n_movies: int = 300, min_year: int = 2000, max_year: int = 2026) -> None:
        self.details: dict[int, dict[str, Any]] = {}
        self.by_year: dict[int, list[dict[str, Any]]] = {}
        rng = random.Random(42)
        for i in range(n_movies):
            tmdb_id = 500_000 + i
            year = rng.randint(min_year, max_year)
            detail = make_movie(tmdb_id, year=year)
            self.details[tmdb_id] = detail
            self.by_year.setdefault(year, []).append(detail)
        self.search_calls = 0
        self.detail_calls = 0
        self._lock = threading.Lock()

    # -- pipeline surface ---------------------------------------------------
    def configuration(self) -> dict[str, Any]:
        return {}

    def discover(
        self, page: int, *, year: int | None = None, min_votes: int = 50
    ) -> dict[str, Any]:
        pool = sorted(self.by_year.get(year, []), key=lambda d: -d["vote_count"])
        pool = [d for d in pool if d["vote_count"] >= min_votes]
        start = (page - 1) * 20
        chunk = pool[start : start + 20]
        return {
            "page": page,
            "results": [
                {
                    k: d[k]
                    for k in (
                        "id",
                        "title",
                        "original_title",
                        "release_date",
                        "overview",
                        "popularity",
                        "vote_average",
                        "vote_count",
                        "poster_path",
                        "backdrop_path",
                        "original_language",
                        "adult",
                    )
                }
                for d in chunk
            ],
            "total_pages": max(1, (len(pool) + 19) // 20),
            "total_results": len(pool),
        }

    def movie_detail(self, tmdb_id: int) -> dict[str, Any] | None:
        self.detail_calls += 1
        return self.details.get(int(tmdb_id))

    def search(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        want = title.lower().strip()
        with self._lock:
            self.search_calls += 1
            hits = [d for d in list(self.details.values()) if d["title"].lower() == want]
            if not hits:
                # Deterministically attach an id to any unknown title so
                # resolution has something plausible to match, as a real
                # search would.
                tmdb_id = 900_000 + (abs(hash(want)) % 50_000)
                if tmdb_id not in self.details:
                    self.details[tmdb_id] = make_movie(tmdb_id, year=year or 2015, title=title)
                hits = [self.details[tmdb_id]]
        return [
            {
                "id": d["id"],
                "title": d["title"],
                "original_title": d["original_title"],
                "release_date": f"{year}-01-01" if year else d["release_date"],
                "vote_count": d["vote_count"],
                "popularity": d["popularity"],
            }
            for d in hits
        ]

    def find_by_imdb(self, imdb_id: str) -> list[dict[str, Any]]:
        return []


class FakeClaudeClient:
    """Returns schema-shaped payloads without touching the network."""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.calls: list[str] = []
        self.cfg = type("Cfg", (), {"llm_max_concurrency": 2})()

    def structured(
        self, *, kind: str, system: str, user: str, schema: dict, **_kw
    ) -> dict[str, Any]:
        self.calls.append(kind)
        rng = random.Random(abs(hash(user)) % (2**31))
        if kind == "review_facts":
            return {
                "verdict": rng.choice(["loved", "liked", "mixed", "disliked"]),
                "sentiment": round(rng.uniform(-1, 1), 2),
                "signal_strength": round(rng.uniform(0.3, 1.0), 2),
                "liked": [{"aspect": "unique concept", "category": "originality", "strength": 0.8}],
                "disliked": [{"aspect": "slow middle", "category": "pacing", "strength": 0.4}],
                "themes": ["identity", "sacrifice"],
                "tone_words": ["tense", "warm"],
                "engagement": {
                    "intellectual": round(rng.random(), 2),
                    "emotional": round(rng.random(), 2),
                    "visceral": round(rng.random(), 2),
                },
                "taste_signals": ["values originality of premise over execution polish"],
            }
        if kind == "dossier":
            return {
                "logline": "A person confronts an escalating problem.",
                "tone": ["tense", "melancholy"],
                "themes": ["obsession", "family"],
                "pacing": rng.choice(["slow", "measured", "brisk", "relentless"]),
                "scales": {
                    k: round(rng.random(), 2)
                    for k in [
                        "intellectual_demand",
                        "emotional_intensity",
                        "originality",
                        "feel_good",
                        "darkness",
                        "spectacle",
                        "realism",
                        "humor",
                        "tension",
                    ]
                },
                "who_its_for": "Viewers who like a slow build.",
                "avoid_if": "You want something light.",
            }
        if kind == "intent":
            return {
                "semantic_query": f"A film matching: {user[-160:]}",
                "interpretation": "Something absorbing and not too long.",
                "include_genres": [],
                "exclude_genres": [],
                "keywords": [],
                "people": [],
                "year_min": None,
                "year_max": None,
                "runtime_max": None,
                "languages": [],
                "novelty": "balanced",
                "taste_weight": 0.6,
                "target_scales": {},
                "allow_rewatch": False,
            }
        if kind == "taste_summary":
            return {
                "headline": "Rewards ambition, punishes coasting.",
                "loves": ["unusual premises", "competence on screen"],
                "dislikes": ["style without substance"],
                "contradictions": [],
                "blind_spots": ["classic world cinema"],
                "rating_style": "Uses the whole scale and is stingy above four.",
            }
        if kind == "pitches":
            ids = [
                int(t.split()[0].strip(":"))
                for t in user.split("tmdb_id ")[1:]
                if t.split() and t.split()[0].strip(":").isdigit()
            ]
            return {
                "pitches": [
                    {
                        "tmdb_id": i,
                        "hook": "A tightly wound thriller.",
                        "because": "You rated a similar film highly.",
                        "caveat": "",
                    }
                    for i in ids
                ]
            }
        return {}

    def map_structured(self, jobs, *, progress=None, progress_span=(0.0, 1.0), label="") -> list:
        return [self.structured(**job) for job in jobs]

    def text(self, *, system: str, user: str, **_kw) -> str:
        return "A plausible sentence."

    def usage_summary(self) -> dict[str, Any]:
        return {"calls": len(self.calls), "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
