"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from movierec.config import Config  # noqa: E402
from movierec.db import init_db  # noqa: E402
from movierec.enrich.embeddings import HashBackend  # noqa: E402


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    cfg = Config(
        root=tmp_path,
        tmdb_api_key="test-key",
        anthropic_api_key="",
        embed_backend="hash",
        catalog_size=200,
        min_votes=50,
        min_year=2000,
        enable_movielens=False,
        enable_imdb=False,
        enable_wikipedia=False,
        candidates_per_source=60,
        db_path=tmp_path / "db" / "test.db",
        data_dir=tmp_path / "data",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def conn(tmp_config: Config):
    connection = init_db(tmp_config.db_path)
    yield connection
    connection.close()


@pytest.fixture
def backend() -> HashBackend:
    return HashBackend(dim=128)


def _find_real_export() -> Path | None:
    candidates = sorted(p for p in (ROOT / "data").glob("letterboxd-*") if p.is_dir())
    return candidates[-1] if candidates else None


@pytest.fixture
def real_export() -> Path:
    """The maintainer's actual Letterboxd export, when it is present.

    Only for the test that specifically validates real-world data. Everything
    else uses :func:`library_export`, which falls back to a generated one — the
    export is git-ignored, so a fixture that merely skips would silently
    disable the entire end-to-end suite on any clean checkout, CI included.
    """
    export = _find_real_export()
    if export is None:
        pytest.skip("no Letterboxd export in data/")
    return export


@pytest.fixture
def library_export(tmp_path: Path) -> Path:
    """A full-sized Letterboxd export: the real one if present, else generated.

    Written outside the config's own ``data/`` directory, because callers copy
    the export *into* that directory and copying a tree onto itself fails.
    """
    export = _find_real_export()
    return export if export is not None else generate_export(tmp_path / "_export_source")


def generate_export(data_dir: Path, *, n_watched: int = 520, seed: int = 5) -> Path:
    """Write a realistic Letterboxd export.

    Reproduces the format's real awkwardness, because that is what the ingest
    layer has to survive:

    * ratings / watched / watchlist / likes / lists carry the **film** URI
    * reviews and diary carry the **entry** URI, which is unique per log
    * lists use a two-block CSV with a banner line
    * a rewatch appears twice in the diary with the same film
    """
    import csv
    import random

    rng = random.Random(seed)
    export = data_dir / "letterboxd-sample-2026-01-01-00-00-utc"
    (export / "likes").mkdir(parents=True, exist_ok=True)
    (export / "lists").mkdir(parents=True, exist_ok=True)

    openers = [
        "Quiet",
        "Burning",
        "The Last",
        "Northern",
        "Paper",
        "Glass",
        "Salt",
        "Winter",
        "Crimson",
        "Hollow",
        "Silver",
        "Broken",
        "Distant",
        "Iron",
        "Amber",
        "Wild",
    ]
    closers = [
        "Harbour",
        "Circuit",
        "Lighthouse",
        "Passage",
        "Orchard",
        "Signal",
        "Chapel",
        "Meridian",
        "Foxes",
        "Machine",
        "Cartography",
        "Weather",
        "Anthem",
        "Divide",
    ]
    films: list[tuple[str, int, str]] = []
    used: set[str] = set()
    while len(films) < n_watched:
        title = f"{rng.choice(openers)} {rng.choice(closers)}"
        if len(films) % 7 == 0:
            title += f" {rng.choice(['II', 'Rising', 'Redux'])}"
        year = rng.randint(1968, 2026)
        key = f"{title}|{year}"
        if key in used:
            continue
        used.add(key)
        films.append((title, year, f"https://boxd.it/{len(films):04x}"))

    def write(path: Path, header: list[str], rows: list[list]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    date = "2025-06-01"
    write(
        export / "watched.csv",
        ["Date", "Name", "Year", "Letterboxd URI"],
        [[date, t, y, u] for t, y, u in films],
    )

    rated = films[:165]
    # A believable distribution: most films land mid-scale, a few at each end.
    scale = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    weights = [1, 3, 4, 8, 12, 20, 18, 18, 9, 7]
    ratings = [rng.choices(scale, weights=weights)[0] for _ in rated]
    write(
        export / "ratings.csv",
        ["Date", "Name", "Year", "Letterboxd URI", "Rating"],
        [[date, t, y, u, r] for (t, y, u), r in zip(rated, ratings)],
    )

    phrases = [
        "A genuinely original premise carried by committed performances.",
        "Beautiful to look at, but it never earns its ending.",
        "Slow in the middle third and it never recovers the tension.",
        "The score does most of the emotional work, and it works.",
        "Clever without being smug. The twist is set up properly.",
        "Too pleased with its own strangeness to be actually moving.",
        "Rewatched it and liked it more the second time.",
        "Competent and forgettable. Nothing here I will think about again.",
    ]
    reviews = []
    for i, ((t, y, _u), r) in enumerate(zip(rated[:130], ratings[:130])):
        reviews.append(
            [
                date,
                t,
                y,
                f"https://boxd.it/{i:06x}rv",
                r,
                "Yes" if i % 11 == 0 else "",
                rng.choice(phrases),
                "",
                date,
            ]
        )
    write(
        export / "reviews.csv",
        [
            "Date",
            "Name",
            "Year",
            "Letterboxd URI",
            "Rating",
            "Rewatch",
            "Review",
            "Tags",
            "Watched Date",
        ],
        reviews,
    )

    diary = []
    for i, ((t, y, _u), r) in enumerate(zip(rated[:150], ratings[:150])):
        diary.append(
            [date, t, y, f"https://boxd.it/{i:06x}dy", r, "Yes" if i % 11 == 0 else "", "", date]
        )
    write(
        export / "diary.csv",
        ["Date", "Name", "Year", "Letterboxd URI", "Rating", "Rewatch", "Tags", "Watched Date"],
        diary,
    )

    # The watchlist is films they have *not* watched, so invent fresh ones.
    watchlist = [
        [
            date,
            f"Unseen {closers[i % len(closers)]} {i}",
            2020 + (i % 6),
            f"https://boxd.it/w{i:03x}",
        ]
        for i in range(110)
    ]
    write(export / "watchlist.csv", ["Date", "Name", "Year", "Letterboxd URI"], watchlist)

    write(
        export / "likes" / "films.csv",
        ["Date", "Name", "Year", "Letterboxd URI"],
        [[date, t, y, u] for t, y, u in rated[:35]],
    )

    with open(export / "lists" / "favourites.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("Letterboxd list export v7\n")
        w = csv.writer(fh)
        w.writerow(["Date", "Name", "Tags", "URL", "Description"])
        w.writerow(["2026-01-01", "favourites", "", "https://boxd.it/list1", ""])
        fh.write("\n")
        w.writerow(["Position", "Name", "Year", "URL", "Description"])
        for i, (t, y, u) in enumerate(rated[:12], start=1):
            w.writerow([i, t, y, u, ""])

    write(
        export / "comments.csv",
        ["Date", "Content", "Comment"],
        [[date, "https://boxd.it/xyz", "A thought about how memory works in this one."]],
    )
    write(
        export / "profile.csv",
        [
            "Date Joined",
            "Username",
            "Given Name",
            "Family Name",
            "Email Address",
            "Location",
            "Website",
            "Bio",
            "Pronoun",
            "Favorite Films",
        ],
        [["2025-01-01", "sample", "Sample", "", "", "", "", "", "", films[0][2]]],
    )
    return export


@pytest.fixture
def synthetic_export(tmp_path: Path) -> Path:
    """A small, fully controlled export used where exact counts matter."""
    export = tmp_path / "data" / "letterboxd-tester-2026-01-01-00-00-utc"
    export.mkdir(parents=True, exist_ok=True)
    (export / "likes").mkdir(exist_ok=True)
    (export / "lists").mkdir(exist_ok=True)

    (export / "ratings.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating\n"
        "2025-01-01,Alpha,2001,https://boxd.it/aa,4.5\n"
        "2025-01-02,Beta,2002,https://boxd.it/bb,2\n"
        "2025-01-03,Gamma,2003,https://boxd.it/cc,5\n",
        encoding="utf-8",
    )
    (export / "watched.csv").write_text(
        "Date,Name,Year,Letterboxd URI\n"
        "2025-01-01,Alpha,2001,https://boxd.it/aa\n"
        "2025-01-02,Beta,2002,https://boxd.it/bb\n"
        "2025-01-03,Gamma,2003,https://boxd.it/cc\n"
        "2025-01-04,Delta,2004,https://boxd.it/dd\n",
        encoding="utf-8",
    )
    (export / "reviews.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review,Tags,Watched Date\n"
        '2025-01-01,Alpha,2001,https://boxd.it/rev001,4.5,,"Loved the concept, hated the ending",,2025-01-01\n'
        '2025-01-02,Beta,2002,https://boxd.it/rev002,2,Yes,"Boring and superficial",,2025-01-02\n',
        encoding="utf-8",
    )
    (export / "diary.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n"
        "2025-01-01,Alpha,2001,https://boxd.it/dia001,4.5,,,2025-01-01\n"
        "2025-01-02,Beta,2002,https://boxd.it/dia002,2,Yes,,2025-01-02\n",
        encoding="utf-8",
    )
    (export / "watchlist.csv").write_text(
        "Date,Name,Year,Letterboxd URI\n"
        "2025-02-01,Epsilon,2005,https://boxd.it/ee\n"
        "2025-02-02,Zeta,2006,https://boxd.it/ff\n",
        encoding="utf-8",
    )
    (export / "likes" / "films.csv").write_text(
        "Date,Name,Year,Letterboxd URI\n2025-01-03,Gamma,2003,https://boxd.it/cc\n",
        encoding="utf-8",
    )
    (export / "lists" / "faves.csv").write_text(
        "Letterboxd list export v7\n"
        "Date,Name,Tags,URL,Description\n"
        "2026-01-01,faves,,https://boxd.it/list1,\n"
        "\n"
        "Position,Name,Year,URL,Description\n"
        "1,Alpha,2001,https://boxd.it/aa,\n"
        "2,Gamma,2003,https://boxd.it/cc,\n",
        encoding="utf-8",
    )
    (export / "comments.csv").write_text(
        "Date,Content,Comment\n"
        "2026-01-05,https://boxd.it/xyz,A thought about memory and narrative\n",
        encoding="utf-8",
    )
    (export / "profile.csv").write_text(
        "Date Joined,Username,Given Name,Family Name,Email Address,Location,Website,Bio,Pronoun,Favorite Films\n"
        "2025-01-01,tester,Test,,,,,,They / them,https://boxd.it/aa\n",
        encoding="utf-8",
    )
    return export
