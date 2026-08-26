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


@pytest.fixture
def real_export() -> Path:
    """The user's actual Letterboxd export, when it is present."""
    data = ROOT / "data"
    candidates = sorted(p for p in data.glob("letterboxd-*") if p.is_dir())
    if not candidates:
        pytest.skip("no Letterboxd export in data/")
    return candidates[-1]


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
