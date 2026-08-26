"""Export parsing and, critically, that repeat runs stay incremental."""

from __future__ import annotations

import shutil

from movierec.db import fetch_all, scalar
from movierec.ingest.letterboxd import (
    FilmRegistry,
    _read_list_csv,
    find_export_dirs,
    ingest_export,
    is_film_uri,
    latest_export_dir,
)
from movierec.text_utils import film_key


def test_is_film_uri_separates_film_from_entry_uris():
    # This distinction is why film_key exists: reviews and diary carry entry URIs.
    assert is_film_uri("https://boxd.it/EMTM")
    assert is_film_uri("https://boxd.it/ufwK")
    assert not is_film_uri("https://boxd.it/aRcQXb")
    assert not is_film_uri("https://boxd.it/dLEyxv")


def test_registry_only_keeps_film_uris():
    reg = FilmRegistry()
    reg.add("Weapons", 2025, "https://boxd.it/aRcQXb")  # entry URI, must be ignored
    reg.add("Weapons", 2025, "https://boxd.it/EMTM")  # film URI, must be kept
    entry = reg.films[film_key("Weapons", 2025)]
    assert entry["film_uri"] == "https://boxd.it/EMTM"


def test_latest_export_dir_picks_newest(tmp_path):
    data = tmp_path / "data"
    for stamp in ["2026-01-01-10-00", "2026-08-26-22-45", "2025-12-31-23-59"]:
        d = data / f"letterboxd-x-{stamp}-utc"
        d.mkdir(parents=True)
        (d / "watched.csv").write_text("Date,Name,Year,Letterboxd URI\n")
        (d / "ratings.csv").write_text("Date,Name,Year,Letterboxd URI,Rating\n")
    assert latest_export_dir(data).name == "letterboxd-x-2026-08-26-22-45-utc"
    assert len(find_export_dirs(data)) == 3


def test_list_csv_two_block_format(synthetic_export):
    meta, films = _read_list_csv(synthetic_export / "lists" / "faves.csv")
    assert meta["Name"] == "faves"
    assert [f["Name"] for f in films] == ["Alpha", "Gamma"]


def test_full_ingest_counts(conn, synthetic_export, tmp_path):
    stats = ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert stats.films_total == 6  # Alpha..Delta + Epsilon, Zeta from the watchlist
    assert stats.ratings_new == 3
    assert stats.watched_new == 4
    assert stats.reviews_new == 2
    assert stats.watchlist_new == 2
    assert stats.likes_new == 1
    assert stats.lists_new == 2
    assert stats.comments_new == 1
    assert scalar(conn, "SELECT COUNT(*) FROM user_films") == 6


def test_reviews_join_to_films_despite_entry_uris(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    rows = fetch_all(
        conn, "SELECT f.title, r.review_text FROM user_reviews r JOIN user_films f USING(film_key)"
    )
    assert {r["title"] for r in rows} == {"Alpha", "Beta"}


def test_unchanged_export_is_skipped(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    again = ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert again.unchanged is True
    assert again.films_new == 0


def test_forced_rerun_writes_nothing_new(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    again = ingest_export(conn, synthetic_export, data_root=tmp_path / "data", force=True)
    assert (again.films_new, again.ratings_new, again.reviews_new, again.watched_new) == (
        0,
        0,
        0,
        0,
    )
    assert scalar(conn, "SELECT COUNT(*) FROM user_films") == 6


def test_new_export_folder_adds_only_the_delta(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")

    newer = tmp_path / "data" / "letterboxd-tester-2026-06-01-00-00-utc"
    shutil.copytree(synthetic_export, newer)
    (newer / "ratings.csv").write_text(
        (synthetic_export / "ratings.csv").read_text()
        + "2025-06-01,Omega,2020,https://boxd.it/zz,3.5\n",
        encoding="utf-8",
    )
    (newer / "watched.csv").write_text(
        (synthetic_export / "watched.csv").read_text()
        + "2025-06-01,Omega,2020,https://boxd.it/zz\n",
        encoding="utf-8",
    )

    assert latest_export_dir(tmp_path / "data").name == newer.name
    stats = ingest_export(conn, newer, data_root=tmp_path / "data")
    assert stats.films_new == 1
    assert stats.ratings_new == 1
    assert stats.reviews_new == 0  # unchanged reviews are not reprocessed
    assert scalar(conn, "SELECT COUNT(*) FROM user_ratings") == 4


def test_edited_review_is_detected_as_changed(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    text = (
        (synthetic_export / "reviews.csv")
        .read_text()
        .replace("Loved the concept, hated the ending", "Actually the ending grew on me")
    )
    (synthetic_export / "reviews.csv").write_text(text, encoding="utf-8")

    stats = ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert stats.reviews_changed == 1
    assert stats.reviews_new == 0
    assert "grew on me" in scalar(
        conn, "SELECT review_text FROM user_reviews ORDER BY review_uri LIMIT 1"
    )


def test_watchlist_removal_is_applied(conn, synthetic_export, tmp_path):
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert scalar(conn, "SELECT COUNT(*) FROM user_watchlist") == 2
    (synthetic_export / "watchlist.csv").write_text(
        "Date,Name,Year,Letterboxd URI\n2025-02-01,Epsilon,2005,https://boxd.it/ee\n",
        encoding="utf-8",
    )
    stats = ingest_export(conn, synthetic_export, data_root=tmp_path / "data")
    assert stats.watchlist_removed == 1
    assert scalar(conn, "SELECT COUNT(*) FROM user_watchlist") == 1


def test_deleted_and_orphaned_folders_are_ignored(conn, synthetic_export, tmp_path):
    (synthetic_export / "deleted").mkdir(exist_ok=True)
    (synthetic_export / "deleted" / "ratings.csv").write_text(
        "Date,Name,Year,Letterboxd URI,Rating\n2025-01-01,Ghost,1999,https://boxd.it/gg,5\n",
        encoding="utf-8",
    )
    ingest_export(conn, synthetic_export, data_root=tmp_path / "data", force=True)
    assert scalar(conn, "SELECT COUNT(*) FROM user_films WHERE title='Ghost'") == 0


def test_real_export_parses_cleanly(conn, real_export, tmp_path):
    stats = ingest_export(conn, real_export, data_root=real_export.parent)
    assert stats.films_total > 100
    assert stats.ratings_new > 50
    # Nothing may reference a film that was not registered.
    for table in [
        "user_ratings",
        "user_reviews",
        "user_diary",
        "user_watchlist",
        "user_likes",
        "user_lists",
    ]:
        orphans = scalar(
            conn,
            f"SELECT COUNT(*) FROM {table} t "
            f"LEFT JOIN user_films f USING(film_key) WHERE f.film_key IS NULL",
        )
        assert orphans == 0, f"{table} has {orphans} orphaned rows"
