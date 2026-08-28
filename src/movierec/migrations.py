"""Versioned SQL migrations.

Each entry is ``(version, name, sql)``. The runner in :mod:`movierec.db`
applies anything newer than what is recorded in ``schema_migrations``, inside a
transaction, so an interrupted upgrade never leaves a half-applied schema.

Never edit a migration that has shipped - add a new one.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[int, str, str]] = []


def _add(version: int, name: str, sql: str) -> None:
    MIGRATIONS.append((version, name, sql))


_add(
    1,
    "initial_schema",
    """
-- ===========================================================================
-- Catalog: films and their metadata, sourced from TMDB / IMDb / MovieLens.
-- ===========================================================================
CREATE TABLE movies (
    tmdb_id             INTEGER PRIMARY KEY,
    imdb_id             TEXT,
    title               TEXT NOT NULL,
    original_title      TEXT,
    year                INTEGER,
    release_date        TEXT,
    runtime             INTEGER,
    original_language   TEXT,
    production_countries TEXT,
    overview            TEXT,
    tagline             TEXT,
    poster_path         TEXT,
    backdrop_path       TEXT,
    homepage            TEXT,
    adult               INTEGER NOT NULL DEFAULT 0,
    status              TEXT,
    budget              INTEGER,
    revenue             INTEGER,
    tmdb_popularity     REAL,
    tmdb_vote_average   REAL,
    tmdb_vote_count     INTEGER,
    imdb_rating         REAL,
    imdb_votes          INTEGER,
    collection_id       INTEGER,
    collection_name     TEXT,
    -- provenance
    origin              TEXT,      -- discover | user | movielens | manual
    detail_level        INTEGER NOT NULL DEFAULT 0,  -- 0 stub, 1 summary, 2 full detail
    detail_fetched_at   TEXT,
    in_catalog          INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX idx_movies_imdb ON movies(imdb_id) WHERE imdb_id IS NOT NULL;
CREATE INDEX idx_movies_year        ON movies(year);
CREATE INDEX idx_movies_popularity  ON movies(tmdb_popularity DESC);
CREATE INDEX idx_movies_votes       ON movies(tmdb_vote_count DESC);
CREATE INDEX idx_movies_title       ON movies(title COLLATE NOCASE);
CREATE INDEX idx_movies_detail      ON movies(detail_level);

CREATE TABLE genres (
    genre_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);
CREATE TABLE movie_genres (
    tmdb_id   INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    genre_id  INTEGER NOT NULL REFERENCES genres(genre_id) ON DELETE CASCADE,
    PRIMARY KEY (tmdb_id, genre_id)
);
CREATE INDEX idx_movie_genres_genre ON movie_genres(genre_id);

CREATE TABLE keywords (
    keyword_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE
);
CREATE TABLE movie_keywords (
    tmdb_id    INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    keyword_id INTEGER NOT NULL REFERENCES keywords(keyword_id) ON DELETE CASCADE,
    PRIMARY KEY (tmdb_id, keyword_id)
);
CREATE INDEX idx_movie_keywords_kw ON movie_keywords(keyword_id);

CREATE TABLE people (
    person_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    popularity  REAL,
    known_for   TEXT
);
CREATE TABLE movie_credits (
    tmdb_id     INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,          -- cast | crew
    job         TEXT NOT NULL DEFAULT '',  -- Director, Screenplay, ...
    character   TEXT,
    cast_order  INTEGER,
    PRIMARY KEY (tmdb_id, person_id, role, job)
);
CREATE INDEX idx_credits_person ON movie_credits(person_id);
CREATE INDEX idx_credits_job    ON movie_credits(job) WHERE job IS NOT NULL;

-- Long-form natural-language material about a film. One row per source so we
-- can re-fetch or re-generate any single source without disturbing the others.
CREATE TABLE movie_texts (
    tmdb_id      INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    source       TEXT NOT NULL,   -- wikipedia_plot | tmdb_reviews | dossier_prose
    text         TEXT NOT NULL,
    lang         TEXT DEFAULT 'en',
    url          TEXT,
    content_hash TEXT NOT NULL,
    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (tmdb_id, source)
);

-- MovieLens tag-genome: dense, interpretable descriptors ("atmospheric",
-- "thought-provoking", "dystopia") with a 0-1 relevance per film.
CREATE TABLE movie_tags (
    tmdb_id    INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    tag        TEXT NOT NULL,
    relevance  REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT 'movielens_genome',
    PRIMARY KEY (tmdb_id, tag, source)
);
CREATE INDEX idx_movie_tags_tag ON movie_tags(tag);

-- Claude-generated structured profile of a film (themes, tone, pacing, ...).
CREATE TABLE movie_dossiers (
    tmdb_id      INTEGER PRIMARY KEY REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    model        TEXT NOT NULL,
    input_hash   TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Item-item collaborative-filtering neighbours derived from MovieLens.
CREATE TABLE cf_neighbors (
    tmdb_id          INTEGER NOT NULL,
    neighbor_tmdb_id INTEGER NOT NULL,
    score            REAL NOT NULL,
    PRIMARY KEY (tmdb_id, neighbor_tmdb_id)
);
CREATE INDEX idx_cf_score ON cf_neighbors(tmdb_id, score DESC);

-- External id crosswalk (MovieLens movieId <-> tmdb/imdb).
CREATE TABLE external_ids (
    namespace   TEXT NOT NULL,      -- movielens | imdb
    external_id TEXT NOT NULL,
    tmdb_id     INTEGER,
    PRIMARY KEY (namespace, external_id)
);
CREATE INDEX idx_external_tmdb ON external_ids(tmdb_id);

-- ===========================================================================
-- User: everything derived from the Letterboxd export.
--
-- NOTE ON KEYS: Letterboxd exports are inconsistent. ratings/watched/
-- watchlist/likes/lists carry the *film* URI (boxd.it/EMTM), while reviews and
-- diary carry the *entry* URI (boxd.it/aRcQXb) which is unique per log, not
-- per film. The only key that joins every file is the normalised title+year,
-- so `film_key` is canonical and `film_uri` is recorded opportunistically.
-- ===========================================================================
CREATE TABLE user_films (
    film_key          TEXT PRIMARY KEY,   -- slug(title)|year
    film_uri          TEXT,               -- boxd.it film URI, when a file supplies one
    title             TEXT NOT NULL,
    year              INTEGER,
    tmdb_id           INTEGER REFERENCES movies(tmdb_id) ON DELETE SET NULL,
    match_confidence  REAL,
    match_method      TEXT,
    needs_review      INTEGER NOT NULL DEFAULT 0,
    resolved_at       TEXT,
    first_seen        TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_user_films_tmdb   ON user_films(tmdb_id);
CREATE UNIQUE INDEX idx_user_films_uri ON user_films(film_uri) WHERE film_uri IS NOT NULL;
CREATE INDEX idx_user_films_review ON user_films(needs_review) WHERE needs_review = 1;

-- Manual match corrections. Deliberately preserved across `rebuild` so a fix
-- made once survives every future rebuild of the catalog.
CREATE TABLE title_overrides (
    film_key   TEXT PRIMARY KEY,
    tmdb_id    INTEGER,
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE user_ratings (
    film_key    TEXT PRIMARY KEY REFERENCES user_films(film_key) ON DELETE CASCADE,
    rating      REAL NOT NULL,
    rated_date  TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE user_watched (
    film_key     TEXT PRIMARY KEY REFERENCES user_films(film_key) ON DELETE CASCADE,
    watched_date TEXT
);

CREATE TABLE user_diary (
    entry_uri    TEXT PRIMARY KEY,   -- Letterboxd diary-entry URI
    film_key     TEXT NOT NULL REFERENCES user_films(film_key) ON DELETE CASCADE,
    rating       REAL,
    rewatch      INTEGER NOT NULL DEFAULT 0,
    watched_date TEXT,
    logged_date  TEXT,
    tags         TEXT
);
CREATE INDEX idx_diary_film ON user_diary(film_key);
CREATE INDEX idx_diary_date ON user_diary(watched_date);

CREATE TABLE user_reviews (
    review_uri   TEXT PRIMARY KEY,   -- Letterboxd review URI, stable per review
    film_key     TEXT NOT NULL REFERENCES user_films(film_key) ON DELETE CASCADE,
    review_text  TEXT NOT NULL,
    text_hash    TEXT NOT NULL,
    rating       REAL,
    rewatch      INTEGER NOT NULL DEFAULT 0,
    watched_date TEXT,
    review_date  TEXT,
    tags         TEXT,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_reviews_film ON user_reviews(film_key);

CREATE TABLE user_watchlist (
    film_key   TEXT PRIMARY KEY REFERENCES user_films(film_key) ON DELETE CASCADE,
    added_date TEXT
);

CREATE TABLE user_likes (
    film_key   TEXT PRIMARY KEY REFERENCES user_films(film_key) ON DELETE CASCADE,
    liked_date TEXT
);

CREATE TABLE user_lists (
    list_name   TEXT NOT NULL,
    film_key    TEXT NOT NULL REFERENCES user_films(film_key) ON DELETE CASCADE,
    position    INTEGER,
    notes       TEXT,
    list_uri    TEXT,
    list_date   TEXT,
    PRIMARY KEY (list_name, film_key)
);

-- Free-text comments the user left on their own or others' entries. Short, but
-- they carry real taste signal, so they feed the review-structuring stage.
CREATE TABLE user_comments (
    comment_hash TEXT PRIMARY KEY,
    target_uri   TEXT,
    comment_text TEXT NOT NULL,
    comment_date TEXT
);

CREATE TABLE user_profile (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Claude's structured reading of a single natural-language review.
CREATE TABLE review_facts (
    review_uri   TEXT PRIMARY KEY REFERENCES user_reviews(review_uri) ON DELETE CASCADE,
    text_hash    TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    model        TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- In-app signals: thumbs up/down, dismissals, "not tonight".
CREATE TABLE feedback (
    feedback_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id      INTEGER,
    action       TEXT NOT NULL,       -- like | dislike | dismiss | saved | watched
    surface      TEXT,                -- tonight | ask | insights
    context_json TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_feedback_movie ON feedback(tmdb_id);

-- ===========================================================================
-- Machine-learning artefacts.
-- ===========================================================================
CREATE TABLE embeddings (
    entity_type  TEXT NOT NULL,       -- movie | review | taste_mode | query
    entity_id    TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    vector       BLOB NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (entity_type, entity_id, model)
);
CREATE INDEX idx_embeddings_type ON embeddings(entity_type, model);

CREATE TABLE model_artifacts (
    name         TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    metrics_json TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    is_active    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (name, version)
);
CREATE INDEX idx_artifacts_active ON model_artifacts(name, is_active);

-- ===========================================================================
-- Operations: provenance and incremental-update bookkeeping.
-- ===========================================================================
CREATE TABLE ingest_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,        -- setup | update | rebuild
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,        -- running | ok | error | skipped
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    stats_json  TEXT,
    error       TEXT
);
CREATE INDEX idx_ingest_started ON ingest_runs(started_at DESC);

-- Fingerprints of every input file we have consumed, so an unchanged export is
-- skipped outright and a changed one only reprocesses the rows that differ.
CREATE TABLE source_files (
    path        TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    size_bytes  INTEGER,
    mtime       REAL,
    first_seen  TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE kv (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
""",
)


_add(
    2,
    "enrichment_attempts",
    """
-- Negative cache for enrichment lookups that legitimately found nothing.
--
-- Roughly one film in six has no confident Wikipedia article - a short, an
-- obscure regional release, a title too generic to disambiguate. Without a
-- record of that, those films stay in the "no plot yet" set forever and eat the
-- fetch budget on every single update.
CREATE TABLE enrichment_attempts (
    tmdb_id      INTEGER NOT NULL REFERENCES movies(tmdb_id) ON DELETE CASCADE,
    source       TEXT NOT NULL,       -- wikipedia_plot | ...
    outcome      TEXT NOT NULL,       -- miss | error
    attempts     INTEGER NOT NULL DEFAULT 1,
    last_attempt TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (tmdb_id, source)
);
CREATE INDEX idx_enrichment_source ON enrichment_attempts(source, last_attempt);
""",
)
