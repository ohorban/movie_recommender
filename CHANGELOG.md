# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a single-user application, "breaking" is interpreted as *requires a rebuild or a
manual migration step*, and is always called out explicitly.

## [Unreleased]

## [0.1.0] — 2026-08-26

First working version: the whole pipeline from Letterboxd export to ranked recommendations.

### Added

**Data layer**
- SQLite schema (33 tables) with a versioned, transactional migration runner.
- Letterboxd export ingestion covering ratings, watched, diary, reviews, watchlist, likes, lists,
  comments and profile, with checksum-based change detection so unchanged files are skipped.
- Canonical `film_key` (normalised title + year) as the join key, because `reviews.csv` and
  `diary.csv` carry entry URIs rather than film URIs and nothing else joins every export file.
- Title normalisation that folds accents and the ligatures NFKD leaves alone (`æ`, `ø`, `ß`, `ł`).

**External catalog**
- TMDB ingestion: year-by-year discovery, full detail with keywords, credits and audience reviews.
- IMDb bulk ratings as an independent quality prior.
- MovieLens 25M: the 1,128-tag genome, plus shrunk-cosine item-item collaborative filtering.
- Wikipedia plot synopses as the deep natural-language source, with article-validity guards and
  tiered fetching (own films first, then the strongest of the catalog, then on demand).
- On-disk HTTP response cache, token-bucket rate limiting and resumable downloads throughout.

**Matching**
- Confidence-scored TMDB resolution blending fuzzy title similarity, year proximity and a
  popularity tiebreak, with ambiguity detection when two candidates score alike.
- Manual overrides that survive a full rebuild, exposed in the CLI and the Data tab.
- A 14-day cooldown before retrying a film that failed to match.

**Enrichment**
- Pluggable embedding backends: local sentence-transformers by default, a deterministic offline
  hash backend for tests.
- Two documents per film — a profile document and, where a real synopsis exists, a plot document
  mean-pooled over chunks.
- Claude review structuring into liked/disliked aspects, themes, engagement and generalisable
  taste signals.
- Lazily generated, cached film dossiers with nine calibrated 0–1 scales.
- Content-hashed everything, so nothing is re-embedded or re-sent to the LLM without cause.

**Taste model**
- Preference expressed as z-scores against the user's own rating mean rather than raw stars.
- Multi-modal taste: liked films clustered into distinct modes with exemplars and labels.
- A repulsion centroid built from low-rated films.
- Empirical-Bayes shrunk affinities across genre, keyword, tag, director, cast, decade, language
  and runtime.
- Dossier-scale sweet spots weighted by how strongly each scale actually correlates with rating.

**Recommendation**
- Six-source candidate generation: per-mode kNN, per-favourite kNN, CF neighbours, facet rules,
  watchlist, and a deliberate exploration slot.
- Cross-validated ranker selecting between ridge and gradient boosting, blended with a hand-tuned
  heuristic in proportion to demonstrated skill.
- Leakage guard removing each film's own CF contribution during training.
- MMR diversification and multi-source agreement bonuses.
- Natural-language intent layer producing a semantic query, hard filters and a taste weight, with
  an over-filtering guard that discards filter sets that eliminate almost everything.
- Batched, grounded recommendation notes citing the user's own rated films.

**Interfaces**
- Streamlit app with Tonight, Ask, Insights and Data tabs, including an in-app update button.
- `movierec` CLI: setup, update, rebuild, status, recommend, unmatched, fix-match.

**Project**
- 125 tests including an end-to-end run of the real pipeline against stand-in clients.
- Ruff lint and format, GitHub Actions CI, Makefile, architecture documentation.

### Known limitations
- Letterboxd exports carry no external ids, so a small number of films need manual matching.
- With ~150 ratings the learned ranker is genuinely small-data; the blend weight reflects this.
- The catalog grows slowly across updates as new releases are added, beyond the configured size.

[Unreleased]: https://github.com/ohorban/movie_recommender/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ohorban/movie_recommender/releases/tag/v0.1.0
