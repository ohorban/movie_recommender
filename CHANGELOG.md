# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a single-user application, "breaking" is interpreted as *requires a rebuild or a
manual migration step*, and is always called out explicitly.

## [Unreleased]

## [0.1.5] — 2026-08-28

### Fixed

- **`AttributeError: 'str' object has no attribute 'get'` killed the pipeline** at the taste-profile
  stage, and the same error broke recommendation explanations. A tool-use `input_schema` constrains
  what the model is *asked* for, not what it returns. Across one real run of 127 reviews and 413
  dossiers, three deviations occurred:
  - a required array returned as `null` (11 of 127 reviews) — already tolerated;
  - one entry inside an array of objects returned as a **JSON string** rather than an object — the
    crash;
  - `tone` and `themes` returned as a **bare string** instead of a list (96 of 413 dossiers) — this
    one never crashed. It iterated the string character by character, so a film's tone rendered as
    "t, e, n, s, e" in the UI and in the prompts sent back to Claude.
- Every LLM payload now passes through `movierec.enrich.coerce`, on the way in **and** on the way
  out of the database. Normalising on read repairs the records already stored, so nothing needs
  regenerating.
- The natural-language intent parser was hardened the same way: enums constrained, numbers clamped,
  list fields accepting a bare string.

### Changed
- **The test Claude client now returns imperfect payloads.** It previously returned flawless
  schema-conformant output, which is exactly why all three deviations reached production untested.
  It now emits each of them on a deterministic slice of calls. Verified to have teeth: removing the
  normalisation layer fails 18 end-to-end tests.

### Added
- `tests/test_coerce.py` (27 tests), every case drawn from the observed payloads.

## [0.1.4] — 2026-08-28

### Fixed

- **The 0.1.2 fix for the `temperature` error never took effect.** The kwargs filter was added and
  unit-tested, but neither `messages.create` call site was actually routed through it, so every LLM
  call still raised exactly as before. The unit test passed because it exercised the filter in
  isolation rather than the code path that uses it. Both call sites now use it, and
  `tests/test_llm.py` drives `structured()` and `text()` end to end against stand-in SDKs with and
  without `temperature` — plus a source-level assertion that no `messages.create` call bypasses the
  filter.
- A `**kwargs`-style `messages.create` signature (an SDK wrapper or decorator) made the filter strip
  *every* argument. Filtering is now skipped when the callee accepts `VAR_KEYWORD`.
- **`Config` resolved relative paths against the current working directory.** `load_config` always
  passed absolute paths so production was unaffected, but a directly constructed `Config` would read
  and write `data/` and `db/` wherever the process happened to start — which caused a test to write
  into the maintainer's live cache. Relative paths are now anchored to `root`.

### Added
- `tests/test_llm.py` (12 tests): SDK-compatibility, caching, batch failure isolation, usage
  accounting, and the call-site guard.

## [0.1.3] — 2026-08-28

### Fixed

- **`MOVIEREC_WIKIPEDIA_LIMIT` behaved as a per-run batch size rather than a coverage target**, so
  every update fetched another full batch of synopses — hours at a time — slowly walking the entire
  30k catalog rather than stopping once the target was met. The budget is now the remaining
  shortfall (`target - already_stored`). On the maintainer's database this took the next update from
  8,000 fetches to 1,417.
- Films with no confident Wikipedia article (about one in six) were never recorded as such, so they
  stayed in the "no synopsis yet" set forever and consumed the budget on every run. Added an
  `enrichment_attempts` negative cache (migration 002), with a 180-day retry window matching the
  HTTP cache TTL.

### Changed
- The user's own films are now exempt from the coverage budget — a newly logged film always gets a
  synopsis even once the catalog target is met — and are still fetched first.

## [0.1.2] — 2026-08-28

### Fixed

- **Every Claude API call failed** with `Messages.create() got an unexpected keyword argument
  'temperature'`. The `anthropic` 1.x SDK removed `temperature` and `top_p` from `Messages.create`.
  Keyword arguments are now filtered against the installed SDK's actual signature at construction
  time, so the same code runs on 0.x and 1.x. This had silently disabled review structuring, film
  dossiers, the taste summary, natural-language intent parsing and all recommendation explanations.

- **The ranker's reported accuracy was inflated by target leakage** — 0.96 rank correlation against
  a true 0.53 on real data. Two independent causes, both now closed:
  - Features are fitted on the ratings being predicted, so a rated film was scored partly against
    itself: its own rating sat inside its director's affinity and inside the centroid of the taste
    mode it belonged to. Training features are now built leave-one-out.
  - Leave-one-out alone still reported ~0.91, because the profile was fitted once over every rating
    and cross-validated on top. The taste profile is now rebuilt inside each fold, and held-out
    films are featurised exactly as an unrated candidate would be.

  This was not only a reporting problem: the blend weight is derived from that score, so the
  system was fully trusting a model that had mostly memorised its own training set. On the
  maintainer's data the honest evaluation now prefers the hand-tuned prior outright.

- Streamlit's source watcher printed a traceback for every lazily-imported `transformers` submodule
  with a missing optional dependency, burying the real logs. Disabled via `.streamlit/config.toml`.
- `FutureWarning` from sentence-transformers 6's renamed `get_sentence_embedding_dimension`.
- The test TMDB fake mutated its own state while iterating it, so concurrent resolution
  intermittently lost a title.

### Added
- `tests/test_leakage.py` — trains on ratings drawn at random and asserts the evaluation finds no
  signal, pinning the failure mode above.
- The Insights tab reports the metric as held-out, explains what that means, and when the prior
  wins it says so and shows the prior's weights instead of an empty table.

### Changed
- `TasteRanker.fit` accepts externally computed out-of-fold predictions; `TasteMode` records its
  member films; `build_profile_from_prefs` builds a profile from an arbitrary subset of ratings.

## [0.1.1] — 2026-08-27

### Fixed
- `make dev-install` failed with *"No virtual environment found"* on a clean checkout. The Makefile
  assumed a `.venv` already existed and was activated. It now creates the environment on demand and
  invokes the venv's own binaries throughout, so no `source .venv/bin/activate` is needed for any
  target. A missing `uv` is reported with install instructions rather than `command not found`.

### Added
- `make doctor` — reports Python version, whether the embedding and web extras are installed,
  whether the API keys are set, and the current database status.
- `make venv` and `make distclean`; `PYTHON_VERSION` is overridable (default 3.12).
- `make help` is now the default target.

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

[Unreleased]: https://github.com/ohorban/movie_recommender/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/ohorban/movie_recommender/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/ohorban/movie_recommender/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/ohorban/movie_recommender/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ohorban/movie_recommender/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ohorban/movie_recommender/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ohorban/movie_recommender/releases/tag/v0.1.0
