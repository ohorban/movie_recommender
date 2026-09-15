# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a single-user application, "breaking" is interpreted as *requires a rebuild or a
manual migration step*, and is always called out explicitly.

## [Unreleased]

### Added

- **Television.** Letterboxd logs shows next to films; TMDB keeps movies and series in separate id
  namespaces and the matcher only ever searched movies. Every show therefore either found nothing
  or attached itself to a film that shared a word with the title — *Baby Reindeer* to
  *A Baby Reindeer's First Christmas*, *Chernobyl* to a documentary about the exclusion zone, and
  *Chernobyl* is a five-star film that anchors one of the taste clusters. Both indexes are now
  searched and the candidates compete on one scale.
- Shows are stored under their TMDB id plus an offset, with `media_type` and the original id
  alongside (migration 003). Shifting the id is far less invasive than making `(id, media_type)`
  the key of a dozen tables, and `movierec.media` is the only place that does the arithmetic.
- A show is taste evidence, never a recommendation: television is excluded from the candidate pool
  explicitly rather than incidentally.
- The recommendation card names the director, links to IMDb and Letterboxd as well as TMDB, and
  shows your own rating when you have one.

- `RankerMetrics` carries `spearman_sd` and `cv_repeats`, both persisted with the model.
- Four tests cover the averaged-score path: that the averages (not a single split) drive selection,
  that a learned model still wins when it earns it, that an untrained model is never selected, and
  that the spread survives a save and reload.
- `test_progress_is_reported_from_the_calling_thread` drives the real `ClaudeClient` and pins the
  contract; two further tests cover ordering under concurrency and that progress closes its span.
- `test_progress_is_never_reported_from_a_worker_thread` runs the whole pipeline with a callback
  that fails on any off-thread call, covering TMDB, Wikipedia and resolution.
- The test LLM client now runs its calls through a thread pool like the real one. Running them
  serially is what hid this bug.

### Fixed

- **Two leakage holes.** `_loo_dislike` was defined and never called, so every film inside the
  dislike centroid was compared against a centroid containing itself — and `sim_dislike` is one of
  the largest weights in the model. `sim_plot_best` reached for the full mode matrix while
  `sim_mode_best`, three lines above, used the leave-one-out one.
- **The stored profile is no longer a lossy copy of the trained one.** Affinities were truncated to
  the 80 largest per facet, which — because the largest values are the ones with a single
  observation behind them — kept the noise and discarded the evidence, and `affinity_stats` was
  never persisted at all, so leave-one-out silently did nothing on a profile read back from the
  database. Both are stored in full now, along with mode membership.
- **The taste profile is reproducible.** The ratings query had no `ORDER BY` and k-means++ seeds
  from row order, so identical ratings could produce different taste modes — and each mode drives
  its own retrieval neighbourhood. Cluster labels also depended on dict ordering, so the same
  cluster was described differently from run to run in the UI and in the prompt sent to Claude.
- **A failed fold no longer degrades the score quietly.** It imputed a constant 0.0, which sits near
  the mean of a z-scored target, so a broken fold landed mid-pack and cost about 0.06 Spearman with
  only a log line. The model is now dropped from that split instead.
- **A failed held-out evaluation no longer falls back to a flattering number.** The fallback ran an
  in-sample cross-validation reporting about 0.89 where the honest figure is 0.51, and stored that
  as the model's accuracy. It now fails loudly.
- **A model that cannot be loaded no longer reports itself as running.** An unusable pickle kept
  reporting "ridge, spearman 0.51" while the heuristic did all the ranking. Feature sets are also
  versioned now, so coefficients fitted on an older definition are refused rather than applied to a
  feature that changed meaning underneath them.


- **Correcting a bad match silently did nothing.** Two independent faults. The id typed into the
  Fix box never reached the handler, because a plain widget beside a plain button hands Streamlit
  the click before the typed value commits — so the old value was submitted. The handler then
  wrote a *null* override, which pinned the film to "no match found" permanently and reapplied
  itself on every subsequent run. It is now a form (which gathers its widgets on submit), it
  accepts a pasted TMDB address as well as a bare id, and an unreadable paste is an error rather
  than a silent erasure. Migration 003 clears the two null overrides this produced.
- **A one-year release-date disagreement outranked an identical title.** Letterboxd dates
  *Obsession* to 2025 and TMDB to 2026; the match was correct and got flagged for review anyway,
  along with *Enemy*, *Talk to Me*, *Oculus*, *Am I OK?* and *Queen of the Ring*. An exact title
  now carries most of the decision when the years are within two. A larger gap gets no such
  benefit — same title, decades apart, is what a remake looks like.
- **A short title inside a much longer one scored as a match.** `fuzz.WRatio` has a partial-ratio
  arm, so *Baby Reindeer* scored 0.64 against *A Baby Reindeer's First Christmas* and cleared the
  review floor. Candidates whose title is much longer than the query are now discounted.
- Everything still awaiting review is re-matched by migration 003, since all of the above was
  decided by a matcher that no longer exists. Confirmed matches are untouched.
- **The Insights tab rendered a raw `<parameter name="loves">` tag.** Claude emitted the XML
  framing of its own tool call *inside* a field value; the content after the tag was well formed,
  only the framing leaked. It rendered verbatim under "Reliably works for you", and the same string
  was being sent back to Claude in the taste brief attached to every Ask request. Leaked framing is
  now unwrapped and the value recovered — on read as well as on write, so your existing profile is
  repaired without paying to regenerate it.
- **The Tonight tab recomputed an identical answer on every page load.** Ranking 30,046 films and
  writing eight explanations is seconds of work plus a paid Claude call, and none of it is random,
  so a refresh bought the same five films twice. The pick is now stored in the database and reused
  until the data changes or Reroll is pressed.

- **The Update button in the Data tab died partway through with `NoSessionContext`.**
  `ClaudeClient.map_structured` reported progress from inside its worker threads, and Streamlit
  raises when a progress widget is touched off the main thread — so a database update crashed at the
  review-structuring stage after twenty minutes of work. Progress is now reported from the calling
  thread while the calls still run concurrently.
- The same change restores cancellation. The main thread previously blocked inside
  `list(pool.map(...))` with no `st.*` call of its own, so Streamlit had no point at which to
  interrupt a long run — which is why a second click started a second pipeline alongside the first
  rather than replacing it.
- The UI progress callback now suppresses `NoSessionContext` as a backstop. Deliberately only that
  one error: Streamlit cancels a running script by raising `StopException` / `RerunException`
  through `st.*` calls, so swallowing those would leave an old run alive beside a new one.

### Changed

- **The ranking model is no longer chosen by a coin flip.** Ridge had replaced the hand-tuned prior
  outright on a held-out margin of 0.0015 Spearman (paired *p* = 0.91), and because the trust weight
  was `score / 0.45` clipped to 1.0, that coin flip did not tilt the blend — it switched the prior
  off entirely. A learned model now has to beat the prior by more than a standard error of the
  paired per-split difference, *and* clear an absolute skill floor: on random ratings the prior
  scores about zero, so merely clearing it proves nothing.
- **Held-out metrics are computed within each fold, then averaged.** Correlating the concatenated
  out-of-fold vector let the offsets between folds — each has its own model and its own rebuilt
  profile — into the number. That artifact was worth 0.017 Spearman, more than ten times the margin
  it was being used to decide, and it was the whole reason ridge appeared to win.
- **The blend weight is searched on held-out folds instead of derived from a constant.** Dividing by
  0.45 and clipping meant only 0.0 and 1.0 were ever reachable, so the blend was never blended. The
  search covers which model *and* how much of it, with weight zero — the prior alone — competing on
  the same footing. Ties inside half a point of Spearman go to the smaller weight, because the peak
  is flat and taking the maximum of eleven correlated estimates flatters itself.
- **`scale_fit` keeps the direction of a preference.** The weight was `abs(r)` and the feature scored
  distance from a target, which cannot express "more of this is better" — a viewer who likes
  spectacle more the more of it there is was modelled as preferring average spectacle. On real data
  the feature correlated **-0.02** with preference; the signed form reaches **+0.44**. Weights are
  now signed, scales below a 0.10 correlation are dropped as noise (`darkness` was clearing the old
  gate at 0.027), and the Insights tab shows the direction as its own column.
- **A missing dossier is neutral rather than disqualifying.** `scale_fit` returned 0.0 when a film
  had none, which sat *below* the worst real fit of 0.354 — about six standard deviations low for
  the 97.7% of candidates without one. Every rated film has a dossier and 2.3% of the candidate pool
  does, so this was a large systematic bonus for having been recommended before: 9 of the shipped
  top 20 had a dossier against that 2.3% base rate, and dossiers are generated for films the
  recommender already picked. The feature is now centred on zero, so absent evidence scores zero.
- **`cf_score` no longer depends on what else is in the batch.** It was normalised against the peak
  of the candidate set, so the same film scored 0.818 in the full catalog and 0.964 in a 200-film
  shortlist — and the ranker was fitted on ~160-film batches for use on batches of thousands. It is
  now normalised against the user's own expressed preference, which does not move.


- **"Why you" no longer justifies one film with another.** It read *"You wrote that Mulan is not
  often a movie that makes me cry, and this is built for exactly that kind of animated gut-punch"* —
  a claim specific enough to be wrong often, and not what a preference is. Explanations are now
  drawn from a closed list of measured traits (genre and tag affinities, scale sweet spots, what
  the reviews praise and complain about), all aggregated over the whole history, and naming a film
  you have seen is forbidden.
- **The Ask tab stopped importing your taste as subject matter.** Asked for "something that will
  make me feel existential about romantic relationships", it read that as a request for science
  fiction, because science fiction is what the profile says you like. The taste summary now fills
  gaps the request leaves open — how dark, how demanding, how long — and may not add a genre,
  premise or setting the request did not ask for.
- **A specific request no longer returns the same film as an open one.** *The Prestige* led both the
  Tonight tab and that relationships query. Two causes, both fixed: the taste-only retrieval sources
  ran at full size even when a request was made, flooding the pool with well-reviewed films that
  had nothing to do with it; and the taste score has a long right tail — a famous, widely loved film
  lands four standard deviations out — while semantic similarity is near-symmetric, so one outlier
  outscored an excellent match whatever weight the request was given. Sources are now scaled back
  when there is a request, and both terms are clipped before blending. On the real database that
  query goes from *Her, The Prestige, Contact, Toy Story* to *Her, Contact, Room, Magnolia*.

- **The reported ranker accuracy is now an average over five fold splits instead of one.** With 163
  rated films, which films land in which fold moves the held-out Spearman by more than any change
  made to the model: measured on unchanged data and an unchanged model, it ranged 0.445–0.514
  (sd 0.023). A single split was therefore reporting mostly noise, and a run-to-run move was easy to
  mistake for a real gain or regression. Training now repeats the cross-validation with five seeds
  and reports the mean with its spread.
- Model selection uses those averaged scores too, so which ranker gets deployed no longer depends on
  one lucky split.
- The Insights tab shows the accuracy as `mean ± sd`. Read the spread first: a difference smaller
  than it is not a difference.

### Note

Measured on the real database, held-out Spearman over five fold splits:

```
before   ridge alone, weight 1.00      0.5221   (what shipped, selected on a 0.0015 margin)
after    20% ridge + 80% prior         0.5600   (sd 0.0283)
```

The prior alone now scores 0.5477, up from 0.5269, which is the leakage and `scale_fit` fixes
showing up in the honest number rather than in the model. The mixture adds a further 0.013 on top
(paired *p* = 0.12 — real but not conclusive, which is why the tie-break leans toward the prior).


The first version of the averaging above averaged the *predictions* across the five splits and
scored those. That measures a five-model ensemble, not the single model that actually gets deployed,
and it flattered the ridge model into winning. Averaging the *metrics* is the correct form: each
split scores the model as it will be used, and the five scores are then summarised.

## [0.1.9] — 2026-08-28

### Fixed
- **Clicking an example query in the Ask tab crashed the app.** Streamlit forbids assigning to
  `st.session_state[key]` once the widget owning that key exists, and the chips did exactly that to
  prefill the search box. They now use an `on_click` callback, which runs before the rerun and is
  the supported way to do it.
- The CI guard step piped `pytest` into `tee`, so the pipe's exit code masked pytest's. The step
  passed even when the tests failed — and even when `pytest` was not installed at all. It now runs
  under `shell: bash`, which sets `-eo pipefail`.

### Added
- `tests/test_app.py` (10 tests) drives the interface through Streamlit's `AppTest`: example chips,
  reroll, thumbs-down writing feedback, tab contents, and a check that rendering never fires the
  destructive update button.
- `test_every_button_survives_a_click` presses **every** button on every tab, one per fresh run, and
  fails on any exception. That is the general guard for this class of bug. Verified to have teeth:
  reverting the chip fix fails three tests, including this one.
- CI installs the `app` extra and fails if either the end-to-end or UI suite skips itself.

### Note
The previous UI checks only called `AppTest.run()` — they rendered the app and asserted it looked
right. This bug raises only on click, so those checks reported a healthy app while a whole tab was
broken. Rendering is not exercising.

## [0.1.8] — 2026-08-28

### Changed
- **Rewrote the README** around how you actually use the thing: setup first, then the interface,
  then a nine-step plain-language walkthrough of how the recommender works.
- **`MOVIEREC_WIKIPEDIA_LIMIT` now defaults to 3,000, down from 8,000.** Measured at roughly 20
  minutes per 1,000 films, the old default made the first build a 3–4 hour job, most of it in one
  step. 3,000 covers the part of the catalog you are realistically shown. Raising it is one line.
- Wikipedia fetching uses 8 workers instead of 6 — about a third faster, and rate limiting already
  backs off safely.

### Added
- `LICENSE` (MIT). The README claimed a licence the repo did not carry. It also records the terms of
  the four datasets the software downloads: TMDB, IMDb (non-commercial), MovieLens and Wikipedia.
- `make app` now prints a one-line fix if Streamlit is missing, instead of a traceback.

### Verified
- A clean clone with no `.env`, no database and no Letterboxd export: installs, passes 176 tests
  (1 skip, the one needing real data), reports missing keys clearly from both the CLI and
  `make doctor`, and renders all four tabs with build instructions rather than crashing.

## [0.1.7] — 2026-08-28

### Fixed
- **CI could not install anything.** `uv pip install --system` assumes a system interpreter, but
  `setup-uv` provisions a *managed* Python — so 3.10 and 3.13 reported "No system Python
  installation found", and 3.12 fell through to Debian's externally-managed interpreter and was
  refused. The job now uses `activate-environment: true`, which puts uv's managed Python on PATH in
  a virtual environment, and installs without `--system`.

### Changed
- **Dropped the Python version matrix.** CI runs on 3.12 alone — the version the Makefile creates
  and the version this is actually run on. For a single-user application a three-version matrix
  triples CI time and offers three ways to fail on something that cannot affect the only user.
  `requires-python` stays at `>=3.10`; 3.10 and 3.13 were verified by hand and remain supported,
  they are just not re-checked on every push. Restoring the matrix is a four-line change if the
  project ever grows a second user.

## [0.1.6] — 2026-08-28

### Fixed

- **CI failed before running a single test.** `astral-sh/setup-uv@v3` with `enable-cache: true`
  globs for a `uv.lock` by default; this project installs with `uv pip install -e .` rather than
  `uv sync`, so there is none, and the cache step errored out. Pinned `cache-dependency-glob` to
  `pyproject.toml`.
- Bumped `actions/checkout` to v7 and `setup-uv` to v10.0.1, both of which run on Node 24 — the old
  versions emitted the Node 20 deprecation warning. (`setup-uv` publishes no floating `v10` tag, so
  the exact version is pinned.)

- **The end-to-end tests had never run in CI.** They require a Letterboxd export and `data/` is
  git-ignored, so on a clean checkout all 21 skipped and CI would have passed on 156 of 177 tests —
  with none of the pipeline, resolution, taste-model, leakage or retrieval coverage. `conftest` now
  generates a realistic 520-film export when none is present, reproducing the format's real
  awkwardness (film URIs vs entry URIs, the two-block list CSV, rewatch duplicates). A CI step fails
  the build if those tests ever silently skip again.

### Changed
- CI matrix is now 3.10, 3.12 and 3.13 — verified locally on all three.

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

[Unreleased]: https://github.com/ohorban/movie_recommender/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/ohorban/movie_recommender/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/ohorban/movie_recommender/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/ohorban/movie_recommender/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/ohorban/movie_recommender/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/ohorban/movie_recommender/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/ohorban/movie_recommender/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/ohorban/movie_recommender/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ohorban/movie_recommender/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ohorban/movie_recommender/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ohorban/movie_recommender/releases/tag/v0.1.0
