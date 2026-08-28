# Architecture

How the system works, and why it is built this way.

---

## The shape of the problem

This is a recommender with roughly 150 ratings and 140 written reviews to learn from. That number
drives nearly every design decision here. Standard collaborative filtering has nothing like enough
signal; a deep model would memorise the training set; a single "taste vector" would average away
the structure that makes the taste interesting.

What *is* abundant is language. The reviews say why a film worked. Wikipedia says what happens in a
film. The tag genome says what a film feels like to a thousand other viewers. So the system spends
its effort turning language into structure, and keeps the statistical machinery deliberately small
and heavily regularised.

---

## Pipeline

```
Letterboxd export ──┐
                    ├──► SQLite ──► embeddings ──► taste model ──► ranker ──► UI
TMDB / IMDb /       │
MovieLens / Wikipedia
```

Ten stages, all incremental, all resumable:

| # | Stage | What it does |
|---|---|---|
| 1 | `letterboxd` | Parse the newest export; upsert only rows that changed |
| 2 | `tmdb_discover` | Walk TMDB year by year, keep the best `catalog_size` films |
| 3 | `tmdb_detail` | Full detail: keywords, credits, audience reviews |
| 4 | `resolve` | Match user films to TMDB with confidence scores |
| 5 | `imdb` | Attach IMDb ratings as a second quality opinion |
| 6 | `movielens` | Tag genome + item-item collaborative filtering |
| 7 | `wikipedia` | Plot synopses — the deep NL layer, to a coverage target |
| 8 | `embeddings` | Profile and plot vectors; review vectors |
| 9 | `review_structuring` + `dossiers` | Claude turns prose into structured data |
| 10 | `taste_profile` + `ranker` | Fit and persist the model |

Every stage is wrapped by `RunLogger`, so the Data tab can show exactly what happened and when.

---

## Why `film_key` and not the Letterboxd URI

Letterboxd exports are internally inconsistent. `ratings.csv`, `watched.csv`, `watchlist.csv` and
`likes/films.csv` carry the **film** URI (`boxd.it/EMTM`, a 2–5 character slug). `reviews.csv` and
`diary.csv` carry the **entry** URI (`boxd.it/aRcQXb`, six characters) — unique per log, not per
film. There is no id that joins all of them.

So the canonical key is `slug(normalised title)|year`, computed identically everywhere. Film URIs
are recorded opportunistically when a file supplies one. This is the single most load-bearing
decision in the data layer, and `test_letterboxd.py` guards it directly.

Title normalisation folds accents *and* the ligatures NFKD leaves intact — `æ`, `ø`, `ß`, `ł`, `þ`
— which matters for the Scandinavian and Icelandic titles that appear in any real watch history.

---

## Incrementality

The user replaces the entire export folder each time, so every run sees the full history. Cheap
updates come from hashing at four levels:

1. **File** — SHA-256 in `source_files`. An unchanged export short-circuits the whole stage.
2. **Row** — ratings, watchlist entries and diary logs are compared before writing.
3. **Review text** — `text_hash` gates both the LLM call and the embedding.
4. **Document** — `content_hash` of the composed film document gates re-embedding.

The end-to-end test asserts the exact invariant: on a no-change update, the number of films
re-embedded equals the number newly detailed, and nothing at all is re-sent to Claude.

---

## Natural language into structure

### Reviews → taste facts

Each review becomes a record of verdict, sentiment, signal strength, liked and disliked aspects
(each categorised and weighted), themes, tone words, three engagement dimensions — and
`taste_signals`: generalisable statements phrased to transfer to unseen films.

The distinction the prompt insists on: *"liked the twist"* is not a taste signal; *"rewards films
that earn their twists structurally"* is. The first describes one film, the second predicts others.

Aspects aggregate into per-category affinities with the same shrinkage used elsewhere, so a single
emphatic review cannot dominate.

### Films → dossiers

Each film's plot, keywords, genome tags and credits become nine calibrated 0–1 scales:
`intellectual_demand`, `emotional_intensity`, `originality`, `feel_good`, `darkness`, `spectacle`,
`realism`, `humor`, `tension` — plus tone, themes, pacing, who it is for and who should avoid it.

Calibration is anchored explicitly in the prompt (*"darkness 0.9 is Come and See; 0.5 is The Dark
Knight; 0.1 is Paddington"*) because consistency across films matters far more than nuance within
any one of them.

Dossiers are generated **lazily** — the user's own films during setup, then finalists as they
surface. Cost tracks use, not catalog size.

---

## Embeddings

Anthropic serves no embeddings endpoint, so vectors come from a local sentence-transformer
(`BAAI/bge-small-en-v1.5` by default). Free, fast on Apple silicon, and cheap enough to recompute
the entire catalog when the model changes. The backend is an interface; an API encoder can be
swapped in without touching callers.

**Two documents per film**, because embedding quality is mostly a function of what you feed the
encoder:

- **Profile** — title, director, cast, genre, runtime, tagline, genome tags, keywords, overview and
  audience-review excerpts, assembled under an explicit ~1,800-character budget because small
  encoders truncate hard at ~512 tokens.
- **Plot** — the Wikipedia synopsis, chunked and mean-pooled. Only created when a genuine synopsis
  exists; re-embedding the overview as a "plot" would double the storage for no signal. Films
  without one fall back to profile similarity at feature time, so they are not penalised for
  lacking a Wikipedia article.

Storage is a `BLOB` column in SQLite rather than a separate vector file, keeping one source of
truth. A 30k × 384 matrix is 46 MB and a full cosine sweep is one matmul — an ANN index would add
dependencies and failure modes for no measurable gain at this scale.

---

## The taste model

Three deliberate choices:

**Preference, not rating.** A 3/5 from someone whose mean is 2.9 is mild approval. Everything
downstream works in standard deviations from the user's own mean, so a harsh rater and a generous
one produce comparable models. An explicit Letterboxd "like" adds a further +0.25.

**Multi-modal, not a single centroid.** Liked films are clustered (weighted k-means, k chosen from
the data and capped at 5). Each mode carries a centroid, a weight, exemplars and a label derived
from its dominant genome tags. A candidate only has to match *one* mode well. A separate repulsion
centroid is built from low-rated films.

**Shrunk affinities.** For every facet value:

```
affinity = mean_preference × n / (n + k)
```

An empirical-Bayes shrink toward zero, with `k` set per facet type (directors shrink less than
genres, because seeing a director twice means more than seeing a genre twice). Without it, a genre
seen once would be the strongest signal in the dataset.

Dossier scales get two numbers each: a preference-weighted **target** (where the sweet spot sits)
and a **weight** (how strongly the scale correlates with rating). The weight is what tells the
ranker whether the user actually cares about that dimension — and the Insights tab surfaces both.

---

## Candidate generation

Retrieval and ranking are separated because each source has a distinct failure mode. Embeddings
drift toward description-similar films; CF drifts toward popular ones; facet rules drift toward
whatever you already watch. The union is far more robust than any single source.

| Source | Retrieves |
|---|---|
| `taste:<mode>` | kNN from each mode centroid, budgeted by mode weight |
| `similar-to:<film>` | kNN from individual highly-rated films |
| `viewers-like-you` | CF neighbours weighted by rating |
| `well-reviewed-in-your-genres` | Facet rules with a quality floor |
| `your-watchlist` | Always eligible — an explicit statement of intent |
| `outside-your-usual` | Acclaimed films *far* from every centroid |
| `matches-your-request` | kNN from the parsed intent query |

Each candidate remembers which sources proposed it, which is what makes the "why am I seeing this"
panel truthful rather than reconstructed. Films proposed by several independent sources get a small
agreement bonus.

---

## Ranking

Seventeen features spanning embedding similarity (best mode, weighted modes, plot, repulsion),
facet affinities, CF evidence, quality and popularity priors, dossier-scale fit and recency.

### The leakage problem

Every one of those features is *fitted* on the ratings the model is trying to predict. That makes
this system unusually easy to fool, and the first version was fooled badly: it reported a
cross-validated rank correlation of **0.96** on real data where the honest figure was **0.53**.

Two distinct leaks, both of which had to be closed:

**A film scored against itself.** A director the user has seen exactly once has an affinity that is
a pure function of that one film's rating; the model "learns" to read the label back out of the
feature. The same applies to the centroid of the taste mode a film belongs to, and to the dislike
centroid. Training features are therefore built **leave-one-out** — each film is scored against a
profile with its own contribution subtracted (`affinity_value(..., exclude_pref=...)`,
`_loo_mode_matrix`, `_loo_dislike`).

**A profile fitted across folds.** Leave-one-out alone still reported ~0.91, because the affinities
and centroids were computed once over every rating and the cross-validation ran on top of them —
each fold's held-out films had helped build the signals they were then scored against. So the
reported metric now comes from `_fold_predictions`, which **rebuilds the entire taste profile inside
each fold** from that fold's training ratings only, and featurises the held-out films exactly as an
unrated candidate is featurised at recommendation time.

The lesson generalises: when your features are themselves fitted on the labels, cross-validating
the final estimator tells you almost nothing. The fitting has to happen inside the fold.

### Model selection

Candidates are ridge on standardised features (always) and gradient boosting (at 80+ examples),
scored by **Spearman rank correlation** on the honest out-of-fold predictions — the ordering is what
matters, not the predicted rating. The hand-tuned heuristic competes as a third candidate on the
same held-out basis, and on a few hundred ratings it frequently wins.

The safeguard that matters most:

```python
blend_weight = clip(held_out_spearman / 0.45, 0, 1)
final_score  = blend_weight × learned + (1 - blend_weight) × heuristic
```

If the honest evaluation says the model learned nothing, its influence goes to zero and the prior
takes over. `tests/test_leakage.py` asserts this directly by training on ratings drawn at random.

Training also removes each film's own CF contribution, so a film cannot predict itself.

Final selection is MMR-diversified over the embedding space, so the list is not five variations of
one film.

## The intent layer

A free-text request becomes: a rich `semantic_query` paragraph (written as though describing a real
film, because it is embedded and matched against film descriptions), optional hard filters, target
scales, a novelty setting, and a `taste_weight` in [0, 1].

`taste_weight` is the interesting number. It says how much the general taste profile should
override the literal request:

| Request | Weight |
|---|---|
| "something to watch tonight" | 0.9 |
| "something funny" | 0.7 |
| "a slow character study set in rural Japan" | 0.25 |
| "the 1974 Coppola one about surveillance" | 0.05 |

Over-filtering is the main failure mode of a system like this — a filter that removes the right
answer cannot be recovered later — so the prompt pushes preferences into `semantic_query` and
`target_scales` rather than into filters, and a filter set that eliminates almost everything (fewer
than 15 films) is discarded outright.

---

## Failure behaviour

Every external dependency degrades rather than breaks:

| Missing | Result |
|---|---|
| `ANTHROPIC_API_KEY` | No review structuring, dossiers, intent parsing or explanations. Ranking, retrieval and Insights all still work. |
| sentence-transformers | Falls back to the hash backend with a loud warning. Runs; recommends poorly. |
| MovieLens | No genome tags, no CF source. Content-based retrieval unaffected. |
| IMDb | Quality prior uses TMDB votes alone. |
| Wikipedia | Profile documents only; the plot feature falls back to profile similarity. |
| `TMDB_API_KEY` | Hard stop — there is no catalog without it. |

---

## Testing

125 tests, no network required.

The centrepiece is `test_pipeline_e2e.py`, which runs the **real** pipeline — real schema, real
ingest, real resolution, real taste model, real ranker — against the actual Letterboxd export, with
stand-in TMDB and Claude clients and the deterministic hash encoder. A break anywhere in the chain
surfaces there.

Tests that earned their keep during development, each of which caught a real bug:

- `æ`/`ø`/`ß` survive NFKD unchanged, breaking title matching for Scandinavian films.
- Newly matched TMDB ids violated the `user_films → movies` foreign key, because resolution ran
  before the films existed.
- Tied scores produced a *perfect* Spearman correlation via `argsort(argsort(x))`, which would have
  reported a flawless model precisely when every candidate scored the same.
- `embed_reviews` silently overwrote `embed_movies`' stats key, corrupting the update report.
- Two identically-scoring match candidates landed exactly *on* the auto-accept threshold.
- The ranker reported 0.96 rank correlation against a true 0.53, because features fitted on the
  ratings were cross-validated without rebuilding them per fold (see **Ranking** above).
- `anthropic` 1.x removed `temperature` from `Messages.create`, so every LLM call raised. Keyword
  arguments are now filtered against the installed SDK's signature.
- The *first* fix for that bug added the filter but never routed the call sites through it, and the
  unit test checked the filter rather than the code path using it — so the test passed while
  production stayed broken. `tests/test_llm.py` now drives the real methods against stand-in SDKs
  and asserts at source level that no `messages.create` call bypasses the filter. A unit test of a
  helper is not evidence that the helper is wired in.
