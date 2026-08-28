# movierec

A personal, multi-stage hybrid movie recommender built on my own Letterboxd history.

It reads my ratings, diary and written reviews, matches them against a catalog assembled from
TMDB, IMDb and MovieLens, turns my prose reviews into structured preference data with Claude, and
learns a ranking model on top of semantic embeddings. The result is a small web app that answers
three questions: *what should I watch tonight*, *find me something specific*, and *what does my
taste actually look like*.

Built for one user. There is no auth, no multi-tenancy, and no ambition to have any.

---

## What it does

**Learns from what I wrote, not just what I scored.** A 3/5 with an enthusiastic review means
something different from a silent 3/5. Every review is parsed into liked and disliked aspects,
themes, engagement levels and — most usefully — generalisable statements about my taste that
transfer to films I have never seen.

**Models taste as several things, not one.** Averaging every film I liked produces a vector that
describes nobody: the midpoint of a war film and an animated musical is neither. Liked films are
clustered into distinct modes, and a candidate only has to match one of them well.

**Knows how much it knows.** Every feature here is fitted on the very ratings the model predicts,
which makes honest evaluation hard — the first version reported 0.96 rank correlation where the
truth was 0.53. The taste profile is now rebuilt inside each cross-validation fold, and the learned
model is blended with a hand-tuned prior in proportion to its held-out score. When it has learned
nothing, its influence goes to zero rather than confidently ranking noise. At a few hundred ratings
the prior often wins, and the Insights tab says so plainly.

**Explains itself against my own history.** Every recommendation cites a specific film I rated or
something I wrote, and says so when a pick is a genuine stretch.

---

## Setup

### 1. Keys

Both are free to obtain. TMDB is required; Claude is required for review analysis and the
natural-language layer, but the system degrades gracefully without it.

```bash
cp .env.example .env
```

| Key | Where | Why |
|---|---|---|
| `TMDB_API_KEY` | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | The film catalog. Required. |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Review structuring, film profiles, the Ask tab. |

### 2. Install

```bash
make dev-install
```

That creates `.venv` and installs everything. No `activate` needed — every `make` target runs the
venv's own binaries. If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

`[embed]` pulls in PyTorch and sentence-transformers (~2 GB). Without it the system falls back to a
hash-based encoder that has no semantic understanding — usable for a smoke test, not for real
recommendations.

`make doctor` checks the environment, your keys and the database at any point.

### 3. Add your Letterboxd export

Export from [letterboxd.com/settings/data](https://letterboxd.com/settings/data), unzip, and drop
the whole folder into `data/`:

```
data/letterboxd-ohorban-2026-08-26-22-45-utc/
```

Export folders are git-ignored on purpose — this repo is public and the export contains your full
watch history and review text. Drop the `data/letterboxd-*/` line from `.gitignore` if you want it
versioned.

### 4. Build

```bash
make setup
```

First build takes roughly 45–90 minutes and downloads about 1.5 GB. Everything is cached and
resumable — interrupt it and re-run, and it picks up where it stopped.

| Stage | Roughly |
|---|---|
| TMDB catalog discovery + detail | 20–30 min |
| MovieLens 25M download + tag genome + CF | 15–25 min |
| IMDb ratings | 2 min |
| Wikipedia plot synopses (8k films) | 40–70 min |
| Embedding the catalog | 5–15 min |
| Claude: reviews + film profiles | 3–6 min |

### 5. Run

```bash
make app
```

---

## Updating

Export from Letterboxd again and replace the folder in `data/`. Then either press **Update
database** in the Data tab, or:

```bash
make update
```

Updates are incremental by design. Unchanged export files are skipped by checksum, films that
already have detail are not refetched, a review is only re-sent to Claude when its text changed,
and an embedding is only recomputed when its source document changed. A typical update after a few
new logs takes seconds and costs a few cents.

Old export folders can be left in `data/` — the newest is always used.

`MOVIEREC_WIKIPEDIA_LIMIT` is a **coverage target**, not a per-run batch: once that many films have
a synopsis the catalog sweep stops, so updates stay short. Raise it to deepen coverage (roughly
4–8 minutes per additional 1,000 films, paid once). Your own films are always fetched regardless.

---

## The interface

**Tonight** — one confident pick plus alternates, with a reroll. Thumbs down hides a film for 60
days.

**Ask** — free text. "Something tense but not bleak, under two hours." Claude turns it into a
semantic query plus hard filters and a weight controlling how much my general taste should override
the literal request; the panel shows exactly what it decided.

**Insights** — taste clusters with their exemplars, genre and tag affinities, the aspects my reviews
praise and criticise, where I disagree with the crowd, genre coverage and blind spots, and honest
model diagnostics including cross-validated rank correlation.

**Data** — update button, unmatched-film corrections, and the full pipeline run history.

---

## CLI

```bash
movierec setup                  # first-time build
movierec update                 # incremental update
movierec update --no-llm        # skip every Claude call
movierec rebuild                # delete and rebuild (keeps manual match corrections)
movierec status                 # what is in the database
movierec recommend              # a pick, in the terminal
movierec recommend "a heist film that is actually about friendship" -n 5
movierec unmatched              # films needing a manual TMDB match
movierec fix-match <film_key> <tmdb_id>
```

---

## Costs

Embeddings run locally and are free. Claude usage with `claude-sonnet-5`:

| | Approx |
|---|---|
| First build (~140 reviews + ~400 film profiles) | $1–3 |
| Each update (only what changed) | a few cents |
| One natural-language request | ~$0.02 |

TMDB, IMDb, MovieLens and Wikipedia are all free.

---

## Matching caveat

Letterboxd exports contain no external ids, and `reviews.csv` and `diary.csv` carry *entry* URIs
rather than *film* URIs — so films are joined on normalised title plus year, and matched to TMDB by
search with a confidence score. Anything below the bar is flagged rather than guessed at; the Data
tab lists them. Corrections are permanent and survive a full rebuild.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pipeline and the model actually work
- [`CHANGELOG.md`](CHANGELOG.md) — versioned history

## Development

```bash
make test        # 150 tests, no network required
make lint
make fmt
make doctor      # environment, keys and database health
```

Python 3.12 by default; override with `make dev-install PYTHON_VERSION=3.11`.

The test suite runs the real pipeline end to end against stand-in TMDB and Claude clients, so a
break anywhere in the chain surfaces there rather than in production.

## Licence

MIT. Personal project. IMDb and MovieLens datasets carry their own non-commercial terms.
