# movierec

A movie recommender that learns from your own Letterboxd history — your ratings, your diary, and
the reviews you wrote. Built for one person. No accounts, no sharing.

---

## Setup

You need two free API keys and about two hours for the first build. After that, updates take
minutes.

**1. Get the keys**

| Key | Where | What it's for |
|---|---|---|
| `TMDB_API_KEY` | [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) | The movie catalog. Required. |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Reading your reviews, and the Ask tab. |

**2. Put them in a file**

```bash
cp .env.example .env
```

Open `.env` and paste the two keys in. Everything else in that file has a working default.

**3. Add your Letterboxd data**

Export it from [letterboxd.com/settings/data](https://letterboxd.com/settings/data). Unzip it. Drop
the whole folder into `data/`, so you end up with something like:

```
data/letterboxd-yourname-2026-08-26-22-45-utc/
```

**4. Install and build**

```bash
make dev-install     # ~2 min, downloads PyTorch (about 2 GB)
make setup           # ~2 hours, see the table below
make app             # opens the web app
```

If `make` says `uv: command not found`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

`make setup` can be stopped with Ctrl-C and restarted. It picks up where it left off.

**How long the first build takes**

| Step | Time |
|---|---|
| Download the movie catalog from TMDB | 25–35 min |
| Download MovieLens (250 MB) and IMDb ratings | 15–20 min |
| Download plot summaries from Wikipedia | 50–70 min |
| Turn everything into embeddings | 5–15 min |
| Claude reads your reviews and profiles your films | 10–15 min |

The Wikipedia step is the slow one. Lower `MOVIEREC_WIKIPEDIA_LIMIT` in `.env` if you want a faster
first build, or set `MOVIEREC_ENABLE_WIKIPEDIA=false` to skip it.

---

## Using it

`make app` opens a web page with four tabs.

**Tonight** — one pick, plus a few alternatives. Hit Reroll for a different angle. Thumbs down hides
a film for 60 days.

**Ask** — type what you're in the mood for. "Something tense but not bleak, under two hours."
Claude reads that, turns it into a search, and the recommender ranks the catalog against it.

**Insights** — what the system has worked out about your taste, and how well it actually knows you.

**Data** — the update button, and any films it couldn't match to the catalog.

---

## Updating

Export from Letterboxd again and replace the folder in `data/`. Then press **Update database** in
the Data tab, or run:

```bash
make update
```

Only new things get processed. Unchanged files are skipped, films already in the catalog aren't
re-downloaded, and a review is only re-read by Claude if you edited it. A normal update takes a few
minutes and costs a few cents.

Old export folders can stay in `data/`. The newest one is always used.

---

## How it works

Nine steps, in order.

**1. Read your Letterboxd export**

Parses your ratings, watched list, diary, reviews, watchlist, likes and lists.

One wrinkle: Letterboxd's files don't share a single ID. `ratings.csv` uses a film ID, but
`reviews.csv` uses a *review* ID, which is different for every review. So films are matched on
their title and year, cleaned up so that "Amélie" and "Amelie" count as the same film.

**2. Build a movie catalog**

Downloads about 30,000 films from TMDB — description, genres, keywords, cast, crew, popularity.
Then adds three more sources:

- **IMDb ratings**, as a second opinion on quality.
- **MovieLens**, which gives two things: 1,128 human-written tags per film ("atmospheric",
  "thought-provoking"), and "people who liked what you liked also liked X".
- **Wikipedia plot summaries.** A TMDB description is two sentences of marketing. A Wikipedia plot
  is several hundred words about what actually happens. That difference is what lets the system
  tell apart two films that share a genre but nothing else.

**3. Match your films to the catalog**

Letterboxd doesn't tell us which TMDB film you watched, so each one is looked up by title and year
and scored for confidence. Anything it isn't sure about gets flagged in the Data tab instead of
being guessed at. Corrections you make there are permanent.

**4. Turn text into numbers**

Every film gets a written description assembled from everything above — director, cast, genre,
tags, keywords, plot. That text is converted into a list of numbers (an "embedding") that captures
its meaning. Two films with similar meaning end up with similar numbers, so similarity becomes
arithmetic.

This runs on your own machine and is free. Your reviews get the same treatment.

**5. Claude reads your reviews**

Each review you wrote is turned into structured data: what you liked, what you didn't, how engaged
you were, and — most usefully — statements that carry over to films you haven't seen.

"Liked the twist" is useless. "Rewards films that earn their twists" predicts other films.

**6. Claude profiles films**

Each film gets rated 0 to 1 on nine scales: how much thinking it demands, how dark it is, how
funny, how tense, how original, how uplifting, and so on. The prompt anchors these against real
examples so the numbers mean the same thing across films.

This happens on demand, so it only costs money for films you might actually be shown.

**7. Build your taste profile**

Three ideas here:

- **Your ratings are graded on your own curve.** If your average is 2.9, a 3 out of 5 from you is
  mild approval, not a middling score.
- **You don't have one taste, you have several.** Averaging every film you liked gives a result
  that describes nothing — the midpoint of a war film and an animated musical is neither. So liked
  films are grouped into clusters, and a recommendation only has to match one of them.
- **Weak evidence is discounted.** A director you've seen once shouldn't outrank a genre you've
  seen forty times. Every preference is pulled toward neutral in proportion to how little supports
  it.

**8. Train the ranker, and check it honestly**

A model is trained to predict your rating from all of the above.

The hard part is knowing whether it works. Every input is built from the same ratings we're trying
to predict, so a careless test scores each film partly against itself. The first version of this
reported 96% accuracy when the truth was 53%. Now the taste profile is rebuilt from scratch inside
each test fold, so a film being scored never helped build the thing scoring it.

The model is then trusted **in proportion to how well it did on that honest test**. If it learned
nothing, it's ignored and a hand-tuned fallback takes over. Right now, on ~155 ratings, the
fallback wins — and the Insights tab says so.

**9. Recommend**

Candidates come from six places at once: films near each of your taste clusters, films near
individual favourites, "people like you also liked", well-reviewed films in genres you rate highly,
your watchlist, and a deliberate slot for good films *far* from your usual taste.

Each source fails in a different way, so the mix is more reliable than any one of them. The ranker
sorts the pile, near-duplicates are dropped so you don't get five versions of the same film, and
Claude writes a short note for each pick citing a film you actually rated.

---

## Commands

```bash
make setup            # first build
make update           # incremental update
make app              # web interface
make doctor           # check keys, install, and database health
make rebuild          # wipe and rebuild (keeps your manual film matches)
make test             # 187 tests, no network needed
```

The `movierec` command does the same and a bit more:

```bash
movierec recommend "a heist film that's actually about friendship" -n 5
movierec unmatched                       # films needing a manual match
movierec fix-match <film_key> <tmdb_id>  # fix one
movierec status                          # what's in the database
```

---

## Cost

Embeddings run on your machine and are free. So are TMDB, IMDb, MovieLens and Wikipedia.

Only Claude costs money:

| | Roughly |
|---|---|
| First build | $1–3 |
| Each update | a few cents |
| One question in the Ask tab | ~$0.02 |

---

## If something goes wrong

`make doctor` checks your Python version, whether everything installed, whether your keys are set,
and what's in the database.

**"No system Python" or install errors** — delete `.venv` and run `make dev-install` again.

**Films missing from recommendations** — check the Data tab. TV series logged on Letterboxd will
never match a movie database; that's expected.

**Recommendations feel generic** — the model gets better with more ratings. Below about 100 it's
mostly working from genre and quality signals.

---

## Notes

Your Letterboxd export is git-ignored on purpose: this repo is public and the export contains your
full watch history and review text. Delete the `data/letterboxd-*/` line from `.gitignore` if you
want it version-controlled.

`docs/ARCHITECTURE.md` has the technical detail — schema, feature list, the leakage problem, and
what each test is guarding against.

Requires Python 3.10 or newer. CI runs on 3.12.

MIT licensed. The datasets it downloads have their own terms; see `LICENSE`.
