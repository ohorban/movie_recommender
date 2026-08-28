"""movierec — the web interface.

Four surfaces:
  Tonight   one confident pick plus alternates, with a reroll
  Ask       free-text requests routed through the natural-language layer
  Insights  what the system has learned about your taste, and how well
  Data      refresh from a new Letterboxd export, and fix bad matches

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from movierec import __version__  # noqa: E402
from movierec.config import load_config  # noqa: E402
from movierec.db import init_db  # noqa: E402
from movierec.ingest.letterboxd import latest_export_dir  # noqa: E402
from movierec.ingest.resolve import set_override, unresolved_report  # noqa: E402
from movierec.recommend.engine import RecommendationEngine  # noqa: E402
from movierec.recommend.ranker import TasteRanker  # noqa: E402
from movierec.taste import insights as ins  # noqa: E402
from movierec.taste.profile import load_profile  # noqa: E402

st.set_page_config(
    page_title="movierec", page_icon="🎬", layout="wide", initial_sidebar_state="expanded"
)

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  .rec-title { font-size: 1.45rem; font-weight: 650; line-height: 1.25; margin-bottom: .1rem; }
  .rec-meta  { color: var(--text-color-light, #8a8f98); font-size: .86rem; margin-bottom: .7rem; }
  .rec-hook  { font-size: 1.02rem; line-height: 1.55; margin-bottom: .55rem; }
  .rec-because { border-left: 3px solid #d4a017; padding-left: .75rem; font-size: .93rem;
                 line-height: 1.5; margin-bottom: .5rem; }
  .rec-caveat { font-size: .86rem; opacity: .75; margin-bottom: .4rem; }
  .pill { display:inline-block; padding: .12rem .55rem; margin: 0 .25rem .28rem 0;
          border-radius: 999px; font-size: .72rem; background: rgba(128,128,128,.16); }
  .stat-label { font-size: .78rem; opacity: .7; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def get_conn(_db_path: str):
    return init_db(_db_path, check_same_thread=False)


@st.cache_resource(show_spinner="Loading the model…")
def get_engine(_db_path: str, _version: int):
    cfg = get_config()
    return RecommendationEngine(get_conn(_db_path), cfg)


def bump_cache() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    get_engine.clear()


def engine() -> RecommendationEngine:
    cfg = get_config()
    return get_engine(str(cfg.db_path), st.session_state.get("data_version", 0))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_rec(item, *, hero: bool = False, key_prefix: str = "") -> None:
    cols = st.columns([1, 3.2] if hero else [1, 3.6])
    with cols[0]:
        if item.poster_url:
            st.image(item.poster_url, width="stretch")
        else:
            st.markdown(
                "<div style='height:180px;background:rgba(128,128,128,.12);border-radius:8px'></div>",
                unsafe_allow_html=True,
            )
    with cols[1]:
        year = f" ({item.year})" if item.year else ""
        st.markdown(f"<div class='rec-title'>{item.title}{year}</div>", unsafe_allow_html=True)

        meta = []
        if item.runtime:
            meta.append(f"{item.runtime} min")
        if item.genres:
            meta.append(" · ".join(item.genres[:3]))
        if item.tmdb_rating:
            meta.append(f"TMDB {item.tmdb_rating:.1f}")
        if item.imdb_rating:
            meta.append(f"IMDb {item.imdb_rating:.1f}")
        if item.on_watchlist:
            meta.append("★ on your watchlist")
        st.markdown(
            f"<div class='rec-meta'>{' &nbsp;·&nbsp; '.join(meta)}</div>", unsafe_allow_html=True
        )

        if item.hook:
            st.markdown(f"<div class='rec-hook'>{item.hook}</div>", unsafe_allow_html=True)
        elif item.overview:
            st.markdown(
                f"<div class='rec-hook'>{item.overview[:320]}</div>", unsafe_allow_html=True
            )

        if item.because:
            st.markdown(
                f"<div class='rec-because'><b>Why you:</b> {item.because}</div>",
                unsafe_allow_html=True,
            )
        if item.caveat:
            st.markdown(f"<div class='rec-caveat'>⚠︎ {item.caveat}</div>", unsafe_allow_html=True)

        if item.dossier:
            tone = item.dossier.get("tone") or []
            themes = item.dossier.get("themes") or []
            pills = "".join(f"<span class='pill'>{t}</span>" for t in [*tone[:4], *themes[:4]])
            if pills:
                st.markdown(pills, unsafe_allow_html=True)

        b = st.columns([1, 1, 1, 1, 3])
        eng = engine()
        with b[0]:
            if st.button("👍", key=f"{key_prefix}up{item.tmdb_id}", help="More like this"):
                eng.record_feedback(item.tmdb_id, "like", "ui")
                st.toast(f"Noted — more like {item.title}")
        with b[1]:
            if st.button(
                "👎", key=f"{key_prefix}dn{item.tmdb_id}", help="Not for me — hide for 60 days"
            ):
                eng.record_feedback(item.tmdb_id, "dislike", "ui")
                st.toast(f"Hidden: {item.title}")
                st.rerun()
        with b[2]:
            if st.button("Seen it", key=f"{key_prefix}sn{item.tmdb_id}"):
                eng.record_feedback(item.tmdb_id, "watched", "ui")
                st.toast("Marked as seen")
                st.rerun()
        with b[3]:
            st.link_button("TMDB", item.tmdb_url)

        with st.expander("Why this surfaced"):
            st.caption("Retrieval sources: " + ", ".join(item.sources))
            if item.features:
                top = sorted(item.features.items(), key=lambda t: -abs(t[1]))[:8]
                st.dataframe(
                    {"signal": [k for k, _ in top], "value": [round(v, 3) for _, v in top]},
                    hide_index=True,
                    width="stretch",
                )
            if item.dossier:
                st.caption(item.dossier.get("who_its_for", ""))


def not_ready(message: str) -> None:
    st.warning(message)
    st.markdown(
        "**To build the database:**\n"
        "1. Copy `.env.example` to `.env` and add your `TMDB_API_KEY` and `ANTHROPIC_API_KEY`.\n"
        "2. Run `make setup` (or `movierec setup`) — the first build takes a while.\n"
        "3. Come back here."
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
cfg = get_config()
db_exists = Path(cfg.db_path).exists()

with st.sidebar:
    st.markdown("### 🎬 movierec")
    st.caption(f"v{__version__}")

    if db_exists:
        conn = get_conn(str(cfg.db_path))
        stats = ins.headline_stats(conn)
        a, b = st.columns(2)
        a.metric("Watched", f"{stats['films_watched']:,}")
        b.metric("Rated", f"{stats['films_rated']:,}")
        a.metric("Reviews", f"{stats['reviews_written']:,}")
        b.metric("Catalog", f"{stats['catalog_size']:,}")
        if stats["unmatched"] or stats["needs_review"]:
            st.warning(
                f"{stats['unmatched']} unmatched, {stats['needs_review']} to confirm — see the Data tab."
            )
    else:
        st.info("No database yet. Run `make setup`.")

    st.divider()
    export = latest_export_dir(cfg.data_dir)
    st.caption(f"**Export:** {export.name if export else 'none found'}")
    missing = cfg.missing_credentials()
    st.caption("**Keys:** " + ("all set ✓" if not missing else "missing " + ", ".join(missing)))
    st.caption(f"**Embeddings:** {cfg.embed_backend} · {cfg.embed_model.split('/')[-1]}")
    st.caption(f"**LLM:** {cfg.llm_model}")


st.title("What should I watch?")

tab_tonight, tab_ask, tab_insights, tab_data = st.tabs(["Tonight", "Ask", "Insights", "Data"])


# --------------------------------------------------------------------------- #
# Tonight
# --------------------------------------------------------------------------- #
with tab_tonight:
    if not db_exists:
        not_ready("The database has not been built yet.")
    else:
        eng = engine()
        ready, reason = eng.readiness()
        if not ready:
            not_ready(reason)
        else:
            left, right = st.columns([3, 1])
            with left:
                st.caption(
                    "One pick, chosen from your whole history. Reroll for a different angle."
                )
            with right:
                reroll = st.button("🎲 Reroll", width="stretch")

            if reroll or "tonight" not in st.session_state:
                st.session_state["tonight_seen"] = (
                    st.session_state.get("tonight_seen", set()) if reroll else set()
                )
                with st.spinner("Thinking about what you'd actually enjoy…"):
                    result = eng.recommend(
                        "", n=5, exclude=st.session_state.get("tonight_seen", set()), explain=True
                    )
                st.session_state["tonight"] = result
                st.session_state["tonight_seen"] = st.session_state.get("tonight_seen", set()) | {
                    i.tmdb_id for i in result.items
                }

            result = st.session_state.get("tonight")
            if result and result.items:
                render_rec(result.items[0], hero=True, key_prefix="t0")
                if len(result.items) > 1:
                    st.divider()
                    st.markdown("##### Or one of these")
                    for item in result.items[1:]:
                        render_rec(item, key_prefix="ta")
                        st.markdown("")
                st.caption(f"Ranked {result.pool_size:,} candidates · model: {result.ranker_kind}")
            elif result:
                st.info("No candidates left — try clearing dismissals or updating your data.")
            for note in result.notes if result else []:
                st.caption(f"ℹ︎ {note}")


# --------------------------------------------------------------------------- #
# Ask
# --------------------------------------------------------------------------- #
EXAMPLES = [
    "something tense but not bleak, under two hours",
    "a smart sci-fi I probably haven't heard of",
    "something inspiring about people who are very good at their job",
    "a comfort watch for a bad day",
    "a foreign-language film that will stay with me",
]

with tab_ask:
    if not db_exists:
        not_ready("The database has not been built yet.")
    else:
        eng = engine()
        if not eng.is_ready():
            not_ready("The taste model has not been trained yet.")
        else:
            st.caption(
                "Describe what you're in the mood for. Claude interprets it against your taste, "
                "then the recommender ranks the catalog."
            )
            query = st.text_input(
                "What are you after?",
                key="ask_query",
                placeholder="e.g. a slow-burn mystery with a great ending",
            )

            chips = st.columns(len(EXAMPLES))
            for col, example in zip(chips, EXAMPLES):
                if col.button(example, key=f"ex_{example[:14]}", width="stretch"):
                    st.session_state["ask_query"] = example
                    query = example

            c1, c2, c3 = st.columns([1, 1, 2])
            n = c1.slider("How many", 3, 12, 6)
            allow_seen = c2.checkbox("Include films I've seen", value=False)

            if st.button("Find it", type="primary") or (
                query and st.session_state.get("last_ask") != query
            ):
                if query.strip():
                    st.session_state["last_ask"] = query
                    with st.spinner("Interpreting your request…"):
                        st.session_state["ask_result"] = eng.recommend(
                            query, n=n, explain=True, allow_seen=allow_seen
                        )

            result = st.session_state.get("ask_result")
            if result and result.items:
                if result.intent.interpretation:
                    st.info(f"**Reading that as:** {result.intent.interpretation}")
                with st.expander("What the request became"):
                    st.write("**Semantic query used for retrieval:**")
                    st.caption(result.intent.semantic_query)
                    meta = {
                        "taste weight": round(result.intent.taste_weight, 2),
                        "novelty": result.intent.novelty,
                        "genres in": result.intent.include_genres or "—",
                        "genres out": result.intent.exclude_genres or "—",
                        "years": f"{result.intent.year_min or '—'}–{result.intent.year_max or '—'}",
                        "max runtime": result.intent.runtime_max or "—",
                        "target scales": result.intent.target_scales or "—",
                    }
                    st.json(meta)
                st.divider()
                for item in result.items:
                    render_rec(item, key_prefix="ask")
                    st.markdown("")
                st.caption(f"Ranked {result.pool_size:,} candidates")
            elif result:
                st.info("Nothing matched. Try loosening the request.")


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
with tab_insights:
    if not db_exists:
        not_ready("The database has not been built yet.")
    else:
        conn = get_conn(str(cfg.db_path))
        cfg2 = get_config()
        profile = load_profile(conn)
        stats = ins.headline_stats(conn)

        cols = st.columns(5)
        cols[0].metric("Films watched", f"{stats['films_watched']:,}")
        cols[1].metric("Rated", f"{stats['films_rated']:,}")
        cols[2].metric("Mean rating", f"{stats['mean_rating']}")
        cols[3].metric("Reviews", f"{stats['reviews_written']:,}")
        cols[4].metric("Watchlist", f"{stats['watchlist']:,}")

        if profile is None:
            st.info("No taste profile yet — run `movierec update`.")
        else:
            summary = profile.summary
            if summary:
                st.markdown(f"### {summary.get('headline', '')}")
                a, b = st.columns(2)
                with a:
                    st.markdown("**Reliably works for you**")
                    for x in summary.get("loves", []):
                        st.markdown(f"- {x}")
                with b:
                    st.markdown("**Reliably doesn't**")
                    for x in summary.get("dislikes", []):
                        st.markdown(f"- {x}")
                if summary.get("contradictions"):
                    with st.expander("Tensions in your taste"):
                        for x in summary["contradictions"]:
                            st.markdown(f"- {x}")
                if summary.get("blind_spots"):
                    with st.expander("Blind spots"):
                        for x in summary["blind_spots"]:
                            st.markdown(f"- {x}")
                if summary.get("rating_style"):
                    st.caption(f"**How you rate:** {summary['rating_style']}")
                st.divider()

            st.markdown("### The clusters in your favourites")
            st.caption(
                "Your liked films don't form one taste — they form several. "
                "Recommendations only have to match one of these well."
            )
            mode_cols = st.columns(max(1, len(profile.modes)))
            for col, mode in zip(mode_cols, profile.modes):
                with col:
                    st.markdown(f"**{mode.label}**")
                    st.caption(f"{mode.size} films · {mode.weight:.0%} of your taste")
                    for e in mode.exemplars[:4]:
                        st.markdown(
                            f"<span style='font-size:.86rem'>{e['title']} — {e['rating']}★</span>",
                            unsafe_allow_html=True,
                        )
                    if mode.top_genres:
                        st.caption(", ".join(mode.top_genres))

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Genre affinity")
                st.caption(
                    "Shrunk mean preference in standard deviations. Positive = you rate it above your own average."
                )
                df = ins.affinity_table(profile, "genre")
                if not df.empty:
                    st.bar_chart(df.set_index("name")["affinity"], horizontal=True, height=380)
            with c2:
                st.markdown("##### What your reviews praise and criticise")
                st.caption("Extracted from your own words by Claude, aggregated by aspect.")
                df = ins.aspect_table(profile)
                if not df.empty:
                    st.bar_chart(df.set_index("aspect")["affinity"], horizontal=True, height=380)
                else:
                    st.info("No structured review data yet.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Your sweet spots")
                st.caption(
                    "Where you sit on each dimension, and how much it actually predicts your rating."
                )
                df = ins.scale_table(profile)
                if not df.empty:
                    st.dataframe(df, hide_index=True, width="stretch")
                else:
                    st.info("Not enough profiled films yet.")
            with c2:
                st.markdown("##### Tag affinity")
                df = ins.affinity_table(profile, "tag", top=12)
                if not df.empty:
                    st.bar_chart(df.set_index("name")["affinity"], horizontal=True, height=340)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Rating distribution")
                df = ins.rating_distribution(conn)
                if not df.empty:
                    st.bar_chart(df.set_index("rating")["films"], height=250)
                st.markdown("##### By decade")
                df = ins.decade_profile(conn)
                if not df.empty:
                    st.dataframe(df, hide_index=True, width="stretch", height=240)
            with c2:
                st.markdown("##### Where you disagree with the crowd")
                dis = ins.crowd_disagreement(conn)
                t1, t2 = st.tabs(["You liked it more", "You liked it less"])
                with t1:
                    if not dis["underrated"].empty:
                        st.dataframe(
                            dis["underrated"], hide_index=True, width="stretch", height=250
                        )
                with t2:
                    if not dis["overrated"].empty:
                        st.dataframe(dis["overrated"], hide_index=True, width="stretch", height=250)

            st.divider()
            st.markdown("### How well does the model know you?")
            ranker = TasteRanker.load(conn)
            if ranker:
                m = ranker.metrics
                cols = st.columns(4)
                cols[0].metric("Model", m.model_kind)
                cols[1].metric(
                    "Rank correlation",
                    f"{m.spearman:.3f}",
                    help="Cross-validated Spearman between predicted and actual preference. "
                    "Above 0.3 is genuinely useful at this data size.",
                )
                cols[2].metric("NDCG@10", f"{m.ndcg_at_10:.3f}")
                cols[3].metric("Trained on", f"{m.n_train} films")
                st.progress(
                    min(1.0, m.blend_weight),
                    text=f"Learned model influence: {m.blend_weight:.0%} "
                    f"(the rest is the hand-tuned prior — this rises as you rate more films)",
                )
                if m.model_kind == "heuristic":
                    st.caption(
                        "No learned model beat the hand-tuned prior on held-out data, so the prior "
                        "is doing the ranking. This is expected at a few hundred ratings and "
                        "resolves itself as you rate more."
                    )
                if m.top_features:
                    label = (
                        "**Weights the prior uses**"
                        if m.model_kind == "heuristic"
                        else "**What actually predicts your rating**"
                    )
                    st.markdown(label)
                    st.dataframe(
                        {
                            "signal": [f for f, _ in m.top_features],
                            "weight": [round(v, 4) for _, v in m.top_features],
                        },
                        hide_index=True,
                        width="stretch",
                    )

            with st.expander("Films that look alike but you rated very differently"):
                st.caption(
                    "These are the cases where description-similarity alone would mislead the recommender."
                )
                df = ins.most_similar_pairs(conn, profile, profile.embed_model or cfg2.embed_model)
                if not df.empty:
                    st.dataframe(df, hide_index=True, width="stretch")

            with st.expander("Genre coverage — what you haven't explored"):
                df = ins.genre_coverage(conn)
                if not df.empty:
                    st.dataframe(df.sort_values("coverage_pct"), hide_index=True, width="stretch")

            if profile.taste_signals:
                with st.expander(
                    f"All {len(profile.taste_signals)} preference signals from your reviews"
                ):
                    for s in profile.taste_signals:
                        st.markdown(f"- {s}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("### Update from a new Letterboxd export")
    st.caption(
        f"Drop the unzipped export folder into `{cfg.data_dir.name}/` — the newest one wins. "
        "Only what actually changed gets reprocessed, so updates are quick and cheap."
    )

    export = latest_export_dir(cfg.data_dir)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.info(
            f"**Current export:** `{export.name}`"
            if export
            else f"**No export found** in `{cfg.data_dir}`."
        )
    with c2:
        skip_llm = st.checkbox(
            "Skip Claude", value=False, help="Faster and free, but no new review analysis."
        )

    disabled = export is None or bool(cfg.missing_credentials(need_llm=False))
    if cfg.missing_credentials(need_llm=False):
        st.error("TMDB_API_KEY is not set in `.env`, so the catalog cannot be updated.")

    if st.button("🔄 Update database", type="primary", disabled=disabled):
        from movierec.pipeline import run as run_pipeline

        bar = st.progress(0.0, text="Starting…")
        log_box = st.empty()

        def on_progress(message: str, fraction: float) -> None:
            bar.progress(min(1.0, max(0.0, fraction)), text=message)

        try:
            report = run_pipeline(
                cfg,
                kind="update",
                progress=on_progress,
                conn=get_conn(str(cfg.db_path)),
                skip_llm=skip_llm,
            )
            bar.progress(1.0, text="Done")
            bump_cache()
            st.success("Update complete.")
            with log_box.container():
                for stage, s in report.stages.items():
                    interesting = {
                        k: v
                        for k, v in s.items()
                        if not str(k).startswith("_") and v not in (0, "", [], None, False)
                    }
                    st.markdown(
                        f"**{stage}** — "
                        + (
                            ", ".join(f"`{k}`: {v}" for k, v in interesting.items())
                            or "nothing to do"
                        )
                    )
                for w in report.warnings:
                    st.warning(w)
        except Exception as exc:
            bar.empty()
            st.error(f"{type(exc).__name__}: {exc}")

    if db_exists:
        conn = get_conn(str(cfg.db_path))
        st.divider()
        st.markdown("### Films that need a match")
        st.caption(
            "Letterboxd exports carry no external ids, so films are matched by title and year. "
            "Anything the matcher wasn't confident about is listed here. Corrections are permanent — "
            "they survive a full rebuild."
        )
        rows = unresolved_report(conn)
        if not rows:
            st.success("Every film in your history is confidently matched.")
        else:
            for r in rows[:40]:
                cols = st.columns([3, 3, 1.4, 1.2])
                cols[0].markdown(f"**{r['title']}** ({r['year']})")
                if r["tmdb_id"]:
                    cols[1].markdown(f"matched → {r['matched_title']} ({r['matched_year']})")
                else:
                    cols[1].markdown("_no match found_")
                conf = r["match_confidence"]
                cols[2].markdown(f"`{conf:.2f}`" if conf is not None else "`—`")
                with cols[3]:
                    with st.popover("Fix"):
                        st.caption(f"`{r['film_key']}`")
                        new_id = st.number_input(
                            "TMDB id",
                            min_value=0,
                            step=1,
                            key=f"fx{r['film_key']}",
                            value=int(r["tmdb_id"] or 0),
                        )
                        cc = st.columns(2)
                        if cc[0].button("Pin", key=f"pin{r['film_key']}"):
                            set_override(conn, r["film_key"], int(new_id) or None, "set from UI")
                            conn.commit()
                            bump_cache()
                            st.rerun()
                        if cc[1].button(
                            "Confirm", key=f"ok{r['film_key']}", help="Accept the current match"
                        ):
                            set_override(conn, r["film_key"], r["tmdb_id"], "confirmed in UI")
                            conn.commit()
                            bump_cache()
                            st.rerun()
            if len(rows) > 40:
                st.caption(f"…and {len(rows) - 40} more.")

        st.divider()
        st.markdown("### Recent pipeline runs")
        hist = ins.ingest_history(conn, limit=30)
        if not hist.empty:
            st.dataframe(
                hist[["run_id", "kind", "stage", "status", "started_at", "stats_json"]],
                hide_index=True,
                width="stretch",
                height=320,
            )

        with st.expander("Danger zone"):
            st.caption(
                "A full rebuild deletes the database and reconstructs it from your export plus the "
                "cached API responses. Your manual match corrections are preserved. "
                "Run it from the terminal: `movierec rebuild`."
            )
