"""RM Copilot - Streamlit UI.

Who to call -> meeting prep (portfolio, signals, suggested topics, cited talking
points) -> fact-check -> RM reviews/edits/finalises. The copilot never contacts
a client: the RM decides what to say and does it themselves.

All client data is SYNTHETIC. Product facts come from OCBC's public API data
(indicative, possibly outdated).
"""
from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src import core
from src import eval as fc

load_dotenv(find_dotenv(usecwd=True))
st.set_page_config(page_title="RM Copilot", layout="wide")


def esc(text) -> str:
    """Escape $ so Streamlit markdown doesn't render S$ amounts as maths."""
    return str(text).replace("$", "\\$")


def freshness(as_of: str) -> str:
    """'2026-10-05 (2 days old)'."""
    n = core.age_days(as_of)
    return f"{as_of} ({n} day{'s' if n != 1 else ''} old)"


@st.cache_data
def get_book():
    """Synthetic book of clients."""
    return core.load_book()


@st.cache_data
def get_sheets():
    """Product fact sheets from the cleaned OCBC file."""
    return core.build_fact_sheets()


@st.cache_data
def get_ranked():
    """Clients ranked by the explainable priority model."""
    return core.rank_clients(get_book())


@st.cache_data
def get_eval():
    """Deterministic eval report (no API calls; doesn't overwrite the saved report)."""
    return fc.run_all(0, save=False)


book, sheets, ranked = get_book(), get_sheets(), get_ranked()
snapshot = book["clients"].as_of.iloc[0]
pulled = core.load_products()[0]["pulled_on"]
ss = st.session_state
ss.setdefault("briefs", {})
ss.setdefault("opened_at", {})

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("RM Copilot")
    rms = book["rms"]
    rm_name = st.selectbox("Relationship manager", rms.rm_name)
    rm_id = rms.set_index("rm_name").loc[rm_name, "rm_id"]
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    use_ai = st.toggle("Draft talking points with Claude", value=has_key,
                       help="Off, or if the API fails, a template generator is used instead.")
    if use_ai and not has_key:
        st.caption("No API key found; the template fallback will be used.")
    st.subheader("Data freshness")
    st.markdown(esc(f"**Client data (synthetic):** snapshot {freshness(snapshot)}"))
    st.markdown(esc(f"**Product facts (OCBC public API):** pulled {freshness(pulled)}"))
    st.warning("Product facts are indicative and may be outdated. Confirm current rates and terms before quoting.")
    st.info("The copilot only drafts. It never contacts clients or takes action. You review and decide everything.")

tab_call, tab_prep, tab_check, tab_metrics = st.tabs(["Who to call", "Meeting prep", "Fact-check & eval", "Metrics"])

# ---------------------------------------------------------------- who to call
book_rm = ranked[ranked.rm_id == rm_id].reset_index(drop=True)
with tab_call:
    st.subheader(esc(f"{rm_name}: {len(book_rm)} clients, ranked by likelihood a conversation leads to action"))
    st.caption("Score = logistic regression trained on past (synthetic) outreach outcomes. "
               "'Why' lists the features pushing this client above an average client (log-odds contribution).")
    view = pd.DataFrame({
        "Rank": range(1, len(book_rm) + 1),
        "Client": book_rm.name + " (" + book_rm.client_id + ")",
        "Score": book_rm.score.round(2),
        "Why": book_rm.why.map(lambda w: "; ".join(w) or "No strong signals"),
        "Deposits (S$)": book_rm.aum_sgd.round(0),
        "Last contact (days)": book_rm.days_since_contact,
    })
    sel = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                       column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    if sel.selection.rows:
        ss["client_id"] = book_rm.client_id.iloc[sel.selection.rows[0]]
        st.success("Client selected. Open the Meeting prep tab.")

# ---------------------------------------------------------------- meeting prep
with tab_prep:
    ids = list(book_rm.client_id)
    default = ids.index(ss["client_id"]) if ss.get("client_id") in ids else 0
    label = dict(zip(book_rm.client_id, book_rm.name + " (" + book_rm.client_id + ")"))
    cid = st.selectbox("Client", ids, index=default, format_func=label.get)
    if cid not in ss["opened_at"]:
        ss["opened_at"][cid] = time.time()
        core.log_event({"event": "brief_opened", "rm_id": rm_id, "client_id": cid})

    summary = core.build_client_summary(book, cid)
    suggestions = core.suggest_products(summary)
    facts = core.context_for_llm(summary, suggestions, sheets)
    facts_by_id = {f["fact_id"]: f for f in facts}
    row = book_rm.set_index("client_id").loc[cid]
    p = summary["profile"]

    st.subheader(esc(f"{summary['name']}, {p['age']}, {p['occupation']} · risk profile {p['risk_profile']}"))
    st.markdown(esc(f"**Priority score {row.score:.2f}.** Why: " + ("; ".join(row.why) or "no strong signals")))

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Portfolio holdings**")
        h = summary["holdings"]
        st.dataframe(pd.DataFrame({
            "Fact id": "C-H-" + h.holding_id,
            "Product": h.product_name,
            "Balance (S$)": h.balance_sgd.round(0),
            "Detail": [(f"matures {m}" if m else "") + (f"lock-in ends {l} · {pk}" if l else "")
                       for m, l, pk in zip(h.maturity_date, h.lock_in_end, h.loan_rate_package)],
            "Source": h.source, "As of": h.as_of,
        }), hide_index=True, width="stretch")
        st.markdown("**Recent interactions**")
        for f in summary["facts"]:
            if f["field"] in ("interaction", "activity"):
                st.markdown(esc(f"- `{f['fact_id']}` {f['text']}"))
    with c2:
        st.markdown("**Signals (computed in code)**")
        for s in summary["signals"]:
            st.markdown(esc(f"- `{s['fact_id']}` {s['text']}"))
        if not summary["signals"]:
            st.caption("No signals.")
        st.caption(esc(f"Source: synthetic tables, snapshot {freshness(summary['as_of'])}"))

    st.markdown("**Suggested topics (rules engine)**")
    if not suggestions:
        st.caption("No product suggestions; consider a relationship check-in.")
    for s in suggestions:
        with st.expander(esc(f"{s['product_name']} ({s['type']}): " + "; ".join(s["reasons"]))):
            st.caption(esc("Evidence: " + ", ".join(s["client_fact_ids"])))
            st.dataframe(pd.DataFrame([{
                "Fact id": f["fact_id"], "Fact": f["text"].split(" | ", 1)[1], "Source": f["source"],
                "Pulled": freshness(f["as_of"]), "Stale?": f["stale"]} for f in facts if f["fact_id"] in s["product_fact_ids"]]),
                hide_index=True, width="stretch")

    st.divider()
    if st.button("Draft talking points", type="primary"):
        with st.spinner("Drafting from the facts above..."):
            brief = core.generate_brief(summary, suggestions, facts, use_ai=use_ai)
        brief["check"] = fc.fact_check_brief(brief["points"], facts)
        brief["brief_id"] = f"{cid}-{brief['generated_at']}"
        ss["briefs"][cid] = brief
        core.log_event({"event": "brief_generated", "rm_id": rm_id, "client_id": cid,
                        "brief_id": brief["brief_id"], "generator": brief["generator"]})

    brief = ss["briefs"].get(cid)
    if brief:
        summ = brief["check"]["summary"]
        st.markdown(esc(f"**Draft by {brief['generator']}** at {brief['generated_at']} · fact-check: "
                        f"{summ['pass']} pass, {summ['warn']} warn, {summ['fail']} fail · "
                        f"citation validity {summ['citation_validity']:.0%}"))
        if brief["error"]:
            st.warning(esc(f"Claude was unavailable, so template points are shown. ({brief['error']})"))
        for bi in summ["brief_issues"]:
            st.warning(esc(bi["detail"]))

        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        for i, (pt, res) in enumerate(zip(brief["points"], brief["check"]["points"])):
            with st.container(border=True):
                a, b = st.columns([1, 6])
                a.checkbox("Keep", value=res["verdict"] != "fail", key=f"keep-{brief['brief_id']}-{i}")
                a.caption(esc(f"{icon[res['verdict']]} {pt['section'].replace('_', ' ')}"))
                b.text_area("Point", pt["text"], key=f"text-{brief['brief_id']}-{i}", label_visibility="collapsed")
                for c in pt["citations"]:
                    f = facts_by_id.get(c)
                    b.caption(esc(f"[{c}] {f['text']} · {f['source']} · as of {freshness(f['as_of'])}"
                                  + (f" · STALE: {f['stale']}" if f["stale"] else "")) if f else esc(f"[{c}] not found"))
                for iss in res["issues"]:
                    b.markdown(esc(f"{icon[iss['severity']]} **{iss['check']}**: {iss['detail']}"))

        if st.button("Finalise my notes (I have reviewed them)"):
            final, kept = [], 0
            for i, pt in enumerate(brief["points"]):
                keep = ss[f"keep-{brief['brief_id']}-{i}"]
                text = ss[f"text-{brief['brief_id']}-{i}"]
                edited = text.strip() != pt["text"].strip()
                v = brief["check"]["points"][i]["verdict"]
                core.log_event({"event": "point_kept" if keep else "point_dismissed", "rm_id": rm_id, "client_id": cid,
                                "brief_id": brief["brief_id"], "generator": brief["generator"], "point_idx": i,
                                "section": pt["section"], "verdict": v, "edited": edited})
                if keep:
                    kept += 1
                    final.append({**pt, "text": text})
            # Re-check the RM's edited version so edits can't silently introduce errors.
            recheck = fc.fact_check_brief(final, facts)
            secs = round(time.time() - ss["opened_at"][cid])
            core.log_event({"event": "finalised", "rm_id": rm_id, "client_id": cid, "brief_id": brief["brief_id"],
                            "generator": brief["generator"], "value": secs})
            st.success(f"Kept {kept} of {len(brief['points'])} points. Prep time {secs // 60} min {secs % 60} s. "
                       "Nothing has been sent; use these notes as you see fit.")
            for r, pt in zip(recheck["points"], final):
                if r["verdict"] != "pass":
                    st.warning(esc(f"Your edited point still has issues: '{pt['text'][:80]}...' "
                                   + "; ".join(x["detail"] for x in r["issues"])))
            notes = "\n".join(f"- [{p['section']}] {p['text']} (sources: {', '.join(p['citations'])})" for p in final)
            notes += f"\n\nClient data snapshot {snapshot}; product facts pulled {pulled} (indicative)."
            st.code(notes, language=None)

        with st.form(f"outcome-{cid}"):
            st.markdown("**After the meeting** (optional, for measuring impact)")
            acted = st.radio("Did you act on any suggestion?", ["Yes", "No"], horizontal=True)
            revenue = st.number_input("Estimated revenue from this meeting (S\\$)", min_value=0.0, step=100.0)
            if st.form_submit_button("Log outcome"):
                core.log_event({"event": "outcome", "rm_id": rm_id, "client_id": cid, "brief_id": brief["brief_id"],
                                "generator": brief["generator"], "verdict": acted, "value": revenue})
                st.success("Outcome logged.")

# ---------------------------------------------------------------- fact-check & eval
with tab_check:
    st.subheader("How every talking point is checked")
    st.markdown(esc("""
- **has_citation / citation_exists**: every point cites fact ids, and they exist in the facts given to the drafter.
- **numbers_grounded / dates_grounded**: every number, S$ amount, % and date appears in the cited facts.
- **product_grounded / product_exists**: products mentioned are cited and exist in the OCBC product file.
- **indicative_caveat**: quoting a product figure requires an 'indicative / confirm' caveat (warn).
- **stale_fact**: citing a stale fact (SIBOR, 2016 returns) must be flagged as outdated (warn).
- **compliance_language**: no 'guaranteed', 'risk-free' and similar phrases.
- **no_action_on_behalf**: no claims that something was sent, booked or opened.
- **suitability_caution**: investment topics need a suitability caution for the RM (brief-level warn).
"""))
    rep = get_eval()
    a, b = st.columns(2)
    with a:
        st.markdown("**Checker self-test (seeded errors in clean briefs)**")
        t = rep["checker_self_test"]
        st.dataframe(pd.DataFrame(t["detection_rate_by_error"].items(), columns=["Seeded error", "Detection rate"]),
                     hide_index=True, width="stretch")
        st.caption(f"False-fail rate on {t['clean_points']} clean points: {t['clean_false_fail_rate']:.0%}")
    with b:
        st.markdown("**'Who to call' ranking (repeated 5-fold CV, synthetic labels)**")
        r = rep["ranking"]
        st.dataframe(pd.DataFrame({"AUC": r["auc"], "Precision @ top 20%": r["precision_at_top_20pct"]}),
                     width="stretch")
        st.caption(f"{r['n_clients']} clients, base take-up rate {r['base_rate']:.0%}. Labels are synthetic, "
                   "so this checks the pipeline, not real-world lift.")
        st.dataframe(pd.Series(r["standardised_coefficients"], name="coefficient"), width="stretch")
    st.caption("Run `python -m src.eval --llm 5` to also fact-check five Claude-drafted briefs end to end.")

# ---------------------------------------------------------------- metrics
with tab_metrics:
    st.subheader("Success metrics (from your actions in this app)")
    ev = core.load_events()
    if ev.empty:
        st.caption("No activity logged yet.")
    else:
        ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
        fin = ev[ev.event == "finalised"]
        kept = ev[ev.event.isin(["point_kept", "point_dismissed"])]
        outc = ev[ev.event == "outcome"]
        baseline = st.number_input("Assumed manual prep time per client (min)", value=30, min_value=1,
                                   help="ASSUMPTION: replace with a measured baseline from RM time studies.")
        m1, m2, m3, m4 = st.columns(4)
        med = fin.value.median() / 60 if len(fin) else float("nan")
        m1.metric("Prep time saved / meeting", f"{baseline - med:.1f} min" if len(fin) else "n/a",
                  help=f"Baseline minus median copilot prep time ({med:.1f} min).")
        m2.metric("RM adoption", f"{fin.rm_id.nunique()} of {len(book['rms'])} RMs",
                  help=f"{len(fin)} finalised briefs.")
        m3.metric("Suggestions kept", f"{(kept.event == 'point_kept').mean():.0%}" if len(kept) else "n/a",
                  help="Share of drafted points the RM kept (edited or not).")
        m4.metric("Revenue / meeting", f"SGD {outc.value.mean():,.0f}" if len(outc) else "n/a",
                  help=f"Acted on: {(outc.verdict == 'Yes').mean():.0%} of logged outcomes." if len(outc) else None)
        if len(kept):
            st.markdown("**Keep rate by fact-check verdict** (do RMs trust what passes?)")
            st.dataframe(kept.assign(kept=kept.event == "point_kept").groupby("verdict").kept.agg(["mean", "count"]),
                         width="stretch")
        st.dataframe(ev.tail(50), hide_index=True, width="stretch")
