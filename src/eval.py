"""Fact-check and evaluation layer for the RM copilot.

Three parts:
  1. fact_check_brief() - deterministic checks on every talking point
     (citations exist, numbers/dates/products are grounded in the cited facts,
     indicative-rate caveat, stale facts, compliance language, and no claims of
     acting on the RM's behalf).
  2. checker_self_test() - seeds known errors into clean briefs to measure
     that the checker actually catches them (and doesn't flag clean ones).
  3. ranking_eval() - held-out AUC / precision@k of the "who to call" model
     against a simple rules baseline and random.

Run: python -m src.eval            (deterministic, no API calls)
     python -m src.eval --llm 3    (also drafts 3 briefs with Claude and checks them)
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src import core

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ID_TOKEN = re.compile(r"\b(?:[A-Z]{1,5}-)*[A-Z]{1,5}\d{2,}\b|\b[CP]-[A-Z0-9-]+\b")
NUMBER = re.compile(r"(?<![\w.])(?:S?\$|US\$)?\d[\d,]*(?:\.\d+)?\s?(?:%|k\b|K\b|m\b|million\b)?")

# ASSUMPTION: a short, illustrative list of phrases a bank compliance team would
# not want in RM talking points (MAS fair-dealing spirit). Not an official list.
BANNED = re.compile(r"\bguarantee(?:d|s)?\b|\brisk[- ]free\b|\bno risk\b|\bcan(?:not|'t) lose\b|"
                    r"\bwill definitely\b|\bsure(?:-fire| to)\b|\bact now\b|\blimited time only\b", re.I)
ACTING = re.compile(r"\b(?:I|we)(?:'ve| have)? (?:already )?(?:sent|booked|transferred|placed|opened|submitted|"
                    r"applied|switched|renewed|executed)\b|\bhas been (?:booked|placed|opened|transferred|renewed|submitted)\b", re.I)
UNGROUNDED_TERMS = ["credit card", "insurance", "bond", "stock", "etf", "cashback", "gold", "crypto",
                    "structured", "annuity", "promotion"]
FAKE_PRODUCT = re.compile(r"\bOCBC\s+(?:[A-Z][\w+]*\s+){1,3}(?:Account|Card|Loan|Plan|Fund|Saver|Deposit)\b")
CAVEAT = re.compile(r"indicative|confirm|subject to change|may have changed", re.I)
STALE_ACK = re.compile(r"outdated|stale|retired|2016|no longer|old data", re.I)


def _to_float(tok: str) -> float | None:
    """Normalise '$70,000', 'S$1.2k', '0.2500', '1.2%' to a float."""
    t = tok.strip().lower().replace("us$", "").replace("s$", "").replace("$", "").replace(",", "").replace("%", "").strip()
    mult = 1.0
    if t.endswith("million"):
        t, mult = t[:-7].strip(), 1e6
    elif t.endswith("k"):
        t, mult = t[:-1].strip(), 1e3
    elif t.endswith("m"):
        t, mult = t[:-1].strip(), 1e6
    try:
        return round(float(t) * mult, 6)
    except ValueError:
        return None


def extract_numbers(text: str) -> set[float]:
    """Numbers in free text, ignoring ISO dates and record ids (C0012, H00405, P-TD-03)."""
    text = ID_TOKEN.sub(" ", ISO_DATE.sub(" ", text))
    out = set()
    for m in NUMBER.finditer(text):
        v = _to_float(m.group())
        if v is not None:
            out.add(v)
    return out


def known_products() -> list[str]:
    """Product names in the cleaned product file."""
    return [p["product_name"] for p in core.load_products()]


# --------------------------------------------------------------------------
# 1. Fact checker
# --------------------------------------------------------------------------


def check_point(point: dict, facts_by_id: dict, products: list[str]) -> dict:
    """Run every check on one talking point; return verdict + issues."""
    issues = []
    text = point["text"]
    cites = point.get("citations") or []

    def issue(check, severity, detail):
        issues.append({"check": check, "severity": severity, "detail": detail})

    if not cites:
        issue("has_citation", "fail", "No citation.")
    unknown = [c for c in cites if c not in facts_by_id]
    if unknown:
        issue("citation_exists", "fail", f"Cited ids not in the provided facts: {unknown}")
    cited = [facts_by_id[c] for c in cites if c in facts_by_id]
    cited_text = " ".join(f["text"] for f in cited)

    # Numbers must appear in cited facts. Small counting words like "1" in
    # "1-2 years" are still checked; that's deliberate (strict grounding).
    allowed = extract_numbers(cited_text) | {float(d[:4]) for d in ISO_DATE.findall(cited_text)}
    missing = sorted(v for v in extract_numbers(text) if v not in allowed)
    if missing:
        issue("numbers_grounded", "fail", f"Numbers not found in cited facts: {missing}")

    allowed_dates = set(ISO_DATE.findall(cited_text)) | {f["as_of"] for f in cited}
    bad_dates = [d for d in ISO_DATE.findall(text) if d not in allowed_dates]
    if bad_dates:
        issue("dates_grounded", "fail", f"Dates not found in cited facts: {bad_dates}")

    cited_subjects = {f["subject"] for f in cited} | {p for p in products if p.lower() in cited_text.lower()}
    for p in products:
        if p.lower() in text.lower() and p not in cited_subjects:
            issue("product_grounded", "fail", f"Mentions '{p}' without citing a fact about it.")
    for m in FAKE_PRODUCT.finditer(text):
        if not any(m.group().lower().replace("ocbc ", "") in p.lower() for p in products):
            issue("product_exists", "fail", f"Unknown product name: '{m.group()}'")
    for term in UNGROUNDED_TERMS:
        if re.search(rf"\b{term}s?\b", text, re.I) and not re.search(rf"\b{term}", cited_text, re.I):
            issue("term_grounded", "warn", f"Mentions '{term}' which is not in the cited facts.")

    product_nums = extract_numbers(" ".join(f["text"] for f in cited if f["kind"] == "product"))
    if point.get("section") != "caution" and product_nums & extract_numbers(text) and not CAVEAT.search(text):
        issue("indicative_caveat", "warn", "Quotes a product figure without saying it is indicative / to be confirmed.")

    stale = [f["fact_id"] for f in cited if f["stale"]]
    if stale and point.get("section") != "caution" and not STALE_ACK.search(text):
        issue("stale_fact", "warn", f"Relies on possibly outdated facts {stale} without flagging it.")

    if BANNED.search(text):
        issue("compliance_language", "fail", f"Prohibited phrasing: '{BANNED.search(text).group()}'")
    if ACTING.search(text):
        issue("no_action_on_behalf", "fail", f"Claims an action was taken: '{ACTING.search(text).group()}'")

    verdict = "fail" if any(i["severity"] == "fail" for i in issues) else ("warn" if issues else "pass")
    return {"verdict": verdict, "issues": issues}


def fact_check_brief(points: list[dict], facts: list[dict], products: list[str] | None = None) -> dict:
    """Check every point, plus brief-level rules; return per-point results and a summary."""
    products = products or known_products()
    by_id = {f["fact_id"]: f for f in facts}
    results = [check_point(p, by_id, products) for p in points]
    brief_issues = []
    mentions_invest = any(re.search(r"unit trust|invest", p["text"], re.I) for p in points)
    has_suitability = any(p.get("section") == "caution" and re.search(r"suitab|risk profile|assess", p["text"], re.I)
                          for p in points)
    if mentions_invest and not has_suitability:
        brief_issues.append({"check": "suitability_caution", "severity": "warn",
                             "detail": "Investment discussed but no suitability caution for the RM."})
    n = len(points) or 1
    all_cites = [c for p in points for c in (p.get("citations") or [])]
    summary = {
        "n_points": len(points),
        "pass": sum(r["verdict"] == "pass" for r in results),
        "warn": sum(r["verdict"] == "warn" for r in results),
        "fail": sum(r["verdict"] == "fail" for r in results),
        "pass_rate": round(sum(r["verdict"] == "pass" for r in results) / n, 3),
        "citation_validity": round(sum(c in by_id for c in all_cites) / max(len(all_cites), 1), 3),
        "brief_issues": brief_issues,
    }
    return {"points": results, "summary": summary}


# --------------------------------------------------------------------------
# 2. Checker self-test with seeded errors
# --------------------------------------------------------------------------


def _mutations(point: dict, facts_by_id: dict) -> dict[str, dict]:
    """Return one corrupted copy of `point` per error type."""
    m = {}
    p = _copy.deepcopy(point)
    p["text"] += " The rate is 4.85% for 12 months."
    m["invented_number"] = p
    p = _copy.deepcopy(point); p["citations"] = p["citations"] + ["P-FAKE-99"]
    m["fake_citation"] = p
    p = _copy.deepcopy(point); p["citations"] = []
    m["missing_citation"] = p
    p = _copy.deepcopy(point); p["text"] += " This gives guaranteed returns."
    m["banned_phrase"] = p
    p = _copy.deepcopy(point); p["text"] += " I have booked the placement for you."
    m["acting_on_behalf"] = p
    p = _copy.deepcopy(point); p["text"] += " You could also consider the OCBC Platinum Saver Account."
    m["invented_product"] = p
    p = _copy.deepcopy(point); p["text"] += " It matures on 2027-02-30."
    m["invented_date"] = p
    return m


def checker_self_test(book: dict, sheets: dict, n_clients: int = 15) -> dict:
    """Seed errors into clean template briefs and measure detection / false alarms."""
    products = known_products()
    ranked = core.rank_clients(book)
    clean_fail, clean_total = 0, 0
    caught: dict[str, list[int]] = {}
    for cid in ranked.client_id.head(n_clients):
        s = core.build_client_summary(book, cid)
        sug = core.suggest_products(s)
        facts = core.context_for_llm(s, sug, sheets)
        by_id = {f["fact_id"]: f for f in facts}
        brief = core.template_points(s, sug, facts)
        for pt in brief:
            r = check_point(pt, by_id, products)
            clean_total += 1
            clean_fail += r["verdict"] == "fail"
            for kind, bad in _mutations(pt, by_id).items():
                caught.setdefault(kind, []).append(int(check_point(bad, by_id, products)["verdict"] == "fail"))
    return {
        "clean_points": clean_total,
        "clean_false_fail_rate": round(clean_fail / max(clean_total, 1), 3),
        "detection_rate_by_error": {k: round(float(np.mean(v)), 3) for k, v in caught.items()},
    }


# --------------------------------------------------------------------------
# 3. Ranking evaluation
# --------------------------------------------------------------------------


def ranking_eval(book: dict, n_repeats: int = 5, top_frac: float = 0.2) -> dict:
    """Repeated 5-fold CV: AUC and precision@top-k of the priority model vs baselines.

    Labels are SYNTHETIC (see data_gen.make_campaign_history), so this shows the
    pipeline works, not real-world lift.
    """
    as_of = pd.Timestamp(book["clients"].as_of.iloc[0]).date()
    feats = core.compute_features(book["clients"], book["holdings"], book["activity"], book["interactions"], as_of)
    y = feats[["client_id"]].merge(book["campaign_outcomes"], on="client_id").took_up_product.to_numpy()
    # Rules baseline: count of active signals (what an RM might do by hand).
    rules = (feats[["td_maturing_60d", "lock_in_ending_120d", "large_inflow_30d", "usd_without_fx_account",
                    "no_investment_growth_profile"]].sum(axis=1) + (feats.idle_cash_sgd > 50_000)).to_numpy(float)
    scores = {"model": [], "rules": [], "random": []}
    precs = {"model": [], "rules": [], "random": []}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    rng = np.random.default_rng(0)
    for tr, te in cv.split(feats, y):
        model = core.train_priority_model(feats.iloc[tr], book["campaign_outcomes"])
        preds = {"model": model.predict_proba(core.model_matrix(feats.iloc[te]))[:, 1],
                 "rules": rules[te] + 1e-6 * rng.random(len(te)),  # random tie-break
                 "random": rng.random(len(te))}
        k = max(1, int(len(te) * top_frac))
        for name, p in preds.items():
            scores[name].append(roc_auc_score(y[te], p))
            precs[name].append(y[te][np.argsort(-p)[:k]].mean())

    def ms(v):
        return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

    full = core.train_priority_model(feats, book["campaign_outcomes"])
    return {
        "n_clients": int(len(y)), "base_rate": round(float(y.mean()), 3), "top_frac": top_frac,
        "auc": {k: ms(v) for k, v in scores.items()},
        "precision_at_top_20pct": {k: ms(v) for k, v in precs.items()},
        "standardised_coefficients": dict(zip(core.MODEL_FEATURES, np.round(full[-1].coef_[0], 3).tolist())),
    }


# --------------------------------------------------------------------------
# Optional: end-to-end check of Claude-drafted briefs
# --------------------------------------------------------------------------


def llm_brief_eval(book: dict, sheets: dict, n: int = 3) -> dict:
    """Draft briefs for the top-n clients with Claude and fact-check them."""
    ranked = core.rank_clients(book)
    rows = []
    for cid in ranked.client_id.head(n):
        s = core.build_client_summary(book, cid)
        sug = core.suggest_products(s)
        facts = core.context_for_llm(s, sug, sheets)
        t0 = time.time()
        brief = core.generate_brief(s, sug, facts, use_ai=True)
        fc = fact_check_brief(brief["points"], facts)
        rows.append({"client_id": cid, "generator": brief["generator"], "error": brief["error"],
                     "latency_s": round(time.time() - t0, 1), **{k: v for k, v in fc["summary"].items() if k != "brief_issues"},
                     "issues": [i for r in fc["points"] for i in r["issues"]] + fc["summary"]["brief_issues"],
                     "points": brief["points"]})
    df = pd.DataFrame(rows)
    return {"briefs": rows, "model": core.MODEL_ID,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "fallback_rate": round(float((df.generator != core.MODEL_ID).mean()), 3),
            "mean_pass_rate": round(float(df.pass_rate.mean()), 3),
            "mean_citation_validity": round(float(df.citation_validity.mean()), 3),
            "total_fail_points": int(df.fail.sum())}


def write_summary(report: dict, path: Path = EVAL_DIR / "SUMMARY.md") -> None:
    """Write a human-readable summary of the eval report (committed to the repo)."""
    r, t = report["ranking"], report["checker_self_test"]
    lines = ["# Evaluation summary: RM copilot", "",
             "Generated by `python -m src.eval`. All client data is synthetic; product facts are OCBC public data "
             "(indicative, possibly outdated).", "",
             "## Fact-checker self-test", "",
             f"Seeded errors were injected into {t['clean_points']} clean template points. "
             f"False-fail rate on the clean points: {t['clean_false_fail_rate']:.0%}.", "",
             "| Seeded error | Detection rate |", "|---|---|"]
    lines += [f"| {k} | {v:.0%} |" for k, v in t["detection_rate_by_error"].items()]
    lines += ["", "## 'Who to call' ranking", "",
              f"Repeated 5-fold CV (5 repeats) on {r['n_clients']} clients, base take-up rate {r['base_rate']:.0%}. "
              "Rules baseline = count of active signals.", "",
              "| Strategy | AUC | Precision @ top 20% |", "|---|---|---|"]
    lines += [f"| {k} | {r['auc'][k]} | {r['precision_at_top_20pct'][k]} |" for k in r["auc"]]
    lines += ["", "**Read with care:** labels are synthetic and come from a rule we wrote, so this checks the "
              "pipeline, not real-world lift. The model is only marginally ahead of the rules baseline, within noise."]
    lb = report.get("llm_briefs")
    if lb:
        lines += ["", "## Claude-drafted briefs, fact-checked end to end", "",
                  f"Run at {lb.get('generated_at', 'unknown')} with `{lb.get('model', core.MODEL_ID)}`.", "",
                  "| Client | Generator | Latency (s) | Points | Pass | Warn | Fail | Citation validity |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {b['client_id']} | {b['generator']} | {b['latency_s']} | {b['n_points']} | {b['pass']} | "
                  f"{b['warn']} | {b['fail']} | {b['citation_validity']:.0%} |" for b in lb["briefs"]]
        lines += ["", f"Fallback rate {lb['fallback_rate']:.0%}; mean pass rate {lb['mean_pass_rate']:.0%}. "
                  "Full points and issues are in `report.json`."]
    path.write_text("\n".join(lines) + "\n")


def run_all(n_llm: int = 0, save: bool = True) -> dict:
    """Run the full evaluation; optionally save report.json and SUMMARY.md under data/eval/.

    Without --llm, the last saved Claude brief results are kept rather than dropped.
    """
    book = core.load_book()
    sheets = core.build_fact_sheets()
    report = {"ranking": ranking_eval(book), "checker_self_test": checker_self_test(book, sheets)}
    saved = EVAL_DIR / "report.json"
    if n_llm:
        report["llm_briefs"] = llm_brief_eval(book, sheets, n_llm)
    elif saved.exists():
        prev = json.loads(saved.read_text()).get("llm_briefs")
        if prev:
            report["llm_briefs"] = prev
    if save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(report, indent=2, default=str))
        write_summary(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="number of Claude-drafted briefs to fact-check")
    args = ap.parse_args()
    rep = run_all(args.llm)
    print(json.dumps(rep["ranking"], indent=1))
    print(json.dumps(rep["checker_self_test"], indent=1))
    if "llm_briefs" in rep:
        lb = rep["llm_briefs"]
        print({k: v for k, v in lb.items() if k != "briefs"})
        for b in lb["briefs"]:
            print(f"\n== {b['client_id']} via {b['generator']} ({b['latency_s']}s) pass={b['pass_rate']} err={b['error']}")
            for p in b["points"]:
                print(f"  [{p['section']}] {p['text']}  {p['citations']}")
            for i in b["issues"]:
                print("   !", i)
