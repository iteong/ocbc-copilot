"""Business logic for the RM copilot.

Pipeline:
  1. Product fact sheets  - real OCBC public product data, split into small
     citable facts (id, text, source_file, pulled_on).
  2. Client summary       - computed in code from SYNTHETIC tables; every line
     is a citable fact with its source table and as_of date.
  3. Priority ("who to call") - logistic regression with per-feature
     contributions so the RM sees WHY a client is ranked high.
  4. Product suggestions  - explicit rules, each with reasons + fact ids.
  5. Talking points       - Claude drafts from the facts only, citing fact ids;
     a template generator is the non-AI fallback.

The copilot only drafts. It never contacts clients or books anything; the RM
reviews, edits and decides what (if anything) to use.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS_PATH = ROOT / "data" / "products_clean" / "products.json"
SYNTH_DIR = ROOT / "data" / "synthetic"
FEEDBACK_PATH = ROOT / "data" / "feedback" / "events.csv"

MODEL_ID = "claude-opus-5-5"
LLM_TIMEOUT_S = 60.0

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def money(x: float) -> str:
    """Format a S$ amount the same way everywhere (the fact-checker relies on it)."""
    return f"S${abs(x):,.0f}"


def age_days(as_of: str, today: date | None = None) -> int:
    """Days between an ISO date string and today."""
    today = today or date.today()
    return (today - date.fromisoformat(str(as_of)[:10])).days


def fact(fact_id: str, text: str, source: str, as_of: str, kind: str, subject: str,
         field: str = "", stale: str = "") -> dict:
    """Build one citable fact. `stale` explains why the fact may be outdated."""
    return {"fact_id": fact_id, "text": text, "source": source, "as_of": str(as_of)[:10],
            "kind": kind, "subject": subject, "field": field, "stale": stale}


# ---------------------------------------------------------------------------
# 1. Product fact sheets
# ---------------------------------------------------------------------------

PRODUCT_CODES = {
    "360 Account": "360", "FRANK Account": "FRANK", "Time Deposit (Fixed Deposit)": "TD",
    "USD Current Account": "USDCA", "Global Savings Account": "GSA",
    "Home Loan (New Purchase)": "HLNP", "Home Loan (Refinancing)": "HLRF",
    "Bonus+ Savings Account": "BONUS", "Monthly Savings Account": "MSA", "Unit Trusts": "UT",
}
FACT_FIELDS = ["description", "eligibility", "initial_deposit", "minimum_balance",
               "fees_and_charges", "benefits", "interest_rates", "loan_amount",
               "overview", "how_it_works", "key_risks"]
FIELD_LABELS = {f: f.replace("_", " ").capitalize() for f in FACT_FIELDS + ["remarks"]}

# Known staleness issues from data/products_clean/DATA_QUALITY.md.
STALE_PATTERNS = [
    (re.compile(r"SIBOR", re.I), "References SIBOR, which has been retired in Singapore."),
    (re.compile(r"March 2016|since 31 August 2012", re.I), "Back-tested performance figure dated March 2016."),
]


def load_products() -> list[dict]:
    """Load the cleaned OCBC product records (indicative, possibly outdated)."""
    with open(PRODUCTS_PATH) as f:
        return json.load(f)


def split_bullets(text: str) -> list[str]:
    """Turn a nested '- bullet' field into leaf lines that keep their headings.

    e.g. 'Singapore Dollar > $5,000 - S$20,000 > 12 - 15 mths:0.2500'
    """
    leaves, root, stack = [], "", {}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, raw in enumerate(lines):
        indent = len(raw) - len(raw.lstrip(" "))
        body = raw.strip()
        bulleted = body.startswith("- ")
        body = body[2:].strip() if bulleted else body
        level = indent // 2 if bulleted else -1
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt_indent = len(nxt) - len(nxt.lstrip(" "))
        nxt_bulleted = nxt.strip().startswith("- ")
        has_children = bool(nxt) and nxt_bulleted and (
            nxt_indent > indent or (not bulleted and nxt_indent == indent))
        # A short unpunctuated line followed by plain text is a sub-heading too.
        short_heading = (not bulleted and bool(nxt) and not nxt_bulleted and len(body) < 50
                         and not body.endswith((".", "!", "?")))
        if has_children or short_heading:
            if level < 0:
                root, stack = body.rstrip(":"), {}
            else:
                stack = {k: v for k, v in stack.items() if k < level}
                stack[level] = body.rstrip(":")
            continue
        if level < 0:
            root, stack = "", {}
        context = ([root] if root and level >= 0 else []) + [stack[k] for k in sorted(stack) if k < level]
        leaves.append(" > ".join(context + [body]))
    return leaves


def build_fact_sheets(products: list[dict] | None = None) -> dict[str, list[dict]]:
    """Return {product_name: [fact, ...]} built from the cleaned product file."""
    products = products or load_products()
    sheets = {}
    for p in products:
        code = PRODUCT_CODES.get(p["product_name"], re.sub(r"\W", "", p["product_name"])[:6].upper())
        fields = FACT_FIELDS + (["remarks"] if p["category"] == "Deposit Accounts" else [])
        facts, n = [], 0
        for field in fields:
            for leaf in split_bullets(p.get(field) or ""):
                n += 1
                stale = next((why for rx, why in STALE_PATTERNS if rx.search(leaf)), "")
                facts.append(fact(
                    f"P-{code}-{n:02d}", f"{p['product_name']} | {FIELD_LABELS[field]}: {leaf}",
                    source=f"{p['source_file']} (OCBC public API, indicative)", as_of=p["pulled_on"],
                    kind="product", subject=p["product_name"], field=field, stale=stale))
        if not (p.get("key_risks") or "").strip():
            n += 1
            facts.append(fact(
                f"P-{code}-{n:02d}",
                f"{p['product_name']} | Data gap: the public data has no key risks listed for this product.",
                source=f"{p['source_file']} (OCBC public API, indicative)", as_of=p["pulled_on"],
                kind="product", subject=p["product_name"], field="data_gap"))
        sheets[p["product_name"]] = facts
    return sheets


def _tier_bounds(label: str) -> tuple[float, float] | None:
    """Parse a Time Deposit tier label like '>$20,000 - S$50,000' into (low, high)."""
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", label)]
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def select_product_facts(sheet: list[dict], amount: float | None = None, max_facts: int = 22) -> list[dict]:
    """Pick the facts worth sending to the LLM for one product.

    For the Time Deposit, keep only the S$ rate tier that matches `amount`.
    """
    out = []
    for f in sheet:
        if f["field"] == "benefits" and "Singapore Dollar >" in f["text"]:
            tier = f["text"].split("Singapore Dollar > ", 1)[1].split(" > ")[0]
            b = _tier_bounds(tier)
            if amount is None or b is None or not (b[0] <= amount <= b[1]):
                continue
        out.append(f)
    return out[:max_facts] if amount is None else out


# ---------------------------------------------------------------------------
# 2. Synthetic book + features + client summary
# ---------------------------------------------------------------------------


def load_book(regenerate: bool = False) -> dict[str, pd.DataFrame]:
    """Load the SYNTHETIC tables, generating them on first use."""
    names = ["rms", "clients", "holdings", "activity", "interactions", "campaign_outcomes"]
    if regenerate or not all((SYNTH_DIR / f"{n}.csv").exists() for n in names):
        from src.data_gen import generate
        generate()
    book = {n: pd.read_csv(SYNTH_DIR / f"{n}.csv", keep_default_na=False,
                           na_values={"tenor_months": [""]}) for n in names}
    return book


# ASSUMPTION: "idle cash" = balances in low-yield accounts plus any 360 Account
# balance above the S$70,000 bonus cap (from the 360 fact sheet). A real bank
# would use its own definition and live rates.
LOW_YIELD = {"FRANK Account", "Bonus+ Savings Account", "Monthly Savings Account"}
CAP_360 = 70_000


def compute_features(clients, holdings, activity, interactions, as_of: date) -> pd.DataFrame:
    """One row per client with the signals used for ranking and suggestions."""
    h = holdings.copy()
    h["days_to_maturity"] = h["maturity_date"].map(lambda d: (date.fromisoformat(d) - as_of).days if d else np.nan)
    h["days_to_lock_end"] = h["lock_in_end"].map(lambda d: (date.fromisoformat(d) - as_of).days if d else np.nan)
    h["idle"] = np.where(h.product_name.isin(LOW_YIELD), h.balance_sgd,
                         np.where(h.product_name == "360 Account", (h.balance_sgd - CAP_360).clip(lower=0), 0))
    g = h.groupby("client_id")
    agg = pd.DataFrame({
        "aum_sgd": g.balance_sgd.apply(lambda s: s[s > 0].sum()),
        "loan_sgd": g.balance_sgd.apply(lambda s: -s[s < 0].sum()),
        "idle_cash_sgd": g.idle.sum(),
        "td_maturing_60d": g.days_to_maturity.apply(lambda s: int(((s >= 0) & (s <= 60)).any())),
        "lock_in_ending_120d": g.days_to_lock_end.apply(lambda s: int(((s >= -90) & (s <= 120)).any())),
        "has_investment": g.product_name.apply(lambda s: int("Unit Trusts" in set(s))),
        "has_fx_account": g.product_name.apply(lambda s: int(bool({"Global Savings Account", "USD Current Account"} & set(s)))),
        "n_products": g.product_name.nunique(),
    }).reset_index()
    last = interactions.groupby("client_id")["date"].max().map(lambda d: (as_of - date.fromisoformat(d)).days)
    f = clients[["client_id", "rm_id", "age", "risk_profile"]].merge(agg, on="client_id", how="left")
    f = f.merge(activity[["client_id", "inflow_30d_sgd", "usd_inflow_30d", "salary_credited_30d"]], on="client_id", how="left")
    f["days_since_contact"] = f.client_id.map(last).fillna(999).astype(int)
    f["large_inflow_30d"] = (f.inflow_30d_sgd > 50_000).astype(int)
    f["usd_without_fx_account"] = ((f.usd_inflow_30d > 0) & (f.has_fx_account == 0)).astype(int)
    f["no_investment_growth_profile"] = ((f.has_investment == 0) & (f.risk_profile != "Conservative")).astype(int)
    return f.fillna(0)


# Features the priority model uses, with plain-English templates for reasons.
MODEL_FEATURES = {
    "td_maturing_60d": "Time deposit matures within 60 days",
    "lock_in_ending_120d": "Home loan lock-in ends (or ended) within ~4 months",
    "idle_cash_over_50k": "Idle cash of {idle}",
    "large_inflow_30d": "Large inflow of {inflow} in last 30 days",
    "log_days_since_contact": "Last contacted {days} days ago",
    "no_investment_growth_profile": "No investments despite {risk} risk profile",
    "usd_without_fx_account": "USD inflows but no foreign-currency account",
}


def model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Transform raw features into the model's input columns."""
    X = features[[c for c in MODEL_FEATURES if c in features.columns]].astype(float).copy()
    X["idle_cash_over_50k"] = (features.idle_cash_sgd > 50_000).astype(float)
    X["log_days_since_contact"] = np.log1p(features.days_since_contact)
    return X[list(MODEL_FEATURES)]


def train_priority_model(features: pd.DataFrame, outcomes: pd.DataFrame):
    """Fit a standardised logistic regression on past (synthetic) outreach outcomes."""
    y = features[["client_id"]].merge(outcomes, on="client_id").took_up_product
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(model_matrix(features), y)
    return model


def explain_scores(model, features: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Score clients and attach the top positive feature contributions as reasons.

    Contribution = coefficient x standardised value (log-odds vs an average client).
    """
    X = model_matrix(features)
    scaler, lr = model[0], model[-1]
    contrib = scaler.transform(X) * lr.coef_[0]
    out = features[["client_id", "rm_id"]].copy()
    out["score"] = model.predict_proba(X)[:, 1]
    reasons = []
    for i, row in enumerate(features.itertuples()):
        order = np.argsort(-contrib[i])
        texts = []
        for j in order[:top_k]:
            if contrib[i, j] <= 0.05:
                break
            texts.append(MODEL_FEATURES[X.columns[j]].format(
                idle=money(row.idle_cash_sgd), inflow=money(row.inflow_30d_sgd),
                days=row.days_since_contact, risk=row.risk_profile) + f" (+{contrib[i, j]:.2f})")
        reasons.append(texts)
    out["why"] = reasons
    out["contributions"] = [dict(zip(X.columns, np.round(c, 3))) for c in contrib]
    return out.sort_values("score", ascending=False)


def rank_clients(book: dict, as_of: date | None = None, model=None) -> pd.DataFrame:
    """Rank every client by modelled take-up propensity with reasons."""
    as_of = as_of or date.fromisoformat(book["clients"].as_of.iloc[0])
    feats = compute_features(book["clients"], book["holdings"], book["activity"], book["interactions"], as_of)
    model = model or train_priority_model(feats, book["campaign_outcomes"])
    ranked = explain_scores(model, feats)
    return ranked.merge(book["clients"][["client_id", "name", "age", "risk_profile"]], on="client_id") \
                 .merge(feats.drop(columns=["rm_id", "age", "risk_profile"]), on="client_id")


def build_client_summary(book: dict, client_id: str) -> dict:
    """Compute a client summary where every line is a citable, dated fact."""
    c = book["clients"].set_index("client_id").loc[client_id]
    as_of = date.fromisoformat(c.as_of)
    hold = book["holdings"][book["holdings"].client_id == client_id]
    act = book["activity"][book["activity"].client_id == client_id].iloc[0]
    ints = book["interactions"][book["interactions"].client_id == client_id].sort_values("date", ascending=False)
    feats = compute_features(book["clients"][book["clients"].client_id == client_id], hold,
                             book["activity"][book["activity"].client_id == client_id], ints, as_of).iloc[0]
    facts = [
        fact("C-PRO-1", f"Client {c['name']} ({client_id}), age {c.age}, {c.residency}, {c.occupation}.",
             c.source, c.as_of, "client", client_id, "profile"),
        fact("C-PRO-2", f"Risk profile on file: {c.risk_profile}. Client since {c.client_since}.",
             c.source, c.as_of, "client", client_id, "profile"),
    ]
    for h in hold.itertuples():
        if h.balance_sgd < 0:
            txt = (f"Holds {h.product_name} ({h.holding_id}): outstanding {money(h.balance_sgd)}, "
                   f"rate package '{h.loan_rate_package}', lock-in ends {h.lock_in_end}.")
        else:
            txt = f"Holds {h.product_name} ({h.holding_id}): balance {money(h.balance_sgd)} (S$ equivalent), opened {h.opened_on}."
            if h.maturity_date:
                txt += f" {int(h.tenor_months)}-month deposit maturing {h.maturity_date}."
        facts.append(fact(f"C-H-{h.holding_id}", txt, h.source, h.as_of, "client", client_id, "holding"))
    facts.append(fact(
        "C-ACT-1",
        f"Last 30 days (from {act.window_start}): inflows {money(act.inflow_30d_sgd)}, "
        f"salary credited: {'yes, ' + money(act.salary_amount_sgd) if act.salary_credited_30d else 'no'}, "
        f"{act.bills_paid_30d} bills paid, card spend {money(act.card_spend_30d_sgd)}"
        + (f", USD inflows {money(act.usd_inflow_30d)} (S$ equivalent)" if act.usd_inflow_30d else "") + ".",
        act.source, act.as_of, "client", client_id, "activity"))
    for it in ints.head(3).itertuples():
        facts.append(fact(f"C-INT-{it.interaction_id}", f"{it.date} {it.channel}: {it.note}",
                          it.source, it.as_of, "client", client_id, "interaction"))

    signals = derive_signals(feats, hold, act, as_of)
    for i, s in enumerate(signals, 1):
        facts.append(fact(f"C-SIG-{i}", s["text"], "computed in src/core.py from " + s["from"], c.as_of,
                          "client", client_id, "signal"))
        s["fact_id"] = f"C-SIG-{i}"
    return {"client_id": client_id, "name": c["name"], "rm_id": c.rm_id, "as_of": c.as_of,
            "profile": c.to_dict(), "holdings": hold, "features": feats.to_dict(),
            "signals": signals, "facts": facts}


def derive_signals(feats: pd.Series, hold: pd.DataFrame, act: pd.Series, as_of: date) -> list[dict]:
    """Rule-based signals; each has a code, explanation and source table."""
    sig = []
    src_h, src_a = "synthetic_core_banking.holdings", "synthetic_core_banking.activity_30d"
    for h in hold.itertuples():
        if h.maturity_date:
            d = (date.fromisoformat(h.maturity_date) - as_of).days
            if 0 <= d <= 60:
                sig.append({"code": "td_maturing", "holding_id": h.holding_id, "amount": h.balance_sgd, "from": src_h,
                            "text": f"Time deposit {h.holding_id} of {money(h.balance_sgd)} matures on {h.maturity_date}, {d} days after the {as_of} snapshot."})
        if h.lock_in_end:
            d = (date.fromisoformat(h.lock_in_end) - as_of).days
            if -90 <= d <= 120:
                when = f"ends in {d} days" if d >= 0 else f"ended {-d} days ago"
                sig.append({"code": "lock_in_ending", "holding_id": h.holding_id, "amount": -h.balance_sgd, "from": src_h,
                            "text": f"Home loan {h.holding_id} lock-in {when} ({h.lock_in_end}); outstanding {money(h.balance_sgd)}."})
        if h.product_name == "360 Account" and h.balance_sgd > CAP_360:
            sig.append({"code": "above_360_cap", "holding_id": h.holding_id, "amount": h.balance_sgd - CAP_360, "from": src_h,
                        "text": f"360 Account balance {money(h.balance_sgd)} is {money(h.balance_sgd - CAP_360)} above the S$70,000 bonus-interest cap."})
        if h.product_name == "360 Account" and not act.salary_credited_30d:
            sig.append({"code": "360_no_salary", "holding_id": h.holding_id, "amount": 0, "from": src_a,
                        "text": "Holds a 360 Account but no salary credit was seen in the last 30 days."})
        if h.product_name == "FRANK Account" and int(feats.age) >= 26 and h.balance_sgd < 1000:
            sig.append({"code": "frank_fee_risk", "holding_id": h.holding_id, "amount": h.balance_sgd, "from": src_h,
                        "text": f"Age {int(feats.age)} with FRANK Account balance {money(h.balance_sgd)}, below S$1,000."})
    if feats.idle_cash_sgd > 50_000:
        sig.append({"code": "idle_cash", "amount": feats.idle_cash_sgd, "from": src_h,
                    "text": f"Idle cash in low-yield balances: {money(feats.idle_cash_sgd)}."})
    if feats.large_inflow_30d:
        sig.append({"code": "large_inflow", "amount": feats.inflow_30d_sgd, "from": src_a,
                    "text": f"Large inflow: {money(feats.inflow_30d_sgd)} received in the last 30 days."})
    if feats.usd_without_fx_account:
        sig.append({"code": "usd_no_fx", "amount": feats.usd_inflow_30d, "from": src_a,
                    "text": "USD inflows in the last 30 days but no foreign-currency account held."})
    if feats.no_investment_growth_profile and feats.aum_sgd > 30_000:
        sig.append({"code": "no_investment", "amount": feats.aum_sgd, "from": src_h,
                    "text": f"No investment holdings; risk profile {feats.risk_profile}; deposits {money(feats.aum_sgd)}."})
    if feats.days_since_contact > 180:
        sig.append({"code": "not_contacted", "amount": 0, "from": "synthetic_crm.interaction_log",
                    "text": f"Not contacted for {int(feats.days_since_contact)} days."})
    return sig


# ---------------------------------------------------------------------------
# 4. Product suggestions (rules)
# ---------------------------------------------------------------------------

# Rule table: signal code -> (product, why). Explicit so the RM can see the logic.
# ASSUMPTION: these mappings are illustrative sales logic, not OCBC policy.
RULES = {
    "td_maturing": [("Time Deposit (Fixed Deposit)", "Maturing deposit: discuss renewal options."),
                    ("Unit Trusts", "Maturing deposit: if the client wants growth and suitability is confirmed, discuss diversifying.")],
    "lock_in_ending": [("Home Loan (Refinancing)", "Lock-in ending: review the mortgage package.")],
    "above_360_cap": [("Time Deposit (Fixed Deposit)", "Balance above the 360 bonus cap earns only base interest on the excess.")],
    "idle_cash": [("360 Account", "Idle cash in low-yield balances."),
                  ("Time Deposit (Fixed Deposit)", "Idle cash the client may not need immediately.")],
    "large_inflow": [("Time Deposit (Fixed Deposit)", "Recent large inflow, possibly fresh funds.")],
    "usd_no_fx": [("Global Savings Account", "USD inflows without a foreign-currency account.")],
    "no_investment": [("Unit Trusts", "No investments despite a non-conservative risk profile.")],
    "360_no_salary": [("360 Account", "Existing 360 Account is missing the salary-credit bonus.")],
    "frank_fee_risk": [("FRANK Account", "Possible monthly service fee: client is 26+ with a low balance.")],
}
NOTE_RULES = [(re.compile(r"child's education", re.I), "Monthly Savings Account", "Client mentioned saving for a child's education."),
              (re.compile(r"mortgage", re.I), "Home Loan (Refinancing)", "Client mentioned reviewing their mortgage.")]
NOT_FOR_CONSERVATIVE = {"Unit Trusts"}


def suggest_products(summary: dict, max_n: int = 3) -> list[dict]:
    """Map signals and notes to candidate products, with reasons and evidence ids."""
    held = set(summary["holdings"].product_name)
    risk = summary["profile"]["risk_profile"]
    age = int(summary["profile"]["age"])
    out: dict[str, dict] = {}

    def add(product, why, fid, amount=None):
        if product in NOT_FOR_CONSERVATIVE and risk == "Conservative":
            return
        if age < 18 and product not in {"FRANK Account", "Monthly Savings Account"}:
            return
        existing = product in held and product not in {"Time Deposit (Fixed Deposit)"}
        s = out.setdefault(product, {"product_name": product, "reasons": [], "client_fact_ids": [],
                                     "type": "review existing" if existing else "new", "amount": amount})
        if why not in s["reasons"]:
            s["reasons"].append(why)
            s["client_fact_ids"].append(fid)
        if amount and not s["amount"]:
            s["amount"] = amount

    for sig in summary["signals"]:
        for product, why in RULES.get(sig["code"], []):
            if sig["code"] in ("idle_cash",) and product == "360 Account" and "360 Account" in held:
                continue
            add(product, why, sig["fact_id"], sig.get("amount"))
    for f in summary["facts"]:
        if f["field"] == "interaction":
            for rx, product, why in NOTE_RULES:
                if rx.search(f["text"]) and (product not in held):
                    if product.startswith("Home Loan") and not any(summary["holdings"].product_name.str.startswith("Home Loan")):
                        continue
                    add(product, why, f["fact_id"])
    ranked = sorted(out.values(), key=lambda s: -len(s["reasons"]))
    return ranked[:max_n]


def context_for_llm(summary: dict, suggestions: list[dict], sheets: dict) -> list[dict]:
    """All facts the drafter may use: client facts + selected product facts."""
    facts = list(summary["facts"])
    for s in suggestions:
        amt = s.get("amount") if s["product_name"].startswith("Time Deposit") else None
        chosen = select_product_facts(sheets[s["product_name"]], amount=amt)
        s["product_fact_ids"] = [f["fact_id"] for f in chosen]
        facts += chosen
    return facts


# ---------------------------------------------------------------------------
# 5. Talking points: Claude with a template fallback
# ---------------------------------------------------------------------------


class TalkingPoint(BaseModel):
    """One line of the meeting brief."""
    section: Literal["opener", "talking_point", "question", "caution"]
    text: str = Field(description="One or two sentences the RM could say or should keep in mind.")
    citations: list[str] = Field(description="fact_id values from the provided facts that support every claim.")


class MeetingBrief(BaseModel):
    """Structured output the LLM must return."""
    points: list[TalkingPoint]


SYSTEM_PROMPT = """You draft meeting-prep notes for a bank relationship manager (RM) in Singapore.
The RM reviews everything and decides what to say; you never contact the client and you never claim an action was taken.

Use ONLY the facts provided. Each fact has a fact_id, text, source and as_of date.
Rules:
- Every point must cite the fact_id(s) that support each claim. Never cite an id that is not in the list.
- Copy numbers, rates, amounts and dates exactly as written in the cited facts (same figures; don't round or compute new ones).
- Product facts come from OCBC public product data that is indicative and may be outdated. When a point mentions a rate, fee or threshold, say it is "indicative" and that the RM should confirm current terms.
- Do not mention any product, promotion or feature that is not in the facts.
- No promises of returns, no "guaranteed", "risk-free" or pressure language. For investment products, add a caution that suitability must be assessed first.
- If a fact is marked stale, either avoid it or add a caution that it is outdated.
- Write 1 opener, 3-5 talking points, 2-3 questions to ask the client, and 1-3 cautions for the RM.
- Refer to the client by name or as "the client"; do not infer gender or other traits from the name.
- Plain, concise English the RM could say out loud."""


def _facts_block(facts: list[dict]) -> str:
    """Serialise facts for the prompt (stable ordering helps caching)."""
    lines = []
    for f in facts:
        stale = f" [STALE: {f['stale']}]" if f["stale"] else ""
        lines.append(f"{f['fact_id']} | as_of {f['as_of']} | {f['source']} | {f['text']}{stale}")
    return "\n".join(lines)


def _llm_points(summary: dict, suggestions: list[dict], facts: list[dict]) -> list[dict]:
    """Call Claude for structured talking points. Raises on any failure."""
    import anthropic
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_S, max_retries=1)
    sugg = "\n".join(f"- {s['product_name']} ({s['type']}): " + "; ".join(s["reasons"]) +
                     f" [evidence: {', '.join(s['client_fact_ids'])}]" for s in suggestions) or "- none"
    user = (f"Client: {summary['name']} ({summary['client_id']}). Data snapshot {summary['as_of']}.\n\n"
            f"Suggested topics from the rules engine:\n{sugg}\n\nFACTS:\n{_facts_block(facts)}")
    resp = client.beta.messages.parse(
        model=MODEL_ID,
        max_tokens=8000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_format=MeetingBrief,
    )
    if resp.stop_reason != "end_turn" or resp.parsed_output is None:
        raise RuntimeError(f"LLM stopped with {resp.stop_reason}")
    return [p.model_dump() for p in resp.parsed_output.points]


def template_points(summary: dict, suggestions: list[dict], facts: list[dict]) -> list[dict]:
    """Deterministic, non-AI talking points built only from the facts."""
    by_id = {f["fact_id"]: f for f in facts}
    pts = [{"section": "opener", "citations": ["C-PRO-1"],
            "text": f"Check in with {summary['name']} and ask how things have been since your last conversation."}]
    if any(f["field"] == "interaction" for f in facts):
        last = next(f for f in facts if f["field"] == "interaction")
        pts.append({"section": "opener", "citations": [last["fact_id"]],
                    "text": "Follow up on the last conversation: " + last["text"].split(": ", 1)[1]})
    for s in suggestions:
        pfacts = [by_id[i] for i in s.get("product_fact_ids", []) if not by_id[i]["stale"]]
        key = next((f for f in pfacts if f["field"] in ("benefits", "interest_rates")), pfacts[0] if pfacts else None)
        text = f"{s['product_name']}: {s['reasons'][0]}"
        cites = s["client_fact_ids"][:1]
        if key:
            text += f" Indicative fact sheet detail (confirm current terms): {key['text'].split(': ', 1)[1]}"
            cites.append(key["fact_id"])
        pts.append({"section": "talking_point", "text": text, "citations": cites})
        if s["product_name"] == "Unit Trusts":
            pts.append({"section": "caution", "citations": [c for c in s["product_fact_ids"] if by_id[c]["field"] == "data_gap"][:1] or cites,
                        "text": "Investment product: complete the suitability assessment before discussing; key risks are not in the public data."})
    pts.append({"section": "question", "citations": ["C-PRO-2"],
                "text": "Have your goals or appetite for risk changed since we last updated your profile?"})
    pts.append({"section": "caution", "citations": [facts[-1]["fact_id"]] if any(f["kind"] == "product" for f in facts) else ["C-PRO-1"],
                "text": "Product details come from public data and are indicative; confirm current rates and terms before quoting."})
    return pts


def generate_brief(summary: dict, suggestions: list[dict], facts: list[dict], use_ai: bool = True) -> dict:
    """Draft talking points with Claude; fall back to templates if AI is off or fails."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if use_ai:
        try:
            return {"points": _llm_points(summary, suggestions, facts), "generator": MODEL_ID,
                    "generated_at": now, "error": None}
        except Exception as e:  # noqa: BLE001 - any API/parse failure falls back
            err = f"{type(e).__name__}: {str(e)[:200]}"
            return {"points": template_points(summary, suggestions, facts), "generator": "template (AI unavailable)",
                    "generated_at": now, "error": err}
    return {"points": template_points(summary, suggestions, facts), "generator": "template",
            "generated_at": now, "error": None}


# ---------------------------------------------------------------------------
# RM feedback log (adoption / acted-on metrics)
# ---------------------------------------------------------------------------


EVENT_COLUMNS = ["ts", "event", "rm_id", "client_id", "brief_id", "generator", "point_idx",
                 "section", "verdict", "edited", "value"]


def log_event(event: dict, path: Path | None = None) -> None:
    """Append one RM action (brief_opened, point_kept/dismissed, finalised, outcome) to the log.

    The log only records what the RM did in the copilot; it never triggers any client action.
    """
    path = path or FEEDBACK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: "" for c in EVENT_COLUMNS}
    row.update({k: v for k, v in event.items() if k in EVENT_COLUMNS})
    row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pd.DataFrame([row], columns=EVENT_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_events(path: Path | None = None) -> pd.DataFrame:
    """Read the event log (empty frame if none yet)."""
    path = path or FEEDBACK_PATH
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    book = load_book()
    sheets = build_fact_sheets()
    print({k: len(v) for k, v in sheets.items()})
    for f in sheets["Time Deposit (Fixed Deposit)"][:4] + sheets["360 Account"][:4]:
        print(f["fact_id"], "|", f["text"][:120])
    ranked = rank_clients(book)
    print(ranked[ranked.rm_id == "RM01"][["client_id", "name", "score", "why"]].head(5).to_string())
    cid = ranked.client_id.iloc[0]
    s = build_client_summary(book, cid)
    sug = suggest_products(s)
    facts = context_for_llm(s, sug, sheets)
    for f in s["facts"]:
        print(f["fact_id"], f["as_of"], f["text"])
    print([(x["product_name"], x["reasons"]) for x in sug])
    brief = generate_brief(s, sug, facts, use_ai=False)
    for p in brief["points"]:
        print(p["section"], p["citations"], p["text"][:150])
