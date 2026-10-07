"""Generate SYNTHETIC relationship-manager book data for the copilot.

Nothing here is real customer data. Product names come from
data/products_clean/products.json so holdings line up with real fact sheets.

Every table carries `source` (which synthetic system the row pretends to come
from) and `as_of` (snapshot date) so every downstream fact is traceable and
its age is known.

Run: python -m src.data_gen
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS_PATH = ROOT / "data" / "products_clean" / "products.json"
OUT_DIR = ROOT / "data" / "synthetic"

# Fixed snapshot date so the data set is reproducible. The app computes data
# age against the real current date.
SNAPSHOT_DATE = date(2026, 10, 6)
SEED = 42
N_RMS = 4
CLIENTS_PER_RM = 100

FIRST = ["Wei Ling", "Arjun", "Siti", "Marcus", "Hui Min", "Daniel", "Priya", "Jun Jie",
         "Nurul", "Kenneth", "Mei Xin", "Ravi", "Aisha", "Benjamin", "Li Ting", "Farhan",
         "Grace", "Hafiz", "Joanne", "Kumar", "Shu Fen", "Ethan", "Zara", "Wen Hao"]
LAST = ["Tan", "Lim", "Ng", "Wong", "Rahman", "Pillai", "Chua", "Goh", "Ismail", "Lee",
        "Teo", "Nair", "Koh", "Yusof", "Ong", "Chen"]
OCCUPATIONS = ["Engineer", "Doctor", "Teacher", "Business owner", "Lawyer", "Analyst",
               "Nurse", "Sales director", "Civil servant", "Consultant", "Retiree", "Student"]

# ASSUMPTION: segment thresholds and residency mix are illustrative only, not
# OCBC's actual segmentation or customer demographics.
SEGMENTS = [("Premier", 200_000), ("Affluent", 50_000), ("Mass", 0)]
RESIDENCY = ["Singapore Citizen", "Singapore PR", "Employment Pass", "S Pass"]
RESIDENCY_P = [0.62, 0.18, 0.15, 0.05]
RISK_PROFILES = ["Conservative", "Balanced", "Growth"]

NOTE_TEMPLATES = [
    "Client mentioned planning to buy a {thing} within the next year.",
    "Client asked about ways to earn more on idle cash.",
    "Client said they receive income in USD from overseas work.",
    "Client is reviewing their mortgage as the lock-in period ends soon.",
    "Client prefers low-risk options and dislikes market volatility.",
    "Client wants to start saving regularly for their child's education.",
    "Routine check-in, no specific needs raised.",
    "Client complained about the fall-below fee last month.",
]


def load_product_names() -> list[str]:
    """Return product names from the cleaned OCBC product file."""
    with open(PRODUCTS_PATH) as f:
        return [p["product_name"] for p in json.load(f)]


def _iso(d: date) -> str:
    """Format a date as ISO string."""
    return d.isoformat()


def make_rms() -> pd.DataFrame:
    """Create the relationship-manager roster."""
    names = ["RM Alicia Tan", "RM Bryan Koh", "RM Chitra Nair", "RM Darren Lim"]
    return pd.DataFrame({
        "rm_id": [f"RM{i + 1:02d}" for i in range(N_RMS)],
        "rm_name": names[:N_RMS],
        "source": "synthetic_hr.rm_roster",
        "as_of": _iso(SNAPSHOT_DATE),
    })


def make_clients(rng: np.random.Generator, rms: pd.DataFrame) -> pd.DataFrame:
    """Create client profiles, CLIENTS_PER_RM per RM."""
    rows = []
    n = N_RMS * CLIENTS_PER_RM
    for i in range(n):
        age = int(np.clip(rng.normal(44, 14), 17, 82))
        occupation = "Student" if age < 22 else ("Retiree" if age > 65 else rng.choice(OCCUPATIONS[:-2]))
        # ASSUMPTION: rough income distribution in S$, not calibrated to any real data.
        income = 0 if occupation in ("Student", "Retiree") else int(rng.lognormal(8.9, 0.5) // 100 * 100)
        rows.append({
            "client_id": f"C{i + 1:04d}",
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "age": age,
            "residency": rng.choice(RESIDENCY, p=RESIDENCY_P),
            "occupation": occupation,
            "monthly_income_sgd": income,
            "risk_profile": "Conservative" if age > 65 else rng.choice(RISK_PROFILES, p=[0.35, 0.45, 0.2]),
            "rm_id": rms.rm_id.iloc[i % N_RMS],
            "client_since": _iso(SNAPSHOT_DATE - timedelta(days=int(rng.integers(200, 6000)))),
            "source": "synthetic_crm.client_profile",
            "as_of": _iso(SNAPSHOT_DATE),
        })
    return pd.DataFrame(rows)


def _holding(client_id: str, product: str, balance: float, opened: date, **extra) -> dict:
    """Build one holding row with provenance columns."""
    row = {
        "client_id": client_id,
        "product_name": product,
        "balance_sgd": round(float(balance), 2),
        "opened_on": _iso(opened),
        "maturity_date": "",
        "tenor_months": np.nan,
        "loan_rate_package": "",
        "lock_in_end": "",
        "currency": "SGD",
        "source": "synthetic_core_banking.holdings",
        "as_of": _iso(SNAPSHOT_DATE),
    }
    row.update(extra)
    return row


def make_holdings(rng: np.random.Generator, clients: pd.DataFrame) -> pd.DataFrame:
    """Assign each client a plausible portfolio of the real OCBC product names."""
    rows = []
    for c in clients.itertuples():
        opened = lambda: SNAPSHOT_DATE - timedelta(days=int(rng.integers(60, 3000)))  # noqa: E731
        # Everyone has one primary transactional account.
        if c.age <= 26 and rng.random() < 0.7:
            rows.append(_holding(c.client_id, "FRANK Account", rng.lognormal(8.5, 1.0), opened()))
        elif rng.random() < 0.55 and c.age >= 18:
            rows.append(_holding(c.client_id, "360 Account", rng.lognormal(10.8, 1.0), opened()))
        else:
            rows.append(_holding(c.client_id, "Bonus+ Savings Account", rng.lognormal(10.3, 1.1), opened()))
        if rng.random() < 0.25:
            rows.append(_holding(c.client_id, "Monthly Savings Account", rng.lognormal(8.0, 0.8), opened()))
        if c.age >= 18 and rng.random() < 0.3:
            tenor = int(rng.choice([6, 12, 18, 24]))
            # Some deposits mature soon -> a timely reason to call.
            mat = SNAPSHOT_DATE + timedelta(days=int(rng.integers(3, 200)))
            rows.append(_holding(c.client_id, "Time Deposit (Fixed Deposit)", rng.lognormal(11.0, 0.8),
                                 mat - timedelta(days=30 * tenor), maturity_date=_iso(mat), tenor_months=tenor))
        if c.age >= 18 and rng.random() < 0.12:
            rows.append(_holding(c.client_id, "Global Savings Account", rng.lognormal(10.0, 0.9) * 1.35,
                                 opened(), currency="USD"))
        if c.age >= 18 and rng.random() < 0.06:
            rows.append(_holding(c.client_id, "USD Current Account", rng.lognormal(9.5, 0.6) * 1.35,
                                 opened(), currency="USD"))
        if c.risk_profile != "Conservative" and rng.random() < 0.3:
            rows.append(_holding(c.client_id, "Unit Trusts", rng.lognormal(10.5, 1.0), opened()))
        if 28 <= c.age <= 65 and rng.random() < 0.25:
            # ASSUMPTION: typical 2-3 year lock-in on fixed packages; synthetic only.
            lock_end = SNAPSHOT_DATE + timedelta(days=int(rng.integers(-60, 700)))
            pkg = rng.choice(["Short-term fixed", "Fixed Deposit-linked", "Variable (board rate)"], p=[0.5, 0.35, 0.15])
            rows.append(_holding(c.client_id, "Home Loan (New Purchase)", -rng.uniform(250_000, 1_200_000),
                                 lock_end - timedelta(days=365 * 3), loan_rate_package=pkg,
                                 lock_in_end=_iso(lock_end)))
    df = pd.DataFrame(rows)
    df.insert(0, "holding_id", [f"H{i + 1:05d}" for i in range(len(df))])
    return df


def make_activity(rng: np.random.Generator, clients: pd.DataFrame) -> pd.DataFrame:
    """Create 30-day activity aggregates (inflows, salary credit, card spend)."""
    rows = []
    for c in clients.itertuples():
        salary = c.monthly_income_sgd > 0 and rng.random() < 0.65
        big_inflow = rng.random() < 0.12
        rows.append({
            "client_id": c.client_id,
            "salary_credited_30d": bool(salary),
            "salary_amount_sgd": c.monthly_income_sgd if salary else 0,
            "inflow_30d_sgd": round(float(rng.lognormal(9, 0.7) + (rng.uniform(50_000, 300_000) if big_inflow else 0)), 2),
            "usd_inflow_30d": round(float(rng.lognormal(8.5, 0.8)) if rng.random() < 0.1 else 0.0, 2),
            "bills_paid_30d": int(rng.integers(0, 6)),
            "card_spend_30d_sgd": round(float(rng.lognormal(6.3, 0.8)) if rng.random() < 0.6 else 0.0, 2),
            "window_start": _iso(SNAPSHOT_DATE - timedelta(days=30)),
            "source": "synthetic_core_banking.activity_30d",
            "as_of": _iso(SNAPSHOT_DATE),
        })
    return pd.DataFrame(rows)


def make_interactions(rng: np.random.Generator, clients: pd.DataFrame) -> pd.DataFrame:
    """Create past RM interaction notes (1-3 per client)."""
    rows = []
    things = ["condo", "HDB flat", "car"]
    for c in clients.itertuples():
        for _ in range(int(rng.integers(1, 4))):
            d = SNAPSHOT_DATE - timedelta(days=int(rng.integers(5, 420)))
            note = rng.choice(NOTE_TEMPLATES).format(thing=rng.choice(things))
            rows.append({
                "client_id": c.client_id,
                "date": _iso(d),
                "channel": rng.choice(["Call", "Branch meeting", "Video call"]),
                "note": note,
                "source": "synthetic_crm.interaction_log",
                "as_of": _iso(SNAPSHOT_DATE),
            })
    df = pd.DataFrame(rows).sort_values(["client_id", "date"])
    df.insert(0, "interaction_id", [f"I{i + 1:05d}" for i in range(len(df))])
    return df


def make_campaign_history(rng: np.random.Generator, features: pd.DataFrame) -> pd.DataFrame:
    """Label whether a past RM outreach led to a product take-up.

    ASSUMPTION: outcomes come from a hand-written latent rule plus noise, so
    the priority model learns a known pattern. Real outcomes would come from
    the bank's campaign / CRM history.
    """
    z = (-2.2
         + 1.3 * features["td_maturing_60d"]
         + 1.1 * features["lock_in_ending_120d"]
         + 0.9 * (features["idle_cash_sgd"] > 50_000)
         + 0.8 * features["large_inflow_30d"]
         + 0.5 * np.log1p(features["days_since_contact"]) / np.log(400)
         + 0.4 * features["no_investment_growth_profile"]
         + rng.normal(0, 0.6, len(features)))
    p = 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "client_id": features["client_id"],
        "contacted_last_campaign": True,
        "took_up_product": (rng.random(len(p)) < p).astype(int),
        "source": "synthetic_crm.campaign_outcomes",
        "as_of": _iso(SNAPSHOT_DATE),
    })


def generate(out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    """Generate all synthetic tables, write them as CSV, and return them."""
    # Imported here to avoid a circular import (core reads the CSVs we write).
    from src.core import compute_features

    rng = np.random.default_rng(SEED)
    rms = make_rms()
    clients = make_clients(rng, rms)
    holdings = make_holdings(rng, clients)
    activity = make_activity(rng, clients)
    interactions = make_interactions(rng, clients)
    feats = compute_features(clients, holdings, activity, interactions, SNAPSHOT_DATE)
    campaigns = make_campaign_history(rng, feats)

    tables = {"rms": rms, "clients": clients, "holdings": holdings, "activity": activity,
              "interactions": interactions, "campaign_outcomes": campaigns}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = generate()
    for k, v in t.items():
        print(f"{k}: {len(v)} rows")
    print(t["holdings"].head(8).to_string())
    print(t["campaign_outcomes"].took_up_product.mean().round(3), "take-up rate")
