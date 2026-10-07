# Handoff: RM copilot (prototype)

The prototype ranks an RM's clients by who needs attention (with reasons), summarises each client's holdings and
signals in code, and drafts cited talking points with Claude. A plain-code fact-checker tests every point. The RM
reviews, edits and decides; the copilot never contacts a client.

## Run it
Python 3.11. Run `source ~/Source/ocbc-env/.venv/bin/activate`, then put `ANTHROPIC_API_KEY` in `.env`. Run
everything from the repo root:
1. `python -m src.data_gen`: synthetic book → `data/synthetic/*.csv` (committed; also rebuilt on first use if missing).
2. `python -m src.core`: prints fact-sheet sizes, the top of RM01's ranking, one client summary and a template brief.
3. `python -m src.eval`: fact-checker self-test and ranking CV → `data/eval/report.json` and `SUMMARY.md`
   (both committed). Add `--llm N` to fact-check N Claude-drafted briefs. Without it, the last saved Claude results
   are kept.
4. `streamlit run app.py`.
5. `pytest tests/ -v`: 6 tests, about 6 s, no API calls.

## Files
| File | What it does |
|---|---|
| `src/data_gen.py` | `make_clients()`, `make_holdings()`, `make_activity()`, `make_interactions()`, `make_campaign_history()` (planted outcome rule), `generate()`. Seed 42, snapshot 2026-10-06. |
| `src/core.py` | `build_fact_sheets()` / `split_bullets()` / `select_product_facts()`; `compute_features()`, `build_client_summary()`, `derive_signals()`; `train_priority_model()`, `explain_scores()`, `rank_clients()`; `RULES` + `suggest_products()`; `generate_brief()` (Claude via `_llm_points()`, else `template_points()`); `log_event()`. |
| `src/eval.py` | `check_point()` / `fact_check_brief()`, `checker_self_test()`, `ranking_eval()`, `llm_brief_eval()`, `write_summary()`, `run_all()`. |
| `app.py` | Streamlit UI. The eval tab recomputes the deterministic eval without overwriting the saved report. |
| `tests/` | `conftest.py` puts the repo root on the path. `test_copilot.py`: provenance on every fact; template briefs pass; checker flags invented numbers and actions; AI-off and API-failure fallbacks; `AppTest` smoke test with the event log redirected to a temp file. |

## Data sources and what should replace them
- **Clients, holdings, activity, interactions and outreach outcomes are synthetic.** Replace them with core banking
  and CRM feeds, with point-in-time snapshots so each fact's `as_of` is real.
- **Outreach outcomes** should come from logged RM contacts and product take-up, ideally with a random holdout of
  clients not contacted, so the ranking can be judged on what the call caused rather than who would act anyway.
- **Product facts** come from OCBC's public API, pulled 5 Oct 2026, with known staleness: SIBOR packages (SIBOR is
  retired) and unit-trust returns as of March 2016. Replace with the bank's approved product disclosures (product
  highlights sheets, rate boards, fee schedules) with effective dates and owners.
- **Not in the product data:** credit cards, insurance, live FX and deposit rates. The drafter can't mention them,
  and the fact-checker warns if they appear.

## Key assumptions
Each is labelled `ASSUMPTION` in the code.
- Segment thresholds, residency mix and income distribution are invented.
- USD balances are a synthetic S$-equivalent figure; there's no FX conversion.
- "Idle cash" = balances in FRANK, Bonus+ and Monthly Savings accounts, plus any 360 Account balance above the
  S$70,000 bonus cap from the fact sheet.
- Home-loan lock-ins are assumed to be about 3 years. A lock-in ending within 120 days, or ended in the last 90,
  counts as a signal.
- Product rules (signal → product) are illustrative sales logic, not OCBC policy.
- Outreach outcomes follow a hand-written rule plus noise, so the model learns a known pattern.
- The banned-phrase list is a short stand-in inspired by fair-dealing principles, not an official list.
- The prep-time baseline (30 min) is a placeholder until RM time studies exist.
- Suitability for unit trusts is flagged as a caution only; no actual suitability or Customer Knowledge Assessment
  is modelled.

## Known shortcuts
- **The ranking barely beats rules.** AUC 0.68 vs 0.67 for counting signals, within noise, on synthetic labels.
  In the app the model is trained and scored on the same clients.
- **Number grounding is presence-only.** The checker confirms each number appears in the cited facts, not that it's
  paired with the right claim (quoting the 24-month rate as the 12-month rate passes if both facts are cited).
- **Strict product check.** Mentioning a product the client holds without citing the holding is a fail. That caught a
  real gap in one Claude brief, but it can also be over-strict.
- **Keyword lists.** Out-of-scope terms, banned phrases and "acting on behalf" patterns are regexes and will miss
  paraphrases.
- **Small LLM eval.** Only 3 Claude briefs have been fact-checked end to end.
- **No identity or access control.** The RM is picked from a dropdown, and the event log is a local CSV.
- **Prep timer is rough.** It starts when a client is first opened in a browser session and stops at finalise.
- **Client data is sent to the Anthropic API** in the prompt (synthetic here). Real client data would need data
  classification, residency and outsourcing approval first.
- **No screenshots yet** in the README.

## To productionise
1. **Data:** governed feeds from core banking and CRM, point-in-time features, and freshness checks that block stale
   loads.
2. **Product facts:** an approved, versioned claims library with effective dates; drop retired benchmarks (SIBOR).
3. **Ranking:** train on real outreach outcomes with a time-based split, compare against the rules baseline, calibrate
   scores, and add contact-frequency caps. Check consent and Do Not Call rules (PDPA) before any outbound contact.
4. **Drafting and checks:** pair numbers with claims (structured fact references rather than free text), add an
   entailment check, and require suitability workflows for investment products.
5. **Governance:** an audit log of prompt, fact versions, output and RM edits; human sign-off stays mandatory; model
   and prompt changes go through the eval in `src/eval.py` before release.
6. **Measurement:** a pilot with treatment and control RMs to measure prep time, adoption, acted-on share and revenue
   per meeting properly.
