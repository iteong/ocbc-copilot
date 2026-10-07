# Session log: building the RM copilot prototype

A record of one Claude Code session (7 Oct 2026): what was asked, what was built, the decisions taken and why, the
results, and the open caveats. For how to run the project, see [`HANDOFF.md`](../HANDOFF.md).

## Starting point
- **Repo:** `ocbc-round` (renamed to `ocbc-copilot` during the session). It held cleaned OCBC product data
  (`data/products_clean/`, 10 deposit, loan and investment products from OCBC's public API, pulled 5 Oct 2026) and
  empty stubs for `src/data_gen.py`, `src/core.py`, `src/eval.py` and `app.py`.
- **Project rules** (`CLAUDE.md`):
  - Python 3.11; pandas, scikit-learn, Streamlit; minimal dependencies;
  - simple, explainable methods first, and every ranking must show WHY;
  - cite product_name and source_file, and never present rates as current;
  - flag banking and regulatory assumptions;
  - every Claude call wrapped in try/except with a timeout and a non-AI fallback;
  - escape `$` in Streamlit text;
  - don't run `streamlit run`.

## The ask
A copilot that helps RMs decide who to call, see each client's holdings and know what to say, measured on prep time
saved, RM adoption, share of suggestions acted on and revenue per meeting. The copilot must not act for the RM, and
every fact must be traceable to a source and timestamped. Deliverables: synthetic client data, product fact sheets
from the cleaned product file, a client summary computed in code, an LLM layer writing cited talking points, a
fact-check evaluation layer, and a Streamlit app.

## Design decisions
- **Facts as the unit of grounding.** Product fields and client data are both turned into small facts with an id,
  text, source and `as_of` date. Claude may only use and cite these ids, and the checker verifies against them.
- **Signals and product suggestions are rules, not the LLM.** The LLM only phrases what code has already worked out,
  which keeps "why" explicit.
- **Ranking:** logistic regression with per-client contributions (coefficient × standardised value) as reasons.
- **Structured output** via `client.beta.messages.parse` with a pydantic schema (`MeetingBrief`), effort `low` for
  latency, and `fallbacks: "default"` for refusals.
- **The Claude code lives in `src/core.py`** to keep to the project's module structure.
- **The RM is in control:** points failing the fact-check start unticked, edits are re-checked on finalise, and the
  output is notes to copy. Nothing is sent.

## Build, and corrections along the way
- **Fact sheet parser.** The first version made headings such as "Fixed Deposit-linked rates" into standalone facts.
  Fixed by treating any line with child bullets (and short unpunctuated sub-headings) as context. Leaves now read
  like "Interest rates: Fixed Deposit-linked rates > 36-month Fixed Deposit Mortgage Rate + 1.23% for the first 3 years".
- **Time Deposit tier bug.** Tier labels such as `>$50,000 - S$99,999` contain `>`, which was also the context
  separator, so every SGD tier was dropped. Fixed by splitting on `" > "`.
- **Ranking honesty.** On one 72-client split the model (AUC 0.71) lost to a count-of-signals rule (0.74). Changed to
  repeated 5-fold CV, grew the book from 240 to 400 clients, and made idle cash a >S$50,000 flag consistent with the
  signal. Result: model 0.68 ± 0.06 vs rules 0.67 ± 0.05 vs random 0.50, reported as "within noise".
- **First Claude run (2 briefs):** all citations valid, no fallbacks. Tuning that followed:
  - "banking with us since 2016" failed number grounding although it came from `client_since`, so years from cited
    dates now count;
  - the indicative-caveat check no longer applies to RM-facing cautions;
  - one draft called the client "her" from the name alone, so the prompt now says to use the name or "the client";
  - synthetic loans had been given SIBOR packages, which is unrealistic since SIBOR is retired. Replaced with a
    variable board-rate package.
- **Final Claude run (3 briefs):** 0% fallback, 100% valid citations, 91% of points pass. The one fail was real: a
  caution mentioned the client's 360 Account without citing the holding.
- **App** checked headlessly with Streamlit `AppTest`: no unescaped `$` in rendered markdown. Replaced the deprecated
  `use_container_width` with `width="stretch"`, and fixed "1 days old".

## Publishing
- Committed `5f2a00f` on branch `rm-copilot`, fast-forwarded `main`, then deleted the branch.
- Push failed: the repo has no remote configured yet.
- The user renamed the folder from `ocbc-round` to `ocbc-copilot`. Nothing in the code depends on the folder name.

## Docs adopted from `ocbc-campaigns`
- README, HANDOFF and this log.
- Pinned `requirements.txt`.
- `src/eval.py` now also writes a committed `data/eval/SUMMARY.md`, and keeps the last Claude results when run
  without `--llm`.
- An `RM_COPILOT_MODEL` setting (default `claude-opus-5-5`).
- **Fix:** the app's eval tab had been overwriting `data/eval/report.json` and dropping the saved Claude results. It
  now recomputes without saving.
- **Reliability:** a real call with an invalid key fell back to template points in 1.9 s. A new test simulates an API
  outage and checks the fallback brief passes the fact-check. 6 tests pass in about 6 s.
- **Test fix:** plain `pytest tests/` couldn't import `src` (only `python -m pytest` worked). Added
  `tests/conftest.py` to put the repo root on the path.

## Screenshots
- The user added 7 screenshots to `data/screenshots/`, which are now in the README (the ranking view as the main
  image, the rest collapsible). Checked: every S$ amount renders correctly, and the Claude brief shown passed all
  11 points.

## Pushed to GitHub
- Checked first: no `.env` or API key anywhere in the git history.
- Added `origin` (https://github.com/iteong/ocbc-copilot, empty at the time). The first push failed with 403: the
  saved fine-grained GitHub token had no write access to the new repo (the same issue as `ocbc-campaigns`). The user
  fixed the token, and the retry pushed `main` (`5d6ddf5`), which now tracks `origin/main`.

## Open caveats
See `HANDOFF.md` → Known shortcuts. The main ones:
- the ranking is no better than simple rules on synthetic labels;
- number grounding doesn't check that numbers are paired with the right claim;
- only 3 Claude briefs have been evaluated end to end;
- client data would go to the Anthropic API, which needs approval for real data;
- nothing has been reviewed by Compliance.
