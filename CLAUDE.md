# Project conventions
- Python 3.11. Use pandas, scikit-learn, streamlit. Keep dependencies minimal.
- All customer data is SYNTHETIC. Real product facts are in data/products_clean/products.json (cleaned from OCBC's public API JSON in data/ocbc_products/, pulled 5 Oct 2026, indicative and possibly outdated). Use the clean file, not the raw JSON. Cite product_name and source_file, and never present its rates as current.
- Structure: data generation in src/data_gen.py, business logic/model in src/core.py, evaluation in src/eval.py, UI in app.py.
- Prefer simple, explainable approaches first (rules, logistic regression) before complex models.
- Every model or ranking must expose WHY (feature contributions, reasons, or cited sources).
- Write small functions with docstrings. No notebooks.
- After each change, run the code and show a sample of output.
- Flag any assumption about banking data or regulation explicitly in comments.
- Python environment: ~/Source/ocbc-env/.venv (already activated). Install anything new with `uv pip install`.
- This is an Intel Mac: never install torch or sentence-transformers. For semantic search use model2vec (minishlab/potion-base-8M).
- Load secrets with `from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))`. Never print or hard-code API keys.
- Don't run `streamlit run` yourself — I run the app in a separate terminal tab; it reloads on save.
- Wrap every Claude API call in try/except with a timeout and a non-AI fallback, so the app never crashes if the API fails.
- In Streamlit, escape every $ in text you display (st.markdown treats text between two $ signs as maths, so 'S$800 … S$160' renders garbled).
- Never name a module after a Python built-in (e.g. copy.py, json.py).

# Engineering practices (learned from practice)
- Every data row and every fact shown to a user or an LLM carries a `source` and an `as_of` date, so it's traceable and its age is known.
- Make the Claude model configurable through an env var read from .env (e.g. `<PROJECT>_MODEL`), defaulting to `claude-opus-5-5`.
- LLM prompts that mention people: tell the model to use the name or "the client" and not infer gender or other traits from a name.
- Compare every model with a simple rules baseline and random, using repeated CV or a holdout, and report results honestly (say "within noise" when it is). Mark synthetic-label results as optimistic.
- Streamlit: use `width="stretch"`, not the deprecated `use_container_width`. Verify the app headlessly with `streamlit.testing.v1.AppTest` (also check no unescaped `$` in rendered markdown).
- The app must never overwrite saved eval artefacts; only the eval script writes them.

# Tests
- Tests live in tests/, with a tests/conftest.py that puts the repo root on sys.path, so plain `pytest tests/` works from any directory.
- Tests make no API calls. Include: a fallback test that simulates an API failure; an AppTest smoke test that redirects any log or output files to tmp_path; and guardrail tests for the core promises (grounding, consent, no prohibited phrases, etc.).

# Docs to keep in every project (update before each commit)
- requirements.txt: pinned to installed versions, `# Python 3.11` at the top, a comment per package saying what it's for; note any non-pip system tools.
- README.md: what it does, an "Unofficial prototype, not affiliated with OCBC, all customer data synthetic, product data pulled <date>, indicative" disclaimer, how it works, a results table with honest caveats, verified run commands, a file table, and links to HANDOFF.md and docs/SESSION_LOG.md.
- HANDOFF.md: run order, files and key functions, data sources and what should replace them, key assumptions (matching the ASSUMPTION comments), known shortcuts, and steps to productionise.
- docs/SESSION_LOG.md: a dated log per session: starting point, the ask, design decisions and why, bugs and corrections, results, publishing, open caveats. Append; don't rewrite history.
- The eval script writes a committed data/eval/SUMMARY.md (readable) alongside any JSON report.
- Before committing: run every command in the README, and make sure every number in the docs matches the latest outputs.

# Git
- Commit straight to main (no feature branches). End commit messages with the Co-Authored-By line.
- If there's no remote, ask for the repo URL rather than guessing. Keep .env, .pytest_cache/ and local logs gitignored.
