"""Tests for the fact-checker, fact sheets and a headless app smoke test (no API calls)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import core
from src import eval as fc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ctx():
    """One client's summary, suggestions and facts."""
    book = core.load_book()
    sheets = core.build_fact_sheets()
    cid = core.rank_clients(book).client_id.iloc[0]
    s = core.build_client_summary(book, cid)
    sug = core.suggest_products(s)
    return s, sug, core.context_for_llm(s, sug, sheets)


def test_every_fact_has_source_and_date(ctx):
    _, _, facts = ctx
    assert facts and all(f["source"] and f["as_of"] and f["fact_id"] for f in facts)


def test_template_brief_passes_fact_check(ctx):
    s, sug, facts = ctx
    res = fc.fact_check_brief(core.template_points(s, sug, facts), facts)
    assert res["summary"]["fail"] == 0


def test_checker_flags_invented_number_and_action(ctx):
    _, _, facts = ctx
    by_id = {f["fact_id"]: f for f in facts}
    bad = {"section": "talking_point", "citations": ["C-PRO-1"],
           "text": "I have booked a deposit at 4.85% for you."}
    checks = {i["check"] for i in fc.check_point(bad, by_id, fc.known_products())["issues"]}
    assert {"numbers_grounded", "no_action_on_behalf"} <= checks


def test_ai_off_falls_back_to_template(ctx):
    s, sug, facts = ctx
    assert core.generate_brief(s, sug, facts, use_ai=False)["generator"] == "template"


def test_app_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "FEEDBACK_PATH", tmp_path / "events.csv")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.toggle[0].set_value(False).run()
    at.button[0].click().run()          # Draft talking points (template)
    assert not at.exception
    at.button[1].click().run()          # Finalise
    assert not at.exception
    assert (tmp_path / "events.csv").exists()
