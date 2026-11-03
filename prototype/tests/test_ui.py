from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from ambifix_hitl.storage import Store

BASE=Path(__file__).resolve().parents[1]


@pytest.fixture
def ui(monkeypatch,tmp_path):
    monkeypatch.setenv("AMBIFIX_DB",str(tmp_path/"ui.sqlite3"));monkeypatch.setenv("AMBIFIX_EXPORT_DIR",str(tmp_path/"exports"))
    monkeypatch.setenv("AMBIFIX_MODE","mock");monkeypatch.delenv("AMBIFIX_MODEL_MANIFEST",raising=False)
    monkeypatch.delenv("AMBIFIX_MODEL_GATE",raising=False)
    app=AppTest.from_file(str(BASE/"app.py"),default_timeout=45).run()
    assert not app.exception
    return app,tmp_path


def click(app,label):
    [b for b in app.button if b.label==label][0].click().run()
    assert not app.exception


def analyze(app,text="The violet test lantern shall flash after the rehearsal preview."):
    app.text_area(key="requirement_input").set_value(text)
    click(app,"Analyze requirement")
    return app.session_state["active_prediction"]


def test_ui_submission_mock_probabilities_and_exact_text(ui):
    app,tmp=ui
    assert any("DEMO / MOCK MODE" in x.value for x in app.warning)
    text="  The violet test lantern shall flash.\n  "
    app.text_input(key="source_name").set_value("new-synthetic-ui-fixture")
    app.text_input(key="source_record").set_value("HITL-SOFTWARE-CASE-1")
    key=analyze(app,text)
    assert len(app.metric)==3 and app.dataframe
    p=Store(tmp/"ui.sqlite3").prediction(key)
    assert p["original_text"]==text and p["is_mock"] and len(p["probabilities"])==6
    assert p["dataset_provenance"]["dataset"]=="new-synthetic-ui-fixture"
    assert p["dataset_provenance"]["source_id"]=="HITL-SOFTWARE-CASE-1"


def test_ui_acceptance_restart_history_and_export(ui):
    app,tmp=ui;key=analyze(app)
    app.text_input(key="reviewer_id").set_value("synthetic-ui-reviewer")
    app.selectbox(key="status_"+key).set_value("Accepted by reviewer")
    click(app,"Save review event")
    store=Store(tmp/"ui.sqlite3");assert store.events()[0]["decision_type"]=="accept"
    click(app,"Save review event");assert len(store.events())==1
    click(app,"Verify audit history");assert any("integrity checks passed" in x.value for x in app.success)
    click(app,"Create versioned audit export")
    assert app.session_state["export_archive"] and app.session_state["export_manifest"]["purpose"]=="review_audit_only"
    restarted=AppTest.from_file(str(BASE/"app.py"),default_timeout=45).run()
    assert not restarted.exception and restarted.selectbox(key="history_choice")
    click(restarted,"Open saved prediction");assert restarted.session_state["active_prediction"]==key


def test_ui_correction_clarification_and_manual_rewrite(ui):
    app,tmp=ui;key=analyze(app);store=Store(tmp/"ui.sqlite3");old=store.prediction(key)
    app.text_input(key="reviewer_id").set_value("synthetic-ui-reviewer")
    app.radio(key="decision_"+key).set_value("Correct class").run()
    corrected=old["predicted_label"]%6+1
    app.selectbox(key="selected_"+key).set_value(corrected)
    app.text_area(key="explanation_"+key).set_value("Software fixture annotation, not a real participant.")
    app.text_area(key="question_"+key).set_value("Which rehearsal light is intended?")
    app.text_area(key="rewrite_"+key).set_value("The violet rehearsal light shall flash exactly once after Preview.")
    app.selectbox(key="rewrite_status_"+key).set_value("Needs stakeholder clarification")
    click(app,"Save review event")
    e=store.events()[0]
    assert e["selected_label"]==corrected and e["prediction"]==old
    assert e["clarification_question"] and e["rewrite_status"]=="needs_stakeholder_clarification"
    assert any("violet rehearsal light" in x.value for x in app.code)


def test_ui_empty_input_and_missing_reviewer(ui):
    app,tmp=ui;click(app,"Analyze requirement")
    assert app.error and not Store(tmp/"ui.sqlite3").predictions()
    analyze(app);click(app,"Save review event")
    assert app.error and Store(tmp/"ui.sqlite3").events()==[]


def test_ui_duplicate_analysis_is_explicit(ui):
    app,tmp=ui;analyze(app);analyze(app)
    assert any("exact text already" in x.value for x in app.warning)
    assert len(Store(tmp/"ui.sqlite3").predictions())==2


def test_ui_real_missing_model_no_silent_mock(ui):
    app,tmp=ui
    app.radio(key="backend").set_value("Real BERT").run()
    assert not app.exception and app.error
    assert any("No alternative model was substituted" in x.value for x in app.caption)
    assert not app.metric and not any(b.label=="Analyze requirement" for b in app.button)
