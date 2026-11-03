import hashlib,json,sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from ambifix_hitl.contracts import ValidationError
from ambifix_hitl.storage import Store,AuditError
from ambifix_hitl.exporting import export_bytes,export_to_directory


def review_fields(prediction, **overrides):
    return {"decision": "accept", "selected_label": prediction["predicted_label"],
            "reviewer_id": "software-fixture-reviewer", "review_status": "accepted_by_reviewer", **overrides}


def test_correction_preserves_original_probabilities_and_text(store,prediction):
    original=store.prediction(prediction["prediction_id"])
    corrected=prediction["predicted_label"]%6+1
    event,duplicate=store.append_review(prediction["prediction_id"],"correction-1",**review_fields(prediction,
        decision="correct",selected_label=corrected,explanation="Reviewer-supplied software fixture annotation.",
        clarification_question="Which rehearsal lamp is intended?",rewritten_text="The violet rehearsal lamp shall flash once after Preview.",
        rewrite_status="accepted_by_reviewer"))
    assert not duplicate and event["selected_label"]==corrected
    assert event["prediction"]==original==store.prediction(prediction["prediction_id"])
    assert event["rewritten_text"] and event["clarification_question"] and event["rewrite_authorship"]=="human"
    assert not event["semantic_equivalence_verified"] and not event["automatically_resolved"] and not event["approved_for_training"]
    assert event["timestamp_utc"].endswith("Z") and event["event_id"]
    assert Store(store.path).events()==[event] and Store(store.path).verify()["status"]=="PASS"


def test_accept_and_needs_context(store,prediction):
    store.append_review(prediction["prediction_id"],"accepted",**review_fields(prediction))
    event,_=store.append_review(prediction["prediction_id"],"context",**review_fields(prediction,
        decision="needs_context",selected_label=None,review_status="needs_stakeholder_clarification"))
    assert event["selected_label"] is None and len(store.events())==2 and store.verify()["event_count"]==2


@pytest.mark.parametrize("overrides",[
    {"decision":"correct","selected_label":7},{"decision":"correct","selected_label":"2"},
    {"decision":"needs_context","selected_label":1},{"decision":"needs_context","selected_label":None,"review_status":"accepted_by_reviewer"},
    {"review_status":"resolved_automatically"},{"reviewer_id":""},{"rewrite_status":"accepted_by_reviewer"},
    {"rewritten_text":"A manual rewrite.","rewrite_status":None}])
def test_invalid_review_rejected_without_event(store,prediction,overrides):
    with pytest.raises(ValidationError):store.append_review(prediction["prediction_id"],"invalid",**review_fields(prediction,**overrides))
    assert store.events()==[]


def test_identical_accept_vs_correction_rule(store,prediction):
    with pytest.raises(ValidationError):store.append_review(prediction["prediction_id"],"wrong",**review_fields(prediction,decision="correct"))
    with pytest.raises(ValidationError):store.append_review(prediction["prediction_id"],"wrong2",**review_fields(prediction,selected_label=prediction["predicted_label"]%6+1))


def test_idempotent_duplicate_feedback_and_explicit_conflict(store,prediction):
    one,dup=store.append_review(prediction["prediction_id"],"double-click",**review_fields(prediction))
    two,dup=store.append_review(prediction["prediction_id"],"double-click",**review_fields(prediction))
    assert dup and one==two and len(store.events())==1
    with pytest.raises(ValidationError):store.append_review(prediction["prediction_id"],"double-click",**review_fields(prediction,explanation="different"))


def test_duplicate_requirement_preserves_both_analyses(service,store,prediction):
    other,duplicate=service.analyze(prediction["original_text"],threshold=0.1)
    assert duplicate==1 and other["requirement_id"]==prediction["requirement_id"]
    assert other["prediction_id"]!=prediction["prediction_id"] and len(store.predictions())==2
    assert store.prediction(prediction["prediction_id"])["uncertainty"]["threshold"]==0.75
    assert other["uncertainty"]["threshold"]==0.1


@pytest.mark.parametrize("table",["predictions","requirements","review_events"])
def test_sql_update_delete_refused(store,prediction,table):
    store.append_review(prediction["prediction_id"],"immutable",**review_fields(prediction))
    with sqlite3.connect(store.path) as db:
        for sql in [f"DELETE FROM {table}",f"UPDATE {table} SET rowid=rowid"]:
            with pytest.raises(sqlite3.IntegrityError,match="append-only"):db.execute(sql)
    assert store.verify()["status"]=="PASS"


def test_tamper_detection_even_if_trigger_removed(store,prediction):
    store.append_review(prediction["prediction_id"],"tamper-check",**review_fields(prediction))
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER immutable_review_events_UPDATE")
        db.execute("UPDATE review_events SET event_hash='altered'")
    with pytest.raises(AuditError):store.verify()
    with pytest.raises(AuditError):export_bytes(store)


def test_concurrent_appends_form_one_chain(store,prediction):
    def save(i):return store.append_review(prediction["prediction_id"],f"concurrent-{i}",**review_fields(prediction))
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(save,range(8)))
    assert store.verify()["event_count"]==8


def test_reproducible_versioned_export_no_training_approval(store,prediction,tmp_path):
    store.append_review(prediction["prediction_id"],"export-check",**review_fields(prediction))
    a=export_bytes(store);b=export_bytes(Store(store.path))
    assert a==b and a[3]["mock_event_count"]==1 and not a[3]["approved_for_training"]
    assert a[3]["purpose"]=="review_audit_only"
    folder,raw,manifest=export_to_directory(store,tmp_path/"exports")
    assert export_to_directory(store,tmp_path/"exports")[1]==raw
    (folder/"manifest.json").write_text("tampered")
    with pytest.raises(ValueError):export_to_directory(store,tmp_path/"exports")


def test_export_latest_review_keeps_complete_history(store,prediction):
    store.append_review(prediction["prediction_id"],"draft",**review_fields(prediction,review_status="draft"))
    first=export_bytes(store)
    store.append_review(prediction["prediction_id"],"accepted",**review_fields(prediction))
    second=export_bytes(store)
    assert first[0]!=second[0] and second[3]["event_count"]==2 and second[3]["latest_review_count"]==1
    assert len(second[2]["review_events.jsonl"].splitlines())==2
    assert json.loads(second[2]["reviewed_records.jsonl"])["review_status"]=="accepted_by_reviewer"
