"""Streamlit entry point. Run: python -m streamlit run app.py"""
from pathlib import Path
import json
import os
import sqlite3
import sys
import uuid
import streamlit as st

BASE=Path(__file__).resolve().parent
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
from ambifix_hitl.contracts import LABELS,ValidationError,ModelUnavailable
from ambifix_hitl.predictors import MockPredictor,RealBERTPredictor
from ambifix_hitl.storage import Store,AuditError
from ambifix_hitl.service import ReviewService
from ambifix_hitl.exporting import export_to_directory

st.set_page_config(page_title="AmbiFix · Human Review",page_icon="◈",layout="wide")
st.markdown("""<style>
.stApp {background:#f6f8fa;} .block-container {padding-top:2rem;max-width:1250px;}
h1,h2,h3 {letter-spacing:-.025em;} [data-testid="stMetric"] {background:white;border:1px solid #dce5eb;border-radius:10px;padding:14px;}
[data-testid="stSidebar"] {background:#eaf0f4;} .stTabs [data-baseweb="tab-list"] {gap:14px;}
</style>""",unsafe_allow_html=True)


@st.cache_resource
def predictor_resource(mode,manifest,gate,device):
    return MockPredictor() if mode=="mock" else RealBERTPredictor(manifest,gate,device)


def display_prediction(p):
    st.subheader("Mock output" if p["is_mock"] else "BERT classifier output")
    if p["is_mock"]:st.warning("DEMO / MOCK MODE — deterministic software-test output; no semantic model inference.")
    cols=st.columns([2,1,1])
    cols[0].metric("Predicted category",p["predicted_name"])
    cols[1].metric("Maximum class probability",f"{p['confidence']:.1%}")
    cols[2].metric("Normalized entropy",f"{p['uncertainty']['value']:.3f}")
    st.dataframe([{"Class ID":c,"Category":LABELS[c],"Probability":p["probabilities"][c-1]} for c in LABELS],
        hide_index=True,width="stretch",
        column_config={"Probability":st.column_config.ProgressColumn("Class probability",min_value=0,max_value=1,format="%.4f")})
    u=p["uncertainty"]
    st.info(("Prioritized for human review" if u["prioritized"] else "Below the review-priority threshold")+
            f" · entropy {u['value']:.3f}, stored threshold {u['threshold']:.2f}. This heuristic is uncalibrated.")
    st.caption("Entropy measures how diffuse the six-class distribution is. It does not establish correctness or the probability of an error.")
    if p["model_metadata"].get("truncated"):
        st.warning(f"The classifier used a truncated token sequence (maximum {p['model_metadata']['max_length']}). The exact original text remains stored.")
    with st.expander("Prediction identity and provenance"):
        st.json({k:p[k] for k in ["prediction_id","requirement_id","model_id","checkpoint_version","is_mock","timestamp_utc","dataset_provenance","model_metadata"]})


st.title("AmbiFix")
st.caption("Human review of software requirements · research prototype")
with st.sidebar:
    st.subheader("Workspace")
    default=1 if os.environ.get("AMBIFIX_MODE","mock")=="real" else 0
    choice=st.radio("Inference backend",["Demo / mock","Real BERT"],index=default,key="backend")
    mode="mock" if choice=="Demo / mock" else "real"
    threshold=st.slider("Review entropy threshold",0.0,1.0,0.75,0.05,key="threshold")
    st.caption("Applies to new analyses. Each saved prediction retains its original threshold.")
    reviewer=st.text_input("Reviewer pseudonym",max_chars=80,key="reviewer_id")
    st.caption("Self-reported identifier; this local prototype does not authenticate reviewers.")

if mode=="mock":
    st.warning("DEMO / MOCK MODE — software demonstration only. Genuine BERT checkpoint inference is not active.")
else:
    st.info("Real BERT selected · local checkpoint and approved reference replay required.")
try:
    predictor=predictor_resource(mode,os.environ.get("AMBIFIX_MODEL_MANIFEST",""),
        os.environ.get("AMBIFIX_MODEL_GATE",""),os.environ.get("AMBIFIX_DEVICE","cpu"))
    store=Store(os.environ.get("AMBIFIX_DB",str(BASE/"runtime/reviews.sqlite3")))
    service=ReviewService(predictor,store)
except (ModelUnavailable,ValidationError,AuditError,sqlite3.Error,OSError) as exc:
    st.error(f"Workspace unavailable: {exc}")
    st.caption("No alternative model was substituted. Select Demo / mock explicitly for software testing, or supply the required local artifacts.")
    st.stop()

views=st.tabs(["Requirement Analysis","Human Review","Original vs Revised Requirement","Feedback History","Export and Audit"])
with views[0]:
    st.subheader("Analyze a requirement")
    st.caption("The exact text is preserved. Classification proposes a category for review.")
    with st.form("analysis_form"):
        text=st.text_area("Original requirement",height=150,max_chars=20000,key="requirement_input")
        provenance=st.selectbox("Known source",["Interactive input / partition unknown","Synthetic UI fixture"],key="provenance")
        with st.expander("Optional source provenance"):
            source_name=st.text_input("Dataset or source name (optional)",max_chars=200,key="source_name")
            source_record=st.text_input("Source record ID (optional)",max_chars=200,key="source_record")
        submitted=st.form_submit_button("Analyze requirement",type="primary")
    if submitted:
        try:
            p,duplicate=service.analyze(text,threshold,{"dataset":source_name or ("synthetic_ui_fixture" if provenance=="Synthetic UI fixture" else "interactive_input"),
                "partition":"synthetic" if provenance=="Synthetic UI fixture" else "unknown","source_id":source_record or None,
                "provenance_verification":"reviewer_supplied_not_independently_verified"})
            st.session_state.active_prediction=p["prediction_id"]
            st.session_state.review_token=str(uuid.uuid4())
            if duplicate:st.warning(f"This exact text already has {duplicate} saved analysis record(s). A new prediction record was appended; prior reviews are preserved.")
            st.success("Prediction snapshot saved. Continue to Human Review.")
        except (ValidationError,ModelUnavailable,sqlite3.Error) as exc:st.error(str(exc))
    p=store.prediction(st.session_state.active_prediction) if st.session_state.get("active_prediction") else None
    if p:display_prediction(p)

with views[1]:
    st.subheader("Record human judgment")
    st.caption("Changing a class label does not rewrite or resolve the requirement. All explanations and revisions below are reviewer-authored.")
    if not p:st.info("Analyze a requirement first, or open a saved prediction from Feedback History.")
    else:
        st.caption("Saved prediction mode: "+p["mode_label"])
        st.code(p["original_text"],language=None,wrap_lines=True)
        st.write("Original prediction:",p["predicted_name"])
        key=p["prediction_id"]
        action=st.radio("Review decision",["Accept predicted class","Correct class","Uncertain / needs domain context"],key="decision_"+key)
        with st.form("human_review_"+key):
            if action=="Correct class":
                selected=st.selectbox("Reviewer-selected class",[None,*LABELS],format_func=lambda x:"Choose a class" if x is None else f"{x} · {LABELS[x]}",key="selected_"+key)
            elif action=="Accept predicted class":selected=p["predicted_label"]
            else:selected=None
            statuses={"Draft":"draft","Needs stakeholder clarification":"needs_stakeholder_clarification","Accepted by reviewer":"accepted_by_reviewer"}
            status=st.selectbox("Classification review status",list(statuses),key="status_"+key)
            explanation=st.text_area("Reviewer explanation (optional)",max_chars=5000,key="explanation_"+key)
            question=st.text_area("Proposed clarification question (optional)",max_chars=5000,key="question_"+key)
            rewrite=st.text_area("Manually rewritten requirement (optional)",height=120,max_chars=20000,key="rewrite_"+key)
            rewrite_status=st.selectbox("Manual rewrite status",list(statuses),key="rewrite_status_"+key)
            save=st.form_submit_button("Save review event",type="primary")
        if save:
            try:
                event,duplicate=service.review(key,st.session_state.setdefault("review_token",str(uuid.uuid4())),
                    decision={"Accept predicted class":"accept","Correct class":"correct","Uncertain / needs domain context":"needs_context"}[action],
                    selected_label=selected,reviewer_id=reviewer,review_status=statuses[status],
                    explanation=explanation,clarification_question=question,rewritten_text=rewrite,
                    rewrite_status=statuses[rewrite_status] if rewrite else None)
                st.success("Identical submission already recorded; no duplicate event added." if duplicate else "Review appended. Original prediction preserved.")
                st.caption("Event ID: "+event["event_id"])
            except (ValidationError,AuditError,sqlite3.Error) as exc:st.error(str(exc))
        if st.button("Start another review of this prediction",key="another_"+key):
            st.session_state.review_token=str(uuid.uuid4())
            st.success("New submission token created. Edit the form and save to append another event.")

with views[2]:
    st.subheader("Original and human-authored revision")
    if not p:st.info("Select a saved prediction to compare its original text and recorded rewrite.")
    else:
        st.caption("Saved prediction mode: "+p["mode_label"])
        reviews=[e for e in store.events() if e["prediction_id"]==p["prediction_id"]]
        latest=reviews[-1] if reviews else None
        left,right=st.columns(2)
        left.markdown("**Original requirement — preserved**");left.code(p["original_text"],language=None,wrap_lines=True)
        right.markdown("**Manual rewrite — reviewer-authored**")
        if latest and latest["rewritten_text"]:
            right.code(latest["rewritten_text"],language=None,wrap_lines=True)
            right.write("Rewrite status: "+latest["rewrite_status"].replace("_"," "))
            if latest["clarification_question"]:st.write("Reviewer clarification question:",latest["clarification_question"])
        else:right.info("No manually authored rewrite has been recorded for this prediction.")
        st.caption("Reviewer acceptance is a workflow status. Semantic equivalence, stakeholder agreement and ambiguity resolution have not been independently established.")

with views[3]:
    st.subheader("Feedback history")
    predictions=store.predictions();events=store.events()
    st.caption(f"{len(predictions)} immutable prediction snapshot(s) · {len(events)} append-only review event(s)")
    if predictions:
        options={x["prediction_id"]:x for x in predictions}
        chosen=st.selectbox("Saved prediction",list(reversed(options)),format_func=lambda x:options[x]["timestamp_utc"]+" · "+("MOCK" if options[x]["is_mock"] else "REAL")+" · "+options[x]["original_text"][:65],key="history_choice")
        if st.button("Open saved prediction"):
            st.session_state.active_prediction=chosen
            st.session_state.review_token=str(uuid.uuid4())
            st.rerun()
    if events:
        st.dataframe([{"Sequence":e["sequence"],"UTC":e["timestamp_utc"],"Mode":"MOCK" if e["prediction"]["is_mock"] else "REAL",
            "Reviewer":e["reviewer_pseudonym"],"Decision":e["decision_type"],"Original prediction":e["prediction"]["predicted_name"],
            "Reviewer class":e["selected_name"],"Review status":e["review_status"],"Rewrite status":e["rewrite_status"]} for e in reversed(events)],hide_index=True,width="stretch")
        with st.expander("Full immutable review-event provenance"):st.json(events)
    else:st.info("No feedback events recorded yet.")

with views[4]:
    st.subheader("Export and audit")
    st.caption("Exports contain review events, latest judgments and provenance. They are not approved training datasets and never trigger model updates.")
    if st.button("Verify audit history"):
        try:st.json(store.verify());st.success("Local prediction and event-chain integrity checks passed.")
        except AuditError as exc:st.error(str(exc))
    if st.button("Create versioned audit export"):
        try:
            folder,archive,manifest=export_to_directory(store,os.environ.get("AMBIFIX_EXPORT_DIR",str(BASE/"runtime/exports")))
            st.session_state.export_archive=archive;st.session_state.export_manifest=manifest
            st.success("Audit snapshot created: "+manifest["version"])
        except (AuditError,OSError,ValueError) as exc:st.error(str(exc))
    if st.session_state.get("export_archive") is not None:
        manifest=st.session_state.export_manifest
        st.json(manifest)
        st.download_button("Download audit snapshot ZIP",data=st.session_state.export_archive,file_name=manifest["version"]+".zip",mime="application/zip")
st.caption("Research prototype · no automatic rewriting, online training or verified usability claim")
