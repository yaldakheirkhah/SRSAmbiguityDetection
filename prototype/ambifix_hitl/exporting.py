"""Deterministic, versioned audit export. Not a model-training dataset approval."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
from .contracts import canonical


def export_bytes(store):
    store.verify()
    events = store.events()  # Committed immutable rows; later appends belong to a later export.
    latest = {}
    for event in events:
        latest[event["prediction_id"]] = event
    files = {
        "review_events.jsonl": "".join(canonical(e)+"\n" for e in events).encode(),
        "reviewed_records.jsonl": "".join(canonical(e)+"\n" for e in sorted(latest.values(), key=lambda e:e["sequence"])).encode(),
    }
    identity = hashlib.sha256(files["review_events.jsonl"]).hexdigest()
    version = "feedback-v1-"+identity[:16]
    manifest = {"schema_version":1,"version":version,"purpose":"review_audit_only",
        "approved_for_training":False,"event_count":len(events),"latest_review_count":len(latest),
        "mock_event_count":sum(e["prediction"]["is_mock"] for e in events),
        "review_statuses_included":"all; explicit statuses preserved, not expert-validated",
        "event_head_hash":events[-1]["event_hash"] if events else "0"*64,
        "source_event_timestamp_max":events[-1]["timestamp_utc"] if events else None,
        "files_sha256":{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        "restrictions":"No automatic retraining. Future use needs deliberate authorization, training-partition provenance checks and label validation. Classification acceptance does not establish textual resolution."}
    files["manifest.json"] = (canonical(manifest)+"\n").encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo(name,date_time=(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,files[name])
    return version,buffer.getvalue(),files,manifest


def export_to_directory(store,destination):
    version,archive,files,manifest=export_bytes(store)
    folder=Path(destination)/version
    folder.mkdir(parents=True,exist_ok=True)
    all_files={**files,version+".zip":archive}
    for name,data in all_files.items():
        target=folder/name
        if target.exists():
            if target.read_bytes()!=data:
                raise ValueError("Existing content-version export differs; refusing overwrite.")
        else:
            with target.open("xb") as f:f.write(data)
    return folder,archive,manifest
