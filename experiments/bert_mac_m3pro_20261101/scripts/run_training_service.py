"""Observe ONE invocation of the existing audited training program.

This is process/report management only: it contains no training implementation,
model forward, dataset discovery, recipe edit, automatic retry or final eval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import uuid

EXPERIMENT = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
LABELS = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_precision", "weighted_recall", "weighted_f1"]
ROOT_DOCS = ["MAC_TRAINING_EXECUTION_REPORT.md", "MANUSCRIPT_RESULTS_READY.md", "COMPLETED.json"]


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic(path, value, is_json=True):
    path = Path(path)
    temporary = path.with_name(path.name + ".service-partial-" + uuid.uuid4().hex)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n" if is_json else value
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def safe_file(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Expected a relative POSIX artifact path")
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts or any(p.casefold() == "test.csv" for p in part.parts):
        raise ValueError("Invalid/protected artifact path")
    root = Path(root).resolve()
    for end in range(1, len(part.parts)+1):
        if root.joinpath(*part.parts[:end]).is_symlink():
            raise ValueError("Linked service artifact is prohibited")
    return root.joinpath(*part.parts)


def run_path(root, relative):
    part = PurePosixPath(relative)
    if len(part.parts) != 2 or part.parts[0] != "outputs" or not part.parts[1].startswith("mac_mps_original_sqrtweighted_"):
        raise ValueError("Unexpected Mac run directory")
    return safe_file(root, relative)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observe(root):
    started = read_json(root / "outputs/STARTED.json")
    status = {"status": "STARTING_PREFLIGHT", "success": False,
              "experiment_id": EXPERIMENT, "timestamp_utc": now(),
              "service_scope": "Observer of existing audited train_bert.py --execute; no recipe changes",
              "training_started_marker_observed": bool(started),
              "MPS_optimizer_progress_observed": False,
              "new_validation_macro_f1": None, "best_checkpoint_independently_verified": False,
              "COMPLETE_ARTIFACTS_zip_verified": False, "held_out_data_accessed": False}
    if started:
        run = run_path(root, started["run_directory"])
        status["run_directory"] = started["run_directory"]
        progress = read_json(run / "progress.json")
        status["status"] = "CORE_RUN_STARTED_WAITING_FOR_OPTIMIZER_PROGRESS"
        if progress:
            status["last_actual_progress"] = progress
            status["MPS_optimizer_progress_observed"] = int(progress.get("successful_optimizer_steps_this_epoch", 0)) > 0
            status["status"] = progress.get("status", "TRAINING_IN_PROGRESS")
    return status


def publish(root, status, verification=None):
    atomic(root / "outputs/SERVICE_STATUS.json", status)
    atomic(root / "COMPLETED.json", status)
    lines = ["# Mac BERT training execution", "", f"Updated UTC: {status['timestamp_utc']}", "",
             f"**Status: {status['status']}**", "",
             "This is the separate original-only MPS FP32 experiment. The audited recipe is unchanged.",
             "The existing training program is invoked once; no automatic restart or hyperparameter search occurs.",
             "The final held-out partition remains sealed; the final-evaluation script is never invoked.", ""]
    if verification:
        run = run_path(root, status["run_directory"])
        scores = verification["metrics"]
        lines += [f"Actual independently verified selected epoch: **{status['selected_epoch']}**.", "",
                  "| Actual validation metric | Value |", "|---|---:|"]
        lines += [f"|{key}|{scores[key]:.15g}|" for key in METRICS]
        lines += ["", "|Class|Precision|Recall|F1|Support|", "|---|---:|---:|---:|---:|"]
        for number, label in enumerate(LABELS, 1):
            entry = scores["per_class"][str(number)]
            lines.append(f"|{number} {label}|{entry['precision']:.12g}|{entry['recall']:.12g}|{entry['f1']:.12g}|{entry['support']}|")
        lines += ["", f"- Best checkpoint: `{run / 'best_model'}`.",
                  f"- Actual figures/source data: `{run / 'figures'}`.",
                  f"- Complete artifact ZIP: `{root / status['return_zip']}`.",
                  f"- Core success marker: `{root / 'outputs/COMPLETED.json'}`.",
                  f"- Additional independent arithmetic/file verification: `{root / 'reports/SERVICE_INDEPENDENT_VERIFICATION.json'}`.", "",
                  "The core reloaded the exported model and saved tokenizer and replayed all1015 validation rows before success.",
                  "The service separately reran the stdlib artifact/metric verifier; it did not independently repeat neural inference."]
    else:
        lines += ["No final new validation Macro F1 is reported until training, selected-checkpoint reload,",
                  "independent verification and complete artifact ZIP checks have all succeeded.", "",
                  f"Observed MPS optimizer progress: **{status.get('MPS_optimizer_progress_observed', False)}**."]
        if status.get("run_directory"):
            lines += [f"Actual run directory: `{root / status['run_directory']}`."]
        if status.get("last_actual_progress"):
            lines += ["", "Last actual progress record:", "```json",
                      json.dumps(status["last_actual_progress"], indent=2), "```"]
        if status.get("failure"):
            lines += ["", "Failure/blocked diagnostic:", "```json", json.dumps(status["failure"], indent=2), "```",
                      "Actual logs and partial artifacts are retained. No automatic second training run is started."]
    lines += ["", "Historical validation Macro F1 0.581140 is reference-only and never a target.",
              "Inherited punctuation-equivalent overlaps, annotation limitations, single-seed selection and MPS/CUDA precision differences remain.",
              "After actual completion, freeze the verified model/protocol and obtain separate explicit authorization before final evaluation.", ""]
    report = "\n".join(lines)
    atomic(root / "MAC_TRAINING_EXECUTION_REPORT.md", report, is_json=False)
    ready = "# Manuscript results readiness\n\n" + ("**ACTUAL MAC VALIDATION EVIDENCE AVAILABLE.**\n\n" if verification else "**NEW MAC RESULTS NOT YET READY.**\n\n") + report
    atomic(root / "MANUSCRIPT_RESULTS_READY.md", ready, is_json=False)


def independently_verify(root, marker):
    if marker.get("status") != "COMPLETED_NEW_VALIDATION_ONLY_RUN" or marker.get("experiment_id") != EXPERIMENT:
        raise ValueError("Core did not publish the expected genuine Mac completion")
    if not all(marker.get(k) is True for k in ["model_checkpoint_complete", "independent_verification_passed", "publication_figures_exported"]):
        raise ValueError("Core success is missing an essential quality gate")
    if marker.get("held_out_evaluation_performed") is not False:
        raise ValueError("Final evaluation boundary differs")
    if marker.get("return_zip") != "outputs/COMPLETE_ARTIFACTS.zip":
        raise ValueError("Unexpected complete return ZIP")
    archive = safe_file(root, marker["return_zip"])
    if archive.stat().st_size != marker["return_zip_bytes"] or sha(archive) != marker["return_zip_sha256"]:
        raise ValueError("Complete return ZIP identity mismatch")
    run = run_path(root, marker["run_directory"])
    if not marker.get("artifact_sha256"):
        raise ValueError("No actual artifact identities")
    for relative, expected in marker["artifact_sha256"].items():
        if sha(safe_file(run, relative)) != expected:
            raise ValueError("Core artifact changed after completion: " + relative)
    command = [str(root / ".venv/bin/python"), str(root / "scripts/verify_artifacts.py"),
               "--root", str(root), "--run-dir", str(run)]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=900)
    (root / "reports/SERVICE_INDEPENDENT_VERIFICATION.stdout.json").write_text(result.stdout, encoding="utf-8")
    (root / "reports/SERVICE_INDEPENDENT_VERIFICATION.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Additional independent artifact verification failed; inspect service verification logs")
    verification = json.loads(result.stdout)
    if verification.get("status") != "PASS_INDEPENDENT_ARTIFACT_VERIFICATION":
        raise ValueError("Additional verifier did not pass")
    if abs(verification["metrics"]["macro_f1"] - marker["selected_validation_macro_f1"]) > 1e-12:
        raise ValueError("Independent Macro F1 differs from actual completion")
    atomic(root / "reports/SERVICE_INDEPENDENT_VERIFICATION.json", verification)
    return verification


def main(root):
    root = Path(root).resolve()
    (root / "outputs").mkdir(exist_ok=True)
    (root / "reports/execution_attempts").mkdir(parents=True, exist_ok=True)
    service = {"status": "SERVICE_RESERVED_ONCE", "timestamp_utc": now(),
               "pid": os.getpid(), "experiment_id": EXPERIMENT,
               "source_sha256": sha(Path(__file__)), "automatic_retry": False}
    # Exclusive permanent reservation prevents concurrent wrappers and reruns.
    with (root / "outputs/SERVICE_STARTED.json").open("x", encoding="utf-8") as handle:
        json.dump(service, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    for name in ROOT_DOCS:
        original = root / name
        if original.is_file():
            shutil.copyfile(original, root / "reports/execution_attempts" / ("service_before_" + stamp + "_" + name))
    child = None
    try:
        publish(root, observe(root))
        # Same executable, audited implementation and flags; inherited process
        # environment is supplied by launchd/caffeinate, never rewritten here.
        child = subprocess.Popen([str(root / ".venv/bin/python"), "-u", str(root / "scripts/train_bert.py"), "--execute"], cwd=root)
        atomic(root / "outputs/SERVICE_PROCESS.json", {**service, "child_pid": child.pid, "child_command": [str(root / ".venv/bin/python"), "-u", "scripts/train_bert.py", "--execute"]})
        while child.poll() is None:
            publish(root, observe(root))
            time.sleep(30)
        exit_code = child.returncode
        if exit_code != 0:
            raise RuntimeError("Audited training process exited with code " + str(exit_code))
        marker = read_json(root / "outputs/COMPLETED.json")
        if not marker:
            raise RuntimeError("Training exited without a genuine complete artifact marker")
        verification = independently_verify(root, marker)
        complete = {**marker, "success": True, "timestamp_utc": now(),
                    "service_status": "COMPLETE_AND_INDEPENDENTLY_VERIFIED",
                    "service_independent_verification_passed": True,
                    "MPS_optimizer_progress_observed": True,
                    "new_validation_macro_f1": verification["metrics"]["macro_f1"],
                    "best_checkpoint_independently_verified": True,
                    "COMPLETE_ARTIFACTS_zip_verified": True,
                    "held_out_data_accessed": False}
        publish(root, complete, verification)
        atomic(root / "outputs/SERVICE_EXIT.json", {"status": "SUCCESS", "timestamp_utc": now(), "child_exit_code": 0})
        return 0
    except BaseException as exc:
        status = observe(root)
        status.update({"status": "FAILED_OR_BLOCKED_ACTUAL_EXECUTION", "timestamp_utc": now(),
                       "success": False, "failure": {"exception_type": type(exc).__name__, "message": str(exc),
                        "child_exit_code": child.poll() if child else None,
                        "preflight": read_json(root / "reports/LAST_EXECUTION_PREFLIGHT.json")}})
        publish(root, status)
        atomic(root / "outputs/SERVICE_EXIT.json", {"status": status["status"], "timestamp_utc": now(), "failure": status["failure"]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(main(args.root))
