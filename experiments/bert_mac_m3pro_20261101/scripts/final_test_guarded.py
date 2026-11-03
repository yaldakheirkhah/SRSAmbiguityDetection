"""INERT by default: one future explicitly authorized frozen-model evaluation.

Do not invoke during development. There is deliberately no default dataset path.
No held-out path is resolved, inspected, hashed or opened before an explicit
authorization, a bound frozen-model manifest and a single-use lock pass.
This script does not authorize itself, freeze a model, train or select a model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata


EXPERIMENT_ID = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
ACTION = "AUTHORIZE_ONE_FINAL_HELD_OUT_EVALUATION"
CLASS_NAMES = ["Lexical Ambiguity", "Syntactic Ambiguity", "Semantic Ambiguity",
               "Syntax Ambiguity", "Pragmatic Ambiguity", "Clean"]
REQUIRED_FROZEN = {
    "best_model/model.safetensors", "best_model/config.json",
    "best_model/tokenizer_config.json", "best_model/special_tokens_map.json",
    "class_mapping.json", "run_identity.json", "selected_checkpoint.json",
    "checkpoint_replay.json", "independent_verification.json", "data_manifest.json",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_json(path, description):
    # Control arguments cannot be redirected to a CSV even by mistake.
    path = Path(path)
    if path.suffix.casefold() != ".json" or path.is_symlink():
        raise PermissionError(description + " must be a separate regular JSON control file")
    if path.stat().st_size > 1_000_000:
        raise PermissionError(description + " is unexpectedly large")
    return json.loads(path.read_text(encoding="utf-8"))


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise PermissionError("Authorization timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def authorize(authorization_path, freeze_manifest_path, dataset_argument, execute=False):
    """No dataset filesystem operation occurs in this function."""
    if not execute or not authorization_path or not freeze_manifest_path or not dataset_argument:
        raise PermissionError("REFUSED: separate explicit authorization and frozen-model manifest are required; no dataset accessed")
    authorization = _control_json(authorization_path, "Authorization")
    if (authorization.get("schema_version") != 1 or authorization.get("action") != ACTION
            or authorization.get("authorized") is not True
            or authorization.get("experiment_id") != EXPERIMENT_ID
            or authorization.get("dataset_role") != "held_out_test"
            or authorization.get("expected_rows") != 1015
            or re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("expected_dataset_sha256", ""))) is None
            or not str(authorization.get("authorized_by", "")).strip()
            or not str(authorization.get("evaluation_id", "")).strip()):
        raise PermissionError("REFUSED: invalid explicit one-evaluation authorization")
    now = datetime.now(timezone.utc)
    if not _utc(authorization["authorized_at_utc"]) <= now < _utc(authorization["expires_at_utc"]):
        raise PermissionError("REFUSED: authorization is future-dated or expired")
    # os.path.abspath is lexical: it does not stat/resolve/open the dataset.
    requested = os.path.abspath(os.fspath(dataset_argument))
    if not os.path.isabs(authorization.get("dataset_absolute_path", "")) or requested != authorization["dataset_absolute_path"]:
        raise PermissionError("REFUSED: explicit authorized dataset path differs")
    frozen = _control_json(freeze_manifest_path, "Frozen-model manifest")
    expected_freeze_sha = authorization.get("model_freeze_manifest_sha256")
    if not isinstance(expected_freeze_sha, str) or len(expected_freeze_sha) != 64 or _sha(freeze_manifest_path) != expected_freeze_sha:
        raise PermissionError("REFUSED: frozen-model manifest identity differs from authorization")
    if (frozen.get("status") != "FROZEN_AFTER_INDEPENDENT_VALIDATION_VERIFICATION"
            or frozen.get("experiment_id") != EXPERIMENT_ID
            or frozen.get("held_out_evaluation_performed") is not False):
        raise PermissionError("REFUSED: an independently verified, unused frozen Mac model is required")
    if authorization.get("evaluation_script_sha256") != _sha(Path(__file__)):
        raise PermissionError("REFUSED: authorized evaluation source identity differs")
    return authorization, frozen


def verify_frozen_model(frozen):
    run = Path(frozen["run_directory"]).resolve(strict=True)
    identities = frozen.get("files_sha256", {})
    if not REQUIRED_FROZEN.issubset(identities):
        raise PermissionError("Frozen manifest is missing model/verification identities")
    if not ({"best_model/tokenizer.json", "best_model/vocab.txt"} & set(identities)):
        raise PermissionError("Frozen tokenizer identity missing")
    for relative, expected in identities.items():
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "test.csv" in [v.casefold() for v in parts.parts]:
            raise PermissionError("Unsafe frozen-model member path")
        # Only model/validation-control artifacts belong in a freeze manifest.
        if relative not in REQUIRED_FROZEN and not relative.startswith("best_model/"):
            raise PermissionError("Unsupported frozen-model control member: " + relative)
        path = run.joinpath(*parts.parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(run) or _sha(path) != expected:
            raise PermissionError("Frozen checkpoint/control file changed: " + relative)
    actual_model_files = {p.relative_to(run).as_posix() for p in (run / "best_model").rglob("*") if p.is_file()}
    if actual_model_files != {name for name in identities if name.startswith("best_model/")}:
        raise PermissionError("Every actual model/tokenizer file must be frozen")
    def control(name):
        return json.loads((run / name).read_text(encoding="utf-8"))
    verification = control("independent_verification.json")
    replay = control("checkpoint_replay.json")
    identity = control("run_identity.json")
    mapping = control("class_mapping.json")
    config = control("best_model/config.json")
    if verification.get("status") != "PASS_INDEPENDENT_ARTIFACT_VERIFICATION" or replay.get("status") != "PASS":
        raise PermissionError("Frozen validation verification or fresh-checkpoint replay has not passed")
    if identity.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("Wrong frozen run identity")
    if mapping.get("model_index_to_dataset_id") != [1, 2, 3, 4, 5, 6]:
        raise PermissionError("Frozen class-index order differs")
    if config.get("id2label") != {str(i): name for i, name in enumerate(CLASS_NAMES)}:
        raise PermissionError("Frozen model label order differs")
    if config.get("label2id") != {name: i for i, name in enumerate(CLASS_NAMES)}:
        raise PermissionError("Frozen model reverse labels differ")
    return run, identity["training_config"]


def _metrics(truth, predicted):
    matrix = [[0] * 6 for _ in range(6)]
    for a, b in zip(truth, predicted):
        matrix[a - 1][b - 1] += 1
    supports = [sum(row) for row in matrix]
    predicted_counts = [sum(row[i] for row in matrix) for i in range(6)]
    result = {"accuracy": sum(matrix[i][i] for i in range(6)) / len(truth),
              "class_order": [1, 2, 3, 4, 5, 6], "confusion_matrix": matrix,
              "per_class": {}, "zero_division": 0}
    for i in range(6):
        p = matrix[i][i] / predicted_counts[i] if predicted_counts[i] else 0.0
        r = matrix[i][i] / supports[i] if supports[i] else 0.0
        f = 2 * matrix[i][i] / (supports[i] + predicted_counts[i]) if supports[i] + predicted_counts[i] else 0.0
        result["per_class"][str(i + 1)] = {"precision": p, "recall": r, "f1": f, "support": supports[i]}
    for key in ("precision", "recall", "f1"):
        values = [result["per_class"][str(i + 1)][key] for i in range(6)]
        result["macro_" + key] = sum(values) / 6
        result["weighted_" + key] = sum(a * n for a, n in zip(values, supports)) / len(truth)
    return result


def execute(authorization_path=None, freeze_manifest_path=None, dataset_argument=None,
            output_dir=None, execute_authorized=False, device_name="mps"):
    authorization, frozen = authorize(authorization_path, freeze_manifest_path, dataset_argument, execute_authorized)
    run, recipe = verify_frozen_model(frozen)
    # Development identities are checked before any held-out data operation.
    # This verifier accepts only the two packaged frozen development CSVs.
    from verify_inputs import verify_inputs
    development = verify_inputs(Path(__file__).resolve().parents[1])
    if development["input_identity"] != frozen["input_identity"] or development["config"] != recipe:
        raise PermissionError("Frozen development identity or complete inference/training configuration differs")
    development_ids = {row["source_id"] for key in ["train_rows", "validation_rows"] for row in development[key]}
    if not output_dir:
        raise PermissionError("A new, separate final-evaluation output directory is required")
    output = Path(output_dir).absolute()
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise PermissionError("Evaluation output ancestors must not be symlinks")
    output = output.resolve(strict=False)
    if output.exists() or output == run or output.is_relative_to(run):
        raise PermissionError("Final evaluation must have a new separate output directory")
    # This is intentionally unavailable during development and imports no ML
    # packages before explicit authorization/freeze checks.
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if device_name == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable automatic MPS-to-CPU operator fallback before authorized evaluation")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU fallback is not automatic")
    if device_name not in ("mps", "cpu"):
        raise ValueError("Only explicitly selected MPS or CPU inference is supported")
    device = torch.device(device_name)
    torch.manual_seed(int(recipe["seed"]))
    if device_name == "mps":
        torch.mps.manual_seed(int(recipe["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        str(run / "best_model"), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation=recipe.get("attention_implementation", "eager"),
        output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Frozen checkpoint is incomplete")
    tokenizer = AutoTokenizer.from_pretrained(str(run / "best_model"), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    model.float().to(device).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Frozen inference must retain FP32 parameters")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "numpy": np.__version__, "device": device_name,
                   "precision": "FP32 forward; CPU FP64 softmax", "attention": recipe["attention_implementation"],
                   "eval_batch_size": recipe["eval_batch_size"], "max_length": recipe["max_length"],
                   "seed": recipe["seed"], "versions": {name: importlib.metadata.version(name) for name in
                       ["torch", "transformers", "tokenizers", "numpy", "safetensors", "scikit-learn", "matplotlib"]},
                   "environment_controls": {name: os.environ.get(name) for name in
                       ["PYTHONHASHSEED", "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"]}}
    # A single-use lock is retained on success OR failure. It must not be deleted
    # to repeatedly evaluate. A new explicit approval is needed after a failure.
    # Run-level location also prevents copied authorization files from bypassing
    # the single-use condition for this frozen experiment.
    lock = run.parent / ("FINAL_EVALUATION_" + EXPERIMENT_ID + ".USED.json")
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"status": "AUTHORIZED_EVALUATION_STARTED_OR_CONSUMED",
                   "evaluation_id": authorization["evaluation_id"],
                   "experiment_id": EXPERIMENT_ID,
                   "authorization_sha256": _sha(authorization_path),
                   "started_utc": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
    output.mkdir(parents=True, exist_ok=False)
    try:
        # FIRST operation on the requested dataset, after all authorization,
        # checkpoint, device and one-use checks above. No default path exists.
        dataset = Path(dataset_argument)
        if dataset.is_symlink():
            raise PermissionError("Authorized data must be a regular, unlinked file")
        with dataset.open("rb") as handle:
            raw_data = handle.read()
        observed_sha256 = hashlib.sha256(raw_data).hexdigest()
        if observed_sha256 != authorization["expected_dataset_sha256"]:
            raise ValueError("Authorized held-out bytes differ from the pre-existing recorded split checksum; no inference performed")
        reader = csv.DictReader(io.StringIO(raw_data.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != ["source_id", "Text", "Class"]:
            raise ValueError("Authorized frozen data schema differs")
        rows = list(reader)
        if len(rows) != authorization["expected_rows"] or len({r["source_id"] for r in rows}) != len(rows):
            raise ValueError("Authorized data count/unique IDs differ")
        truth = [int(row["Class"]) for row in rows]
        if any(c not in range(1, 7) for c in truth) or any(not r["source_id"] or not r["Text"] for r in rows):
            raise ValueError("Invalid authorized data labels/records")
        for row in rows:
            key = " ".join(unicodedata.normalize("NFKC", row["Text"]).split()).casefold()
            if re.fullmatch(r"[0-9a-f]{16}", row["source_id"]) is None or hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] != row["source_id"]:
                raise ValueError("Authorized stable source identifier does not match original content-derived identity")
        if {row["source_id"] for row in rows} & development_ids:
            raise ValueError("Held-out source IDs overlap frozen development data; no final inference performed")
        ordered_ids_sha256 = hashlib.sha256(("\n".join(r["source_id"] for r in rows) + "\n").encode("utf-8")).hexdigest()
        ordered_rows_sha256 = hashlib.sha256(json.dumps(
            [{"source_id": r["source_id"], "Text": r["Text"], "Class": int(r["Class"])} for r in rows],
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        with (output / "data_identity_before_inference.json").open("x", encoding="utf-8") as handle:
            json.dump({"status": "PASS_AUTHORIZED_FROZEN_DATA_IDENTITY_BEFORE_INFERENCE", "sha256": observed_sha256,
                       "rows": len(rows), "ordered_source_ids_sha256": ordered_ids_sha256,
                       "ordered_source_rows_sha256": ordered_rows_sha256,
                       "timestamp_utc": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
        parts = []
        seen = 0
        with torch.inference_mode():
            for start in range(0, len(rows), int(recipe["eval_batch_size"])):
                batch = rows[start:start + int(recipe["eval_batch_size"])]
                encoded = tokenizer([row["Text"] for row in batch], padding=True, truncation=True,
                                    max_length=int(recipe["max_length"]), return_tensors="pt")
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
                if logits.dtype != torch.float32:
                    raise RuntimeError("Final inference must use FP32 without autocast")
                parts.append(logits.cpu().numpy())
                seen += len(batch)
                with (output / "inference_batches.jsonl").open("a", encoding="utf-8") as journal:
                    journal.write(json.dumps({"start_row_inclusive": start, "end_row_exclusive": start+len(batch),
                        "rows": len(batch), "cumulative_rows_inferred": seen,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()})+"\n")
        logits = np.concatenate(parts).astype(np.float32, copy=False)
        if logits.shape != (1015, 6) or not np.isfinite(logits).all():
            raise ValueError("Invalid frozen-model final logits")
        probabilities = logits.astype(np.float64)
        probabilities -= probabilities.max(axis=1, keepdims=True)
        probabilities = np.exp(probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predicted = (probabilities.argmax(axis=1) + 1).tolist()
        metrics = _metrics(truth, predicted)
        # Hash checks only: no second model forward or second source-data read.
        verify_frozen_model(frozen)
        np.save(output / "final_logits_fp32.npy", logits, allow_pickle=False)
        np.save(output / "final_probabilities_fp64.npy", probabilities, allow_pickle=False)
        with (output / "final_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
            fields = ["source_id", "Text", "true_label", "predicted_label", *[f"p_class_{c}" for c in range(1, 7)]]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, target, prediction, p in zip(rows, truth, predicted, probabilities):
                writer.writerow({"source_id": row["source_id"], "Text": row["Text"],
                                 "true_label": target, "predicted_label": prediction,
                                 **{f"p_class_{c}": format(float(p[c - 1]), ".17g") for c in range(1, 7)}})
        with (output / "final_confusion_counts.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_label", *[f"pred_{c}" for c in range(1, 7)]])
            for c, row in enumerate(metrics["confusion_matrix"], 1):
                writer.writerow([c, *row])
        with (output / "final_metrics.json").open("x", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        result = {"status": "COMPLETED_ONCE_AUTHORIZED_FINAL_EVALUATION", "experiment_id": EXPERIMENT_ID,
                  "evaluation_id": authorization["evaluation_id"], "evaluation_device": device_name,
                  "model_freeze_manifest_sha256": _sha(freeze_manifest_path),
                  "authorization_sha256": _sha(authorization_path), "evaluated_data_sha256": observed_sha256,
                  "recorded_expected_data_sha256": authorization["expected_dataset_sha256"],
                  "ordered_source_ids_sha256": ordered_ids_sha256, "ordered_source_rows_sha256": ordered_rows_sha256,
                  "source_file_byte_reads": 1, "source_rows_forwarded_once": seen,
                  "source_ids_disjoint_from_frozen_development": True,
                  "model_files_unchanged_after_inference": True,
                  "evaluation_script_sha256": _sha(Path(__file__)), "environment": environment,
                  "evaluated_rows": len(rows), "class_order": list(range(1, 7)),
                  "preprocessing": "Original raw Text with frozen tokenizer/settings only",
                  "training_or_model_selection_performed": False,
                  "metrics": metrics, "completed_utc": datetime.now(timezone.utc).isoformat()}
        with (output / "FINAL_EVALUATION_COMPLETED.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result
    except BaseException as exc:
        with (output / "FINAL_EVALUATION_FAILED.json").open("x", encoding="utf-8") as handle:
            json.dump({"status": "FAILED_AUTHORIZATION_CONSUMED_NO_AUTOMATIC_RETRY",
                       "exception_type": type(exc).__name__, "message": str(exc)}, handle, indent=2)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--dataset")  # deliberately no default and no Path conversion
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--execute-authorized-final-evaluation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.authorization, args.freeze_manifest, args.dataset,
                             args.output_dir, args.execute_authorized_final_evaluation,
                             args.device), indent=2))
