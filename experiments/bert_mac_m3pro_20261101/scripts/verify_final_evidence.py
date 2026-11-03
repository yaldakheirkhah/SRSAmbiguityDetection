"""Independently audit EXPORTED final predictions; never load a model/dataset.

Only named artifacts inside the supplied completed output directory are read.
The held-out source CSV is never discovered, opened, hashed or evaluated here.
"""
from __future__ import annotations

import argparse
import array
import ast
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

ORDER = [1, 2, 3, 4, 5, 6]
CLASS_NAMES = ["Lexical Ambiguity", "Syntactic Ambiguity", "Semantic Ambiguity",
               "Syntax Ambiguity", "Pragmatic Ambiguity", "Clean"]
EXPERIMENT = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
ORIGINAL_EXPORTS = ["final_predictions.csv", "final_logits_fp32.npy",
    "final_probabilities_fp64.npy", "final_metrics.json", "final_confusion_counts.csv",
    "FINAL_EVALUATION_COMPLETED.json"]
METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
               "weighted_precision", "weighted_recall", "weighted_f1"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member(root, name):
    # Deliberately accepts no arbitrary dataset path or recursive discovery.
    part = Path(name)
    require(not part.is_absolute() and ".." not in part.parts
            and not any(p.casefold() == "test.csv" for p in part.parts), "Unsafe artifact path")
    path = root / part
    for end in range(1, len(part.parts) + 1):
        require(not root.joinpath(*part.parts[:end]).is_symlink(), "Linked evidence is prohibited")
    require(path.resolve().is_relative_to(root), "Artifact escapes output directory")
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def metrics_from_exported_labels(truth, predicted):
    """Separate count-based implementation, not the evaluator's metric function."""
    require(len(truth) == len(predicted) and len(truth) > 0, "Invalid exported cardinality")
    require(all(type(x) is int and x in ORDER for x in truth + predicted), "Labels must be integer IDs1..6")
    actual = Counter(truth)
    assigned = Counter(predicted)
    joint = Counter(zip(truth, predicted))
    per_class = {}
    for label in ORDER:
        tp = joint[label, label]
        precision = tp / assigned[label] if assigned[label] else 0.0
        recall = tp / actual[label] if actual[label] else 0.0
        denominator = actual[label] + assigned[label]
        f1 = 2 * tp / denominator if denominator else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall,
                                 "f1": f1, "support": actual[label]}
    metrics = {"accuracy": sum(joint[c, c] for c in ORDER) / len(truth),
               "class_order": ORDER, "zero_division": 0,
               "confusion_matrix": [[joint[a, b] for b in ORDER] for a in ORDER],
               "per_class": per_class}
    for key in ["precision", "recall", "f1"]:
        metrics["macro_" + key] = sum(per_class[str(c)][key] for c in ORDER) / 6
        metrics["weighted_" + key] = sum(per_class[str(c)][key] * actual[c] for c in ORDER) / len(truth)
    return metrics


def numeric_equal(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12, message)


def check_metric_report(stored, recomputed):
    require(stored.get("class_order") == ORDER and stored.get("zero_division") == 0,
            "Reported metric class order/zero division differs")
    for key in METRIC_KEYS:
        numeric_equal(stored.get(key), recomputed[key], "Reported metric differs: " + key)
    matrix = stored.get("confusion_matrix")
    require(isinstance(matrix, list) and len(matrix) == 6
            and all(isinstance(row, list) and len(row) == 6
                    and all(type(value) is int and value >= 0 for value in row) for row in matrix),
            "Reported confusion must contain integer counts")
    require(matrix == recomputed["confusion_matrix"], "Reported confusion counts differ")
    require(set(stored.get("per_class", {})) == {str(c) for c in ORDER}, "Per-class report order/domain differs")
    for label in ORDER:
        for key in ["precision", "recall", "f1", "support"]:
            numeric_equal(stored["per_class"][str(label)].get(key), recomputed["per_class"][str(label)][key],
                          "Reported per-class result differs")


def read_numeric_npy(path, rows, dtype):
    """Bounded stdlib parser: no object arrays, NumPy import or pickle loading."""
    require(dtype in {"<f4", "<f8"}, "Only bounded floating-point exports are supported")
    width = 4 if dtype == "<f4" else 8
    with path.open("rb") as handle:
        require(handle.read(6) == b"\x93NUMPY", "Invalid NPY magic")
        version = tuple(handle.read(2))
        require(version in {(1, 0), (2, 0)}, "Unsupported NPY version")
        count_width = 2 if version == (1, 0) else 4
        count_bytes = handle.read(count_width)
        require(len(count_bytes) == count_width, "Truncated NPY header")
        length = int.from_bytes(count_bytes, "little")
        require(0 < length <= 4096, "Unbounded NPY header")
        header = ast.literal_eval(handle.read(length).decode("latin1"))
        require(header.get("descr") == dtype and header.get("shape") == (rows, 6)
                and header.get("fortran_order") is False, "NPY dtype/shape/order differs")
        data = handle.read(rows * 6 * width + 1)
    require(len(data) == rows * 6 * width, "Truncated/extra NPY payload")
    values = array.array("f" if width == 4 else "d")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(x) for x in values), "Nonfinite exported logits/probabilities")
    return [list(values[start:start + 6]) for start in range(0, len(values), 6)]


def _write_immutable(path, data):
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == data,
                "Existing derived evidence differs; refusing overwrite: " + path.name)
    else:
        with path.open("xb") as handle:
            handle.write(data)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def verify_final_evidence(output_dir, expected_ids_sha256=None):
    candidate = Path(output_dir)
    require(candidate.suffix.casefold() != ".csv" and not any(p.casefold() == "test.csv" for p in candidate.parts),
            "Only an exported-output directory is accepted, never a source dataset")
    root = candidate.resolve(strict=True)
    require(root.is_dir() and not any(p.casefold() == "test.csv" for p in root.parts), "Expected completed output directory")
    for name in ORIGINAL_EXPORTS:
        path = member(root, name)
        require(path.is_file() and path.stat().st_size > 0, "Missing actual final export: " + name)
    completion = json.loads(member(root, "FINAL_EVALUATION_COMPLETED.json").read_text())
    require(completion.get("status") == "COMPLETED_ONCE_AUTHORIZED_FINAL_EVALUATION"
            and completion.get("experiment_id") == EXPERIMENT
            and completion.get("evaluated_rows") == 1015 and completion.get("class_order") == ORDER
            and completion.get("training_or_model_selection_performed") is False,
            "Genuine once-authorized completed final evaluation is required")
    for key in ["model_freeze_manifest_sha256", "authorization_sha256", "evaluated_data_sha256"]:
        require(re.fullmatch(r"[0-9a-f]{64}", completion.get(key, "")) is not None, "Missing evaluation identity: " + key)
    stamp = datetime.fromisoformat(completion["completed_utc"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "Evaluation timestamp lacks timezone")
    fields = ["source_id", "Text", "true_label", "predicted_label", *[f"p_class_{c}" for c in ORDER]]
    with member(root, "final_predictions.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, "Prediction schema/class order differs")
        rows = list(reader)
    require(len(rows) == 1015, "Final prediction cardinality differs")
    logits = read_numeric_npy(member(root, "final_logits_fp32.npy"), 1015, "<f4")
    probabilities = read_numeric_npy(member(root, "final_probabilities_fp64.npy"), 1015, "<f8")
    truth, predicted, ids, original_rows = [], [], [], []
    maximum_softmax_error = 0.0
    for index, row in enumerate(rows):
        require(set(row) == set(fields) and None not in row.values(), "Missing/extra exported prediction fields")
        sid, text = row["source_id"], row["Text"]
        require(re.fullmatch(r"[0-9a-f]{16}", sid) is not None and sid not in ids
                and bool(text.strip()), "Invalid/duplicate exported requirement identifier or blank text")
        require(hashlib.sha256(normalized(text).encode()).hexdigest()[:16] == sid,
                "Exported requirement ID does not match its recorded text")
        require(row["true_label"] in [str(c) for c in ORDER] and row["predicted_label"] in [str(c) for c in ORDER],
                "Exported label mapping differs")
        probability = [float(row[f"p_class_{c}"]) for c in ORDER]
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in probability)
                and abs(sum(probability) - 1) <= 1e-12, "Invalid six-class probabilities")
        require(probability == probabilities[index], "CSV probabilities do not round-trip the FP64 export")
        estimate = max(range(6), key=lambda c: probability[c]) + 1
        require(int(row["predicted_label"]) == estimate, "Predicted class is not first-argmax in fixed class order")
        shift = max(logits[index])
        exponent = [math.exp(x - shift) for x in logits[index]]
        total = sum(exponent)
        error = max(abs(a - b / total) for a, b in zip(probability, exponent))
        require(error <= 1e-12, "Exported probabilities and FP32 logits disagree")
        maximum_softmax_error = max(maximum_softmax_error, error)
        ids.append(sid); truth.append(int(row["true_label"])); predicted.append(estimate)
        original_rows.append({"source_id": sid, "Text": text, "Class": int(row["true_label"])})
    ids_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    rows_sha = hashlib.sha256(json.dumps(original_rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if expected_ids_sha256 is not None:
        require(ids_sha == expected_ids_sha256, "Exported requirement ID order differs from the supplied single-read identity")
    if "ordered_source_ids_sha256" in completion:
        require(ids_sha == completion["ordered_source_ids_sha256"], "Exported ID order differs from authorized source read")
    if "ordered_source_rows_sha256" in completion:
        require(rows_sha == completion["ordered_source_rows_sha256"], "Exported IDs/text/true labels differ from authorized source read")
    metrics = metrics_from_exported_labels(truth, predicted)
    stored = json.loads(member(root, "final_metrics.json").read_text())
    check_metric_report(stored, metrics)
    check_metric_report(completion["metrics"], metrics)
    with member(root, "final_confusion_counts.csv").open(newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    require(table and table[0] == ["true_label", *[f"pred_{c}" for c in ORDER]]
            and len(table) == 7 and all(len(row) == 7 for row in table), "Confusion schema/order differs")
    require([int(row[0]) for row in table[1:]] == ORDER
            and [[int(x) for x in row[1:]] for row in table[1:]] == metrics["confusion_matrix"],
            "Saved raw confusion matrix differs from independent counts")
    # Additional independent library cross-check. Imports no torch/model stack.
    import sklearn
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    numeric_equal(float(accuracy_score(truth, predicted)), metrics["accuracy"], "sklearn accuracy differs")
    require(confusion_matrix(truth, predicted, labels=ORDER).tolist() == metrics["confusion_matrix"], "sklearn confusion differs")
    for average in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(truth, predicted, labels=ORDER, average=average, zero_division=0)
        for key, value in [("precision", p), ("recall", r), ("f1", f)]:
            numeric_equal(float(value), metrics[average + "_" + key], "sklearn aggregate differs")
    classification = classification_report(truth, predicted, labels=ORDER, output_dict=True, zero_division=0)
    for label in ORDER:
        for report_key, key in [("precision", "precision"), ("recall", "recall"), ("f1-score", "f1"), ("support", "support")]:
            numeric_equal(classification[str(label)][report_key], metrics["per_class"][str(label)][key], "sklearn per-class differs")
    text_report = "FINAL HELD-OUT TEST — fixed six-class order; no model inference repeated\n\n" + classification_report(
        truth, predicted, labels=ORDER, target_names=CLASS_NAMES, digits=12, zero_division=0)
    derived = ["final_classification_report.json", "final_classification_report.txt", "final_confusion_row_normalized.csv"]
    _write_immutable(member(root, derived[0]), _json_bytes(classification))
    _write_immutable(member(root, derived[1]), text_report.encode())
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["true_label", *[f"pred_{c}" for c in ORDER]])
    for label, counts in zip(ORDER, metrics["confusion_matrix"]):
        support = metrics["per_class"][str(label)]["support"]
        writer.writerow([label, *[format(value / support if support else 0.0, ".17g") for value in counts]])
    _write_immutable(member(root, derived[2]), buffer.getvalue().encode())
    result = {"status": "PASS_INDEPENDENT_FINAL_EVIDENCE_VERIFICATION", "dataset_role": "final_held_out_test",
        "experiment_id": EXPERIMENT, "evaluation_id": completion["evaluation_id"], "evaluated_rows": 1015,
        "evaluation_completed_utc": completion["completed_utc"],
        "evaluated_data_sha256": completion["evaluated_data_sha256"],
        "model_freeze_manifest_sha256": completion["model_freeze_manifest_sha256"],
        "authorization_sha256": completion["authorization_sha256"],
        "evaluation_device": completion.get("evaluation_device"),
        "metrics": metrics, "class_names": CLASS_NAMES, "class_order": ORDER,
        "ordered_exported_ids_sha256": ids_sha, "ordered_exported_source_rows_sha256": rows_sha,
        "single_read_source_ids_bound": "ordered_source_ids_sha256" in completion or expected_ids_sha256 is not None,
        "single_read_source_rows_labels_bound": "ordered_source_rows_sha256" in completion,
        "maximum_logits_softmax_abs_error": maximum_softmax_error,
        "independent_methods": ["Counter-based integer confusion arithmetic", "sklearn metrics cross-check"],
        "sklearn_version": sklearn.__version__, "source_dataset_reopened": False, "model_inference_repeated": False,
        "training_or_model_selection_performed": False, "verifier_source_sha256": digest(Path(__file__)),
        "artifact_sha256": {name: digest(member(root, name)) for name in ORIGINAL_EXPORTS + derived}}
    destination = member(root, "independent_final_verification.json")
    if destination.exists():
        require(json.loads(destination.read_text()) == result, "Existing independent final verification differs")
    else:
        _write_immutable(destination, _json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ids-sha256")
    args = parser.parse_args()
    print(json.dumps(verify_final_evidence(args.output_dir, args.expected_ids_sha256), indent=2))
