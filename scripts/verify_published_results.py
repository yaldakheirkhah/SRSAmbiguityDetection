"""Aggregate arithmetic only. Never opens datasets, predictions or checkpoints."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [1, 2, 3, 4, 5, 6]
NAMES = ["Lexical Ambiguity", "Syntactic Ambiguity", "Semantic Ambiguity",
         "Syntax Ambiguity", "Pragmatic Ambiguity", "Clean"]


def metrics_from_matrix(matrix):
    if (not isinstance(matrix, list) or len(matrix) != 6
            or any(not isinstance(row, list) or len(row) != 6 for row in matrix)
            or any(type(v) is not int or v < 0 for row in matrix for v in row)):
        raise ValueError("Expected six by six nonnegative integer aggregate counts")
    supports = [sum(row) for row in matrix]
    total = sum(supports)
    if not total:
        raise ValueError("Empty confusion matrix")
    per_class = {}
    for i in range(6):
        tp = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        p = tp / predicted if predicted else 0.0
        r = tp / supports[i] if supports[i] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[str(i + 1)] = {"precision": p, "recall": r, "f1": f, "support": supports[i]}
    result = {"accuracy": sum(matrix[i][i] for i in range(6)) / total, "per_class": per_class}
    for key in ("precision", "recall", "f1"):
        result["macro_" + key] = sum(c[key] for c in per_class.values()) / 6
        result["weighted_" + key] = sum(c[key] * c["support"] for c in per_class.values()) / total
    return result


def compare_metrics(reported, calculated):
    for key, expected in calculated.items():
        if key == "per_class":
            for cls, values in expected.items():
                compare_metrics(reported[key][cls], values)
        else:
            actual = reported[key]
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
                raise ValueError("Invalid metric: " + key)
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Aggregate metric disagreement: " + key)


def verify():
    # Fixed, code-controlled aggregate paths only. No data/model argument exists.
    result = json.loads((ROOT / "results/frozen_bert_summary.json").read_text())
    if result["dataset_class_ids"] != ORDER or result["model_indices"] != list(range(6)) or result["class_names"] != NAMES:
        raise ValueError("Incorrect fixed class order")
    if (result["training_rows"], result["validation_rows"], result["held_out_rows"], result["augmented_training_rows"]) != (4731, 1015, 1015, 0):
        raise ValueError("Unexpected recorded experiment identity")
    for key in ("validation", "final_test"):
        record = result[key]
        if record["class_order"] != ORDER or sum(map(sum, record["confusion_matrix"])) != 1015:
            raise ValueError("Wrong aggregate class order or count: " + key)
        compare_metrics(record, metrics_from_matrix(record["confusion_matrix"]))
    with (ROOT / "results/mac_training_history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != result["completed_epochs"] or [int(r["epoch"]) for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Missing/reordered epoch history")
    for row in history:
        for key in ("train_weighted_loss", "val_macro_f1", "val_loss"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Nonfinite recorded history")
    best = max(history, key=lambda row: float(row["val_macro_f1"]))
    if int(best["epoch"]) != result["best_epoch"] or not math.isclose(float(best["val_macro_f1"]), result["validation"]["macro_f1"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Checkpoint selection disagrees with literal history maximum")
    updates = sum(int(row["successful_optimizer_steps"]) for row in history)
    if updates != 14504 or any(int(row["skipped_optimizer_steps"]) for row in history):
        raise ValueError("Unexpected optimizer update history")
    return {"status": "PASS_AGGREGATE_ARITHMETIC_AND_SELECTION", "rows_per_partition": 1015,
            "epochs": len(history), "best_epoch": result["best_epoch"], "optimizer_updates": updates,
            "validation_macro_f1": result["validation"]["macro_f1"],
            "final_test_macro_f1": result["final_test"]["macro_f1"],
            "source_datasets_read": False, "model_inference_run": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
