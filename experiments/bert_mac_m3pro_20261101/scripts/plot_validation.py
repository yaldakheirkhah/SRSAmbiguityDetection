"""Create validation figures from genuine, independently verified run outputs.

This module has no model training, dataset discovery or held-out input path.
No figure is produced when a complete verified validation run is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


EXPERIMENT_ID = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
CLASS_NAMES = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"]
SOURCES = (
    "epoch_metrics.csv", "validation_metrics.json", "classification_report.json",
    "confusion_matrix.csv", "selected_checkpoint.json", "independent_verification.json",
    "run_identity.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_recorded_results(run_dir):
    """Validate recorded counts and history before importing any plotting library."""
    run = Path(run_dir).resolve(strict=True)
    if any(p.casefold() == "test.csv" for p in run.parts):
        raise ValueError("Held-out input is not accepted by this validation-only tool")
    for name in SOURCES:
        path = run / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Actual complete validation evidence required: " + name)
    identity = _json(run / "run_identity.json")
    if identity.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Figures must retain the separate Mac experiment identity")
    verification = _json(run / "independent_verification.json")
    if verification.get("status") != "PASS_INDEPENDENT_ARTIFACT_VERIFICATION":
        raise ValueError("Independent validation artifact verification has not passed")
    identities = verification.get("artifact_sha256", {})
    for name in SOURCES:
        if name == "independent_verification.json":
            continue
        if identities.get(name) != _sha(run / name):
            raise ValueError("Figure source changed or is not bound by independent verification: " + name)
    metrics = _json(run / "validation_metrics.json")
    selected = _json(run / "selected_checkpoint.json")
    report = _json(run / "classification_report.json")
    with (run / "epoch_metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
        history = list(csv.DictReader(handle))
    if not history or [int(r["epoch"]) for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Complete sequential actual epoch history is required")
    for row in history:
        for field in ("train_weighted_loss", "val_macro_f1"):
            if not math.isfinite(float(row[field])):
                raise ValueError("Non-finite recorded epoch value")
        if not 0 <= float(row["val_macro_f1"]) <= 1 or float(row["train_weighted_loss"]) < 0:
            raise ValueError("Invalid recorded epoch loss/score")
    best = max(history, key=lambda row: float(row["val_macro_f1"]))
    if int(best["epoch"]) != selected["epoch"]:
        raise ValueError("Selected epoch is not the earliest literal maximum")
    if abs(float(best["val_macro_f1"]) - metrics["macro_f1"]) > 1e-12:
        raise ValueError("Selected metrics do not agree with recorded history")
    if abs(selected["metric_value"] - metrics["macro_f1"]) > 1e-12:
        raise ValueError("Selection record differs from metrics")
    independent = verification["metrics"]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1"):
        if abs(metrics[key] - independent[key]) > 1e-12:
            raise ValueError("Independent saved-prediction metric differs: " + key)
    with (run / "confusion_matrix.csv").open(newline="", encoding="utf-8-sig") as handle:
        matrix_rows = list(csv.reader(handle))
    if len(matrix_rows) != 7 or any(len(row) != 7 for row in matrix_rows):
        raise ValueError("Expected labeled six-class confusion matrix")
    matrix = [[int(v) for v in row[1:]] for row in matrix_rows[1:]]
    if [int(row[0]) for row in matrix_rows[1:]] != list(range(1, 7)):
        raise ValueError("Confusion row label order is not 1..6")
    if matrix_rows[0] != ["true_label", *[f"pred_{c}" for c in range(1, 7)]]:
        raise ValueError("Confusion column label order is not 1..6")
    if any(v < 0 for row in matrix for v in row) or sum(map(sum, matrix)) != 1015:
        raise ValueError("Confusion matrix is not the actual full 1,015-row validation set")
    if matrix != independent["confusion_matrix"]:
        raise ValueError("Saved confusion counts differ from independent verification")
    per_class = []
    for c, name in enumerate(CLASS_NAMES, 1):
        entry = report.get(str(c), report.get(name + " Ambiguity", report.get(name)))
        expected = independent["per_class"][str(c)]
        if not isinstance(entry, dict):
            raise ValueError("Per-class report absent")
        for source_key, key in (("precision", "precision"), ("recall", "recall"),
                                ("f1-score", "f1"), ("support", "support")):
            if abs(entry[source_key] - expected[key]) > 1e-12:
                raise ValueError("Per-class report differs from independent verification")
        per_class.append(expected)
    return run, history, matrix, per_class, selected, metrics


def make_figures(run_dir, output_dir=None):
    run, history, matrix, per_class, selected, metrics = load_recorded_results(run_dir)
    output = Path(output_dir) if output_dir is not None else run / "figures"
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing publication figures")
    # Imports occur only after actual evidence has passed the checks above.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=False)
    sources = output / "source_data"
    sources.mkdir()
    for name in SOURCES:
        shutil.copyfile(run / name, sources / name)
    shutil.copyfile(Path(__file__), output / "plot_validation.py")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    outputs = []

    def save(fig, stem):
        for suffix in ("png", "pdf", "svg"):
            target = output / (stem + "." + suffix)
            fig.savefig(target, dpi=400, bbox_inches="tight", facecolor="white")
            outputs.append(target.name)
        plt.close(fig)

    epochs = [int(row["epoch"]) for row in history]
    fig, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    axis.plot(epochs, [float(row["train_weighted_loss"]) for row in history],
              marker="o", markersize=3, linewidth=1.5, color="#245b84")
    axis.set(xlabel="Epoch", ylabel="Training weighted cross-entropy",
             title="Original-only BERT — Mac MPS")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "training_weighted_loss")

    fig, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    axis.plot(epochs, [float(row["val_macro_f1"]) for row in history],
              marker="o", markersize=3, linewidth=1.5, color="#245b84")
    axis.axvline(selected["epoch"], linestyle="--", linewidth=1,
                 color="#994525", label=f"Selected epoch {selected['epoch']}")
    axis.set(xlabel="Epoch", ylabel="Validation Macro F1", ylim=(0, 1),
             title="Frozen validation — Mac MPS")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save(fig, "validation_macro_f1")

    raw = np.asarray(matrix, dtype=int)
    totals = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, totals, out=np.zeros_like(raw, dtype=float), where=totals > 0)
    with (sources / "confusion_matrix_row_normalized.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *[f"pred_{c}" for c in range(1, 7)]])
        for c, row in enumerate(normalized, 1):
            writer.writerow([c, *[format(float(v), ".17g") for v in row]])
    for values, normalized_flag, stem in ((raw, False, "validation_confusion_counts"),
                                           (normalized, True, "validation_confusion_row_normalized")):
        fig, axis = plt.subplots(figsize=(7.3, 5.8), layout="constrained")
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized_flag else None)
        for i in range(6):
            for j in range(6):
                text = f"{values[i,j]:.2f}" if normalized_flag else str(values[i,j])
                threshold = 0.5 if normalized_flag else float(values.max()) / 2
                axis.text(j, i, text, ha="center", va="center",
                          color="white" if values[i, j] > threshold else "black", fontsize=9)
        axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES,
                 yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class",
                 title="Validation confusion — " + ("row proportions" if normalized_flag else "raw counts"))
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        fig.colorbar(image, ax=axis, label="Proportion within true class" if normalized_flag else "Examples")
        save(fig, stem)

    fig, axis = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    bars = axis.bar(CLASS_NAMES, [row["f1"] for row in per_class], color="#245b84", width=0.65)
    for bar, row in zip(bars, per_class):
        axis.text(bar.get_x() + bar.get_width()/2, row["f1"] + 0.02,
                  f"{row['f1']:.3f}\nn={row['support']}", ha="center", fontsize=9)
    axis.set(ylabel="Validation F1", ylim=(0, 1.13), title="Per-class validation F1 — Mac MPS")
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "validation_per_class_f1")
    manifest = {"status": "GENERATED_FROM_ACTUAL_VERIFIED_VALIDATION_OUTPUTS",
                "experiment_id": EXPERIMENT_ID, "selected_epoch": selected["epoch"],
                "validation_rows": 1015, "validation_macro_f1": metrics["macro_f1"],
                "class_order": list(range(1, 7)), "class_names": CLASS_NAMES,
                "raw_confusion_distinct_from_row_normalized": True,
                "smoothing_or_interpolated_data_used": False,
                "held_out_evaluation_performed": False,
                "source_sha256": {name: _sha(run / name) for name in SOURCES},
                "generated_files_sha256": {name: _sha(output / name) for name in outputs},
                "matplotlib_version": matplotlib.__version__, "png_dpi": 400}
    (output / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    print(json.dumps(make_figures(arguments.run_dir, arguments.output_dir), indent=2))
