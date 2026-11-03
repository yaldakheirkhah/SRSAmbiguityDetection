"""Plot verified final held-out exports; never load a model or source dataset.

Only the explicitly named completed-evaluation exports below are accepted.
No score is calculated by inference, changed, smoothed or invented here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

EXPERIMENT_ID = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
ORDER = [1, 2, 3, 4, 5, 6]
CLASS_NAMES = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"]
FULL_NAMES = [name + " Ambiguity" for name in CLASS_NAMES[:-1]] + ["Clean"]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_precision", "weighted_recall", "weighted_f1"]
SOURCES = (
    "final_metrics.json", "final_confusion_counts.csv",
    "final_confusion_row_normalized.csv", "final_classification_report.json",
    "FINAL_EVALUATION_COMPLETED.json", "independent_final_verification.json",
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _matrix(path, normalized=False):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    _require(len(rows) == 7 and all(len(row) == 7 for row in rows),
             "Expected a labeled six-class confusion matrix")
    _require(rows[0] == ["true_label", *[f"pred_{c}" for c in ORDER]],
             "Confusion column label order is not 1..6")
    _require([row[0] for row in rows[1:]] == [str(c) for c in ORDER],
             "Confusion row label order is not 1..6")
    return [[float(value) if normalized else int(value) for value in row[1:]]
            for row in rows[1:]]


def load_recorded_results(output_dir):
    """Fail before plotting unless completed exports match independent hashes."""
    given = Path(output_dir)
    # Reject a source-data-file mistake before resolving or probing it.
    if given.suffix.casefold() in {".csv", ".tsv"} or any(p.casefold() == "test.csv" for p in given.parts):
        raise ValueError("Use the completed evaluation output directory, never a source dataset")
    if given.is_symlink():
        raise ValueError("Evaluation output directory may not be a symlink")
    root = given.resolve(strict=True)
    for name in SOURCES:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("Complete verified final exports required: " + name)
    verification = _json(root / "independent_final_verification.json")
    _require(verification.get("status") == "PASS_INDEPENDENT_FINAL_EVIDENCE_VERIFICATION",
             "Independent final saved-prediction verification has not passed")
    _require(verification.get("evaluated_rows") == 1015,
             "The verified result must cover all 1,015 held-out requirements")
    identities = verification.get("artifact_sha256", {})
    hashes = {name: _sha(root / name) for name in SOURCES}
    for name in SOURCES:
        if name != "independent_final_verification.json":
            _require(identities.get(name) == hashes[name],
                     "Plot source changed or is not hash-bound by independent verification: " + name)
    completion = _json(root / "FINAL_EVALUATION_COMPLETED.json")
    _require(completion.get("status") == "COMPLETED_ONCE_AUTHORIZED_FINAL_EVALUATION"
             and completion.get("experiment_id") == EXPERIMENT_ID,
             "The one authorized frozen-Mac evaluation has not completed")
    _require(completion.get("evaluated_rows") == 1015
             and completion.get("training_or_model_selection_performed") is False,
             "Invalid final-evaluation scope or row count")
    _require(completion.get("class_order") == ORDER,
             "Completed evaluation label order differs")
    metrics = _json(root / "final_metrics.json")
    independent = verification["metrics"]
    _require(metrics.get("class_order") == independent.get("class_order") == ORDER,
             "Final metric class order differs")
    for key in METRICS:
        _require(all(isinstance(source[key], (int, float)) and math.isfinite(source[key])
                     and 0 <= source[key] <= 1
                     for source in [metrics, independent, completion["metrics"]]),
                 "Non-finite or invalid actual final metric: " + key)
        _require(abs(metrics[key] - independent[key]) < 1e-12
                 and abs(metrics[key] - completion["metrics"][key]) < 1e-12,
                 "Actual final and independent metric reports differ: " + key)
    counts = _matrix(root / "final_confusion_counts.csv")
    _require(all(v >= 0 for row in counts for v in row)
             and sum(map(sum, counts)) == 1015,
             "Raw confusion does not represent 1,015 actual observations")
    _require(counts == independent["confusion_matrix"] == metrics["confusion_matrix"],
             "Raw confusion counts differ from independently verified predictions")
    normalized = _matrix(root / "final_confusion_row_normalized.csv", normalized=True)
    for i, row in enumerate(counts):
        support = sum(row)
        expected = [v / support if support else 0. for v in row]
        _require(all(math.isfinite(v) and abs(v - target) < 1e-12
                     for v, target in zip(normalized[i], expected)),
                 "Confusion normalization is not proportion within true class")
    report = _json(root / "final_classification_report.json")
    per_class = []
    for i, name in enumerate(FULL_NAMES, 1):
        observed = report.get(str(i), report.get(name))
        expected = independent["per_class"][str(i)]
        _require(isinstance(observed, dict), "Missing actual class report: " + name)
        _require(expected["support"] == sum(counts[i-1]), "Actual class support differs")
        for key in ["precision", "recall", "support"]:
            _require(abs(observed[key] - expected[key]) < 1e-12,
                     "Actual class report differs from independent verification")
        _require(abs(observed.get("f1-score", observed.get("f1", float("nan"))) - expected["f1"]) < 1e-12,
                 "Actual class F1 differs from independent verification")
        per_class.append(expected)
    return root, metrics, counts, normalized, per_class, hashes


def plot_final_evidence(output_dir):
    root, metrics, counts, normalized, per_class, hashes = load_recorded_results(output_dir)
    destination = root / "figures"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite final publication figures")
    # No ML imports. Plotting packages load only after verified actual evidence.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    staged = root / (".figures_staging_" + uuid.uuid4().hex)
    staged.mkdir(exist_ok=False)
    sources = staged / "source_data"
    sources.mkdir()
    for name in SOURCES:
        if _sha(root / name) != hashes[name]:
            raise ValueError("Final output changed before figure-source copying")
        shutil.copyfile(root / name, sources / name)
    shutil.copyfile(Path(__file__), staged / "plot_final_evidence.py")
    rendered = []
    try:
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "axes.spines.top": False, "axes.spines.right": False}):
            def save(figure, stem):
                for suffix in ["png", "pdf", "svg"]:
                    target = staged / (stem + "." + suffix)
                    figure.savefig(target, dpi=600, bbox_inches="tight", facecolor="white")
                    if suffix == "svg" and "<image" in target.read_text():
                        raise ValueError("Unexpected raster embedding in vector SVG")
                    if suffix == "pdf" and b"/Subtype /Image" in target.read_bytes():
                        raise ValueError("Unexpected raster embedding in vector PDF")
                    rendered.append(target.name)
                plt.close(figure)

            for values, row_normalized, stem in [
                (counts, False, "test_confusion_counts"),
                (normalized, True, "test_confusion_row_normalized")]:
                array = np.asarray(values, dtype=float)
                fig, axis = plt.subplots(figsize=(7.5, 6.1), layout="constrained")
                mesh = axis.pcolormesh(np.arange(7), np.arange(7), array,
                    shading="flat", cmap="Blues", vmin=0,
                    vmax=1 if row_normalized else float(array.max()),
                    edgecolors="white", linewidth=0.5, rasterized=False)
                axis.set_aspect("equal")
                axis.set(xlim=(0, 6), ylim=(6, 0),
                    xticks=np.arange(6)+0.5, yticks=np.arange(6)+0.5,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    xlabel="Predicted class", ylabel="True class",
                    title="Final held-out test — " + ("row proportions" if row_normalized else "raw confusion counts"))
                plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
                for i in range(6):
                    for j in range(6):
                        label = f"{array[i,j]:.2f}" if row_normalized else str(int(array[i,j]))
                        threshold = 0.5 if row_normalized else float(array.max())/2
                        axis.text(j+0.5, i+0.5, label, ha="center", va="center",
                                  color="white" if array[i,j] > threshold else "black", fontsize=9)
                colorbar = fig.colorbar(mesh, ax=axis,
                    label="Proportion within true class" if row_normalized else "Requirements")
                if colorbar.solids is not None:
                    colorbar.solids.set_rasterized(False)
                save(fig, stem)

            fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
            bars = axis.bar(CLASS_NAMES, [r["f1"] for r in per_class], color="#245b84", width=0.65)
            for bar, row in zip(bars, per_class):
                axis.text(bar.get_x()+bar.get_width()/2, row["f1"]+0.025,
                          f"{row['f1']:.3f}\nn={row['support']}", ha="center", fontsize=9)
            axis.set(ylabel="Held-out test F1", ylim=(0, 1.14),
                     title="Final held-out test — per-class F1")
            axis.set_yticks([0, .2, .4, .6, .8, 1.])
            plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.2)
            save(fig, "test_per_class_f1")
        for name, expected in hashes.items():
            _require(_sha(root / name) == expected, "Final source changed during plotting")
        manifest = {"status": "GENERATED_FROM_ACTUAL_INDEPENDENTLY_VERIFIED_FINAL_EXPORTS",
            "scope": "FINAL_HELD_OUT_TEST", "experiment_id": EXPERIMENT_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_rows": 1015, "class_order": ORDER, "class_names": CLASS_NAMES,
            "actual_test_macro_f1": metrics["macro_f1"],
            "supports": [r["support"] for r in per_class],
            "raw_confusion_distinct_from_row_normalized": True,
            "confusion_normalization": "Divide each raw true-class row by its own support",
            "source_data_smoothed_interpolated_or_fabricated": False,
            "model_or_dataset_loaded_by_plotter": False,
            "additional_inference_or_evaluation_performed": False,
            "png_dpi": 600, "PDF_and_SVG_fully_vector": True,
            "matplotlib_version": matplotlib.__version__,
            "source_sha256": hashes,
            "plotting_code_sha256": _sha(staged / "plot_final_evidence.py"),
            "generated_files_sha256": {name:_sha(staged/name) for name in rendered}}
        (staged / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
        os.replace(staged, destination)
        return manifest
    except BaseException as exc:
        plt.close("all")
        (staged / "FIGURE_GENERATION_FAILED.json").write_text(json.dumps({
            "status":"FAILED_PRESERVED_NO_INFERENCE_RERUN", "exception":repr(exc)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args=parser.parse_args()
    print(json.dumps(plot_final_evidence(args.output_dir), indent=2))
