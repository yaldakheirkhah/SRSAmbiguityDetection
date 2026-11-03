"""Recreate figures using actual aggregate evidence, without model/data access."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verify_published_results import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify()
    # Refuse dangerous output destinations before creating anything.
    output = args.output_dir.absolute()
    if output.exists() or any(p.casefold() in {"data", "outputs", "runs", "model_cache"} for p in output.parts):
        raise ValueError("Choose a new figure directory outside protected research paths")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    data = json.loads((ROOT / "results/frozen_bert_summary.json").read_text())
    with (ROOT / "results/mac_training_history.csv").open(newline="") as f:
        history = list(csv.DictReader(f))
    output.mkdir(parents=True, exist_ok=False)

    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)

    epochs = [int(row["epoch"]) for row in history]
    for key, ylabel, name in [("train_weighted_loss", "Recorded training weighted loss", "training_loss"),
                               ("val_macro_f1", "Validation Macro F1", "validation_macro_f1")]:
        fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
        ax.plot(epochs, [float(r[key]) for r in history], color="#255c99", linewidth=1.4)
        ax.axvline(data["best_epoch"], color="#9e5025", linestyle="--", linewidth=1, label="Selected epoch 41")
        ax.set(xlabel="Completed epoch", ylabel=ylabel)
        ax.legend(frameon=False); ax.grid(alpha=0.15)
        save(fig, name)
    labels = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"]
    for partition in ("validation", "final_test"):
        record = data[partition]
        counts = np.array(record["confusion_matrix"], dtype=int)
        title = "Validation" if partition == "validation" else "Final held-out Test"
        for normalize in (False, True):
            values = counts / counts.sum(axis=1, keepdims=True) if normalize else counts
            fig, ax = plt.subplots(figsize=(7.4, 6.0), layout="constrained")
            im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
            fig.colorbar(im, ax=ax, label="Row proportion" if normalize else "Requirement count", shrink=0.8)
            for i in range(6):
                for j in range(6):
                    txt = f"{values[i,j]:.2f}" if normalize else str(counts[i,j])
                    ax.text(j, i, txt, ha="center", va="center", color="white" if values[i,j] > values.max() * 0.55 else "black", fontsize=9)
            ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
                   xlabel="Predicted class", ylabel="True class", title=title + (" — row-normalized" if normalize else " — raw counts"))
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            save(fig, partition + ("_confusion_normalized" if normalize else "_confusion_counts"))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    x = np.arange(6)
    for offset, partition, label, color in [(-0.19, "validation", "Validation", "#255c99"), (0.19, "final_test", "Final held-out Test", "#b96634")]:
        values = [data[partition]["per_class"][str(i)]["f1"] for i in range(1, 7)]
        ax.bar(x + offset, values, 0.38, label=label, color=color)
    ax.set(xticks=x, xticklabels=labels, ylabel="Per-class F1", ylim=(0, 1))
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.15)
    save(fig, "per_class_f1")
    (output / "FIGURE_PROVENANCE.json").write_text(json.dumps({"inputs": ["results/frozen_bert_summary.json", "results/mac_training_history.csv"], "code": "scripts/plot_published_results.py", "inference_performed": False, "smoothing_or_interpolation": False, "figure_count": 7}, indent=2) + "\n")
    print("Created 7 figures in PNG/PDF/SVG from recorded aggregates and history.")


if __name__ == "__main__":
    main()
