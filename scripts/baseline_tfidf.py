"""Explicit train/validation TF-IDF baseline. No fitting or data reads on import."""
import argparse
import json

if __package__:
    from .data_tools import (LABEL_IDS, LABEL_NAMES, load_requirements, new_output_directory,
                             pair_audit, require_execution, sha256, write_json_exclusive)
else:
    from data_tools import (LABEL_IDS, LABEL_NAMES, load_requirements, new_output_directory,
                            pair_audit, require_execution, sha256, write_json_exclusive)


def make_model():
    """Historical baseline recipe; vocabulary/IDF and classifier fit train only."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])


def run_baseline(train_path, validation_path, output_dir, *, execute=False, encoding="utf-8-sig"):
    require_execution(execute)
    output = new_output_directory(output_dir)
    train = load_requirements(train_path, encoding=encoding, source_ids=True)
    validation = load_requirements(validation_path, encoding=encoding, source_ids=True)
    overlap = pair_audit(train, validation)
    if overlap["shared_source_ids"] or overlap["normalized_text_pairs"]:
        raise ValueError("Shared source IDs or normalized Text across development partitions; no training performed")
    from sklearn.metrics import accuracy_score, classification_report
    output.mkdir(parents=True, exist_ok=False)
    model = make_model()
    model.fit(train["Text"], train["Class"].astype(int))
    predicted = model.predict(validation["Text"])
    truth = validation["Class"].astype(int)
    report = classification_report(truth, predicted, labels=LABEL_IDS, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "dataset_role": "validation", "rows": len(validation),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"], "weighted_f1": report["weighted avg"]["f1-score"],
        "class_order": LABEL_IDS, "class_names": LABEL_NAMES, "classification_report": report,
    }
    predictions = validation[["source_id", "Class"]].copy()
    predictions["predicted_class"] = predicted
    with (output / "validation_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        predictions.to_csv(handle, index=False)
    write_json_exclusive(output / "validation_metrics.json", metrics)
    write_json_exclusive(output / "run_config.json", {
        "model": "TF-IDF + LogisticRegression", "fit_scope": "training Text and labels only",
        "ngram_range": [1, 2], "min_df": 2, "max_features": 50000,
        "max_iter": 1500, "class_weight": "balanced", "random_state": 42,
        "input_field": "Text", "external_preprocessing": "none", "encoding": encoding,
        "train_sha256": sha256(train_path), "validation_sha256": sha256(validation_path),
        "train_rows": len(train), "validation_rows": len(validation), "overlap_audit": overlap,
        "final_evaluation_executed": False,
    })
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True, help="New nonexistent directory")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--execute", action="store_true", help="Explicitly fit the training-only baseline")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required; no data accessed")
    print(json.dumps(run_baseline(args.train, args.validation, args.output_dir,
                                  execute=True, encoding=args.encoding), indent=2))


if __name__ == "__main__":
    main()
