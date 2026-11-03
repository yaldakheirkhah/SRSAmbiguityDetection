"""Historical split algorithm, guarded for NEW isolated reproductions only.

This is not a command to recreate or replace the frozen manuscript partitions.
No work occurs on import. All input/output paths and execution are explicit.
"""
import argparse
import hashlib
import json

if __package__:
    from .data_tools import (class_counts, load_requirements, new_output_directory,
                             normalize, require_execution, sha256, write_json_exclusive)
else:
    from data_tools import (class_counts, load_requirements, new_output_directory,
                            normalize, require_execution, sha256, write_json_exclusive)


def create_splits(input_path, output_dir, *, execute=False, encoding="latin-1", seed=42):
    """Quarantine conflicts, deduplicate groups, then stratify 70/15/15.

    Returns aggregate metadata; writes new CSVs exclusively after all split
    calculations/checks succeed. Labels and original Text values are preserved.
    """
    require_execution(execute)
    output = new_output_directory(output_dir)
    frame = load_requirements(input_path, encoding=encoding)
    frame["group_key"] = frame["Text"].map(normalize)
    counts = frame.groupby("group_key")["Class"].nunique()
    conflict_keys = set(counts[counts > 1].index)
    disputed = frame[frame["group_key"].isin(conflict_keys)].copy()
    consistent = frame[~frame["group_key"].isin(conflict_keys)].copy()
    clean = consistent.drop_duplicates(subset=["group_key"]).copy()
    clean["source_id"] = clean["group_key"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
    if not clean["source_id"].is_unique:
        raise ValueError("Truncated source-ID collision; refuse to create partitions")
    from sklearn.model_selection import train_test_split
    development, test = train_test_split(clean, test_size=0.15, random_state=seed, stratify=clean["Class"])
    train, validation = train_test_split(development, test_size=0.15 / 0.85,
                                        random_state=seed, stratify=development["Class"])
    parts = {"train": train, "validation": validation, "test": test}
    keys = [set(part["group_key"]) for part in parts.values()]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Grouping identity overlap; no output written")
    report = {
        "purpose": "New isolated reproduction of historical splitting algorithm; does not replace frozen partitions",
        "seed": seed, "encoding": encoding, "input_sha256": sha256(input_path),
        "original_rows": len(frame), "conflicting_groups": len(conflict_keys),
        "quarantined_rows": len(disputed), "consistent_duplicate_rows_removed": len(consistent) - len(clean),
        "unique_usable_rows": len(clean),
        "partitions": {name: {"rows": len(part), "class_counts": class_counts(part)} for name, part in parts.items()},
        "normalized_group_overlap": 0,
        "normalization": "NFKC + collapse whitespace + strip + casefold; grouping only",
        "limitations": "No automatic relabeling. Punctuation variants and semantic near duplicates are not grouped by this historical algorithm; no leakage-free claim.",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "conflicting_annotations.csv").open("x", encoding="utf-8", newline="") as handle:
        disputed.to_csv(handle, index=False)
    for name, part in parts.items():
        with (output / (name + ".csv")).open("x", encoding="utf-8", newline="") as handle:
            part[["source_id", "Text", "Class"]].to_csv(handle, index=False)
    write_json_exclusive(output / "split_audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Authorized raw CSV; never the sealed held-out partition")
    parser.add_argument("--output-dir", required=True, help="New nonexistent directory outside data/splits")
    parser.add_argument("--encoding", default="latin-1", help="Historical source encoding default: latin-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="Explicitly create a NEW isolated reproduction")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required; no data accessed")
    report = create_splits(args.input, args.output_dir, execute=True, encoding=args.encoding, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
