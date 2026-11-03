"""Audit explicitly supplied development CSVs; never defaults to any dataset."""
import argparse
import json

if __package__:
    from .data_tools import (frame_audit, load_requirements, new_output_directory,
                             pair_audit, require_execution, sha256, write_json_exclusive)
else:
    from data_tools import (frame_audit, load_requirements, new_output_directory,
                            pair_audit, require_execution, sha256, write_json_exclusive)


def audit_datasets(train_path, validation_path, output_dir, *, execute=False, encoding="utf-8-sig"):
    require_execution(execute)
    output = new_output_directory(output_dir)
    train = load_requirements(train_path, encoding=encoding, source_ids=True)
    validation = load_requirements(validation_path, encoding=encoding, source_ids=True)
    report = {
        "dataset_roles": ["train", "validation"], "encoding": encoding,
        "train": {**frame_audit(train), "sha256": sha256(train_path)},
        "validation": {**frame_audit(validation), "sha256": sha256(validation_path)},
        "train_validation_overlap": pair_audit(train, validation),
        "mutations_performed": False, "automatic_relabeling": False,
        "normalization": "NFKC + collapse whitespace + strip + casefold; auditing only",
        "limitation": "Aggregate checks are not proof of semantic independence or verified augmentation ancestry.",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output / "development_data_audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True, help="New nonexistent directory")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--execute", action="store_true", help="Explicitly read authorized development CSVs")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required; no data accessed")
    print(json.dumps(audit_datasets(args.train, args.validation, args.output_dir,
                                    execute=True, encoding=args.encoding), indent=2))


if __name__ == "__main__":
    main()
