"""Shared data-only checks. No dataset is opened at import time.

Only caller-supplied development data may be loaded. Normalization identifies
groups for auditing/splitting; the original Text field is never rewritten.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

LABEL_IDS = [1, 2, 3, 4, 5, 6]
LABEL_NAMES = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"]


def normalize(text):
    """Historical NFKC, whitespace collapse and casefold grouping only."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def punctuation_key(text):
    """Diagnostic key only: punctuation equivalence is not verified lineage."""
    return re.sub(r"[\W_]+", "", normalize(text))


def allowed_input(path):
    path = Path(path).expanduser()
    for candidate in (path,):
        if any(p.casefold() == "test.csv" for p in candidate.parts):
            raise ValueError("Held-out source paths are prohibited")
    resolved = path.resolve()
    if any(p.casefold() == "test.csv" for p in resolved.parts):
        raise ValueError("Held-out source paths are prohibited")
    return resolved


def _frozen_directory(path):
    parts = [p.casefold() for p in Path(path).parts]
    return any(parts[i:i + 2] == ["data", "splits"] for i in range(len(parts) - 1))


def new_output_directory(path):
    """Validate a new isolated output location; this does not create it."""
    path = Path(path).expanduser()
    if _frozen_directory(path):
        raise ValueError("Outputs inside data/splits are prohibited; use a new reproduction directory")
    resolved = path.resolve()
    if _frozen_directory(resolved):
        raise ValueError("Outputs inside data/splits are prohibited")
    if resolved.exists():
        raise FileExistsError("Output directory already exists; no files will be overwritten")
    return resolved


def require_execution(execute):
    if not execute:
        raise PermissionError("Explicit --execute is required; no data has been accessed")


def load_requirements(path, *, encoding="utf-8-sig", source_ids=False):
    import pandas as pd
    path = allowed_input(path)
    frame = pd.read_csv(path, encoding=encoding, dtype={"source_id": str})
    if not {"Text", "Class"}.issubset(frame.columns) or len(frame) == 0:
        raise ValueError("A nonempty CSV with Text and Class columns is required")
    if frame[["Text", "Class"]].isna().any().any():
        raise ValueError("Missing Text or Class values are prohibited")
    if not frame["Text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Text must contain strings; no automatic coercion is performed")
    if frame["Text"].map(normalize).eq("").any():
        raise ValueError("Empty normalized Text is prohibited")
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    if labels.isna().any() or not labels.isin(LABEL_IDS).all():
        raise ValueError("Class must use the six original numeric labels 1..6")
    if set(labels) != set(LABEL_IDS):
        raise ValueError("All six original labels must be represented")
    if source_ids:
        if "source_id" not in frame or frame["source_id"].isna().any():
            raise ValueError("Stable nonempty source_id values are required")
        if frame["source_id"].str.strip().eq("").any() or not frame["source_id"].is_unique:
            raise ValueError("source_id must be nonempty and unique within each partition")
    return frame


def class_counts(frame):
    values = frame["Class"].astype(int).value_counts()
    return {str(label): int(values.get(label, 0)) for label in LABEL_IDS}


def frame_audit(frame):
    keys = frame["Text"].map(normalize)
    punctuation = frame["Text"].map(punctuation_key)
    conflicts = frame.assign(_key=keys).groupby("_key")["Class"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    return {
        "rows": len(frame), "class_counts": class_counts(frame),
        "exact_text_duplicate_excess": int(frame["Text"].duplicated().sum()),
        "normalized_text_duplicate_excess": int(keys.duplicated().sum()),
        "punctuation_equivalent_duplicate_excess": int(punctuation.duplicated().sum()),
        "normalized_conflict_groups": len(conflict_keys),
        "normalized_conflict_rows": int(keys.isin(conflict_keys).sum()),
    }


def pair_audit(train, validation):
    def overlap_pairs(left, right):
        a, b = left.value_counts(), right.value_counts()
        return sum(int(a[k]) * int(b[k]) for k in a.index.intersection(b.index))
    result = {
        "exact_text_pairs": overlap_pairs(train["Text"], validation["Text"]),
        "normalized_text_pairs": overlap_pairs(train["Text"].map(normalize), validation["Text"].map(normalize)),
        "punctuation_equivalent_pairs": overlap_pairs(train["Text"].map(punctuation_key), validation["Text"].map(punctuation_key)),
        "limitation": "Punctuation and normalization checks do not establish semantic independence or augmentation lineage.",
    }
    if "source_id" in train and "source_id" in validation:
        result["shared_source_ids"] = len(set(train["source_id"]) & set(validation["source_id"]))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with allowed_input(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
