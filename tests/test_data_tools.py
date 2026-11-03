"""Invented CSV fixtures only; no original partitions, model fitting or inference.

Baseline fitting is replaced by a spy. Metrics in that test describe only its
synthetic contract fixture and are never experimental performance evidence.
"""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_datasets, baseline_tfidf, create_splits, data_tools


def invented_rows(per_class=1, prefix="training"):
    return [
        {"source_id": f"{prefix}-{label}-{index}",
         "Text": f"The {prefix} system shall record type{label} event {index}.", "Class": label}
        for label in range(1, 7) for index in range(per_class)
    ]


def write_fixture(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_imports_never_read_data_fit_or_create_output(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Import must not perform data I/O or fitting")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    for module in [data_tools, create_splits, baseline_tfidf, audit_datasets]:
        importlib.reload(module)


@pytest.mark.parametrize("module", [create_splits, baseline_tfidf, audit_datasets])
def test_cli_requires_execution_before_data_access(module, monkeypatch):
    monkeypatch.setattr(module, "load_requirements", lambda *a, **k: pytest.fail("No execution approval"))
    args = ["--output-dir", "unused"]
    args += ["--input", "unused"] if module is create_splits else ["--train", "unused", "--validation", "unused"]
    with pytest.raises(SystemExit) as caught:
        module.main(args)
    assert caught.value.code == 2


@pytest.mark.parametrize("call", [
    lambda: create_splits.create_splits("unused", "unused"),
    lambda: baseline_tfidf.run_baseline("unused", "unused", "unused"),
    lambda: audit_datasets.audit_datasets("unused", "unused", "unused"),
])
def test_api_requires_execution_without_resolving_paths(call, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("Unapproved path operation"))
    with pytest.raises(PermissionError):
        call()


@pytest.mark.parametrize("name", ["test.csv", "TeSt.CsV"])
def test_heldout_input_name_rejected_before_resolve(name, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("Protected source must remain untouched"))
    with pytest.raises(ValueError, match="Held-out"):
        data_tools.allowed_input(Path("fictional") / name)


@pytest.mark.parametrize("suffix", ["data/splits", "data/splits/new_reproduction", "DATA/SPLITS/new"])
def test_frozen_output_location_rejected_before_read(tmp_path, suffix, monkeypatch):
    monkeypatch.setattr(create_splits, "load_requirements", lambda *a, **k: pytest.fail("Must reject output first"))
    with pytest.raises(ValueError, match="data/splits"):
        create_splits.create_splits("unused", tmp_path / suffix, execute=True)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("immutable fixture")
    monkeypatch.setattr(create_splits, "load_requirements", lambda *a, **k: pytest.fail("Must reject output first"))
    with pytest.raises(FileExistsError):
        create_splits.create_splits("unused", output, execute=True)
    assert marker.read_text() == "immutable fixture"


def test_output_symlink_cannot_escape_to_frozen_directory(tmp_path):
    protected = tmp_path / "data" / "splits"
    protected.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="data/splits"):
        data_tools.new_output_directory(alias / "new")


def test_historical_normalization_is_conservative_and_does_not_remove_punctuation():
    assert data_tools.normalize("  Ａlpha\t BETA \n") == "alpha beta"
    assert data_tools.normalize("Must stop.") != data_tools.normalize("Must stop")
    assert data_tools.punctuation_key("Must stop.") == data_tools.punctuation_key("Must stop")


def test_deterministic_stratified_splits_quarantine_and_duplicate_behavior(tmp_path):
    rows = invented_rows(40)
    rows += [dict(rows[0]), {**rows[0], "Text": "  " + rows[0]["Text"].upper() + "  "}]
    rows += [{"Text": "Disputed invented fixture.", "Class": 1, "source_id": "conflict-1"},
             {"Text": "Disputed invented fixture.", "Class": 2, "source_id": "conflict-2"}]
    source = write_fixture(tmp_path / "invented_raw.csv", rows)
    source_before = source.read_bytes()
    a, b = tmp_path / "reproduction_a", tmp_path / "reproduction_b"
    report = create_splits.create_splits(source, a, execute=True, encoding="utf-8")
    repeated = create_splits.create_splits(source, b, execute=True, encoding="utf-8")
    assert report == repeated
    assert (report["original_rows"], report["quarantined_rows"], report["conflicting_groups"]) == (244, 2, 1)
    assert report["consistent_duplicate_rows_removed"] == 2
    assert report["unique_usable_rows"] == 240
    assert {k: v["rows"] for k, v in report["partitions"].items()} == {"train": 168, "validation": 36, "test": 36}
    expected = {"train": 28, "validation": 6, "test": 6}
    identities = []
    texts = set()
    for name, count in expected.items():
        # These are NEW, invented temporary fixtures, never the frozen source.
        first, second = a / f"{name}.csv", b / f"{name}.csv"
        assert first.read_bytes() == second.read_bytes()
        part = pd.read_csv(first, dtype={"source_id": str})
        assert list(part.columns) == ["source_id", "Text", "Class"]
        assert data_tools.class_counts(part) == {str(i): count for i in range(1, 7)}
        assert part.source_id.is_unique
        identities.append(set(part.source_id))
        texts.update(part.Text)
    assert not any(identities[i] & identities[j] for i in range(3) for j in range(i + 1, 3))
    assert texts == {row["Text"] for row in rows[:240]}
    quarantine = pd.read_csv(a / "conflicting_annotations.csv")
    assert quarantine.Class.tolist() == [1, 2]
    assert source.read_bytes() == source_before


@pytest.mark.parametrize("mutation", ["null_text", "blank_text", "invalid_label", "missing_class", "duplicate_id"])
def test_invalid_development_inputs_fail_closed(tmp_path, mutation):
    rows = invented_rows()
    if mutation == "null_text": rows[0]["Text"] = None
    elif mutation == "blank_text": rows[0]["Text"] = "  \t "
    elif mutation == "invalid_label": rows[0]["Class"] = 7
    elif mutation == "missing_class": rows.pop()
    else: rows[1]["source_id"] = rows[0]["source_id"]
    source = write_fixture(tmp_path / "invented.csv", rows)
    with pytest.raises(ValueError):
        data_tools.load_requirements(source, source_ids=True)


def test_audit_reports_conflicts_duplicates_and_punctuation_risk_without_changing_inputs(tmp_path):
    train = invented_rows()
    train += [{**train[0], "source_id": "same-label-duplicate"},
              {**train[0], "source_id": "cross-label-conflict", "Class": 2}]
    validation = invented_rows(prefix="validation")
    validation[0]["Text"] = train[0]["Text"].rstrip(".")
    t = write_fixture(tmp_path / "invented_train.csv", train)
    v = write_fixture(tmp_path / "invented_validation.csv", validation)
    before = [p.read_bytes() for p in [t, v]]
    report = audit_datasets.audit_datasets(t, v, tmp_path / "audit", execute=True)
    assert report["train"]["exact_text_duplicate_excess"] == 2
    assert report["train"]["normalized_conflict_groups"] == 1
    assert report["train"]["normalized_conflict_rows"] == 3
    assert report["train_validation_overlap"]["normalized_text_pairs"] == 0
    assert report["train_validation_overlap"]["punctuation_equivalent_pairs"] == 3
    assert [p.read_bytes() for p in [t, v]] == before


def test_baseline_fit_boundary_uses_training_only_with_mock_model(tmp_path, monkeypatch):
    train = invented_rows(2)
    validation = invented_rows(prefix="validation")
    t = write_fixture(tmp_path / "invented_train.csv", train)
    v = write_fixture(tmp_path / "invented_validation.csv", validation)
    calls = []
    class SpyModel:
        def fit(self, texts, labels):
            calls.append(("fit", list(texts), list(labels)))
        def predict(self, texts):
            calls.append(("predict", list(texts)))
            return [1, 2, 3, 4, 5, 6]
    monkeypatch.setattr(baseline_tfidf, "make_model", SpyModel)
    output = tmp_path / "baseline"
    metrics = baseline_tfidf.run_baseline(t, v, output, execute=True)
    assert calls[0] == ("fit", [r["Text"] for r in train], [r["Class"] for r in train])
    assert calls[1] == ("predict", [r["Text"] for r in validation])
    assert len(calls) == 2
    assert metrics["dataset_role"] == "validation" and metrics["class_order"] == [1, 2, 3, 4, 5, 6]
    assert metrics["rows"] == 6
    predictions = pd.read_csv(output / "validation_predictions.csv")
    assert predictions.source_id.tolist() == [r["source_id"] for r in validation]


def test_baseline_refuses_overlap_before_constructing_model(tmp_path, monkeypatch):
    rows = invented_rows()
    t = write_fixture(tmp_path / "invented_train.csv", rows)
    v = write_fixture(tmp_path / "invented_validation.csv", rows)
    monkeypatch.setattr(baseline_tfidf, "make_model", lambda: pytest.fail("Overlap must block fitting"))
    with pytest.raises(ValueError, match="Shared source"):
        baseline_tfidf.run_baseline(t, v, tmp_path / "blocked", execute=True)
    assert not (tmp_path / "blocked").exists()
