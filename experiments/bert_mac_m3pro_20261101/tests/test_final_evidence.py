"""Synthetic arithmetic/parser/guard tests only; no requirement corpus or model."""
from pathlib import Path
import copy
import hashlib
import json
import math
import struct

import pytest

import verify_final_evidence as audit


def test_known_integer_counts_and_both_averages():
    result = audit.metrics_from_exported_labels([1, 1, 2, 2, 3, 4, 5, 6], [1, 2, 2, 2, 3, 4, 5, 6])
    assert result["accuracy"] == 7 / 8
    assert result["macro_precision"] == pytest.approx((1 + 2 / 3 + 4) / 6)
    assert result["macro_recall"] == pytest.approx((0.5 + 1 + 4) / 6)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + 4) / 6)
    assert result["weighted_f1"] == pytest.approx((2 * 2 / 3 + 2 * 4 / 5 + 4) / 8)
    assert result["per_class"]["1"] == {"precision": 1, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert result["confusion_matrix"][0] == [1, 1, 0, 0, 0, 0]
    assert result["weighted_recall"] == result["accuracy"]


def test_fixed_six_class_macro_includes_zero_support_classes():
    result = audit.metrics_from_exported_labels([1], [1])
    assert result["macro_f1"] == 1 / 6
    assert result["weighted_f1"] == 1
    assert result["per_class"]["6"]["support"] == 0


@pytest.mark.parametrize("truth,pred", [([], []), ([1], []), ([0], [1]), ([1], [7]), ([True], [1]), ([1.0], [1])])
def test_invalid_exported_label_domain_refused(truth, pred):
    with pytest.raises(ValueError):
        audit.metrics_from_exported_labels(truth, pred)


def test_report_mismatch_cannot_pass():
    metrics = audit.metrics_from_exported_labels([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    changed = copy.deepcopy(metrics)
    changed["weighted_f1"] = 0.75
    with pytest.raises(ValueError, match="weighted_f1"):
        audit.check_metric_report(changed, metrics)


def test_nonfinite_report_refused():
    with pytest.raises(ValueError):
        audit.numeric_equal(float("nan"), 0.0, "nonfinite")


def test_boolean_is_not_a_confusion_count():
    metrics = audit.metrics_from_exported_labels([1], [1])
    changed = copy.deepcopy(metrics)
    changed["confusion_matrix"][0][0] = True
    with pytest.raises(ValueError, match="integer counts"):
        audit.check_metric_report(changed, metrics)


def test_source_dataset_argument_rejected_before_filesystem(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No filesystem call is permitted for a source-dataset argument")
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="exported-output directory"):
        audit.verify_final_evidence("/fictional/sealed/test.csv")


def test_absent_exports_never_make_success_report(tmp_path):
    with pytest.raises(ValueError, match="Missing actual final export"):
        audit.verify_final_evidence(tmp_path)
    assert not (tmp_path / "independent_final_verification.json").exists()


def npy_bytes(values, rows, descr="<f4"):
    header = repr({"descr": descr, "fortran_order": False, "shape": (rows, 6)}).encode("latin1")
    header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
    data = struct.pack("<" + ("f" if descr == "<f4" else "d") * len(values), *values)
    return b"\x93NUMPY" + b"\x01\x00" + len(header).to_bytes(2, "little") + header + data


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_bounded_npy_export_parser(tmp_path, dtype):
    values = list(map(float, range(12)))
    path = tmp_path / "synthetic_array.npy"
    path.write_bytes(npy_bytes(values, 2, dtype))
    assert audit.read_numeric_npy(path, 2, dtype) == [values[:6], values[6:]]


def test_nonfinite_npy_and_wrong_cardinality_refused(tmp_path):
    path = tmp_path / "synthetic_array.npy"
    path.write_bytes(npy_bytes([float("nan")] + [0.0] * 5, 1))
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.read_numeric_npy(path, 1, "<f4")
    path.write_bytes(npy_bytes([0.0] * 6, 1))
    with pytest.raises(ValueError, match="shape"):
        audit.read_numeric_npy(path, 2, "<f4")


def test_immutable_evidence_write_refuses_change(tmp_path):
    p = tmp_path / "synthetic_receipt.json"
    audit._write_immutable(p, b'{"synthetic_unit_fixture":true}\n')
    audit._write_immutable(p, b'{"synthetic_unit_fixture":true}\n')
    with pytest.raises(ValueError, match="refusing overwrite"):
        audit._write_immutable(p, b'{"synthetic_unit_fixture":false}\n')


def test_id_normalization_is_conservative_and_not_punctuation_stripping():
    assert audit.normalized("  FOO\tBAR  ") == "foo bar"
    assert audit.normalized("foo, bar") != audit.normalized("foo bar")
