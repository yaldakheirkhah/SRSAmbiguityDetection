"""Pure guard/source tests. No actual dataset, model, result or figure is read."""
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_guard_default_never_touches_any_path(monkeypatch):
    module = load("final_test_guarded")
    def forbidden(*args, **kwargs):
        pytest.fail("No filesystem operation is permitted with absent authorization")
    for method in ("open", "stat", "resolve", "read_text", "read_bytes", "exists", "is_file", "is_symlink"):
        monkeypatch.setattr(Path, method, forbidden)
    with pytest.raises(PermissionError, match="no dataset accessed"):
        module.execute(dataset_argument="/this/file/must/never/be/touched.csv")


@pytest.mark.parametrize("arguments", [
    (None, None, None, False),
    ("separate_authorization.json", None, "never_access.csv", True),
    (None, "frozen_manifest.json", "never_access.csv", True),
    ("separate_authorization.json", "frozen_manifest.json", "never_access.csv", False),
])
def test_missing_gate_controls_refuse_before_control_or_data_reads(monkeypatch, arguments):
    module = load("final_test_guarded")
    monkeypatch.setattr(module, "_control_json", lambda *args: pytest.fail("Gate must stop before control reads"))
    with pytest.raises(PermissionError):
        module.authorize(*arguments)


def test_no_ml_packages_imported_by_guard_or_plot_module():
    before = set(sys.modules)
    load("final_test_guarded")
    load("plot_validation")
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "transformers", "matplotlib", "numpy"} for name in imported)


def test_denied_authorization_never_reads_freeze_or_dataset(monkeypatch):
    module = load("final_test_guarded")
    reads = []
    def denied(path, description):
        reads.append(description)
        return {"schema_version": 1, "action": module.ACTION, "authorized": False}
    monkeypatch.setattr(module, "_control_json", denied)
    monkeypatch.setattr(module, "_sha", lambda *args: pytest.fail("Denied authorization must not hash anything"))
    with pytest.raises(PermissionError, match="invalid explicit"):
        module.authorize("author_approval.json", "frozen.json", "never_access.csv", True)
    assert reads == ["Authorization"]


def test_control_json_cannot_be_redirected_to_csv(monkeypatch):
    module = load("final_test_guarded")
    for method in ("is_symlink", "stat", "read_text", "open"):
        monkeypatch.setattr(Path, method, lambda *args: pytest.fail("A CSV control argument must not be inspected"))
    with pytest.raises(PermissionError, match="regular JSON"):
        module._control_json("never_access.csv", "Frozen-model manifest")


def test_plot_refuses_missing_actual_evidence_before_output(tmp_path):
    module = load("plot_validation")
    output = tmp_path / "figures"
    with pytest.raises(FileNotFoundError, match="Actual complete validation evidence required"):
        module.make_figures(tmp_path, output)
    assert not output.exists()


def test_separate_mac_identity_is_fixed():
    guard = load("final_test_guarded")
    plots = load("plot_validation")
    assert guard.EXPERIMENT_ID == plots.EXPERIMENT_ID == "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
    assert len(guard.CLASS_NAMES) == 6
    assert guard.CLASS_NAMES[1] == "Syntactic Ambiguity"
    assert guard.CLASS_NAMES[3] == "Syntax Ambiguity"
