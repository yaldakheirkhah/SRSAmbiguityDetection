"""CPU-only tests for the separate MPS entry point; no model/data execution."""
from pathlib import Path
import importlib.util
import json
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_bert.py"
spec = importlib.util.spec_from_file_location("ambifix_mps_core_for_test", SCRIPT)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def config():
    return {
        "model_id": core.MODEL_ID,
        "pretrained_revision": core.REVISION,
        "pretrained_weights_sha256": core.WEIGHTS_SHA,
        "device_type": "mps",
        "train_precision": "float32",
        "validation_precision": "float32",
        "train_batch_size": 8,
        "gradient_accumulation": 2,
        "eval_batch_size": 16,
        "max_length": 192,
    }


def mac(monkeypatch):
    monkeypatch.setattr(core.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(core.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")


def test_source_compiles_without_ml_import():
    compile(SCRIPT.read_text(), str(SCRIPT), "exec")
    assert "torch" not in core.__dict__
    assert "np" not in core.__dict__


@pytest.mark.parametrize("setting", ["PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"])
def test_unsafe_acceleration_environment_refused(monkeypatch, setting):
    mac(monkeypatch)
    monkeypatch.setenv(setting, "1")
    with pytest.raises(RuntimeError):
        core.check_mac_contract(config())


@pytest.mark.parametrize("key,value", [("train_precision", "bf16"), ("validation_precision", "float16"),
                                       ("device_type", "cpu"), ("train_batch_size", 4),
                                       ("gradient_accumulation", 4), ("eval_batch_size", 8),
                                       ("max_length", 128)])
def test_unrecorded_recipe_changes_refused(monkeypatch, key, value):
    mac(monkeypatch)
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        core.check_mac_contract(cfg)


def test_non_native_mac_refused(monkeypatch):
    mac(monkeypatch)
    monkeypatch.setattr(core.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError):
        core.check_mac_contract(config())


def test_literal_best_and_meaningful_stop_are_distinct():
    assert core.checkpoint_decision(0.5000001, 0.5, 0.5, 3, 0.001) == (True, False, 4)
    assert core.checkpoint_decision(0.5, 0.5, 0.5, 3, 0.001) == (False, False, 4)
    assert core.checkpoint_decision(0.502, 0.5, 0.5, 3, 0.001) == (True, True, 0)


def test_failed_preflight_prevents_training_lock_and_ml_import(monkeypatch, tmp_path):
    mac(monkeypatch)
    cfg = config()
    monkeypatch.setattr(core, "input_check", lambda root: {"config": cfg, "input_identity": {"fixture": True}})
    monkeypatch.setattr(core, "completed_run", lambda root, verified: None)
    preflight = types.ModuleType("mac_preflight")
    def blocked(root, full_smoke):
        assert (tmp_path / "outputs/EXECUTION_LOCK.json").is_file()
        return {"status": "BLOCKED_INSUFFICIENT_DISK", "blockers": ["Only 4.8GiB available; 12GiB required"]}
    preflight.run_preflight = blocked
    monkeypatch.setitem(sys.modules, "mac_preflight", preflight)
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="preflight did not pass"):
        core.execute(tmp_path)
    assert not (tmp_path / "outputs/STARTED.json").exists()
    assert not (tmp_path / "outputs/EXECUTION_LOCK.json").exists()
    assert not (tmp_path / "outputs/COMPLETED.json").exists()
    assert not (tmp_path / "outputs/COMPLETE_ARTIFACTS.zip").exists()
    assert "torch" not in (set(sys.modules) - before)
    failure = json.loads((tmp_path / "reports/LAST_EXECUTION_PREFLIGHT.json").read_text())
    assert failure["status"] == "BLOCKED_EXECUTION_PREFLIGHT"
    assert failure["training_started"] is False
    assert "Only 4.8GiB available; 12GiB required" in failure["blockers"]
    assert failure["preflight_report"]["status"] == "BLOCKED_INSUFFICIENT_DISK"


def test_preflight_must_bind_exact_frozen_input_identity(monkeypatch, tmp_path):
    mac(monkeypatch)
    monkeypatch.setattr(core, "input_check", lambda root: {"config": config(), "input_identity": {"expected": True}})
    monkeypatch.setattr(core, "completed_run", lambda root, verified: None)
    preflight = types.ModuleType("mac_preflight")
    preflight.run_preflight = lambda root, full_smoke: {"status": "PASS_MAC_PREFLIGHT", "input_identity": {"different": True}}
    monkeypatch.setitem(sys.modules, "mac_preflight", preflight)
    with pytest.raises(ValueError, match="different input identities"):
        core.execute(tmp_path)
    assert not (tmp_path / "outputs/STARTED.json").exists()
    assert not (tmp_path / "outputs/EXECUTION_LOCK.json").exists()


def test_concurrent_session_refused_before_second_smoke(tmp_path):
    reserved = tmp_path / "outputs/reserved_model_run"
    with core.reserve_session(tmp_path, reserved):
        first = json.loads((tmp_path / "outputs/EXECUTION_LOCK.json").read_text())
        with pytest.raises(RuntimeError, match="already reserved"):
            with core.reserve_session(tmp_path, reserved):
                pytest.fail("Concurrent preflight must not enter")
        assert json.loads((tmp_path / "outputs/EXECUTION_LOCK.json").read_text()) == first
    assert not (tmp_path / "outputs/EXECUTION_LOCK.json").exists()


def test_reservation_cleanup_never_removes_replaced_owner(tmp_path):
    reserved = tmp_path / "outputs/reserved_model_run"
    with core.reserve_session(tmp_path, reserved):
        lock = tmp_path / "outputs/EXECUTION_LOCK.json"
        replacement = {"reservation_token": "another-owner"}
        lock.write_text(json.dumps(replacement))
    assert json.loads(lock.read_text()) == replacement


def test_started_training_keeps_its_owned_session_lock(tmp_path):
    reserved = tmp_path / "outputs/reserved_model_run"
    with core.reserve_session(tmp_path, reserved):
        (tmp_path / "outputs/STARTED.json").write_text('{"status":"STARTED_NEW_RUN"}')
    assert (tmp_path / "outputs/EXECUTION_LOCK.json").exists()
