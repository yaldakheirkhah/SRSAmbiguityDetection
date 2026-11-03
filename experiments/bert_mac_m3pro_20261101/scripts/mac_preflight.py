"""Fail-closed Mac resource/dependency checks and train-only BERT smoke test.

The resource gate precedes imports of the ML stack, model downloads and smoke
training. No validation inference or final evaluation is performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from verify_inputs import verify_inputs, digest

PINS = {"torch": "2.9.0", "transformers": "4.51.3", "huggingface-hub": "0.36.2",
        "safetensors": "0.5.3", "numpy": "1.26.4", "pandas": "2.2.3",
        "scikit-learn": "1.6.1", "matplotlib": "3.10.0"}
EXPERIMENT = "ambifix_original_sqrtweighted_bert_mac_mps_v1_20261101"
PROTOCOL_FIELDS = ["model_id", "pretrained_revision", "pretrained_weights_sha256",
    "dataset_class_ids", "dataset_class_names", "dataset_to_model_index", "seed",
    "max_length", "train_batch_size", "eval_batch_size", "gradient_accumulation",
    "max_epochs", "min_epochs", "early_stop_patience", "min_delta", "learning_rate",
    "weight_decay", "scheduler_patience", "scheduler_factor", "scheduler_min_lr",
    "scheduler_threshold", "scheduler_threshold_mode", "max_grad_norm",
    "class_weight_power", "gradient_checkpointing", "validation_precision",
    "optimizer_name", "optimizer_betas", "optimizer_eps", "tokenizer_use_fast",
    "dynamic_padding", "model_input_column", "external_preprocessing",
    "checkpoint_selection", "early_stopping", "weighted_gradient_reduction",
    "attention_implementation", "training_rows", "validation_rows",
    "training_data_file", "validation_data_file", "augmented_examples_permitted"]


def memory_bytes():
    p = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True,
                       capture_output=True, timeout=10)
    if p.returncode or not p.stdout.strip().isdigit():
        raise RuntimeError("Installed unified memory could not be verified: " + p.stderr.strip())
    return int(p.stdout.strip())


def check_protocol(root, config):
    original = json.loads((root / "provenance/nvidia_training_config.json").read_text())
    changed = [name for name in PROTOCOL_FIELDS if config.get(name) != original.get(name)]
    if changed:
        raise ValueError("Scientific protocol drift from audited NVIDIA source: " + str(changed))
    if (config["experiment_id"] != EXPERIMENT or config["device_type"] != "mps"
            or config["train_precision"] != "float32"
            or config["validation_precision"] != "float32"
            or config["optimizer_foreach"] is not False):
        raise ValueError("Unapproved Mac run identity/precision/optimizer policy")
    return {"unchanged_scientific_fields": PROTOCOL_FIELDS,
            "documented_hardware_deviations": config["hardware_deviations"],
            "nvidia_config_sha256": digest(root / "provenance/nvidia_training_config.json")}


def run_preflight(root, full_smoke=True):
    root = Path(root).resolve()
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "BLOCKED_PREFLIGHT", "training_executed": False,
              "full_experiment_training_executed": False,
              "held_out_data_accessed": False, "blockers": [],
              "smoke_status": "NOT_RUN", "model_downloaded": False,
              "smoke_training_executed": False, "snapshot_download_invoked": False,
              "environment_created_by_this_check": False}
    try:
        verified = verify_inputs(root)
        cfg = verified["config"]
        result.update({"experiment_id": cfg["experiment_id"],
                       "input_identity": verified["input_identity"],
                       "training_rows": len(verified["train_rows"]),
                       "validation_rows": len(verified["validation_rows"]),
                       "protocol_comparison": check_protocol(root, cfg)})
    except Exception as exc:
        result["blockers"].append("Input/protocol integrity: " + repr(exc))
        return result
    result["hardware"] = {"platform": platform.platform(), "system": platform.system(),
                          "architecture": platform.machine(), "python": sys.version,
                          "python_executable": sys.executable,
                          "free_disk_bytes": shutil.disk_usage(root).free}
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        result["blockers"].append("This run requires native arm64 macOS.")
    if sys.version_info[:2] != (3, 12):
        result["blockers"].append("This fixed environment requires Python 3.12.")
    try:
        result["hardware"]["unified_memory_bytes"] = memory_bytes()
        if result["hardware"]["unified_memory_bytes"] < cfg["minimum_unified_memory_gib"] * 1024**3:
            result["blockers"].append("Installed unified memory is below the fixed 18GiB requirement.")
    except Exception as exc:
        result["blockers"].append(str(exc))
    if result["hardware"]["free_disk_bytes"] < cfg["minimum_free_disk_gib"] * 1024**3:
        result["blockers"].append("Insufficient disk: at least 12GiB actually free is required AFTER environment setup.")
    versions = {}
    for name, expected in PINS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        versions[name] = {"expected": expected, "installed": actual}
        if actual != expected:
            result["blockers"].append(f"Dedicated environment requires {name}=={expected}; found {actual}.")
    result["dependencies"] = versions
    for name in ["PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"]:
        if os.environ.get(name) not in (None, "0"):
            result["blockers"].append(name + " must be unset or 0; no silent fallback/fast-math change.")
    result["resource_and_dependency_gate_passed"] = not result["blockers"]
    if result["blockers"]:
        return result
    # Set before importing torch, in a fresh dedicated process.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "0"
    os.environ["PYTORCH_MPS_PREFER_METAL"] = "0"
    try:
        import torch
        result["mps"] = {"built": torch.backends.mps.is_built(),
                         "available": torch.backends.mps.is_available(),
                         "torch": torch.__version__}
        if not result["mps"]["available"]:
            raise RuntimeError("MPS unavailable to this process. CPU fallback is prohibited.")
        if not full_smoke:
            result["status"] = "PASS_MAC_RESOURCE_DEPENDENCY_CHECKS_SMOKE_PENDING"
            return result
        result["smoke"] = bert_smoke(root, verified, progress=result)
        result["smoke_status"] = "PASS_TRAIN_ONLY_BERT_FORWARD_BACKWARD_SAVE_RELOAD"
        result["status"] = "PASS_MAC_PREFLIGHT"
    except Exception as exc:
        result["blockers"].append("MPS/BERT smoke failed: " + repr(exc))
    return result


def bert_smoke(root, verified, progress=None):
    """Discarded diagnostic model, freshly initialized again for the actual run."""
    import random
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cfg = verified["config"]
    before = time.monotonic()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.mps.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    cached = root / "model_cache" / ("models--"+cfg["model_id"].replace("/","--")) / "snapshots" / cfg["pretrained_revision"] / "model.safetensors"
    if progress is not None:
        progress.update({"snapshot_download_invoked":True, "model_downloaded":None,
                         "pretrained_cache_file_present_before":cached.is_file(),
                         "download_note":"Hub snapshot resolution may use cache or network; transferred bytes are not independently measured."})
    snapshot = Path(snapshot_download(repo_id=cfg["model_id"], revision=cfg["pretrained_revision"],
        token=False, cache_dir=str(root / "model_cache"),
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]))
    if snapshot.name != cfg["pretrained_revision"] or digest(snapshot / "model.safetensors") != cfg["pretrained_weights_sha256"]:
        raise ValueError("Smoke initialization is not the pinned original pretrained weights.")
    if progress is not None: progress["pretrained_snapshot"] = str(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model, loading = AutoModelForSequenceClassification.from_pretrained(str(snapshot),
        num_labels=6, id2label={i-1:cfg["dataset_class_names"][str(i)] for i in range(1,7)},
        label2id={cfg["dataset_class_names"][str(i)]:i-1 for i in range(1,7)},
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        attn_implementation="eager", output_loading_info=True)
    if set(loading.get("missing_keys", [])) != {"classifier.weight", "classifier.bias"} or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Unexpected original-pretrained initialization keys.")
    if any(not (k.startswith("cls.predictions.") or k.startswith("cls.seq_relationship.")) for k in loading.get("unexpected_keys", [])):
        raise ValueError("Unexpected unused original-pretrained keys.")
    model = model.float().to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["optimizer_betas"]), eps=cfg["optimizer_eps"], foreach=False)
    weights = torch.tensor(verified["class_weights"], dtype=torch.float32, device="mps")
    # Use actual training rows only. Full 192-token padding tests the fixed recipe's
    # maximum padded sequence footprint, with 8 x 2 accumulation unchanged.
    count = cfg["train_batch_size"] * cfg["gradient_accumulation"]
    rows = sorted(verified["train_rows"], key=lambda r: (-len(r["Text"]), r["source_id"]))[:count]
    denominator = 0.0
    model.train(); optimizer.zero_grad(set_to_none=True)
    max_loss_error = 0.0
    for offset in range(0, len(rows), cfg["train_batch_size"]):
        chunk = rows[offset:offset+cfg["train_batch_size"]]
        batch = tokenizer([r["Text"] for r in chunk], padding="max_length", truncation=True,
                          max_length=cfg["max_length"], return_tensors="pt")
        batch = {k:v.to("mps") for k,v in batch.items()}
        targets = torch.tensor([r["Class"]-1 for r in chunk], dtype=torch.long, device="mps")
        logits = model(**batch).logits
        if logits.dtype != torch.float32 or not torch.isfinite(logits).all().item():
            raise FloatingPointError("Invalid MPS FP32 smoke logits.")
        numerator = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction="sum")
        values = logits.detach().cpu().double().numpy()
        logsumexp = np.log(np.exp(values-values.max(1,keepdims=True)).sum(1)) + values.max(1)
        cpu_targets = targets.cpu().numpy()
        manual = sum(verified["class_weights"][int(y)] * (logsumexp[i]-values[i,y]) for i,y in enumerate(cpu_targets))
        error = abs(float(numerator.detach().cpu())-manual)
        max_loss_error = max(max_loss_error,error)
        if error > 1e-4 * max(1.,abs(manual)):
            raise FloatingPointError("MPS weighted CE differs from manual CPU FP64 numerator.")
        numerator.backward()
        if progress is not None:
            progress["training_executed"] = True
            progress["smoke_training_executed"] = True
        denominator += float(weights[targets].sum().item())
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Nonfinite MPS smoke gradient.")
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True, foreach=False)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.mps.synchronize()
    model.eval()
    with torch.inference_mode():
        expected = model(**batch).logits.cpu().numpy()
    memory = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
              "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
              "recommended_max_bytes": torch.mps.recommended_max_memory()}
    with tempfile.TemporaryDirectory(prefix="train-only-smoke-", dir=root) as temp:
        destination = Path(temp)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        checkpoint_digest = digest(destination / "model.safetensors")
        del model, optimizer, logits, numerator, weights
        gc.collect(); torch.mps.empty_cache()
        reloaded, info = AutoModelForSequenceClassification.from_pretrained(temp, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, attn_implementation="eager", output_loading_info=True)
        if any(info.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]):
            raise ValueError("Incomplete smoke checkpoint reload.")
        reloaded = reloaded.float().to("mps").eval()
        reloaded_tokenizer = AutoTokenizer.from_pretrained(temp, local_files_only=True, trust_remote_code=False, use_fast=True)
        chunk = rows[-cfg["train_batch_size"]:]
        replay_batch = {k:v.to("mps") for k,v in reloaded_tokenizer([r["Text"] for r in chunk],
            padding="max_length", truncation=True, max_length=cfg["max_length"],return_tensors="pt").items()}
        with torch.inference_mode():
            actual = reloaded(**replay_batch).logits.cpu().numpy()
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-5) or not np.array_equal(expected.argmax(1), actual.argmax(1)):
            raise ValueError("Train-only saved-model/tokenizer replay failed.")
        difference = float(np.abs(expected-actual).max())
        del reloaded, replay_batch, batch, targets
        gc.collect(); torch.mps.empty_cache()
    return {"status":"PASS", "rows":len(rows), "training_source_ids":[r["source_id"] for r in rows],
            "uses_validation":False, "optimizer_steps":1, "device":"mps", "precision":"float32",
            "batch_size":cfg["train_batch_size"], "gradient_accumulation":cfg["gradient_accumulation"],
            "padded_length":cfg["max_length"], "maximum_weighted_CE_numerator_abs_error":max_loss_error,
            "reload_maximum_logit_abs_difference":difference, "discarded_smoke_checkpoint_sha256":checkpoint_digest,
            "memory":memory, "elapsed_seconds":time.monotonic()-before,
            "smoke_model_used_for_actual_training":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resources-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    report=run_preflight(args.root, full_smoke=not args.resources_only)
    if args.output:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["status"].startswith("PASS") else 2)
