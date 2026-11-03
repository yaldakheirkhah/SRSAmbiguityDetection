"""One fixed CNN run and one fixed LSTM run; never opens held-out data.

Usage: python scripts/train_baselines.py --smoke
       python scripts/train_baselines.py --architecture cnn
       python scripts/train_baselines.py --architecture lstm
"""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import (BASE, ROOT, CONFIG_PATH, CLASSES, sha, write_json, config, load_frozen,
    encode, seed_all, loader, make_model, evaluate, metrics, probabilities, write_predictions)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    versions = {}
    for p in ["torch", "numpy", "pandas", "requests"]:
        versions[p] = importlib.metadata.version(p)
    return {"captured_at_utc": timestamp(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(), "dependencies": versions,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "MPS_operator_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "not set"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
        "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "battery": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout,
        "memory": subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True).stdout,
        "processor": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "git_head_read_only": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}


def inputs():
    cfg = config()
    train, val = load_frozen()
    prep = json.loads((BASE / "artifacts/preparation_manifest.json").read_text())
    assert prep["configuration_sha256"] == sha(CONFIG_PATH), "Configuration changed after preparation"
    for filename, digest in prep["files"].items():
        assert sha(BASE / "artifacts" / filename) == digest
    vocabulary = json.loads((BASE / "artifacts/vocabulary.json").read_text())
    matrix = np.load(BASE / "artifacts/initial_embedding_matrix_fp32.npy", allow_pickle=False)
    assert matrix.dtype == np.float32 and matrix.shape == (len(vocabulary), 100)
    assert np.array_equal(matrix[0], np.zeros(100, dtype=np.float32))
    weights = json.loads((BASE / "artifacts/class_weights.json").read_text())
    return cfg, train, val, vocabulary, matrix, weights


def smoke_one(architecture, cfg, train, vocab, matrix, device):
    """Forward/backward + padding invariance; training rows only, no val scores."""
    seed_all(cfg["seed"], cfg["training"]["cpu_threads"])
    model = make_model(architecture, matrix, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    seqs = encode(train.Text.iloc[:64], vocab, cfg["tokenizer"]["max_tokens"])
    # Include short and maximum-length probes to catch padding/backend defects.
    seqs += [np.full(n, 1, dtype=np.int64) for n in [1, 2, 3, 4, 5, 191, 192]]
    labels = list(train.Class.iloc[:64]) + [1, 2, 3, 4, 5, 6, 1]
    dl = loader(seqs, labels, 32, False, cfg["seed"])
    losses = []
    start = time.monotonic()
    model.train()
    for i, (ids, lengths, y) in enumerate(dl):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids.to(device), lengths.to(device))
        loss = nn.functional.cross_entropy(logits, y.to(device))
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    logits, p, _ = evaluate(model, dl, device)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-12)
    # Compare each very short sequence alone with additional right padding.
    model.eval()
    padding_deltas = []
    with torch.inference_mode():
        for n in [1, 2, 3, 4, 5, 13]:
            ids = torch.full((1, max(5,n)), 1, dtype=torch.long, device=device)
            if n < 5:
                ids[:, n:] = 0
            lengths = torch.tensor([n], device=device)
            a = model(ids, lengths).cpu().numpy()
            b = model(nn.functional.pad(ids, (0, 17)), lengths).cpu().numpy()
            delta = float(np.max(np.abs(a-b)))
            assert delta < 1e-4, f"padding affects {architecture}: {delta}"
            padding_deltas.append(delta)
    assert torch.equal(model.embedding.weight[0].detach().cpu(), torch.zeros(100))
    memory = {}
    if device == "mps":
        torch.mps.synchronize()
        memory = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                  "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    return {"architecture": architecture, "device": device, "status": "PASS", "training_rows_used": 64,
        "synthetic_integer_sequence_probes": 7, "held_out_validation_scores_computed": False,
        "optimization_steps": len(losses), "finite_losses": losses, "max_padding_invariance_abs_difference": max(padding_deltas),
        "seconds": time.monotonic()-start, "memory": memory}


def smoke():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1", "Implicit MPS operator fallback is disabled by design"
    out = BASE / "reports/SMOKE_TEST.json"
    assert not out.exists(), "Smoke evidence already exists; do not overwrite"
    cfg, train, val, vocab, matrix, weights = inputs()
    result = {"status": "PASS", "timestamp_utc": timestamp(), "config_sha256": sha(CONFIG_PATH),
              "scope": "train-only implementation/backend checks, no validation scoring", "architectures": {}}
    for architecture in ["cnn", "lstm"]:
        preferred = "mps" if torch.backends.mps.is_available() else "cpu"
        attempts = []
        try:
            attempt = smoke_one(architecture, cfg, train, vocab, matrix, preferred)
            attempts.append(attempt)
            chosen = preferred
        except Exception as e:
            attempts.append({"device": preferred, "status": "FAILED", "exception": repr(e), "traceback": traceback.format_exc()})
            if preferred == "cpu":
                raise
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            attempts.append(smoke_one(architecture, cfg, train, vocab, matrix, "cpu"))
            chosen = "cpu"
        result["architectures"][architecture] = {"selected_device": chosen, "attempts": attempts}
        print(json.dumps(result["architectures"][architecture], indent=2), flush=True)
    result["environment"] = environment()
    write_json(out, result, exclusive=True)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def report_text(result, cfg):
    lines = ["Frozen original validation; six-class single-label classification", "class_order=[1,2,3,4,5,6]; zero_division=0",
             "class  name                    precision       recall           F1     support"]
    for c, name in zip(CLASSES, cfg["labels"]["names"]):
        x = result["per_class"][str(c)]
        lines.append(f"{c:5d}  {name:23s} {x['precision']:.10f} {x['recall']:.10f} {x['f1']:.10f} {x['support']:7d}")
    for avg in ["macro", "weighted"]:
        lines.append(f"{avg:30s} {result[avg+'_precision']:.10f} {result[avg+'_recall']:.10f} {result[avg+'_f1']:.10f}")
    lines.append(f"Accuracy {result['accuracy']:.10f}")
    return "\n".join(lines) + "\n"


def train_model(architecture):
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1"
    cfg, train, val, vocab, matrix, weights_record = inputs()
    cfg_sha = sha(CONFIG_PATH)
    smoke_record = json.loads((BASE / "reports/SMOKE_TEST.json").read_text())
    assert smoke_record["status"] == "PASS" and smoke_record["config_sha256"] == cfg_sha
    device = smoke_record["architectures"][architecture]["selected_device"]
    out = BASE / "runs" / f"glove_{architecture}_seed42_v1"
    if out.parent.is_symlink():
        raise ValueError('Generated runs directory may not be a symlink')
    out.parent.mkdir(exist_ok=True)
    out.mkdir(exist_ok=False)
    seed_all(cfg["seed"], cfg["training"]["cpu_threads"])
    env = environment()
    write_json(out / "environment.json", env, exclusive=True)
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (out / "pip_freeze.txt").write_text(pip.stdout)
    # Preserve exact configuration bytes so the checkpoint hash binds this copy.
    with (out / "configuration.json").open("xb") as f:
        f.write(CONFIG_PATH.read_bytes())
    write_json(out / "class_weights.json", weights_record, exclusive=True)
    (out / "vocabulary.json").write_bytes((BASE / "artifacts/vocabulary.json").read_bytes())
    (out / "tokenizer.json").write_bytes((BASE / "artifacts/tokenizer.json").read_bytes())
    sources = {p.name: sha(p) for p in [Path(__file__), BASE / "scripts/common.py", BASE / "scripts/prepare_data.py"]}
    write_json(out / "run_identity.json", {"architecture": architecture, "seed": cfg["seed"], "started_utc": timestamp(),
        "device": device, "config_sha256": cfg_sha, "training_sha256": cfg["dataset"]["train_sha256"],
        "validation_sha256": cfg["dataset"]["validation_sha256"], "source_code_sha256": sources,
        "preparation_manifest_sha256": sha(BASE / "artifacts/preparation_manifest.json"),
        "embedding_source_sha256": sha(BASE / "provenance/glove_source.json"),
        "initial_embeddings_sha256": sha(BASE / "artifacts/initial_embedding_matrix_fp32.npy")}, exclusive=True)
    xs = encode(train.Text, vocab, cfg["tokenizer"]["max_tokens"])
    vs = encode(val.Text, vocab, cfg["tokenizer"]["max_tokens"])
    tc = cfg["training"]
    train_dl = loader(xs, train.Class, tc["batch_size"], True, cfg["seed"])
    val_dl = loader(vs, val.Class, tc["evaluation_batch_size"], False, cfg["seed"])
    model = make_model(architecture, matrix, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
                                  betas=tuple(tc["betas"]), eps=tc["epsilon"])
    w = torch.tensor(weights_record["weights"], dtype=torch.float32, device=device)
    history = []
    best = -float("inf")
    meaningful_best = -float("inf")
    best_epoch = None
    stale = 0
    started = time.monotonic()
    max_driver = 0
    for epoch in range(1, tc["maximum_epochs"] + 1):
        t0 = time.monotonic()
        model.train()
        summed_loss = 0.0
        summed_weight = 0.0
        updates = 0
        for ids, lengths, labels in train_dl:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, lengths)
            loss_sum = nn.functional.cross_entropy(logits, labels, weight=w, reduction="sum")
            weight_sum = w[labels].sum()
            loss = loss_sum / weight_sum
            assert torch.isfinite(loss).item(), "Non-finite training loss"
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            summed_loss += float(loss_sum.detach().cpu())
            summed_weight += float(weight_sum.detach().cpu())
            updates += 1
        logits, probs, predicted = evaluate(model, val_dl, device)
        result = metrics(val.Class.to_numpy(), predicted)
        improved = result["macro_f1"] > best
        if improved:
            best = result["macro_f1"]
            best_epoch = epoch
            state = {"model_state_dict": cpu_state(model), "epoch": epoch, "architecture": architecture,
                "configuration_sha256": cfg_sha, "vocabulary_sha256": sha(out / "vocabulary.json"),
                "training_sha256": cfg["dataset"]["train_sha256"],
                "validation_sha256": cfg["dataset"]["validation_sha256"], "validation_macro_f1": best}
            tmp = out / "best_checkpoint.pt.partial"
            torch.save(state, tmp)
            tmp.replace(out / "best_checkpoint.pt")
            write_predictions(out, val, logits, probs, predicted, cfg)
            write_json(out / "validation_metrics.json", result)
        if result["macro_f1"] > meaningful_best + tc["meaningful_improvement_delta"]:
            meaningful_best = result["macro_f1"]
            stale = 0
        else:
            stale += 1
        memory = None
        if device == "mps":
            torch.mps.synchronize()
            memory = torch.mps.driver_allocated_memory()
            max_driver = max(max_driver, memory)
        row = {"epoch": epoch, "train_loss_weighted": summed_loss/summed_weight,
            "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_updates": updates,
            "validation_accuracy": result["accuracy"], "validation_macro_f1": result["macro_f1"],
            "validation_weighted_f1": result["weighted_f1"], "checkpoint_updated": improved,
            "meaningful_patience_counter": stale, "epoch_seconds": time.monotonic()-t0,
            "mps_driver_bytes_sampled": memory}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "epoch_history.csv", index=False, float_format="%.17g")
        print(f"{architecture} epoch {epoch:02d}: loss={row['train_loss_weighted']:.5f} macro_f1={result['macro_f1']:.9f} "
              f"weighted_f1={result['weighted_f1']:.9f} accuracy={result['accuracy']:.9f} best_epoch={best_epoch} "
              f"patience={stale}/{tc['early_stopping_patience']} seconds={row['epoch_seconds']:.1f}", flush=True)
        if epoch >= tc["minimum_epochs"] and stale >= tc["early_stopping_patience"]:
            break
    # Reconstruct a new model and replay the selected checkpoint on the SAME validation.
    checkpoint = torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    replay = make_model(architecture, matrix, cfg).to(device)
    replay.load_state_dict(checkpoint["model_state_dict"], strict=True)
    replay_logits, replay_probs, replay_pred = evaluate(replay, val_dl, device)
    saved_logits = np.load(out / "validation_logits_fp32.npy", allow_pickle=False)
    saved_probs = np.load(out / "validation_probabilities_fp64.npy", allow_pickle=False)
    assert np.allclose(replay_logits, saved_logits, atol=1e-5, rtol=1e-5)
    assert np.array_equal(replay_pred, saved_probs.argmax(1)+1)
    assert metrics(val.Class, replay_pred)["macro_f1"] == best
    assert checkpoint["epoch"] == best_epoch
    assert best == max(x["validation_macro_f1"] for x in history)
    assert best_epoch == next(x["epoch"] for x in history if x["validation_macro_f1"] == best)
    assert all(sha(ROOT/cfg["dataset"][role]) == cfg["dataset"][role+"_sha256"] for role in ["train", "validation"])
    result = json.loads((out / "validation_metrics.json").read_text())
    (out / "classification_report.txt").write_text(report_text(result, cfg))
    pd.DataFrame(result["confusion_matrix"], index=[f"true_{c}" for c in CLASSES],
                 columns=[f"predicted_{c}" for c in CLASSES]).to_csv(out / "confusion_matrix.csv")
    write_json(out / "checkpoint_replay.json", {"status": "PASS", "device": device,
        "max_abs_logit_difference": float(np.max(np.abs(replay_logits-saved_logits))),
        "max_abs_probability_difference": float(np.max(np.abs(replay_probs-saved_probs))),
        "prediction_mismatches": int(np.sum(replay_pred != saved_probs.argmax(1)+1)),
        "macro_f1": best, "best_epoch": best_epoch}, exclusive=True)
    summary = {"status": "COMPLETED_VALIDATION_ONLY", "architecture": architecture, "seed": cfg["seed"],
        "device": device, "ended_utc": timestamp(), "epochs_completed": len(history), "selected_epoch": best_epoch,
        "selection": tc["checkpoint_rule"], "total_seconds": time.monotonic()-started,
        "max_mps_driver_bytes_sampled": max_driver, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_rows": len(train), "validation_rows": len(val), "training_only_vocabulary": True,
        "class_order": CLASSES, "validation_metrics": result,
        "test_accessed": False, "historical_reproduction": False,
        "artifacts_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "COMPLETED.json", summary, exclusive=True)
    print(json.dumps({k: summary[k] for k in ["status", "architecture", "device", "selected_epoch", "epochs_completed", "total_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--architecture", choices=["cnn", "lstm"])
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        train_model(args.architecture)
