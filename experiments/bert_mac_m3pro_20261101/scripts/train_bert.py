"""One NEW unaugmented Apple Silicon MPS BERT training run.

--inputs-only is standard-library-only; --preflight performs guarded Mac checks
and a disposable training-only smoke test. --execute requires a successful full
preflight before one fixed-seed training run. FP32 MPS is an explicit hardware
adaptation from the separate NVIDIA BF16 recipe, not historical reproduction.
No held-out evaluation entry point exists here. Original files are untouched.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import warnings
import zipfile

MODEL_ID = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
WEIGHTS_SHA = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
ORDER = [1, 2, 3, 4, 5, 6]


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Artifact paths must be package-relative POSIX strings")
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or any(x.casefold() == "test.csv" for x in p.parts):
        raise ValueError("Escaped/protected artifact path")
    resolved = (Path(root)/p).resolve()
    if not resolved.is_relative_to(Path(root).resolve()) or any(x.casefold() == "test.csv" for x in resolved.parts):
        raise ValueError("Artifact path escapes its allowed directory")
    return resolved


def sha(path):
    path = Path(path)
    if any(x.casefold() == "test.csv" for x in path.parts):
        raise ValueError("Held-out file access is prohibited")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_atomic(path, value, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    temp = path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    if exclusive and path.exists():
        raise FileExistsError(f"Completion already exists: {path}")
    os.replace(temp, path)


def input_check(root):
    # Lazy import: this helper itself is stdlib-only and reads only allowlisted inputs.
    from verify_inputs import verify_inputs
    return verify_inputs(Path(root))


def checkpoint_decision(score, literal_best, meaningful_best, stale, min_delta):
    """Saving and stopping use separate clocks; exact score ties keep first epoch."""
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation Macro F1")
    literal = score > literal_best
    meaningful = score > meaningful_best + min_delta
    return literal, meaningful, (0 if meaningful else stale+1)


def check_mac_contract(cfg):
    """Refuse silent scientific/hardware adaptations before importing Torch."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This separate recipe requires native Apple Silicon macOS")
    if cfg.get("device_type") != "mps":
        raise ValueError("Mac recipe must explicitly use MPS")
    if cfg.get("train_precision") != "float32" or cfg.get("validation_precision") != "float32":
        raise ValueError("This fixed MPS recipe requires FP32 training AND validation")
    if cfg.get("train_batch_size") != 8 or cfg.get("gradient_accumulation") != 2 or cfg.get("eval_batch_size") != 16:
        raise ValueError("Physical/effective batches must retain 8 x 2 / evaluation 16")
    if cfg.get("max_length") != 192:
        raise ValueError("The intended 192-token maximum must not change silently")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS CPU fallback must be disabled; restart with PYTORCH_ENABLE_MPS_FALLBACK=0")
    if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
        raise RuntimeError("MPS fast math must be disabled for this fixed FP32 recipe")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "0"


def cpu_tree(value, torch):
    """Portable optimizer state export; no MPS tensors need be loaded on return."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v, torch) for v in value)
    return value


class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, text):
        for target in self.targets: target.write(text); target.flush()
    def flush(self):
        for target in self.targets: target.flush()


def completed_run(root, verified):
    marker = root/"outputs/COMPLETED.json"
    if not marker.is_file():
        return None
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("status") != "COMPLETED_NEW_VALIDATION_ONLY_RUN":
        raise ValueError("Invalid completion status")
    if saved.get("input_identity") != verified["input_identity"] or saved.get("training_config") != verified["config"]:
        raise ValueError("Existing result belongs to different immutable inputs/configuration")
    run = safe_path(root, saved["run_directory"])
    if not run.is_relative_to((root/"outputs").resolve()):
        raise ValueError("Completed run is outside outputs")
    for name, digest in saved["artifact_sha256"].items():
        if sha(safe_path(run,name)) != digest:
            raise ValueError("Completed run artifact changed: "+name)
    archive = safe_path(root, saved["return_zip"])
    if sha(archive) != saved["return_zip_sha256"]:
        raise ValueError("Completed return ZIP changed")
    return saved


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def package_result(root, run, artifact_hashes):
    """Exact result checkpoint bytes, not a source-only/report-only return archive."""
    name = "COMPLETE_ARTIFACTS.zip"
    destination = root/"outputs"/name
    temp = destination.with_suffix(".zip.partial")
    if destination.exists() or temp.exists():
        raise FileExistsError("Never overwrite a result archive")
    with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for relative, digest in sorted(artifact_hashes.items()):
            source = safe_path(run, relative)
            if sha(source) != digest:
                raise ValueError("Run changed before return packaging: "+relative)
            z.write(source, arcname=run.name+"/"+relative)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise IOError("Return archive CRC verification failed")
        for relative, digest in artifact_hashes.items():
            h = hashlib.sha256()
            with z.open(run.name+"/"+relative) as f:
                for block in iter(lambda:f.read(8*1024*1024), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise IOError("Return ZIP changed a checkpoint/artifact byte")
    os.replace(temp,destination)
    return destination


@contextmanager
def reserve_session(root, reserved_run):
    """Serialize full smoke tests and training; release only OUR pre-start lock.

    An interrupted actual training run retains the lock. A blocked smoke check
    creates no STARTED marker and releases its UUID-matched reservation, so a
    later corrected resource preflight can run without deleting another session.
    """
    outputs=root/"outputs";outputs.mkdir(exist_ok=True)
    lock_path=outputs/"EXECUTION_LOCK.json"
    token=uuid.uuid4().hex
    try:
        with lock_path.open("x",encoding="utf-8") as lock:
            json.dump({"reservation_token":token,"reserved_run_directory":reserved_run.relative_to(root).as_posix(),
                "reserved_utc":utc_now(),"scope":"FULL_PREFLIGHT_AND_SINGLE_TRAINING_SESSION"},lock)
            lock.flush();os.fsync(lock.fileno())
    except FileExistsError as exc:
        raise RuntimeError("Another execution or full preflight already reserved this package. Preserve its outputs; do not launch concurrently.") from exc
    try:
        yield
    finally:
        if not (outputs/"STARTED.json").exists():
            # Never unlink a replaced reservation belonging to another process.
            try:
                saved=json.loads(lock_path.read_text(encoding="utf-8"))
            except (FileNotFoundError,ValueError):
                saved={}
            if saved.get("reservation_token")==token:
                lock_path.unlink()


def record_execution_preflight(root, preflight, error=None):
    reports=root/"reports";reports.mkdir(exist_ok=True)
    record={"status":"PASS_MAC_PREFLIGHT" if error is None else "BLOCKED_EXECUTION_PREFLIGHT",
        "timestamp_utc":utc_now(),"training_started":False,"preflight_report":preflight,
        "held_out_data_accessed":False}
    if error is not None:
        record.update({"exception_type":type(error).__name__,"message":str(error),
            "blockers":[*(preflight or {}).get("blockers",[]),str(error)]})
    json_atomic(reports/"LAST_EXECUTION_PREFLIGHT.json",record)


def checked_full_preflight(root, verified):
    from mac_preflight import run_preflight
    preflight=None
    try:
        preflight=run_preflight(root,full_smoke=True)
        if preflight.get("status")!="PASS_MAC_PREFLIGHT":
            raise RuntimeError("Mac full preflight did not pass; training is prohibited")
        if preflight.get("input_identity")!=verified["input_identity"]:
            raise ValueError("Preflight was performed on different input identities")
        if shutil.disk_usage(root).free<12*1024**3:
            raise RuntimeError("At least 12GiB free disk is required for transactional checkpoints and complete return ZIP")
    except BaseException as exc:
        record_execution_preflight(root,preflight,exc)
        raise
    record_execution_preflight(root,preflight)
    return preflight


def execute(root):
    if sys.flags.optimize:
        raise RuntimeError("Python optimization is prohibited: verification assertions must remain enabled")
    root = Path(root).resolve()
    verified = input_check(root)
    cfg = verified["config"]
    if cfg["model_id"] != MODEL_ID or cfg["pretrained_revision"] != REVISION or cfg["pretrained_weights_sha256"] != WEIGHTS_SHA:
        raise ValueError("Initialization is restricted to the pinned original pretrained BERT, never a fine-tuned checkpoint")
    check_mac_contract(cfg)
    outputs = root/"outputs"; outputs.mkdir(exist_ok=True)
    existing = completed_run(root,verified)
    if existing:
        print("Existing completed result verified and reused; no retraining:", existing["return_zip"])
        return existing
    started_marker = outputs/"STARTED.json"
    if started_marker.exists():
        raise RuntimeError("A started/interrupted run already exists. Preserve and return its actual logs; automatic retries/validation-selected repeated trials are disabled.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    run = outputs/("mac_mps_original_sqrtweighted_"+stamp)
    # Reserve BEFORE the disposable full-model smoke, not just before training:
    # concurrent smoke models could otherwise exhaust unified memory.
    with reserve_session(root,run):
        preflight=checked_full_preflight(root,verified)
        try:
            import numpy as np
            import pandas as pd
            import torch
            import transformers
            import sklearn
            from torch.utils.data import Dataset,DataLoader
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            from huggingface_hub import snapshot_download
            from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is required; no CPU training fallback is permitted")
            device=torch.device("mps")
            torch.mps.synchronize();torch.mps.empty_cache()
        except BaseException as exc:
            record_execution_preflight(root,preflight,exc)
            raise
        run.mkdir(exist_ok=False)
        json_atomic(started_marker,{"status":"STARTED_NEW_RUN","run_directory":run.relative_to(root).as_posix(),
            "input_identity":verified["input_identity"],"started_utc":utc_now()},exclusive=True)
    # STARTED now exists, so this session's lock is intentionally retained.
    trained_model = None; selected_model = None; chosen = None; history=[]; warning_rows=[]
    wall_start = time.monotonic()
    stdout = sys.stdout; stderr = sys.stderr
    with (run/"console.log").open("x",encoding="utf-8",buffering=1) as console:
        try:
            with redirect_stdout(Tee(stdout,console)),redirect_stderr(Tee(stderr,console)),warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                print("NEW original-data weighted BERT run; historical 0.581140 is a reference, not a target.",flush=True)
                print("Run:",run,"device: Apple Silicon MPS; FP32 training and validation",flush=True)
                json_atomic(run/"mac_preflight.json",preflight)
                cfg_path=run/"training_config.json"
                cfg_path.write_bytes((root/"config/training_config.json").read_bytes())
                (run/"class_weights.json").write_bytes((root/"config/class_weights.json").read_bytes())
                mapping={"dataset_class_names":cfg["dataset_class_names"],"dataset_to_model_index":cfg["dataset_to_model_index"],"model_index_to_dataset_id":ORDER}
                json_atomic(run/"class_mapping.json",mapping)
                code_hashes={"scripts/train_bert.py":sha(Path(__file__)),"scripts/verify_inputs.py":sha(root/"scripts/verify_inputs.py"),
                    "scripts/verify_artifacts.py":sha(root/"scripts/verify_artifacts.py"),
                    "scripts/mac_preflight.py":sha(root/"scripts/mac_preflight.py"),
                    "scripts/plot_validation.py":sha(root/"scripts/plot_validation.py")}
                snapshot_dir=run/"source_snapshot";snapshot_dir.mkdir(exist_ok=False)
                for relative in code_hashes:
                    original=Path(__file__) if relative=="scripts/train_bert.py" else root/relative
                    shutil.copyfile(original,snapshot_dir/Path(relative).name)
                (run/"data_manifest.json").write_bytes((root/"data_manifest.json").read_bytes())
                identity={"experiment_id":cfg["experiment_id"],"input_identity":verified["input_identity"],"training_config":cfg,
                    "pretrained_revision":REVISION,"pretrained_model_id":MODEL_ID,"started_utc":utc_now(),
                    "historical_checkpoint_recovered":False,"new_training_run":True,"source_code_sha256":code_hashes,
                    "hardware_experiment":"Separate Apple Silicon MPS FP32 run; not the NVIDIA BF16 run"}
                json_atomic(run/"run_identity.json",identity)
                environment={"python":sys.version,"platform":platform.platform(),"architecture":platform.machine(),
                    "torch":torch.__version__,"transformers":transformers.__version__,
                    "sklearn":sklearn.__version__,"numpy":np.__version__,"pandas":pd.__version__,
                    "device":"mps","mps_is_built":torch.backends.mps.is_built(),"mps_is_available":torch.backends.mps.is_available(),
                    "mac_preflight":preflight,"PYTORCH_ENABLE_MPS_FALLBACK":os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                    "PYTORCH_MPS_FAST_MATH":os.environ.get("PYTORCH_MPS_FAST_MATH"),
                    "seed":cfg["seed"],"PYTHONHASHSEED":os.environ.get("PYTHONHASHSEED"),
                    "training_precision":"FP32 forward, parameters, loss and gradients; no autocast",
                    "validation_precision":"FP32 MPS forward; CPU NumPy FP64 softmax",
                    "adamw_foreach":False,"gradient_clipping_foreach":False,
                    "deterministic_algorithms":"warn_only; MPS kernels and cross-hardware bit identity are not guaranteed"}
                json_atomic(run/"environment.json",environment)
                with (run/"pip_freeze.txt").open("w",encoding="utf-8") as f:
                    subprocess.run([sys.executable,"-m","pip","freeze"],stdout=f,check=True)
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
                cached_snapshot=preflight.get("pretrained_snapshot")
                snapshot=Path(cached_snapshot) if cached_snapshot else Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,token=False,
                    allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.txt","special_tokens_map.json"],
                    cache_dir=str(root/"model_cache")))
                if snapshot.name != REVISION or sha(snapshot/"model.safetensors") != WEIGHTS_SHA:
                    raise ValueError("Pinned original pretrained weight identity failed")
                json_atomic(run/"pretrained_revision.json",{"repository":MODEL_ID,"revision":REVISION,
                    "weights_sha256":WEIGHTS_SHA,"weights_filename":"model.safetensors","training_started_from_original_pretrained":True,
                    "fine_tuned_initialization_used":False,"tokenizer_revision":REVISION})
                tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=cfg["tokenizer_use_fast"])

                def seed_all():
                    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
                    torch.mps.manual_seed(cfg["seed"])
                    torch.use_deterministic_algorithms(True,warn_only=True)
                seed_all()
                generator=torch.Generator().manual_seed(cfg["seed"])

                class Requirements(Dataset):
                    def __init__(self,rows): self.rows=rows
                    def __len__(self): return len(self.rows)
                    def __getitem__(self,i):
                        row=self.rows[i];return row["Text"],int(row["Class"])-1
                def collate(items):
                    texts,labels=zip(*items)
                    encoded=tokenizer(list(texts),padding=True,truncation=True,max_length=cfg["max_length"],return_tensors="pt")
                    encoded["labels"]=torch.tensor(labels,dtype=torch.long)
                    return encoded
                train_loader=DataLoader(Requirements(verified["train_rows"]),batch_size=cfg["train_batch_size"],shuffle=True,
                    generator=generator,num_workers=0,pin_memory=False,collate_fn=collate)
                val_loader=DataLoader(Requirements(verified["validation_rows"]),batch_size=cfg["eval_batch_size"],shuffle=False,
                    num_workers=0,pin_memory=False,collate_fn=collate)
                seed_all()
                trained_model,loading=AutoModelForSequenceClassification.from_pretrained(str(snapshot),num_labels=6,
                    id2label={i-1:cfg["dataset_class_names"][str(i)] for i in ORDER},
                    label2id={cfg["dataset_class_names"][str(i)]:i-1 for i in ORDER},
                    local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation=cfg["attention_implementation"],output_loading_info=True)
                missing=set(loading.get("missing_keys",[]))
                unexpected=loading.get("unexpected_keys",[])
                if missing != {"classifier.weight","classifier.bias"} or any(not(k.startswith("cls.predictions.") or k.startswith("cls.seq_relationship.")) for k in unexpected):
                    raise ValueError("Unexpected encoder/pretrained loading keys; only a new classifier head and unused original pretraining heads are allowed: "+str(loading))
                if loading.get("mismatched_keys") or loading.get("error_msgs") or trained_model.config.model_type!="bert":
                    raise ValueError("Pinned pretrained loading/configuration failed")
                json_atomic(run/"pretrained_loading_info.json",loading)
                trained_model=trained_model.float().to(device);trained_model.gradient_checkpointing_disable()
                if trained_model.is_gradient_checkpointing: raise RuntimeError("Gradient checkpointing unexpectedly enabled")
                optimizer=torch.optim.AdamW(trained_model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],
                    betas=tuple(cfg["optimizer_betas"]),eps=cfg["optimizer_eps"],foreach=False)
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=cfg["scheduler_factor"],
                    patience=cfg["scheduler_patience"],threshold=cfg["scheduler_threshold"],threshold_mode=cfg["scheduler_threshold_mode"],min_lr=cfg["scheduler_min_lr"])
                weights=torch.tensor(verified["class_weights"],device=device,dtype=torch.float32)

                @torch.inference_mode()
                def evaluate(model):
                    model.eval()
                    if any(p.dtype!=torch.float32 for p in model.parameters()): raise TypeError("Validation must use FP32 parameters")
                    parts=[];truth=[];loss_sum=0.0
                    for batch in val_loader:
                        target=batch["labels"].to(device)
                        inputs={k:v.to(device,non_blocking=False) for k,v in batch.items() if k!="labels"}
                        logits=model(**inputs).logits
                        if logits.dtype!=torch.float32 or not torch.isfinite(logits).all().item(): raise FloatingPointError("Invalid FP32 validation logits")
                        loss_sum+=float(torch.nn.functional.cross_entropy(logits,target,reduction="sum").item())
                        parts.append(logits.cpu().numpy());truth.extend(target.cpu().tolist())
                    logits=np.concatenate(parts).astype(np.float32,copy=False)
                    x=logits.astype(np.float64);x-=x.max(axis=1,keepdims=True);p=np.exp(x);p/=p.sum(axis=1,keepdims=True)
                    truth=np.asarray(truth,dtype=np.int64)+1;pred=p.argmax(axis=1)+1
                    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,rtol=0,atol=1e-12):raise FloatingPointError("Invalid validation softmax")
                    scores={"accuracy":float(accuracy_score(truth,pred)),"unweighted_loss":loss_sum/len(truth)}
                    for average in ["macro","weighted"]:
                        precision,recall,f1,_=precision_recall_fscore_support(truth,pred,labels=ORDER,average=average,zero_division=0)
                        scores.update({average+"_precision":float(precision),average+"_recall":float(recall),average+"_f1":float(f1)})
                    if not all(math.isfinite(x) for x in scores.values()):raise FloatingPointError("Nonfinite validation score")
                    return scores,truth,pred,p,logits

                def save_candidate(epoch,scores,truth,pred,probabilities,logits):
                    checkpoints=run/"checkpoints";checkpoints.mkdir(exist_ok=True)
                    staged=checkpoints/(".epoch_%04d_staging"%epoch)
                    final=checkpoints/("epoch_%04d"%epoch)
                    staged.mkdir(exist_ok=False)
                    model_dir=staged/"best_model"
                    # Export explicit CPU copies without moving the live model.
                    # The resulting safetensors preserves every FP32 parameter
                    # and can load on another device without an MPS dependency.
                    trained_model.save_pretrained(model_dir,safe_serialization=True,
                        state_dict={k:v.detach().cpu() for k,v in trained_model.state_dict().items()})
                    tokenizer.save_pretrained(model_dir)
                    if not (model_dir/"model.safetensors").is_file():raise FileNotFoundError("Required full model.safetensors was not exported")
                    torch.save({"epoch":epoch,"optimizer":cpu_tree(optimizer.state_dict(),torch),"scheduler":scheduler.state_dict(),
                        "python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),
                        "mps_rng_state":torch.mps.get_rng_state(),"dataloader_generator_state":generator.get_state(),
                        "input_identity":verified["input_identity"],"training_config":cfg},staged/"best_training_state.pt")
                    np.save(staged/"validation_logits_fp32.npy",logits,allow_pickle=False)
                    np.save(staged/"validation_probabilities_fp64.npy",probabilities,allow_pickle=False)
                    prediction_rows=[]
                    for row,t,p,pvec in zip(verified["validation_rows"],truth,pred,probabilities):
                        prediction_rows.append({"source_id":row["source_id"],"Text":row["Text"],"true_label":int(t),"predicted_label":int(p),
                            **{f"p_class_{c}":format(float(pvec[c-1]),".17g") for c in ORDER}})
                    write_csv(staged/"validation_predictions.csv",prediction_rows,["source_id","Text","true_label","predicted_label",*[f"p_class_{c}" for c in ORDER]])
                    json_atomic(staged/"validation_metrics.json",{**scores,"selected_epoch":epoch,"class_order":ORDER,"zero_division":0,
                        "scope":"New frozen-validation run only; historical 0.581140 is not a target or recreated checkpoint","held_out_evaluation_performed":False})
                    report=classification_report(truth,pred,labels=ORDER,output_dict=True,zero_division=0)
                    json_atomic(staged/"classification_report.json",report)
                    (staged/"classification_report.txt").write_text(classification_report(truth,pred,labels=ORDER,digits=10,zero_division=0),encoding="utf-8")
                    cm=confusion_matrix(truth,pred,labels=ORDER)
                    with (staged/"confusion_matrix.csv").open("w",newline="",encoding="utf-8") as f:
                        writer=csv.writer(f);writer.writerow(["true_label",*[f"pred_{c}" for c in ORDER]])
                        writer.writerows([c,*cm[c-1].tolist()] for c in ORDER)
                    selection={"epoch":epoch,"selection_metric":"val_macro_f1","metric_value":scores["macro_f1"],
                        "rule":"strict literal maximum; earliest epoch wins an exact tie","class_order":ORDER}
                    json_atomic(staged/"selected_checkpoint.json",selection)
                    members={p.relative_to(staged).as_posix():sha(p) for p in staged.rglob("*") if p.is_file()}
                    json_atomic(staged/"CHECKPOINT_COMPLETE.json",{"epoch":epoch,"artifact_sha256":members})
                    os.replace(staged,final)
                    # Publishing the pointer is the commit; old complete checkpoint survives until then.
                    json_atomic(run/"selected_candidate.json",{"relative_directory":final.relative_to(run).as_posix(),"epoch":epoch,"metric_value":scores["macro_f1"]})
                    return final

                literal_best=-math.inf;meaningful_best=-math.inf;stale=0;best_epoch=None
                for epoch in range(1,cfg["max_epochs"]+1):
                    epoch_start=time.monotonic();epoch_started_utc=utc_now();trained_model.train();optimizer.zero_grad(set_to_none=True)
                    if shutil.disk_usage(root).free<6*1024**3:raise RuntimeError("Free disk fell below 6GiB mid-run; preserving actual partial evidence rather than risking corrupt exports")
                    numerator_total=0.0;denominator_total=0.0;window_denominator=0.0;steps=0
                    for batch_number,batch in enumerate(train_loader,1):
                        target=batch["labels"].to(device)
                        inputs={k:v.to(device,non_blocking=False) for k,v in batch.items() if k!="labels"}
                        out=trained_model(**inputs).logits
                        if out.dtype!=torch.float32:raise TypeError("MPS training forward must retain FP32")
                        if not torch.isfinite(out).all().item():raise FloatingPointError("Nonfinite training logits")
                        numerator=torch.nn.functional.cross_entropy(out.float(),target,weight=weights,reduction="sum")
                        denominator=float(weights[target].sum().item())
                        if not torch.isfinite(numerator).item() or denominator<=0:raise FloatingPointError("Invalid weighted loss")
                        numerator.backward()
                        numerator_total+=float(numerator.item());denominator_total+=denominator;window_denominator+=denominator
                        if batch_number%cfg["gradient_accumulation"]==0 or batch_number==len(train_loader):
                            for parameter in trained_model.parameters():
                                if parameter.grad is not None:parameter.grad.div_(window_denominator)
                            torch.nn.utils.clip_grad_norm_(trained_model.parameters(),cfg["max_grad_norm"],error_if_nonfinite=True,foreach=False)
                            optimizer.step();steps+=1;optimizer.zero_grad(set_to_none=True);window_denominator=0.0
                            if steps==1 or steps%20==0 or batch_number==len(train_loader):
                                progress={"status":"TRAINING_IN_PROGRESS","epoch":epoch,"batch_number":batch_number,
                                    "successful_optimizer_steps_this_epoch":steps,"training_rows":len(verified["train_rows"]),
                                    "actual_cumulative_train_weighted_loss":numerator_total/denominator_total,
                                    "elapsed_epoch_seconds":time.monotonic()-epoch_start,"timestamp_utc":utc_now(),
                                    "device":"mps","held_out_evaluation_performed":False}
                                json_atomic(run/"progress.json",progress)
                                print(f"Training epoch {epoch}, optimizer step {steps}: cumulative weighted loss={progress['actual_cumulative_train_weighted_loss']:.6f}",flush=True)
                    json_atomic(run/"progress.json",{"status":"VALIDATING_FROZEN_DEVELOPMENT_PARTITION","epoch":epoch,
                        "successful_optimizer_steps_this_epoch":steps,"timestamp_utc":utc_now(),"held_out_evaluation_performed":False})
                    scores,truth,pred,probabilities,logits=evaluate(trained_model)
                    literal,meaningful,stale=checkpoint_decision(scores["macro_f1"],literal_best,meaningful_best,stale,cfg["min_delta"])
                    before_lr=optimizer.param_groups[0]["lr"];scheduler.step(scores["macro_f1"])
                    row={"epoch":epoch,"train_weighted_loss":numerator_total/denominator_total,"val_loss":scores["unweighted_loss"],
                        "val_macro_f1":scores["macro_f1"],"val_weighted_f1":scores["weighted_f1"],"val_accuracy":scores["accuracy"],
                        "learning_rate_before_scheduler":before_lr,"learning_rate_after_scheduler":optimizer.param_groups[0]["lr"],
                        "successful_optimizer_steps":steps,"skipped_optimizer_steps":0,"literal_best_updated":literal,
                        "meaningful_improvement":meaningful,"early_stop_stale_epochs":stale,"elapsed_seconds":time.monotonic()-epoch_start,
                        "epoch_started_utc":epoch_started_utc,"epoch_completed_utc":utc_now()}
                    history.append(row);write_csv(run/"epoch_metrics.csv",history,list(row))
                    with (run/"training_log.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
                    if meaningful:meaningful_best=scores["macro_f1"]
                    if literal:
                        old=chosen
                        chosen=save_candidate(epoch,scores,truth,pred,probabilities,logits)
                        literal_best=scores["macro_f1"];best_epoch=epoch
                        if old is not None:
                            if not old.is_relative_to(run/"checkpoints") or not (old/"CHECKPOINT_COMPLETE.json").is_file():
                                raise RuntimeError("Refusing to remove an unowned/incomplete previous checkpoint")
                            shutil.rmtree(old)
                    print(f"Epoch {epoch:02d}: Macro F1={scores['macro_f1']:.9f}, Weighted F1={scores['weighted_f1']:.9f}, accuracy={scores['accuracy']:.9f}; selected epoch={best_epoch}; stopping counter={stale}/{cfg['early_stop_patience']}",flush=True)
                    if epoch>=cfg["min_epochs"] and stale>=cfg["early_stop_patience"]:
                        print("Early stopping. Returning literal best checkpoint, not the final epoch.",flush=True);break
                if chosen is None or best_epoch is None:raise RuntimeError("No actual best checkpoint was saved")
                candidate=json.loads((chosen/"CHECKPOINT_COMPLETE.json").read_text())
                for name,digest in candidate["artifact_sha256"].items():
                    if sha(safe_path(chosen,name))!=digest:raise ValueError("Selected checkpoint file changed")
                for p in chosen.iterdir():
                    if p.name=="CHECKPOINT_COMPLETE.json":continue
                    destination=run/p.name
                    if destination.exists():raise FileExistsError("Canonical export already exists")
                    if p.is_dir():shutil.copytree(p,destination)
                    else:shutil.copyfile(p,destination)
                # Drop the final training graph as well as the module before reload.
                del out,numerator,target,inputs,batch
                del trained_model,optimizer,scheduler,weights;trained_model=None;torch.mps.empty_cache()
                selected_model,info=AutoModelForSequenceClassification.from_pretrained(str(run/"best_model"),local_files_only=True,
                    trust_remote_code=False,use_safetensors=True,attn_implementation=cfg["attention_implementation"],output_loading_info=True)
                if any(info.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]):
                    raise ValueError("Exported best checkpoint is incomplete or incompatible")
                selected_model=selected_model.float().to(device)
                selected_tokenizer=AutoTokenizer.from_pretrained(str(run/"best_model"),local_files_only=True,trust_remote_code=False,use_fast=True)
                # val_loader's collate closure now uses the EXPORTED tokenizer for
                # the complete 1,015-row replay, not the original cached tokenizer.
                tokenizer=selected_tokenizer
                replay,_,replay_pred,replay_p,replay_logits=evaluate(selected_model)
                saved_logits=np.load(run/"validation_logits_fp32.npy",allow_pickle=False)
                saved_p=np.load(run/"validation_probabilities_fp64.npy",allow_pickle=False)
                if not np.allclose(replay_logits,saved_logits,rtol=1e-6,atol=1e-5) or not np.allclose(replay_p,saved_p,rtol=0,atol=1e-6):
                    raise RuntimeError("Reloaded checkpoint does not numerically reproduce its selected validation outputs")
                if not np.array_equal(replay_pred,saved_p.argmax(1)+1) or abs(replay["macro_f1"]-literal_best)>1e-12:
                    raise RuntimeError("Reloaded checkpoint predictions/Macro F1 changed")
                json_atomic(run/"checkpoint_replay.json",{"status":"PASS","selected_epoch":best_epoch,"prediction_mismatches":0,
                    "maximum_logit_abs_difference":float(np.abs(replay_logits-saved_logits).max()),
                    "maximum_probability_abs_difference":float(np.abs(replay_p-saved_p).max()),"macro_f1":replay["macro_f1"],
                    "source":"fresh local best_model reload, not the still-in-memory trained model"})
                json_atomic(run/"python_warnings.json",[{"category":w.category.__name__,"message":str(w.message)} for w in captured])
                input_check(root)  # Original package train/validation/config bytes still match.
                from verify_artifacts import verify_artifacts
                identity.update({"selected_epoch":best_epoch,"epochs_completed":len(history),"completed_validation_utc":utc_now(),
                    "validation_metrics":replay,"wall_seconds":time.monotonic()-wall_start,"test_evaluation_performed":False})
                json_atomic(run/"run_identity.json",identity)
                verification=verify_artifacts(root,run)
                if not str(verification.get("status","")).startswith("PASS"):
                    raise RuntimeError("Independent artifact verification did not pass")
                if not (run/"independent_verification.json").is_file():raise FileNotFoundError("Independent verification report missing")
                from plot_validation import make_figures
                make_figures(run)
                figures=run/"figures"
                if len(list(figures.glob("*.png")))<4 or len(list(figures.glob("*.pdf")))<4:
                    raise RuntimeError("Required reproducible validation figures were not fully exported")
                json_atomic(run/"run_status.json",{"status":"VERIFIED_VALIDATION_ARTIFACTS","selected_epoch":best_epoch,
                    "held_out_evaluation_performed":False,"new_run_not_historical_recovery":True})
                print("All selected model artifacts and independent metrics passed. Preparing the complete return ZIP.",flush=True)
        except BaseException as exc:
            json_atomic(run/"run_status.json",{"status":"FAILED_OR_INTERRUPTED","exception_type":type(exc).__name__,"message":str(exc),"timestamp_utc":utc_now()})
            (run/"failure_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
            print("Run stopped. Actual partial checkpoints/logs retained; no completion marker was published.",file=stderr)
            raise
        finally:
            if trained_model is not None:del trained_model
            if selected_model is not None:del selected_model
            torch.mps.empty_cache()
    # console.log is closed before hashing; exclude checkpoint scratch to avoid duplicate GB exports.
    artifact_hashes={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob("*")) if p.is_file() and "checkpoints" not in p.relative_to(run).parts}
    try:
        archive=package_result(root,run,artifact_hashes)
        result={"status":"COMPLETED_NEW_VALIDATION_ONLY_RUN","experiment_id":cfg["experiment_id"],
            "run_directory":run.relative_to(root).as_posix(),"input_identity":verified["input_identity"],"training_config":cfg,
            "selected_epoch":best_epoch,"selected_validation_macro_f1":literal_best,"epochs_completed":len(history),
            "artifact_sha256":artifact_hashes,"return_zip":archive.relative_to(root).as_posix(),"return_zip_sha256":sha(archive),
            "return_zip_bytes":archive.stat().st_size,"model_checkpoint_complete":True,"independent_verification_passed":True,"publication_figures_exported":True,
            "hardware_experiment":"MPS_FP32_SEPARATE_FROM_NVIDIA_BF16",
            "historical_0_581140_is_reference_not_target":True,"held_out_evaluation_performed":False,"completed_utc":utc_now()}
        json_atomic(outputs/"COMPLETED.json",result,exclusive=True)
    except BaseException as exc:
        json_atomic(run/"packaging_failure.json",{"status":"RETURN_PACKAGE_OR_COMPLETION_FAILED","exception_type":type(exc).__name__,"message":str(exc)})
        raise
    print("COMPLETE. Return these actual files, plus the saved executed notebook:")
    print(archive);print(outputs/"COMPLETED.json")
    print("No held-out evaluation has been run. Freeze these artifacts before seeking separate final-evaluation approval.")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root",type=Path,default=Path(__file__).resolve().parents[1])
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs-only",action="store_true")
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if a.inputs_only:
        checked=input_check(a.package_root)
        print(json.dumps({"status":checked.get("status","PASS"),"training_rows":len(checked["train_rows"]),
            "validation_rows":len(checked["validation_rows"]),"class_weights":checked["class_weights"],
            "input_identity":checked["input_identity"],"torch_imported":False,"training_executed":False},indent=2))
    elif a.preflight:
        root=a.package_root.resolve();verified=input_check(root);check_mac_contract(verified["config"])
        if (root/"outputs/STARTED.json").exists():
            raise RuntimeError("A training run already started; preserve its actual outputs rather than launching another smoke")
        reserved=root/"outputs"/("preflight_only_"+uuid.uuid4().hex)
        with reserve_session(root,reserved):
            checked=checked_full_preflight(root,verified)
            print(json.dumps(checked,indent=2))
    else:execute(a.package_root)


if __name__=="__main__":main()
