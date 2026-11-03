"""Deterministic demo backend and strictly local BERT checkpoint adapter.

The UI never falls back from real to mock. Ordinary tests need no ML dependency.
"""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import math
from .contracts import LABELS, RawPrediction, ModelUnavailable, ValidationError, validate_text, canonical


def allowed_path(path):
    p=Path(path).expanduser()
    if any(part.casefold()=="test.csv" for part in p.parts):
        raise ModelUnavailable("Held-out file paths are prohibited.")
    resolved=p.resolve()
    if any(part.casefold()=="test.csv" for part in resolved.parts):
        raise ModelUnavailable("Held-out file paths are prohibited.")
    return resolved


def file_sha(path):
    p=allowed_path(path)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


class MockPredictor:
    """Hash-derived arbitrary test probabilities, without semantic rationales."""
    def predict(self,text):
        validate_text(text)
        h=hashlib.sha256(("ambifix-mock-v1\n"+text).encode()).digest()
        scores=[1+h[i] for i in range(6)];total=sum(scores)
        return RawPrediction(tuple(x/total for x in scores),"mock-sha256-v1","DEMO-NO-MODEL-WEIGHTS",True,
            {"verification_scope":"deterministic software fixture only", "semantic_inference":False,
             "truncated":False,"input_character_count":len(text)})


def inspect_manifest(manifest_path):
    path=allowed_path(manifest_path)
    if not path.is_file():raise ModelUnavailable("No checkpoint manifest was found. Supply the genuine trained model, tokenizer and provenance separately.")
    try:
        m=json.loads(path.read_text())
        if m["schema_version"]!=1 or m["model_id"] not in {"corrected_weighted_bert","lineage_variant_A"}:
            raise ValueError("Unsupported checkpoint identity/schema.")
        validate_text(m["checkpoint_version"],200)
        if "REPLACE" in m["checkpoint_version"]:raise ValueError("Template values are not checkpoint evidence.")
        if m["trained_checkpoint_attestation"] is not True or m["training_data_provenance"] not in {"frozen_original_train_only","lineage_original_training_parents_only"}:
            raise ValueError("Author-supplied trained/train-only provenance attestation is required; legacy augmented checkpoints are prohibited.")
        validate_text(m["provenance_note"],5000)
        if m["model_index_to_dataset_id"]!=[1,2,3,4,5,6]:raise ValueError("Model index mapping must be explicitly 0..5 -> dataset 1..6.")
        if m["label_names"]!=list(LABELS.values()):raise ValueError("Six-class names/order differ from trusted exported mapping.")
        if m["preprocessing"]["external_preprocessing"]!="none" or m["preprocessing"]["input_field"]!="Text":
            raise ValueError("Adapter supports raw original Text without external transformations.")
        length=m["preprocessing"]["max_length"]
        if type(length) is not int or not 8<=length<=512:raise ValueError("Supply actual maximum token length in [8,512]; it is not inferred.")
        if m["preprocessing"]["forward_precision"] not in {"fp32","cuda_bf16","cuda_fp16"}:raise ValueError("Unsupported explicit forward precision.")
        folder=allowed_path(path.parent/m["checkpoint_dir"])
        required={"config.json","model.safetensors","tokenizer_config.json","special_tokens_map.json"}
        files=m["files_sha256"]
        if not isinstance(files,dict) or not required<=set(files) or not ({"tokenizer.json","vocab.txt"}&set(files)):
            raise ValueError("A complete safetensors classifier/config/local tokenizer is required; .bin/sharded/adapter-only formats are not silently converted.")
        for name in ["tokenizer.json","vocab.txt","added_tokens.json"]:
            if (folder/name).exists() and name not in files:
                raise ValueError(f"Tokenizer artifact is present but not covered by the manifest: {name}")
        for name,digest in files.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).name!=name:
                raise ValueError("Checkpoint files must be immediate local filenames.")
            if Path(name).suffix not in {".json",".txt",".safetensors"}:raise ValueError("Unsupported checkpoint artifact type.")
            p=allowed_path(folder/name)
            if p.parent!=folder or (folder/name).is_symlink():raise ValueError("Checkpoint files may not escape the supplied folder.")
            if not isinstance(digest,str) or len(digest)!=64 or file_sha(p)!=digest:
                raise ValueError(f"Missing or SHA-256 mismatching checkpoint artifact: {name}")
        c=json.loads((folder/"config.json").read_text())
        if not isinstance(c,dict):raise ValueError("Model configuration must be a JSON object.")
        if c.get("model_type")!="bert" or c.get("architectures")!=["BertForSequenceClassification"]:
            raise ValueError("Expected a complete BertForSequenceClassification checkpoint, not pretrained-only or another architecture.")
        ids=c.get("id2label",{})
        if not isinstance(ids,dict) or set(ids)!=set(str(i) for i in range(6)):raise ValueError("Model config must declare six output labels.")
        names=[ids[str(i)] for i in range(6)]
        if names not in [list(LABELS.values()),[f"LABEL_{i}" for i in range(6)]]:
            raise ValueError("Named checkpoint label order conflicts with the explicit mapping.")
        if "label2id" in c and c["label2id"]!={name:i for i,name in enumerate(names)}:
            raise ValueError("Checkpoint label2id and id2label are inconsistent.")
        if length>c.get("max_position_embeddings",0):raise ValueError("Maximum length exceeds the model's positional capacity.")
        fingerprint=hashlib.sha256(canonical(files).encode()).hexdigest()
    except (KeyError,ValueError,TypeError,OSError) as exc:
        raise ModelUnavailable(f"Checkpoint is absent or incompatible: {exc}") from exc
    return m,folder,fingerprint,file_sha(path)


def check_reference_gate(gate_path, manifest_sha, fingerprint, device):
    try:
        g=json.loads(allowed_path(gate_path).read_text())
        # Boundary checks must remain active under python -O/PYTHONOPTIMIZE.
        # This is a software compatibility gate, not proof of training provenance.
        if not isinstance(g,dict) or g.get("status")!="PASS_REFERENCE_REPLAY":
            raise ValueError("Successful reference replay is required.")
        if g["manifest_sha256"]!=manifest_sha or g["checkpoint_fingerprint"]!=fingerprint:
            raise ValueError("Reference replay checkpoint identity differs.")
        if g["device"]!=device:
            raise ValueError("Reference replay must cover the selected device/precision contract.")
        order=g["class_order"]
        if (not isinstance(order,list) or any(type(v) is not int for v in order)
                or order!=[1,2,3,4,5,6] or g["reference_role"]!="validation_reference"):
            raise ValueError("Only the explicit six-class validation reference is accepted.")
        if type(g["rows"]) is not int or g["rows"]<=0:
            raise ValueError("Reference row count must be a positive integer.")
        if type(g["prediction_mismatches"]) is not int or g["prediction_mismatches"]!=0:
            raise ValueError("Reference predictions must match exactly.")
        tolerance=g["probability_tolerance"];error=g["maximum_probability_abs_error"]
        if (type(tolerance) not in (int,float) or not math.isfinite(tolerance)
                or not 0<=tolerance<=0.01):
            raise ValueError("Reference tolerance must be finite and in [0,0.01].")
        if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=tolerance:
            raise ValueError("Reference probability error exceeds the finite declared tolerance.")
        digest=g["approved_reference_sha256"]
        if (not isinstance(digest,str) or len(digest)!=64
                or any(c not in "0123456789abcdef" for c in digest)
                or digest!=g["actual_reference_sha256"]):
            raise ValueError("Approved and actual reference identities must match.")
        if type(g["schema_version"]) is not int or g["schema_version"]!=1:
            raise ValueError("Unsupported reference replay schema.")
    except (KeyError,TypeError,ValueError,OSError) as exc:
        raise ModelUnavailable(f"A matching successful approved-reference replay report is required: {exc}") from exc
    return g


class RealBERTPredictor:
    def __init__(self,manifest_path,gate_path=None,device="cpu",*,_verification_only=False):
        if device not in {"cpu","cuda"}:raise ModelUnavailable("Select explicit CPU or CUDA; no device fallback.")
        m,folder,fingerprint,manifest_sha=inspect_manifest(manifest_path)
        self.manifest,self.fingerprint,self.manifest_sha,self.device=m,fingerprint,manifest_sha,device
        self.gate=None
        if not _verification_only:
            if not gate_path:raise ModelUnavailable("Real mode requires a successful reference replay report.")
            self.gate=check_reference_gate(gate_path,manifest_sha,fingerprint,device)
        precision=m["preprocessing"]["forward_precision"]
        if precision!="fp32" and device!="cuda":raise ModelUnavailable("Recorded CUDA mixed precision cannot silently run as CPU FP32.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification,AutoTokenizer
            if device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA is unavailable; no automatic CPU fallback.")
            if precision=="cuda_bf16" and not torch.cuda.is_bf16_supported():raise ValueError("Recorded BF16 is unavailable.")
            self.tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,use_fast=True)
            model,info=AutoModelForSequenceClassification.from_pretrained(str(folder),local_files_only=True,
                trust_remote_code=False,use_safetensors=True,output_loading_info=True,torch_dtype=torch.float32,
                attn_implementation="eager")
            if any(info.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]):
                raise ValueError(f"Incomplete/mismatching weights would initialize or ignore tensors: {info}")
            if model.num_labels!=6:raise ValueError("The loaded classifier does not have six classes.")
            self.torch=torch;self.model=model.to(device).eval()
            if device=="cuda":
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        except Exception as exc:
            raise ModelUnavailable(f"Local BERT loading failed: {type(exc).__name__}: {exc}") from exc

    def predict_batch(self,texts):
        if not 1<=len(texts)<=16:raise ValidationError("Inference batches must contain 1..16 requirements.")
        for text in texts:validate_text(text)
        t=self.torch
        maxlen=self.manifest["preprocessing"]["max_length"]
        counts=[len(self.tokenizer.encode(text,add_special_tokens=True,truncation=False)) for text in texts]
        encoded=self.tokenizer(texts,return_tensors="pt",padding=True,truncation=True,max_length=maxlen)
        encoded={k:v.to(self.device) for k,v in encoded.items()}
        precision=self.manifest["preprocessing"]["forward_precision"]
        context=t.autocast("cuda",dtype=t.bfloat16 if precision=="cuda_bf16" else t.float16) if precision!="fp32" else nullcontext()
        try:
            with t.inference_mode(),context:
                logits=self.model(**encoded).logits.float().cpu()
            # Softmax in CPU FP64, without claiming FP64 model inference.
            probs=t.softmax(logits.double(),dim=-1).tolist()
            return [RawPrediction(tuple(p),self.manifest["model_id"],self.manifest["checkpoint_version"]+":"+self.fingerprint,False,
                {"checkpoint_fingerprint":self.fingerprint,"manifest_sha256":self.manifest_sha,
                 "verification_scope":"approved-reference replay" if self.gate else "compatibility-only; not research-verified",
                 "device":self.device,"forward_precision":precision,"input_token_count":n,
                 "max_length":maxlen,"truncated":n>maxlen,"author_training_attestation_independently_reproduced":False})
                for p,n in zip(probs,counts)]
        except Exception as exc:
            raise ModelUnavailable(f"BERT inference failed; no replacement model used: {exc}") from exc

    def predict(self,text):
        return self.predict_batch([text])[0]
