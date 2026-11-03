"""Hash an actual owner-supplied local checkpoint; does not prove training."""
from pathlib import Path
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ambifix_hitl.contracts import LABELS
from ambifix_hitl.predictors import allowed_path,file_sha,inspect_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--model-id",choices=["corrected_weighted_bert","lineage_variant_A"],required=True)
    p.add_argument("--checkpoint-version",required=True)
    p.add_argument("--max-length",type=int,required=True)
    p.add_argument("--precision",choices=["fp32","cuda_bf16","cuda_fp16"],required=True)
    p.add_argument("--training-provenance",choices=["frozen_original_train_only","lineage_original_training_parents_only"],required=True)
    p.add_argument("--provenance-note",required=True)
    p.add_argument("--attest-trained-train-only",action="store_true",required=True)
    a=p.parse_args();folder=allowed_path(a.checkpoint_dir);out=allowed_path(a.output)
    if out.exists():p.error("Refusing to overwrite an existing model manifest.")
    required=["config.json","model.safetensors","tokenizer_config.json","special_tokens_map.json"]
    names=required+[n for n in ["tokenizer.json","vocab.txt","added_tokens.json"] if (folder/n).is_file()]
    files={name:file_sha(folder/name) for name in names}
    value={"schema_version":1,"checkpoint_dir":os.path.relpath(folder,out.parent),"model_id":a.model_id,
        "checkpoint_version":a.checkpoint_version,"trained_checkpoint_attestation":a.attest_trained_train_only,
        "training_data_provenance":a.training_provenance,"provenance_note":a.provenance_note,
        "model_index_to_dataset_id":[1,2,3,4,5,6],"label_names":list(LABELS.values()),
        "preprocessing":{"input_field":"Text","external_preprocessing":"none","max_length":a.max_length,"forward_precision":a.precision},
        "files_sha256":files}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:f.write(json.dumps(value,indent=2)+"\n")
    _,_,fingerprint,_=inspect_manifest(out)
    print(json.dumps({"status":"FILES_AND_CONFIGURATION_CHECKED_NOT_REFERENCE_VERIFIED","manifest":str(out),
        "checkpoint_fingerprint":fingerprint,"training_provenance":"author attestation, not independently reproduced",
        "next":"Run validate_checkpoint.py with approved reference artifacts before enabling the real UI."},indent=2))


if __name__=="__main__":main()
