"""Offline best-checkpoint inference restricted to the frozen validation file.

This executable deliberately has no arbitrary dataset-path or held-out option.
It never fits a vocabulary, weights, model, or other preprocessing.
"""
import argparse
import json
import numpy as np
import torch
from common import BASE, ROOT, sha, write_json, encode, loader, make_model, evaluate, metrics, write_predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--architecture",required=True,choices=["cnn","lstm"])
    args=parser.parse_args()
    out=BASE/"runs"/f"glove_{args.architecture}_seed42_v1"
    done=json.loads((out/"COMPLETED.json").read_text())
    assert done["status"]=="COMPLETED_VALIDATION_ONLY"
    for f in ["configuration.json","vocabulary.json","tokenizer.json","best_checkpoint.pt"]:
        assert sha(out/f)==done["artifacts_sha256"][f]
    cfg=json.loads((out/"configuration.json").read_text())
    val_path=ROOT/"data/splits/validation.csv"
    assert sha(val_path)==cfg["dataset"]["validation_sha256"]
    import pandas as pd
    validation=pd.read_csv(val_path,dtype={"source_id":str},keep_default_na=False)
    assert len(validation)==1015 and validation.source_id.is_unique
    vocabulary=json.loads((out/"vocabulary.json").read_text())
    checkpoint=torch.load(out/"best_checkpoint.pt",map_location="cpu",weights_only=True)
    assert checkpoint["vocabulary_sha256"]==sha(out/"vocabulary.json")
    assert checkpoint["configuration_sha256"]==sha(out/"configuration.json")
    assert checkpoint["validation_sha256"]==sha(val_path)
    matrix=checkpoint["model_state_dict"]["embedding.weight"].numpy().astype(np.float32,copy=True)
    device=done["device"]
    if device=="mps":
        assert torch.backends.mps.is_available(),"Recorded MPS backend required for this replay; no silent device change"
    model=make_model(args.architecture,matrix,cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    sequences=encode(validation.Text,vocabulary,cfg["tokenizer"]["max_tokens"])
    dl=loader(sequences,validation.Class,cfg["training"]["evaluation_batch_size"],False,cfg["seed"])
    logits,p,predicted=evaluate(model,dl,device)
    recorded=pd.read_csv(out/"validation_predictions.csv",dtype={"source_id":str})
    assert recorded.source_id.tolist()==validation.source_id.tolist()
    assert np.array_equal(recorded.predicted_label,predicted)
    result=metrics(validation.Class,predicted)
    dest=out/"standalone_validation_replay"
    dest.mkdir(exist_ok=False)
    write_predictions(dest,validation,logits,p,predicted,cfg)
    write_json(dest/"metrics.json",result,exclusive=True)
    write_json(dest/"REPLAY.json",{"status":"PASS","device":device,"best_epoch":checkpoint["epoch"],
        "original_predictions_match":True,"training_or_fitting_performed":False,
        "validation_sha256":sha(val_path),"checkpoint_sha256":sha(out/"best_checkpoint.pt")},exclusive=True)
    print(json.dumps({"status":"PASS","architecture":args.architecture,"validation_macro_f1":result["macro_f1"]}))


if __name__=="__main__":
    main()
