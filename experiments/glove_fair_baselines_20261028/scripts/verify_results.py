"""Independent sklearn + integer-confusion verification, no training module import.

Run with the repository's existing .venv (sklearn), not necessarily training Python.
Only frozen train/validation and saved predictions are read; no held-out access.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
ORDER = [1,2,3,4,5,6]


def digest(path):
    path = Path(path)
    assert path.name.casefold() != "test.csv", "Held-out file prohibited"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output(path, value):
    assert not path.exists(), f"Do not overwrite verification evidence: {path}"
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def from_predictions(y, p):
    report = classification_report(y, p, labels=ORDER, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=ORDER)
    # Second derivation uses integer counts rather than sklearn's F1 implementation.
    per_class = []
    for i in range(6):
        tp = int(cm[i,i]); actual = int(cm[i].sum()); predicted = int(cm[:,i].sum())
        per_class.append((tp/predicted if predicted else 0.0, tp/actual if actual else 0.0,
                          2*tp/(actual+predicted) if actual+predicted else 0.0, actual))
    result = {"accuracy": float(accuracy_score(y,p))}
    for avg in ["macro","weighted"]:
        a,b,c,_ = precision_recall_fscore_support(y,p,labels=ORDER,average=avg,zero_division=0)
        result.update({f"{avg}_precision": float(a), f"{avg}_recall": float(b), f"{avg}_f1": float(c)})
        manual = np.mean([x[2] for x in per_class]) if avg=="macro" else sum(x[2]*x[3] for x in per_class)/len(y)
        assert abs(c-manual) < 1e-14
    assert abs(result["weighted_recall"]-result["accuracy"]) < 1e-14
    return result,report,cm


def check_preparation(train, val, cfg):
    prep = json.loads((BASE/"artifacts/preparation_manifest.json").read_text())
    assert prep["configuration_sha256"] == digest(BASE/"config/experiment.json")
    for filename, sha in prep["files"].items():
        assert digest(BASE/"artifacts"/filename) == sha
    # Rebuild vocabulary independently, from train alone, without common.py.
    def tokens(t): return re.findall(r"\w+|[^\w\s]", t.lower(), flags=re.UNICODE)
    counter = Counter(token for text in train.Text for token in tokens(text))
    expected = {"<PAD>":0,"<UNK>":1}
    expected.update({word:i+2 for i,word in enumerate(sorted(counter,key=lambda w:(-counter[w],w)))})
    actual = json.loads((BASE/"artifacts/vocabulary.json").read_text())
    assert actual == expected
    val_only = {w for t in val.Text for w in tokens(t)} - set(counter)
    assert not (val_only & set(actual)), "Validation-only tokens expanded vocabulary"
    counts = np.array([sum(train.Class==c) for c in ORDER])
    w = np.sqrt(len(train)/(6*counts));w /= w.mean()
    wr = json.loads((BASE/"artifacts/class_weights.json").read_text())
    assert wr["counts"] == counts.tolist() and np.array_equal(w,wr["weights"])
    membership = pd.read_csv(BASE/"artifacts/training_membership.csv",dtype={"source_id":str})
    validation_membership = pd.read_csv(BASE/"artifacts/validation_membership.csv",dtype={"source_id":str})
    assert list(membership.source_id)==list(train.source_id) and list(membership.Class)==list(train.Class)
    assert list(validation_membership.source_id)==list(val.source_id) and list(validation_membership.Class)==list(val.Class)
    assert not(set(train.source_id)&set(val.source_id))
    source = json.loads((BASE/"provenance/glove_source.json").read_text())
    assert source["archive_sha256"]==cfg["embedding"]["archive_sha256"] and source["revision"]==cfg["embedding"]["revision"]
    # Publisher-hashed original text member + exact independently reconstructed matrix.
    assert digest(BASE/"embeddings"/source["member"]) == source["member_sha256"]
    matrix = np.load(BASE/"artifacts/initial_embedding_matrix_fp32.npy",allow_pickle=False)
    reference = np.random.default_rng(cfg["seed"]).normal(0,0.05,matrix.shape).astype(np.float32)
    reference[0] = 0
    found = set();total = 0
    with (BASE/"embeddings"/source["member"]).open(encoding="utf-8") as f:
        for line in f:
            total+=1
            word,values=line.rstrip("\n").split(" ",1)
            if word in expected:
                reference[expected[word]]=np.fromstring(values,sep=" ",dtype=np.float32)
                found.add(word)
    assert np.array_equal(matrix,reference) and total==400001
    assert found == set(json.loads((BASE/"artifacts/glove_matched_training_tokens.json").read_text()))
    assert len(found)==prep["glove_matched_training_types"]>0
    return {"status":"PASS", "vocabulary_rebuilt_from_train_only":True,"validation_only_token_types_excluded":len(val_only),
        "class_weights_recomputed_from_train_only":True,"original_membership_and_order_match":True,
        "no_train_validation_ID_overlap":True,"embedding_matrix_rebuilt_exactly_from_original_GloVe_member":True,
        "train_only_oov_seeded_initialization_verified":True,"glove_training_vocabulary_matches":len(found)}


def check_run(architecture,val,cfg):
    out=BASE/"runs"/f"glove_{architecture}_seed42_v1"
    done=json.loads((out/"COMPLETED.json").read_text())
    assert done["status"]=="COMPLETED_VALIDATION_ONLY"
    for filename,sha in done["artifacts_sha256"].items():
        assert digest(out/filename)==sha,filename
    assert json.loads((out/"configuration.json").read_text())==cfg
    identity=json.loads((out/"run_identity.json").read_text())
    assert identity["config_sha256"]==digest(BASE/"config/experiment.json")==digest(out/"configuration.json")
    assert identity["training_sha256"]==cfg["dataset"]["train_sha256"]
    assert identity["validation_sha256"]==cfg["dataset"]["validation_sha256"]
    for filename,sha in identity["source_code_sha256"].items():
        assert digest(BASE/"scripts"/filename)==sha,"Executed training implementation changed"
    assert json.loads((out/"class_weights.json").read_text())==json.loads((BASE/"artifacts/class_weights.json").read_text())
    assert digest(out/"vocabulary.json")==digest(BASE/"artifacts/vocabulary.json")
    assert json.loads((out/"tokenizer.json").read_text())==cfg["tokenizer"]
    d=pd.read_csv(out/"validation_predictions.csv",dtype={"source_id":str},keep_default_na=False,float_precision="round_trip")
    assert len(d)==1015 and d.source_id.is_unique
    assert d.source_id.tolist()==val.source_id.tolist(),"Validation row order/IDs differ"
    assert d.Text.tolist()==val.Text.tolist(),"Validation texts differ"
    assert d.true_label.tolist()==val.Class.tolist(),"Validation labels differ"
    assert set(d.predicted_label)<=set(ORDER)
    probs=d[[f"p_class_{c}" for c in ORDER]].to_numpy(dtype=np.float64)
    lossless=np.load(out/"validation_probabilities_fp64.npy",allow_pickle=False)
    logits=np.load(out/"validation_logits_fp32.npy",allow_pickle=False)
    assert probs.shape==(1015,6) and np.array_equal(probs,lossless)
    assert logits.dtype==np.float32 and lossless.dtype==np.float64
    assert np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
    assert np.allclose(probs.sum(1),1,rtol=0,atol=1e-14)
    x=logits.astype(np.float64);x-=x.max(1,keepdims=True);x=np.exp(x);x/=x.sum(1,keepdims=True)
    assert np.allclose(x,probs,rtol=0,atol=1e-15)
    assert np.array_equal(probs.argmax(1)+1,d.predicted_label)
    assert np.array_equal(probs.max(1),d.confidence)
    assert np.array_equal(d.correct.to_numpy(),d.true_label.to_numpy()==d.predicted_label.to_numpy())
    names=cfg["labels"]["names"]
    assert list(d.true_name)==[names[c-1] for c in d.true_label]
    assert list(d.predicted_name)==[names[c-1] for c in d.predicted_label]
    results,classification,cm=from_predictions(d.true_label,d.predicted_label)
    original=json.loads((out/"validation_metrics.json").read_text())
    for k,v in results.items():
        assert abs(v-original[k])<1e-14,(k,v,original[k])
    assert np.array_equal(cm,original["confusion_matrix"])
    assert np.array_equal(cm,pd.read_csv(out/"confusion_matrix.csv",index_col=0).to_numpy())
    for c in ORDER:
        for name,newname in [("precision","precision"),("recall","recall"),("f1-score","f1"),("support","support")]:
            assert abs(classification[str(c)][name]-original["per_class"][str(c)][newname])<1e-14
    history=pd.read_csv(out/"epoch_history.csv",float_precision="round_trip")
    assert history.epoch.tolist()==list(range(1,len(history)+1))
    epoch=int(history.loc[history.validation_macro_f1.idxmax(),"epoch"])
    assert epoch==done["selected_epoch"]
    assert abs(results["macro_f1"]-history.loc[history.epoch==epoch,"validation_macro_f1"].iloc[0])<1e-14
    for m in ["accuracy","weighted_f1"]:
        assert abs(results[m]-history.loc[history.epoch==epoch,"validation_"+m].iloc[0])<1e-14
    assert int(history.optimizer_updates.min())==int(np.ceil(4731/cfg["training"]["batch_size"]))
    replay=json.loads((out/"checkpoint_replay.json").read_text())
    assert replay["status"]=="PASS" and replay["prediction_mismatches"]==0
    verification={"status":"PASS_INDEPENDENT_SKLEARN_AND_CONFUSION_COUNT_RECALCULATION", "architecture":architecture,
        "metrics":results,"classification_report":classification,"confusion_matrix":cm.tolist(),
        "selected_epoch":epoch,"epochs_completed":len(history),"device":done["device"],
        "checks":{"exact_frozen_validation_rows_ids_text_labels_order":True,"class_order":ORDER,"zero_division":0,
            "probability_csv_lossless_npy_exact_roundtrip":True,"probabilities_match_saved_logits":True,
            "argmax_labels_match":True,"saved_metrics_and_confusion_matrix_agree":True,
            "best_checkpoint_is_earliest_literal_maximum_macro_f1":True,"saved_checkpoint_replay_passed":True,
            "all_run_artifact_hashes_match":True},
        "predictions_sha256":digest(out/"validation_predictions.csv"),"checkpoint_sha256":digest(out/"best_checkpoint.pt")}
    output(out/"INDEPENDENT_VERIFICATION.json",verification)
    return verification


def references(val):
    # Recalculate both references from their existing saved predictions, without refitting.
    result={}
    tf=ROOT/"reports/REVISION_AUDIT_TFIDF_VALIDATION_PREDICTIONS.csv"
    d=pd.read_csv(tf,dtype={"source_id":str},keep_default_na=False)
    assert len(d)==len(val) and d.source_id.tolist()==val.source_id.tolist()
    assert d.y_true.tolist()==val.Class.tolist()
    scores,report,cm=from_predictions(d.y_true,d.y_pred)
    result["tfidf_logistic_regression"]={"status":"Existing audit-run predictions independently recalculated; not refitted here",
        "metrics":scores,"prediction_file":str(tf.relative_to(ROOT)),"prediction_sha256":digest(tf),"row_alignment":"exact IDs/labels/order; text absent from historical prediction CSV; original Text input is documented by its source script and frozen validation hash"}
    bert=ROOT/"experiments/weighted_bert/runs/bert_sqrtweighted_max50_20261018_214144/validation_predictions.csv"
    d=pd.read_csv(bert,dtype={"source_id":str},keep_default_na=False)
    assert len(d)==len(val) and d.source_id.tolist()==val.source_id.tolist()
    assert d.Class.tolist()==val.Class.tolist() and d.Text.tolist()==val.Text.tolist()
    scores,report,cm=from_predictions(d.Class,d.predicted_class)
    result["weighted_bert"]={"status":"Existing external-run predictions independently recalculated; training/checkpoint not reproduced",
        "metrics":scores,"prediction_file":str(bert.relative_to(ROOT)),"prediction_sha256":digest(bert),"row_alignment":"exact IDs/text/labels/order",
        "limitations":"Actual executed BERT training recipe, original checkpoint and environment remain missing; no architecture-only causal or significance claim"}
    return result


def main():
    cfg=json.loads((BASE/"config/experiment.json").read_text())
    frames=[]
    for role in ["train","validation"]:
        p=ROOT/cfg["dataset"][role]
        assert p.name==role+".csv" and p.name in {"train.csv","validation.csv"}
        assert digest(p)==cfg["dataset"][role+"_sha256"]
        frames.append(pd.read_csv(p,dtype={"source_id":str},keep_default_na=False))
    train,val=frames
    prep=check_preparation(train,val,cfg)
    models={a:check_run(a,val,cfg) for a in ["cnn","lstm"]}
    refs=references(val)
    result={"status":"PASS", "verification_environment":{"python":sys.version,"numpy":np.__version__,
        "pandas":pd.__version__,"scikit-learn":importlib.metadata.version("scikit-learn")},
        "scope":"Independent fitting audit, saved predictions/metrics and checkpoint replay evidence; no test access or model retraining",
        "preparation":prep,"new_models":models,"references":refs}
    output(BASE/"reports/INDEPENDENT_COMPARISON_VERIFICATION.json",result)
    rows=[]
    for name,value in [("GloVe-CNN (new implementation)",models["cnn"]),("GloVe-LSTM (new implementation)",models["lstm"]),
                       ("TF-IDF + balanced Logistic Regression",refs["tfidf_logistic_regression"]),("Corrected weighted BERT",refs["weighted_bert"])]:
        rows.append({"model":name,**value["metrics"]})
    pd.DataFrame(rows).to_csv(BASE/"reports/validation_comparison.csv",index=False,float_format="%.17g")
    print(json.dumps({"status":"PASS","comparison":rows},indent=2))


if __name__=="__main__":
    main()
