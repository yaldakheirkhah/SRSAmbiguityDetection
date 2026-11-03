import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ambifix_hitl.contracts import LABELS,ModelUnavailable,ValidationError
from ambifix_hitl.predictors import MockPredictor,RealBERTPredictor,inspect_manifest,file_sha,allowed_path,check_reference_gate


def fixture_manifest(tmp_path):
    # Text placeholders ONLY for file-integrity/label contract tests. Never loaded as model weights.
    folder=tmp_path/"explicit_fake_checkpoint_for_unit_test";folder.mkdir()
    config={"model_type":"bert","architectures":["BertForSequenceClassification"],"max_position_embeddings":512,
        "id2label":{str(i):name for i,name in enumerate(LABELS.values())}}
    (folder/"config.json").write_text(json.dumps(config))
    for n in ["model.safetensors","tokenizer_config.json","special_tokens_map.json","vocab.txt"]:(folder/n).write_text("NOT A MODEL: software file-contract fixture only")
    m={"schema_version":1,"checkpoint_dir":folder.name,"model_id":"corrected_weighted_bert","checkpoint_version":"unit-fixture-not-trained",
        "trained_checkpoint_attestation":True,"training_data_provenance":"frozen_original_train_only","provenance_note":"Artificial author-attestation field for unit control flow; no actual checkpoint or experimental evidence",
        "model_index_to_dataset_id":[1,2,3,4,5,6],"label_names":list(LABELS.values()),
        "preprocessing":{"external_preprocessing":"none","input_field":"Text","max_length":192,"forward_precision":"fp32"},
        "files_sha256":{p.name:file_sha(p) for p in folder.iterdir()}}
    out=tmp_path/"manifest.json";out.write_text(json.dumps(m));return out,folder,m


def test_missing_model_fails_without_mock_fallback(tmp_path):
    with pytest.raises(ModelUnavailable,match="manifest"):RealBERTPredictor(tmp_path/"absent.json")


def test_file_and_class_order_contract_only(tmp_path):
    path,folder,m=fixture_manifest(tmp_path)
    _,_,fingerprint,manifest_sha=inspect_manifest(path)
    assert len(fingerprint)==64 and len(manifest_sha)==64
    with pytest.raises(ModelUnavailable,match="reference replay"):RealBERTPredictor(path)
    (folder/"vocab.txt").write_text("changed")
    with pytest.raises(ModelUnavailable,match="SHA-256"):inspect_manifest(path)


@pytest.mark.parametrize("mutation",["labels","mapping","architecture","unknown_length","contaminated","untrained"])
def test_incompatible_checkpoint_rejected(tmp_path,mutation):
    path,folder,m=fixture_manifest(tmp_path)
    if mutation=="labels":m["label_names"][1],m["label_names"][3]=m["label_names"][3],m["label_names"][1]
    if mutation=="mapping":m["model_index_to_dataset_id"]=[0,1,2,3,4,5]
    if mutation=="unknown_length":m["preprocessing"]["max_length"]=None
    if mutation=="contaminated":m["training_data_provenance"]="legacy_24949_augmented"
    if mutation=="untrained":m["trained_checkpoint_attestation"]=False
    if mutation=="architecture":
        c=json.loads((folder/"config.json").read_text());c["architectures"]=["BertForMaskedLM"]
        (folder/"config.json").write_text(json.dumps(c));m["files_sha256"]["config.json"]=file_sha(folder/"config.json")
    path.write_text(json.dumps(m))
    with pytest.raises(ModelUnavailable):inspect_manifest(path)


def test_forbidden_path_rejected_before_open(tmp_path):
    # No held-out file is created or accessed: this exercises the string-path guard.
    with pytest.raises(ModelUnavailable):allowed_path(tmp_path/"test.csv")


def test_unmanifested_tokenizer_file_rejected(tmp_path):
    path,folder,m=fixture_manifest(tmp_path)
    (folder/"added_tokens.json").write_text('{}')
    with pytest.raises(ModelUnavailable,match="not covered"):inspect_manifest(path)


def test_malformed_model_config_fails_safely(tmp_path):
    path,folder,m=fixture_manifest(tmp_path)
    (folder/"config.json").write_text('[]');m["files_sha256"]["config.json"]=file_sha(folder/"config.json")
    path.write_text(json.dumps(m))
    with pytest.raises(ModelUnavailable,match="JSON object"):inspect_manifest(path)


def test_reference_gate_pending_or_wrong_identity_rejected(tmp_path):
    p=tmp_path/"gate.json";p.write_text(json.dumps({"status":"COMPATIBILITY_ONLY_REFERENCE_PENDING"}))
    with pytest.raises(ModelUnavailable):check_reference_gate(p,"a"*64,"b"*64,"cpu")


def test_replay_rejects_mock_and_nonapproved_identity(tmp_path):
    path=Path(__file__).resolve().parents[1]/"scripts/validate_checkpoint.py"
    spec=importlib.util.spec_from_file_location("checkpoint_validator_software_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=tmp_path/"synthetic_reference.csv"
    reference.write_text("source_id,Text,Class,predicted_class,p1,p2,p3,p4,p5,p6\nsyn1,new software fixture,1,1,1,0,0,0,0,0\n")
    approval={"reference_role":"validation_reference","class_order":[1,2,3,4,5,6],"model_id":"corrected_weighted_bert",
        "reference_sha256":file_sha(reference),"expected_rows":1,"probability_tolerance":1e-4,
        "columns":{"source_id":"source_id","text":"Text","true_label":"Class","predicted_label":"predicted_class","probabilities":[f"p{i}" for i in range(1,7)]}}
    fake=SimpleNamespace(manifest={"model_id":"corrected_weighted_bert"},predict_batch=lambda texts:[MockPredictor().predict(t) for t in texts])
    with pytest.raises(ValueError,match="Mock outputs"):module.replay_reference(fake,reference,approval)
    approval["reference_sha256"]="0"*64
    with pytest.raises(ValueError,match="identity differs"):module.replay_reference(fake,reference,approval)
