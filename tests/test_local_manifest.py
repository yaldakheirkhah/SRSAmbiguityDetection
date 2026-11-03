"""Synthetic release-preparation controls only; no research input or ML stack."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]


def source_module(relative,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest():
    return source_module('experiments/legacy_25k_grouped/scripts/prepare_local_manifest.py','synthetic_local_manifest')


def synthetic_legacy(root):
    config={'benchmark_id':'SYNTHETIC_UNIT_FIXTURE_ONLY','eligible_for_original_frozen_test':False}
    (root/'config').mkdir();(root/'data').mkdir();(root/'reports').mkdir()
    raw=(json.dumps(config)+'\n').encode();(root/'config/protocol.json').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    (root/'config/protocol.sha256').write_text(digest+'  protocol.json\n')
    (root/'data/train.csv').write_text('invented_fixture_only\nalpha\n')
    (root/'data/validation.csv').write_text('invented_fixture_only\nbeta\n')
    gate={'status':'PASS_PARTIAL_RECOVERY','approved_for_gpu_package':True,
        'eligible_for_original_frozen_test':False,'benchmark_id':config['benchmark_id'],
        'protocol_sha256':digest,'checks':{'synthetic_control_flow_fixture_only':True}}
    (root/'reports/QUALITY_GATE.json').write_text(json.dumps(gate))
    return gate


def test_default_preparation_refuses_before_filesystem(manifest,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No path may be accessed')
    for method in ['resolve','stat','open','exists','is_symlink','is_file']:
        monkeypatch.setattr(Path,method,forbidden)
    with pytest.raises(PermissionError,match='nothing accessed'):
        manifest.prepare('SYNTHETIC_PATH','legacy')


@pytest.mark.parametrize('execute,authorized',[(False,True),(True,False)])
def test_both_explicit_controls_required(manifest,tmp_path,execute,authorized):
    with pytest.raises(PermissionError):manifest.prepare(tmp_path,'legacy',execute=execute,authorized=authorized)
    assert list(tmp_path.iterdir())==[]


def test_missing_inputs_never_create_manifest(manifest,tmp_path):
    with pytest.raises(FileNotFoundError):manifest.prepare(tmp_path,'legacy',execute=True,authorized=True)
    assert not (tmp_path/'package_manifest.json').exists()


def test_synthetic_manifest_contains_identities_only_and_refuses_overwrite(manifest,tmp_path):
    synthetic_legacy(tmp_path)
    result=manifest.prepare(tmp_path,'legacy',execute=True,authorized=True)
    assert len(result['files'])==4
    assert result['copied_or_published_records'] is False
    text=(tmp_path/'package_manifest.json').read_text()
    assert 'alpha' not in text and 'beta' not in text
    with pytest.raises(FileExistsError):manifest.prepare(tmp_path,'legacy',execute=True,authorized=True)
    assert (tmp_path/'package_manifest.json').read_text()==text


@pytest.mark.parametrize('mutation',['failed_check','different_protocol','wrong_benchmark','test_eligible'])
def test_failed_synthetic_gate_never_creates_manifest(manifest,tmp_path,mutation):
    gate=synthetic_legacy(tmp_path)
    if mutation=='failed_check':gate['checks']['synthetic_control_flow_fixture_only']=False
    if mutation=='different_protocol':gate['protocol_sha256']='f'*64
    if mutation=='wrong_benchmark':gate['benchmark_id']='OTHER_SYNTHETIC_FIXTURE'
    if mutation=='test_eligible':gate['eligible_for_original_frozen_test']=True
    (tmp_path/'reports/QUALITY_GATE.json').write_text(json.dumps(gate))
    with pytest.raises(ValueError):manifest.prepare(tmp_path,'legacy',execute=True,authorized=True)
    assert not (tmp_path/'package_manifest.json').exists()


def test_synthetic_symlink_input_refused(manifest,tmp_path):
    synthetic_legacy(tmp_path)
    original=tmp_path/'data/train.csv';original.rename(tmp_path/'invented_fixture.csv')
    original.symlink_to(tmp_path/'invented_fixture.csv')
    with pytest.raises(ValueError,match='symlinks'):manifest.prepare(tmp_path,'legacy',execute=True,authorized=True)
    assert not (tmp_path/'package_manifest.json').exists()


def test_lineage_and_legacy_use_identical_helper_source():
    legacy=ROOT/'experiments/legacy_25k_grouped/scripts/prepare_local_manifest.py'
    lineage=ROOT/'experiments/lineage_augmentation_bert_20261027/scripts/prepare_local_manifest.py'
    assert legacy.read_bytes()==lineage.read_bytes()


def synthetic_lineage(root,manifest):
    for name in ['config','data','reports']:(root/name).mkdir()
    for name in manifest.FILES['lineage']:
        if name.endswith('.csv'):(root/name).write_text('invented_fixture_only\n'+name+'\n')
    hash_of=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    config={'benchmark_id':'SYNTHETIC_LINEAGE_UNIT_FIXTURE_ONLY'}
    (root/'config/training_config.json').write_text(json.dumps(config))
    gate={'status':'PASS_SYNTHETIC_CONTROL_FLOW_ONLY','approved_for_gpu_package':True,
        'checks':{'synthetic_fixture_only':True},'protected_test_accessed':False,
        'benchmark_id':config['benchmark_id'],'training_config_sha256':hash_of('config/training_config.json'),
        'variants':{}}
    for variant in ['A','B']:
        data='data/train_A_Conservative_v1.csv' if variant=='A' else 'data/train_B_Full_Provisional_v1.csv'
        weights={'training_only':True,'training_data_path':data,'training_data_sha256':hash_of(data)}
        (root/f'config/variant_{variant}_class_weights.json').write_text(json.dumps(weights))
        gate['variants'][variant]={'sha256':hash_of(data)}
    (root/'reports/DATASET_VERIFICATION.json').write_text(json.dumps(gate))
    return {name:hash_of(name) for name in manifest.EXPECTED_ORIGINAL}


def test_lineage_synthetic_bindings_only(manifest,tmp_path,monkeypatch):
    # Replace expected hashes ONLY inside this isolated unit-test module instance.
    expected=synthetic_lineage(tmp_path,manifest)
    monkeypatch.setattr(manifest,'EXPECTED_ORIGINAL',expected)
    result=manifest.prepare(tmp_path,'lineage',execute=True,authorized=True)
    assert len(result['files'])==8 and result['training_performed'] is False


@pytest.mark.parametrize('mutation',['original_hash','weight_binding','configuration_binding'])
def test_invalid_lineage_bindings_refused(manifest,tmp_path,monkeypatch,mutation):
    expected=synthetic_lineage(tmp_path,manifest)
    monkeypatch.setattr(manifest,'EXPECTED_ORIGINAL',expected)
    if mutation=='original_hash':expected['data/train_original.csv']='0'*64
    if mutation=='weight_binding':
        path=tmp_path/'config/variant_A_class_weights.json'
        record=json.loads(path.read_text());record['training_data_sha256']='0'*64
        path.write_text(json.dumps(record))
    if mutation=='configuration_binding':
        path=tmp_path/'reports/DATASET_VERIFICATION.json'
        record=json.loads(path.read_text());record['training_config_sha256']='0'*64
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):manifest.prepare(tmp_path,'lineage',execute=True,authorized=True)
    assert not (tmp_path/'package_manifest.json').exists()


def test_variant_csv_serialization_matches_original_writer(tmp_path):
    module=source_module('experiments/lineage_augmentation_bert_20261027/scripts/build_variants.py','synthetic_variant_builder')
    rows=[{'fixture':'invented, quoted "text"','Class':'1'},{'fixture':'invented café\nline','Class':'6'}]
    path=tmp_path/'synthetic_serialization.csv';module.write_csv(path,rows,['fixture','Class'])
    assert path.read_bytes()==module.csv_bytes(rows,['fixture','Class'])
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')


def test_weight_mismatch_refused_before_any_export(tmp_path,monkeypatch):
    module=source_module('experiments/lineage_augmentation_bert_20261027/scripts/build_variants.py','synthetic_variant_weights')
    monkeypatch.setattr(module,'BASE',tmp_path)
    (tmp_path/'config').mkdir();(tmp_path/'data').mkdir()
    path=tmp_path/'config/variant_A_class_weights.json';original=b'{"fixture_only": 1}\n'
    path.write_bytes(original)
    module.check_existing_weight_records({'A':{'fixture_only':1}})
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='differ'):
        module.check_existing_weight_records({'A':{'fixture_only':2}})
    assert path.read_bytes()==original and list((tmp_path/'data').iterdir())==[]
