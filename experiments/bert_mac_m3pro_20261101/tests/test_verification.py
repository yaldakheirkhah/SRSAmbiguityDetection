"""CPU software checks only: no model training, weights or invented run results."""
from pathlib import Path
import math,struct,sys
import pytest
from verify_inputs import verify_inputs,child
from verify_artifacts import metrics_from_counts,read_npy,inspect_safetensors,verify_artifacts

ROOT=Path(__file__).resolve().parents[1]


def npy_bytes(values,dtype,shape):
    header=repr({'descr':dtype,'fortran_order':False,'shape':shape}).encode('latin1')
    padding=64-((10+len(header)+1)%64);header+=b' '*padding+b'\n'
    packed=struct.pack('<'+('f' if dtype=='<f4' else 'd')*len(values),*values)
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+packed


@pytest.mark.skip(reason="Private development data excluded from public release; use explicit verify_inputs CLI after authorized access")
def test_actual_frozen_inputs_and_train_only_weights():
    r=verify_inputs(ROOT)
    assert len(r['train_rows'])==4731 and len(r['validation_rows'])==1015
    assert r['audit']['train']['class_counts']==[324,418,415,332,568,2674]
    assert r['audit']['validation']['class_counts']==[70,89,89,71,122,574]
    assert r['audit']['inherited_punctuation_overlap']['pairs']==10
    assert math.isclose(sum(r['class_weights'])/6,1,abs_tol=1e-14)
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules


@pytest.mark.parametrize('relative',['../escape.csv','/absolute.csv','data/test.csv'])
def test_prohibited_paths_rejected_before_read(relative):
    with pytest.raises(ValueError):child(ROOT,relative)


def test_synthetic_confusion_metrics_global_not_batch_averages():
    r=metrics_from_counts([1,2,3,4,5,6,6],[1,1,3,6,5,6,2])
    assert math.isclose(r['accuracy'],4/7)
    assert math.isclose(r['macro_f1'],19/36)
    assert math.isclose(r['weighted_f1'],11/21)
    assert r['per_class']['2']['f1']==0 and r['class_order']==[1,2,3,4,5,6]
    assert math.isclose(r['weighted_recall'],r['accuracy'])


def test_missing_predicted_classes_keep_six_class_macro():
    r=metrics_from_counts([1,2,3,4,5,6],[6,6,6,6,6,6])
    assert math.isclose(r['macro_f1'],(2/7)/6)
    assert len(r['per_class'])==6


@pytest.mark.parametrize('a,b',[([0],[1]),([1],[7]),([True],[1]),([1],[])])
def test_invalid_metric_labels_rejected(a,b):
    with pytest.raises(ValueError):metrics_from_counts(a,b)


@pytest.mark.parametrize('dtype,values',[('<f4',[0,1,2,3,4,5]),('<f8',[0.1,0.2,0.3,0.1,0.1,0.2])])
def test_small_synthetic_array_parser_roundtrip(tmp_path,dtype,values):
    p=tmp_path/'synthetic_array.npy';p.write_bytes(npy_bytes(values,dtype,(1,6)))
    r=read_npy(p,1,dtype)
    assert len(r)==1 and len(r[0])==6
    assert all(math.isclose(a,b,rel_tol=1e-6) for a,b in zip(r[0],values))


def test_array_nan_or_wrong_shape_rejected(tmp_path):
    p=tmp_path/'synthetic_invalid.npy';p.write_bytes(npy_bytes([float('nan')]*6,'<f4',(1,6)))
    with pytest.raises(ValueError,match='Non-finite'):read_npy(p,1,'<f4')
    p.write_bytes(npy_bytes([0]*6,'<f4',(2,3)))
    with pytest.raises(ValueError,match='shape'):read_npy(p,1,'<f4')


def test_incomplete_weights_cannot_pass(tmp_path):
    p=tmp_path/'truncated_tensor_file';p.write_bytes(b'')
    with pytest.raises(ValueError,match='Truncated'):inspect_safetensors(p)


def test_no_real_run_cannot_produce_verified_completion(tmp_path, monkeypatch):
    # Isolate missing-artifact refusal from private development-data verification.
    monkeypatch.setitem(verify_artifacts.__globals__, "verify_inputs", lambda root: {"config": {}})
    with pytest.raises(ValueError,match='tokenizer export missing'):verify_artifacts(ROOT,tmp_path)
    assert not (tmp_path/'independent_verification.json').exists()
