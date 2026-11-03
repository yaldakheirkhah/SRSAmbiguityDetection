"""Static/pure numerical tests only; no Torch, training or model downloads."""
from pathlib import Path
import ast
import importlib.util
import json
import math
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("new_original_train_source",ROOT/"scripts/train_bert.py")
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def test_import_does_not_import_training_stack():
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_literal_best_distinct_from_stopping_clock():
    literal,meaningful,stale=MODULE.checkpoint_decision(.5812,.5811,.5811,3,.001)
    assert literal and not meaningful and stale==4
    assert MODULE.checkpoint_decision(.5811,.5811,.5811,2,.001)==(False,False,3)
    assert MODULE.checkpoint_decision(.59,.5811,.5811,2,.001)==(True,True,0)
    with pytest.raises(FloatingPointError):MODULE.checkpoint_decision(float('nan'),0,0,0,.001)


def test_blackwell_runtime_checks():
    assert MODULE.check_cuda_build((12,0),'12.8',['sm_120'])['capability']==[12,0]
    with pytest.raises(RuntimeError):MODULE.check_cuda_build((12,0),'12.6',['sm_120'])
    with pytest.raises(RuntimeError):MODULE.check_cuda_build((12,0),'12.8',['sm_90'])
    with pytest.raises(RuntimeError):MODULE.check_cuda_build((12,0),None,[])


def test_full_window_weighted_gradient_math():
    # Sum CE-gradient numerators / total target-class-weight denominator; uneven microbatches.
    numerators=[17.0,3.0];denominators=[5.0,1.0]
    combined=sum(numerators)/sum(denominators)
    separately_averaged=sum(n/d for n,d in zip(numerators,denominators))/2
    assert math.isclose(combined,20/6) and not math.isclose(combined,separately_averaged)
    # Last one-microbatch accumulation window is not divided by a nominal window size.
    assert sum([7.0])/sum([2.0])==3.5
    source=(ROOT/'scripts/train_bert.py').read_text()
    assert 'reduction="sum"' in source and 'parameter.grad.div_(window_denominator)' in source
    assert 'batch_number==len(train_loader)' in source
    assert source.index('parameter.grad.div_(window_denominator)') < source.index('torch.nn.utils.clip_grad_norm_')


def test_protected_or_escaped_output_path_rejected(tmp_path):
    with pytest.raises(ValueError):MODULE.safe_path(tmp_path,'../elsewhere')
    with pytest.raises(ValueError):MODULE.safe_path(tmp_path,'test.csv')
    with pytest.raises(ValueError):MODULE.safe_path(tmp_path,'nested\\artifact')


def test_pretrained_init_and_actual_export_guards_present():
    source=(ROOT/'scripts/train_bert.py').read_text()
    assert MODULE.MODEL_ID=='google-bert/bert-base-uncased'
    assert MODULE.REVISION=='86b5e0934494bd15c9632b12f734a8a67f723594'
    assert 'local_files_only=True' in source and 'safe_serialization=True' in source
    assert 'missing != {"classifier.weight","classifier.bias"}' in source
    assert source.index('verification=verify_artifacts(root,run)') < source.index('archive=package_result(root,run,artifact_hashes)')
    assert source.index('archive=package_result(root,run,artifact_hashes)') < source.index('json_atomic(outputs/"COMPLETED.json"')
    assert 'if started_marker.exists()' in source
    assert 'No held-out evaluation has been run' in source


def test_windows_artifact_paths_exported_tokenizer_and_snapshot():
    source=(ROOT/'scripts/train_bert.py').read_text()
    assert 'str(run.relative_to(root))' not in source and 'str(archive.relative_to(root))' not in source
    assert 'p.relative_to(run).as_posix()' in source and 'p.relative_to(staged).as_posix()' in source
    assert source.index('tokenizer=selected_tokenizer') < source.index('replay,_,replay_pred,replay_p,replay_logits=evaluate(selected_model)')
    assert 'source_snapshot' in source and 'data_manifest.json' in source
    assert '(outputs/"EXECUTION_LOCK.json").open("x"' in source


def test_notebook_run_all_and_unexecuted_cells():
    p=ROOT/'notebooks/Train_Original_Weighted_BERT_RTX5060Ti.ipynb'
    nb=json.loads(p.read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==5 and all(c.get('execution_count') is None and c.get('outputs')==[] for c in cells)
    for c in cells:compile(''.join(c['source']),str(p),'exec')
    text='\n'.join(''.join(c['source']) for c in cells)
    assert '"--execute"' in text and '"--preflight"' not in text
    assert 'platform.system() == "Darwin"' in text
    assert '"torch==2.9.0+cu128"' in text and 'if installed("torch") != "2.9.0+cu128"' in text
    assert 'COMPLETED.json' in text and 'FileLink' in text
    assert not any('test.csv' in ''.join(c['source']) for c in cells)
