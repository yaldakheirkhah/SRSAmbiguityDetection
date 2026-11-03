"""Independent stdlib-only validation arithmetic/input/checkpoint-file audit.

No training, model download, model forward or pickle loading. This does not
independently reproduce training; checkpoint replay is separately recorded.
"""
from pathlib import Path
import argparse,array,ast,csv,json,math,struct,sys
from verify_inputs import verify_inputs,child,digest,require,LABELS


def metrics_from_counts(truth,predicted):
    require(len(truth)==len(predicted)>0,"Invalid prediction cardinality")
    cm=[[0]*6 for _ in range(6)]
    for a,b in zip(truth,predicted):
        require(type(a) is int and type(b) is int and a in range(1,7) and b in range(1,7),"Labels must be dataset IDs 1..6")
        cm[a-1][b-1]+=1
    actual=[sum(row) for row in cm];counts=[sum(row[c] for row in cm) for c in range(6)]
    p=[cm[c][c]/counts[c] if counts[c] else 0.0 for c in range(6)]
    r=[cm[c][c]/actual[c] if actual[c] else 0.0 for c in range(6)]
    f=[2*cm[c][c]/(actual[c]+counts[c]) if actual[c]+counts[c] else 0.0 for c in range(6)]
    result={'accuracy':sum(cm[i][i] for i in range(6))/len(truth),'confusion_matrix':cm,'class_order':[1,2,3,4,5,6],'zero_division':0}
    for name,values in [('precision',p),('recall',r),('f1',f)]:
        result['macro_'+name]=sum(values)/6
        result['weighted_'+name]=sum(v*n for v,n in zip(values,actual))/len(truth)
    result['per_class']={str(i+1):{'precision':p[i],'recall':r[i],'f1':f[i],'support':actual[i]} for i in range(6)}
    return result


def read_npy(path,rows,dtype):
    """Bounded parser for the two small, non-object probability/logit NPY arrays."""
    with Path(path).open('rb') as f:
        require(f.read(6)==b'\x93NUMPY',"Not an NPY array")
        version=tuple(f.read(2));require(version in {(1,0),(2,0)},"Unsupported NPY version")
        width=2 if version==(1,0) else 4
        raw=f.read(width);require(len(raw)==width,"Truncated NPY header")
        length=int.from_bytes(raw,'little');require(0<length<=4096,"Unexpected NPY header length")
        header=ast.literal_eval(f.read(length).decode('latin1'))
        require(header.get('descr')==dtype and header.get('shape')==(rows,6) and header.get('fortran_order') is False,"Array dtype/shape/order mismatch")
        data=f.read(rows*6*(4 if dtype=='<f4' else 8)+1)
    size=4 if dtype=='<f4' else 8
    require(len(data)==rows*6*size,"Array data length mismatch")
    values=array.array('f' if dtype=='<f4' else 'd');values.frombytes(data)
    if sys.byteorder!='little':values.byteswap()
    require(all(math.isfinite(x) for x in values),"Non-finite logits/probabilities")
    return [list(values[i:i+6]) for i in range(0,len(values),6)]


def inspect_safetensors(path):
    size=Path(path).stat().st_size
    with Path(path).open('rb') as f:
        length_bytes=f.read(8);require(len(length_bytes)==8,"Truncated safetensors")
        length=int.from_bytes(length_bytes,'little');require(0<length<=1_000_000,"Invalid safetensors header length")
        header=json.loads(f.read(length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    required={'bert.embeddings.word_embeddings.weight','bert.embeddings.position_embeddings.weight','bert.embeddings.token_type_embeddings.weight',
              'bert.embeddings.LayerNorm.weight','bert.embeddings.LayerNorm.bias','bert.pooler.dense.weight','bert.pooler.dense.bias','classifier.weight','classifier.bias'}
    suffixes=['attention.self.query','attention.self.key','attention.self.value','attention.output.dense','attention.output.LayerNorm',
              'intermediate.dense','output.dense','output.LayerNorm']
    for i in range(12):
        for suffix in suffixes:
            for param in ['weight','bias']:required.add(f'bert.encoder.layer.{i}.{suffix}.{param}')
    require(set(tensors)==required,"Incomplete/unexpected BERT-base classifier tensor set")
    require(tensors['classifier.weight']['shape']==[6,768] and tensors['classifier.bias']['shape']==[6],"Classifier shape mismatch")
    require(tensors['bert.embeddings.word_embeddings.weight']['shape']==[30522,768],"Wrong pretrained vocabulary/hidden size")
    require(tensors['bert.embeddings.position_embeddings.weight']['shape']==[512,768],"Wrong positional embedding shape")
    offsets=[]
    for name,t in tensors.items():
        require(t['dtype']=='F32',"Selected checkpoint must preserve FP32 parameters")
        start,end=t['data_offsets'];require(type(start) is int and type(end) is int and 0<=start<end,"Invalid tensor offsets")
        require(end-start==math.prod(t['shape'])*4,"Tensor shape/byte length mismatch")
        offsets.append((start,end))
    offsets.sort();require(offsets[0][0]==0 and offsets[-1][1]==size-8-length,"Truncated/incomplete checkpoint bytes")
    require(all(a[1]==b[0] for a,b in zip(offsets,offsets[1:])),"Overlapping/gapped tensor data")
    return {'tensor_count':len(tensors),'parameters_dtype':'F32','full_encoder_classifier_structure_checked':True,'bytes':size}


def verify_artifacts(package_root,run_dir):
    inputs=verify_inputs(package_root);cfg=inputs['config'];run=Path(run_dir).resolve()
    require(not any(p.casefold()=='test.csv' for p in run.parts),"Held-out path prohibited")
    required=['best_model/model.safetensors','best_model/config.json','best_model/tokenizer_config.json','best_model/special_tokens_map.json',
        'class_mapping.json','class_weights.json','epoch_metrics.csv','validation_predictions.csv','validation_logits_fp32.npy','validation_probabilities_fp64.npy',
        'validation_metrics.json','classification_report.json','classification_report.txt','confusion_matrix.csv','run_identity.json','environment.json','pip_freeze.txt',
        'best_training_state.pt','checkpoint_replay.json','selected_checkpoint.json','pretrained_revision.json','pretrained_loading_info.json',
        'data_manifest.json','source_snapshot/train_bert.py','source_snapshot/verify_inputs.py','source_snapshot/verify_artifacts.py']
    tokenizer=[name for name in ['best_model/tokenizer.json','best_model/vocab.txt','best_model/added_tokens.json'] if child(run,name).is_file()]
    require(any(name in tokenizer for name in ['best_model/tokenizer.json','best_model/vocab.txt']),"Matching tokenizer export missing")
    for name in required+tokenizer:require(child(run,name).is_file() and child(run,name).stat().st_size>0,"Missing/incomplete required artifact: "+name)
    def j(name):return json.loads(child(run,name).read_text())
    identity=j('run_identity.json')
    require(identity['experiment_id']==cfg['experiment_id'] and identity['input_identity']==inputs['input_identity'],"Run/input identity mismatch")
    require(identity['training_config']==cfg and identity['pretrained_revision']==cfg['pretrained_revision'] and identity['pretrained_model_id']==cfg['model_id'],"Run recipe/original initialization mismatch")
    require(digest(child(run,'data_manifest.json'))==inputs['input_identity']['data_manifest_sha256'],"Returned input manifest differs")
    for rel in ['scripts/train_bert.py','scripts/verify_inputs.py','scripts/verify_artifacts.py']:
        expected_sha=digest(child(package_root,rel))
        require(identity['source_code_sha256'].get(rel)==expected_sha and digest(child(run,'source_snapshot/'+Path(rel).name))==expected_sha,"Actual executed source differs from this release: "+rel)
    pretrained=j('pretrained_revision.json')
    require(pretrained['repository']==cfg['model_id'] and pretrained['revision']==cfg['pretrained_revision'] and pretrained['weights_sha256']==cfg['pretrained_weights_sha256'],"Original pretrained identity differs")
    require(pretrained['training_started_from_original_pretrained'] is True and pretrained['fine_tuned_initialization_used'] is False,"Fine-tuned initialization is prohibited")
    mapping=j('class_mapping.json')
    require(mapping['dataset_class_names']==cfg['dataset_class_names'] and mapping['dataset_to_model_index']==cfg['dataset_to_model_index'] and mapping['model_index_to_dataset_id']==[1,2,3,4,5,6],"Saved class mapping mismatch")
    weights=j('class_weights.json');expected=json.loads(child(package_root,'config/class_weights.json').read_text())
    require(weights==expected,"Saved class weights are not actual train-only weights")
    model=j('best_model/config.json')
    require(model.get('model_type')=='bert' and model.get('architectures')==['BertForSequenceClassification'],"Wrong selected model architecture")
    require(model.get('id2label')=={str(i):LABELS[i] for i in range(6)},"Model config class order mismatch")
    require(model.get('label2id')=={LABELS[i]:i for i in range(6)},"Model config reverse mapping mismatch")
    structure=inspect_safetensors(child(run,'best_model/model.safetensors'))
    with child(run,'validation_predictions.csv').open(newline='',encoding='utf-8-sig') as f:preds=list(csv.DictReader(f))
    val=inputs['validation_rows'];require(len(preds)==len(val),"Validation prediction cardinality mismatch")
    logits=read_npy(child(run,'validation_logits_fp32.npy'),len(val),'<f4')
    probs=read_npy(child(run,'validation_probabilities_fp64.npy'),len(val),'<f8')
    labels=[];max_softmax_error=0.0
    for i,(p,v) in enumerate(zip(preds,val)):
        require(p['source_id']==v['source_id'] and p['Text']==v['Text'] and p['true_label']==str(v['Class']),"Frozen validation ID/text/label/order mismatch")
        row=[float(p[f'p_class_{c}']) for c in range(1,7)]
        require(all(math.isfinite(x) and 0<=x<=1 for x in row) and abs(sum(row)-1)<1e-12,"Invalid saved probabilities")
        require(row==probs[i],"CSV probabilities do not exactly round-trip the FP64 NPY")
        predicted=max(range(6),key=lambda k:row[k])+1
        require(p['predicted_label']==str(predicted),"Probability class order/argmax mismatch")
        labels.append(predicted)
        exps=[math.exp(x-max(logits[i])) for x in logits[i]];total=sum(exps)
        error=max(abs(a-b/total) for a,b in zip(row,exps));max_softmax_error=max(max_softmax_error,error)
        require(error<=1e-12,"Saved logits and probabilities disagree")
    metrics=metrics_from_counts([v['Class'] for v in val],labels)
    stored=j('validation_metrics.json')
    for name in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1']:
        require(abs(stored[name]-metrics[name])<1e-12,"Reported metric mismatch: "+name)
    report=j('classification_report.json')
    for c in range(1,7):
        entry=report.get(str(c),report.get(LABELS[c-1]));require(isinstance(entry,dict),"Missing per-class report")
        for report_key,metric_key in [('precision','precision'),('recall','recall'),('f1-score','f1'),('support','support')]:
            require(abs(entry[report_key]-metrics['per_class'][str(c)][metric_key])<1e-12,"Per-class report mismatch")
    for avg in ['macro','weighted']:
        for report_key,metric_key in [('precision','precision'),('recall','recall'),('f1-score','f1')]:
            require(abs(report[avg+' avg'][report_key]-metrics[avg+'_'+metric_key])<1e-12,"Aggregate report mismatch")
    with child(run,'confusion_matrix.csv').open(newline='') as f:matrix=list(csv.reader(f))
    require(len(matrix)==7 and all(len(row) in [6,7] for row in matrix),"Invalid confusion CSV shape")
    require([[int(x) for x in row[-6:]] for row in matrix[1:]]==metrics['confusion_matrix'],"Confusion matrix mismatch")
    with child(run,'epoch_metrics.csv').open(newline='') as f:epochs=list(csv.DictReader(f))
    require(epochs and [int(r['epoch']) for r in epochs]==list(range(1,len(epochs)+1)),"Invalid epoch history")
    require(all(math.isfinite(float(r['val_macro_f1'])) for r in epochs),"Nonfinite validation history")
    expected_updates=math.ceil(math.ceil(len(inputs['train_rows'])/cfg['train_batch_size'])/cfg['gradient_accumulation'])
    require(all(int(r['successful_optimizer_steps'])==expected_updates and int(r['skipped_optimizer_steps'])==0 for r in epochs),"Incomplete/changed original-data update count")
    best=max(epochs,key=lambda r:float(r['val_macro_f1']));selected=j('selected_checkpoint.json')
    require(selected['epoch']==int(best['epoch']),"Checkpoint is not earliest literal maximum Macro F1")
    require(abs(float(best['val_macro_f1'])-metrics['macro_f1'])<1e-12 and abs(selected['metric_value']-metrics['macro_f1'])<1e-12,"Selected score does not match predictions")
    for rowkey,metrickey in [('val_weighted_f1','weighted_f1'),('val_accuracy','accuracy')]:require(abs(float(best[rowkey])-metrics[metrickey])<1e-12,"Selected epoch report differs")
    replay=j('checkpoint_replay.json')
    require(replay.get('status')=='PASS' and replay.get('selected_epoch')==selected['epoch'] and replay.get('prediction_mismatches')==0,"Fresh selected-checkpoint replay has not passed")
    require(abs(replay['macro_f1']-metrics['macro_f1'])<1e-12,"Replayed Macro F1 differs")
    logit_tolerance=1e-5+1e-6*max(abs(x) for row in logits for x in row)
    for key,bound in [('maximum_probability_abs_difference',1e-6),('maximum_logit_abs_difference',logit_tolerance)]:
        require(type(replay[key]) in (int,float) and math.isfinite(replay[key]) and 0<=replay[key]<=bound,"Replayed numeric difference exceeds its recorded contract")
    hashes={name:digest(child(run,name)) for name in required+tokenizer}
    result={'status':'PASS_INDEPENDENT_ARTIFACT_VERIFICATION','scope':'Independent saved-output arithmetic, frozen development inputs and checkpoint file structure/hashes; no model forward or training reproduced by this verifier',
        'experiment_id':cfg['experiment_id'],'input_identity':inputs['input_identity'],'metrics':metrics,'selected_epoch':selected['epoch'],
        'epochs_logged':len(epochs),'maximum_logits_softmax_error':max_softmax_error,'checkpoint_structure':structure,'artifact_sha256':hashes,
        'checks':{'exact_original_train_validation_bytes':True,'validation_ids_text_labels_and_order':True,'training_only_sqrt_weights':True,
                  'six_class_probability_order_argmax':True,'full_precision_array_csv_agreement':True,'per_class_global_metrics_and_confusion':True,
                  'earliest_literal_best_epoch':True,'complete_selected_model_and_tokenizer_export':True,'recorded_fresh_checkpoint_replay_pass':True},
        'held_out_data_accessed':False,'training_reproduced_by_verifier':False}
    out=child(run,'independent_verification.json')
    if out.exists():require(json.loads(out.read_text())==result,"Existing independent verification differs; refusing overwrite")
    else:out.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--run-dir',required=True);args=p.parse_args()
    print(json.dumps(verify_artifacts(args.root,args.run_dir),indent=2))
