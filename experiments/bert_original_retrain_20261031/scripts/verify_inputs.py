"""CPU-only original-partition integrity/weight check. No ML imports or fitting."""
from pathlib import Path
import argparse,csv,hashlib,json,math,re,unicodedata
from collections import defaultdict

EXPECTED_HASHES={"train":"c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8",
                 "validation":"9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42"}
EXPECTED_COUNTS={"train":[324,418,415,332,568,2674],"validation":[70,89,89,71,122,574]}
LABELS=["Lexical Ambiguity","Syntactic Ambiguity","Semantic Ambiguity","Syntax Ambiguity","Pragmatic Ambiguity","Clean"]


def require(condition,message):
    if not condition:raise ValueError(message)


def child(root,relative):
    root=Path(root).resolve();p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts,"Package-relative paths only")
    require(not any(x.casefold()=='test.csv' for x in p.parts),"Held-out path prohibited")
    out=root/p
    for i in range(1,len(p.parts)+1):require(not root.joinpath(*p.parts[:i]).is_symlink(),"Package symlinks prohibited")
    resolved=out.resolve()
    require(resolved.is_relative_to(root),"Path escapes package")
    require(not any(x.casefold()=='test.csv' for x in resolved.parts),"Held-out path prohibited")
    return resolved


def digest(path):
    p=Path(path)
    require(not any(x.casefold()=='test.csv' for x in p.parts),"Held-out path prohibited")
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).split()).casefold()
def punctuation_key(text):return re.sub(r'[\W_]+','',normalized(text))


def duplicates(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[key(r['Text'])].append(r)
    multi=[r for r in groups.values() if len(r)>1]
    return {'duplicate_keys':len(multi),'extra_rows':sum(len(g)-1 for g in multi),
            'cross_label_keys':sum(len({r['Class'] for r in g})>1 for g in multi)}


def verify_inputs(package_root):
    root=Path(package_root).resolve()
    manifest=json.loads(child(root,'data_manifest.json').read_text())
    allowed={'data/train.csv','data/validation.csv','config/training_config.json','config/class_weights.json','config/pretrained_checkpoint.json'}
    require(set(manifest['files_sha256'])==allowed,"Unexpected input manifest file set")
    for rel,sha in manifest['files_sha256'].items():require(digest(child(root,rel))==sha,"Input SHA mismatch: "+rel)
    cfg=json.loads(child(root,'config/training_config.json').read_text())
    require(cfg['experiment_id']==manifest['experiment_id'],"Experiment identity mismatch")
    require(cfg['model_id']=='google-bert/bert-base-uncased' and cfg['pretrained_revision']=='86b5e0934494bd15c9632b12f734a8a67f723594',"Only pinned original pretrained BERT is permitted")
    require(cfg['augmented_examples_permitted'] is False and cfg['test_evaluation_implemented'] is False,"Invalid protocol")
    require(cfg['training_data_file']=='data/train.csv' and cfg['validation_data_file']=='data/validation.csv',"Only original packaged development inputs permitted")
    require(cfg['dataset_class_ids']==[1,2,3,4,5,6] and [cfg['dataset_class_names'][str(i)] for i in range(1,7)]==LABELS,"Class order/name mismatch")
    require(cfg['dataset_to_model_index']=={str(i):i-1 for i in range(1,7)},"Model class mapping mismatch")
    datasets={};audit={}
    for role in ['train','validation']:
        path=child(root,'data/'+role+'.csv')
        require(digest(path)==EXPECTED_HASHES[role],"Frozen original "+role+" bytes changed")
        with path.open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            require(reader.fieldnames==['source_id','Text','Class'],"Unexpected frozen schema")
            rows=list(reader)
        require(len(rows)==sum(EXPECTED_COUNTS[role]),"Unexpected frozen cardinality")
        ids=set()
        for r in rows:
            require(set(r)=={'source_id','Text','Class'} and None not in r.values(),"Missing/extra CSV fields")
            require(isinstance(r['Text'],str) and bool(r['Text'].strip()),"Blank original text")
            require(re.fullmatch(r'[0-9a-f]{16}',r['source_id']) is not None and r['source_id'] not in ids,"Invalid or duplicate source ID")
            ids.add(r['source_id'])
            require(r['Class'] in ['1','2','3','4','5','6'],"Unsupported original label")
            r['Class']=int(r['Class'])
            require(hashlib.sha256(normalized(r['Text']).encode()).hexdigest()[:16]==r['source_id'],"Original content-derived ID mismatch")
        counts=[sum(r['Class']==c for r in rows) for c in range(1,7)]
        require(counts==EXPECTED_COUNTS[role],"Frozen class supports changed")
        exact=duplicates(rows,lambda t:t);norm=duplicates(rows,normalized)
        require(exact['extra_rows']==norm['extra_rows']==0,"Unexpected exact/normalized duplicates")
        audit[role]={'rows':len(rows),'class_counts':counts,'exact':exact,'normalized':norm,
                     'punctuation_equivalent':duplicates(rows,punctuation_key),
                     'rows_with_C1_controls':sum(any(0x80<=ord(c)<=0x9f for c in r['Text']) for r in rows)}
        datasets[role]=rows
    a,b=datasets['train'],datasets['validation']
    require(not({r['source_id'] for r in a}&{r['source_id'] for r in b}),"Train/validation ID overlap")
    for transform in [lambda t:t,normalized]:
        require(not({transform(r['Text']) for r in a}&{transform(r['Text']) for r in b}),"Exact/normalized train/validation overlap")
    keys=defaultdict(list)
    for r in a:keys[punctuation_key(r['Text'])].append(r)
    pairs=[(left,right) for right in b for left in keys.get(punctuation_key(right['Text']),[])]
    audit['inherited_punctuation_overlap']={'pairs':len(pairs),'validation_rows':len({v['source_id'] for _,v in pairs}),
        'cross_label_pairs':sum(x['Class']!=y['Class'] for x,y in pairs),
        'scope':'Heuristic text equivalence, not proof of parent/project independence; frozen originals preserved'}
    record=json.loads(child(root,'config/class_weights.json').read_text())
    counts=EXPECTED_COUNTS['train'];raw=[math.sqrt(len(a)/(6*n)) for n in counts];mean=sum(raw)/6;weights=[w/mean for w in raw]
    require(record['counts']==counts and record['class_order']==[1,2,3,4,5,6],"Training class-weight support mismatch")
    require(len(record['weights'])==6 and all(math.isclose(x,y,rel_tol=0,abs_tol=1e-14) for x,y in zip(record['weights'],weights)),"Incorrect train-only square-root class weights")
    require(record['training_sha256']==EXPECTED_HASHES['train'],"Class weights not bound to frozen original training")
    identity={'experiment_id':cfg['experiment_id'],'files_sha256':manifest['files_sha256'],
              'data_manifest_sha256':digest(child(root,'data_manifest.json')),
              'ordered_train_ids_sha256':hashlib.sha256(('\n'.join(r['source_id'] for r in a)+'\n').encode()).hexdigest(),
              'ordered_validation_ids_sha256':hashlib.sha256(('\n'.join(r['source_id'] for r in b)+'\n').encode()).hexdigest()}
    return {'status':'PASS_FROZEN_ORIGINAL_INPUTS','train_rows':a,'validation_rows':b,'class_weights':weights,
            'config':cfg,'input_identity':identity,'audit':audit}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output');args=p.parse_args()
    r=verify_inputs(args.root);summary={k:v for k,v in r.items() if k not in ['train_rows','validation_rows']}
    if args.output:
        out=Path(args.output);require(not out.exists(),"Do not overwrite input verification evidence")
        out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
