"""Write-once A/B variants. No split generation, relabeling or model execution."""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,io,json,math,re,shutil,unicodedata
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
BT=ROOT/'experiments/lineage_backtranslation_mac_20261027'
LABELS=list(range(1,7))
EXPECTED={'train':'c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8','validation':'9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42'}
SOURCES={
 'frozen_train':ROOT/'data/splits/train.csv','frozen_validation':ROOT/'data/splits/validation.csv',
 'augmented_output':BT/'runs/full/train_augmented_PROVISIONAL.csv',
 'accepted_synthetic':BT/'runs/full/synthetic_accepted_PROVISIONAL.csv',
 'quality_flags':BT/'reports/candidate_suspicious_flags.csv',
 'generation_audit':BT/'reports/INDEPENDENT_OUTPUT_AUDIT.json',
 'generation_config':BT/'runs/full/generation_config.json',
 'translation_model_lock':BT/'runs/full/model_lock.json',
}

def safe(path):
 p=Path(path)
 if any(x.casefold()=='test.csv' for x in p.parts):raise ValueError('Protected input')
 p=p.resolve()
 if any(x.casefold()=='test.csv' for x in p.parts):raise ValueError('Protected target')
 if p not in {x.resolve() for x in SOURCES.values()} and not p.is_relative_to(BASE):raise ValueError('Input not allowlisted')
 return p

def sha(path):return hashlib.sha256(safe(path).read_bytes()).hexdigest()

def read_csv(path):
 return list(csv.DictReader(io.StringIO(safe(path).read_text(encoding='utf-8-sig'),newline='')))

def norm(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()

def punct(text):return re.sub(r'[\W_]+','',norm(text),flags=re.UNICODE)

def write_csv(path,rows,fields):
 with path.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n',extrasaction='raise');w.writeheader();w.writerows(rows)

def csv_bytes(rows,fields):
 buffer=io.StringIO(newline='')
 writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n',extrasaction='raise')
 writer.writeheader();writer.writerows(rows)
 return buffer.getvalue().encode('utf-8-sig')

def check_existing_weight_records(records):
 # Check both records before any new data export; preserve supplied exact bytes.
 for name,record in records.items():
  path=BASE/'config'/f'variant_{name}_class_weights.json'
  if path.is_symlink():raise ValueError('Class-weight records may not be symlinks')
  if path.exists() and json.loads(path.read_text())!=record:
   raise ValueError('Recomputed training-only weights or data identity differ: '+name)

def write_json(path,value):
 with path.open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def duplicate(rows,key):
 groups=defaultdict(list)
 for r in rows:groups[key(r['Text'])].append(r)
 repeated=[v for v in groups.values() if len(v)>1]
 cross=[v for v in groups.values() if len({r['Class'] for r in v})>1]
 return {'duplicate_keys':len(repeated),'repeated_rows_beyond_first':sum(len(v)-1 for v in repeated),
         'rows_in_duplicate_groups':sum(map(len,repeated)),'cross_label_keys':len(cross),'rows_in_cross_label_groups':sum(map(len,cross))}

def overlap(a,b,key):
 aa=defaultdict(list);bb=defaultdict(list)
 for r in a:aa[key(r['Text'])].append(r)
 for r in b:bb[key(r['Text'])].append(r)
 common=set(aa)&set(bb)
 cross=sum(bool({r['Class'] for r in aa[k]}!={r['Class'] for r in bb[k]}) for k in common)
 return {'shared_keys':len(common),'left_rows':sum(len(aa[k]) for k in common),'right_rows':sum(len(bb[k]) for k in common),'cross_label_shared_keys':cross}

def diagnostics(rows,valid,originals):
 syn=[r for r in rows if r['is_synthetic']=='1']
 return {'training_exact':duplicate(rows,lambda s:s),'training_normalized':duplicate(rows,norm),'training_punctuation':duplicate(rows,punct),
         'training_validation_exact':overlap(rows,valid,lambda s:s),'training_validation_normalized':overlap(rows,valid,norm),
         'training_validation_punctuation':overlap(rows,valid,punct),
         'synthetic_normalized':duplicate(syn,norm),'synthetic_punctuation':duplicate(syn,punct),
         'synthetic_original_normalized':overlap(syn,originals,norm),'synthetic_original_punctuation':overlap(syn,originals,punct),
         'synthetic_validation_normalized':overlap(syn,valid,norm),'synthetic_validation_punctuation':overlap(syn,valid,punct)}

def main():
 for name in ['data','audit','reports','provenance']:
  directory=BASE/name
  if directory.is_symlink():raise ValueError('Generated directories may not be symlinks')
  directory.mkdir(exist_ok=True)
 if (BASE/'reports/DATASET_VERIFICATION.json').exists() or any((BASE/'data').iterdir()):raise SystemExit('Existing versioned data found; refuse overwrite')
 before={name:{'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for name,path in SOURCES.items()}
 assert before['frozen_train']['sha256']==EXPECTED['train'] and before['frozen_validation']['sha256']==EXPECTED['validation']
 original=read_csv(SOURCES['frozen_train']);validation=read_csv(SOURCES['frozen_validation'])
 augmented=read_csv(SOURCES['augmented_output']);syn=read_csv(SOURCES['accepted_synthetic']);flagrows=read_csv(SOURCES['quality_flags'])
 previous=json.loads(safe(SOURCES['generation_audit']).read_text());lock=json.loads(safe(SOURCES['translation_model_lock']).read_text())
 assert previous['status'].startswith('PASS_') and all(previous['checks'].values())
 assert len(original)==4731 and len(validation)==1015 and len(augmented)==7174 and len(syn)==2443
 parents={r['source_id']:r for r in original};validids={r['source_id'] for r in validation}
 assert len(parents)==4731 and len(validids)==1015 and not set(parents)&validids
 assert [{k:r[k] for k in ['source_id','Text','Class']} for r in augmented[:4731]]==original
 assert augmented[4731:]==syn
 synids={r['example_id'] for r in syn};assert len(synids)==2443
 flagids={r['example_id'] for r in flagrows};assert len(flagids)==len(flagrows)
 flagged={r['example_id']:r for r in flagrows if r['example_id'] in synids and r['review_flags'].strip()}
 assert set(flagged)=={r['example_id'] for r in flagrows if r['final_decision']=='accepted' and r['review_flags'].strip()}
 for r in syn:
  p=parents[r['source_id']]
  assert r['parent_text']==p['Text'] and r['Class']==p['Class'] and r['Class']!='6'
  assert r['label_review_status']=='UNVERIFIED' and r['is_synthetic']=='1'
  rev=lock[r['pivot_language']]
  assert r['forward_model']==rev['forward_id'] and r['reverse_model']==rev['reverse_id']
  assert r['forward_revision']==rev['forward'] and r['reverse_revision']==rev['reverse']
  if r['example_id'] in flagged:
   f=flagged[r['example_id']]
   assert all(f[k]==r[k] for k in ['source_id','Text','Class','parent_text','pivot_language'])
 assert len(flagged)==previous['review_only_flags']['accepted_rows_with_any_flag']==1270
 fields=list(augmented[0])+['quality_review_flags','quality_flag_count','variant_policy']
 variants={};diags={};weights={};counts={};members=[];excluded=[];planned_data={}
 for name in ['A','B']:
  selected=[]
  for r in augmented:
   flags=flagged.get(r['example_id'],{}).get('review_flags','') if r['is_synthetic']=='1' else ''
   keep=name=='B' or r['is_synthetic']=='0' or not flags
   enhanced={**r,'quality_review_flags':flags,'quality_flag_count':str(len(flags.split(';')) if flags else 0),
             'variant_policy':'Conservative_unflagged' if name=='A' else 'Full_Provisional'}
   if keep:selected.append(enhanced)
   if name=='A':
    members.append({'example_id':r['example_id'],'source_id':r['source_id'],'Class':r['Class'],'is_synthetic':r['is_synthetic'],
                    'included_A':str(keep),'included_B':'True','quality_review_flags':flags,'reason':'flagged_synthetic_excluded_from_A' if not keep else 'retained_original' if r['is_synthetic']=='0' else 'accepted_synthetic_no_recorded_flags'})
    if not keep:excluded.append(enhanced)
  variants[name]=selected
  assert len({r['example_id'] for r in selected})==len(selected)
  assert [{k:r[k] for k in ['source_id','Text','Class']} for r in selected[:4731]]==original
  assert all(r['source_id'] in parents and r['parent_text']==parents[r['source_id']]['Text'] and r['Class']==parents[r['source_id']]['Class'] for r in selected)
  counts[name]=[sum(int(r['Class'])==c for r in selected) for c in LABELS]
  n=len(selected);raw=[math.sqrt(n/(6*c)) for c in counts[name]];mean=sum(raw)/6
  w=[x/mean for x in raw]
  weights[name]={'variant':name,'dataset_class_order':LABELS,'training_counts':counts[name],'training_rows':n,'values':w,
                 'formula':'(N_train / (6 * n_train_class)) ** 0.5, normalized to arithmetic mean 1',
                 'training_only':True,'dtype_saved':'float64 JSON values; torch loss weights cast float32 at runtime',
                 'validation_labels_used':False,'synthetic_labels':'UNVERIFIED; originals SOURCE_LABEL'}
  filename='train_A_Conservative_v1.csv' if name=='A' else 'train_B_Full_Provisional_v1.csv'
  planned_data[filename]=csv_bytes(selected,fields)
  weights[name]['training_data_path']='data/'+filename
  weights[name]['training_data_sha256']=hashlib.sha256(planned_data[filename]).hexdigest()
  diags[name]=diagnostics(selected,validation,original)
  assert diags[name]['training_exact']['repeated_rows_beyond_first']==0
  assert diags[name]['training_normalized']['repeated_rows_beyond_first']==0
  assert diags[name]['training_validation_normalized']['shared_keys']==0
  assert diags[name]['training_punctuation']['duplicate_keys']==19 and diags[name]['training_punctuation']['cross_label_keys']==3
  assert diags[name]['training_validation_punctuation']['shared_keys']==10 and diags[name]['training_validation_punctuation']['cross_label_shared_keys']==1
  for key in ['synthetic_normalized','synthetic_punctuation']:assert diags[name][key]['repeated_rows_beyond_first']==0
  for key in ['synthetic_original_normalized','synthetic_original_punctuation','synthetic_validation_normalized','synthetic_validation_punctuation']:assert diags[name][key]['shared_keys']==0
 assert len(variants['A'])==5904 and len(variants['A'])-4731==1173 and len(variants['B'])==7174 and len(excluded)==1270
 check_existing_weight_records(weights)
 for filename,payload in planned_data.items():
  with (BASE/'data'/filename).open('xb') as f:f.write(payload)
 for name,record in weights.items():
  destination=BASE/'config'/f'variant_{name}_class_weights.json'
  if not destination.exists():write_json(destination,record)
 for source,dest in [('frozen_train','data/train_original.csv'),('frozen_validation','data/validation.csv')]:
  with (BASE/dest).open('xb') as f:f.write(safe(SOURCES[source]).read_bytes())
 write_csv(BASE/'audit/variant_membership_v1.csv',members,list(members[0]))
 write_csv(BASE/'audit/flagged_synthetic_excluded_A_v1.csv',excluded,fields)
 (BASE/'provenance').mkdir(exist_ok=True)
 for name,dest in [('quality_flags','quality_flags_from_generation.csv'),('accepted_synthetic','accepted_synthetic_from_generation.csv'),
                   ('generation_audit','generation_output_audit.json'),('generation_config','generation_config.json'),('translation_model_lock','translation_model_lock.json')]:
  with (BASE/'provenance'/dest).open('xb') as f:f.write(safe(SOURCES[name]).read_bytes())
 assert all(sha(SOURCES[k])==v['sha256'] for k,v in before.items())
 config=json.loads((BASE/'config/training_config.json').read_text())
 checks={'source_generation_independent_audit_passed':True,'original_4731_preserved_in_both_in_original_order':True,
         'frozen_train_and_validation_hashes_match':True,'validation_bytes_and_order_unchanged':True,
         'all_2443_synthetic_parents_exact_original_train_text_class':True,'all_synthetic_generation_provenance_preserved':True,
         'flagged_accepted_rows_1270_unique_and_aligned':True,'A_exact_unflagged_subset_1173':True,'B_all_accepted_2443':True,
         'all_synthetic_labels_UNVERIFIED':True,'numeric_classes_1_to_6_preserved_without_merging':True,
         'training_example_ids_unique_parent_ids_preserved':True,'no_new_exact_normalized_duplicates_conflicts_or_validation_overlaps':True,
         'no_synthetic_punctuation_duplicates_conflicts_or_overlap':True,'inherited_punctuation_risks_preserved_and_disclosed':True,
         'class_weights_computed_separately_from_training_only':True,'all_input_artifact_hashes_unchanged':True,'no_split_regeneration':True}
 out={'status':'PASS_DATASET_LINEAGE_WITH_INHERITED_RISKS','approved_for_gpu_package':True,
      'benchmark_id':config['benchmark_id'],'checks':checks,'training_config_sha256':sha(BASE/'config/training_config.json'),
      'train_original_sha256':EXPECTED['train'],'validation_sha256':EXPECTED['validation'],'protected_test_accessed':False,
      'source_inputs':before,'source_inputs_unchanged':True,'original_training_rows':4731,'validation_rows':1015,
      'accepted_synthetic_source_rows':len(syn),'accepted_synthetic_rows_with_flags':len(flagged),'excluded_synthetic_rows_A':len(excluded),
      'variants':{v:{'data_path':weights[v]['training_data_path'],'sha256':weights[v]['training_data_sha256'],'training_rows':len(variants[v]),
                     'original_rows':4731,'synthetic_rows':len(variants[v])-4731,'class_counts':{str(c):counts[v][c-1] for c in LABELS},
                     'synthetic_class_counts':dict(sorted(Counter(r['Class'] for r in variants[v] if r['is_synthetic']=='1').items())),
                     'synthetic_pivot_counts':dict(sorted(Counter(r['pivot_language'] for r in variants[v] if r['is_synthetic']=='1').items())),
                     'class_weight_path':f'config/variant_{v}_class_weights.json'} for v in ['A','B']},
      'duplicate_overlap_diagnostics':diags,'quality_policy':'A excludes every accepted synthetic with any recorded automatic review flag. B retains all accepted synthetic. Unflagged does not mean expert-verified.',
      'scientific_limits':['All synthetic labels remain UNVERIFIED, including A unflagged rows.',
       'Original punctuation-equivalent duplicate/conflict/overlap risks remain in both variants; no originals removed.',
       'Similarity keys do not prove semantic independence or absence of unknown project/document relations.',
       'Historical baseline full training recipe/checkpoint is missing; same-validation score comparisons are exploratory, not a controlled causal augmentation claim.',
       'No classifier training or model scores are produced during preparation.'],
      'training_executed':False,'new_split_created':False,'labels_changed':False}
 write_json(BASE/'reports/DATASET_VERIFICATION.json',out)
 distribution=[]
 for c in LABELS:
  distribution.append({'Class':c,'original_train':sum(int(r['Class'])==c for r in original),
                       'A_synthetic':sum(int(r['Class'])==c for r in variants['A'] if r['is_synthetic']=='1'),
                       'A_total':counts['A'][c-1],'B_synthetic':sum(int(r['Class'])==c for r in syn),
                       'B_total':counts['B'][c-1],'excluded_flagged_synthetic_A':sum(int(r['Class'])==c for r in excluded),
                       'validation_unchanged':sum(int(r['Class'])==c for r in validation)})
 write_csv(BASE/'reports/class_distributions.csv',distribution,list(distribution[0]))
 for path in (BASE/'data').iterdir():path.chmod(0o444)
 print(json.dumps({'status':out['status'],'variants':out['variants'],'excluded_synthetic_rows_A':len(excluded)},indent=2))

if __name__=='__main__':main()
