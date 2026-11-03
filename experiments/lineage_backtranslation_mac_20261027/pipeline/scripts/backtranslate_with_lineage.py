#!/usr/bin/env python3
"""AmbiFix: conservative TRAIN-ONLY back-translation with auditable parent lineage.

Reads ONLY the frozen train.csv and validation.csv supplied in this package.
Does not read or require test.csv. Does NOT train a classifier.
Generated labels are INHERITED ASSUMPTIONS, not independently validated labels.
"""
from __future__ import annotations

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

import argparse
import csv
import hashlib
import json
import os
# Allow CPU fallback for unsupported Metal operations. Set before torch import.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_TRAIN_HASH = 'c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8'
EXPECTED_VALID_HASH = '9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42'
EXPECTED_TRAIN_ROWS = 4731
EXPECTED_VALID_ROWS = 1015
CLASSES = (1, 2, 3, 4, 5, 6)
MODELS = {
    'fr': ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
    'de': ('Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
    'es': ('Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'),
}


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def punct_normalize(text):
    return re.sub(r'[\W_]+', '', normalize(text), flags=re.UNICODE)


def safe_path(path):
    p = Path(path)
    if any(x.casefold() == 'test.csv' for x in p.parts):
        raise ValueError('Protected test path: no access permitted')
    resolved = p.resolve()
    if any(x.casefold() == 'test.csv' for x in resolved.parts):
        raise ValueError('Protected test target: no access permitted')
    return resolved


def digest(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def parent_id(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()[:16]


def read_csv(path):
    with safe_path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def check_inputs(train_path, valid_path):
    train_path, valid_path = safe_path(train_path), safe_path(valid_path)
    for p in [train_path, valid_path]:
        if not p.is_file():
            raise FileNotFoundError(f'Missing input {p}')
    if digest(train_path) != EXPECTED_TRAIN_HASH or digest(valid_path) != EXPECTED_VALID_HASH:
        raise ValueError('Input SHA256 differs from the frozen AmbiFix split. DO NOT regenerate/swap the splits. See README.')
    train, valid = read_csv(train_path), read_csv(valid_path)
    if len(train) != EXPECTED_TRAIN_ROWS or len(valid) != EXPECTED_VALID_ROWS:
        raise ValueError('Frozen split row counts changed.')
    for name, rows in [('train', train), ('validation', valid)]:
        if not rows or not {'source_id','Text','Class'}.issubset(rows[0]):
            raise ValueError(f'{name}: missing source_id/Text/Class columns')
        keys, ids = set(), set()
        for i, r in enumerate(rows):
            if not r['Text'].strip() or r['Class'] not in list(map(str, CLASSES)):
                raise ValueError(f'{name} invalid text or class at row {i}')
            if r['source_id'] != parent_id(r['Text']):
                raise ValueError(f'{name} source_id mismatch at row {i}')
            key=normalize(r['Text']); sid=r['source_id']
            if key in keys or sid in ids:
                raise ValueError(f'{name} repeated text/source_id')
            keys.add(key); ids.add(sid)
    if {r['source_id'] for r in train} & {r['source_id'] for r in valid}:
        raise ValueError('Train/validation source IDs overlap')
    if {normalize(r['Text']) for r in train} & {normalize(r['Text']) for r in valid}:
        raise ValueError('Train/validation normalized text overlaps')
    return train, valid


def protected_terms(text):
    t = normalize(text)
    # Distinguish negation, exception, restriction and quantification: changes can alter the requirement.
    keyphrases = [
        'not', 'never', 'without', 'unless', 'except', 'only',
        'at least', 'at most', 'no more than', 'no less than',
        'before', 'after', 'until', 'if', 'when',
    ]
    return {x for x in keyphrases if re.search(r'(?<!\w)' + re.escape(x) + r'(?!\w)', t)}


def numeric_tokens(text):
    return Counter(re.findall(r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)', text))


def candidate_reason(original, candidate, observed, blocked_punct):
    o, c = normalize(original), normalize(candidate)
    if not candidate.strip() or not c:
        return 'empty'
    if len(c) < 12 or len(c) > 2500:
        return 'invalid_length'
    if re.search(r'[\x80-\x9f]', candidate):
        return 'control_characters'
    if c == o or c in observed:
        return 'same_or_duplicate'
    ratio = len(c)/max(1,len(o))
    if ratio < .72 or ratio > 1.35:
        return 'large_length_change'
    if numeric_tokens(original) != numeric_tokens(candidate):
        return 'numbers_changed'
    if protected_terms(original) != protected_terms(candidate):
        return 'scope_condition_or_negation_changed'
    if punct_normalize(candidate) in blocked_punct:
        return 'punctuation_equivalent_duplicate'
    # Exact words in identifiers/acronyms are often functionally meaningful.
    key_ids = {s for s in re.findall(r'\b[A-Z]{2,}\b',original) if s not in {'THE','AND','FOR','GUI'}}
    if key_ids - set(re.findall(r'\b[A-Z]{2,}\b',candidate)):
        return 'acronym_changed'
    return None


def csv_write(path, rows, columns):
    path = Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    temp.replace(path)


RECIPE_VERSION = 'mac_lineage_correctness_v2'


def json_write(path, value):
    p = safe_path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(p)


def resolve_model_lock(pivots):
    from huggingface_hub import HfApi
    api = HfApi()
    result = {}
    for pivot in pivots:
        a,b = MODELS[pivot]
        result[pivot] = {'forward_id':a, 'reverse_id':b,
                         'forward':api.model_info(a, revision='main').sha,
                         'reverse':api.model_info(b, revision='main').sha}
        if not all(re.fullmatch('[0-9a-f]{40}', result[pivot][x] or '') for x in ['forward','reverse']):
            raise ValueError('Cannot freeze translation model commits')
    return result


def extract_provenance(rows, args):
    return {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'script':'backtranslate_with_lineage.py',
        'recipe_version':RECIPE_VERSION, 'recipe_sha256':digest(__file__),
        'train_sha256':EXPECTED_TRAIN_HASH, 'validation_sha256':EXPECTED_VALID_HASH,
        'seed':args.seed, 'target_per_minority_class':args.target_per_class,
        'pivot_languages':args.pivots, 'variants_per_pivot':args.variants_per_pivot,
        'max_input_tokens':args.max_input_tokens, 'max_output_tokens':args.max_output_tokens,
        'translation_pairs':{x:list(MODELS[x]) for x in args.pivots},
        'generation':{'do_sample':False, 'forward_num_beams':4,
                      'reverse_num_beams':max(4,2*args.variants_per_pivot)},
        'requested_device':args.device, 'initial_batch_size':args.batch_size,
        'model_revision_policy':'Immutable model/tokenizer commits locked before generation and bound on resume',
        'human_verified_labels':False, 'read_test':False,
        'warning':'Inherited labels may be wrong even for fluent, automatically filtered translations.'}


class MarianPair:
    def __init__(self, language, device, max_input_tokens, max_output_tokens, revisions=None, seed=42):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        from huggingface_hub import snapshot_download
        self.torch, self.device = torch, device
        self.cache_dir=Path(__file__).resolve().parents[2]/'model_cache/hub'
        self.max_input_tokens, self.max_output_tokens = max_input_tokens, max_output_tokens
        self.revisions = revisions or resolve_model_lock([language])[language]
        self.fallback_events = []
        self.memory_samples = []
        torch.manual_seed(seed)
        if hasattr(torch,'mps') and torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Precache exact locked files, then load offline. Transformers 4.51.3 otherwise
        # starts an unsolicited safetensors-conversion thread for .bin repositories.
        # Local-only loading prevents extra PR-revision weights/conversion requests.
        for direction in ['forward','reverse']:
            snapshot_download(self.revisions[direction+'_id'],revision=self.revisions[direction],cache_dir=self.cache_dir,
                              allow_patterns=['pytorch_model.bin','config.json','generation_config.json',
                                              'source.spm','target.spm','vocab.json','tokenizer_config.json',
                                              'special_tokens_map.json'],max_workers=2)
        self.forward_tokenizer = MarianTokenizer.from_pretrained(self.revisions['forward_id'], revision=self.revisions['forward'], local_files_only=True, cache_dir=self.cache_dir)
        self.forward_model = MarianMTModel.from_pretrained(self.revisions['forward_id'], revision=self.revisions['forward'],
                                                          use_safetensors=False, local_files_only=True, cache_dir=self.cache_dir).to(device).eval()
        self.reverse_tokenizer = MarianTokenizer.from_pretrained(self.revisions['reverse_id'], revision=self.revisions['reverse'], local_files_only=True, cache_dir=self.cache_dir)
        self.reverse_model = MarianMTModel.from_pretrained(self.revisions['reverse_id'], revision=self.revisions['reverse'],
                                                          use_safetensors=False, local_files_only=True, cache_dir=self.cache_dir).to(device).eval()
        for direction in ['forward','reverse']:
            model = getattr(self, direction+'_model')
            if getattr(model.config,'_commit_hash',None) != self.revisions[direction]:
                raise ValueError('Loaded model revision differs from frozen lock')
        self.sample_memory('loaded')

    def sample_memory(self, stage):
        if self.device == 'mps':
            self.memory_samples.append({'stage':stage, 'current_allocated_bytes':self.torch.mps.current_allocated_memory(),
                                        'driver_allocated_bytes':self.torch.mps.driver_allocated_memory()})

    def _infer(self,texts,tok,model,n):
        features=tok(texts,padding=True,truncation=False,return_tensors='pt')
        if features['input_ids'].shape[-1] > self.max_input_tokens:
            raise ValueError('One translation input exceeds declared input limit')
        features={key:value.to(self.device) for key,value in features.items()}
        with self.torch.inference_mode():
            generated=model.generate(**features,max_new_tokens=self.max_output_tokens,
                                     num_beams=max(4,2*n),num_return_sequences=n,do_sample=False)
        ids=generated.cpu().tolist()
        result=[]
        for row,text in zip(ids,tok.batch_decode(generated,skip_special_tokens=True)):
            # Marian can force EOS at the cap; reject capped rows even if EOS exists.
            rest=row[1:]
            try: ended=rest.index(tok.eos_token_id)+1
            except ValueError: ended=len(rest)
            capped=(tok.eos_token_id not in rest or ended >= self.max_output_tokens)
            result.append({'Text':text, 'capped':capped, 'generated_tokens_through_eos':ended})
        if len(result) != len(texts)*n: raise ValueError('Translation output alignment failure')
        return result

    def eligible_reason(self,text):
        if re.search(r'[\x80-\x9f]',text): return 'input_control_characters'
        if len(self.forward_tokenizer.encode(text,add_special_tokens=True)) > self.max_input_tokens:
            return 'input_token_limit'
        return None

    def eligible(self,text): return self.eligible_reason(text) is None

    def _batch_records(self,texts,variants):
        pivots=self._infer(texts,self.forward_tokenizer,self.forward_model,1)
        answer=[[] for _ in texts]
        eligible=[]
        for i,p in enumerate(pivots):
            reason=('forward_generation_limit' if p['capped'] else
                    'forward_empty' if not p['Text'].strip() else
                    'reverse_input_token_limit' if len(self.reverse_tokenizer.encode(p['Text'],add_special_tokens=True))>self.max_input_tokens else None)
            if reason:
                answer[i]=[{'Text':'','pivot_text':p['Text'],'failure_reason':reason}]
            else: eligible.append(i)
        if eligible:
            translated=self._infer([pivots[i]['Text'] for i in eligible],self.reverse_tokenizer,self.reverse_model,variants)
            for j,i in enumerate(eligible):
                answer[i]=[{'Text':x['Text'],'pivot_text':pivots[i]['Text'],
                            'failure_reason':'reverse_generation_limit' if x['capped'] else
                                             'reverse_empty' if not x['Text'].strip() else None}
                           for x in translated[j*variants:(j+1)*variants]]
        return answer

    def batch_records(self,texts,variants):
        try: result=self._batch_records(texts,variants)
        except (RuntimeError,NotImplementedError) as error:
            message=str(error).lower()
            if self.device!='mps' or not any(x in message for x in ['mps','metal','not implemented','out of memory']): raise
            self.fallback_events.append({'from':'mps','to':'cpu','reason':str(error),
                                         'utc':datetime.now(timezone.utc).isoformat()})
            self.forward_model.to('cpu'); self.reverse_model.to('cpu')
            self.device='cpu'; self.torch.mps.empty_cache()
            result=self._batch_records(texts,variants)
        self.sample_memory('batch_complete')
        return result

    def batch(self,texts,variants):
        return [[x['Text'] for x in r if not x.get('failure_reason')] for r in self.batch_records(texts,variants)]

    def close(self):
        del self.forward_model,self.reverse_model
        import gc
        gc.collect()
        if self.device=='cuda':self.torch.cuda.empty_cache()
        elif hasattr(self.torch,'mps') and self.torch.backends.mps.is_available():self.torch.mps.empty_cache()


def repair_terminal_journal(journal):
    if not journal.exists(): return
    raw=journal.read_bytes(); lines=raw.splitlines(keepends=True)
    offset=0
    for i,line in enumerate(lines):
        if not line.strip(): offset+=len(line); continue
        try: json.loads(line.decode('utf-8'))
        except (json.JSONDecodeError,UnicodeDecodeError) as error:
            if i!=len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'Damaged journal at line {i+1}: {error}')
            backup=journal.with_name('generation_journal_damaged_terminal_'+hashlib.sha256(line).hexdigest()[:12]+'.bin')
            backup.write_bytes(line)
            journal.write_bytes(raw[:offset])
            json_write(journal.with_name('journal_tail_recovery.json'),
                       {'status':'RECOVERED_INCOMPLETE_TERMINAL_RECORD_ONLY','line':i+1,
                        'discarded_bytes':len(line),'preserved_damaged_tail':backup.name})
            return
        offset+=len(line)
    if raw and not raw.endswith(b'\n'):journal.write_bytes(raw+b'\n')


def replay_journal(journal,initial_texts,initial_punct,parents=None,allowed_pivots=None,model_lock=None,config=None):
    journal=safe_path(journal)
    repair_terminal_journal(journal)
    decisions=[];seen=set(initial_texts);punct=set(initial_punct);done=set();counts=Counter();by_id={}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            event=json.loads(line);key=(event['source_id'],event['pivot'])
            if key in done:raise ValueError(f'Repeated journal key {key}')
            if parents is not None:
                parent=parents.get(event['source_id'])
                if parent is None or event.get('parent_text')!=parent['Text'] or str(event.get('Class'))!=str(parent['Class']):
                    raise ValueError('Journal parent/class does not match frozen training source')
            if allowed_pivots is not None and event['pivot'] not in allowed_pivots:raise ValueError('Journal pivot not permitted')
            if model_lock is not None and event.get('model_revisions')!=model_lock[event['pivot']]:raise ValueError('Journal model revision mismatch')
            done.add(key)
            for cand in event['candidates']:
                if cand['source_id']!=event['source_id'] or cand['parent_text']!=event['parent_text'] or str(cand['Class'])!=str(event['Class']) or cand['pivot_language']!=event['pivot']:
                    raise ValueError('Journal candidate parent alignment failed')
                if cand['example_id'] in by_id:raise ValueError('Duplicate candidate example ID')
                expected_id=hashlib.sha256((cand['source_id']+'|'+event['pivot']+'|'+str(cand['variant_index'])+'|'+cand['Text']).encode()).hexdigest()[:24]
                if cand['example_id']!=expected_id:raise ValueError('Journal example ID mismatch')
                if model_lock is not None:
                    frozen=model_lock[event['pivot']]
                    for field,value in [('forward_model',frozen['forward_id']),('reverse_model',frozen['reverse_id']),
                                        ('forward_revision',frozen['forward']),('reverse_revision',frozen['reverse'])]:
                        if cand.get(field)!=value:raise ValueError('Journal candidate revision/model mismatch')
                if config is not None:
                    for field in ['seed','max_input_tokens','max_output_tokens']:
                        if cand.get(field)!=config[field]:raise ValueError('Journal candidate generation settings mismatch')
                    if cand.get('num_beams')!=config['generation']['reverse_num_beams']:raise ValueError('Journal beam setting mismatch')
                if cand['decision']=='accepted':
                    reason=candidate_reason(cand['parent_text'],cand['Text'],seen,punct)
                    if reason is not None:raise ValueError('Journal accepted candidate fails duplicate/filter replay: '+reason)
                    counts[int(cand['Class'])]+=1;seen.add(normalize(cand['Text']));punct.add(punct_normalize(cand['Text']))
                decisions.append(cand);by_id[cand['example_id']]=cand
            for example_id in event.get('quarantined_example_ids',[]):
                prior=by_id.get(example_id)
                if prior is None:raise ValueError('Unknown cross-label quarantine reference')
                if prior['decision']=='accepted':
                    prior['decision']='quarantined_cross_label_duplicate';counts[int(prior['Class'])]-=1
    return decisions,done,counts,seen,punct


def produce(args,translator_factory=None):
    train,valid=check_inputs(args.train,args.validation)
    counts=Counter(int(r['Class']) for r in train)
    print('Frozen inputs verified: train=',len(train),'validation=',len(valid),flush=True)
    print('Original train classes:',dict(sorted(counts.items())),flush=True)
    print('No test.csv is accessed.',flush=True)
    if args.dry_run:
        print('DRY RUN COMPLETE — no translation, GPU, or model downloads.',flush=True);return None
    out=safe_path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=extract_provenance(train,args)
    config_file=out/'generation_config.json'
    if config_file.exists():
        previous=json.loads(config_file.read_text())
        ignored={'created_utc','requested_device','initial_batch_size'}
        if {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in config.items() if k not in ignored}:
            raise ValueError('Output belongs to DIFFERENT configuration/recipe; use a new output directory')
        config=previous
    else:json_write(config_file,config)
    lock_path=out/'model_lock.json'
    requested=getattr(args,'model_lock',None)
    lock=(json.loads(safe_path(requested).read_text()) if requested else
          json.loads(lock_path.read_text()) if lock_path.exists() else
          {p:{'forward':'OFFLINE_MOCK','reverse':'OFFLINE_MOCK','forward_id':MODELS[p][0],'reverse_id':MODELS[p][1]} for p in args.pivots}
          if translator_factory else resolve_model_lock(args.pivots))
    lock={p:lock[p] for p in args.pivots}
    if any((lock[p]['forward_id'],lock[p]['reverse_id'])!=MODELS[p] for p in args.pivots):
        raise ValueError('Translation model identities differ from declared protocol')
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Model revision lock changed on resume')
    if not lock_path.exists():json_write(lock_path,lock)
    if not translator_factory and any(not re.fullmatch('[0-9a-f]{40}',lock[p][d]) for p in args.pivots for d in ['forward','reverse']):
        raise ValueError('Real model execution requires immutable model revisions')
    rng=random.Random(args.seed)
    sources={k:[r for r in train if int(r['Class'])==k] for k in range(1,6)}
    for k in sources:rng.shuffle(sources[k])
    parents={r['source_id']:r for r in train}
    initial={normalize(r['Text']) for r in train+valid};initial_punct={punct_normalize(r['Text']) for r in train+valid}
    journal=out/'generation_journal.jsonl'
    decisions,done,accepted_counts,seen,seen_punct=replay_journal(journal,initial,initial_punct,parents,args.pivots,lock,config)
    label_index={};by_id={r['example_id']:r for r in decisions}
    def add_index(text,cls,eid,origin):
        for kind,key in [('norm',normalize(text)),('punct',punct_normalize(text))]:
            label_index.setdefault((kind,key),[]).append((int(cls),eid,origin))
    for origin,rows in [('original_train',train),('original_validation',valid)]:
        for r in rows:add_index(r['Text'],r['Class'],r['source_id'],origin)
    for r in decisions:
        if r['decision'] in ['accepted','quarantined_cross_label_duplicate','cross_label_duplicate_conflict']:
            add_index(r['Text'],r['Class'],r['example_id'],'synthetic')
    print('Resuming:',len(done),'completed source/pivot jobs; accepted',dict(accepted_counts),flush=True)
    runtime=[]
    for pivot in args.pivots:
        if all(counts[c]+accepted_counts[c]>=args.target_per_class for c in range(1,6)):break
        if all((r['source_id'],pivot) in done for c in range(1,6) for r in sources[c] if counts[c]+accepted_counts[c]<args.target_per_class):continue
        if translator_factory:model=translator_factory(pivot)
        else:
            import torch
            device=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
            if device=='mps' and not torch.backends.mps.is_available():raise EnvironmentError('Requested MPS unavailable')
            if device=='cuda' and not torch.cuda.is_available():raise EnvironmentError('Requested CUDA unavailable')
            print('Loading',pivot,'on',device,'at',lock[pivot],flush=True)
            try:
                model=MarianPair(pivot,device,args.max_input_tokens,args.max_output_tokens,lock[pivot],args.seed)
            except (RuntimeError,NotImplementedError) as error:
                if device!='mps' or not any(x in str(error).lower() for x in ['mps','metal','not implemented','out of memory']):raise
                import gc
                gc.collect();torch.mps.empty_cache()
                model=MarianPair(pivot,'cpu',args.max_input_tokens,args.max_output_tokens,lock[pivot],args.seed)
                model.fallback_events.append({'from':'mps','to':'cpu','stage':'initialization','reason':str(error)})
        json_write(out/'model_revisions.json',lock)  # Before first candidate, including interruptions.
        try:
            with journal.open('a',encoding='utf-8') as handle:
                for cls in range(1,6):
                    pool=[r for r in sources[cls] if (r['source_id'],pivot) not in done]
                    for start in range(0,len(pool),args.batch_size):
                        if counts[cls]+accepted_counts[cls]>=args.target_per_class:break
                        batch=pool[start:start+args.batch_size];eligible=[];skip={}
                        for r in batch:
                            reason=(model.eligible_reason(r['Text']) if hasattr(model,'eligible_reason') else None if model.eligible(r['Text']) else 'input_ineligible')
                            if reason:skip[r['source_id']]=reason
                            else:eligible.append(r)
                        if eligible:
                            if hasattr(model,'batch_records'):outputs=model.batch_records([r['Text'] for r in eligible],args.variants_per_pivot)
                            else:outputs=[[{'Text':t,'pivot_text':'OFFLINE_MOCK','failure_reason':None} for t in rr] for rr in model.batch([r['Text'] for r in eligible],args.variants_per_pivot)]
                            if len(outputs)!=len(eligible):raise ValueError('Translation source/output alignment failure')
                            responses={r['source_id']:rr for r,rr in zip(eligible,outputs)}
                        else:responses={}
                        for r in batch:
                            # Every generated member of this batch is journaled, even at the target boundary.
                            sid=r['source_id'];records=responses.get(sid,[])
                            if not records:records=[{'Text':'','pivot_text':'','failure_reason':skip.get(sid,'reverse_empty')}]
                            event={'source_id':sid,'parent_text':r['Text'],'Class':cls,'pivot':pivot,
                                   'model_revisions':lock[pivot],'candidates':[],'quarantined_example_ids':[]}
                            for variant,record in enumerate(records):
                                candidate=record['Text'];reason=record.get('failure_reason')
                                conflicts=[]
                                if not reason and candidate.strip():
                                    conflicts=[x for kind,key in [('norm',normalize(candidate)),('punct',punct_normalize(candidate))]
                                               for x in label_index.get((kind,key),[]) if x[0]!=cls]
                                    if conflicts:reason='cross_label_duplicate_conflict'
                                if not reason:reason=candidate_reason(r['Text'],candidate,seen,seen_punct)
                                if reason is None and counts[cls]+accepted_counts[cls]>=args.target_per_class:reason='target_reached'
                                entry={'source_id':sid,'parent_text':r['Text'],'Class':cls,'Text':candidate,
                                       'pivot_language':pivot,'variant_index':variant,
                                       'decision':reason or 'accepted',
                                       'example_id':hashlib.sha256((sid+'|'+pivot+'|'+str(variant)+'|'+candidate).encode()).hexdigest()[:24],
                                       'pivot_text':record.get('pivot_text',''), 'forward_model':lock[pivot]['forward_id'],
                                       'reverse_model':lock[pivot]['reverse_id'],'forward_revision':lock[pivot]['forward'],
                                       'reverse_revision':lock[pivot]['reverse'],'seed':args.seed,
                                       'max_input_tokens':args.max_input_tokens,'max_output_tokens':args.max_output_tokens,
                                       'num_beams':max(4,2*args.variants_per_pivot),'generation_device':getattr(model,'device','OFFLINE_MOCK'),
                                       'generated_utc':datetime.now(timezone.utc).isoformat(),
                                       'conflict_with_example_ids':json.dumps(sorted(set(x[1] for x in conflicts)))}
                                if reason=='cross_label_duplicate_conflict':
                                    for _,eid,origin in conflicts:
                                        if origin=='synthetic' and by_id[eid]['decision']=='accepted':
                                            prior=by_id[eid];prior['decision']='quarantined_cross_label_duplicate';accepted_counts[int(prior['Class'])]-=1
                                            event['quarantined_example_ids'].append(eid)
                                    add_index(candidate,cls,entry['example_id'],'synthetic')
                                if reason is None:
                                    accepted_counts[cls]+=1;seen.add(normalize(candidate));seen_punct.add(punct_normalize(candidate))
                                    add_index(candidate,cls,entry['example_id'],'synthetic')
                                decisions.append(entry);event['candidates'].append(entry);by_id[entry['example_id']]=entry
                            handle.write(json.dumps(event,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
                            done.add((sid,pivot))
                        if (start//args.batch_size)%10==0:
                            print(f'{pivot} class {cls}: processed {min(start+len(batch),len(pool))}/{len(pool)}, accepted={accepted_counts[cls]}',flush=True)
        finally:
            runtime.append({'pivot':pivot,'final_device':getattr(model,'device','OFFLINE_MOCK'),
                            'fallback_events':getattr(model,'fallback_events',[]),'memory_samples':getattr(model,'memory_samples',[])})
            runtime[-1]['exception_type']=sys.exc_info()[0].__name__ if sys.exc_info()[0] else None
            runtime[-1]['utc']=datetime.now(timezone.utc).isoformat()
            stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            json_write(out/f'runtime_{pivot}_{stamp}.json',runtime[-1])
            if hasattr(model,'close'):model.close()
    accepted=[r for r in decisions if r['decision']=='accepted']
    keys={normalize(r['Text']) for r in accepted};punct_keys={punct_normalize(r['Text']) for r in accepted}
    assert len(keys)==len(accepted) and len(punct_keys)==len(accepted)
    assert not keys & initial and not punct_keys & initial_punct
    assert all(r['source_id'] in parents and r['parent_text']==parents[r['source_id']]['Text'] and int(r['Class'])==int(parents[r['source_id']]['Class']) for r in decisions)
    cols=['source_id','Text','Class','example_id','parent_text','pivot_language','variant_index','is_synthetic','label_review_status',
          'pivot_text','forward_model','reverse_model','forward_revision','reverse_revision','seed','max_input_tokens','max_output_tokens','num_beams','generation_device','generated_utc']
    syn=[{**r,'is_synthetic':1,'label_review_status':'UNVERIFIED'} for r in accepted]
    originals=[{**r,'example_id':r['source_id'],'parent_text':r['Text'],'is_synthetic':0,'label_review_status':'SOURCE_LABEL'} for r in train]
    csv_write(out/'train_augmented_PROVISIONAL.csv',originals+syn,cols)
    csv_write(out/'synthetic_accepted_PROVISIONAL.csv',syn,cols)
    csv_write(out/'lineage.csv',syn,cols)
    decision_cols=cols+['decision','conflict_with_example_ids']
    csv_write(out/'all_translation_decisions.csv',decisions,decision_cols)
    rejected=[r for r in decisions if r['decision']!='accepted']
    csv_write(out/'rejected_candidates.csv',rejected,decision_cols)
    csv_write(out/'cross_label_conflicts.csv',[r for r in rejected if 'cross_label' in r['decision']],decision_cols)
    review=[]
    for cls in range(1,6):
        rows=[dict(r) for r in syn if int(r['Class'])==cls];random.Random(args.seed+cls).shuffle(rows)
        review.extend(rows[:20])
    csv_write(out/'MANUAL_REVIEW_SAMPLE.csv',review,cols+['preserves_ambiguity_label_Y_N_U','review_notes'])
    summary={**config,'original_train_count':len(train),'validation_count_unchanged':len(valid),
             'synthetic_accepted_count':len(syn),'provisional_train_count':len(train)+len(syn),
             'per_class_train_original':{str(k):counts[k] for k in CLASSES},
             'per_class_train_provisional':{str(k):counts[k]+accepted_counts[k] for k in CLASSES},
             'generated_decision_counts':dict(Counter(r['decision'] for r in decisions)),
             'target_unmet':{str(k):max(0,args.target_per_class-counts[k]-accepted_counts[k]) for k in range(1,6)},
             'model_revisions':lock,'overlap_with_validation_normalized':0,'overlap_with_validation_punctuation_normalized':0,
             'synthetic_training_original_overlap_normalized':0,'synthetic_training_original_overlap_punctuation':0,
             'synthetic_duplicate_normalized':0,'synthetic_duplicate_punctuation':0,
             'parent_lineage_all_train':True,'verified_ambiguity_labels':False,'not_a_test_result':True,
             'runtime_records':[json.loads(p.read_text()) for p in sorted(out.glob('runtime_*.json'))],'status':'COMPLETED_FINITE_PREDECLARED_PARENT_PIVOT_ATTEMPTS'}
    json_write(out/'AUDIT.json',summary)
    print('DONE: accepted',len(syn),'provisional train',len(train)+len(syn),'unmet',summary['target_unmet'],flush=True)
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',default='data/splits/train.csv');ap.add_argument('--validation',default='data/splits/validation.csv')
    ap.add_argument('--output',default='outputs/lineage_backtranslation_20261027')
    ap.add_argument('--target-per-class',type=int,default=900)
    ap.add_argument('--pivots',nargs='+',default=['fr','de','es'],choices=MODELS)
    ap.add_argument('--variants-per-pivot',type=int,default=1,choices=[1,2,3])
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--max-input-tokens',type=int,default=192);ap.add_argument('--max-output-tokens',type=int,default=192)
    ap.add_argument('--model-lock',default=None,help='JSON immutable model revisions to share across smoke/full runs')
    ap.add_argument('--device',default='auto',choices=['auto','cuda','mps','cpu']);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    if min(args.target_per_class,args.batch_size,args.max_input_tokens,args.max_output_tokens)<=0:ap.error('Counts/limits must be positive')
    if len(set(args.pivots))!=len(args.pivots):ap.error('Duplicate pivot language')
    produce(args)


if __name__=='__main__':main()
