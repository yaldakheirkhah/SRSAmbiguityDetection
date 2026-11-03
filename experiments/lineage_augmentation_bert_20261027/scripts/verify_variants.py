#!/usr/bin/env python3
"""Independent, stdlib-only audit of immutable A/B augmentation variants.

The code does not import the variant builder or Torch. It never traverses split
folders, reads a protected test filename, modifies input data, or evaluates a
model. Run only after the parent has finished constructing all declared inputs.
"""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import re
import unicodedata

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
BT = ROOT / 'experiments/lineage_backtranslation_mac_20261027'
EXPECTED_SOURCE_HASHES = {
    'train': 'c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8',
    'validation': '9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42',
}
LABELS = [1, 2, 3, 4, 5, 6]
INPUTS = {
    'frozen_train': ROOT / 'data/splits/train.csv',
    'frozen_validation': ROOT / 'data/splits/validation.csv',
    'source_full': BT / 'runs/full/train_augmented_PROVISIONAL.csv',
    'source_synthetic': BT / 'runs/full/synthetic_accepted_PROVISIONAL.csv',
    'source_flags': BT / 'reports/candidate_suspicious_flags.csv',
    'source_output_audit': BT / 'reports/INDEPENDENT_OUTPUT_AUDIT.json',
    'source_flag_code': BT / 'scripts/audit_outputs.py',
    'source_generation_config': BT / 'runs/full/generation_config.json',
    'source_run_model_lock': BT / 'runs/full/model_lock.json',
    'translation_lock': BT / 'config/model_lock.json',
    'original_clone': BASE / 'data/train_original.csv',
    'validation_clone': BASE / 'data/validation.csv',
    'A': BASE / 'data/train_A_Conservative_v1.csv',
    'B': BASE / 'data/train_B_Full_Provisional_v1.csv',
    'weights_A': BASE / 'config/variant_A_class_weights.json',
    'weights_B': BASE / 'config/variant_B_class_weights.json',
}
OPTIONAL_REPORT = BASE / 'reports/DATASET_VERIFICATION.json'
ALLOWED = set(INPUTS.values()) | {OPTIONAL_REPORT}
MODALS = ['shall', 'should', 'must', 'may', 'might', 'can', 'could', 'will', 'would',
          'ought to', 'have to', 'has to']
CONDITIONS = ['if', 'when', 'unless', 'except', 'only', 'before', 'after', 'until',
              'provided that', 'as long as', 'at least', 'at most', 'no more than', 'no less than']


def guarded(path):
    p = Path(path)
    # The guard runs before resolution/stat/read of even an imagined protected path.
    if any(part.casefold() == 'test.csv' for part in p.parts):
        raise ValueError('Protected filename rejected before filesystem access')
    if p not in ALLOWED:
        raise ValueError('Input is not explicitly allowlisted: ' + str(p))
    resolved = p.resolve()
    if any(part.casefold() == 'test.csv' for part in resolved.parts):
        raise ValueError('Protected resolved target rejected')
    return resolved


def blob(path):
    return guarded(path).read_bytes()


def sha(path):
    return hashlib.sha256(blob(path)).hexdigest()


def rows(path):
    return list(csv.DictReader(io.StringIO(blob(path).decode('utf-8-sig'), newline='')))


def obj(path):
    return json.loads(blob(path).decode('utf-8-sig'))


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip().casefold()


def punct(text):
    return re.sub(r'[\W_]+', '', norm(text), flags=re.UNICODE)


def number_counts(text):
    return Counter(re.findall(r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)', str(text)))


def phrase_counts(text, phrases):
    text = norm(text)
    return {phrase: len(re.findall(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text))
            for phrase in phrases
            if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text)}


def quality_flags(parent, candidate):
    """Exact independent reproduction of the existing audit's review cues."""
    pm, cm = phrase_counts(parent, MODALS), phrase_counts(candidate, MODALS)
    pc, cc = phrase_counts(parent, CONDITIONS), phrase_counts(candidate, CONDITIONS)
    pn = Counter(re.findall(r"\b(?:not|never|no|without|cannot)\b|n['’]t\b", norm(parent)))
    cn = Counter(re.findall(r"\b(?:not|never|no|without|cannot)\b|n['’]t\b", norm(candidate)))
    po = Counter(re.findall(r'<=|>=|!=|==|[<>=+*/%−-]', parent))
    co = Counter(re.findall(r'<=|>=|!=|==|[<>=+*/%−-]', candidate))
    names = []
    if set(pm) != set(cm): names.append('modal_set_changed')
    if pm != cm: names.append('modal_counts_changed')
    if pn != cn: names.append('negation_counts_changed')
    if pc != cc: names.append('condition_scope_counts_changed')
    if number_counts(parent) != number_counts(candidate): names.append('numeric_tokens_changed')
    if po != co: names.append('operator_tokens_changed')
    if not candidate.strip(): names.append('empty_candidate')
    return names


def diagnostic(data, function):
    groups = defaultdict(list)
    for row in data: groups[function(row['Text'])].append(row)
    repeated = [r for r in groups.values() if len(r) > 1]
    conflict = [r for r in groups.values() if len({int(x['Class']) for x in r}) > 1]
    return {'rows': len(data), 'unique_keys': len(groups), 'duplicate_keys': len(repeated),
            'repeated_rows_beyond_first': sum(len(x)-1 for x in repeated),
            'rows_in_duplicate_groups': sum(map(len, repeated)),
            'cross_label_keys': len(conflict), 'rows_in_cross_label_groups': sum(map(len, conflict))}


def overlap(left, right, function):
    a, b = defaultdict(list), defaultdict(list)
    for r in left: a[function(r['Text'])].append(r)
    for r in right: b[function(r['Text'])].append(r)
    shared = set(a) & set(b)
    return {'shared_keys': len(shared), 'left_rows': sum(len(a[k]) for k in shared),
            'right_rows': sum(len(b[k]) for k in shared),
            'cross_label_shared_keys': sum(len({int(x['Class']) for x in a[k]+b[k]}) > 1 for k in shared)}


def main():
    targets = [BASE / 'reports/INDEPENDENT_VARIANT_REVIEW.json',
               BASE / 'reports/INDEPENDENT_VARIANT_REVIEW.md']
    if any(path.exists() for path in targets):
        raise SystemExit('Existing independent review found; refuse overwrite')
    absent = [str(p.relative_to(ROOT)) for p in INPUTS.values() if not guarded(p).is_file()]
    if absent:
        raise SystemExit('Variant inputs not yet complete; no audit executed: ' + ', '.join(absent))
    before = {key: sha(path) for key, path in INPUTS.items()}
    checks, details = {}, {}
    def check(name, passed, detail=None):
        checks[name] = bool(passed)
        if detail is not None: details[name] = detail

    original, validation = rows(INPUTS['frozen_train']), rows(INPUTS['frozen_validation'])
    source_full, source_syn, flag_rows = rows(INPUTS['source_full']), rows(INPUTS['source_synthetic']), rows(INPUTS['source_flags'])
    original_clone, val_clone = rows(INPUTS['original_clone']), rows(INPUTS['validation_clone'])
    variants = {'A': rows(INPUTS['A']), 'B': rows(INPUTS['B'])}
    source_audit, lock = obj(INPUTS['source_output_audit']), obj(INPUTS['translation_lock'])
    check('generation_model_lock_copies_agree', lock == obj(INPUTS['source_run_model_lock']))
    check('frozen_train_hash_and_count', before['frozen_train'] == EXPECTED_SOURCE_HASHES['train'] and len(original) == 4731)
    check('frozen_validation_hash_and_count', before['frozen_validation'] == EXPECTED_SOURCE_HASHES['validation'] and len(validation) == 1015)
    check('original_clone_byte_identical', before['original_clone'] == before['frozen_train'] and original_clone == original)
    check('validation_clone_bytes_and_order_identical', before['validation_clone'] == before['frozen_validation'] and val_clone == validation)
    check('previous_generation_independent_audit_passed', source_audit.get('status', '').startswith('PASS_') and all(source_audit['checks'].values()))
    check('source_generation_artifact_hashes_still_match',
          before['source_full'] == source_audit['run_artifact_sha256']['train_augmented_PROVISIONAL.csv']
          and before['source_synthetic'] == source_audit['run_artifact_sha256']['synthetic_accepted_PROVISIONAL.csv'])
    check('source_generation_original_prefix_and_synthetic_suffix', len(source_full) == 7174 and len(source_syn) == 2443
          and len(source_full) == len(original)+len(source_syn)
          and all({k:r[k] for k in ['source_id','Text','Class']} == o for r,o in zip(source_full[:len(original)], original))
          and source_full[len(original):] == source_syn)
    parents = {r['source_id']: r for r in original}
    val_ids = {r['source_id'] for r in validation}
    check('original_and_validation_ids_unique_disjoint', len(parents) == len(original) and len(val_ids) == len(validation) and not set(parents)&val_ids)
    source_columns = list(source_full[0])
    source_by_id = {r['example_id']:r for r in source_syn}
    expected_flags = {r['example_id']: quality_flags(r['parent_text'], r['Text']) for r in source_syn}
    accepted_ids = set(expected_flags)
    check('accepted_source_example_ids_unique', len(accepted_ids) == len(source_syn))
    check('quality_review_table_ids_unique', len({r['example_id'] for r in flag_rows}) == len(flag_rows))
    flags_by_id = {r['example_id']: r for r in flag_rows if r['example_id'] in accepted_ids}
    flag_metadata_bad = []
    for eid, r in flags_by_id.items():
        source = source_by_id[eid]
        if (r['final_decision'] != 'accepted' or r['review_flags'].split(';') != expected_flags[eid]
            or any(r[k] != source[k] for k in ['source_id','parent_text','Text','Class','pivot_language'])
            or r['label_review_status'] != 'UNVERIFIED'):
            flag_metadata_bad.append(eid)
    flagged = {eid for eid, flag in expected_flags.items() if flag}
    unflagged = accepted_ids - flagged
    check('all_accepted_quality_flags_independently_recomputed', not flag_metadata_bad and set(flags_by_id) == flagged,
          {'computed_flagged':len(flagged), 'computed_unflagged':len(unflagged), 'metadata_violations':flag_metadata_bad[:10],
           'definition':'Existing lexical review cues reproduced exactly, not semantic/label validation'})
    check('recomputed_counts_match_published_generation_audit', len(flagged) == source_audit['review_only_flags']['accepted_rows_with_any_flag'] == 1270
          and len(unflagged) == 1173)
    expected = {'A':source_full[:len(original)] + [r for r in source_syn if r['example_id'] in unflagged],
                'B':source_full}
    variant_results = {}
    for name, data in variants.items():
        syn = data[len(original):]
        compare = [{key:r.get(key, '') for key in source_columns} for r in data]
        check(f'{name}_exact_source_order_membership_and_generation_metadata', compare == expected[name])
        check(f'{name}_all_originals_retained_unchanged_in_order', len(data)>=len(original)
              and all({key:r.get(key) for key in ['source_id','Text','Class']} == o for r,o in zip(data,original)))
        check(f'{name}_versioned_data_file_read_only', guarded(INPUTS[name]).stat().st_mode & 0o222 == 0)
        check(f'{name}_example_ids_unique', len({r['example_id'] for r in data}) == len(data))
        check(f'{name}_original_status_and_parent_text', all(r['is_synthetic']=='0' and r['label_review_status']=='SOURCE_LABEL'
              and r['parent_text']==r['Text'] and r['example_id']==r['source_id'] for r in data[:len(original)]))
        bad = []
        for r in syn:
            parent = parents.get(r['source_id'])
            p = r['pivot_language']
            correct = (parent is not None and r['source_id'] not in val_ids and r['parent_text']==parent['Text']
                       and r['Class']==parent['Class'] and int(r['Class']) in range(1,6)
                       and r['is_synthetic']=='1' and r['label_review_status']=='UNVERIFIED' and p in lock)
            if correct:
                correct = all(r[field]==lock[p][key] for field,key in [
                    ('forward_model','forward_id'),('reverse_model','reverse_id'),('forward_revision','forward'),('reverse_revision','reverse')])
                eid=hashlib.sha256((r['source_id']+'|'+p+'|'+r['variant_index']+'|'+r['Text']).encode()).hexdigest()[:24]
                correct = correct and r['example_id']==eid
            if not correct:bad.append(r['example_id'])
        check(f'{name}_every_synthetic_parent_label_model_and_id_verified', not bad, {'violations':len(bad),'first_ids':bad[:10]})
        check(f'{name}_quality_columns_match_existing_review_definition', all(
            set(filter(None,r.get('quality_review_flags','').split(';'))) == set(expected_flags[r['example_id']])
            for r in syn) and all(not r.get('quality_review_flags','').strip() for r in data[:len(original)]))
        if name=='A':
            check('A_no_flagged_synthetic_included', all(not expected_flags[r['example_id']] for r in syn))
            check('A_all_unflagged_synthetic_retained', {r['example_id'] for r in syn} == unflagged)
        else:
            check('B_all_accepted_synthetic_retained', {r['example_id'] for r in syn} == accepted_ids)
        check(f'{name}_no_original_parent_overlap_with_validation', not {r['source_id'] for r in data}&val_ids)
        diagnostics = {key:diagnostic(data, fn) for key,fn in [('exact',str),('normalized',norm),('punctuation',punct)]}
        overlaps = {key:overlap(data,validation,fn) for key,fn in [('exact',str),('normalized',norm),('punctuation',punct)]}
        syn_diags = {key:diagnostic(syn,fn) for key,fn in [('exact',str),('normalized',norm),('punctuation',punct)]}
        syn_vs_original = {key:overlap(syn,original,fn) for key,fn in [('exact',str),('normalized',norm),('punctuation',punct)]}
        syn_vs_validation = {key:overlap(syn,validation,fn) for key,fn in [('exact',str),('normalized',norm),('punctuation',punct)]}
        check(f'{name}_exact_normalized_duplicates_conflicts_and_eval_overlap_zero', all(
            diagnostics[k]['repeated_rows_beyond_first']==diagnostics[k]['cross_label_keys']==overlaps[k]['shared_keys']==0
            for k in ['exact','normalized']))
        check(f'{name}_inherited_punctuation_risks_unchanged', diagnostics['punctuation']['duplicate_keys']==19
              and diagnostics['punctuation']['repeated_rows_beyond_first']==19 and diagnostics['punctuation']['cross_label_keys']==3
              and overlaps['punctuation']['shared_keys']==10 and overlaps['punctuation']['cross_label_shared_keys']==1)
        check(f'{name}_new_synthetic_duplicates_conflicts_and_overlap_zero', all(
            syn_diags[k]['repeated_rows_beyond_first']==syn_diags[k]['cross_label_keys']==syn_vs_original[k]['shared_keys']==syn_vs_validation[k]['shared_keys']==0
            for k in ['exact','normalized','punctuation']))
        counts = Counter(int(r['Class']) for r in data)
        syn_counts = Counter(int(r['Class']) for r in syn)
        check(f'{name}_six_classes_and_clean_class_unchanged', set(counts)==set(LABELS) and counts[6]==2674 and not syn_counts[6])
        w = obj(INPUTS['weights_'+name])
        raw = [math.sqrt(len(data)/(6*counts[c])) for c in LABELS]
        mean = sum(raw)/6
        values = [x/mean for x in raw]
        check(f'{name}_class_weights_training_only_and_exact', w['dataset_class_order']==LABELS
              and w['training_counts']==[counts[c] for c in LABELS] and len(w['values'])==6
              and all(math.isclose(float(a),b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(w['values'],values)),
              {'computed_counts':[counts[c] for c in LABELS],'computed_values':values})
        variant_results[name] = {'rows':len(data),'original_rows':len(original),'synthetic_rows':len(syn),
            'class_counts':{str(c):counts[c] for c in LABELS}, 'synthetic_class_counts':{str(c):syn_counts[c] for c in LABELS},
            'weights':values, 'all_synthetic_labels':'UNVERIFIED', 'diagnostics':diagnostics,
            'validation_overlaps':overlaps, 'synthetic_diagnostics':syn_diags,
            'synthetic_original_overlaps':syn_vs_original,'synthetic_validation_overlaps':syn_vs_validation,
            'model_example_id_count':len({r['example_id'] for r in data}), 'parent_group_count':len({r['source_id'] for r in data})}
    check('A_is_originals_plus_expected_1173_B_is_originals_plus_2443', len(variants['A'])==5904 and len(variants['B'])==7174
          and len(variants['A'])-len(original)==1173 and len(variants['B'])-len(original)==2443)
    if guarded(OPTIONAL_REPORT).is_file():
        builder = obj(OPTIONAL_REPORT)
        declared_sources = builder.get('source_inputs', {})
        source_agreement = {}
        for key, entry in declared_sources.items():
            path = ROOT / entry['path']
            source_agreement[key] = path in ALLOWED and sha(path) == entry['sha256']
        check('all_builder_source_hashes_independently_agree', len(source_agreement)==8 and all(source_agreement.values()), source_agreement)
        check('builder_variant_hashes_and_counts_independently_agree', all(
            builder['variants'][name]['sha256']==before[name]
            and builder['variants'][name]['training_rows']==variant_results[name]['rows']
            and builder['variants'][name]['class_counts']==variant_results[name]['class_counts'] for name in ['A','B']))
        check('builder_gate_reports_all_checks_passed', builder.get('status','').startswith('PASS')
              and builder.get('approved_for_gpu_package') is True and all(builder.get('checks',{}).values()))
    after = {key:sha(path) for key,path in INPUTS.items()}
    check('all_inputs_unchanged_during_independent_review', before==after)
    report = {'status':'PASS_INDEPENDENT_VARIANT_REVIEW' if all(checks.values()) else 'FAIL_INDEPENDENT_VARIANT_REVIEW',
        'date':'2026-10-27','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Independent stdlib arithmetic/content/lineage/flags/hash audit; no generation or model training',
        'checks':checks,'failed_checks':[key for key,value in checks.items() if not value],'check_details':details,
        'input_sha256':before,'input_sha256_after':after,'quality_flags_definition_source':str(INPUTS['source_flag_code'].relative_to(ROOT)),
        'accepted_source_synthetic_count':len(source_syn),'flagged_accepted_synthetic_excluded_from_A':len(flagged),
        'unflagged_accepted_synthetic_count':len(unflagged),'variants':variant_results,
        'limitations':['No quality flag is not proof of semantic equivalence or correct ambiguity label.',
                       'All inherited synthetic labels remain UNVERIFIED, including every Variant A row.',
                       'The aggressive punctuation key retains 19 inherited original duplicate keys (three cross-label) and ten original train/validation overlap keys (one cross-label). These are not new synthetic additions.',
                       'Semantic/paraphrase/project/document independence and expert label validity remain unverified.',
                       'The protected held-out partition was never accessed for any check; no final performance claim is possible.'],
        'test_accessed':False,'training_performed':False,'datasets_modified':False,'original_labels_changed':False}
    lines=['# Independent A/B dataset and lineage review','',report['status'],'',
           f"Recomputed the existing review cues for all {len(source_syn):,} accepted synthetic examples: {len(flagged):,} have flags and {len(unflagged):,} have none. This is a lexical review-flag policy, not expert semantic validation.",'',
           '| Variant | Original rows | Synthetic rows | Total |','|---|---:|---:|---:|']
    for name,v in variant_results.items():lines.append(f"| {name} | {v['original_rows']:,} | {v['synthetic_rows']:,} | {v['rows']:,} |")
    lines += ['',f"All {len(checks)} independent checks passed." if all(checks.values()) else 'Failed checks: '+', '.join(report['failed_checks']),
              '', 'Verified source/validation byte identities and ordering, exact source-column preservation, quality-filter membership/order, every synthetic parent/class/model revision/content ID, separate training-only class weights, and duplicate/overlap diagnostics. New synthetic duplicates or overlap are zero under exact, NFKC/whitespace/casefold and punctuation keys. Frozen originals are retained unchanged.',
              '', 'Inherited original punctuation diagnostics remain: 19 duplicate keys (three cross-label); ten train/validation shared keys (one cross-label). These aggressive-key matches do not themselves establish equal meaning. No original example was removed or relabeled.',
              '', 'All synthetic labels remain UNVERIFIED. No-flag selection does not establish semantic correctness, population representativeness or a verified augmentation benefit. No model was trained/evaluated, and no held-out file was accessed.',
              '', '## Check results','']
    lines += [('- PASS: ' if passed else '- FAIL: ')+name for name,passed in checks.items()]
    with targets[0].open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    with targets[1].open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({key:report[key] for key in ['status','failed_checks','flagged_accepted_synthetic_excluded_from_A','unflagged_accepted_synthetic_count']},indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
