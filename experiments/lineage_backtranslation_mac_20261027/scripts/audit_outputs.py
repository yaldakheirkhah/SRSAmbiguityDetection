#!/usr/bin/env python3
"""Independent stdlib-only post-run audit of the Mac back-translation outputs.

Refuses to run before runs/full/AUDIT.json exists. Reads an explicit allowlist,
never traverses split directories, never imports the generation implementation,
and never changes generated/source datasets. Review flags are not adjudication.
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
import re
import sys
import unicodedata


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
RUN = BASE / 'runs/full'
SOURCE = BASE / 'source_package/AmbiFix_Lineage_Backtranslation_Mac_M3Pro'
EXPECTED_HASHES = {
    'train': 'c5d623c1738400170fe7b5feb23a82d4a3c15f071a77fe5c4bfba8c6c5cc70f8',
    'validation': '9b5c7c76dad306bff4b306753d04a0169f84dd6f27eb8312bc63f5f3322bef42',
}
EXPECTED_ROWS = {'train': 4731, 'validation': 1015}
MODEL_IDS = {
    'fr': ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
    'de': ('Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
    'es': ('Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'),
}
CSV_NAMES = [
    'train_augmented_PROVISIONAL.csv', 'synthetic_accepted_PROVISIONAL.csv',
    'lineage.csv', 'all_translation_decisions.csv', 'rejected_candidates.csv',
    'cross_label_conflicts.csv', 'MANUAL_REVIEW_SAMPLE.csv',
]
ALLOWLIST = {
    *(RUN / name for name in ['AUDIT.json', 'generation_config.json', 'model_lock.json',
                              'model_revisions.json', 'generation_journal.jsonl', *CSV_NAMES]),
    BASE / 'pipeline/scripts/backtranslate_with_lineage.py',
    *(where / f'data/splits/{name}.csv' for where in [SOURCE, BASE / 'pipeline', ROOT]
      for name in ['train', 'validation']),
}


def read_bytes(path):
    path = Path(path)
    if any(part.casefold() == 'test.csv' for part in path.parts):
        raise ValueError('Protected path rejected before access')
    if path not in ALLOWLIST:
        raise ValueError('Path not explicitly allowlisted: ' + str(path))
    resolved = path.resolve()
    if any(part.casefold() == 'test.csv' for part in resolved.parts):
        raise ValueError('Protected symlink target rejected')
    return resolved.read_bytes()


def read_json(path):
    return json.loads(read_bytes(path).decode('utf-8-sig'))


def read_csv(path):
    return list(csv.DictReader(io.StringIO(read_bytes(path).decode('utf-8-sig'), newline='')))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


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


PROTECTED = ['not', 'never', 'without', 'unless', 'except', 'only', 'at least',
             'at most', 'no more than', 'no less than', 'before', 'after', 'until', 'if', 'when']
MODALS = ['shall', 'should', 'must', 'may', 'might', 'can', 'could', 'will', 'would',
          'ought to', 'have to', 'has to']
CONDITIONS = ['if', 'when', 'unless', 'except', 'only', 'before', 'after', 'until',
              'provided that', 'as long as', 'at least', 'at most', 'no more than', 'no less than']


def filter_reason(parent, text, seen_norm, seen_punct):
    """Independent reproduction of documented non-model candidate filtering."""
    original, candidate = norm(parent), norm(text)
    if not str(text).strip() or not candidate:
        return 'empty'
    if not 12 <= len(candidate) <= 2500:
        return 'invalid_length'
    if re.search(r'[\x80-\x9f]', text):
        return 'control_characters'
    if candidate == original or candidate in seen_norm:
        return 'same_or_duplicate'
    if not .72 <= len(candidate) / max(1, len(original)) <= 1.35:
        return 'large_length_change'
    if number_counts(parent) != number_counts(text):
        return 'numbers_changed'
    if set(phrase_counts(parent, PROTECTED)) != set(phrase_counts(text, PROTECTED)):
        return 'scope_condition_or_negation_changed'
    if punct(text) in seen_punct:
        return 'punctuation_equivalent_duplicate'
    acronyms = set(re.findall(r'\b[A-Z]{2,}\b', parent)) - {'THE', 'AND', 'FOR', 'GUI'}
    if acronyms - set(re.findall(r'\b[A-Z]{2,}\b', text)):
        return 'acronym_changed'
    return None


def group_diagnostic(rows, key_function):
    groups = defaultdict(list)
    for row in rows:
        groups[key_function(row['Text'])].append(row)
    duplicates = {key: members for key, members in groups.items() if len(members) > 1}
    conflicts = {key: members for key, members in groups.items()
                 if len({str(r['Class']) for r in members}) > 1}
    return {'unique_keys': len(groups), 'duplicate_keys': len(duplicates),
            'repeated_rows_beyond_first': len(rows) - len(groups),
            'rows_in_duplicate_groups': sum(map(len, duplicates.values())),
            'cross_label_keys': len(conflicts),
            'rows_in_cross_label_groups': sum(map(len, conflicts.values()))}


def overlap_diagnostic(left, right, key_function):
    a = [key_function(r['Text']) for r in left]
    b = [key_function(r['Text']) for r in right]
    shared = set(a) & set(b)
    labels = defaultdict(set)
    for row in left + right:
        key = key_function(row['Text'])
        if key in shared:
            labels[key].add(str(row['Class']))
    return {'shared_keys': len(shared), 'left_rows': sum(k in shared for k in a),
            'right_rows': sum(k in shared for k in b),
            'cross_label_shared_keys': sum(len(v) > 1 for v in labels.values())}


def csv_scalar(value):
    return '' if value is None else str(value)


def write_csv_exclusive(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not (RUN / 'AUDIT.json').is_file():
        raise SystemExit('Full run completion artifact absent; independent post-run audit refused.')
    reports = BASE / 'reports'
    destinations = [reports / name for name in [
        'INDEPENDENT_OUTPUT_AUDIT.json', 'class_distributions.csv', 'candidate_suspicious_flags.csv']]
    if any(p.exists() for p in destinations):
        raise SystemExit('Existing independent audit output found; refuse silent overwrite.')
    summary = read_json(RUN / 'AUDIT.json')
    config = read_json(RUN / 'generation_config.json')
    lock = read_json(RUN / 'model_lock.json')
    revisions = read_json(RUN / 'model_revisions.json')
    tables = {name: read_csv(RUN / name) for name in CSV_NAMES}
    checks, details, warnings = {}, {}, []

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail

    source_hashes = {}
    source_rows = {}
    for name in ['train', 'validation']:
        source_rows[name] = read_csv(SOURCE / f'data/splits/{name}.csv')
        for origin, where in [('immutable_package', SOURCE), ('working_package', BASE / 'pipeline'),
                              ('frozen_repository', ROOT)]:
            blob = read_bytes(where / f'data/splits/{name}.csv')
            source_hashes[f'{origin}/{name}'] = sha(blob)
            check(f'{origin}_{name}_hash_matches_approved', sha(blob) == EXPECTED_HASHES[name])
        check(f'{name}_row_count_matches_approved', len(source_rows[name]) == EXPECTED_ROWS[name])
    train, validation = source_rows['train'], source_rows['validation']
    parents = {r['source_id']: r for r in train}
    validation_ids = {r['source_id'] for r in validation}
    check('original_source_ids_unique_disjoint', len(parents) == len(train)
          and len(validation_ids) == len(validation) and not set(parents) & validation_ids)
    check('original_source_ids_match_normalized_content_hash', all(
        r['source_id'] == sha(norm(r['Text']).encode())[:16] for r in train + validation))
    check('configuration_has_approved_source_hashes',
          config.get('train_sha256') == EXPECTED_HASHES['train']
          and config.get('validation_sha256') == EXPECTED_HASHES['validation'])
    check('initial_protocol_target_and_pivots', config.get('target_per_minority_class') == 900
          and config.get('pivot_languages') == ['fr', 'de', 'es'])
    check('configuration_declares_no_test_and_unverified_labels', config.get('read_test') is False
          and config.get('human_verified_labels') is False)
    check('generation_is_deterministic_beam_recipe', config.get('generation', {}).get('do_sample') is False
          and config.get('generation', {}).get('forward_num_beams') == 4
          and config.get('generation', {}).get('reverse_num_beams') == max(4, 2 * int(config['variants_per_pivot'])))
    check('current_exact_recipe_matches_run_config', config.get('recipe_sha256') == sha(
        read_bytes(BASE / 'pipeline/scripts/backtranslate_with_lineage.py')))
    check('model_lock_matches_revisions_and_summary', lock == revisions == summary.get('model_revisions'))
    lock_valid = set(lock) == set(config['pivot_languages'])
    for pivot in config['pivot_languages']:
        item = lock[pivot]
        lock_valid = lock_valid and (item.get('forward_id'), item.get('reverse_id')) == MODEL_IDS[pivot]
        lock_valid = lock_valid and all(re.fullmatch('[0-9a-f]{40}', item.get(x, '') or '')
                                       for x in ['forward', 'reverse'])
    check('all_model_identities_and_revisions_valid', lock_valid)
    check('configuration_translation_pairs_match_locked_models', config.get('translation_pairs') == {
        pivot: list(MODEL_IDS[pivot]) for pivot in config['pivot_languages']})
    check('summary_contains_exact_run_configuration', all(summary.get(k) == v for k, v in config.items()))
    check('summary_completion_status_is_finite_attempt_completion',
          summary.get('status') == 'COMPLETED_FINITE_PREDECLARED_PARENT_PIVOT_ATTEMPTS')

    errors = Counter()
    samples = defaultdict(list)

    def record(kind, condition, identifier):
        if not condition:
            errors[kind] += 1
            if len(samples[kind]) < 20:
                samples[kind].append(identifier)

    # Replay all lineage and decisions independently; no generation code import.
    by_id, final_decisions, journal_order, seen_jobs = {}, {}, [], set()
    norm_seen = {norm(r['Text']) for r in train + validation}
    punct_seen = {punct(r['Text']) for r in train + validation}
    index = defaultdict(list)
    original_counts = Counter(int(r['Class']) for r in train)
    accepted_counts = Counter()

    def add_index(text, cls, eid, origin):
        for kind, value in [('norm', norm(text)), ('punct', punct(text))]:
            index[(kind, value)].append((int(cls), eid, origin))

    for origin, rows in [('original_train', train), ('original_validation', validation)]:
        for row in rows:
            add_index(row['Text'], row['Class'], row['source_id'], origin)
    model_failures = {'input_control_characters', 'input_token_limit', 'input_ineligible',
                      'forward_generation_limit', 'forward_empty', 'reverse_input_token_limit',
                      'reverse_generation_limit', 'reverse_empty'}
    reason_counts_not_independently_token_verified = Counter()
    journal_blob = read_bytes(RUN / 'generation_journal.jsonl')
    raw_lines = journal_blob.decode('utf-8').splitlines()
    for line_number, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        event = json.loads(line)
        sid, pivot = event['source_id'], event['pivot']
        job = (sid, pivot)
        parent = parents.get(sid)
        record('journal_job_uniqueness', job not in seen_jobs, line_number)
        seen_jobs.add(job)
        event_ok = parent is not None and pivot in lock
        event_ok = event_ok and event.get('parent_text') == parent['Text']
        event_ok = event_ok and str(event.get('Class')) == parent['Class']
        event_ok = event_ok and event.get('model_revisions') == lock[pivot]
        event_ok = event_ok and isinstance(event.get('candidates'), list) and bool(event['candidates'])
        record('journal_event_parent_class_pivot_revision_alignment', event_ok, line_number)
        if not event_ok:
            raise ValueError(f'Journal event lineage invalid at line {line_number}; refuse misleading downstream audit.')
        computed_quarantines = set()
        for entry in event['candidates']:
            eid, text = entry['example_id'], entry['Text']
            cls = int(entry['Class'])
            identity = {'line': line_number, 'example_id': eid}
            expected_id = sha((sid + '|' + pivot + '|' + str(entry['variant_index']) + '|' + text).encode())[:24]
            record('candidate_ids_unique_and_valid', eid not in by_id and eid == expected_id, identity)
            record('candidate_exact_training_parent_and_label', entry.get('source_id') == sid
                   and sid not in validation_ids and entry.get('parent_text') == parent['Text']
                   and str(entry.get('Class')) == parent['Class'] and 1 <= cls <= 5, identity)
            record('candidate_pivot_and_model_metadata', entry.get('pivot_language') == pivot
                   and all(entry.get(field) == lock[pivot][key] for field, key in [
                       ('forward_model', 'forward_id'), ('reverse_model', 'reverse_id'),
                       ('forward_revision', 'forward'), ('reverse_revision', 'reverse')]), identity)
            record('candidate_generation_metadata', all(entry.get(k) == config[k]
                   for k in ['seed', 'max_input_tokens', 'max_output_tokens'])
                   and entry.get('num_beams') == config['generation']['reverse_num_beams']
                   and isinstance(entry.get('variant_index'), int)
                   and 0 <= entry['variant_index'] < int(config['variants_per_pivot'])
                   and entry.get('generation_device') in {'mps', 'cpu'}
                   and isinstance(entry.get('pivot_text'), str), identity)
            try:
                stamp = datetime.fromisoformat(entry['generated_utc'])
                valid_stamp = stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            except (ValueError, TypeError, KeyError):
                valid_stamp = False
            record('candidate_utc_metadata', valid_stamp, identity)
            raw_decision = entry['decision']
            conflicts = []
            if raw_decision not in model_failures and text.strip():
                conflicts = [x for kind, value in [('norm', norm(text)), ('punct', punct(text))]
                             for x in index.get((kind, value), []) if x[0] != cls]
            conflict_ids = sorted({x[1] for x in conflicts})
            stored_conflicts = json.loads(entry.get('conflict_with_example_ids', '[]'))
            # All generated batch members are journaled at the quota boundary.
            # Explicit failure/conflict/filter reasons retain precedence; only an
            # otherwise acceptable candidate may become target_reached.
            record('candidate_conflict_reference_alignment', stored_conflicts == conflict_ids, identity)
            if raw_decision in model_failures:
                expected_reason = raw_decision
                reason_counts_not_independently_token_verified[raw_decision] += 1
            elif conflicts:
                expected_reason = 'cross_label_duplicate_conflict'
            else:
                expected_reason = filter_reason(parent['Text'], text, norm_seen, punct_seen) or 'accepted'
            if expected_reason == 'accepted' and original_counts[cls] + accepted_counts[cls] >= config['target_per_minority_class']:
                expected_reason = 'target_reached'
            record('nonmodel_filter_decision_replay', raw_decision == expected_reason, identity)
            by_id[eid] = entry
            final_decisions[eid] = raw_decision
            journal_order.append(eid)
            if raw_decision == 'accepted':
                accepted_counts[cls] += 1
                norm_seen.add(norm(text))
                punct_seen.add(punct(text))
                add_index(text, cls, eid, 'synthetic')
            elif raw_decision == 'cross_label_duplicate_conflict':
                for other_cls, prior_id, origin in conflicts:
                    if origin == 'synthetic' and final_decisions.get(prior_id) == 'accepted':
                        final_decisions[prior_id] = 'quarantined_cross_label_duplicate'
                        accepted_counts[other_cls] -= 1
                        computed_quarantines.add(prior_id)
                add_index(text, cls, eid, 'synthetic')
        declared_quarantines = event.get('quarantined_example_ids', [])
        record('retroactive_cross_label_quarantine_replay',
               len(declared_quarantines) == len(set(declared_quarantines))
               and set(declared_quarantines) == computed_quarantines, line_number)

    for kind in ['journal_job_uniqueness', 'journal_event_parent_class_pivot_revision_alignment',
                 'candidate_ids_unique_and_valid', 'candidate_exact_training_parent_and_label',
                 'candidate_pivot_and_model_metadata', 'candidate_generation_metadata',
                 'candidate_utc_metadata', 'candidate_conflict_reference_alignment',
                 'nonmodel_filter_decision_replay', 'retroactive_cross_label_quarantine_replay']:
        check(kind, errors[kind] == 0, {'violations': errors[kind], 'first_examples': samples[kind]})

    all_csv = tables['all_translation_decisions.csv']
    rejected_csv = tables['rejected_candidates.csv']
    accepted_csv = tables['synthetic_accepted_PROVISIONAL.csv']
    lineage_csv = tables['lineage.csv']
    augmented = tables['train_augmented_PROVISIONAL.csv']
    accepted_ids = [eid for eid in journal_order if final_decisions[eid] == 'accepted']
    rejected_ids = [eid for eid in journal_order if final_decisions[eid] != 'accepted']
    check('all_csv_ids_equal_journal_in_order', [r['example_id'] for r in all_csv] == journal_order)
    check('accepted_csv_ids_equal_final_journal_decisions', [r['example_id'] for r in accepted_csv] == accepted_ids)
    check('lineage_csv_ids_equal_accepted_in_order', [r['example_id'] for r in lineage_csv] == accepted_ids)
    check('rejected_csv_ids_equal_final_journal_decisions', [r['example_id'] for r in rejected_csv] == rejected_ids)
    compare_fields = ['source_id', 'parent_text', 'Class', 'Text', 'pivot_language', 'variant_index',
                      'example_id', 'pivot_text', 'forward_model', 'reverse_model', 'forward_revision',
                      'reverse_revision', 'seed', 'max_input_tokens', 'max_output_tokens', 'num_beams',
                      'generation_device', 'generated_utc']
    for name, rows in [('all_decisions', all_csv), ('rejected', rejected_csv),
                       ('accepted', accepted_csv), ('lineage', lineage_csv)]:
        mismatch = []
        for row in rows:
            entry = by_id.get(row.get('example_id'))
            bad = entry is None or any(row.get(k) != csv_scalar(entry.get(k)) for k in compare_fields)
            if name in {'all_decisions', 'rejected'}:
                bad = bad or row.get('decision') != final_decisions.get(row.get('example_id'))
                bad = bad or row.get('conflict_with_example_ids') != csv_scalar(entry.get('conflict_with_example_ids'))
            else:
                bad = bad or row.get('is_synthetic') != '1' or row.get('label_review_status') != 'UNVERIFIED'
            if bad:
                mismatch.append(row.get('example_id'))
        check(f'{name}_csv_full_candidate_metadata_matches_journal', not mismatch,
              {'violations': len(mismatch), 'first_example_ids': mismatch[:20]})
    conflict_ids = [eid for eid in rejected_ids if 'cross_label' in final_decisions[eid]]
    check('cross_label_csv_matches_all_quarantined_conflicts',
          [r['example_id'] for r in tables['cross_label_conflicts.csv']] == conflict_ids
          and tables['cross_label_conflicts.csv'] == [r for r in rejected_csv if 'cross_label' in r['decision']])
    original_export, synthetic_export = augmented[:len(train)], augmented[len(train):]
    check('augmented_original_prefix_preserves_every_source_in_order', len(original_export) == len(train)
          and all(all(export.get(k) == source[k] for k in ['source_id', 'Text', 'Class'])
                  and export.get('parent_text') == source['Text']
                  and export.get('example_id') == source['source_id']
                  and export.get('is_synthetic') == '0' and export.get('label_review_status') == 'SOURCE_LABEL'
                  for export, source in zip(original_export, train)))
    check('augmented_synthetic_suffix_exactly_matches_accepted_csv', synthetic_export == accepted_csv)
    check('augmented_example_ids_unique', len({r['example_id'] for r in augmented}) == len(augmented))
    manual = tables['MANUAL_REVIEW_SAMPLE.csv']
    accepted_map = {r['example_id']: r for r in accepted_csv}
    check('manual_review_sample_drawn_only_from_unverified_accepted',
          len({r['example_id'] for r in manual}) == len(manual)
          and all(r['example_id'] in accepted_map and r.get('label_review_status') == 'UNVERIFIED'
                  and all(r.get(k) == accepted_map[r['example_id']].get(k) for k in compare_fields)
                  for r in manual)
          and all(sum(int(r['Class']) == cls for r in manual) <= 20 for cls in range(1, 6)))
    check('accepted_lineage_export_byte_identical', sha(read_bytes(RUN / 'lineage.csv'))
          == sha(read_bytes(RUN / 'synthetic_accepted_PROVISIONAL.csv')))

    synthetic_counts = Counter(int(r['Class']) for r in accepted_csv)
    rejected_counts = Counter(r['decision'] for r in rejected_csv)
    all_decision_counts = Counter(final_decisions.values())
    final_counts = Counter(int(r['Class']) for r in augmented)
    target_unmet = {str(k): max(0, config['target_per_minority_class'] - final_counts[k]) for k in range(1, 6)}
    check('accepted_count_and_rejection_counts_match_audit_summary',
          summary.get('synthetic_accepted_count') == len(accepted_csv)
          and summary.get('generated_decision_counts') == dict(all_decision_counts)
          and len(accepted_csv) + len(rejected_csv) == len(journal_order))
    check('class_distributions_and_unmet_targets_match_summary',
          summary.get('per_class_train_original') == {str(k): original_counts[k] for k in range(1, 7)}
          and summary.get('per_class_train_provisional') == {str(k): final_counts[k] for k in range(1, 7)}
          and summary.get('target_unmet') == target_unmet
          and final_counts == original_counts + synthetic_counts
          and accepted_counts == synthetic_counts)
    check('original_and_final_row_counts_match_summary', summary.get('original_train_count') == len(train)
          and summary.get('validation_count_unchanged') == len(validation)
          and summary.get('provisional_train_count') == len(augmented) == len(train) + len(accepted_csv))
    check('clean_class_not_synthetically_oversampled', synthetic_counts[6] == 0
          and final_counts[6] == original_counts[6])
    check('ambiguity_classes_do_not_exceed_requested_target',
          all(final_counts[k] <= max(original_counts[k], config['target_per_minority_class']) for k in range(1, 6)))

    diagnostics = {
        'original_train_exact_text': group_diagnostic(train, lambda text: text),
        'original_train_normalized': group_diagnostic(train, norm),
        'original_train_punctuation': group_diagnostic(train, punct),
        'validation_normalized': group_diagnostic(validation, norm),
        'validation_punctuation': group_diagnostic(validation, punct),
        'original_train_validation_normalized': overlap_diagnostic(train, validation, norm),
        'original_train_validation_punctuation': overlap_diagnostic(train, validation, punct),
        'accepted_synthetic_exact_text': group_diagnostic(accepted_csv, lambda text: text),
        'accepted_synthetic_normalized': group_diagnostic(accepted_csv, norm),
        'accepted_synthetic_punctuation': group_diagnostic(accepted_csv, punct),
        'synthetic_original_train_normalized': overlap_diagnostic(accepted_csv, train, norm),
        'synthetic_original_train_punctuation': overlap_diagnostic(accepted_csv, train, punct),
        'synthetic_validation_normalized': overlap_diagnostic(accepted_csv, validation, norm),
        'synthetic_validation_punctuation': overlap_diagnostic(accepted_csv, validation, punct),
        'augmented_exact_text': group_diagnostic(augmented, lambda text: text),
        'augmented_normalized': group_diagnostic(augmented, norm),
        'augmented_punctuation': group_diagnostic(augmented, punct),
        'augmented_validation_normalized': overlap_diagnostic(augmented, validation, norm),
        'augmented_validation_punctuation': overlap_diagnostic(augmented, validation, punct),
    }
    check('accepted_synthetic_duplicates_zero', all(diagnostics[name]['repeated_rows_beyond_first'] == 0
          for name in ['accepted_synthetic_normalized', 'accepted_synthetic_punctuation']))
    check('accepted_synthetic_overlap_with_both_partitions_zero', all(diagnostics[name]['shared_keys'] == 0
          for name in ['synthetic_original_train_normalized', 'synthetic_original_train_punctuation',
                       'synthetic_validation_normalized', 'synthetic_validation_punctuation']))
    check('normalized_augmented_duplicates_and_validation_overlap_zero',
          diagnostics['augmented_normalized']['repeated_rows_beyond_first'] == 0
          and diagnostics['augmented_validation_normalized']['shared_keys'] == 0)
    # Unique-key count naturally increases with novel synthetic rows; compare the
    # unchanged risk counts separately from population size in the inherited check.
    inherited_fields = ['duplicate_keys', 'repeated_rows_beyond_first', 'rows_in_duplicate_groups',
                        'cross_label_keys', 'rows_in_cross_label_groups']
    check('inherited_punctuation_risks_preserved_without_new_additions',
        all(diagnostics['original_train_punctuation'][k] == diagnostics['augmented_punctuation'][k]
            for k in inherited_fields)
        and diagnostics['original_train_validation_punctuation'] == diagnostics['augmented_validation_punctuation']
        and diagnostics['original_train_punctuation']['duplicate_keys'] == 19
        and diagnostics['original_train_punctuation']['cross_label_keys'] == 3
        and diagnostics['original_train_validation_punctuation']['shared_keys'] == 10
        and diagnostics['original_train_validation_punctuation']['cross_label_shared_keys'] == 1,
        {'interpretation': 'Inherited original punctuation risks are reported separately; they are not new synthetic overlap.'})

    flags, flag_counts, accepted_flag_counts = [], Counter(), Counter()
    for eid in journal_order:
        entry = by_id[eid]
        parent, text = entry['parent_text'], entry['Text']
        pm, cm = phrase_counts(parent, MODALS), phrase_counts(text, MODALS)
        pc, cc = phrase_counts(parent, CONDITIONS), phrase_counts(text, CONDITIONS)
        pn = Counter(re.findall(r"\b(?:not|never|no|without|cannot)\b|n['’]t\b", norm(parent)))
        cn = Counter(re.findall(r"\b(?:not|never|no|without|cannot)\b|n['’]t\b", norm(text)))
        po = Counter(re.findall(r'<=|>=|!=|==|[<>=+*/%−-]', parent))
        co = Counter(re.findall(r'<=|>=|!=|==|[<>=+*/%−-]', text))
        pnum, cnum = number_counts(parent), number_counts(text)
        names = []
        if set(pm) != set(cm): names.append('modal_set_changed')
        if pm != cm: names.append('modal_counts_changed')
        if pn != cn: names.append('negation_counts_changed')
        if pc != cc: names.append('condition_scope_counts_changed')
        if pnum != cnum: names.append('numeric_tokens_changed')
        if po != co: names.append('operator_tokens_changed')
        if not text.strip(): names.append('empty_candidate')
        if names:
            flag_counts.update(names)
            if final_decisions[eid] == 'accepted': accepted_flag_counts.update(names)
            flags.append({
                'example_id': eid, 'source_id': entry['source_id'], 'Class': entry['Class'],
                'pivot_language': entry['pivot_language'], 'final_decision': final_decisions[eid],
                'parent_text': parent, 'Text': text, 'review_flags': ';'.join(names),
                'label_review_status': 'UNVERIFIED', 'action': 'REVIEW_ONLY_NO_FILTER_OR_RELABEL',
                'parent_modal_set': json.dumps(sorted(pm)), 'candidate_modal_set': json.dumps(sorted(cm)),
                'parent_modal_counts': json.dumps(pm, sort_keys=True), 'candidate_modal_counts': json.dumps(cm, sort_keys=True),
                'parent_negation_counts': json.dumps(pn, sort_keys=True), 'candidate_negation_counts': json.dumps(cn, sort_keys=True),
                'parent_condition_counts': json.dumps(pc, sort_keys=True), 'candidate_condition_counts': json.dumps(cc, sort_keys=True),
                'parent_numeric_tokens': json.dumps(pnum, sort_keys=True), 'candidate_numeric_tokens': json.dumps(cnum, sort_keys=True),
                'parent_operator_tokens': json.dumps(po, sort_keys=True), 'candidate_operator_tokens': json.dumps(co, sort_keys=True),
            })
    remaining_jobs = {str(cls): sum((r['source_id'], pivot) not in seen_jobs
                      for r in train if int(r['Class']) == cls for pivot in config['pivot_languages'])
                      for cls in range(1, 6)}
    if any(target_unmet[str(c)] and remaining_jobs[str(c)] for c in range(1, 6)):
        warnings.append('Some below-target classes still have unattempted parent/pivot jobs. Retroactive quarantine after earlier class loops can leave eligible jobs for an unchanged-protocol resume; finite completion is not proof of exhaustive attempts.')
    warnings.extend([
        'All accepted synthetic ambiguity labels remain UNVERIFIED. Automatic lexical filters and a clean lineage audit do not establish semantic label preservation.',
        'Nineteen inherited original-training punctuation duplicate keys (three cross-label) and ten original train/validation punctuation-overlap keys (one cross-label) remain in unchanged originals. No original was removed or relabeled.',
        'Semantic paraphrase/project/document independence is not established. The protected held-out dataset was not accessed for any overlap check.',
        'Generation-limit/input-token failure reasons are recorded model-stage decisions; token-level cap correctness is not independently rerun by this stdlib auditor.',
        'Suspicion flags are broad review cues, including grammatical or harmless paraphrases and punctuation hyphens. They do not adjudicate or filter labels.',
    ])
    end_hashes = {name: sha(read_bytes(ROOT / f'data/splits/{name}.csv')) for name in ['train', 'validation']}
    check('frozen_repository_sources_unchanged_at_end', end_hashes == EXPECTED_HASHES)
    check('summary_declares_labels_unverified_and_validation_only', summary.get('verified_ambiguity_labels') is False
          and summary.get('not_a_test_result') is True and summary.get('read_test') is False)
    # Each independent safety result must agree with the pipeline's summary.
    expected_summary_safety = {
        'overlap_with_validation_normalized': diagnostics['synthetic_validation_normalized']['shared_keys'],
        'overlap_with_validation_punctuation_normalized': diagnostics['synthetic_validation_punctuation']['shared_keys'],
        'synthetic_training_original_overlap_normalized': diagnostics['synthetic_original_train_normalized']['shared_keys'],
        'synthetic_training_original_overlap_punctuation': diagnostics['synthetic_original_train_punctuation']['shared_keys'],
        'synthetic_duplicate_normalized': diagnostics['accepted_synthetic_normalized']['repeated_rows_beyond_first'],
        'synthetic_duplicate_punctuation': diagnostics['accepted_synthetic_punctuation']['repeated_rows_beyond_first'],
    }
    check('pipeline_overlap_summary_matches_independent_counts',
          all(summary.get(k) == v for k, v in expected_summary_safety.items()))
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_WITH_UNVERIFIED_LABELS_AND_INHERITED_PUNCTUATION_RISKS' if all(checks.values()) else 'FAIL',
        'scope': 'Independent stdlib post-run audit, no generation module import/model execution and no dataset changes. Explicit allowed source/run files only; no split-directory traversal.',
        'checks': checks, 'failed_checks': [k for k, v in checks.items() if not v], 'check_details': details,
        'input_hashes': source_hashes, 'final_frozen_repository_hashes': end_hashes,
        'run_artifact_sha256': {name: sha(read_bytes(RUN / name)) for name in [
            'AUDIT.json', 'generation_config.json', 'model_lock.json', 'model_revisions.json',
            'generation_journal.jsonl', *CSV_NAMES]},
        'original_train_rows': len(train), 'validation_rows': len(validation),
        'journal_events': len(seen_jobs), 'journal_candidates': len(journal_order),
        'accepted_synthetic_rows': len(accepted_csv), 'rejected_rows': len(rejected_csv),
        'final_augmented_rows': len(augmented), 'final_decision_counts': dict(all_decision_counts),
        'rejection_counts_by_final_reason': dict(rejected_counts),
        'retroactively_quarantined_prior_acceptances': all_decision_counts['quarantined_cross_label_duplicate'],
        'class_counts_original': {str(k): original_counts[k] for k in range(1, 7)},
        'class_counts_synthetic': {str(k): synthetic_counts[k] for k in range(1, 7)},
        'class_counts_augmented': {str(k): final_counts[k] for k in range(1, 7)},
        'target_unmet': target_unmet, 'unattempted_parent_pivot_jobs_by_class': remaining_jobs,
        'duplicate_and_overlap_diagnostics': diagnostics,
        'model_lock': lock, 'reported_candidate_generation_devices': dict(Counter(r['generation_device'] for r in all_csv)),
        'model_stage_failure_reasons_not_independently_token_verified': dict(reason_counts_not_independently_token_verified),
        'review_only_flags': {'candidate_rows_with_any_flag': len(flags),
            'accepted_rows_with_any_flag': sum(r['final_decision'] == 'accepted' for r in flags),
            'all_candidate_flag_counts_overlapping': dict(flag_counts),
            'accepted_flag_counts_overlapping': dict(accepted_flag_counts),
            'label_review_status': 'UNVERIFIED', 'dataset_mutations': 0},
        'warnings_and_scientific_limitations': warnings,
        'test_accessed': False, 'labels_changed': False, 'datasets_changed': False,
        'model_execution_performed': False,
    }
    reports.mkdir(parents=True, exist_ok=True)
    distributions = [{'Class': k, 'original_train': original_counts[k],
                      'synthetic_accepted_UNVERIFIED': synthetic_counts[k], 'augmented_train_PROVISIONAL': final_counts[k],
                      'unchanged_validation': sum(int(r['Class']) == k for r in validation),
                      'target': config['target_per_minority_class'] if k <= 5 else 'NO_SYNTHETIC_OVERSAMPLING',
                      'target_unmet': target_unmet[str(k)] if k <= 5 else 0} for k in range(1, 7)]
    flag_fields = ['example_id', 'source_id', 'Class', 'pivot_language', 'final_decision', 'parent_text', 'Text',
                   'review_flags', 'label_review_status', 'action', 'parent_modal_set', 'candidate_modal_set',
                   'parent_modal_counts', 'candidate_modal_counts', 'parent_negation_counts', 'candidate_negation_counts',
                   'parent_condition_counts', 'candidate_condition_counts', 'parent_numeric_tokens', 'candidate_numeric_tokens',
                   'parent_operator_tokens', 'candidate_operator_tokens']
    write_csv_exclusive(destinations[1], distributions, list(distributions[0]))
    write_csv_exclusive(destinations[2], flags, flag_fields)
    with destinations[0].open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'failed_checks', 'original_train_rows', 'validation_rows',
                                           'journal_candidates', 'accepted_synthetic_rows', 'rejected_rows',
                                           'final_augmented_rows', 'target_unmet']}, indent=2))
    if report['failed_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
