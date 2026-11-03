"""Independent output reconciliation, metric arithmetic and preservation check.

Never traverses or reads data/splits. Does not refit a model or revise grouping.
"""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def sha(path):
    path = Path(path).resolve()
    if (ROOT / 'data/splits').resolve() in path.parents or path.name.lower() == 'test.csv':
        raise ValueError('Protected path')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((BASE / 'reports/CONSTRUCTION_SUMMARY.json').read_text())
    baseline = json.loads((BASE / 'reports/TFIDF_BASELINE.json').read_text())
    ledger = pd.read_csv(BASE / 'audit/lineage_group_table.csv', keep_default_na=False)
    train = pd.read_csv(BASE / 'data/train.csv')
    valid = pd.read_csv(BASE / 'data/validation.csv')
    pred = pd.read_csv(BASE / 'reports/tfidf_validation_predictions.csv')
    families = pd.read_csv(BASE / 'audit/candidate_families.csv', keep_default_na=False)
    checks = {}
    checks['all_source_rows_have_unique_audit_ids'] = len(ledger) == 24949 and ledger.row_id.is_unique
    checks['exact_disposition_accounting'] = ledger.disposition.value_counts().to_dict() == {
        'quarantined': 20380, 'retained_model_example': 4234,
        'excluded_repeated_normalized_original_text': 335}
    checks['no_verified_parent_ids_fabricated'] = ledger.verified_augmentation_parent_id.eq('').all()
    checks['all_unlinked_excluded'] = ledger.loc[~ledger.normalized_original_content_match].disposition.eq('quarantined').all()
    retained = ledger[ledger.disposition.eq('retained_model_example')]
    combined = pd.concat([train, valid], ignore_index=True)
    checks['partitions_exactly_match_retained_ids'] = set(combined.row_id) == set(retained.row_id) and combined.row_id.is_unique
    checks['group_disjoint'] = not set(train.group_id) & set(valid.group_id)
    checks['candidate_family_sizes_reconcile'] = int(families.legacy_rows.sum()) == 24949
    checks['retained_family_label_sets_unambiguous'] = all(
        families.set_index('group_id').loc[combined.group_id, 'labels'].astype(str).to_numpy() == combined.Class.astype(str).to_numpy())
    for name, frame in [('train', train), ('validation', valid)]:
        checks[f'{name}_hash_bound_before_baseline'] = sha(BASE / f'data/{name}.csv') == baseline['inputs_sha256'][f'data/{name}.csv'] == summary['new_partition_sha256'][name]
    checks['protocol_unchanged_since_freeze'] = sha(BASE / 'config/protocol.json') == (BASE / 'config/protocol.sha256').read_text().split()[0] == baseline['inputs_sha256']['config/protocol.json']
    checks['prediction_ids_order_labels_group_aligned'] = (
        pred.row_id.tolist() == valid.row_id.tolist() and pred.group_id.tolist() == valid.group_id.tolist()
        and pred.y_true.tolist() == valid.Class.tolist())
    probability = pred[[f'p_class_{i}' for i in range(1, 7)]].to_numpy()
    checks['probability_order_and_normalization'] = (
        np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
        and np.array_equal(probability.argmax(axis=1) + 1, pred.y_pred))
    # Recalculate from explicit confusion counts without calling sklearn metrics.
    cm = np.zeros((6, 6), dtype=int)
    np.add.at(cm, (pred.y_true.to_numpy(dtype=int)-1, pred.y_pred.to_numpy(dtype=int)-1), 1)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    tp = np.diag(cm)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(6, dtype=float), where=(support+predicted)>0)
    calculated = {'macro_f1': float(f1.mean()), 'weighted_f1': float(np.dot(f1, support)/support.sum()),
                  'accuracy': float(tp.sum()/cm.sum())}
    checks['saved_metrics_match_independent_confusion_arithmetic'] = all(
        np.isclose(calculated[k], baseline[k], atol=1e-12, rtol=0) for k in calculated)
    checks['saved_confusion_counts_match'] = cm.tolist() == baseline['confusion_matrix_rows_true_columns_predicted']
    checks['all_new_data_ineligible_for_original_test'] = combined.original_frozen_test_eligible.eq(False).all()
    before = json.loads((BASE / 'audit/EXISTING_FILES_BEFORE.json').read_text())
    changed, missing = [], []
    for relative, entry in before.items():
        p = (ROOT / relative).resolve()
        if p.name.lower() == 'test.csv' or (ROOT / 'data/splits').resolve() in p.parents:
            raise ValueError('Protected path accidentally included in preservation snapshot')
        if not p.is_file():
            missing.append(relative)
        elif sha(p) != entry['sha256']:
            changed.append(relative)
    preservation = {'status': 'PASS' if not changed and not missing else 'FAIL',
                    'preexisting_files_checked': len(before), 'changed': changed, 'missing': missing,
                    'scope': 'Pre-existing files from the explicit before snapshot; entire data/splits excluded from traversal, reading and hashing',
                    'protected_split_files_checked': False}
    (BASE / 'reports/SOURCE_PRESERVATION.json').write_text(json.dumps(preservation, indent=2)+'\n')
    checks['preexisting_authorized_files_unchanged'] = preservation['status'] == 'PASS'
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': {k: bool(v) for k, v in checks.items()},
              'independently_recalculated_validation_metrics': calculated,
              'per_class_f1_explicit_order_1_to_6': f1.tolist(),
              'confusion_matrix': cm.tolist(),
              'source_preservation': preservation,
              'no_refitting_or_grouping_changes': True,
              'test_or_old_validation_accessed': False}
    (BASE / 'reports/OUTPUT_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit('Verification failed')


if __name__ == '__main__':
    main()
