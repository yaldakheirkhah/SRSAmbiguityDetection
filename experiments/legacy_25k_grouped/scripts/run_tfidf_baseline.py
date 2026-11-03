"""One training-only fitted baseline on the new partial-recovery partitions."""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
import hashlib
import importlib.metadata
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
LABELS = [1, 2, 3, 4, 5, 6]


def sha(path):
    path = Path(path).resolve()
    if BASE not in path.parents:
        raise ValueError('Only this experiment may be read')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = BASE / 'reports/TFIDF_BASELINE.json'
    if output.exists():
        raise SystemExit('Existing baseline found; no repeated outcome-driven runs.')
    cfg = json.loads((BASE / 'config/baseline.json').read_text())
    gate = json.loads((BASE / 'reports/QUALITY_GATE.json').read_text())
    summary = json.loads((BASE / 'reports/CONSTRUCTION_SUMMARY.json').read_text())
    assert gate['status'] == 'PASS_PARTIAL_RECOVERY' and gate['approved_for_gpu_package'] is True
    assert gate['eligible_for_original_frozen_test'] is False
    assert all(gate['checks'].values())
    for name in ['train', 'validation']:
        assert sha(BASE / f'data/{name}.csv') == summary['new_partition_sha256'][name]
    train = pd.read_csv(BASE / 'data/train.csv')
    valid = pd.read_csv(BASE / 'data/validation.csv')
    assert train.row_id.is_unique and valid.row_id.is_unique
    assert not set(train.row_id) & set(valid.row_id)
    assert not set(train.group_id) & set(valid.group_id)
    assert set(train.Class) == set(valid.Class) == set(LABELS)
    kwargs = dict(cfg['vectorizer'])
    kwargs['ngram_range'] = tuple(kwargs['ngram_range'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**kwargs)),
        ('lr', LogisticRegression(**cfg['classifier'])),
    ])
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(train.Text, train.Class)
    # No validation text/labels enter any fit operation.
    prediction = pipeline.predict(valid.Text)
    probabilities = pipeline.predict_proba(valid.Text)
    assert list(pipeline.classes_) == LABELS
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(prediction, np.asarray(LABELS)[probabilities.argmax(axis=1)])
    report = classification_report(valid.Class, prediction, labels=LABELS, output_dict=True, zero_division=0)
    macro = float(f1_score(valid.Class, prediction, labels=LABELS, average='macro', zero_division=0))
    weighted = float(f1_score(valid.Class, prediction, labels=LABELS, average='weighted', zero_division=0))
    accuracy = float(accuracy_score(valid.Class, prediction))
    assert np.isclose(macro, report['macro avg']['f1-score'])
    assert np.isclose(weighted, report['weighted avg']['f1-score'])
    pred = valid[['row_id', 'group_id', 'Class']].rename(columns={'Class': 'y_true'}).copy()
    pred['y_pred'] = prediction
    for j, label in enumerate(LABELS):
        pred[f'p_class_{label}'] = probabilities[:, j]
    pred.to_csv(BASE / 'reports/tfidf_validation_predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(valid.Class, prediction, labels=LABELS),
                 index=[f'true_{x}' for x in LABELS], columns=[f'pred_{x}' for x in LABELS]).to_csv(
                     BASE / 'reports/tfidf_confusion_matrix.csv')
    majority_class = int(train.Class.value_counts().idxmax())
    majority = np.full(len(valid), majority_class)
    result = {
        'benchmark_id': summary['benchmark_id'],
        'status': 'LOCALLY_EXECUTED_VALIDATION_ONLY',
        'recipe': cfg,
        'inputs_sha256': {str(p.relative_to(BASE)): sha(p) for p in [
            BASE / 'data/train.csv', BASE / 'data/validation.csv', BASE / 'config/baseline.json',
            BASE / 'config/protocol.json', BASE / 'reports/QUALITY_GATE.json']},
        'train_rows': len(train), 'validation_rows': len(valid), 'class_order': LABELS,
        'macro_f1': macro, 'weighted_f1': weighted, 'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix_rows_true_columns_predicted': confusion_matrix(valid.Class, prediction, labels=LABELS).tolist(),
        'majority_reference': {
            'class_selected_from_train': majority_class,
            'macro_f1': float(f1_score(valid.Class, majority, labels=LABELS, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(valid.Class, majority, labels=LABELS, average='weighted', zero_division=0)),
            'accuracy': float(accuracy_score(valid.Class, majority)),
        },
        'training_only_vocabulary_features': len(pipeline.named_steps['tfidf'].vocabulary_),
        'training_only_balanced_class_weights_formula': 'n_train / (6 * n_train_class)',
        'classifier_iterations': pipeline.named_steps['lr'].n_iter_.tolist(),
        'warnings': [str(w.message) for w in caught],
        'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught),
        'fit_and_evaluation_seconds': time.monotonic() - start,
        'environment': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib']},
        'test_evaluation': False,
        'eligible_for_original_frozen_test': False,
        'old_validation_comparable': False,
        'grouping_or_threshold_changes_after_metrics': False,
    }
    (BASE / 'models').mkdir(exist_ok=True)
    joblib.dump(pipeline, BASE / 'models/tfidf_balanced_lr.joblib', compress=3)
    result['saved_pipeline_sha256'] = sha(BASE / 'models/tfidf_balanced_lr.joblib')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'train_rows', 'validation_rows', 'macro_f1', 'weighted_f1', 'accuracy', 'classifier_iterations', 'convergence_warning']}, indent=2))


if __name__ == '__main__':
    main()
