"""Offline integrity fixtures only: these are NOT actual translation results.

All source rows below are invented in-memory fixtures. The input-loader boundary
is patched, outputs live in temporary folders, and Torch/network are never used.
"""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ambifix_integrity_test_fixture', ROOT / 'scripts/backtranslate_with_lineage.py')
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)


def fixture_sources():
    train = []
    for cls, token in enumerate(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'], 1):
        text = f'The system shall record {token} events.'
        train.append({'source_id': bt.parent_id(text), 'Text': text, 'Class': str(cls)})
    text = 'The portal shall export archived items.'
    valid = [{'source_id': bt.parent_id(text), 'Text': text, 'Class': '6'}]
    return train, valid


def arguments(output, **changes):
    values = dict(
        train='IN_MEMORY_FIXTURE_TRAIN', validation='IN_MEMORY_FIXTURE_VALIDATION',
        output=str(output), target_per_class=2, pivots=['fr'], variants_per_pivot=1,
        seed=42, batch_size=1, max_input_tokens=192, max_output_tokens=192,
        device='cpu', dry_run=False, model_lock=None,
    )
    values.update(changes)
    return argparse.Namespace(**values)


class FakeTranslator:
    """Deterministic mock, intentionally compatible with the supplied old API."""
    revisions = {
        'forward': 'OFFLINE_MOCK', 'reverse': 'OFFLINE_MOCK',
        'forward_id': 'OFFLINE_MOCK_FORWARD', 'reverse_id': 'OFFLINE_MOCK_REVERSE',
    }
    device = 'OFFLINE_MOCK'

    def __init__(self, pivot):
        self.pivot = pivot
        self.calls = 0

    def eligible(self, text):
        return True

    def batch(self, texts, variants):
        self.calls += 1
        return [[text.replace('record', 'log') for _ in range(variants)] for text in texts]

    def close(self):
        pass


class InterruptingTranslator(FakeTranslator):
    def batch(self, texts, variants):
        if self.calls == 1:
            raise KeyboardInterrupt('INTENTIONAL_OFFLINE_FIXTURE_INTERRUPTION')
        return super().batch(texts, variants)


class SharedCrossLabelTranslator(FakeTranslator):
    def batch(self, texts, variants):
        return [['The system shall log shared events.' for _ in range(variants)] for _ in texts]


class LoggedFailureTranslator(FakeTranslator):
    def eligible_reason(self, text):
        return 'input_token_limit' if 'alpha' in text else None

    def batch_records(self, texts, variants):
        records = []
        for text in texts:
            reason = ('forward_generation_limit' if 'beta' in text else
                      'reverse_generation_limit' if 'gamma' in text else
                      'reverse_empty' if 'delta' in text else
                      'reverse_input_token_limit')
            records.append([{'Text': '', 'pivot_text': 'OFFLINE_MOCK_PIVOT', 'failure_reason': reason}])
        return records


class DuplicateTranslator(FakeTranslator):
    def batch(self, texts, variants):
        answer = []
        for text in texts:
            if 'alpha' in text:
                candidate = '  ' + text.upper() + '  '
            elif 'beta' in text:
                candidate = text + '!'
            else:
                candidate = text
            answer.append([candidate for _ in range(variants)])
        return answer


class IntegrityResumeTests(unittest.TestCase):
    def run_fixture(self, output, translator=FakeTranslator, **changes):
        train, valid = fixture_sources()
        with patch.object(bt, 'check_inputs', return_value=(train, valid)), redirect_stdout(io.StringIO()):
            return bt.produce(arguments(output, **changes), translator_factory=translator)

    @staticmethod
    def stable_membership(output):
        rows = bt.read_csv(Path(output) / 'synthetic_accepted_PROVISIONAL.csv')
        return sorted((r['source_id'], r['parent_text'], r['Class'], r['Text'],
                       r['pivot_language'], r['example_id']) for r in rows)

    def test_interrupted_mock_job_resumes_to_same_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete, resumed = root / 'complete', root / 'resumed'
            expected = self.run_fixture(complete)
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(resumed, InterruptingTranslator)
            journal = resumed / 'generation_journal.jsonl'
            self.assertTrue(journal.is_file())
            committed = [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
            self.assertGreater(len(committed), 0)
            actual = self.run_fixture(resumed)
            self.assertEqual(self.stable_membership(complete), self.stable_membership(resumed))
            self.assertEqual(expected['per_class_train_provisional'], actual['per_class_train_provisional'])
            events = [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
            keys = [(e['source_id'], e['pivot']) for e in events if e.get('source_id')]
            self.assertEqual(len(keys), len(set(keys)))

    def test_incomplete_final_journal_write_is_recovered_and_logged(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            self.run_fixture(out)
            before = self.stable_membership(out)
            journal = out / 'generation_journal.jsonl'
            with journal.open('ab') as handle:
                handle.write(b'{"source_id":"UNCOMMITTED_OFFLINE_TAIL"')
            self.run_fixture(out)
            self.assertEqual(before, self.stable_membership(out))
            self.assertTrue((out / 'journal_tail_recovery.json').is_file())
            for line in journal.read_text().splitlines():
                if line.strip():
                    json.loads(line)

    def test_journal_middle_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            self.run_fixture(out)
            journal = out / 'generation_journal.jsonl'
            lines = journal.read_bytes().splitlines(keepends=True)
            self.assertGreater(len(lines), 1)
            journal.write_bytes(b'{INTENTIONAL_OFFLINE_CORRUPTION}\n' + b''.join(lines[1:]))
            with self.assertRaises((ValueError, RuntimeError)):
                self.run_fixture(out)

    def test_valid_json_candidate_metadata_or_identity_tampering_fails(self):
        changes = {
            'source_id': 'UNRELATED_OFFLINE_SOURCE',
            'parent_text': 'A different invented requirement.',
            'Class': 2,
            'pivot_language': 'de',
            'forward_model': 'UNDECLARED_OFFLINE_FORWARD_MODEL',
            'reverse_model': 'UNDECLARED_OFFLINE_REVERSE_MODEL',
            'forward_revision': 'a' * 40,
            'reverse_revision': 'b' * 40,
            'seed': 99,
            'max_input_tokens': 128,
            'max_output_tokens': 128,
            'num_beams': 7,
            'example_id': 'f' * 24,
            'Text': 'The system shall log changed events.',
        }
        for field, replacement in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'out'
                self.run_fixture(out)
                journal = out / 'generation_journal.jsonl'
                events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
                self.assertEqual(events[0]['candidates'][0]['Class'], 1)
                events[0]['candidates'][0][field] = replacement
                # Preserve valid JSON and newline termination: syntax recovery must
                # never be mistaken for validation of altered scientific metadata.
                journal.write_text(''.join(json.dumps(e) + '\n' for e in events))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_fixture(out)

    def test_resume_refuses_changed_revision_or_model_identity_lock(self):
        for field, value in [('forward', 'a' * 40),
                             ('forward_id', 'UNDECLARED_OFFLINE_TRANSLATION_MODEL')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / 'out'
                self.run_fixture(out)
                changed = json.loads((out / 'model_lock.json').read_text())
                changed['fr'][field] = value
                requested = root / 'changed_model_lock.json'
                requested.write_text(json.dumps(changed))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_fixture(out, model_lock=str(requested))

    def test_literal_protected_input_is_rejected_before_filesystem_access(self):
        # These are fictitious path strings. Patching every path-resolution/read
        # boundary proves the actual held-out file is never resolved or touched.
        for name in ['test.csv', 'TeSt.CsV']:
            forbidden = 'OFFLINE_NEVER_CREATED/' + name
            callbacks = [
                lambda p=forbidden: bt.safe_path(p),
                lambda p=forbidden: bt.digest(p),
                lambda p=forbidden: bt.read_csv(p),
                lambda p=forbidden: bt.check_inputs(p, 'OFFLINE_NEVER_CREATED/validation.csv'),
            ]
            for number, callback in enumerate(callbacks):
                with self.subTest(name=name, boundary=number):
                    with patch.object(Path, 'resolve', side_effect=AssertionError('path resolution attempted')) as resolve, \
                         patch.object(Path, 'stat', side_effect=AssertionError('stat attempted')) as stat, \
                         patch.object(Path, 'is_file', side_effect=AssertionError('file check attempted')) as is_file, \
                         patch.object(Path, 'open', side_effect=AssertionError('open attempted')) as open_file, \
                         patch.object(Path, 'read_bytes', side_effect=AssertionError('read attempted')) as read_bytes:
                        with self.assertRaises(ValueError):
                            callback()
                        for method in [resolve, stat, is_file, open_file, read_bytes]:
                            method.assert_not_called()

    def test_changed_token_limits_cannot_resume_in_same_output(self):
        for setting in ['max_input_tokens', 'max_output_tokens']:
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'out'
                self.run_fixture(out)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_fixture(out, **{setting: 128})

    def test_target_boundary_logs_every_already_generated_batch_member(self):
        # Two class-1 originals need one addition to reach three. Both members
        # are translated together, so both generated candidates must be audited.
        train, valid = fixture_sources()
        text = 'The system shall record omega events.'
        train.append({'source_id': bt.parent_id(text), 'Text': text, 'Class': '1'})
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            with patch.object(bt, 'check_inputs', return_value=(train, valid)), redirect_stdout(io.StringIO()):
                summary = bt.produce(arguments(out, target_per_class=3, batch_size=2),
                                     translator_factory=FakeTranslator)
            rows = bt.read_csv(out / 'all_translation_decisions.csv')
            class_one = [r for r in rows if str(r['Class']) == '1']
            self.assertEqual(len(class_one), 2)
            self.assertEqual({r['source_id'] for r in class_one},
                             {r['source_id'] for r in train if r['Class'] == '1'})
            self.assertEqual(sorted(r['decision'] for r in class_one), ['accepted', 'target_reached'])
            journal = [json.loads(line) for line in (out / 'generation_journal.jsonl').read_text().splitlines()
                       if line.strip()]
            class_events = [e for e in journal if str(e['Class']) == '1']
            self.assertEqual(len(class_events), 2)
            self.assertEqual(sum(len(e['candidates']) for e in class_events), 2)
            self.assertEqual(summary['per_class_train_provisional']['1'], 3)
            accepted = bt.read_csv(out / 'synthetic_accepted_PROVISIONAL.csv')
            self.assertEqual(sum(str(r['Class']) == '1' for r in accepted), 1)

    def test_changed_variants_or_seed_cannot_resume(self):
        for change in [{'variants_per_pivot': 2}, {'seed': 99}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'out'
                self.run_fixture(out)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_fixture(out, **change)

    def test_parent_label_and_generation_metadata_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            self.run_fixture(out)
            train, valid = fixture_sources()
            parents = {r['source_id']: r for r in train}
            valid_ids = {r['source_id'] for r in valid}
            synthetic = bt.read_csv(out / 'synthetic_accepted_PROVISIONAL.csv')
            self.assertEqual(len(synthetic), 5)
            for row in synthetic:
                parent = parents[row['source_id']]
                self.assertEqual(row['parent_text'], parent['Text'])
                self.assertEqual(str(row['Class']), parent['Class'])
                self.assertNotEqual(str(row['Class']), '6')
                self.assertNotIn(row['source_id'], valid_ids)
                self.assertEqual(row['pivot_language'], 'fr')
                self.assertEqual(row['label_review_status'], 'UNVERIFIED')
                for field in ['forward_model', 'reverse_model', 'forward_revision', 'reverse_revision',
                              'generation_device', 'seed', 'max_input_tokens', 'max_output_tokens', 'num_beams']:
                    self.assertIn(field, row)
                    self.assertNotEqual(str(row[field]).strip(), '', field)
            augmented = bt.read_csv(out / 'train_augmented_PROVISIONAL.csv')
            original = [r for r in augmented if str(r['is_synthetic']) == '0']
            self.assertEqual({(r['source_id'], r['Text'], str(r['Class'])) for r in original},
                             {(r['source_id'], r['Text'], r['Class']) for r in train})

    def test_cross_label_collision_quarantines_prior_synthetic(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            summary = self.run_fixture(out, SharedCrossLabelTranslator)
            accepted = bt.read_csv(out / 'synthetic_accepted_PROVISIONAL.csv')
            self.assertFalse(any(bt.normalize(r['Text']) == bt.normalize('The system shall log shared events.')
                                 for r in accepted))
            decisions = bt.read_csv(out / 'all_translation_decisions.csv')
            reasons = {r['decision'] for r in decisions}
            self.assertIn('cross_label_duplicate_conflict', reasons)
            self.assertIn('quarantined_cross_label_duplicate', reasons)
            train, _ = fixture_sources()
            labels = {r['source_id']: r['Class'] for r in train}
            for row in decisions:
                self.assertEqual(str(row['Class']), labels[row['source_id']])
            augmented = bt.read_csv(out / 'train_augmented_PROVISIONAL.csv')
            actual_counts = {str(cls): sum(str(r['Class']) == str(cls) for r in augmented)
                             for cls in range(1, 7)}
            self.assertEqual(summary['per_class_train_provisional'], actual_counts)
            before = self.stable_membership(out)
            resumed = self.run_fixture(out, SharedCrossLabelTranslator)
            self.assertEqual(before, self.stable_membership(out))
            self.assertEqual(resumed['per_class_train_provisional'], actual_counts)

    def test_unsafe_generation_paths_have_logged_reasons(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            self.run_fixture(out, LoggedFailureTranslator)
            rows = bt.read_csv(out / 'all_translation_decisions.csv')
            reasons = {r['decision'] for r in rows}
            self.assertTrue({'input_token_limit', 'forward_generation_limit', 'reverse_generation_limit',
                             'reverse_empty', 'reverse_input_token_limit'}.issubset(reasons))
            self.assertEqual(len(bt.read_csv(out / 'synthetic_accepted_PROVISIONAL.csv')), 0)

    def test_normalized_and_punctuation_duplicates_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'out'
            self.run_fixture(out, DuplicateTranslator)
            rows = bt.read_csv(out / 'all_translation_decisions.csv')
            reasons = {r['decision'] for r in rows}
            self.assertIn('same_or_duplicate', reasons)
            self.assertIn('punctuation_equivalent_duplicate', reasons)
            self.assertEqual(len(bt.read_csv(out / 'synthetic_accepted_PROVISIONAL.csv')), 0)


if __name__ == '__main__':
    unittest.main()
