import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsi_memory.demo_v4 import OUTPUTS, bundle, render_html
from rsi_memory.topology import fingerprint


def settled_report(seed, kind='training_paired_declared_identity_v41'):
    blocks = ['t0', 't1']
    program = {'schema': 1, 'kind': 'kv_memory_program', 'rules': []}
    programs = {'native': None, **dict.fromkeys(OUTPUTS, program)}
    scores = {'native': [5, 3], 'manager_best': [2, 4], 'admitted_output': [3, 5], 'fixed': [2, 4]}
    manifest = {'seed_sha256': seed, 'config': {'min_improvement': 0.5, 'rounds': 1}, 'contract': {
        'selection_protocol': {'kind': kind}, 'paired_block_ids': {'test': blocks},
        'split_case_sha256': {'test': ['digest']}}, 'manual_program': None}
    trial = {'id': 'p0', 'harness_sha256': seed, 'program': program, 'program_sha256': fingerprint(program)}
    choice = {'proposal_id': 'p0', 'source_sha256': seed, 'effect_sha256': 'effect'}
    matrix = {'cohort_proposals': {'p0': trial}, 'plan': {}, 'measurements': {}}
    row = {'round': 0, 'active_before': seed, 'active_after': seed, 'proposed_child_sha256': None,
           'decision': 'rejected', 'v4_training': matrix, 'v4_selection': dict.fromkeys(OUTPUTS, choice),
           'parent': [trial], 'child': [], 'fixed': [],
           'adaptive_meta_plan': {'expected_effect': '</script><b>hypothesis</b>'}}
    frozen = {**{name: {'trial': trial, 'selection': choice} for name in OUTPUTS}, 'partial': False,
              'rounds': 1, 'active_harness_sha256': seed, 'admitted_harness_sha256s': [seed],
              'manifest_sha256': fingerprint(manifest), 'rounds_sha256': fingerprint([row]),
              'training_matrix_sha256': fingerprint(matrix)}
    pair = {'status': 'completed', 'input': {'split': 'test', 'block_ids': blocks, 'programs': programs},
            'result': {name: {'feasible': True, 'samples': [
                {'block_id': block, 'score': ms, 'case_scores_ms': {'agent': ms}}
                for block, ms in zip(blocks, values, strict=True)]} for name, values in scores.items()}}
    comparisons, relations = {}, {}
    for name in OUTPUTS[:2]:
        for reference in ('native', 'fixed'):
            key = f'{name}_vs_{reference}'
            first, second = (b - a for a, b in zip(scores[name], scores[reference], strict=True))
            comparisons[key] = {'passed': False, 'reason': 'insufficient_stable_gain',
                                'median_improvement': (first + second) / 2,
                                'deltas': [{'block_id': 't0', 'reference_minus_candidate': first},
                                           {'block_id': 't1', 'reference_minus_candidate': second}]}
            relations[key] = {'same_raw_program': reference == 'fixed'}
    return {'status': 'test_settled', 'manifest': manifest, 'rounds': [row], 'frozen': frozen,
            'paired_evaluations': {'pair/frozen/test': pair}, 'usage': {},
            'test_comparisons': comparisons, 'test_program_relations': relations}


class DemoV4Tests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        source = b'# archived source\n'
        self.seed = sha256(source).hexdigest()
        (self.root / (self.seed + '.py')).write_bytes(source)
        self.path = self.root / 'report.json'

    def read(self, report):
        self.path.write_text(json.dumps(report), encoding='utf-8')
        return bundle(self.path, self.root)

    def test_settled_versions_and_paired_sign(self):
        for kind, version in [('training_paired_declared_identity_v4', '4'),
                              ('training_paired_declared_identity_v41', '4.1')]:
            with self.subTest(kind=kind):
                data = self.read(settled_report(self.seed, kind))
                self.assertEqual(data['version'], version)
                self.assertEqual(data['case_deltas'][0]['deltas_ms'], [3, -1])
                self.assertEqual(data['case_deltas'][0]['median_ms'], 1)
                self.assertTrue(data['relations']['manager_best_vs_fixed'])
                html = render_html(data)
                self.assertIn(r'\u003c/script\u003e\u003cb\u003ehypothesis', html)
                self.assertNotIn('</script><b>hypothesis', html)

    def test_rejects_unsafe_threshold_and_source_before_source_read(self):
        report = settled_report(self.seed)
        report['manifest']['config']['min_improvement'] = '<b>untrusted</b>'
        with self.assertRaisesRegex(ValueError, 'finite number'):
            self.read(report)
        report = settled_report(self.seed)
        report['rounds'][0]['proposed_child_sha256'] = '../outside'
        report['frozen']['rounds_sha256'] = fingerprint(report['rounds'])
        original = Path.read_bytes

        def read_only_report(path):
            self.assertEqual(path, self.path)
            return original(path)

        with patch.object(Path, 'read_bytes', read_only_report):
            with self.assertRaisesRegex(ValueError, 'lowercase SHA-256'):
                self.read(report)

    def test_rejects_partial_failed_mismatched_and_invalid_test(self):
        for defect, message in [('checkpoint', 'test_settled'), ('partial', 'completed training archive'),
                                ('failed', 'did not complete'), ('program', 'differs from frozen programs'),
                                ('frozen_program', 'differs from its selected program'),
                                ('blocks', 'predeclared blocks'), ('cases', 'disagree on the case set'),
                                ('score', 'invalid scores'), ('summary', 'disagree with paired samples')]:
            with self.subTest(defect=defect):
                report = settled_report(self.seed)
                pair = report['paired_evaluations']['pair/frozen/test']
                if defect == 'checkpoint':
                    report.update(status='checkpointed', frozen=None)
                elif defect == 'partial':
                    report['frozen']['partial'] = True
                elif defect == 'failed':
                    pair['status'] = 'failed'
                elif defect == 'program':
                    pair['input']['programs']['fixed'] = {'kind': 'different'}
                elif defect == 'frozen_program':
                    report['frozen']['fixed']['trial'] = {'program': {'kind': 'different'}}
                elif defect == 'blocks':
                    for arm in pair['result'].values():
                        arm['samples'].pop()
                elif defect == 'cases':
                    pair['result']['fixed']['samples'][0]['case_scores_ms'] = {'unexpected': 2}
                elif defect == 'score':
                    pair['result']['fixed']['samples'][0]['score'] = float('inf')
                else:
                    report['test_comparisons']['admitted_output_vs_native']['passed'] = True
                with self.assertRaisesRegex(ValueError, message):
                    self.read(report)
