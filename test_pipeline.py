"""Offline workflow tests: replace model calls with schema-validated fixtures."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pipeline


class PipelineTests(unittest.TestCase):
    def test_paired_workflow_continues_with_unverified_quote_without_leaking_reference(self):
        calls = []
        row = {'id': 'sample', 'paper': 'Paper content.',
               'review_segment': 'The review raises a specific concern.',
               'author_rebuttal': 'REFERENCE_ONLY_SECRET'}

        def init(runner, args):
            runner.args = args
            runner.root = Path(args.output)
            runner.root.mkdir()

        def call(runner, stage, data, schema, judge=False):
            calls.append((stage, copy.deepcopy(data)))
            if stage in ('dp', 'etp', 'rap'):
                label = {'dp': 'no', 'etp': 'none', 'rap': 'answer_question'}[stage]
                quote = data['review_segment']
                if stage == 'rap' and 'author_rebuttal' in data:
                    quote = 'Invented evidence that appears in neither source.'
                value = dict(label=label, explanation='Fixture explanation.', uncertainty='',
                             evidence=[dict(source='review_segment', quote=quote)])
            elif stage == 'synthesis':
                value = dict(coherent_plan='Fixture plan.', rebuttal='Fixture reply.', author_checks=[])
            elif stage == 'verify':
                # Even an incorrectly approving judge cannot override bad quotes.
                value = dict(dp=2, etp=2, rap=2, synthesis=2, issues=[],
                             inconsistent_stages=[], passed=True)
            else:
                value = {key: 'Fixture comparison.' for key in schema.model_fields}
            return schema.model_validate(value).model_dump()

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / 'input.jsonl'
            source.write_text(json.dumps(row) + '\n', encoding='utf-8')
            output = Path(directory) / 'run'
            argv = ['pipeline.py', '--input', str(source), '--output', str(output)]
            with patch.object(pipeline.Runner, '__init__', init), patch.object(pipeline.Runner, 'call', call), patch.object(sys, 'argv', argv):
                pipeline.main()
            result = json.loads((output/'results.jsonl').read_text())
        self.assertEqual([s for s, _ in calls], ['dp','etp','rap','synthesis','verify']*2+['compare'])
        for _, data in calls[:5]:
            self.assertNotIn('REFERENCE_ONLY_SECRET', json.dumps(data))
        for _, data in calls[5:10]:
            self.assertEqual(data['author_rebuttal'], row['author_rebuttal'])
        self.assertNotIn('error', result)
        self.assertIn('comparison', result)
        self.assertTrue(result['without_rebuttal']['accepted'])
        self.assertFalse(result['with_rebuttal']['accepted'])
        self.assertTrue(result['with_rebuttal']['rap']['unverified_evidence'])

    def test_saved_failed_rap_quote_restores_original_source(self):
        run = Path(__file__).parent/'runs'/'trial_fixed_01'
        if not run.exists():
            self.skipTest('Historical trial is not included in distribution')
        request = json.loads((run/'call_00008_rap.request.json').read_text(encoding='utf-8'))
        ctx = json.loads(request['messages'][1]['content'])
        response = json.loads((run/'call_00008_rap.response.json').read_text(encoding='utf-8'))
        decision = response['parsed']
        changes, issues = pipeline.validate_evidence(decision, ctx, 'rap')
        self.assertTrue(changes)
        for change in changes:
            self.assertEqual(ctx[change['source']][change['source_start']:change['source_end']], change['corrected_quote'])
        self.assertTrue(any('primary reason' in x['original_quote'] for x in changes))
        # The same response also omitted citation markers in another quote.
        # Keep that substantive mismatch flagged; it must not stop the workflow.
        self.assertTrue(any('Moreover' in x['quote'] for x in issues))


if __name__ == '__main__':
    unittest.main()
