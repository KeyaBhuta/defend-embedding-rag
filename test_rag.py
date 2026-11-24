"""Offline invariants with fake tokenizer/embeddings, never downloaded models."""
import copy
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rag import (Retriever, chunk_paper, section_chunks, select_chunks, evidence_text,
                 token_count, context_hash, validate_retrieved)
from prepare_rag import select_inputs
from prepare_rag_sft import review_template, build_examples
from rag_experiments import student_messages, check_method, parse_output, Rebuttal
from pipeline import context
import rag_experiments
import prepare_rag_sft


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i+1) for i in range(len(text))]}


class FakeEncoder:
    max_seq_length = 10000
    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, prompt='', **kwargs):
        self.calls.append((list(texts), prompt))
        return np.array([[1, t.lower().count('images') * 5, t.lower().count('noise') * 5]
                         for t in texts], dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()
        self.row = {'id': 'p:r0:s0', 'paper_id': 'p', 'record_unit': 'atomic_concern',
                    'paper': 'We use twenty images.\n\nNoise is a method detail.\n\nLimitations remain.',
                    'review_segment': 'Why twenty images?', 'author_rebuttal': 'SECRET_REFERENCE'}

    def get_row(self, root, encoder=None, row=None):
        r = Retriever(self.tok, encoder or FakeEncoder(), root, {'model': 'fake'}, 25, 4)
        return r.retrieve(row or self.row, top_k=2, budget=150)

    def test_chunk_coverage_exact_offsets_and_limits(self):
        for paper in (self.row['paper'], 'a'*83, 'Î±Î²\n\nç ”ç©¶ paper\nTable 1\na b c'):
            chunks = chunk_paper(paper, self.tok, 20, 3)
            covered = set()
            for c in chunks:
                self.assertEqual(c['text'], paper[c['start']:c['end']])
                self.assertLessEqual(c['tokens'], 20)
                covered.update(range(c['start'], c['end']))
            self.assertEqual(covered, set(range(len(paper))))
        chunks = chunk_paper('abcdefghij\n\nklmnopqrstuvwxyz', self.tok, 20, 3)
        self.assertTrue(chunks[0]['text'].endswith('\n\n'))

    def test_rank_budget_and_overlap_deduplication(self):
        chunks = chunk_paper('abcdefghij'*8, self.tok, 20, 8)
        selected = select_chunks(chunks, list(reversed(range(len(chunks)))), self.tok, 6, 190)
        self.assertEqual(selected[0]['rank'], 1)
        self.assertLessEqual(token_count(self.tok, evidence_text(selected)), 190)
        spans = [set(range(c['start'], c['end'])) for c in selected]
        for i in range(len(spans)):
            for j in range(i):
                self.assertFalse(spans[i] & spans[j])
        with self.assertRaises(ValueError):
            select_chunks(chunks, [1]*len(chunks), self.tok, 1, 1)

    def test_section_chunks_keep_paragraphs_and_display_math_whole(self):
        first = '# Short\r\n\r\nSmall.\r\n\r\n'
        equation = '$$\r\na + b\r\n\r\n= c\r\n$$'
        paper = first + '# Long\r\n\r\nLead paragraph.\r\n\r\n' + equation + '\r\n\r\nAfter.'
        chunks = section_chunks(paper, self.tok, size=22)
        self.assertEqual(chunks[0]['text'], first)
        self.assertEqual(''.join(c['text'] for c in chunks), paper)
        self.assertEqual(sum(equation in c['text'] for c in chunks), 1)
        self.assertTrue(any(c['tokens'] > 22 for c in chunks))
        for chunk in chunks:
            self.assertEqual(chunk['text'], paper[chunk['start']:chunk['end']])

    def test_whole_selection_skips_passages_that_do_not_fit(self):
        paper = '# Big\n\n' + 'a'*100 + '\n\n# Small\n\nShort.'
        chunks = section_chunks(paper, self.tok, size=20)
        selected = select_chunks(chunks, [1, 0], self.tok, top_k=6, budget=70, whole=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['text'], chunks[1]['text'])

    def test_cache_reuse_invalidation_and_reference_isolation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            encoder = FakeEncoder()
            first = self.get_row(root, encoder)
            second = self.get_row(root, encoder, dict(self.row, author_rebuttal='DIFFERENT_SECRET'))
            self.assertFalse(first['retrieval']['cache_hit'])
            self.assertTrue(second['retrieval']['cache_hit'])
            self.assertEqual(first['paper'], second['paper'])
            self.assertEqual(len(encoder.calls), 3)  # documents once, query twice
            self.assertNotIn('SECRET', json.dumps(encoder.calls))
            self.assertIn('twenty images', first['paper'])
            changed = self.get_row(root, encoder, dict(self.row, paper=self.row['paper']+' Extra.'))
            self.assertNotEqual(first['retrieval']['index_key'], changed['retrieval']['index_key'])
            other_paper = self.get_row(root, encoder, dict(self.row, paper_id='other'))
            self.assertNotEqual(first['retrieval']['index_key'], other_paper['retrieval']['index_key'])

    def test_no_silent_embedding_truncation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            encoder = FakeEncoder()
            encoder.max_seq_length = 10
            with self.assertRaisesRegex(ValueError, 'no truncation'):
                self.get_row(root, encoder)

    def test_source_and_context_integrity(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(root)
            validate_retrieved(row)
            self.assertNotIn('SECRET', json.dumps(context(row, 'without_rebuttal')))
            self.assertNotIn('SECRET', json.dumps(student_messages(row)))
            row['paper'] += 'invented'
            with self.assertRaises(ValueError):
                validate_retrieved(row)

    def test_cache_tampering_detected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(root)
            path = Path(root)/row['retrieval']['index_key']/'vectors.npy'
            vectors = np.load(path)
            np.save(path, vectors + 1)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.get_row(root)

    def test_full_review_is_not_inferred_from_atomic_concern(self):
        with self.assertRaises(ValueError):
            select_inputs([self.row], 'review')
        with self.assertRaises(ValueError):
            select_inputs([self.row, dict(self.row, paper='conflicting')], 'concern')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(root)
            with self.assertRaises(ValueError):
                check_method([row], 'drg')
            check_method([row], 'swrg')

    def test_sft_approval_context_and_split_checks(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(root)
            teacher = {row['id']: {'without_rebuttal': {'accepted': True,
                'synthesis': {'coherent_plan': 'Candidate plan.', 'rebuttal': 'Candidate reply.'}},
                'student_context_sha256': 'DIFFERENT_FULL_PAPER_CONTEXT'}}
            target = list(review_template([row], teacher))[0]
            self.assertFalse(target['approved'])
            self.assertFalse(target['teacher_context_matches'])
            self.assertEqual(build_examples([row], {row['id']: target}, 'both', 'train')[0], [])
            target.update(approved=True, review_note='Checked source and target.',
                          coherent_reasoning='Evidence supports the stated limitation.')
            examples, rejected = build_examples([row], {row['id']: target}, 'both', 'train')
            self.assertEqual(len(examples), 1)
            self.assertFalse(rejected)
            plain, reasoning = examples[0]['plain'], examples[0]['reasoning']
            self.assertEqual(plain['messages'][1], reasoning['messages'][1])
            self.assertEqual(json.loads(plain['messages'][2]['content'])['rebuttal'],
                             json.loads(reasoning['messages'][2]['content'])['rebuttal'])
            self.assertNotIn('Candidate reply', plain['messages'][1]['content'])
            with self.assertRaises(ValueError):
                build_examples([row], {row['id']: target}, 'both', 'train', {'p'})
            target['student_context_sha256'] = 'stale'
            with self.assertRaises(ValueError):
                build_examples([row], {row['id']: target}, 'both', 'train')
            target['student_context_sha256'] = context_hash(row)
            row['retrieval']['source_split'] = 'eval'
            with self.assertRaises(ValueError):
                build_examples([row], {row['id']: target}, 'both', 'train')

    def test_invalid_model_output_not_silently_accepted(self):
        self.assertEqual(parse_output('```json\n{"rebuttal":"reply"}\n```', Rebuttal), {'rebuttal':'reply'})
        with self.assertRaises(ValueError):
            parse_output('Here is a reply but not JSON', Rebuttal)

    def test_drg_coverage_accepts_numbered_concern_paragraphs(self):
        response = '{"rebuttal":{"concern_10":"Last.","concern_2":"Second.","concern_1":"First."}}'
        with self.assertRaises(ValueError):
            parse_output(response, Rebuttal)
        self.assertEqual(
            parse_output(response, Rebuttal, allow_concern_map=True),
            {'rebuttal': 'First.\n\nSecond.\n\nLast.'},
        )
        with self.assertRaises(ValueError):
            parse_output('{"rebuttal":{"concern_1":["First."]}}', Rebuttal, allow_concern_map=True)

    def test_local_generation_cli_uses_saved_evidence_without_reference(self):
        captured = []

        def fake_init(runner, args):
            runner.root = Path(args.output)
            runner.root.mkdir()

        def fake_generate(runner, stage, messages, schema):
            captured.append(messages)
            return {'rebuttal': 'A fixture reply.'}

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(Path(root)/'cache')
            source = Path(root)/'retrieval.jsonl'
            source.write_text(json.dumps(row)+'\n', encoding='utf-8')
            output = Path(root)/'run'
            argv = ['rag_experiments.py', '--input', str(source), '--limit', '1',
                    '--method', 'swrg', '--output', str(output), '--run']
            with patch.object(sys, 'argv', argv), \
                 patch.object(rag_experiments.LocalRunner, '__init__', fake_init), \
                 patch.object(rag_experiments.LocalRunner, 'generate', fake_generate):
                rag_experiments.main()
            result = json.loads((output/'results.jsonl').read_text())
            self.assertEqual(result['student_context_sha256'], context_hash(row))
            self.assertEqual(json.loads(captured[0][1]['content'])['paper'], row['paper'])
            self.assertNotIn('SECRET', json.dumps(captured))

    def test_sft_export_cli_writes_matched_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            row = self.get_row(Path(root)/'cache')
            source, targets = Path(root)/'retrieval.jsonl', Path(root)/'targets.jsonl'
            source.write_text(json.dumps(row)+'\n', encoding='utf-8')
            target = list(review_template([row]))[0]
            target.update(approved=True, review_note='Offline fixture only.',
                          coherent_reasoning='The evidence describes the sample size.', rebuttal='Fixture.')
            targets.write_text(json.dumps(target)+'\n', encoding='utf-8')
            out = Path(root)/'sft'
            argv = ['prepare_rag_sft.py', '--input', str(source), '--limit', '1',
                    '--targets', str(targets), '--partition', 'train', '--output', str(out)]
            with patch.object(sys, 'argv', argv):
                prepare_rag_sft.main()
            self.assertTrue((out/'plain.jsonl').exists())
            self.assertTrue((out/'reasoning.jsonl').exists())
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['exported'], 1)
            self.assertFalse(report['training_started'])


if __name__ == '__main__':
    unittest.main()
