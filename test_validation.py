"""Offline regression tests. Run: python -m unittest test_validation -v"""
import unittest
from validation import validate_evidence, audit_accepted

SOURCE = 'The paper uses only the 20 images available in the SinGAN Github repository;'

class ValidationTests(unittest.TestCase):
    def decision(self, quote, source='review_segment'):
        return {'evidence': [{'source': source, 'quote': quote}]}

    def test_exact(self):
        self.assertEqual(validate_evidence(self.decision(SOURCE), {'review_segment': SOURCE}, 'dp'), ([], []))

    def test_reported_punctuation_failure(self):
        d = self.decision(SOURCE[:-1] + '.')
        adjustments, issues = validate_evidence(d, {'review_segment': SOURCE}, 'dp')
        self.assertEqual(issues, [])
        self.assertEqual(d['evidence'][0]['quote'], SOURCE)
        self.assertEqual(adjustments[0]['original_quote'], SOURCE[:-1] + '.')

    def test_reject_changed_meaning_empty_wrong_source_and_ambiguity(self):
        for quote, source, text in [
            (SOURCE.replace('20', '200'), 'review_segment', SOURCE),
            ('', 'review_segment', SOURCE),
            (' ', 'review_segment', SOURCE),
            (SOURCE, 'paper', SOURCE),
            (SOURCE[:-1] + '.', 'review_segment', SOURCE + '\n' + SOURCE)]:
            with self.subTest(quote=quote, source=source):
                _, issues = validate_evidence(self.decision(quote, source), {'review_segment': text}, 'dp')
                self.assertTrue(issues)

    def test_encoding_offsets_after_multiple_broken_characters(self):
        text = 'Earlier SinGANâ€™s text. The primary reason to use SinGANâ€™s repository is model security.'
        original = text[text.index('The primary'):]
        quote = original.replace('â€™', '’').lower()
        d = self.decision(quote, 'author_rebuttal')
        changes, issues = validate_evidence(d, {'author_rebuttal': text}, 'rap')
        self.assertEqual(issues, [])
        self.assertEqual(d['evidence'][0]['quote'], original)
        self.assertEqual(text[changes[0]['source_start']:changes[0]['source_end']], original)

    def test_whitespace_and_missing_evidence(self):
        text = 'The paper uses\n\nonly 20 images.'
        d = self.decision('the paper uses only 20 images.')
        _, issues = validate_evidence(d, {'review_segment': text}, 'dp')
        self.assertFalse(issues)
        self.assertEqual(d['evidence'][0]['quote'], text)
        self.assertTrue(validate_evidence({'evidence': []}, {}, 'dp')[1])

    def test_inconsistent_audit_cannot_pass(self):
        a = dict(dp=2, etp=2, rap=2, synthesis=2, passed=True,
                 issues=[], inconsistent_stages=['dp'])
        self.assertFalse(audit_accepted(a))
        a['inconsistent_stages'] = []
        self.assertTrue(audit_accepted(a))
        self.assertFalse(audit_accepted(a, has_unverified_evidence=True))

if __name__ == '__main__':
    unittest.main()
