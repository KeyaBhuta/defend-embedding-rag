import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from data_input import load_records

class InputTests(unittest.TestCase):
    def test_zip_mapping_limit_and_split_isolation(self):
        row = dict(id='train1', paper_id='p1', paper_text='Paper.',
                   review_text='Concern.', reference_rebuttal='Reply.', record_unit='atomic_concern')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'data.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('data/prepared_train/part-00001.jsonl', json.dumps(row)+'\nBROKEN SECOND RECORD')
                z.writestr('data/prepared_eval/part-00001.jsonl', json.dumps({**row,'id':'eval1'}))
                z.writestr('data/train_real_real.json', 'not a prepared record')
            selected = load_records(path, 'train', 1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['paper'], 'Paper.')
            self.assertEqual(selected[0]['review_segment'], 'Concern.')
            self.assertEqual(selected[0]['author_rebuttal'], 'Reply.')
            self.assertEqual(load_records(path, 'eval', 1)[0]['id'], 'eval1')
            with self.assertRaises(ValueError): load_records(path, 'train')
            with self.assertRaises(ValueError): load_records(path, None, 1)
            with self.assertRaises(ValueError): load_records(path, 'train', 0)

    def test_prepared_jsonl_mapping(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'pairs.jsonl'
            path.write_text(json.dumps(dict(id='x',paper_text='P',review_text='R',reference_rebuttal='A')),encoding='utf-8')
            self.assertEqual(load_records(path)[0]['author_rebuttal'], 'A')

if __name__ == '__main__': unittest.main()
