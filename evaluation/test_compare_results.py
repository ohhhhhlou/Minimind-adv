import json
from pathlib import Path
import tempfile
import unittest
from evaluation.compare_results import compare
from evaluation.metrics import score


class CompareTests(unittest.TestCase):
    def test_pairing_nested_manifest_and_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            left, right = Path(temp)/'left', Path(temp)/'right'
            row = dict(line_number=1, messages=[dict(role='user', content='类型#裙')],
                       reference='裙', attributes=[['类型','裙']], output='裙',
                       output_metrics=score('裙', [('类型','裙')], [4], True))
            original = dict(checkpoint_sha256='abc', generation={'do_sample':False})
            for directory in [left, right]:
                directory.mkdir()
                (directory/'summary.json').write_text('{"status":"complete","samples":1}', encoding='utf-8')
                (directory/'manifest.json').write_text(json.dumps(dict(protocol='v2',original_manifest=original)), encoding='utf-8')
                (directory/'samples.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
            report = compare(left,right)
            self.assertEqual(report['checks']['checkpoint_sha256']['status'], 'same')
            self.assertEqual(report['checks']['tokenizer_sha256']['status'], 'unknown')
            self.assertEqual(report['compared_samples'], 1)
            row['messages'][0]['content'] = '类型#裤'
            (right/'samples.jsonl').write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaises(ValueError):
                compare(left,right)
