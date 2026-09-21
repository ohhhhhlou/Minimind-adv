"""验证旧生成不变、非覆盖分数不变，以及结果目录保护。"""
import json
from pathlib import Path
import tempfile
import unittest
from evaluation.metrics import score
from evaluation.rescore import COVERAGE_FIELDS, rescore_row, rescore_directory


class RescoreTests(unittest.TestCase):
    def sample(self):
        """模拟 v1 把多值当作一个字符串的已生成记录。"""
        attrs = [('材质','混纺、纤维')]
        text = '纤维混纺面料'
        return dict(messages=[dict(role='user', content='材质为混纺、纤维。')],
                    attributes=attrs, reference=text, output=text,
                    reference_metrics=score(text, attrs, [5,6]),
                    output_metrics=score(text, attrs, [3,4], True),
                    generation=dict(raw_token_ids=[3,4,2], eos_normal=True))

    def test_preserve_original_and_noncoverage(self):
        old = self.sample()
        new = rescore_row(old)
        self.assertEqual(old['output_metrics']['literal_coverage'], 0)
        for field in ['output_metrics','reference_metrics']:
            self.assertEqual(new[field]['literal_coverage'], 1)
            for key in old[field].keys() - set(COVERAGE_FIELDS):
                self.assertEqual(new[field][key], old[field][key])
        self.assertEqual(new['generation'], old['generation'])
        self.assertEqual(new['output'], old['output'])

    def test_directory_guards_and_report(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source, dest = Path(folder)/'old', Path(folder)/'new'
            source.mkdir()
            (source/'samples.jsonl').write_text(json.dumps(self.sample())+'\n', encoding='utf-8')
            (source/'manifest.json').write_text('{"protocol":"advertisegen-literal-v1"}', encoding='utf-8')
            (source/'summary.json').write_text('{"status":"complete","samples":2}', encoding='utf-8')
            with self.assertRaises(ValueError):
                rescore_directory(source, dest)
            self.assertFalse(dest.exists())
            (source/'summary.json').write_text('{"status":"complete","samples":1}', encoding='utf-8')
            report = rescore_directory(source, dest)
            self.assertEqual(report['overall']['output_metrics']['literal_coverage']['mean'], 1)
            with self.assertRaises(FileExistsError):
                rescore_directory(source, dest)


if __name__ == '__main__':
    unittest.main()
