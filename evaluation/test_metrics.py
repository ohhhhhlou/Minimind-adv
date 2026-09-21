import unittest
from evaluation.metrics import parse_attributes, score, aggregate, repetition_ratio, is_loop


class MetricsTests(unittest.TestCase):
    def test_multivalue_style_parity_and_partial_credit(self):
        structured = '类型#裤*材质#混纺*材质#纤维*风格#街头*风格#简约*衣款式#拼接*衣款式#口袋'
        natural = '类型为裤，材质为混纺、纤维，风格为街头、简约，衣款式为拼接、口袋。'
        self.assertEqual(parse_attributes(structured), parse_attributes(natural))
        s = score('街头裤，纤维混纺面料，拼接设计', parse_attributes(natural))
        self.assertEqual(s['attribute_hits'], 5)
        self.assertEqual(s['missing_attributes'], [['风格','简约'], ['衣款式','口袋']])

    def test_multivalue_boundaries(self):
        self.assertEqual(parse_attributes('材质为混纺、 纤维 、混纺'), [('材质','混纺'),('材质','纤维')])
        self.assertEqual(parse_attributes('图案为黑/白，风格为甜美和休闲'), [('图案','黑/白'),('风格','甜美和休闲')])
        self.assertEqual(parse_attributes('图案#黑、白'), [('图案','黑、白')])
        for text in ['材质为混纺、、纤维', '材质为、混纺', '材质为混纺、']:
            with self.assertRaises(ValueError):
                parse_attributes(text)

    def test_input_styles_equivalent(self):
        expected = [('类型', '裙'), ('版型', '宽松'), ('风格', '文艺')]
        self.assertEqual(parse_attributes('商品属性：类型#裙*版型#宽松*风格#文艺'), expected)
        self.assertEqual(parse_attributes('请根据以下商品信息生成一段中文电商广告文案：类型为裙，版型为宽松，风格为文艺。'), expected)

    def test_no_silent_partial_parsing(self):
        for text in ['', '商品属性：类型#', '类型#裙*坏字段', '类型为裙，随便发挥', '类型#裙#裤']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_attributes(text)

    def test_duplicates_and_multi_values(self):
        self.assertEqual(parse_attributes('颜色#红色*颜色#蓝色*颜色#红色'), [('颜色','红色'), ('颜色','蓝色')])

    def test_literal_not_semantic(self):
        s = score('宽松裙子', [('类型','裙'), ('版型','宽松'), ('风格','文艺')], [1, 3], True)
        self.assertEqual(s['literal_coverage'], 2/3)
        self.assertEqual(s['key_literal_coverage'], 1)
        self.assertFalse(s['all_attributes_covered'])
        self.assertEqual(score('宽大的裙子', [('版型','宽松')])['literal_coverage'], 0)

    def test_empty_and_not_applicable(self):
        s = score('', [('风格','文艺')])
        self.assertEqual(s['literal_coverage'], 0)
        self.assertIsNone(s['key_literal_coverage'])
        self.assertIsNone(s['eos_normal'])
        self.assertIsNone(s['token_length'])
        self.assertFalse(s['loop'])

    def test_repetition_and_loop(self):
        self.assertEqual(repetition_ratio('aaaaa', 4), 0.5)
        self.assertEqual(repetition_ratio('abc', 4), 0)
        self.assertTrue(is_loop('前缀好看好看好看后缀'))
        self.assertFalse(is_loop('好看好看'))

    def test_macro_micro_and_eos_denominator(self):
        scores = [score('红', [('颜色','红')], [3], True),
                  score('', [('颜色','红'), ('版型','宽松')], [], False)]
        result = aggregate(scores)
        self.assertEqual(result['literal_coverage']['mean'], 0.5)
        self.assertEqual(result['literal_coverage_micro'], 1/3)
        self.assertEqual(result['eos_normal'], {'mean':0.5, 'n':2})
        self.assertEqual(aggregate([score('文艺', [('风格','文艺')])])['eos_normal'], {'mean':None, 'n':0})


if __name__ == '__main__':
    unittest.main()
