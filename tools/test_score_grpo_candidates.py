import unittest
from tools.score_grpo_candidates import score_group


class ScoreCandidatesTests(unittest.TestCase):
    def group(self):
        return dict(group_id='g', messages=[dict(role='system', content='写广告'),
                    dict(role='user', content='颜色：白色')], candidates=[
                    dict(candidate_id='a', text='白色裙子'), dict(candidate_id='b', text='红色裙子')])

    def test_preserves_raw_scores_and_isolates_candidates(self):
        group = self.group()
        calls = []
        def scorer(chat):
            calls.append(chat)
            return 7.5 if chat[-1]['content'] == '白色裙子' else -4
        result = score_group(group, scorer)
        self.assertEqual([s['reward_model_score_raw'] for s in result['scores']], [7.5, -4])
        self.assertEqual([len(chat) for chat in calls], [3, 3])
        self.assertEqual(len(group['messages']), 2)
        self.assertNotIn('reward_model_score_raw', group['candidates'][0])

    def test_rejects_reference_leakage(self):
        group = self.group()
        group['messages'].append(dict(role='assistant', content='参考答案'))
        with self.assertRaises(ValueError):
            score_group(group, lambda _: 0)

    def test_rejects_nonfinite_scores(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                score_group(self.group(), lambda _: value)

    def test_rejects_duplicate_ids(self):
        group = self.group()
        group['candidates'][1]['candidate_id'] = 'a'
        with self.assertRaises(ValueError):
            score_group(group, lambda _: 0)


if __name__ == '__main__':
    unittest.main()
