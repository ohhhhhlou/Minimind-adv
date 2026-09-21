import tempfile
import unittest
from pathlib import Path

from tools.evaluate_grpo_candidates import evaluate_file, evaluate_group


class EvaluateCandidatesTests(unittest.TestCase):
    def group(self):
        return dict(
            group_id="g",
            line_number=1,
            attributes=[["颜色", "白色"]],
            candidates=[
                dict(candidate_id="a", seed=1, text="白色裙子", content_token_ids=[1, 2], eos_normal=True),
                dict(candidate_id="b", seed=2, text="红色裙子", content_token_ids=[1, 2], eos_normal=False),
            ],
        )

    def test_scores_each_candidate_without_reference_text(self):
        result = evaluate_group(self.group())
        self.assertEqual(result["candidates"][0]["metrics"]["literal_coverage"], 1.0)
        self.assertEqual(result["candidates"][1]["metrics"]["literal_coverage"], 0.0)

    def test_writes_auditable_artifacts(self):
        with tempfile.TemporaryDirectory(dir="F:\\codex\\MokioMind\\MokioMind\\tmp") as temp:
            source = Path(temp) / "candidates.jsonl"
            source.write_text('{"group_id":"g","attributes":[["颜色","白色"]],"candidates":'
                              '[{"candidate_id":"a","text":"白色","content_token_ids":[1]},'
                              '{"candidate_id":"b","text":"红色","content_token_ids":[2]}]}\n',
                              encoding="utf-8")
            output = evaluate_file(source, Path(temp) / "evaluation")
            self.assertTrue((output / "candidate_metrics.jsonl").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
