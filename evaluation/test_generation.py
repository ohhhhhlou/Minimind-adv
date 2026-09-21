"""不加载模型验证唯一协议及 CLI 的配置记录。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from evaluation.generation import generation_kwargs, SEED


class GenerationTests(unittest.TestCase):
    def test_diagnostics_change_exactly_one_factor(self):
        expected = {'length256': 'max_new_tokens', 'greedy': 'do_sample',
                    'no_penalty': 'repetition_penalty', 'no_ngram': 'no_repeat_ngram_size'}
        base = generation_kwargs()
        for name, key in expected.items():
            changed = generation_kwargs(name)
            self.assertEqual([k for k in base if base[k] != changed[k]], [key])

    def test_frozen_single_candidate_protocol(self):
        expected = dict(do_sample=True, temperature=0.75, top_p=0.90, top_k=0,
                        repetition_penalty=1.15, no_repeat_ngram_size=3,
                        max_new_tokens=96, num_beams=1, num_return_sequences=1, use_cache=True)
        self.assertEqual(generation_kwargs(), expected)
        self.assertEqual(SEED, 2026)
        changed = generation_kwargs()
        changed['max_new_tokens'] = 999
        self.assertEqual(generation_kwargs(), expected)

    def test_cli_manifest_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'evaluation') as folder:
            command = [sys.executable, '-B', '-m', 'evaluation.run', '--reference-only',
                       '--limit', '2', '--generation-profile', 'standard',
                       '--output-dir', folder, '--run-name', 'smoke']
            subprocess.run(command, cwd=root, check=True, capture_output=True)
            manifest = json.loads((Path(folder)/'smoke/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['requested_generation'], generation_kwargs())
            self.assertEqual(manifest['seed'], 2026)
            self.assertEqual(manifest['generation_profile'], 'standard')
            self.assertEqual(subprocess.run(command, cwd=root, capture_output=True).returncode, 1)
            invalid = subprocess.run([sys.executable, '-B', '-m', 'evaluation.run',
                                      '--generation-profile', 'baseline'], cwd=root, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
