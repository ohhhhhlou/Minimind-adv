"""Score saved candidates with the project's existing InternLM2 loader. No training."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score_group(group, score_fn):
    messages = group['messages']
    if [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Expected system/user only; reference must not enter scoring')
    results = []
    seen = set()
    for candidate in group['candidates']:
        cid = candidate['candidate_id']
        if cid in seen:
            raise ValueError(f'Duplicate candidate ID: {cid}')
        seen.add(cid)
        conversation = messages + [dict(role='assistant', content=candidate['text'])]
        value = float(score_fn(conversation))
        if not math.isfinite(value):
            raise ValueError(f'Non-finite reward for {cid}: {value}')
        results.append(dict(candidate_id=cid, reward_model_score_raw=value))
    return dict(group_id=group['group_id'], scores=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--reward-model-path', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    out = (args.output_dir or args.candidates.parent/'external_reward').resolve()
    if out.drive.lower() == 'c:':
        p.error('Output on C: is forbidden')
    if not args.candidates.is_file() or not args.reward_model_path.is_dir():
        p.error('Candidate file and reward model directory must already exist')
    # Import the verified project implementation; never substitute tokenizer logic.
    import torch
    import transformers
    import sentencepiece
    from trainer.train_grpo import load_reward_components, get_checked_reward_score
    from tools.generate_grpo_candidates import digest

    if transformers.__version__ != '4.56.2' or sentencepiece.__version__ != '0.2.0':
        p.error('Use the existing GRPO environment: transformers=4.56.2, sentencepiece=0.2.0. Do not upgrade it.')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='loading', candidate_sha256=digest(args.candidates),
                    reward_model_path=str(args.reward_model_path.resolve()),
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    sentencepiece_version=sentencepiece.__version__,
                    loader_sha256=digest(ROOT/'trainer/train_grpo.py'),
                    script_sha256=digest(Path(__file__)),
                    reward_files_sha256={str(f.relative_to(args.reward_model_path)):digest(f)
                                         for f in sorted(args.reward_model_path.rglob('*'))
                                         if f.is_file() and f.suffix in ('.json', '.py', '.model', '.safetensors', '.bin')},
                    score_policy='raw scalar; no clipping, normalization, weights or preference labels')
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    model, tokenizer = load_reward_components(str(args.reward_model_path.resolve()), args.device)
    def score_fn(conversation):
        with torch.inference_mode():
            return get_checked_reward_score(model, tokenizer, conversation)
    count = 0
    with args.candidates.open(encoding='utf-8') as src, (out/'scores.jsonl').open('w', encoding='utf-8') as dst:
        for line in src:
            result = score_group(json.loads(line), score_fn)
            dst.write(json.dumps(result, ensure_ascii=False)+'\n')
            dst.flush()
            count += 1
            print(f'{count} groups scored', flush=True)
    if not count:
        raise ValueError('Empty candidate file')
    manifest.update(status='complete', groups=count)
    save()
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()
