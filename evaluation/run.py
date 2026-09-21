### 测评入口
"""python -B -m evaluation.run --help"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone
from evaluation.metrics import VERSION, KEY_ATTRIBUTES, read_rows, score, aggregate
from evaluation.generation import PROFILE, GENERATION_VERSION, SEED, generation_kwargs, DIAGNOSTICS

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--data', type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'out/full_sft_best_512.pth')
    p.add_argument('--tokenizer', type=Path, default=ROOT/'model')
    p.add_argument('--output-dir', type=Path, default=Path('/root/autodl-tmp/MokioMind/evaluation_results'))
    p.add_argument('--run-name', default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--hidden-size', type=int, default=512)
    p.add_argument('--num-hidden-layers', type=int, default=8)
    p.add_argument('--use-moe', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--generation-profile', choices=[PROFILE], default=PROFILE,
                   help='唯一 standard 采样协议：96 tokens，seed=2026；不支持旧配置模式')
    p.add_argument('--reference-only', action='store_true', help='No model or tokenizer; token/EOS metrics are null')
    p.add_argument('--diagnostic', choices=DIAGNOSTICS, default='none',
                   help='单因素诊断：相对于 standard 仅改变一个配置，不是正式新协议')
    args = p.parse_args()
    config = generation_kwargs(args.diagnostic)
    if args.diagnostic != 'none' and (args.split != 'val' or args.limit not in (20, 200)):
        p.error('diagnostic runs require --split val and --limit 20 or 200')
    if args.limit < 0:
        p.error('limit >= 0 required')
    output = args.output_dir.resolve()
    if output.drive.lower() == 'c:':
        p.error('Writing evaluation artifacts to C: is forbidden')
    name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if Path(name).name != name or name in ('.', '..'):
        p.error('run-name must be a single directory name')
    data = (args.data or ROOT/f'dataset/advertisegen/advertisegen_sft_{args.split}.jsonl').resolve()
    rows = read_rows(data, args.limit)
    run = output/name
    run.mkdir(parents=True, exist_ok=False)
    # Set caches before importing any ML/EvalScope modules.
    for key, suffix in {'HF_HOME':'hf', 'MODELSCOPE_CACHE':'modelscope', 'XDG_CACHE_HOME':'xdg',
                        'TORCH_HOME':'torch', 'TMPDIR':'tmp', 'TEMP':'tmp', 'TMP':'tmp'}.items():
        target = output/'cache'/suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    api = None
    manifest = dict(protocol=VERSION, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    generation_profile=PROFILE, generation_protocol=GENERATION_VERSION,
                    requested_generation=config, seed=SEED, diagnostic=args.diagnostic,
                    seed_policy='reset torch seed before each sample',
                    sample_selection='first N rows in file order; limit=0 means all',
                    data_sha256=sha256(data), key_attributes=sorted(KEY_ATTRIBUTES),
                    python=platform.python_version(), mode='reference-only' if args.reference_only else 'model',
                    code_sha256={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
                    model_code_sha256=sha256(ROOT/'model/MokioModel.py'))
    if not args.reference_only:
        import torch
        from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
        from evalscope.api.model import GenerateConfig
        from evaluation.model_api import MokioMindAPI
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        api = MokioMindAPI(str(args.checkpoint.resolve()), tokenizer_path=str(args.tokenizer.resolve()),
                           device=args.device, hidden_size=args.hidden_size,
                           num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe,
                           diagnostic=args.diagnostic, max_new_tokens=config['max_new_tokens'])
        manifest.update(checkpoint_sha256=sha256(args.checkpoint),
                        tokenizer_sha256={f.name:sha256(f) for f in args.tokenizer.glob('*.json')},
                        generation=api.generation.to_dict(), model_config=api.model.config.to_dict(),
                        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','evalscope']},
                        device=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
                        cuda=torch.version.cuda, seed=SEED)
    (run/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    with (run/'samples.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            reference = row['conversations'][-1]['content']
            reference_ids = api.tokenizer.encode(reference, add_special_tokens=False) if api else None
            item = dict(line_number=row['line_number'], metadata=row.get('metadata', {}),
                        messages=row['conversations'][:-1], attributes=row['attributes'], reference=reference,
                        reference_metrics=score(reference, row['attributes'], reference_ids))
            if api:
                messages = [ChatMessageSystem(content=item['messages'][0]['content']),
                            ChatMessageUser(content=item['messages'][1]['content'])]
                response = api.generate(messages, [], 'none', GenerateConfig(max_tokens=config['max_new_tokens']))
                text = response.choices[0].message.content
                item.update(output=text, generation=api.last_result,
                            output_metrics=score(text, row['attributes'], api.last_result['content_token_ids'],
                                                 api.last_result['eos_normal']))
            handle.write(json.dumps(item, ensure_ascii=False)+'\n')
            handle.flush()
            results.append(item)
            if len(results) % 20 == 0:
                print(f'{len(results)}/{len(rows)}', flush=True)
    def summary(group):
        return {field:aggregate([r[field] for r in group]) for field in
                (['reference_metrics', 'output_metrics'] if api else ['reference_metrics'])}
    report = dict(status='complete', samples=len(results), overall=summary(results),
                  by_input_style={style:summary([r for r in results if r['metadata'].get('input_style','unknown') == style])
                                  for style in sorted({r['metadata'].get('input_style','unknown') for r in results})})
    (run/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(run)


if __name__ == '__main__':
    main()
