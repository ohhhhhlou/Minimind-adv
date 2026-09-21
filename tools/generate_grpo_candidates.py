"""Generate training-side candidates for AI preference annotation; no training."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import parse_attributes


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line, value in enumerate(f, 1):
            row = json.loads(value)
            messages = row['conversations']
            if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                raise ValueError(f'{path}:{line}: expected system/user/assistant')
            attributes = parse_attributes(messages[1]['content'])
            key = tuple(sorted(attributes))
            yield line, row, attributes, key


def select_rows(train, excluded, count, seed):
    blocked = {key for path in excluded for _, _, _, key in rows(path)}
    pool = {}
    for line, row, attrs, key in rows(train):
        if row.get('metadata', {}).get('source') != 'train.json':
            raise ValueError(f'{train}:{line}: expected metadata.source=train.json')
        if key not in blocked:
            pool.setdefault(key, (line, row, attrs, key))
    if len(pool) < count:
        raise ValueError(f'Only {len(pool)} eligible unique attribute groups')
    return random.Random(seed).sample(list(pool.values()), count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--train-data', type=Path, default=ROOT/'dataset/advertisegen/advertisegen_sft_train.jsonl')
    p.add_argument('--val-data', type=Path, default=ROOT/'dataset/advertisegen/advertisegen_sft_val.jsonl')
    p.add_argument('--test-data', type=Path, default=ROOT/'dataset/advertisegen/advertisegen_sft_test.jsonl')
    p.add_argument('--output-dir', type=Path, default=ROOT/'grpo_candidates/pilot10')
    p.add_argument('--num-prompts', type=int, default=10)
    p.add_argument('--num-candidates', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-only', action='store_true', help='Validate/select data without loading a model')
    args = p.parse_args()
    if args.num_prompts < 1 or args.num_candidates < 2:
        p.error('num-prompts >= 1 and num-candidates >= 2 required')
    out = args.output_dir.resolve()
    if out.drive.lower() == 'c:':
        p.error('Output on C: is forbidden')
    selected = select_rows(args.train_data, [args.val_data, args.test_data], args.num_prompts, args.seed)
    if not args.prepare_only and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error(f'Checkpoint not found: {args.checkpoint}')
    # Never overwrite a previous run, including incomplete runs.
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='prepared', seed=args.seed, num_prompts=args.num_prompts,
                    num_candidates=args.num_candidates,
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                    data_sha256={str(path.resolve()): digest(path) for path in
                                 [args.train_data, args.val_data, args.test_data]},
                    selection='random unique sorted attribute groups; exclude val/test exact groups',
                    label_source='unlabelled', script_sha256=digest(Path(__file__)),
                    metrics_sha256=digest(ROOT/'evaluation/metrics.py'))
    def save_manifest():
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'selected_prompts.jsonl').open('w', encoding='utf-8') as f:
        for line, row, attrs, key in selected:
            f.write(json.dumps(dict(line_number=line, metadata=row.get('metadata', {}),
                                   messages=row['conversations'][:2], attributes=attrs), ensure_ascii=False)+'\n')
    save_manifest()
    if args.prepare_only:
        print(f'Data selection validated: {out}', flush=True)
        return

    import torch
    import transformers
    from transformers import AutoTokenizer, GenerationConfig
    from model.MokioModel import MokioMindConfig, MokioMindForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(ROOT/'model', local_files_only=True)
    model = MokioMindForCausalLM(MokioMindConfig(hidden_size=512, num_hidden_layers=8,
                                               use_moe=False, flash_attention=False))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device=args.device, dtype=torch.float32)
    config = GenerationConfig(do_sample=True, temperature=0.8, top_p=1.0, top_k=0,
                              repetition_penalty=1.0, no_repeat_ngram_size=0,
                              max_new_tokens=96, num_beams=1, num_return_sequences=1,
                              use_cache=True, pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
    manifest.update(status='running', checkpoint_sha256=digest(args.checkpoint),
                    generation=config.to_dict(), model_config=model.config.to_dict(),
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    dtype='float32', device=args.device,
                    tokenizer_sha256={f.name: digest(f) for f in (ROOT/'model').glob('*.json')})
    save_manifest()
    with (out/'candidates.jsonl').open('w', encoding='utf-8') as f:
        for idx, (line, row, attrs, key) in enumerate(selected):
            group_id = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]
            messages = row['conversations'][:2]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False,
                               return_token_type_ids=False).to(args.device)
            prompt_len = inputs['input_ids'].shape[1]
            if prompt_len + 96 > model.config.max_position_embeddings:
                raise ValueError(f'Context overflow for {group_id}; truncation forbidden')
            candidates = []
            for j in range(args.num_candidates):
                seed = args.seed + idx * args.num_candidates + j
                torch.manual_seed(seed)
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=config)
                raw = generated[0, prompt_len:].tolist()
                eos = tokenizer.eos_token_id in raw
                content = raw[:raw.index(tokenizer.eos_token_id)] if eos else raw
                candidates.append(dict(candidate_id=f'{group_id}-{j+1}', seed=seed,
                                       text=tokenizer.decode(content, skip_special_tokens=True),
                                       raw_token_ids=raw, content_token_ids=content, eos_normal=eos))
            item = dict(group_id=group_id, line_number=line, metadata=row.get('metadata', {}),
                        messages=messages, attributes=attrs, rendered_prompt=prompt,
                        prompt_tokens=prompt_len, candidates=candidates)
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
            f.flush()
            print(f'{idx+1}/{len(selected)} groups saved', flush=True)
    manifest['status'] = 'complete'
    save_manifest()
    print(f'Finished. Return candidates.jsonl and manifest.json from: {out}', flush=True)


if __name__ == '__main__':
    main()
