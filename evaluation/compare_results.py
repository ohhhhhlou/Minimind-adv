"""只读对比两次生成：检查来源、按行号匹配、对共同样本重新汇总。"""
import argparse
import json
from pathlib import Path
from evaluation.metrics import aggregate


def load_run(path):
    """要求完整生成结果，展开重新评分 manifest 以追溯真正生成配置。"""
    def read(name):
        return json.loads((path/name).read_text(encoding='utf-8'))
    summary, manifest = read('summary.json'), read('manifest.json')
    if summary.get('status') != 'complete':
        raise ValueError(f'{path}: incomplete run')
    rows = [json.loads(line) for line in (path/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != summary['samples'] or not rows:
        raise ValueError(f'{path}: sample count mismatch')
    indexed = {r['line_number']: r for r in rows}
    if len(indexed) != len(rows) or any('output_metrics' not in r for r in rows):
        raise ValueError(f'{path}: duplicate IDs or missing generation')
    generation_manifest = manifest
    while 'original_manifest' in generation_manifest:
        generation_manifest = generation_manifest['original_manifest']
    return manifest, generation_manifest, indexed


def compare(left, right, limit=200, samples=10):
    """仅比较相同输入和参考答案；未知/不同来源显式列出，不宣称严格控制。"""
    lm, lg, lr = load_run(Path(left))
    rm, rg, rr = load_run(Path(right))
    checks = {}
    for key in ['checkpoint_sha256', 'data_sha256', 'tokenizer_sha256', 'model_code_sha256',
                'model_config', 'packages', 'python', 'device', 'cuda', 'seed', 'seed_policy']:
        a, b = lg.get(key), rg.get(key)
        checks[key] = dict(status='unknown' if a is None or b is None else 'same' if a == b else 'different', left=a, right=b)
    for key in ['protocol', 'key_attributes']:
        a, b = lm.get(key), rm.get(key)
        checks[key] = dict(status='unknown' if a is None or b is None else 'same' if a == b else 'different', left=a, right=b)
    ids = sorted(set(lr) & set(rr))
    if limit:
        ids = ids[:limit]
    if not ids:
        raise ValueError('No shared samples')
    for i in ids:
        for key in ['messages', 'reference', 'attributes']:
            # attributes serialized as lists; order irrelevant for aggregate coverage.
            a, b = lr[i].get(key), rr[i].get(key)
            if key == 'attributes':
                a, b = sorted(a), sorted(b)
            if a != b:
                raise ValueError(f'Line {i}: {key} differs; paired comparison refused')
    lc, rc = lg.get('generation', {}), rg.get('generation', {})
    differences = {k: {'left': lc.get(k), 'right': rc.get(k)} for k in sorted(lc.keys() | rc.keys()) if lc.get(k) != rc.get(k)}
    def group(selected):
        return {name: aggregate([rows[i]['output_metrics'] for i in selected]) for name, rows in [('left', lr), ('right', rr)]}
    examples = []
    for i in ids[:samples]:
        examples.append(dict(line_number=i, messages=lr[i]['messages'], reference=lr[i]['reference'],
                             **{name: dict(output=rows[i]['output'], metrics=rows[i]['output_metrics'],
                                           finish_reason=rows[i].get('generation', {}).get('finish_reason'))
                                for name, rows in [('left', lr), ('right', rr)]}))
    return dict(left=str(left), right=str(right), checks=checks, generation_differences=differences,
                compared_samples=len(ids), selection='first shared line numbers, ascending',
                left_only=len(set(lr)-set(rr)), right_only=len(set(rr)-set(lr)),
                overall=group(ids), by_input_style={style: group([i for i in ids if lr[i].get('metadata', {}).get('input_style', 'unknown') == style])
                for style in sorted({lr[i].get('metadata', {}).get('input_style', 'unknown') for i in ids})}, examples=examples)


def main():
    """输出完整 JSON 对照，可由终端重定向为报告文件。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--samples', type=int, default=10)
    a = p.parse_args()
    if min(a.limit, a.samples) < 0:
        p.error('limit/samples must be >= 0')
    try:
        print(json.dumps(compare(a.left, a.right, a.limit, a.samples), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(1, f'Comparison failed: {exc}\n')


if __name__ == '__main__':
    main()
