"""仅修正保存结果的属性覆盖指标，不加载模型/tokenizer，不重新生成。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from evaluation.metrics import VERSION, KEY_ATTRIBUTES, parse_attributes, score, aggregate
from evaluation.run import sha256

COVERAGE_FIELDS = (
    'attribute_hits', 'attribute_count', 'key_attribute_hits', 'key_attribute_count',
    'literal_coverage', 'key_literal_coverage', 'all_attributes_covered', 'missing_attributes',
)


def rescore_row(row):
    """从原用户消息重新解析属性，只更新覆盖字段，保留生成及其余指标。"""
    result = deepcopy(row)
    users = [m['content'] for m in result['messages'] if m['role'] == 'user']
    if len(users) != 1:
        raise ValueError('Expected exactly one user message')
    result['attributes'] = parse_attributes(users[0])
    for text_key, metric_key in [('reference', 'reference_metrics'), ('output', 'output_metrics')]:
        if text_key == 'output' and text_key not in result:
            if metric_key in result:
                raise ValueError('Output metrics without output text')
            continue
        metrics = score(result[text_key], result['attributes'])
        for key in COVERAGE_FIELDS:
            result[metric_key][key] = metrics[key]
    return result


def rescore_directory(source, destination):
    """验证原报告完整性，另存新版逐条结果、汇总及来源哈希，拒绝覆盖目录。"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.drive.lower() == 'c:':
        raise ValueError('Writing evaluation artifacts to C: is forbidden')
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('Source evaluation is not complete')
    rows = []
    with (source/'samples.jsonl').open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            try:
                rows.append(rescore_row(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'samples.jsonl:{number}: {exc}') from exc
    if not rows or len(rows) != summary['samples']:
        raise ValueError('Source summary/sample count mismatch or empty result')
    generated = ['output' in row for row in rows]
    if any(generated) != all(generated):
        raise ValueError('Mixed reference-only and generated samples')
    def summarize(group):
        """使用新版覆盖指标重建汇总，其余逐条分数原样参与统计。"""
        fields = ['reference_metrics', 'output_metrics'] if all(generated) else ['reference_metrics']
        return {field: aggregate([r[field] for r in group]) for field in fields}
    report = dict(status='complete', protocol=VERSION, samples=len(rows), overall=summarize(rows),
                  by_input_style={style: summarize([r for r in rows if r.get('metadata', {}).get('input_style', 'unknown') == style])
                                  for style in sorted({r.get('metadata', {}).get('input_style', 'unknown') for r in rows})})
    original_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest = dict(protocol=VERSION, mode='coverage-rescore', key_attributes=sorted(KEY_ATTRIBUTES),
                    source_directory=str(source), original_manifest=original_manifest,
                    source_sha256={name: sha256(source/name) for name in ['samples.jsonl', 'summary.json', 'manifest.json']},
                    scoring_code_sha256={name: sha256(Path(__file__).parent/name) for name in ['metrics.py', 'rescore.py']},
                    preserved_metrics='All non-coverage fields copied unchanged; no generation or tokenization')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (destination/'samples.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    (destination/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    """命令行接收旧结果目录与新目录，错误时退出，不覆盖旧结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = rescore_directory(args.source, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Rescore failed: {exc}\n')
    print(f"{VERSION}: {report['samples']} samples -> {args.output_dir.resolve()}")


if __name__ == '__main__':
    main()
