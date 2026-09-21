"""全量只读检查属性解析和同源双格式的一致性，向终端输出 JSON 报告。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from evaluation.metrics import VERSION, read_rows
from evaluation.run import sha256


def audit(path):
    """按 source/source_index 配对，报告多值规模、结构化标点及不一致行号。"""
    rows = read_rows(path)
    groups = defaultdict(lambda: defaultdict(list))
    punctuation = Counter()
    multi = 0
    for row in rows:
        meta = row['metadata']
        style = meta['input_style']
        groups[(meta['source'], meta['source_index'])][style].append(row)
        if style == 'natural' and '、' in row['conversations'][1]['content']:
            multi += 1
        if style == 'structured':
            for _, value in row['attributes']:
                punctuation.update(c for c in '、，,；;/' if c in value)
    pairs, mismatches, ambiguous, unpaired = 0, [], [], 0
    for group in groups.values():
        if any(len(items) != 1 for items in group.values()):
            ambiguous.append([r['line_number'] for items in group.values() for r in items])
        elif 'structured' in group and 'natural' in group:
            pairs += 1
            left, right = group['structured'][0], group['natural'][0]
            if set(left['attributes']) != set(right['attributes']):
                mismatches.append([left['line_number'], right['line_number']])
        else:
            unpaired += 1
    return dict(path=str(path.resolve()), sha256=sha256(path), rows=len(rows), paired_groups=pairs,
                unpaired_groups=unpaired, natural_multivalue_rows=multi,
                structured_value_punctuation=dict(punctuation), mismatches=mismatches,
                ambiguous_groups=ambiguous)


def main():
    """检查默认 train/val/test；发现同源歧义或配对差异时返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1]/'dataset/advertisegen')
    args = parser.parse_args()
    reports = {split: audit(args.data_dir/f'advertisegen_sft_{split}.jsonl') for split in ['train','val','test']}
    print(json.dumps(dict(protocol=VERSION, splits=reports), ensure_ascii=False, indent=2))
    if any(r['mismatches'] or r['ambiguous_groups'] for r in reports.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
