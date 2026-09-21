"""只用标准库读取测评结果，不加载模型、不修改结果文件。"""
import argparse
import json
from itertools import islice
from pathlib import Path


METRICS = [
    ('literal_coverage', '整体精确覆盖率（宏平均）', True),
    ('key_literal_coverage', '关键精确覆盖率（宏平均）', True),
    ('literal_coverage_micro', '整体精确覆盖率（微平均）', True),
    ('key_literal_coverage_micro', '关键精确覆盖率（微平均）', True),
    ('all_attributes_covered', '全属性覆盖样本率', True),
    ('repeated_char_4gram_ratio', '字符 4-gram 重复率', True),
    ('repeated_token_4gram_ratio', 'token 4-gram 重复率', True),
    ('loop', '循环样本率', True),
    ('eos_normal', 'EOS 正常结束率', True),
    ('char_length', '平均字符长度', False),
    ('token_length', '平均 token 长度', False),
]


def format_metric(value, percentage):
    """格式化均值或微平均；保留有效分母，缺失值显示不适用。"""
    suffix = ''
    if isinstance(value, dict):
        suffix = f" (n={value.get('n', '?')})"
        value = value.get('mean')
    if value is None:
        return '不适用/未提供' + suffix
    return (f'{value:.2%}' if percentage else f'{value:.2f}') + suffix


def print_group(title, group):
    """并列打印一组模型与参考答案指标，支持仅参考答案的报告。"""
    print(f'\n【{title}】')
    output = group.get('output_metrics', {})
    reference = group.get('reference_metrics', {})
    print('指标 | 模型输出 | 参考答案')
    for key, label, percentage in METRICS:
        print(f'{label} | {format_metric(output.get(key), percentage)} | '
              f'{format_metric(reference.get(key), percentage)}')


def main():
    """读取指定结果目录或 summary.json，展示汇总及可选的前 N 条样本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='结果目录或 summary.json 路径')
    parser.add_argument('--samples', type=int, default=0, help='展示前 N 条文案，默认不展示')
    parser.add_argument('--by-style', action='store_true', help='同时展示两种输入格式的分组指标')
    args = parser.parse_args()
    if args.samples < 0:
        parser.error('--samples 必须 >= 0')
    path = args.result / 'summary.json' if args.result.is_dir() else args.result
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        print(f'结果文件: {path.resolve()}')
        print(f"状态: {report.get('status', '未知')}；样本数: {report.get('samples', '未知')}")
        if report.get('status') != 'complete':
            print('注意：该报告没有标记为完整完成。')
        print_group('整体', report['overall'])
        if args.by_style:
            for style, group in report.get('by_input_style', {}).items():
                print_group(f'输入格式: {style}', group)
        print('\n注：覆盖率是字面匹配；参考答案的 EOS 不适用。n 是有效样本数。')
        if args.samples:
            with (path.parent / 'samples.jsonl').open(encoding='utf-8') as handle:
                for line in islice(handle, args.samples):
                    row = json.loads(line)
                    print(f"\n【样本 {row.get('line_number', '?')}】")
                    for message in row.get('messages', []):
                        if message.get('role') == 'user':
                            print('输入:', message['content'])
                    print('生成:', row.get('output', '未生成（仅参考答案测评）'))
                    print('参考:', row.get('reference', '未提供'))
                    print('模型遗漏属性:', row.get('output_metrics', {}).get('missing_attributes', '不适用'))
                    print('结束原因:', row.get('generation', {}).get('finish_reason', '不适用'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'无法读取结果：{exc}\n请确认测评已完成，且路径指向结果目录或 summary.json。\n')


if __name__ == '__main__':
    main()
