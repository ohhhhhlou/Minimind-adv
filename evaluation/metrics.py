###指标逻辑
"""Versioned literal metrics. No semantic matching or external dependencies."""
import json
import re

VERSION = 'advertisegen-literal-v2'
KEY_ATTRIBUTES = frozenset(('类型', '颜色', '材质', '面料', '版型', '长度', '衣长', '裙长', '裤长', '袖长'))


def parse_attributes(prompt):
    """将用户输入解析为按原顺序排列的 (属性名, 属性值) 列表。

    支持“类型#裙*版型#宽松”和本数据集的“类型为裙，版型为宽松”模板。
    自然语言模板中的顿号表示同名多值，拆开后逐个计分；结构化值保持原样。
    不猜测“和”、斜杠等分隔方式。去除属性边缘空白、去重完全相同的属性对。
    字段不完整或格式不受支持时抛出 ValueError，避免漏解析造成覆盖率虚高。
    这里只提取原文，不推断同义词或其他语义关系。
    """
    text = prompt.strip()
    if '#' in text:
        text = re.sub(r'^商品属性\s*[:：]\s*', '', text)
        parts = text.split('*')
        pairs = []
        for part in parts:
            if part.count('#') != 1:
                raise ValueError(f'Invalid structured attribute: {part!r}')
            pairs.append(tuple(x.strip() for x in part.split('#')))
    else:
        text = re.sub(r'^请根据以下商品信息生成一段中文电商广告文案\s*[:：]\s*', '', text)
        text = text.rstrip('。.!！')
        pairs = []
        for part in re.split('[，,；;]', text):
            match = re.fullmatch(r'\s*([^为:：]+?)\s*(?:为|:|：)\s*(.+?)\s*', part)
            if not match:
                raise ValueError(f'Unsupported natural attribute: {part!r}')
            key, value = match.groups()
            pairs.extend((key, item.strip()) for item in value.split('、'))
    if not pairs or any(not k or not v for k, v in pairs):
        raise ValueError('Empty attributes')
    return list(dict.fromkeys(pairs))


def read_rows(path, limit=0):
    """从 Path 对象指定的 JSONL 文件读取并校验测评样本，返回字典列表。

    limit=0 表示全量；正数表示按文件顺序取前 limit 条，不随机抽样。
    每条必须包含非空的 system/user/assistant 三条消息，且顺序一致。
    返回的样本增加 attributes（用户输入的属性）和 line_number（原文件行号），
    不改写源文件。空数据、负数 limit 或非法样本会报错；样本错误附带行号。
    """
    if limit < 0:
        raise ValueError('limit must be >= 0')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if limit and len(rows) >= limit:
                break
            try:
                row = json.loads(line)
                messages = row['conversations']
                if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
                    raise ValueError('Expected system/user/assistant')
                if any(not isinstance(m['content'], str) or not m['content'].strip() for m in messages):
                    raise ValueError('Empty/non-text message')
                row['attributes'] = parse_attributes(messages[1]['content'])
                row['line_number'] = line_no
                rows.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
    if not rows:
        raise ValueError('Empty evaluation set')
    return rows


def repetition_ratio(sequence, n=4):
    """计算字符序列或 token ID 序列中重复 n-gram 窗口所占比例。

    n 应为正整数，默认取连续 4 个元素；窗口允许重叠。
    公式为 (窗口总数 - 不同窗口数) / 窗口总数，首次出现不算重复。
    例如 'aaaaa' 有两个 'aaaa' 窗口，重复率为 1/2；不足 n 个元素时为 0。
    本函数不清理空白，字符评分所需的清理由调用方完成。
    """
    grams = [tuple(sequence[i:i+n]) for i in range(max(0, len(sequence)-n+1))]
    return (len(grams)-len(set(grams))) / len(grams) if grams else 0.0


def is_loop(text):
    """判断文案是否出现连续循环，返回布尔值。

    去除所有空白后，检查任意位置是否有长度为 2～32 字符的片段连续出现
    至少三次，例如“好看好看好看”。这是固定规则，不代表语义上的重复判断。
    """
    text = re.sub(r'\s+', '', text)
    return any(text[i:i+n] == text[i+n:i+2*n] == text[i+2*n:i+3*n]
               for n in range(2, min(32, len(text)//3)+1)
               for i in range(len(text)-3*n+1))


def score(text, attributes, token_ids=None, eos=None):
    """为一条生成文案或参考文案计算指标，返回指标字典。

    text 是待评分原文，attributes 是 (属性名, 属性值) 列表。
    覆盖率按属性值是否为原文子串计算；关键覆盖仅统计 KEY_ATTRIBUTES 中
    的属性名。不识别同义词和否定语气，同时返回命中数、总数及遗漏属性。
    没有属性或没有关键属性时，对应覆盖率为 None，而不是默认满分。

    字符 4-gram 重复率和循环检测忽略空白，字符长度则包含原文空白。
    token_ids 由调用方提供：生成结果应去掉末尾 EOS，参考答案应无特殊 token；
    未提供时，token 长度和 token 重复率为 None。
    eos 是调用方根据实际生成 token 判断的结束状态，本函数仅记录它；
    参考答案无生成过程，应保持 None。这里不会加载 tokenizer 或调用模型。
    """
    hits = [value in text for _, value in attributes]
    key_hits = [hit for (key, _), hit in zip(attributes, hits) if key in KEY_ATTRIBUTES]
    chars = re.sub(r'\s+', '', text)
    return dict(attribute_hits=sum(hits), attribute_count=len(hits),
                key_attribute_hits=sum(key_hits), key_attribute_count=len(key_hits),
                literal_coverage=sum(hits)/len(hits) if hits else None,
                key_literal_coverage=sum(key_hits)/len(key_hits) if key_hits else None,
                all_attributes_covered=all(hits) if hits else None,
                repeated_char_4gram_ratio=repetition_ratio(chars),
                repeated_token_4gram_ratio=repetition_ratio(token_ids) if token_ids is not None else None,
                loop=is_loop(text), eos_normal=eos, char_length=len(text),
                token_length=len(token_ids) if token_ids is not None else None,
                missing_attributes=[list(pair) for pair, hit in zip(attributes, hits) if not hit])


def aggregate(scores):
    """汇总同一组样本的 score() 结果，返回可写入 JSON 的统计字典。

    各数值指标记录 mean（逐样本平均）及 n（非 None 的有效样本数）；
    布尔值按 0/1 求平均，因此 loop、eos_normal 等得到对应样本率。
    覆盖率的 mean 是宏平均；另算 micro：总命中属性数 / 总属性数。
    无有效值或分母为零时为 None。遗漏属性列表只保留在逐条结果中，不求平均。
    输入为空时返回 samples=0 及两个值为 None 的 micro 覆盖率。
    """
    result = {'samples': len(scores)}
    for key in scores[0] if scores else []:
        if key == 'missing_attributes':
            continue
        values = [s[key] for s in scores if s[key] is not None]
        result[key] = {'mean': sum(values)/len(values) if values else None, 'n': len(values)}
    for prefix in ('', 'key_'):
        total = sum(s[prefix+'attribute_count'] for s in scores)
        result[prefix+'literal_coverage_micro'] = sum(s[prefix+'attribute_hits'] for s in scores)/total if total else None
    return result
