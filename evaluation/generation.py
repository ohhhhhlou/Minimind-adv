"""唯一生成协议；纯标准库配置，供适配器、命令行和测试共同使用。"""
PROFILE = 'standard'
GENERATION_VERSION = 'standard-sampling-v1'
SEED = 2026
DIAGNOSTICS = ('none', 'length256', 'greedy', 'no_penalty', 'no_ngram')


def generation_kwargs(diagnostic='none'):
    """返回独立配置副本；top_k=0 明确关闭 Transformers 默认的 top-k 截断。"""
    if diagnostic not in DIAGNOSTICS:
        raise ValueError(f'Unknown diagnostic: {diagnostic}')
    config = dict(do_sample=False, temperature=0.75, top_p=0.90, top_k=0,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                max_new_tokens=96, num_beams=1, num_return_sequences=1, use_cache=True)
    changes = {'none': {}, 'length256': {'max_new_tokens': 256},
               'greedy': {'do_sample': False}, 'no_penalty': {'repetition_penalty': 1.0},
               'no_ngram': {'no_repeat_ngram_size': 0}}
    config.update(changes[diagnostic])
    return config
