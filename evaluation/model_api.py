###模型适配器
"""EvalScope 1.8 ModelAPI for native MokioMind state dictionaries."""
import torch
from transformers import AutoTokenizer, GenerationConfig
from evalscope.api.model import ModelAPI, ModelOutput, GenerateConfig
from evalscope.api.registry import register_model_api
from model.MokioModel import MokioMindConfig, MokioMindForCausalLM
from evaluation.generation import generation_kwargs, SEED


@register_model_api(name='mokiomind_native')
class MokioMindAPI(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None, config=None, **kwargs):
        super().__init__(model_name, base_url, api_key, config or GenerateConfig())
        self.device = kwargs.get('device', 'cuda:0')
        protocol = generation_kwargs(kwargs.get('diagnostic', 'none'))
        if 'max_new_tokens' in kwargs and kwargs['max_new_tokens'] != protocol['max_new_tokens']:
            raise ValueError('max_new_tokens differs from selected diagnostic')
        self.max_new_tokens = protocol['max_new_tokens']
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_path'], local_files_only=True)
        self.model = MokioMindForCausalLM(MokioMindConfig(
            hidden_size=kwargs.get('hidden_size', 512),
            num_hidden_layers=kwargs.get('num_hidden_layers', 8),
            use_moe=kwargs.get('use_moe', False), flash_attention=False))
        weights = torch.load(model_name, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().to(device=self.device, dtype=torch.float32)
        self.generation = GenerationConfig(
            **protocol, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id)
        self.last_result = None  # Runner is deliberately serial.

    def generate(self, input, tools, tool_choice, config):
        if tools:
            raise ValueError('Tools are outside this benchmark')
        if config.max_tokens is not None and config.max_tokens != self.max_new_tokens:
            raise ValueError('Generation configuration differs from frozen protocol')
        messages = [{'role': m.role, 'content': m.content} for m in input]
        if any(not isinstance(m['content'], str) for m in messages):
            raise ValueError('Only text messages are supported')
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False,
                                return_token_type_ids=False).to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        if prompt_length + self.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Context overflow; truncation is forbidden')
        # 每次请求重置同一种子，使前20/前200/全量中的同一样本不依赖前序输出长度。
        torch.manual_seed(SEED)
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation)
        raw_ids = output[0, prompt_length:].tolist()
        eos = bool(raw_ids and raw_ids[-1] == self.tokenizer.eos_token_id)
        content_ids = raw_ids[:-1] if eos else raw_ids
        text = self.tokenizer.decode(content_ids, skip_special_tokens=True)
        self.last_result = dict(raw_token_ids=raw_ids, content_token_ids=content_ids,
                                eos_normal=eos, finish_reason='eos' if eos else 'length',
                                prompt_tokens=prompt_length, rendered_prompt=prompt, seed=SEED)
        return ModelOutput.from_content(model=self.model_name, content=text)
