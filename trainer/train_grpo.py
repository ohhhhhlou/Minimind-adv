import os
import sys
import re
import gc
import argparse
import warnings
import math
import json
import importlib.util
import subprocess
import torch
import torch.distributed as dist
import atexit
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoConfig, AutoModel

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)
from evaluation.metrics import parse_attributes, score as score_advertise

warnings.filterwarnings("ignore")


def load_reward_components(model_path, device):
    """加载并验证 InternLM2 reward model，尽量在进入 CUDA 前暴露配置问题。"""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"奖励模型目录不存在：{model_path}")

    tokenizer_model = os.path.join(model_path, "tokenizer.model")
    if not os.path.isfile(tokenizer_model):
        raise FileNotFoundError(
            f"缺少奖励模型原生分词器文件：{tokenizer_model}"
        )

    reward_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # InternLM2 旧版远程代码读取 rope_scaling['type']；新版配置可能写 rope_type。
    if isinstance(getattr(reward_config, "rope_scaling", None), dict):
        rope_type = reward_config.rope_scaling.get(
            "type", reward_config.rope_scaling.get("rope_type", "default")
        )
        if rope_type == "default":
            reward_config.rope_scaling = None
        else:
            reward_config.rope_scaling["type"] = rope_type
            reward_config.rope_scaling.setdefault("factor", 1.0)

    # Transformers 4.56.2 没有内置 InternLM2Tokenizer，因此直接加载奖励
    # 模型目录自带的慢分词器源码。这样只读取 tokenizer.model，不会触发
    # tokenizer.json 的快速分词器转换和特殊 token 重复追加。
    tokenizer_code = os.path.join(model_path, "tokenization_internlm2.py")
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_code):
        raise FileNotFoundError(f"缺少奖励模型分词器源码：{tokenizer_code}")
    if not os.path.isfile(tokenizer_config_path):
        raise FileNotFoundError(f"缺少奖励模型分词器配置：{tokenizer_config_path}")

    spec = importlib.util.spec_from_file_location(
        "internlm2_reward_tokenizer_slow", tokenizer_code
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载奖励模型分词器源码：{tokenizer_code}")
    tokenizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizer_module)

    with open(tokenizer_config_path, "r", encoding="utf-8") as file:
        tokenizer_config = json.load(file)

    reward_tokenizer = tokenizer_module.InternLM2Tokenizer(
        vocab_file=tokenizer_model,
        unk_token=tokenizer_config.get("unk_token", "<unk>"),
        bos_token=tokenizer_config.get("bos_token", "<s>"),
        eos_token=tokenizer_config.get("eos_token", "</s>"),
        pad_token=tokenizer_config.get("pad_token", "</s>"),
        add_bos_token=tokenizer_config.get("add_bos_token", True),
        add_eos_token=tokenizer_config.get("add_eos_token", False),
        chat_template=tokenizer_config.get("chat_template"),
    )
    if isinstance(reward_tokenizer, bool) or not hasattr(
        reward_tokenizer, "apply_chat_template"
    ):
        raise TypeError(
            "奖励分词器加载失败："
            f"类型={type(reward_tokenizer)}，值={reward_tokenizer!r}"
        )

    reward_model = AutoModel.from_pretrained(
        model_path,
        config=reward_config,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    embedding = reward_model.get_input_embeddings()
    embedding_size = embedding.num_embeddings
    tokenizer_size = len(reward_tokenizer)
    reward_token_id = getattr(reward_model, "reward_token_id", None)
    if reward_token_id is None:
        reward_token_id = getattr(reward_config, "reward_token_id", None)

    # tokenizer 可以比 embedding 多出未使用的附加 token，但实际输入 ID 绝不能越界。
    probe_chat = [
        {"role": "user", "content": "测试"},
        {"role": "assistant", "content": "正常"},
    ]
    probe_text = reward_tokenizer.apply_chat_template(
        probe_chat, tokenize=False, add_generation_prompt=False
    )
    probe_ids = reward_tokenizer.encode(probe_text, add_special_tokens=False)
    ids_to_check = list(probe_ids)
    if reward_token_id is not None:
        ids_to_check.append(int(reward_token_id))
    if not ids_to_check or min(ids_to_check) < 0 or max(ids_to_check) >= embedding_size:
        raise ValueError(
            "奖励模型与分词器不匹配，token ID 超出 embedding 范围："
            f"ID范围={min(ids_to_check) if ids_to_check else None}.."
            f"{max(ids_to_check) if ids_to_check else None}，"
            f"embedding_size={embedding_size}，tokenizer_size={tokenizer_size}，"
            f"reward_token_id={reward_token_id}"
        )

    Logger(
        "奖励模型检查通过："
        f"embedding_size={embedding_size}, tokenizer_size={tokenizer_size}, "
        f"reward_token_id={reward_token_id}"
    )
    return reward_model.to(device).eval().requires_grad_(False), reward_tokenizer


def validate_token_ids(name, input_ids, vocab_size):
    """在送入 CUDA embedding/gather 前检查 token ID，给出同步且易读的错误。"""
    if input_ids.numel() == 0:
        raise ValueError(f"{name} 为空，无法训练")
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"{name} token越界：范围={min_id}..{max_id}，模型词表大小={vocab_size}"
        )


def get_checked_reward_score(reward_model, reward_tokenizer, conversation):
    """先在 CPU 检查奖励模型输入 ID，再调用模型评分。"""
    text = reward_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    input_ids = reward_tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    )
    reward_token_id = getattr(reward_model, "reward_token_id", None)
    if reward_token_id is not None and input_ids[0, -1].item() != reward_token_id:
        input_ids = torch.cat(
            [input_ids, torch.tensor([[reward_token_id]], dtype=torch.long)], dim=1
        )
    validate_token_ids(
        "奖励模型输入",
        input_ids,
        reward_model.get_input_embeddings().num_embeddings,
    )
    return reward_model.get_score(reward_tokenizer, conversation)

class RewardServiceClient:
    def __init__(self, python_bin, model_path, device):
        self.proc = subprocess.Popen(
            [
                python_bin,
                "tools/reward_service.py",
                "--model-path",
                model_path,
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        atexit.register(self.close)
    def score(self, conversation):
        self.proc.stdin.write(json.dumps({'conversation': conversation}, ensure_ascii=False)+'\n'); self.proc.stdin.flush()
        result=json.loads(self.proc.stdout.readline())
        if 'error' in result: raise RuntimeError(result['error'])
        return float(result['score'])
    def close(self):
        if self.proc.poll() is None: self.proc.terminate()


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    def reasoning_model_reward(rewards_tensor):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        format_rewards = []
        for response in responses:
            matched = re.match(pattern, response, re.S) or re.match(
                pattern2, response, re.S
            )
            format_rewards.append(0.5 if matched else 0.0)
        rewards_tensor += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0.0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        rewards_tensor += torch.tensor(
            [mark_num(response) for response in responses], device=args.device
        )
        return rewards_tensor

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # Deterministic rule reward for AdvertiseGen.  The same versioned metrics
    # are used by tools/evaluate_grpo_candidates.py.
    rule_scores = []
    for i, prompt in enumerate(prompts):
        matches = re.findall(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", prompt, re.DOTALL)
        user_text = next((content.strip() for role, content in matches if role == "user"), "")
        try:
            attributes = parse_attributes(user_text)
        except ValueError:
            attributes = []
        for j in range(args.num_generations):
            response = responses[i * args.num_generations + j]
            m = score_advertise(response, attributes)
            coverage = m["literal_coverage"] or 0.0
            repeat = m["repeated_char_4gram_ratio"]
            length_penalty = 1.0 if len(response) < 20 or len(response) > 520 else 0.0
            rule_scores.append(
                args.coverage_weight * coverage
                - args.repeat_penalty * repeat
                - args.loop_penalty * float(m["loop"])
                - args.length_penalty * length_penalty
            )
    rewards += torch.tensor(rule_scores, device=args.device, dtype=torch.float32)

    if args.reward_model_weight == 0.0:
        return rewards

    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, prompt in enumerate(prompts):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]

            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.score(tmp_chat) if isinstance(reward_model, RewardServiceClient) else get_checked_reward_score(reward_model, reward_tokenizer, tmp_chat)
                if isinstance(score, torch.Tensor):
                    score = score.detach().float().cpu().item()
                if not math.isfinite(float(score)):
                    raise ValueError(f"奖励模型返回非有限值：{score}")
                score = max(min(score, scale), -scale)

                if args.reasoning == 1:
                    answer_match = re.search(
                        r"<answer>(.*?)</answer>", response, re.DOTALL
                    )
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        answer_chat = messages + [
                            {"role": "assistant", "content": answer_content}
                        ]
                        answer_score = reward_model.score(answer_chat) if isinstance(reward_model, RewardServiceClient) else get_checked_reward_score(reward_model, reward_tokenizer, answer_chat)
                        if isinstance(answer_score, torch.Tensor):
                            answer_score = answer_score.detach().float().cpu().item()
                        if not math.isfinite(float(answer_score)):
                            raise ValueError(f"奖励模型返回非有限值：{answer_score}")
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # The external RM is an optional ablation.  Keep its raw scores out of
        # the first rule-reward run unless explicitly enabled by the CLI.
        rewards += args.reward_model_weight * torch.tensor(
            reward_model_scores, device=args.device
        )

    return rewards


def grpo_train_epoch(
    epoch,
    loader,
    iters,
    ref_model,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(args.device)

        validate_token_ids(
            "策略模型prompt",
            prompt_inputs["input_ids"],
            model.config.vocab_size,
        )

        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -args.max_seq_len :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -args.max_seq_len :
            ]

        with torch.no_grad():
            model_for_gen = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]
        if completion_ids.size(1) == 0:
            raise RuntimeError("模型没有生成任何新token，请检查max_gen_len和EOS设置")
        validate_token_ids("策略模型生成结果", outputs, model.config.vocab_size)

        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = (
                input_ids.detach().clone() if input_ids.is_inference() else input_ids
            )
            logits = mdl(input_ids=input_ids, logits_to_keep=n_keep + 1).logits[
                :, :-1, :
            ]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = (
                    ids_row.detach().clone() if ids_row.is_inference() else ids_row
                )
                token_logps = torch.gather(
                    logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_logps)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model, outputs, completion_ids.size(1)
            )

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(
            prompts, completions, reward_model, reward_tokenizer
        ).to(args.device)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(
            args.num_generations
        )
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)
            <= eos_idx.unsqueeze(1)
        ).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach())
            * advantages.unsqueeze(1)
            - args.beta * per_token_kl
        )

        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean() / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}"
            )

            if wandb and is_main_process():
                wandb.log(
                    {
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": advantages.mean().item(),
                        "learning_rate": current_lr,
                    }
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = (
                model.module.state_dict()
                if isinstance(model, DistributedDataParallel)
                else model.state_dict()
            )
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=scheduler,
            )
            model.train()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del (
            completions,
            rewards,
            grouped_rewards,
            mean_r,
            std_r,
            advantages,
            completion_mask,
        )
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MokioMind GRPO (Group Relative Policy Optimization)"
    )

    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--model_dir", type=str, default="out", help="初始模型权重目录")
    parser.add_argument("--init_weight", type=str, default="full_sft", help="初始权重前缀，不含隐藏维度和.pth")
    parser.add_argument(
        "--save_weight", default="grpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    parser.add_argument("--max_seq_len", default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=96, help="生成的最大长度")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/rlaif-mini.jsonl",
        help="RLAIF数据路径",
    )
    parser.add_argument(
        "--num_generations", type=int, default=8, help="每个prompt生成的样本数"
    )
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument(
        "--reasoning",
        type=int,
        default=0,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型）",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="../../internlm2-1_8b-reward",
        help="Reward模型路径",
    )
    parser.add_argument("--reward_model_weight", type=float, default=0.0)
    parser.add_argument("--reward_python", type=str, default=".venv-eval/bin/python")
    parser.add_argument("--coverage_weight", type=float, default=1.0)
    parser.add_argument("--repeat_penalty", type=float, default=0.5)
    parser.add_argument("--loop_penalty", type=float, default=0.5)
    parser.add_argument("--length_penalty", type=float, default=0.2)
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-GRPO", help="wandb项目名"
    )
    args = parser.parse_args()

    if args.num_generations < 2:
        parser.error("--num_generations 至少为2，否则GRPO无法计算组内相对优势")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("--batch_size 和 --accumulation_steps 必须大于0")
    if args.max_seq_len < 1 or args.max_gen_len < 1:
        parser.error("--max_seq_len 和 --max_gen_len 必须大于0")
    if not os.path.isfile(args.data_path):
        parser.error(f"训练数据不存在：{args.data_path}")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MokioMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    base_weight = args.init_weight

    model, tokenizer = init_model(
        lm_config, base_weight, save_dir=args.model_dir, device=args.device
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model, _ = init_model(
        lm_config, base_weight, save_dir=args.model_dir, device=args.device
    )
    ref_model = ref_model.eval().requires_grad_(False)

    if args.reward_model_weight != 0.0:
        reward_model = RewardServiceClient(args.reward_python, args.reward_model_path, args.device)
        reward_tokenizer = None
    else:
        reward_model, reward_tokenizer = None, None

    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=lm_config.max_position_embeddings
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = max(
        1, math.ceil(iters / args.accumulation_steps) * args.epochs
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                sampler=train_sampler,
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader),
                ref_model,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )
