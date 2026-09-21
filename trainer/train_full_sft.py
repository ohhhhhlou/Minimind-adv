import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.lm_dataset import SFTDataset
from model.MokioModel import MokioMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def assert_finite(value: torch.Tensor, name: str, epoch: int, step: int) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} 出现 NaN/Inf: epoch={epoch + 1}, step={step}"
        )


def save_training_state(weight_name: str, epoch: int, step: int, wandb=None) -> Path:
    raw_model = unwrap_model(model)
    output_path = args.save_dir / (
        f"{weight_name}_{lm_config.hidden_size}"
        f"{'_moe' if lm_config.use_moe else ''}.pth"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {key: value.detach().half().cpu() for key, value in raw_model.state_dict().items()},
        tmp_path,
    )
    os.replace(tmp_path, output_path)

    if weight_name == args.save_weight:
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=raw_model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            wandb=wandb,
            save_dir=str(args.checkpoint_dir),
            scaler=scaler,
            optimizer_step=optimizer_step,
            best_val_loss=best_val_loss,
        )
    Logger(f"权重已保存: {output_path}")
    return output_path


@torch.no_grad()
def evaluate(loader, max_batches: int = 0) -> float:
    raw_model = unwrap_model(model)
    raw_model.eval()
    loss_sum = torch.zeros(1, device=args.device, dtype=torch.float64)
    batch_count = torch.zeros(1, device=args.device, dtype=torch.float64)

    for batch_index, (input_ids, labels, attention_mask) in enumerate(loader, 1):
        if max_batches > 0 and batch_index > max_batches:
            break
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        attention_mask = attention_mask.to(args.device, non_blocking=True)
        with autocast_ctx:
            result = raw_model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = result.loss + result.aux_loss
        assert_finite(loss, "validation loss", current_epoch, current_step)
        loss_sum += loss.detach().double()
        batch_count += 1

    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    raw_model.train()
    if batch_count.item() == 0:
        raise RuntimeError("验证集为空，或 --val_batches 设置导致没有验证 batch")
    return (loss_sum / batch_count).item()


def train_epoch(epoch, loader, full_epoch_iters, start_step=0, wandb=None):
    global current_epoch, current_step, optimizer_step
    current_epoch = epoch
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0

    for local_index, (input_ids, labels, attention_mask) in enumerate(loader, 1):
        step = start_step + local_index
        current_step = step
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        attention_mask = attention_mask.to(args.device, non_blocking=True)

        global_micro_step = epoch * full_epoch_iters + step
        total_micro_steps = max(1, args.epochs * full_epoch_iters)
        lr = get_lr(global_micro_step, total_micro_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            result = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            total_loss = result.loss + result.aux_loss
            assert_finite(total_loss, "training loss", epoch, step)
            scaled_loss = total_loss / args.accumulation_steps
        scaler.scale(scaled_loss).backward()
        accumulated += 1

        is_last_batch = local_index == len(loader)
        should_update = accumulated == args.accumulation_steps or is_last_batch
        if should_update:
            if accumulated < args.accumulation_steps:
                correction = args.accumulation_steps / accumulated
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            assert_finite(grad_norm, "gradient norm", epoch, step)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            optimizer_step += 1

            if optimizer_step % args.log_interval == 0 or optimizer_step == 1:
                elapsed = time.time() - start_time
                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}] batch:{step}/{full_epoch_iters}, "
                    f"optimizer_step:{optimizer_step}, loss:{total_loss.item():.4f}, "
                    f"grad_norm:{grad_norm.item():.4f}, lr:{lr:.8f}, "
                    f"elapsed:{elapsed / 60:.1f}min"
                )
                if wandb and is_main_process():
                    wandb.log(
                        {
                            "train/loss": total_loss.item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/learning_rate": lr,
                            "train/optimizer_step": optimizer_step,
                        }
                    )

            if args.save_interval > 0 and optimizer_step % args.save_interval == 0:
                if is_main_process():
                    save_training_state(args.save_weight, epoch, step, wandb)

            if args.val_interval > 0 and optimizer_step % args.val_interval == 0:
                run_validation(epoch, step, wandb)

            if args.max_steps > 0 and optimizer_step >= args.max_steps:
                return True

        del input_ids, labels, attention_mask, result, total_loss, scaled_loss
    return False


def run_validation(epoch: int, step: int, wandb=None) -> float | None:
    global best_val_loss
    if val_loader is None:
        return None
    val_loss = evaluate(val_loader, args.val_batches)
    Logger(f"Validation: optimizer_step:{optimizer_step}, loss:{val_loss:.4f}")
    if wandb and is_main_process():
        wandb.log({"val/loss": val_loss, "train/optimizer_step": optimizer_step})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if is_main_process():
            save_training_state(f"{args.save_weight}_best", epoch, step, wandb)
    return val_loss


@torch.no_grad()
def reload_and_generate(checkpoint_weight: str, sample_count: int) -> None:
    if sample_count <= 0 or val_ds is None or not is_main_process():
        return
    Logger(f"重新加载 {checkpoint_weight} checkpoint 并执行生成检查")
    fresh_model, fresh_tokenizer = init_model(
        lm_config,
        checkpoint_weight,
        save_dir=str(args.save_dir),
        device=args.device,
    )
    fresh_model.eval()
    for index in range(min(sample_count, len(val_ds))):
        row = val_ds.samples[index]
        messages = list(row["conversations"][:-1])
        prompt = fresh_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = fresh_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(args.device)
        outputs = fresh_model.generate(
            **inputs,
            max_new_tokens=args.smoke_max_new_tokens,
            do_sample=False,
            pad_token_id=fresh_tokenizer.pad_token_id,
            eos_token_id=fresh_tokenizer.eos_token_id,
        )
        completion = fresh_tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        Logger(f"[生成检查 {index + 1}] 输入: {messages[-1]['content']}")
        Logger(f"[生成检查 {index + 1}] 输出: {completion}")
    del fresh_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="MokioMind Full SFT")
    parser.add_argument("--save_dir", default="out", help="推理权重目录（相对项目根目录）")
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", help="断点目录（相对项目根目录）"
    )
    parser.add_argument("--save_weight", default="full_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=0, help="0 表示每个 epoch 验证")
    parser.add_argument("--val_batches", type=int, default=0, help="0 表示验证完整验证集")
    parser.add_argument("--max_steps", type=int, default=0, help="优化器步数上限；0 表示不限")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--data_path", required=True, help="训练 JSONL")
    parser.add_argument("--val_data_path", help="验证 JSONL；正式训练必须提供")
    parser.add_argument("--from_weight", default="pretrain")
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_generate_samples", type=int, default=0)
    parser.add_argument("--smoke_max_new_tokens", type=int, default=128)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MokioMind-Full-SFT")
    parser.add_argument("--use_compile", type=int, choices=[0, 1], default=0)
    parsed = parser.parse_args()
    if parsed.accumulation_steps < 1 or parsed.batch_size < 1:
        parser.error("--batch_size 和 --accumulation_steps 必须大于 0")
    if parsed.max_steps < 0 or parsed.val_batches < 0:
        parser.error("--max_steps 和 --val_batches 不能小于 0")
    if parsed.smoke_generate_samples > 0 and not parsed.val_data_path:
        parser.error("生成检查需要提供 --val_data_path")
    return parsed


if __name__ == "__main__":
    args = parse_args()
    args.save_dir = resolve_path(args.save_dir)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir)
    args.data_path = resolve_path(args.data_path)
    args.val_data_path = resolve_path(args.val_data_path) if args.val_data_path else None
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {args.device}，但 CUDA 不可用")
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 bfloat16，请改用 --dtype float16")
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=str(args.checkpoint_dir))
        if args.from_resume
        else None
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"{args.save_weight}-ep{args.epochs}-bs{args.batch_size}-lr{args.learning_rate}",
            id=wandb_id,
            resume="must" if wandb_id else None,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        save_dir=str(args.save_dir),
        device=args.device,
    )
    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    train_ds = SFTDataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)
    val_ds = (
        SFTDataset(str(args.val_data_path), tokenizer, max_length=args.max_seq_len)
        if args.val_data_path
        else None
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() and val_ds else None
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device_type == "cuda",
        )
        if val_ds
        else None
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, start_step, optimizer_step = 0, 0, 0
    best_val_loss = math.inf
    if ckp_data:
        unwrap_model(model).load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scaler" in ckp_data:
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        optimizer_step = ckp_data.get("optimizer_step", 0)
        best_val_loss = ckp_data.get("best_val_loss", math.inf)

    if dist.is_initialized():
        unwrap_model(model)._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    current_epoch, current_step = start_epoch, start_step
    stopped_early = False
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    full_epoch_iters = math.ceil(len(train_ds) / (args.batch_size * world_size))
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device_type == "cuda",
        )
        stopped_early = train_epoch(
            epoch, train_loader, full_epoch_iters, skip, wandb
        )
        run_validation(epoch, current_step, wandb)
        if is_main_process():
            save_training_state(args.save_weight, epoch, current_step, wandb)
        start_step = 0
        if stopped_early:
            break

    preferred_weight = (
        f"{args.save_weight}_best" if best_val_loss < math.inf else args.save_weight
    )
    reload_and_generate(preferred_weight, args.smoke_generate_samples)
    if dist.is_initialized():
        dist.destroy_process_group()
