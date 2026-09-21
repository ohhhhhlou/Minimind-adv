"""AdvertiseGen SFT two-step smoke-test launcher.

Upload this file together with ``train_full_sft.py`` and run it from the
AutoDL project with:

    uv run python trainer/run_new.py

Edit ``BASE_CONFIG`` or the selected entry in ``PROFILES`` when changing
settings. The three profiles keep smoke, benchmark, and full-run artifacts
separate so they cannot overwrite one another. Dataset and weight names can
also be supplied explicitly on the command line.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "trainer" / "train_full_sft.py"

BASE_CONFIG = {
    "data_path": "dataset/advertisegen/advertisegen_sft_train.jsonl",
    "val_data_path": "dataset/advertisegen/advertisegen_sft_val.jsonl",
    "from_weight": "pretrain",
    "save_dir": "out",
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "max_seq_len": 512,
    "epochs": 1,
    "batch_size": 8,
    "accumulation_steps": 1,
    "learning_rate": 1e-6,
    "dtype": "bfloat16",
    "num_workers": 2,
    "smoke_max_new_tokens": 96,
    "use_compile": 0,
}

PROFILES = {
    "smoke": {
        "checkpoint_dir": "checkpoints/sft_smoke",
        "save_weight": "advertise_sft_smoke_20260905",
        "max_steps": 2,
        "val_batches": 2,
        "val_interval": 0,
        "log_interval": 1,
        "save_interval": 0,
        "smoke_generate_samples": 2,
    },
    "benchmark": {
        "checkpoint_dir": "checkpoints/sft_benchmark",
        "save_weight": "advertise_sft_benchmark_50step",
        "max_steps": 50,
        "val_batches": 2,
        "val_interval": 0,
        "log_interval": 5,
        "save_interval": 0,
        "smoke_generate_samples": 2,
    },
    "train": {
        "checkpoint_dir": "checkpoints/advertise_sft_ep1",
        "save_weight": "advertise_sft_ep1",
        "max_steps": 0,
        "val_batches": 0,
        "val_interval": 0,
        "log_interval": 50,
        "save_interval": 500,
        "smoke_generate_samples": 5,
    },
}


def require_file(relative_path: str, description: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少{description}: {path}")
    return path


def build_command(config: dict[str, object]) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for name, value in config.items():
        command.extend([f"--{name}", str(value)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="AdvertiseGen SFT 启动器")
    parser.add_argument(
        "--mode",
        choices=sorted(PROFILES),
        default="smoke",
        help="smoke=2步全链路；benchmark=50步测速；train=正式1轮",
    )
    parser.add_argument("--data_path", help="覆盖所选模式的训练集路径")
    parser.add_argument("--val_data_path", help="覆盖所选模式的验证集路径")
    parser.add_argument("--from_weight", help="覆盖所选模式的起始权重名（不含 _512.pth）")
    parser.add_argument("--save_weight", help="覆盖所选模式的输出权重名（不含 _512.pth）")
    parser.add_argument("--checkpoint_dir", help="覆盖所选模式的断点目录")
    parser.add_argument("--learning_rate", type=float, help="覆盖所选模式的学习率")
    selected = parser.parse_args()
    config = {**BASE_CONFIG, **PROFILES[selected.mode]}
    for name in (
        "data_path",
        "val_data_path",
        "from_weight",
        "save_weight",
        "checkpoint_dir",
        "learning_rate",
    ):
        value = getattr(selected, name)
        if value is not None:
            config[name] = value

    require_file(config["data_path"], "训练集")
    require_file(config["val_data_path"], "验证集")
    require_file(
        f"{config['save_dir']}/{config['from_weight']}_{config['hidden_size']}.pth",
        "预训练权重",
    )
    require_file(str(TRAIN_SCRIPT.relative_to(PROJECT_ROOT)), "SFT训练脚本")

    (PROJECT_ROOT / config["save_dir"]).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"sft_{selected.mode}_{timestamp}.log"

    command = build_command(config)
    print("运行模式:", selected.mode)
    print("项目目录:", PROJECT_ROOT)
    print("日志文件:", log_path)
    print("执行命令:")
    print(" ".join(command))
    print("=" * 80)

    started_at = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        print(f"冒烟测试失败，退出码: {return_code}")
        print(f"请查看日志: {log_path}")
        return return_code

    print("=" * 80)
    elapsed_seconds = time.monotonic() - started_at
    print(f"SFT {selected.mode} 模式完成")
    print(f"总耗时: {elapsed_seconds / 60:.2f} 分钟")
    if config["max_steps"]:
        print(
            "包含加载、验证、保存和生成的平均耗时: "
            f"{elapsed_seconds / int(config['max_steps']):.2f} 秒/step"
        )
    print(f"日志: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
