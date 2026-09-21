from pathlib import Path
import subprocess
import sys


# ======================== CONFIG: edit these values ========================
CONFIG = {
    # Use "../dataset/dpo_test_1000.jsonl" for a short test,
    # and "../dataset/dpo.jsonl" for full training.
    "data_path": "../dataset/dpo.jsonl", #训练数据要修改
    "save_dir": "../out",
    "save_weight": "dpo_lr2e7",
    "from_weight": "full_sft",
    "from_resume": 0,
    "epochs": 1,
    "batch_size": 8,
    "accumulation_steps": 2,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "max_seq_len": 340,
    "num_workers": 4,
    "learning_rate": 2e-7,
    "beta": 0.1,
    "log_interval": 10,
    "save_interval": 1000,
}

# Test log by default. Change to "../train_dpo.log" for full training.
# 把训练过程记录到哪个日志文件
LOG_PATH = "../训练记录/train_dpo_lr2e7_rebuild.log"
# ===========================================================================


def main() -> int:
    trainer_dir = Path(__file__).resolve().parent
    train_script = trainer_dir / "train_dpo.py"
    log_path = (trainer_dir / LOG_PATH).resolve()

    command = [sys.executable, str(train_script)]
    for name, value in CONFIG.items():
        command.extend((f"--{name}", str(value)))

    print(f"Working directory: {trainer_dir}")
    print(f"Log file: {log_path}")
    print("Starting DPO training...")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=trainer_dir,
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
            log_file.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())

# 关掉电脑仍旧训练的代码
"""
cd /root/autodl-tmp/MokioMind

nohup uv run python trainer/run.py \
  > run_dpo_launcher.log 2>&1 &
"""

# 实时显示训练代码
"""
tail -f /root/autodl-tmp/MokioMind/train_dpo.log
"""

# 实时查看GPU使用情况
"""
watch -n 1 nvidia-smi
"""

# 训练完成后，查看模型权重
"""
ls -lh /root/autodl-tmp/MokioMind/out/dpo_512.pth
"""

# 训练完成后，查看日志有无异常
"""
grep -nEi "error|traceback|nan|inf|out of memory" \
  /root/autodl-tmp/MokioMind/train_dpo.log
"""
# 跑程序
"""
uv run python eval.py \
  --weight full_sft \
  --save_dir out \
  --max_new_tokens 128
"""