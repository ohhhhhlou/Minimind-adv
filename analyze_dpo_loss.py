"""分析 DPO 训练日志中的 loss 变化。

默认读取项目根目录下的 train_dpo_lr2e7.log。
也可以在运行时传入另一个日志文件：
    uv run python analyze_dpo_loss.py train_dpo.log
"""

# argparse 用来接收终端参数，例如用户指定的日志文件名。
import argparse

# re 是 Python 自带的正则表达式模块，用来从文字中提取 loss 数字。
import re

# Path 用来安全地处理文件路径。
from pathlib import Path


def main() -> None:
    # 创建终端参数解析器。
    parser = argparse.ArgumentParser(description="统计 DPO 训练日志中的 loss")

    # 增加一个可选的位置参数 log_file。
    # 如果运行时没有传入文件名，就读取 train_dpo_lr2e7.log。
    parser.add_argument(
        "log_file",
        nargs="?",
        default="train_dpo_lr2e7.log",
        help="训练日志路径，默认：train_dpo_lr2e7.log",
    )

    # 把用户在终端输入的参数解析出来。
    args = parser.parse_args()

    # 把普通字符串转换成 Path 路径对象。
    log_path = Path(args.log_file)

    # 先检查日志是否存在，避免直接打开不存在的文件时报难懂的异常。
    if not log_path.is_file():
        print(f"错误：找不到日志文件：{log_path.resolve()}")
        print("请确认当前位于 /root/autodl-tmp/MokioMind，或传入正确路径。")
        return

    # 一次性读取完整日志。errors='replace' 可避免少量异常字符导致读取失败。
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    # 查找形如 loss:0.692253 的内容。
    # 括号中的部分是我们真正需要提取的数字。
    loss_strings = re.findall(r"loss:([0-9]+(?:\.[0-9]+)?)", log_text)

    # 正则表达式返回的是字符串，因此将每个数字转换成 float 小数。
    loss_values = [float(value) for value in loss_strings]

    # 如果一条 loss 都没有找到，就停止统计并给出原因提示。
    if not loss_values:
        print(f"日志中没有找到 loss：{log_path.resolve()}")
        print("训练可能尚未开始，或者日志格式不是 loss:0.123456。")
        return

    # 日志通常每 log_interval 个 step 才记录一次 loss。
    # 这里统计开头和结尾各 20 条；不足 20 条时使用全部记录。
    sample_count = min(20, len(loss_values))

    # 取出最前面的 sample_count 条，代表训练前段。
    first_values = loss_values[:sample_count]

    # 取出最后面的 sample_count 条，代表训练后段。
    last_values = loss_values[-sample_count:]

    # 分别计算前段、后段与全部记录的平均 loss。
    first_average = sum(first_values) / len(first_values)
    last_average = sum(last_values) / len(last_values)
    total_average = sum(loss_values) / len(loss_values)

    # 后段平均减去前段平均：负数表示下降，正数表示上升。
    average_change = last_average - first_average

    # 输出统计结果。
    print(f"日志文件：{log_path.resolve()}")
    print(f"loss 记录数：{len(loss_values)}")
    print(f"前 {sample_count} 条平均：{first_average:.9f}")
    print(f"后 {sample_count} 条平均：{last_average:.9f}")
    print(f"后段减前段：{average_change:+.9f}")
    print(f"最低 loss：{min(loss_values):.9f}")
    print(f"最高 loss：{max(loss_values):.9f}")
    print(f"全部平均：{total_average:.9f}")

    # DPO 的初始随机基线约为 -log(0.5) = 0.693147。
    dpo_baseline = 0.693147
    print(f"DPO 参考基线：{dpo_baseline:.6f}")

    # 给出非常保守的趋势说明。
    # 阈值只用于快速观察日志，不等同于最终模型质量评价。
    if average_change < -0.005:
        print("趋势判断：后段 loss 有较明显下降，模型可能学到了偏好差异。")
    elif average_change > 0.005:
        print("趋势判断：后段 loss 明显上升，需要检查训练稳定性。")
    else:
        print("趋势判断：前后变化很小，DPO 偏好学习信号可能较弱。")

    print("注意：最终仍要比较模型回答，不能只根据训练 loss 判断效果。")


# 只有直接运行这个文件时才调用 main()。
# 如果以后从其他 Python 文件导入它，就不会自动开始分析。
if __name__ == "__main__":
    main()
