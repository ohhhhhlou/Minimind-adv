# AdvertiseGen 独立规则测评 v2

## 覆盖下降诊断（新增）

standard 默认参数保持不变；目前它是待验证配置，不视为优于旧贪心配置。
先对已有两份结果核验 manifest、相同行号的输入/参考/属性，查看前20条文案：

```bash
.venv-eval/bin/python -B -m evaluation.compare_results evaluation_results/sft-best-val998-v2 evaluation_results/sft-ep1-best-standard-val998 --limit 200 --samples 20 > evaluation_results/compare-old-standard-first200.json
```

报告 checks 的 same/different/unknown 分别表示一致、不同、证据缺失；重新评分的
manifest 自动追溯 original_manifest，指标版本从最外层读取。输入、参考或属性集合不同
则拒绝配对。两侧只对共同前200行重新汇总，不能拿旧全998均值直接比新200条。
examples 是相同行号的前20条，含文案、遗漏、长度、结束原因。报告不会修改源结果。

确认权重/数据/tokenizer一致、人工查看新输出后，再做独立于 standard 的单因素干预：
length256 只把长度改为256；greedy 只把 do_sample 改为False（temperature/top_p仍记录但不生效）；
no_penalty 只把 repetition_penalty 改为1；no_ngram 只把 no_repeat_ngram_size 改为0。
每个实验都相对于 standard 改一个配置，不能把不同实验之间的差异归因于单一因素。
固定 seed/单候选/逐条重置不变，manifest 增加 diagnostic 字段。

```bash
# 每次只运行一个；diagnostic 只允许 val 前20或前200，不接触最终test。
for experiment in length256 greedy no_penalty no_ngram; do
  .venv-eval/bin/python -B -m evaluation.run --checkpoint out/advertise_sft_ep1_best_512.pth --split val --limit 200 --diagnostic "$experiment" --run-name "sft-ep1-diag-${experiment}-val200" || break
done
# 示例：standard全量报告自动截取共同前200，与长度实验比较
.venv-eval/bin/python -B -m evaluation.compare_results evaluation_results/sft-ep1-best-standard-val998 evaluation_results/sft-ep1-diag-length256-val200 --limit 200 --samples 20 > evaluation_results/compare-standard-length256.json
```

不需要重新生成 standard 前200：其每条固定种子的全量结果可以直接取前200作对照。
旧贪心结果仍保留。禁止覆盖已有运行目录；所有诊断均使用新名称。
当前 repetition_penalty/no_repeat_ngram_size 使用 Transformers 默认 logits processors，
decoder-only 的 input_ids 包含提示词，故限制可能影响输入属性复述；本次不改为仅输出范围，
先通过禁用对应限制的实验确认影响。长度、采样、限制存在交互，单因素结果不保证可相加。

## 当前唯一 standard 生成协议

入口默认或显式使用 `--generation-profile standard`，不提供 baseline/anti_repeat 两套模式。
配置定义于 `generation.py`：do_sample=True、temperature=0.75、top_p=0.90、
repetition_penalty=1.15、no_repeat_ngram_size=3、max_new_tokens=96、seed=2026。
显式 top_k=0（不叠加默认 top-k 截断），num_beams=1、num_return_sequences=1，单次生成，无重排。
每条生成前重置 torch seed=2026，固定文件顺序取前 N 条；固定 200 即 val 文件前 200 条。
manifest 保存 generation_profile、generation_protocol、requested_generation、完整实际 generation、
seed 及 seed_policy；逐条也保存 seed。reference-only 只记录请求协议，不代表执行了生成。

旧 `sft-best-val998-v2` 保留为历史确定性配置对照（用户报告：字符重复率41.34%、
token重复率40.27%、循环率30.06%、EOS率78.66%），不覆盖也不把旧输出重评分当作新采样实验。
新旧解码不同，差异不能单独归因于训练；后续各阶段统一使用 standard。

```bash
cd /root/autodl-tmp/MokioMind
.venv-eval/bin/python -B -m evaluation.run --checkpoint out/advertise_sft_ep1_best_512.pth --generation-profile standard --split val --limit 20 --run-name sft-ep1-best-standard-val20
.venv-eval/bin/python -B -m evaluation.run --checkpoint out/advertise_sft_ep1_best_512.pth --generation-profile standard --split val --limit 200 --run-name sft-ep1-best-standard-val200
.venv-eval/bin/python -B -m evaluation.run --checkpoint out/advertise_sft_ep1_best_512.pth --generation-profile standard --split val --run-name sft-ep1-best-standard-val998
.venv-eval/bin/python -B -m evaluation.view_results evaluation_results/sft-ep1-best-standard-val998 --by-style
```

目录默认位于 `/root/autodl-tmp/MokioMind/evaluation_results`，同名目录存在时直接拒绝运行。
本地无模型验证：`python -B -m unittest evaluation.test_generation evaluation.test_metrics evaluation.test_rescore -v`。

## 多值解析修正与旧结果重新评分

指标版本为 `advertisegen-literal-v2`。自然语言模板的顿号表示同属性的多个值，
例如 `材质为混纺、纤维` 解析为 `(材质, 混纺)`、`(材质, 纤维)`，与结构化重复键一致。
只拆自然语言的顿号，不推断斜杠、“和”等分隔符，结构化属性值保持原样。
空子项（如 `混纺、、纤维`）报错。此协议针对当前 AdvertiseGen 模板，不保证任意自由文本。

本地全量审计：train 78,552 条，其中自然语言多值 17,189 条，无同源双格式配对；
val 998 条/499 对、test 540 条/270 对，全部配对属性集合一致，无歧义组。
结构化值中未发现 `、，,；;/`。这些检查证明当前 val/test 的双格式一致性，
不等于 train 每条均获得结构化对照验证。

上传新增或更新的 evaluation 文件后，从服务器项目根目录执行：

```bash
# 可复现全量审计（只读数据）与测试
.venv-eval/bin/python -B -m evaluation.audit_attributes
.venv-eval/bin/python -B -m unittest evaluation.test_metrics evaluation.test_rescore -v
# 原结果必须完整；新目录必须不存在，不重新加载模型或 tokenizer
.venv-eval/bin/python -B -m evaluation.rescore evaluation_results/sft-best-val20 --output-dir evaluation_results/sft-best-val20-v2
.venv-eval/bin/python -B -m evaluation.view_results evaluation_results/sft-best-val20-v2 --by-style --samples 3
```

重新评分仅更新两组文案的属性、命中数、覆盖率和遗漏列表；原文、生成 token、
EOS、长度和重复/循环分数原样保留。参考 token 序列未保存也不影响此修正。
新 manifest 记录旧 manifest、源文件哈希及新评分代码哈希；旧报告不覆盖。
不同指标版本不能混比：已有各 checkpoint 报告均须重新评分后再比较。

## 通用结果查看脚本

`overall.output_metrics` 不是文件，而是结果目录中 `summary.json` 的嵌套字段：
`overall` → `output_metrics`（模型指标），`overall` → `reference_metrics`（参考答案指标）。

```bash
# 在项目根目录执行；更换结果目录即可查看任意一次 val/test 测评。
.venv-eval/bin/python -B -m evaluation.view_results evaluation_results/sft-best-val20
# 同时查看分组指标和前 3 条文案。
.venv-eval/bin/python -B -m evaluation.view_results evaluation_results/sft-best-val20 --by-style --samples 3
# 也可传 summary.json 文件路径；导出便于阅读的普通文本。
.venv-eval/bin/python -B -m evaluation.view_results evaluation_results/sft-best-val20/summary.json --by-style --samples 3 > evaluation_results/sft-best-val20/readable_report.txt
```

此脚本仅依赖 Python 标准库，不调用模型或重新测评。默认只打印汇总，缺失指标显示
“不适用/未提供”，比例显示为百分比；`--samples N` 展示文件顺序的前 N 条文案。
若没有 `summary.json`，先检查原测评是否完成，不把部分 JSONL 当作完整报告。

入口：`python -B -m evaluation.run`（从项目根目录执行）。不修改训练、数据或权重。
使用注册的 EvalScope `ModelAPI` / `ModelOutput`，本地串行 runner 直接调用该接口。
本版不接入 `run_task` 的数据集注册、缓存和 Dashboard；JSONL/JSON 为本实验的报告格式。
适配器注册名为 `mokiomind_native`，加载原生 `.pth` state_dict，`strict=True`。
默认 `out/full_sft_best_512.pth` 来自训练脚本的命名约定，服务器上应核对实际 best 文件位置。

## 依赖与兼容性

拟固定 `evalscope==1.8.0`，接口依据 [官方自定义模型说明](https://evalscope.readthedocs.io/zh-cn/v1.8.0/advanced_guides/custom_model.html)。
[发布元数据](https://pypi.org/project/evalscope/1.8.0/)列出 Python 3.10–3.12；仓库 pyproject 声明 Python >=3.13，
且 torch/transformers 只有下界。不能将“pip 能解析”视为运行兼容性证明。
本地未安装新依赖，也未验证 EvalScope 与真实模型的联合运行。
建议在 AutoDL 单独准备 Python 3.12 测评环境，沿用训练成功的 torch、transformers 精确版本；
直接从仓库运行模块，不用 `pip install -e .`（后者受项目 Python >=3.13 约束）。
若使用 Python 3.13，必须先完成依赖解析和 20 条真实生成验证。
不要为安装 EvalScope 改动训练环境。先在训练环境保存精确版本约束，再到独立环境执行以下命令：

```bash
cd /root/autodl-tmp/MokioMind
mkdir -p evaluation_results
python -m pip freeze > evaluation_results/training-environment.txt
# 将其中 torch、transformers、tokenizers、numpy 等关键版本复制为 evaluation_results/constraints.txt。
# 下列命令在独立测评环境执行；先审核 dry-run，冲突时停止，不强行升级训练依赖。
python -m pip install --dry-run -c evaluation_results/constraints.txt -r evaluation/requirements.txt
python -m pip install -c evaluation_results/constraints.txt -r evaluation/requirements.txt
python -m pip check
python -m pip freeze > evaluation_results/evaluation-environment.txt
python -B -c "from evaluation.model_api import MokioMindAPI; from evalscope.api.messages import ChatMessageSystem, ChatMessageUser; from evalscope.api.model import GenerateConfig; print(GenerateConfig(max_tokens=96))"
```

## 冻结协议

- 所有阶段统一使用数据中的原始 system/user 和当前 tokenizer 聊天模板，`add_generation_prompt=True`，不插入 thinking，不截断输入，不改变 Pretrain 的提示词模式。
- 单样本、单候选，使用上述 standard 采样协议、FP32、禁用 flash attention/TF32、严格确定性算法，超出上下文直接报错。确定性算法设置不等于贪心解码，采样依靠固定种子复现。
- 跨 GPU/torch/transformers 版本不承诺逐位一致；比较实验须固定环境、模型结构、tokenizer、解码和指标版本。旧 `--max-new-tokens` 参数已移除，长度由唯一协议固定。
- 关键属性固定为：类型、颜色、材质、面料、版型、长度、衣长、裙长、裤长、袖长。其他属性计入整体覆盖；改变该列表需要新指标版本。
- 精确覆盖：属性值是否为输出原文的子串。只裁剪属性边缘空白，不做同义词、语义、简繁转换或大小写归一。同一 `(key,value)` 去重，同 key 不同 value 保留；这是字符串诊断，不能判断事实真实性/否定语气。
- `literal_coverage` 和 `key_literal_coverage` 为逐样本覆盖比例；汇总含 macro 均值与 micro 总命中/总属性数。无关键属性时为 null，报告有效分母 `n`。`all_attributes_covered` 均值为完整覆盖样本率。
- 重复 4-gram 比率 `(窗口数-唯一窗口数)/窗口数`，分别计算去空白字符和内容 token；短于 4 为 0。循环样本：去空白文本中，任意 2–32 字符片段连续出现至少 3 次。循环率为布尔均值；只是可复现启发式。
- EOS 正常结束仅在新生成 token 的末尾实际出现 EOS 时为 true（即使正好达到长度上限）。prompt 中的 EOS 不计入。参考原文没有生成过程，EOS 为 null，不能人为记成 100%。
- 字符长度是原始解码文本的 Python `len`（含空白）；token 长度为新生成序列去掉末尾 EOS 后的长度。参考答案按相同 tokenizer、无特殊 token 编码。逐条同时保存完整生成 token 序列和停止原因。
- `--limit N` 固定取文件前 N 条，0 为全量。structured/natural 是同源改写样本，整体按行统计，并分别输出 `by_input_style`，不代表独立商品数。
- 不支持的自然语言格式直接报错，避免“只解析出部分属性”使覆盖率虚高。本仓库两种模板已全量解析检查。

## AutoDL 命令

```bash
cd /root/autodl-tmp/MokioMind
# 先确认这个路径是真正的 best checkpoint；若是其他结构同步传 --hidden-size/--num-hidden-layers/--use-moe。
python -B -m evaluation.run --checkpoint out/full_sft_best_512.pth --split val --limit 20 --run-name sft-best-val20
python -B -m evaluation.run --checkpoint out/full_sft_best_512.pth --split val --run-name sft-best-val998
# 冻结方案后才使用最终 test，不用 test 调参。
python -B -m evaluation.run --checkpoint out/full_sft_best_512.pth --split test --run-name sft-best-test540
```

默认目录 `/root/autodl-tmp/MokioMind/evaluation_results/<run-name>`。
每次运行目录必须不存在，避免覆盖。包含 manifest.json（配置、版本、SHA256）、samples.jsonl、summary.json。
失败直接报错，保留已完成行；只有成功才写入 summary.json，失败目录不能视为完整结果。重新运行请换名称。
后续 Pretrain/GRPO 只替换 `--checkpoint` 和 `--run-name`，其余设置保持一致。
导出的权重必须为完整同架构 state_dict；本版不支持仅 LoRA adapter 权重。

## 不加载模型的本地验证

```powershell
.venv/Scripts/python.exe -B -m unittest evaluation.test_metrics -v
.venv/Scripts/python.exe -B -m evaluation.run --reference-only --limit 20 --output-dir F:/codex/MokioMind/MokioMind/evaluation/local_results --run-name reference-val20
```

`--reference-only` 仅依赖标准库，token/EOS 指标为 null，不能充当生成冒烟。
所有本地输出位于 F 盘；程序拒绝 C 盘输出并在导入 ML 库前将缓存/临时路径放入指定输出根目录。
运行使用 `-B` 避免生成 Python bytecode。未调用训练工具，不联网下载模型/tokenizer。
