# KG-LMF 实验入口（kbs-v2）

当前论文的方法实现位于此目录，支持 Qwen3.5-4B 和 Gemma4 E2B。所有命令在仓库根目录执行。

## 环境与数据

```bash
python -m pip install -r KGLMF/experiments/kbs/requirements.txt
python -m KGLMF.run_kbs --help
```

在 `configs/qwen.json` 或 `configs/gemma.json` 中设置模型路径、模型及编码器版本和评分器解释器。默认使用本地权重；数据和模型需自行准备。代码评分需要 Docker，数学评分使用 Math-Verify。

原始数据须保留题目 ID、来源版本、答案别名、选择题选项和代码测试。用 `recover_sources` 转换后，执行：

```bash
python -m KGLMF.run_kbs prepare --data-dir RECOVERED --output runs/kbs-v2/manifest.json
```

数据按重复/派生题组划分，固定 seed=42，train/dev/eval 目标比例为 60/20/20。成绩对应本地 held-out 子集。

## 训练与评估

按以下顺序执行子命令，完整参数见 [详细实验流程](EXPERIMENTS.md)：

1. `check` → `calibrate`：检查配置，在开发集上校准熵预算。
2. `initialize` → `feedback` → `score-feedback`：用训练集建图并准备配对反馈。
3. `train`：校准 Router/scorer，再训练 Synthesizer、adapter 和 gate 等参数。
4. `feedback` → `score-feedback` → `finalize`：为选定模型重新收集反馈并完成初始图。
5. `run` → `score-run` → `summarize`：运行实验、评分并汇总。

| `--suite` | 实验内容 |
|---|---|
| `main` | 无记忆、文本记忆、潜在图记忆；单/多智能体 |
| `ablation` | 图结构、角色、跳数、门控和 Capsule 消融 |
| `budget` | 固定、熵自适应和随机预算 |
| `transfer` | 容量匹配的 source/target/full 图 |
| `stream` | 静态、追加和完整在线维护 |

两种骨干分别训练，至少使用 3 个独立训练种子。CMA-only 对照需单独训练；命令见详细流程。

## 关键约定

- 初始图和训练反馈仅来自 train；dev 用于校准和模型选择，eval 标签不进入记忆。
- 静态实验禁止写图，在线实验仅写入无标签交互。
- 代码/数学预测必须评分后再汇总，`null` 分数不能当作 0。
- 各条件固定评分器版本，外部基线共享相同的允许训练信息。

36 项 CPU 合成测试已通过；未运行真实模型训练或基准实验。检查命令：

```bash
python -m pytest KGLMF/experiments/kbs/tests -q
```
