# KG-LMF 实验代码

2026-10-29：结构成本改为组合 token 预算、活跃节点数与活跃边数之和（带可学习权重）。
实现定义、版本兼容性和测试说明见 [STRUCTURAL_COST.md](STRUCTURAL_COST.md)。
此变更未重跑基准；历史结果不可按新的成本定义重新解释。

当前入口实现论文的逐题在线协议（`kbs-online-v3`）：同一任务领域共享连续问题流；从空经验图开始，先保存答案，再评分、训练和写图。所有基准均使用内部反馈与允许的题后基准反馈。

在仓库根目录执行：

```bash
python -m pip install -r KGLMF/experiments/kbs/requirements.txt
python -m KGLMF.run_kbs prepare --data-dir RECOVERED --output runs/online/manifest.json
python -m KGLMF.run_kbs check --config KGLMF/experiments/kbs/configs/qwen.json --manifest runs/online/manifest.json
python -m KGLMF.run_kbs run --config KGLMF/experiments/kbs/configs/qwen.json --manifest runs/online/manifest.json --suite main --seed 42 --output runs/online/qwen-main-42
```

先在配置中填写本地模型路径和版本。数据需保留来源版本、题目ID、选项及代码测试；代码/数学评分器需提前配置。代码参考实现和隐藏测试不进入代理提示或记忆。

`--suite` 可选 `main`、`ablation`、`budget`、`sensitivity`、`transfer`、`stream`。同一实验用相同问题清单，按配置的独立种子分别运行。中断后用原命令加 `--resume`；不要复用旧离线 checkpoint。

汇总：

```bash
python -m KGLMF.run_kbs summarize --runs runs/online/qwen-main-42 runs/online/qwen-main-43 runs/online/qwen-main-44 --output runs/online/summary.json
python -m pytest KGLMF/experiments/kbs/tests -q
```

详细协议、转移定义及其他实验命令见 [EXPERIMENTS.md](EXPERIMENTS.md)。旧离线流程仅保留作历史代码，默认入口不再使用。此版本未运行真实实验。
