# 在线实验流程

## 数据与状态

原始数据转换可使用 `recover_sources`；保留 `source_revision`、原始题目ID、答案别名、选择题选项、代码任务ID与可执行测试。`prepare` 不按60/20/20划分，而是将重复/派生题组成组，独立于答案选择一个代表，并保存固定的题目顺序和每个基准的保留数量。`--limit N` 是每个基准的样本上限；这类结果须标明子集。

四个独立问题流分别为：

| 流 | 基准 |
|---|---|
| reasoning | StrategyQA、HotpotQA |
| qa | TriviaQA、PopQA |
| code | KodCode、BigCodeBench |
| science_math | MATH-500、MMLU-Pro、GPQA |

同一流内图、Router、Synthesizer、scorer、adapter/gate、优化器和熵历史连续保留；不同领域、条件和独立种子分别重置。冻结语言模型和编码器权重。初始图仅有无经验的Role锚点，第一题不检索记忆。

## 每题执行顺序

1. 用题前图/参数快照逐个运行planner、solver、critic；single条件只有solver。代理输入视图没有参考答案、代码实现或测试字段。
2. 将首次最终答案、生成种子、检索集合和开销写入并刷新 `answers.jsonl`，然后才能访问评分器。
3. 完成基准评分，在题前快照上收集内部critic反馈与配对探测。全部9个基准都允许题后外部反馈。内部判决不会覆盖失败的基准判决。
4. Router/scorer通过辅助损失更新；Synthesizer/write/adapter/gate/成本参数通过NLL更新。各请求使用记录的检索集合与预算，硬Top-K索引不接收NLL梯度。缺少目标或经验时跳过相应更新。
5. 重新编码旧节点，记录读和单节点增益，写入本题经验，再晋升、合并、重算和剪枝。更新结果只服务下一题。

参考答案只直接监督solver；其他角色使用格式一致且获得内部接受与基准验证的生成响应。代码目标必须来自验证通过的生成程序；不使用参考实现。标量reward不是NLL目标。熵预算只使用已完成问题的最大记录熵，第一题使用log词表大小作为尺度。

首次答案耗时与题后反馈、探测、训练、重新编码及维护开销分别记录。`all_text_tokens` 为首次作答成本；`total_text_tokens_including_adaptation` 包含题后模型调用。隐藏测试仅在隔离评分器执行，不在主机Python进程执行。

## 实验入口

先编辑 `configs/qwen.json` / `configs/gemma.json`，填写本地权重及固定版本。Docker代码评估器的配置沿用 `grader` 和 `evaluator/Dockerfile`；数学使用Math-Verify。`check` 不加载模型。只有显式调用 `run` 会进行真实实验。

```bash
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite main --seed 42 --output runs/main-42
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite ablation --seed 42 --output runs/ablation-42
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite budget --seed 42 --output runs/budget-42
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite sensitivity --seed 42 --output runs/sensitivity-42
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite transfer --seed 42 --output runs/transfer-42
python -m KGLMF.run_kbs run --config CONFIG --manifest MANIFEST --suite stream --seed 42 --output runs/stream-42
```

`--conditions NAME...` 可选取配置中的条件。`main` 比较单/多智能体及无、文本、潜在图记忆；`ablation` 包含结构/角色/跳数/gate/Capsule/固定CMA；`budget` 比较固定、熵自适应和随机预算；`sensitivity` 固定B=16扫描K=1/2/4/8，并固定K=4扫描B=4/8/16/32；`stream` 比较完整维护、追加和不写图。

`flat` 仅取消图传播，`flat_pool` 才同时取消边、Role锚点检索及图维护，保留相同的latent writer/composer；不可把两者都叫完整去图。`no_role_retrieval` 和 `no_role_composition` 分别关闭角色路由和合成，`no_role` 同时关闭；`no_merge` / `no_prune` 单独关闭效用合并/剪枝，容量限制仍执行。从空经验图开始的 `no_write` 无法进行记忆条件NLL，属于无持久经验参照，不能单独证明图维护收益。

## 外部方法

12个外部方法必须使用各自原始实现，按同一在线协议收集首次答案；当前代码提供导入和审计接口，没有把简化提示模拟当作原方法实现。正文七基准比较单独生成七基准manifest，KG-LMF也使用该manifest，避免从额外两个基准获得经验。

```bash
python -m KGLMF.experiments.kbs.import_external --source original-method-records.json --manifest MANIFEST7 --config CONFIG --output runs/external-method-42
```

导入JSON需包含 `method`、原始实现URL/固定revision、相同manifest哈希、seed、与内部运行相同的resolved backend字段、`reset_boundary="task_category"`、`feedback="internal_and_post_answer_benchmark"`、`initial_benchmark_experience_ids=[]`。每条记录包含首次prediction/哈希、`answer_recorded_before_feedback=true`、按顺序列出的本领域全部先前反馈ID `available_feedback_ids`、实际检索来源ID `retrieved_source_ids` 及完整cost。所需cost字段见 `online_external.REQUIRED_COST`；未知费用不能填0。导入检查题序/权限声明并重新评分；声明检查不等于证明外部实现没有泄漏，应保留其原始题前答案及题后更新日志以供核验。

## 跨任务经验转移

转移方向为 non_science_to_science（目标MMLU-Pro、GPQA）与 non_reasoning_to_reasoning（目标StrategyQA、HotpotQA）。先完成所有非目标基准的源流；不同源领域仍独立初始化。将其源经验导入目标状态时，使用相同目标初始化的write模块重新编码，保留合并节点的加权原始记录。

| 条件 | 初始经验 | 目标流写图 | 目标流参数学习 |
|---|---|---|---|
| source_only | 已完成源流 | 不写目标经验 | 题后更新 |
| target_only | 无源经验 | 只写完成的目标经验 | 题后更新 |
| full | 已完成源流 | 追加完成的目标经验 | 题后更新 |

三者目标入口参数相同，题目顺序、反馈权限、检索K和经验容量上限相同。容量是上限，不意味着冷启动时占用量相同。source_only仍会适应目标反馈，所以该实验不是zero-shot测试。target_only是域内记忆参照，不是上界。源流成本保存在独立源日志中，目标评分只汇总首次答案；其开销不应被藏在目标延迟之外而称为全流程免费。

## 恢复、汇总与分析

中断使用完全相同参数加 `--resume`。状态按领域/条件保存，包含图、全部模块、两个优化器、控制器、随机状态和已完成题序。首次答案和反馈已有日志时会复用，不重新计分；更新和写图通过原子checkpoint一起提交。旧版本manifest/checkpoint或更改后的顺序/配置不能用于续跑。

```bash
python -m KGLMF.run_kbs summarize --runs RUN42 RUN43 RUN44 --output summary.json
python -m KGLMF.run_kbs failures --runs RUN42 --output failures.json
python -m KGLMF.run_kbs curves --runs RUN42 RUN43 RUN44 --window 50 --output curves.json
python -m pytest KGLMF/experiments/kbs/tests -q
```

汇总报告种子均值、标准差及配对种子差值的bootstrap区间，不混合离线和在线结果。`failures` 生成待人工标注的失败清单。每题记录图增长、晋升/合并/剪枝、参数更新次数，可用于时间曲线分析。CPU合成验证不代表真实模型或基准成绩。

逐表逐图的实验设计、最低必要实验及原设计的保留/修改项见工作区 `docs/online_implementation_2026-10-13/revision_and_experiment_plan.md`。它是运行计划，不是已取得的实验结果。
