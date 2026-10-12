# KG-LMF 正式方法与实验入口（kbs-v2，2026-10-10）

本入口对应 `paper/kbs_revision_2026-10-09/main_kbs_revised.tex` 的方法和附录 D。工作目录为 `D:/KGLM`。使用 `python -m KGLMF.experiments.kbs`，或项目根目录下 `python -m KGLMF.run_kbs`。旧 `experiments/scripts` 的 AAAI 训练、launcher、自动链以及 `kbs-v1` checkpoint/manifest 不适用于这份方法，继续保留用于历史追溯。

**本次只修改代码、文稿和运行说明；只运行 CPU 合成测试，没有运行以下真实训练、校准、推理、代码评分或 Docker 构建命令。** 论文实验数值暂时沿用原稿；投稿前必须使用此入口实际产生的结果更新它们，不能将本次代码修改当作旧数值的复现认证。

## 安装与代码验证

在仓库根目录执行以下命令；先按 GPU/CUDA 环境安装对应的 PyTorch。依赖文件固定本地已确认具有 Qwen3.5/Gemma4 模型类的 Transformers 版本。CPU 合成测试不加载模型权重，也不执行真实实验。

```powershell
python -m pip install -r KGLMF/experiments/kbs/requirements.txt
python -m pytest KGLMF/experiments/kbs/tests -q
python -m KGLMF.experiments.kbs --help
```

示例中的 `$py` 和 `backend.model` 来自作者本地环境，请替换为自己的解释器及模型路径，并填写真实的模型/编码器源版本。发布的 JSON 配置以 PATH 中的 `python` 调用评分器；若主实验使用其他解释器，请将 `scoring.command[0]` 设置为同一解释器。`python -m KGLMF.experiments.kbs.create_configs` 会按当前解释器重新生成配置。原始数据、manifest、权重、checkpoint、预测和结果文件不随源码提交。

## 1. 数据与评分环境

原始数据按许可自行获取。旧本地导出中的 GPQA/MMLU-Pro 选项和代码题测试不完整，不能直接作为正式输入。`recover_sources` 支持九个基准的显式原始 schema；不推测缺失选项、正确答案或测试。各源必须有真实版本号/commit，所有恢复文件按 `protocol.DATASETS` 中的文件名放入同一目录。

```powershell
$py = 'D:/anaconda3/envs/kglmf/python.exe'
& $py -m KGLMF.experiments.kbs.recover_sources --dataset gpqa --input ORIGINAL_GPQA.json --output RECOVERED/gpqa_real.json --revision SOURCE_COMMIT
& $py -m KGLMF.experiments.kbs.recover_sources --dataset bigcodebench --input ORIGINAL_BIGCODEBENCH.json --output RECOVERED/bigcodebench_real.json --revision SOURCE_COMMIT
& $py -m KGLMF.experiments.kbs.audit_sources --data-dir RECOVERED --output runs/kbs-v2/source_audit.json
& $py -m KGLMF.experiments.kbs prepare --data-dir RECOVERED --output runs/kbs-v2/manifest.json
```

正式协议：全局规范化问题、输入去重和已知派生关系分组，固定 seed=42，按组分配 60/20/20 目标比例。相同题干但选项不同的记录保留并放入同组。`id/task_id/question_id`、`parent_id/derivative_of/group_id/task_family_id` 和显式 `related_source_ids` 用于联合分组；未知语义改写需要在原始数据中补充来源关联，程序不会猜测。模型训练 seed 不改变 manifest。默认保留全部可用记录；`prepare --limits TRAIN DEV TEST` 可在划分后按完整组设每个基准的上限，并写入 manifest。任何上限/比例变动须同步最终论文，不得跨 manifest 混合成绩。

这属于**本地基准划分**，不是官方完整测试或 leaderboard 成绩。MATH-500 的评估部分只能称其 held-out subset。真实 ID、源版本、题数、分区、别名、选项和代码测试均记录在 manifest，训练入口会拒绝缺失源版本/代码测试/结构化选项的数据。

数学和代码在**训练反馈、开发集选择、最终评估**中使用同一版本的评分器。默认 `scoring.command` 指向本项目 `grader`：数学调用安装的 Math-Verify，代码在 Docker 容器中运行源测试。只有显式执行评分命令才会启动容器。

```powershell
& $py -m pip install math-verify
docker build -t kglmf-evaluator:1 KGLMF/experiments/kbs/evaluator
```

源测试需要完整 Python 字符串，并可由 unittest/pytest 收集。镜像缺少某个数据版本的依赖时，按其实际 requirements 增补 Dockerfile，再固定该镜像供所有条件使用。解析失败、评分器缺失、无测试或基础设施错误会终止评分；代码超时/测试失败才记 0。该容器是**源测试执行器**，不会冒充 BigCodeBench 官方实现。镜像 SHA 和执行器摘要写入 verdict。容器无主机挂载、无网络/GPU、只读文件系统，限制 CPU/内存/进程数。也可以把 `scoring.command` 换成原始官方评分器的隔离桥接程序：stdin 请求和 stdout verdict 格式见 `training.TaskScorer`，必须返回完整预测哈希、score、harness 和 revision。

## 2. 一个 seed 的完整准备与训练

配置已与论文统一：K=4、4-hop、写入16行/合成最多32行、Bmin=12、β=.5、τs=.1，四类型各500容量，CMA锚点与 ridge=1e-5，连续前缀和输入门控。配置的模型路径来自现有项目，运行前确认本地路径、模型源版本和支持 Qwen3.5/Gemma4 的 transformers。默认 BF16、本地权重、batch=1、温度1，不自动下载或静默量化。编码器和骨干冻结；loss 保留前缀梯度。Gemma 的 embedding scale 同时应用于文本与前缀，前缀的 token-identity PLE 置零并保留连续 context projection。

```powershell
$cfg = 'KGLMF/experiments/kbs/configs/qwen.json'
$manifest = 'runs/kbs-v2/manifest.json'
& $py -m KGLMF.experiments.kbs check --config $cfg --manifest $manifest --output runs/kbs-v2/check.json
& $py -m KGLMF.experiments.kbs calibrate --config $cfg --manifest $manifest --output runs/kbs-v2/qwen_calibrated.json
$cfg = 'runs/kbs-v2/qwen_calibrated.json'

# 初始化只取 train：确定性 round-robin 最多500个源；生成训练交互后写 Context/Trajectory。
& $py -m KGLMF.experiments.kbs initialize --config $cfg --manifest $manifest --seed 42 --output runs/kbs-v2/qwen/s42_initial.pt

# 全部训练问题 × 三种角色：with/no-memory 和每个检索节点的 with/without-node。
& $py -m KGLMF.experiments.kbs feedback --config $cfg --manifest $manifest --seed 42 --checkpoint runs/kbs-v2/qwen/s42_initial.pt --output runs/kbs-v2/qwen/s42_initial_feedback
& $py -m KGLMF.experiments.kbs score-feedback --config $cfg --manifest $manifest --feedback runs/kbs-v2/qwen/s42_initial_feedback/pairs.jsonl --output runs/kbs-v2/qwen/s42_initial_feedback/pairs.scored.jsonl

# 三轮 Router/scorer 辅助校准；随后冻结二者，再三轮 Synthesizer/adapter/gate NLL。
& $py -m KGLMF.experiments.kbs train --config $cfg --manifest $manifest --seed 42 --initial runs/kbs-v2/qwen/s42_initial.pt --feedback runs/kbs-v2/qwen/s42_initial_feedback/pairs.scored.jsonl --output runs/kbs-v2/qwen/s42_selected.pt

# 选定模型后重新测 train 单节点收益，再形成正式初始图。未 finalize 的 checkpoint 不能 run。
& $py -m KGLMF.experiments.kbs feedback --config $cfg --manifest $manifest --seed 42 --checkpoint runs/kbs-v2/qwen/s42_selected.pt --output runs/kbs-v2/qwen/s42_graph_feedback
& $py -m KGLMF.experiments.kbs score-feedback --config $cfg --manifest $manifest --feedback runs/kbs-v2/qwen/s42_graph_feedback/pairs.jsonl --output runs/kbs-v2/qwen/s42_graph_feedback/pairs.scored.jsonl
& $py -m KGLMF.experiments.kbs finalize --config $cfg --manifest $manifest --seed 42 --checkpoint runs/kbs-v2/qwen/s42_selected.pt --feedback runs/kbs-v2/qwen/s42_graph_feedback/pairs.scored.jsonl --output runs/kbs-v2/qwen/s42_final.pt
```

有配置评分器时 `feedback` 已逐题评分，`score-feedback` 会保留已评分值并绑定文件哈希；没有配置时可以先导出 null 分数，再配置隔离评分器评分。正式训练/finalize 不接受缺失反馈分数、不匹配的 checkpoint/manifest 或未完成的运行。所有输出用全新文件/目录，避免覆盖和混入半次实验。

dev 只负责归一化任务分数选择 checkpoint 和最大熵校准，没有梯度更新或初始图写入。测试标签不进入 Router、gate、Capsule 或图维护。训练时检索排除整道题及其派生组。合成预算按12/16/24/32循环，选定 checkpoint 刷新缓存。Capsule 增益是选定模型的单节点反事实差，不使用总体 memory/no-memory 差代替。

公式9的结构成本系数为可学习参数：`gamma = 0.1 + 1.8 * softmax(cost_logits)`，范围0.1–1.9，总和固定为2；零初始化得到1/1。联合训练使用未分离梯度的结构成本更新 logits，硬检索及离散计数不传梯度。记录到结果文件的是分离梯度后的数值；参数随 memory state 保存，评估时冻结。旧的固定系数 checkpoint 缺少 `cost_logits`，需从新版初始化入口重新生成，不能直接当作新版训练结果。

**A3/CMA-only 对照**：同一 initialized checkpoint 另做初始 feedback，添加 `--adapter cma`；train 添加 `--freeze-adapter` 并使用该对照自己的反馈文件，之后同样 feedback/score-feedback/finalize。最终 run 自动使用线性 CMA，并给条件名添加 `_fixed_adapter`。不能只冻结随机 MLP、也不能共用 full adapter 的配对反馈。

## 3. 至少五组实验

|组|对照|问题|
|---|---|---|
|E1 主结果|none/text/latent × single/multi，两骨干；接入可核验的外部强基线|总体性能、重复种子与计算公平性|
|E2 机制消融|flat、shuffled graph、no-role、1-hop、fixed-gate、no-Capsule、single；另训CMA-only|图、角色、adapter、门控分别贡献多少|
|E3 预算效率|固定4/8/12/16/24/32、连续entropy、随机12–32，K固定|熵预算的收益及probe开销|
|E4 跨域来源|source/target/full，匹配容量，全部来源于train|排除目标评估样本，区分来源与容量|
|E5 在线长期记忆|static、bounded append、full evolution、no-Capsule；固定保留探针|晋升、遗忘、适应和图增长|

```powershell
$ckpt = 'runs/kbs-v2/qwen/s42_final.pt'
& $py -m KGLMF.experiments.kbs run --config $cfg --manifest $manifest --checkpoint $ckpt --seed 42 --suite main --output runs/kbs-v2/qwen/s42_main
& $py -m KGLMF.experiments.kbs run --config $cfg --manifest $manifest --checkpoint $ckpt --seed 42 --suite ablation --output runs/kbs-v2/qwen/s42_ablation
& $py -m KGLMF.experiments.kbs run --config $cfg --manifest $manifest --checkpoint $ckpt --seed 42 --suite budget --output runs/kbs-v2/qwen/s42_budget
& $py -m KGLMF.experiments.kbs run --config $cfg --manifest $manifest --checkpoint $ckpt --seed 42 --suite transfer --output runs/kbs-v2/qwen/s42_transfer
& $py -m KGLMF.experiments.kbs run --config $cfg --manifest $manifest --checkpoint $ckpt --seed 42 --suite stream --output runs/kbs-v2/qwen/s42_stream
```

至少3个独立训练种子，建议配置中42–46五个；Gemma使用另一配置重复完整训练。训练后 main 也采用 entropy，因此必须先 calibrate。`--conditions` 可选择套件部分条件。transfer 的默认64节点若某种来源不足会报错，三种来源须共同下调容量；共同角色节点保留。E4 的模型参数仍接受混合域训练，只能说明记忆来源迁移，不是未见域参数泛化。no-Capsule 保留原节点内容，仅取消其 Capsule 身份和后续晋升；不通过删除所有来源人为制造收益。

## 4. 分数、外部基线与报告

run 保存实际预测及全部成本，code/math 的分数保持 null，直到明确评分。对每个条件生成 `.scored.jsonl` 后再汇总；不要把 null 当作0。

```powershell
& $py -m KGLMF.experiments.kbs score-run --config $cfg --manifest $manifest --predictions runs/kbs-v2/qwen/s42_main/kg_multi.jsonl --output runs/kbs-v2/qwen/s42_main/kg_multi.scored.jsonl
& $py -m KGLMF.experiments.kbs summarize --runs runs/kbs-v2/qwen/s42_main runs/kbs-v2/qwen/s43_main runs/kbs-v2/qwen/s44_main --output runs/kbs-v2/qwen/summary.json
& $py -m KGLMF.experiments.kbs failures --runs runs/kbs-v2/qwen/s42_main --output runs/kbs-v2/qwen/failures.json
& $py -m KGLMF.experiments.kbs.graph_analysis --checkpoint runs/kbs-v2/qwen/s42_stream/evolve.graph.pt --output runs/kbs-v2/qwen/graph_analysis.json
```

汇总 SD/95% CI 的重采样单位是独立训练 seed，配对对比匹配同一seed/样本；重复副本或未完成目录会被拒绝。记录 prompt/output/probe token、latent位置、通信、编码/路由/合成/生成耗时、维护耗时和结构 cost；通信已进入 prompt 时不重复相加，结构 cost 不能当 token/秒。保存硬件、精度、模型/库版本、解码参数、graph provenance、维护事件和原始输出。失败分类需要人工核查，图随机对照不自动证明 small-world。

原 `harness` 和 `export-harness/import-scores` 仍支持 Math-Verify、BigCodeBench 官方流程及版本化 verdict。代码测试镜像和评分器版本必须在所有条件/种子间固定。

外部基线使用核实的原始实现，不能把旧提示词近似标成正式方法。`export-memory` 导出允许的训练经验供基线共享；`import_external` 检查同一训练 ID、完整评估 ID、实现 commit、预测及成本字段，再纳入相同汇总。

```powershell
& $py -m KGLMF.experiments.kbs export-memory --checkpoint $ckpt --manifest $manifest --output runs/kbs-v2/allowed_memory.json
& $py -m KGLMF.experiments.kbs.import_external --source ACTUAL_BASELINE_RECORDS.json --manifest $manifest --output runs/kbs-v2/qwen/s42_external
```

## 5. 本次验证范围

CPU 小张量/假模型测试覆盖分组隔离、硬检索梯度边界、16/32写入合成、CMA与embedding scale、role composition、gate、四类型容量、Capsule配对增益、token block合并、全链训练/finalize/静态评估和统计/评分桥接。真实 GPU 兼容性、实际数据规模、源测试依赖和性能收益尚未通过真实实验验证。

```powershell
& $py -m pytest KGLMF/experiments/kbs/tests -q
& $py -m compileall -q KGLMF/experiments/kbs
```
