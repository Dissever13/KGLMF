# KG-LMF

**KG-LMF: Online Latent Graph Memory for Multi-Agent Reasoning**

KG-LMF combines a shared latent reasoning graph, role-conditioned memory composition, and adaptive memory budgets for multi-agent reasoning.

## Installation

```bash
pip install -r KGLMF/runtime/requirements.txt
```

## Main experiment

Download the model, encoder and benchmark data separately. Include public evidence in question inputs and reserve reference answers and tests for scoring.

```bash
python -m KGLMF.experiments.configure --model /path/to/model --encoder /path/to/encoder --output runs/config.json
python -m KGLMF.experiments.prepare_data --data-dir /path/to/data --dataset strategyqa --output runs/manifest.json
python -m KGLMF.experiments.run_main run --config runs/config.json --manifest runs/manifest.json --seed 42 --output runs/strategyqa-42
python -m KGLMF.experiments.summarize --output runs/strategyqa-42
```

Use a separate output directory for each benchmark and seed. Add `--resume` to resume the same run. Code evaluation requires Docker with the official tests' dependencies.

`KGLMF/experiments/` contains experiment scripts; `KGLMF/runtime/` contains their shared implementation.

```bash
python -m pytest KGLMF/runtime/tests -q
```
