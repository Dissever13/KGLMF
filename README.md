# KG-LMF

**KG-LMF: Online Latent Graph Memory for Multi-Agent Reasoning**

KG-LMF combines a shared latent reasoning graph, role-conditioned memory composition, and adaptive memory budgets for multi-agent reasoning.

## Installation

```bash
pip install -r KGLMF/runtime/requirements.txt
```

## Quick Start

Run KG-LMF on a benchmark with online memory learning.

```bash
python -m KGLMF.experiments.configure --model /path/to/model --encoder /path/to/encoder --output runs/config.json
python -m KGLMF.experiments.prepare_data --data-dir /path/to/data --dataset strategyqa --output runs/manifest.json
python -m KGLMF.experiments.run_main run --config runs/config.json --manifest runs/manifest.json --seed 42 --output runs/strategyqa-42
python -m KGLMF.experiments.summarize --output runs/strategyqa-42
```

## Code

`KGLMF/src/` contains modules for agents, graph memory, routing, memory composition, injection, and graph evolution.
