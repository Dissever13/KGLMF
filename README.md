# KG-LMF

**Knowledge Graph-Driven Latent Memory Framework for Multi-Agent Test-Time Scaling**

KG-LMF stores agent experiences in a shared latent graph, retrieves memories by agent role, and composes continuous memory prefixes for a frozen language model. An entropy controller adjusts the memory token budget.

The current experiment entry point is **kbs-v2**, with configurations for Qwen3.5-4B and Gemma4 E2B.

## Setup

Run from the repository root using Python 3.10 or later. Install a PyTorch build suitable for your CUDA environment first.

```bash
python -m pip install -r KGLMF/experiments/kbs/requirements.txt
python -m KGLMF.run_kbs --help
```

Set local model paths and source revisions in `KGLMF/experiments/kbs/configs/`. Obtain benchmark data and model weights separately.

## Experiments

Five suites cover the main comparison, mechanism ablations, memory budgets, cross-task transfer, and online graph evolution.

See the [experiment quick start](KGLMF/experiments/kbs/README.md) for the workflow and [complete commands](KGLMF/experiments/kbs/EXPERIMENTS.md) for training, scoring, and reporting.

## Checks

```bash
python -m pytest KGLMF/experiments/kbs/tests -q
```

The 36 CPU synthetic tests pass. Real-model training and benchmark experiments have not been run for this implementation.

`KGLMF/experiments/kbs/` contains the current pipeline. `KGLMF/src/` and `KGLMF/experiments/scripts/` retain earlier implementations.
