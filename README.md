# KG-LMF

**KG-LMF: Online Latent Graph Memory for Multi-Agent Reasoning**

KG-LMF retrieves related experience from a shared latent graph and composes a continuous memory prefix for each requesting agent. A frozen language model performs reasoning; memory modules learn from feedback after the first answer has been saved and scored.

## Start here

Use Python 3.10 or later. Install a CUDA-compatible PyTorch build for real-model inference, then install the remaining dependencies:

```bash
python -m pip install -r KGLMF/experiments/kbs/requirements.txt
python -m KGLMF.experiments.kbs.benchmark --help
python -m pytest KGLMF/experiments/kbs/tests -q
```

The [experiment guide](KGLMF/experiments/kbs/README.md) describes the portable single-benchmark entry point, inputs, checkpoints and scoring. The [version notes](KGLMF/experiments/kbs/EXPERIMENTS.md) distinguish it from historical mixed-stream experiment drivers.

## Repository contents

- `KGLMF/experiments/kbs/`: engine, graph and memory modules, grading, entry points, configurations and CPU software tests.
- `KGLMF/experiments/kbs/prompt_resources.py`: domain and role prompts, available without a manuscript checkout.
- `KGLMF/src/` and `KGLMF/experiments/scripts/`: earlier implementations retained for history; not the current aligned entry point.

Model weights, benchmark files, journals, checkpoints, manuscript sources, figures and unpublished results are not included. Obtain each dataset under its own terms and preserve its source revision. No benchmark score is claimed by synthetic tests or the example configuration.

The local aligned backend does not require an API key. The legacy DeepSeek adapter reads `DEEPSEEK_API_KEY` from the environment; no credential belongs in a source file.
