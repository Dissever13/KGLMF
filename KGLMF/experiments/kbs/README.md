# Online KG-LMF experiments

Run commands from the repository root. The portable `benchmark` module uses `AlignedEngine`, domain-specific prompts and the existing durable journal runner. One invocation evaluates one benchmark, one condition and one seed. Every new output directory starts with fresh graph, memory parameters and optimizer state. Do not point it at a historical run directory.

## Inputs

Download model weights and data separately. Normalize exports to `protocol.py` and `online_protocol.py`; preserve original IDs, source revisions, answer aliases, options and choice labels. Code tests are available only to the scorer. Never insert references or solutions into questions.

`prediction_view` sends only the public question and options to agents. For evidence-based tasks, pack public evidence into the question before preparing the manifest. This loader does not fetch documents or recover evidence from private answer fields. `aligned_data.evidence_text` provides the historical label-independent 1800-token HotpotQA paragraph packer. Record truncation and evidence budgets alongside your input export.

```bash
python -m KGLMF.run_kbs prepare --data-dir /path/to/normalized-data --datasets strategyqa --seed 42 --output runs/strategyqa/manifest.json
```

Preparation uses the existing manifest format. Supplying one benchmark makes its single stream the benchmark reset boundary. The portable runner rejects multi-benchmark manifests. The order stays fixed when the subsequent run seed changes. Use `--limit N` only for a declared subset, never to replace difficult or failed questions.

## Configure and run

Copy `configs/aligned_qwen.json` and set your model/encoder paths, immutable revisions when applicable, and scorer environment. The example uses K=4, eight propagation rounds, 16 stored tokens per node, composition budgets between 12 and 32, and fixed structural-cost coefficients of one. It is not a completed run record.

```bash
python -m KGLMF.experiments.kbs.benchmark check --config /path/to/config.json --manifest runs/strategyqa/manifest.json --seed 42
python -m KGLMF.experiments.kbs.benchmark run --config /path/to/config.json --manifest runs/strategyqa/manifest.json --seed 42 --output runs/strategyqa/seed42
```

`check` validates inputs and bindings without loading a model. Only `run` starts inference. Repeat with seeds 43 and 44 in separate output directories for variability. Other benchmarks require new directories and their own manifests; state never carries between them.

## Scoring

Math uses Math-Verify. Generated programs run in Docker without network, host mounts or GPUs. Build `evaluator/Dockerfile` and provide the dependencies required by official source tests. Validate reference programs and negative controls before a run. Missing image dependencies are infrastructure failures, not incorrect answers.

The example selects `source_grader`, adding the source `solution` module alias, a `/tmp` working directory, source stdin/stdout test cases and bounded Python diagnostic capture. Verdicts identify the immutable image and worker hash. Private evaluator images and dataset-specific assets are not included. Keep the evaluator and dependencies fixed during a run.

## Resume and results

Resume the identical invocation with `--resume`. Sources, configuration, manifest, seed and resolved models must match. Source snapshots and checkpoint identities prevent silent version mixing. Answers are fsynced before scoring or feedback. Feedback binds to the saved prediction and pre-update state. A completed run is only summarized on resume.

```bash
python -m KGLMF.experiments.kbs.benchmark summarize --output runs/strategyqa/seed42
python -m pytest KGLMF/experiments/kbs/tests -q
```

Summaries separate inference text tokens from feedback text tokens and include scoring, feedback, training and maintenance in total latency. A single seed yields no SD or confidence interval. Latent positions are not silently counted as text tokens. Preserve full journals for further breakdown and independent audit.
