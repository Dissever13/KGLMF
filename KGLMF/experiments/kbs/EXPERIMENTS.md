# Implementation and experiment versions

## Portable benchmark entry point

`benchmark.py` exposes the existing aligned engine without unpublished paper files or hardcoded workstation paths. It accepts one benchmark per manifest and independently initializes each run. See [README.md](README.md). Prompts are packaged as Python resources. This release repairs packaging and entry-point portability; it does not rerun or replace any historical or author-supplied result.

The engine retains role-local assessment targets, zero-residual CMA initialization, separate routing/composition representations, and durable first-answer journals. Structural cost uses active nodes, active edges and composition length with fixed coefficients; see [STRUCTURAL_COST.md](STRUCTURAL_COST.md).

## Historical drivers

`KGLMF.run_kbs` uses the earlier `online-v3` pipeline. Its multi-benchmark manifests group benchmarks by task category. `temporal_experiment.py`, `paper_experiment.py` and `aligned_experiment.py` are historical drivers. Some preparation steps require frozen private manifests and local exports which are intentionally not distributed. They are not the portable entry point and must not be silently adapted to resume older runs.

Original four-hop/eight-hop graph runs, mixed StrategyQA/HotpotQA streams and later reference runs are separate experiments. A new default does not retroactively change their configuration or evidence inputs.

## Correspondence limits

The aligned graph retains its semantic-neighbor edge prior and new-node protection period. A manuscript formula restricted to empirical co-retrieval edges does not fully describe that implementation. This release does not remove the prior or claim that past results used a different graph. Likewise, a per-benchmark wrapper does not reclassify historical mixed-stream results as per-benchmark cold starts.

External baselines require their original implementations. Import/audit helpers do not supply them or prove absence of leakage. Tables containing author-supplied aggregate values are not reproduced by software tests or the example configuration.

For new studies, freeze source, inputs, model revisions, evaluator and configuration. Record prompts, evidence inputs and token/latency definitions. Evaluate first answers, retain negative results, and use independent repeats for uncertainty.
