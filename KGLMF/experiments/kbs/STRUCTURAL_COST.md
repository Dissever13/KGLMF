# Structural cost revision — 2026-10-29

New executions use `B_e + gamma_1 * N_active + gamma_2 * M_active`.
`M_active` is the number of positive undirected edges in the eligible subgraph
used for propagation. Each edge is counted once; self-transitions for isolated
nodes are excluded. Flat retrieval and zero propagation rounds contribute zero
active edges. `graph_depth` remains diagnostic metadata and is no longer a cost
input. This descriptor is not a wall-clock latency estimate.

Retrieval metadata records `active_edges` and `cost_version=active-edges-v1`.
Likelihood training and reported structural cost use the same edge count.
Online checkpoints also record this version and reject the previous cost
semantics before loading state. The aligned engine has a new method revision.
Source-hash checks must remain enabled for all resumed runs.
Do not resume historical depth-cost experiments using this implementation or
reinterpret their recorded costs as edge costs. Historical results and source
snapshots are not modified by this revision; no benchmark was rerun for it.

The current package includes the aligned method implementation and its earlier
online/temporal interfaces. Use the relevant frozen source snapshot to reproduce
an older experiment. CPU checks are run with:

```bash
python -m pytest KGLMF/experiments/kbs/tests -q
```
