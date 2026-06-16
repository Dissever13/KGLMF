# KG-LMF: Knowledge Graph-Driven Latent Memory Framework

**Knowledge Graph-Driven Latent Memory Framework for Multi-Agent Test-Time Scaling**

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A latent memory framework that constructs a dynamic **Latent Reasoning Graph** to enable efficient multi-agent collaboration. Instead of passing verbose text between agents, KG-LMF retrieves, composes, and injects compact **latent memory tokens** into the LLM's hidden states — reducing token consumption while maintaining reasoning quality.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Knowledge  │ →  │   Memory    │ →  │   Latent    │
│   Graph     │    │  Composer   │    │  Injection  │
│  G=(V,E,T)  │    │ (4L Transf) │    │  (Dynamic   │
│             │    │             │    │   Gates)    │
└─────────────┘    └─────────────┘    └─────────────┘
       ↑                                    ↓
┌─────────────┐                      ┌─────────────┐
│   Graph     │ ← ─ ─ ─ ─ ─ ─ ─ ─ │  Multi-Agent │
│  Evolution  │                      │ Orchestrator│
└─────────────┘                      └─────────────┘
```

**Three phases:**
1. **Graph Construction** — Build latent reasoning graph from task interactions
2. **Retrieval & Composition** — Router retrieves top-k nodes, Composer produces memory tokens
3. **Injection & Evolution** — Dynamic gates inject memory into LLM at layers [0,8,16]

## Repository Structure

```
KGLMF/
├── src/
│   ├── agents/          # Multi-agent orchestrator and prompts
│   ├── api/             # LLM client interface (Ollama, DeepSeek API)
│   ├── baselines/       # Baseline implementations (BM25, CoT, Multi-Agent)
│   ├── composer/        # Memory composer with LoRA adapter
│   ├── eval/            # Evaluation metrics and datasets
│   ├── evolution/       # Graph evolution (merge/expand/prune)
│   ├── graph/           # Latent reasoning graph (nodes, edges, ops)
│   ├── injection/       # Latent injector with dynamic gating + MemoryScorer
│   └── router/          # Graph router with adaptive test-time scaling (TTS)
├── experiments/
│   ├── configs/         # Experiment configuration YAMLs
│   └── scripts/         # Training, evaluation, and figure generation scripts
└── paper/               # AAAI paper source (LaTeX)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/DisseverYOLO/KGLMF.git
cd KGLMF

# Create conda environment
conda create -n kglmf python=3.10
conda activate kglmf

# Install dependencies
pip install torch numpy scipy tqdm sentence-transformers datasets rank-bm25 requests matplotlib
```

## Quick Start

### 1. Download benchmark datasets

```bash
python experiments/scripts/download_datasets.py
```

### 2. Train the KG-LMF components (with LLM feedback)

```bash
python experiments/scripts/train_e2e_v3.py \
    --max_total_samples 300 \
    --n_variations 3 \
    --epochs 100 \
    --n_iterations 7 \
    --device cuda \
    --model llama3.1:8b
```

### 3. Run the fair comparison experiment

```bash
python experiments/scripts/run_llama_fair_v2.py \
    --dataset all \
    --n_samples 81 \
    --checkpoint checkpoints_v3_llama/best_model.pt \
    --model llama3.1:8b
```

### 4. Evaluate results

```bash
python experiments/scripts/statistical_tests.py
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **LatentReasoningGraph** | `src/graph/graph.py` | Dynamic knowledge graph with typed nodes (Role, Trajectory, Context, Capsule) |
| **MemoryComposer** | `src/composer/composer.py` | 4-layer Transformer with learnable pooling queries and role-conditioned cross-attention |
| **GraphRouter** | `src/router/router.py` | Cosine similarity scoring + top-k retrieval with temperature scaling |
| **LatentInjector** | `src/injection/injector.py` | Dynamic gating (`σ(base + confidence × scale)`) with EmbeddingAdapter |
| **MemoryScorer** | `src/injection/injector.py` | MLP predicting accuracy delta from (query, memory_tokens) — differentiable LLM feedback proxy |
| **Orchestrator** | `src/agents/orchestrator.py` | Multi-agent Planner→Coder→Critic pipeline with latent memory retrieval |
| **AdaptiveTTS** | `src/router/tts.py` | Adaptive test-time scaling based on task complexity |

## Benchmark Datasets

- **Reasoning**: StrategyQA, HotpotQA
- **QA**: TriviaQA, PopQA
- **Code**: BigCodeBench (MBPP), KodCode
- **Math**: MATH-500
- **Science**: MMLU-Pro, GPQA

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Task correctness score |
| Latency | Wall-clock time per sample |
| Tokens | LLM text tokens consumed |
| Latent Tokens | Latent memory vectors injected (FREE — no text cost) |
| Memory Hit Rate | Average cosine similarity of retrieved graph nodes |
| Efficiency Score | Accuracy / tokens × 1000 |

## Training Methodology

The end-to-end training uses **LLM feedback as the training signal**:

1. **Phase 1** — Collect feedback: Run Llama with vs without memory, compute accuracy delta
2. **Phase 2** — Train MemoryScorer: Predict accuracy delta from (query, memory_tokens)
3. **Phase 3** — Train Composer: Maximize scorer's prediction (differentiable proxy for LLM feedback)
4. **Repeat** 7 iterations with fresh random training samples

## Known Limitations

- Latent memory injection requires LLM hidden-state access (not available through Ollama HTTP API)
- Current training uses LLM text feedback rather than direct gradient through the LLM
- Code evaluation uses text overlap rather than execution-based scoring

## Citation

```bibtex
@inproceedings{wang2026kglmf,
  title={Knowledge Graph-Driven Latent Memory Framework for Multi-Agent Test-Time Scaling},
  author={Wang, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
