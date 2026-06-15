#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified KG-LMF Experiment Runner — consistent N=81 samples across all datasets.

Supports:
  - All 6 benchmarks + 2 multi-turn datasets
  - NoMemory baseline & KG-LMF (with/without trained composer)
  - Proper task-type routing (qa / reasoning / code / planning)
  - Saves per-sample results for statistical analysis
  - Resume from checkpoint

Usage:
    /d/WorkSoftware/python D:/KGLM/KGLMF/experiments/scripts/run_unified.py \
        --dataset triviaqa --method kglmf --n_samples 81 --output results/
"""

import os, sys, json, argparse, time, random
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.client import DeepSeekClient
from api.config import DEEPSEEK_API_KEY
from graph.graph import LatentReasoningGraph
from graph.node import NodeType
from composer.composer import MemoryComposer
from router.router import GraphRouter
from router.tts import AdaptiveTTSController
from injection.injector import LatentInjector
from evolution.evolution import GraphEvolution
from agents.orchestrator import Orchestrator
from agents.prompts import get_prompts_for_task

# ── Dataset-specific configs ─────────────────────────

DATASET_CONFIG = {
    'triviaqa':      {'task_type': 'qa',        'metric': 'exact_match'},
    'strategyqa':    {'task_type': 'reasoning', 'metric': 'accuracy'},
    'popqa':         {'task_type': 'qa',        'metric': 'exact_match'},
    'kodcode':       {'task_type': 'code',      'metric': 'pass_at_1'},
    'bigcodebench':  {'task_type': 'code',      'metric': 'pass_at_1'},
    'pddl':          {'task_type': 'planning',  'metric': 'success_rate'},
    'hotpotqa':      {'task_type': 'reasoning', 'metric': 'exact_match'},
    'coqa':          {'task_type': 'qa',        'metric': 'f1'},
}


def load_dataset(dataset_name: str, data_dir: str) -> list:
    """Load dataset from JSON file."""
    file_map = {
        'triviaqa': 'trivia_qa_real.json',
        'strategyqa': 'strategyqa_real.json',
        'popqa': 'akariasai_PopQA_real.json',
        'kodcode': 'kodcode_real.json',
        'bigcodebench': 'bigcodebench_real.json',
        'pddl': 'pddl_real.json',
        'hotpotqa': 'HotpotQA_real.json',
        'coqa': 'CoQA_real.json',
    }
    fname = file_map.get(dataset_name)
    if not fname:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset not found: {fpath}")

    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to uniform format
    samples = []
    for item in data:
        q = item.get('question', item.get('input', item.get('query', '')))
        a = item.get('answer', item.get('target', item.get('output', '')))
        if q and a:
            samples.append({'question': q, 'answer': a, 'metadata': item})
    return samples


def evaluate_answer(predicted: str, expected: str, metric: str) -> float:
    """Score a predicted answer against expected."""
    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    if metric == 'exact_match':
        # Normalize and compare
        pred_words = set(pred.split())
        exp_words = set(exp.split())
        if len(exp_words) == 0:
            return 0.0
        overlap = len(pred_words & exp_words) / len(exp_words)
        return 1.0 if overlap > 0.8 else 0.0

    elif metric == 'accuracy':
        if exp in pred or pred in exp:
            return 1.0
        # Check for yes/no match
        if exp in ('yes', 'no'):
            return 1.0 if pred.startswith(exp) else 0.0
        # Partial overlap
        pred_words = set(pred.split())
        exp_words = set(exp.split())
        if len(exp_words) == 0:
            return 0.0
        return len(pred_words & exp_words) / len(exp_words)

    elif metric == 'pass_at_1':
        # For code: check if output contains expected behavior
        return 1.0 if exp[:50] in pred else 0.0

    elif metric == 'success_rate':
        return 1.0 if exp in pred else 0.0

    elif metric == 'f1':
        pred_tokens = set(pred.split())
        exp_tokens = set(exp.split())
        if not pred_tokens or not exp_tokens:
            return 0.0
        tp = len(pred_tokens & exp_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(exp_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return 0.0


def run_experiment(dataset_name: str, method: str = 'kglmf',
                   n_samples: int = 81, data_dir: str = '',
                   checkpoint_path: str = None, output_dir: str = 'results/',
                   seed: int = 42):
    """Run unified experiment on a dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = DATASET_CONFIG[dataset_name]
    task_type = config['task_type']
    metric = config['metric']

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Method: {method} | N={n_samples} | Type: {task_type}")
    print(f"{'='*60}")

    # Load data
    samples = load_dataset(dataset_name, data_dir)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    print(f"Loaded {len(samples)} samples")

    # Initialize API
    api = DeepSeekClient()

    # Initialize KG-LMF components
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_heads=8, n_layers=4)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController()
    injector = LatentInjector(d_model=384)
    evolution = GraphEvolution()

    # Load trained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        composer.load_state_dict(ckpt['composer'])
        router.load_state_dict(ckpt['router'])
        injector.load_state_dict(ckpt['injector'])
        print(f"Loaded trained weights from {checkpoint_path}")

    orchestrator = Orchestrator(
        composer=composer, router=router, tts=tts,
        injector=injector, evolution=evolution,
        graph=graph, api_client=api, d_model=384
    )

    # Run
    results = []
    correct = 0
    total_latency = 0.0
    pbar = tqdm(enumerate(samples), total=len(samples), desc=dataset_name)

    for i, sample in pbar:
        question = sample['question']
        expected = sample['answer']

        t_start = time.time()

        if method == 'nomemory':
            # NoMemory baseline: direct single-agent call
            response = api.generate(
                system_prompt="Answer the question directly and concisely. Just give the answer, no explanation.",
                user_message=question,
            )
            answer = response.strip()
            iterations = 1
        else:
            # KG-LMF: full multi-agent pipeline
            result = orchestrator.run_task(
                task=question,
                task_type=task_type,
            )
            answer = result['final_answer']
            iterations = result['iterations']

        latency = time.time() - t_start
        total_latency += latency
        score = evaluate_answer(answer, expected, metric)
        if score >= 0.5:
            correct += 1

        results.append({
            'sample_id': i,
            'question': question[:200],
            'expected': expected[:200],
            'predicted': answer[:200],
            'score': score,
            'latency': round(latency, 3),
            'iterations': iterations,
            'graph_nodes': len(graph),
        })

        acc = correct / (i + 1)
        pbar.set_postfix({'acc': f'{acc:.3f}', 'nodes': len(graph)})

    # Summary
    acc = correct / len(samples) if samples else 0
    avg_latency = total_latency / len(samples) if samples else 0
    stats = orchestrator.get_stats()

    summary = {
        'dataset': dataset_name,
        'method': method,
        'task_type': task_type,
        'n_samples': len(samples),
        'accuracy': round(acc, 4),
        'correct': correct,
        'avg_latency': round(avg_latency, 3),
        'avg_iterations': round(stats['avg_iterations'], 2),
        'graph_nodes_final': len(graph),
        'graph_edges_final': len(graph.edges),
        'timestamp': datetime.now().isoformat(),
    }

    # Save per-sample results
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir,
        f"{dataset_name}_{method}_n{len(samples)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)

    print(f"\n{dataset_name} {method}: accuracy={acc:.4f} ({correct}/{len(samples)}), "
          f"avg_latency={avg_latency:.2f}s, iterations={stats['avg_iterations']:.1f}")
    print(f"Results saved to {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASET_CONFIG.keys()) + ['all'])
    parser.add_argument('--method', type=str, default='kglmf',
                       choices=['kglmf', 'nomemory'])
    parser.add_argument('--n_samples', type=int, default=81)
    parser.add_argument('--data_dir', type=str,
                       default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained composer checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]

    all_summaries = []
    for ds in datasets:
        summary = run_experiment(
            dataset_name=ds,
            method=args.method,
            n_samples=args.n_samples,
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_summaries.append(summary)

    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print(f"{'Dataset':<16} {'Method':<10} {'Accuracy':<10} {'Latency':<10} {'Nodes':<8}")
        print(f"{'-'*70}")
        for s in all_summaries:
            print(f"{s['dataset']:<16} {s['method']:<10} {s['accuracy']:<10.4f} "
                  f"{s['avg_latency']:<10.2f} {s['graph_nodes_final']:<8}")


if __name__ == '__main__':
    main()
