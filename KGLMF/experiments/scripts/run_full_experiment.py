#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full KG-LMF Experiment Runner.
Runs NoMemory baseline and KG-LMF on all 8 datasets with consistent N samples.

Usage:
    cd D:/KGLM/KGLMF
    /d/anaconda3/envs/kglmf/python experiments/scripts/run_full_experiment.py \
        --method both --n_samples 81 --output_dir results/
"""

import os, sys, json, time, random, argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.api.client import DeepSeekClient
from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

DATASET_CONFIG = {
    'triviaqa':      {'task_type': 'qa',        'metric': 'exact_match', 'file': 'trivia_qa_real.json'},
    'strategyqa':    {'task_type': 'reasoning', 'metric': 'accuracy',    'file': 'strategyqa_real.json'},
    'popqa':         {'task_type': 'qa',        'metric': 'exact_match', 'file': 'akariasai_PopQA_real.json'},
    'kodcode':       {'task_type': 'code',      'metric': 'pass_at_1',  'file': 'kodcode_real.json'},
    'bigcodebench':  {'task_type': 'code',      'metric': 'pass_at_1',  'file': 'bigcodebench_real.json'},
    'pddl':          {'task_type': 'planning',  'metric': 'success_rate','file': 'pddl_real.json'},
    'hotpotqa':      {'task_type': 'reasoning', 'metric': 'exact_match', 'file': 'HotpotQA_real.json'},
    'coqa':          {'task_type': 'qa',        'metric': 'f1',          'file': 'CoQA_real.json'},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_dataset(name: str, n: int, seed: int = 42):
    config = DATASET_CONFIG[name]
    fpath = os.path.join(DATA_DIR, config['file'])
    if not os.path.exists(fpath):
        print(f"  WARNING: {fpath} not found, skipping {name}")
        return []
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) > n:
        random.seed(seed)
        data = random.sample(data, n)
    samples = []
    for item in data:
        q = item.get('question', item.get('prompt', item.get('input', item.get('query', ''))))
        a = item.get('answer', item.get('target', item.get('output', item.get('solution', ''))))
        if isinstance(a, list): a = ' '.join(str(x) for x in a)
        if q and str(a):
            samples.append({'question': str(q), 'answer': str(a)})
    return samples


def evaluate(pred: str, expected: str, metric: str) -> float:
    p = pred.strip().lower()
    e = expected.strip().lower()

    # Normalize yes/no <-> true/false for binary QA datasets
    yes_words = {'yes', 'true', 'correct', 'right'}
    no_words = {'no', 'false', 'incorrect', 'wrong'}
    p_is_yes = any(p.startswith(w) for w in yes_words)
    p_is_no = any(p.startswith(w) for w in no_words)
    e_is_yes = e in yes_words or e.startswith('yes') or e.startswith('true')
    e_is_no = e in no_words or e.startswith('no') or e.startswith('false')

    if metric == 'accuracy':
        # Binary yes/no matching (handles true<->yes, false<->no)
        if e_is_yes and p_is_yes:
            return 1.0
        if e_is_no and p_is_no:
            return 1.0
        # Substring match
        if len(e) > 3 and (e in p or p in e):
            return 1.0
        # Word overlap for longer answers
        pw = set(p.split()); ew = set(e.split())
        if not ew: return 0.0
        overlap = len(pw & ew) / len(ew)
        return 1.0 if overlap > 0.7 else overlap
    elif metric == 'exact_match':
        if e_is_yes and p_is_yes: return 1.0
        if e_is_no and p_is_no: return 1.0
        pw = set(p.split()); ew = set(e.split())
        if not ew: return 0.0
        return 1.0 if len(pw & ew) / len(ew) > 0.8 else 0.0
    elif metric == 'pass_at_1':
        return 1.0 if e[:60] in p else 0.0
    elif metric == 'success_rate':
        return 1.0 if e[:80] in p else 0.0
    elif metric == 'f1':
        pt = set(p.split()); et = set(e.split())
        if not pt or not et: return 0.0
        tp = len(pt & et)
        prec = tp / len(pt); rec = tp / len(et)
        return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return 0.0


def load_trained_weights(checkpoint_path, composer, router, injector, graph):
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        composer.load_state_dict(ckpt['composer'])
        router.load_state_dict(ckpt['router'])
        injector.load_state_dict(ckpt['injector'])
        from src.graph.node import NodeType
        for nid, emb, ntype in ckpt.get('graph_nodes', []):
            graph.add_node(emb, NodeType(ntype))
        print(f"  Loaded trained weights + {len(ckpt.get('graph_nodes',[]))} graph nodes")


def run_dataset(dataset_name: str, method: str, n_samples: int,
                api, checkpoint: str = None, seed: int = 42):
    config = DATASET_CONFIG[dataset_name]
    samples = load_dataset(dataset_name, n_samples, seed)
    if not samples:
        return None
    n = len(samples)
    print(f"\n{'='*55}")
    print(f"{dataset_name} | {method} | N={n} | {config['task_type']}")
    print(f"{'='*55}")

    results = []
    correct = 0
    total_time = 0.0

    # Init KG-LMF components (fresh per dataset)
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_heads=8, n_layers=4)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController()
    injector = LatentInjector(d_model=384, injection_layers=[0, 4, 8])
    evolution = GraphEvolution()

    if checkpoint and os.path.exists(checkpoint) and method == 'kglmf':
        load_trained_weights(checkpoint, composer, router, injector, graph)

    orchestrator = Orchestrator(
        composer=composer, router=router, tts=tts,
        injector=injector, evolution=evolution,
        graph=graph, api_client=api, d_model=384
    )

    pbar = tqdm(enumerate(samples), total=n, desc=f"  {dataset_name}")
    for i, sample in pbar:
        q, expected = sample['question'], sample['answer']
        t0 = time.time()

        if method == 'nomemory':
            answer = api.generate(
                "Answer the question directly and concisely. Output only the answer.",
                q).strip()
            iters = 1
        else:
            result = orchestrator.run_task(task=q, task_type=config['task_type'])
            answer = result['final_answer']
            iters = result['iterations']

        elapsed = time.time() - t0
        total_time += elapsed
        score = evaluate(answer, expected, config['metric'])
        if score >= 0.5:
            correct += 1

        results.append({
            'id': i, 'question': q[:200], 'expected': expected[:200],
            'predicted': answer[:200], 'score': score,
            'latency': round(elapsed, 3), 'iterations': iters,
        })
        pbar.set_postfix({'acc': f'{correct/(i+1):.3f}'})

    acc = correct / n
    avg_lat = total_time / n
    summary = {
        'dataset': dataset_name, 'method': method, 'n': n,
        'accuracy': round(acc, 4), 'correct': correct,
        'avg_latency': round(avg_lat, 3),
        'timestamp': datetime.now().isoformat(),
    }
    print(f"  => {dataset_name} {method}: {acc:.4f} ({correct}/{n}), {avg_lat:.1f}s avg")
    return {'summary': summary, 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='both', choices=['nomemory', 'kglmf', 'both'])
    parser.add_argument('--n_samples', type=int, default=81)
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--checkpoint', default='checkpoints_v2/best_model.pt')
    parser.add_argument('--output_dir', default='results/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = DeepSeekClient()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    methods = ['nomemory', 'kglmf'] if args.method == 'both' else [args.method]

    all_summaries = []
    for ds in datasets:
        for method in methods:
            result = run_dataset(ds, method, args.n_samples, api,
                                args.checkpoint, args.seed)
            if result is None:
                continue
            all_summaries.append(result['summary'])

            # Save per-dataset results
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"{ds}_{method}_n{result['summary']['n']}_{ts}.json"
            with open(os.path.join(args.output_dir, fname), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary table
    if len(all_summaries) > 0:
        print(f"\n{'='*70}")
        print(f"SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Dataset':<16} {'Method':<10} {'N':<5} {'Accuracy':<10} {'Latency':<10}")
        print(f"{'-'*70}")
        for s in all_summaries:
            print(f"{s['dataset']:<16} {s['method']:<10} {s['n']:<5} "
                  f"{s['accuracy']:<10.4f} {s['avg_latency']:<10.2f}")
        print(f"{'='*70}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
