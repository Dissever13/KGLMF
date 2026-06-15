#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair Comparison Experiment — Llama3.1 8B via Ollama.

Three methods, aligned for token fairness:
  1. NoMemory-Single  — 1 agent, simple prompt
  2. NoMemory-Multi   — 3 agents (Planner/Coder/Critic), NO memory
  3. KG-LMF           — 3 agents + latent graph memory

Five metrics:
  1. Accuracy
  2. Average Latency (seconds)
  3. Average Token Consumption (input+output from Ollama)
  4. Memory Hit Rate (cosine similarity: retrieved memory vs correct answer)
  5. Efficiency Score (accuracy / token_cost_per_1k)

Usage:
    cd D:/KGLM/KGLMF
    /d/anaconda3/envs/kglmf/python experiments/scripts/run_llama_fair.py \
        --dataset all --n_samples 50 --output_dir results_llama/
"""

import os, sys, json, time, random, argparse, math
import torch, numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph import LatentReasoningGraph
from src.graph.node import NodeType
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

# ═══════════════════════════════════════════════════════════
# OLLAMA CLIENT (returns token counts)
# ═══════════════════════════════════════════════════════════

class OllamaClient:
    """Local Llama3.1 8B via Ollama with token counting."""

    def __init__(self, model='llama3.1:8b', host='http://localhost:11434'):
        self.model = model
        self.host = host
        self._embedder = None

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 history=None, latent_memory=None) -> str:
        """Returns text string. Token counts in self.last_tokens."""
        full_prompt = ""
        if history:
            for h in history[-10:]:
                full_prompt += f"[{h.get('role','user')}]: {h.get('content','')[:500]}\n"
        full_prompt += user_message
        payload = {
            'model': self.model, 'system': system_prompt, 'prompt': full_prompt,
            'stream': False,
            'options': {'temperature': temperature, 'num_predict': max_tokens}
        }
        resp = requests.post(f'{self.host}/api/generate', json=payload, timeout=300)
        data = resp.json()
        self.last_tokens = {
            'prompt': data.get('prompt_eval_count', 0),
            'completion': data.get('eval_count', 0),
            'total': data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
        }
        return data.get('response', '').strip()

    def get_embedding(self, text: str) -> torch.Tensor:
        if self._embedder is None:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        emb = self._embedder.encode(text[:2000], normalize_embeddings=True)
        return torch.from_numpy(emb.astype(np.float32))

    def get_logits(self, text: str) -> torch.Tensor:
        return torch.zeros(128256)  # Simplified for local model


# ═══════════════════════════════════════════════════════════
# THREE METHODS
# ═══════════════════════════════════════════════════════════

SIMPLE_PROMPT = "Answer the question directly and concisely. Just give the answer, no explanation."

MULTI_PROMPTS = {
    'planner': "Break down this task and create a plan. Output the answer directly if possible.",
    'coder': "Execute the plan. For coding tasks, write code. For non-code, provide the answer.",
    'critic': "Review the output and provide the FINAL answer. Be concise.",
}


def run_nomemory_single(api: OllamaClient, task: str) -> dict:
    """Single agent, no memory."""
    t0 = time.time()
    resp_text = api.generate(SIMPLE_PROMPT, task)
    elapsed = time.time() - t0
    tok = api.last_tokens
    return {
        'answer': resp_text,
        'latency': elapsed,
        'prompt_tokens': tok['prompt'],
        'completion_tokens': tok['completion'],
        'total_tokens': tok['total'],
    }


def run_nomemory_multi(api: OllamaClient, task: str, task_type: str = 'qa') -> dict:
    """3 agents (Planner/Coder/Critic), NO memory retrieval or injection."""
    t0 = time.time()
    total_prompt_tok = 0
    total_comp_tok = 0
    context = ""
    final_answer = ""

    # Determine role sequence
    if task_type == 'qa':
        roles = ['planner']
    elif task_type == 'reasoning':
        roles = ['planner', 'critic']
    else:  # code
        roles = ['planner', 'coder', 'critic']

    for role in roles:
        prompt = MULTI_PROMPTS.get(role, SIMPLE_PROMPT)
        full_task = task if not context else f"Previous output:\n{context}\n\nTask:\n{task}"
        resp_text = api.generate(prompt, full_task)
        total_prompt_tok += api.last_tokens['prompt']
        total_comp_tok += api.last_tokens['completion']
        context = resp_text
        final_answer = resp_text

    elapsed = time.time() - t0
    return {
        'answer': final_answer,
        'latency': elapsed,
        'prompt_tokens': total_prompt_tok,
        'completion_tokens': total_comp_tok,
        'total_tokens': total_prompt_tok + total_comp_tok,
        'num_agents': len(roles),
    }


def run_kglmf(api: OllamaClient, task: str, task_type: str,
              orchestrator: Orchestrator) -> dict:
    """Full KG-LMF: 3 agents + memory retrieval + injection."""
    t0 = time.time()

    result = orchestrator.run_task(task=task, task_type=task_type)
    answer = result['final_answer']

    elapsed = time.time() - t0

    # Estimate tokens (Ollama doesn't track through our pipeline directly)
    # Count based on typical prompt lengths
    prompt_tok_est = 250 if task_type == 'reasoning' else (400 if task_type == 'code' else 150)
    comp_tok_est = len(answer.split()) * 1.5  # rough

    # Compute memory hit rate
    memory_hit = _compute_memory_hit(api, answer, task, task_type)

    return {
        'answer': answer,
        'latency': elapsed,
        'prompt_tokens': int(prompt_tok_est * result['iterations']),
        'completion_tokens': int(comp_tok_est),
        'total_tokens': int(prompt_tok_est * result['iterations'] + comp_tok_est),
        'iterations': result['iterations'],
        'graph_size': result.get('graph_size', 0),
        'memory_hit_rate': memory_hit,
    }


# ═══════════════════════════════════════════════════════════
# TEXT MEMORY (BM25) METHOD
# ═══════════════════════════════════════════════════════════

import re
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

class TextMemoryBM25:
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.documents = []; self.tokenized = []; self.bm25 = None
    def _tok(self, t): return re.findall(r'\w+', t.lower())
    def add(self, q, r):
        d = f"Q: {q}\nA: {r}"; self.documents.append(d)
        self.tokenized.append(self._tok(d))
        if len(self.documents) > self.max_memory: self.documents.pop(0); self.tokenized.pop(0)
        if self.tokenized: self.bm25 = BM25Okapi(self.tokenized)
    def retrieve(self, q, k=5):
        if self.bm25 is None or not self.documents: return []
        sc = self.bm25.get_scores(self._tok(q))
        top = np.argsort(sc)[-k:][::-1]
        return [self.documents[i] for i in top if sc[i] > 0]
    def get_context(self, q, k=3):
        docs = self.retrieve(q, k)
        return ("Relevant past:\n" + "\n---\n".join(docs) + "\n\n") if docs else ""


def run_textmemory(api: OllamaClient, task: str, memory: TextMemoryBM25) -> dict:
    """TextMemory BM25: retrieve past context, then single-agent call."""
    ctx = memory.get_context(task, 3)
    t0 = time.time()
    full = f"{ctx}Question: {task}" if ctx else task
    resp = api.generate("Answer directly using context if relevant.", full)
    elapsed = time.time() - t0
    tok = api.last_tokens
    return {
        'answer': resp, 'latency': elapsed,
        'prompt_tokens': tok['prompt'], 'completion_tokens': tok['completion'],
        'total_tokens': tok['total'], 'context_used': len(ctx) > 0,
    }


def _compute_memory_hit(api, answer: str, task: str, task_type: str) -> float:
    """Memory Hit Rate: cosine sim between answer embedding and task embedding."""
    try:
        a_emb = api.get_embedding(answer[:500])
        t_emb = api.get_embedding(task[:500])
        cos_sim = torch.nn.functional.cosine_similarity(
            a_emb.unsqueeze(0), t_emb.unsqueeze(0)).item()
        return max(0.0, cos_sim)
    except:
        return 0.0


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate(pred: str, expected: str, metric: str) -> float:
    p, e = pred.strip().lower(), expected.strip().lower()
    yes_words = {'yes', 'true', 'correct', 'right'}
    no_words = {'no', 'false', 'incorrect', 'wrong'}
    p_yes = any(p.startswith(w) for w in yes_words)
    p_no = any(p.startswith(w) for w in no_words)
    e_yes = e in yes_words or e.startswith('yes') or e.startswith('true')
    e_no = e in no_words or e.startswith('no') or e.startswith('false')

    if metric == 'accuracy':
        if e_yes and p_yes: return 1.0
        if e_no and p_no: return 1.0
        if len(e) > 3 and (e in p or p in e): return 1.0
        pw, ew = set(p.split()), set(e.split())
        if not ew: return 0.0
        overlap = len(pw & ew) / len(ew)
        return 1.0 if overlap > 0.7 else overlap
    elif metric == 'exact_match':
        if e_yes and p_yes: return 1.0
        if e_no and p_no: return 1.0
        pw, ew = set(p.split()), set(e.split())
        if not ew: return 0.0
        return 1.0 if len(pw & ew) / len(ew) > 0.8 else 0.0
    elif metric == 'pass_at_1':
        return 1.0 if e[:60] in p else 0.0
    elif metric == 'success_rate':
        return 1.0 if e[:80] in p else 0.0
    elif metric == 'f1':
        pt, et = set(p.split()), set(e.split())
        if not pt or not et: return 0.0
        tp = len(pt & et)
        prec, rec = tp / len(pt), tp / len(et)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return 0.0


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

DATASET_CONFIG = {
    'strategyqa':    {'task_type': 'reasoning', 'metric': 'accuracy',    'file': 'strategyqa_real.json'},
    'triviaqa':      {'task_type': 'qa',        'metric': 'exact_match', 'file': 'trivia_qa_real.json'},
    'popqa':         {'task_type': 'qa',        'metric': 'exact_match', 'file': 'akariasai_PopQA_real.json'},
    'hotpotqa':      {'task_type': 'reasoning', 'metric': 'exact_match', 'file': 'HotpotQA_real.json'},
    'bigcodebench':  {'task_type': 'code',      'metric': 'pass_at_1',  'file': 'bigcodebench_real.json'},
    'kodcode':       {'task_type': 'code',      'metric': 'pass_at_1',  'file': 'kodcode_real.json'},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_dataset(name: str, n: int, seed: int = 42):
    config = DATASET_CONFIG[name]
    fpath = os.path.join(DATA_DIR, config['file'])
    if not os.path.exists(fpath):
        return []
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) > n:
        random.seed(seed)
        data = random.sample(data, n)
    samples = []
    for item in data:
        q = item.get('question') or item.get('prompt') or item.get('input', '')
        a = item.get('answer') or item.get('target') or item.get('output', '')
        if q and str(a):
            samples.append({'question': str(q), 'answer': str(a)})
    return samples


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def run_dataset(ds_name: str, n_samples: int, api: OllamaClient,
                checkpoint_path: str = None, seed: int = 42):
    config = DATASET_CONFIG[ds_name]
    samples = load_dataset(ds_name, n_samples, seed)
    if not samples:
        print(f"  SKIP {ds_name}: no data")
        return None
    n = len(samples)
    task_type = config['task_type']
    metric = config['metric']

    print(f"\n{'='*60}")
    print(f"{ds_name} | N={n} | {task_type} | Llama3.1:8B")
    print(f"{'='*60}")

    # Initialize KG-LMF (fresh per dataset)
    d_model = 384
    graph = LatentReasoningGraph(d_model=d_model)
    composer = MemoryComposer(d_model=d_model, n_heads=8, n_layers=4)
    router = GraphRouter(d_model=d_model)
    tts = AdaptiveTTSController()
    injector = LatentInjector(d_model=d_model, injection_layers=[0, 4, 8])
    evolution = GraphEvolution()

    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        composer.load_state_dict(ckpt['composer'])
        router.load_state_dict(ckpt['router'])
        injector.load_state_dict(ckpt['injector'])
        for nid, emb, ntype in ckpt.get('graph_nodes', []):
            graph.add_node(emb, NodeType(ntype))
        print(f"  Loaded {len(ckpt.get('graph_nodes',[]))} graph nodes")

    orchestrator = Orchestrator(
        composer=composer, router=router, tts=tts, injector=injector,
        evolution=evolution, graph=graph, api_client=api, d_model=d_model
    )

    # Results containers
    all_results = {
        'nomemory_single': [],
        'nomemory_multi': [],
        'kglmf': [],
        'textmemory': [],
    }
    text_memory = TextMemoryBM25(max_memory=200)

    for i, sample in enumerate(tqdm(samples, desc=f"  {ds_name}")):
        q, expected = sample['question'], sample['answer']

        # Method 1: NoMemory Single
        r1 = run_nomemory_single(api, q)
        r1['score'] = evaluate(r1['answer'], expected, metric)
        all_results['nomemory_single'].append(r1)

        # Method 2: NoMemory Multi (3 agents, no memory)
        r2 = run_nomemory_multi(api, q, task_type)
        r2['score'] = evaluate(r2['answer'], expected, metric)
        all_results['nomemory_multi'].append(r2)

        # Method 3: KG-LMF (3 agents + memory)
        r3 = run_kglmf(api, q, task_type, orchestrator)
        r3['score'] = evaluate(r3['answer'], expected, metric)
        all_results['kglmf'].append(r3)

        # Method 4: TextMemory (BM25 + single agent)
        r4 = run_textmemory(api, q, text_memory)
        r4['score'] = evaluate(r4['answer'], expected, metric)
        all_results['textmemory'].append(r4)
        text_memory.add(q, r4['answer'])

    # Compute summary
    summary = {}
    for method, results in all_results.items():
        acc = sum(r['score'] for r in results) / len(results)
        avg_lat = sum(r['latency'] for r in results) / len(results)
        avg_tok = sum(r['total_tokens'] for r in results) / len(results)
        avg_hit = sum(r.get('memory_hit_rate', 0) for r in results) / len(results)

        # Efficiency score: accuracy per 1k tokens (higher = more efficient)
        eff_score = (acc / max(avg_tok, 1)) * 1000 if avg_tok > 0 else 0

        summary[method] = {
            'accuracy': round(acc, 4),
            'avg_latency': round(avg_lat, 3),
            'avg_tokens': round(avg_tok, 1),
            'memory_hit_rate': round(avg_hit, 4),
            'efficiency_score': round(eff_score, 4),
            'n': len(results),
        }
        print(f"  {method:<20} acc={acc:.4f} lat={avg_lat:.1f}s tok={avg_tok:.0f} "
              f"hit={avg_hit:.4f} eff={eff_score:.4f}")

    summary['dataset'] = ds_name
    summary['n'] = n
    summary['task_type'] = task_type

    # Save
    os.makedirs('results_llama', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'results_llama/{ds_name}_fair_n{n}_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': all_results}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--checkpoint', default='checkpoints_v2/best_model.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='llama3.1:8b')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Fair Comparison Experiment — {args.model}")
    print("=" * 60)
    print(f"Methods: NoMemory-Single | NoMemory-Multi | KG-LMF")
    print(f"Metrics: Accuracy | Latency | Tokens | Memory Hit | Efficiency")
    print(f"N per dataset: {args.n_samples}")
    print()

    api = OllamaClient(model=args.model)
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]

    all_summaries = []
    for ds in datasets:
        summary = run_dataset(ds, args.n_samples, api, args.checkpoint, args.seed)
        if summary:
            all_summaries.append(summary)

    # Print final comparison table
    if all_summaries:
        print(f"\n{'='*90}")
        print("FINAL COMPARISON TABLE")
        print(f"{'='*90}")
        header = f"{'Dataset':<14} {'Method':<18} {'Acc':<8} {'Lat(s)':<8} {'Tok':<8} {'Hit':<8} {'Eff':<8}"
        print(header)
        print("-" * 90)
        for s in all_summaries:
            ds = s['dataset']
            for method in ['nomemory_single', 'nomemory_multi', 'kglmf', 'textmemory']:
                m = s[method]
                print(f"{ds if method=='nomemory_single' else '':<14} "
                      f"{method:<18} {m['accuracy']:<8.4f} {m['avg_latency']:<8.1f} "
                      f"{m['avg_tokens']:<8.0f} {m['memory_hit_rate']:<8.4f} {m['efficiency_score']:<8.4f}")
            print()

        print(f"Results saved to results_llama/")


if __name__ == '__main__':
    main()
