#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextMemory (BM25) Baseline — Llama3.1 8B via Ollama.
Runs on all datasets with N samples, saves results to merge with fair comparison.

Usage:
    cd D:/KGLM/KGLMF
    /d/anaconda3/envs/kglmf/python experiments/scripts/run_textmemory_llama.py \
        --dataset all --n_samples 50
"""

import os, sys, json, time, random, argparse, re
import requests
import numpy as np
from datetime import datetime
from tqdm import tqdm
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


class OllamaClient:
    def __init__(self, model='llama3.1:8b', host='http://localhost:11434'):
        self.model = model; self.host = host; self.last_tokens = {}

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 history=None, latent_memory=None) -> str:
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


class TextMemoryBM25:
    """Real BM25 text memory for Llama experiments."""

    def __init__(self, max_memory: int = 100):
        self.max_memory = max_memory
        self.documents = []
        self.tokenized = []
        self.bm25 = None

    def _tokenize(self, text: str):
        return re.findall(r'\w+', text.lower())

    def add(self, query: str, response: str):
        doc = f"Q: {query}\nA: {response}"
        self.documents.append(doc)
        self.tokenized.append(self._tokenize(doc))
        if len(self.documents) > self.max_memory:
            self.documents.pop(0); self.tokenized.pop(0)
        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, k: int = 5):
        if self.bm25 is None or not self.documents:
            return []
        tok = self._tokenize(query)
        scores = self.bm25.get_scores(tok)
        top = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top if scores[i] > 0]

    def get_context(self, query: str, k: int = 3) -> str:
        docs = self.retrieve(query, k)
        if not docs: return ""
        return "Relevant past interactions:\n" + "\n---\n".join(docs) + "\n\n"


# ═══════════════════════════════════════════════════════════
# DATASETS & EVALUATION
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
    if not os.path.exists(fpath): return []
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) > n:
        random.seed(seed); data = random.sample(data, n)
    samples = []
    for item in data:
        q = item.get('question') or item.get('prompt') or item.get('input', '')
        a = item.get('answer') or item.get('target') or item.get('output', '')
        if q and str(a): samples.append({'question': str(q), 'answer': str(a)})
    return samples

def evaluate(pred: str, expected: str, metric: str) -> float:
    p, e = pred.strip().lower(), expected.strip().lower()
    yes_w = {'yes', 'true', 'correct', 'right'}
    no_w = {'no', 'false', 'incorrect', 'wrong'}
    p_yes = any(p.startswith(w) for w in yes_w)
    p_no = any(p.startswith(w) for w in no_w)
    e_yes = e in yes_w or e.startswith('yes') or e.startswith('true')
    e_no = e in no_w or e.startswith('no') or e.startswith('false')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    api = OllamaClient()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    os.makedirs('results_llama', exist_ok=True)

    for ds in datasets:
        config = DATASET_CONFIG[ds]
        samples = load_dataset(ds, args.n_samples, args.seed)
        if not samples: continue
        n = len(samples)
        metric = config['metric']

        print(f"\n{'='*50}")
        print(f"{ds} | TextMemory | N={n} | Llama3.1:8B")
        print(f"{'='*50}")

        memory = TextMemoryBM25(max_memory=100)
        results = []
        correct = 0
        total_lat = 0.0
        total_tok = 0

        TEXTMEM_PROMPT = "Answer the question directly and concisely. Use the provided context if relevant."

        for s in tqdm(samples, desc=f"  {ds}"):
            q, expected = s['question'], s['answer']
            context = memory.get_context(q, k=3)

            t0 = time.time()
            full_q = f"{context}Question: {q}" if context else q
            resp = api.generate(TEXTMEM_PROMPT, full_q)
            elapsed = time.time() - t0
            total_lat += elapsed
            tok = api.last_tokens
            total_tok += tok['total']

            score = evaluate(resp, expected, metric)
            if score >= 0.5: correct += 1

            results.append({
                'question': q[:150], 'expected': expected[:150],
                'predicted': resp[:150], 'score': score,
                'latency': elapsed, 'tokens': tok['total'],
                'context_len': len(context)
            })

            memory.add(q, resp)

        acc = correct / n
        avg_lat = total_lat / n
        avg_tok = total_tok / n
        eff = (acc / max(avg_tok, 1)) * 1000

        print(f"  textmemory acc={acc:.4f} lat={avg_lat:.1f}s tok={avg_tok:.0f} eff={eff:.4f}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f'results_llama/{ds}_textmemory_n{n}_{ts}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'dataset': ds, 'method': 'textmemory', 'n': n,
                    'accuracy': acc, 'avg_latency': avg_lat, 'avg_tokens': avg_tok,
                    'efficiency_score': eff,
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
