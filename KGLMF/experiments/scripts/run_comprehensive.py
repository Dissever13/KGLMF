"""Comprehensive benchmark: download datasets, run ALL experiments, NEVER lose data."""
import sys, os, time, json, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.client import DeepSeekClient
from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator
from src.agents.prompts import get_prompts_for_task

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(RESULTS_DIR, 'comprehensive_results.json')

OPTIMAL_QA_PROMPT = "Answer the question directly in ONE concise sentence. For yes/no: start with Yes or No. No explanation. No thinking traces."
OPTIMAL_CODE_PROMPT = "Write correct working Python code. Output ONLY the code. No explanation."
OPTIMAL_PLAN_PROMPT = "Generate a valid PDDL plan with proper syntax. Output ONLY the plan."

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_results(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def download_real_dataset(name, n=101):
    """Download from HuggingFace or use curated"""
    path = os.path.join(DATA_DIR, f'{name}_real.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if len(data) >= n:
            return data[:n]

    try:
        from datasets import load_dataset
        if name == 'TriviaQA':
            ds = load_dataset('trivia_qa', 'rc.nocontext', split='validation', trust_remote_code=True)
            data = []
            for item in ds:
                ans = item.get('answer', {})
                if isinstance(ans, dict):
                    ans = ans.get('value', ans.get('text', ''))
                if ans:
                    data.append({'id': str(len(data)), 'question': item['question'], 'answer': str(ans).strip(), 'type': 'qa', 'tokens': len(item['question'].split())})
                if len(data) >= n: break
            random.seed(42); random.shuffle(data)
            with open(path, 'w') as f: json.dump(data, f)
            return data[:n]
        elif name == 'StrategyQA':
            ds = load_dataset('wics/strategy-qa', split='test', trust_remote_code=True)
            data = []
            for item in ds:
                data.append({'id': str(len(data)), 'question': item['question'], 'answer': str(item['answer']).strip().lower(), 'type': 'reasoning', 'tokens': len(item['question'].split())})
                if len(data) >= n: break
            random.seed(42); random.shuffle(data)
            with open(path, 'w') as f: json.dump(data, f)
            return data[:n]
        elif name == 'PopQA':
            ds = load_dataset('akariasai/PopQA', split='test', trust_remote_code=True)
            data = []
            for item in ds:
                data.append({'id': str(item.get('id', len(data))), 'question': item['question'], 'answer': str(item.get('answer', item.get('possible_answer', ''))).strip(), 'type': 'qa', 'tokens': len(item['question'].split())})
                if len(data) >= n: break
            random.seed(42); random.shuffle(data)
            with open(path, 'w') as f: json.dump(data, f)
            return data[:n]
    except Exception as e:
        print(f'  HuggingFace download failed for {name}: {e}')

    # Fallback: load existing data
    fallback_path = os.path.join(DATA_DIR, f'{name}.json')
    if os.path.exists(fallback_path):
        with open(fallback_path) as f:
            return json.load(f)[:n]
    return []

def run_no_memory(api, dataset, questions, n_samples):
    results = []
    dtype = questions[0].get('type', 'qa') if questions else 'qa'
    sys_prompt = OPTIMAL_CODE_PROMPT if dtype == 'code' else (OPTIMAL_PLAN_PROMPT if dtype == 'planning' else OPTIMAL_QA_PROMPT)
    for item in questions[:n_samples]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        t0 = time.time()
        try:
            resp = api.generate(sys_prompt, q, max_tokens=512)
        except Exception as e:
            resp = f'[ERROR: {e}]'
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split()), 'pred': resp[:200]})
    return results

def run_text_memory(api, dataset, questions, n_samples):
    results = []
    memory_bank = []
    dtype = questions[0].get('type', 'qa') if questions else 'qa'
    sys_prompt = OPTIMAL_CODE_PROMPT if dtype == 'code' else (OPTIMAL_PLAN_PROMPT if dtype == 'planning' else OPTIMAL_QA_PROMPT)
    for item in questions[:n_samples]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        context = '\n'.join(memory_bank[-5:]) if memory_bank else ''
        augmented = f'Context:\n{context}\n\nQuestion: {q}' if context else q
        t0 = time.time()
        try:
            resp = api.generate(sys_prompt, augmented, max_tokens=512)
        except Exception as e:
            resp = f'[ERROR: {e}]'
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(augmented.split()), 'pred': resp[:200]})
        memory_bank.append(f'Q: {q}\nA: {resp[:300]}')
    return results

def run_kg_lmf(api, dataset, questions, n_samples, ablation=None):
    prompt_cfg = get_prompts_for_task('code' if dataset in ['KodCode','BigCodeBench'] else 'reasoning' if dataset == 'StrategyQA' else 'qa')
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=32, scaling_factor=2.0)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)

    if ablation == 'no-graph':
        graph = LatentReasoningGraph(d_model=384, max_nodes_per_type=10000)
    elif ablation == 'no-role':
        composer = MemoryComposer(d_model=384, n_learnable_queries=16, role_types=['generic'])
    elif ablation == 'no-tts':
        tts = AdaptiveTTSController(budget_min=16, budget_max=16)
    elif ablation == 'no-evolution':
        evolution = None
    elif ablation == 'no-compression':
        composer = MemoryComposer(d_model=384, n_learnable_queries=64)
    elif ablation == 'no-injection':
        injector = None

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    orch.agents = {}
    roles = prompt_cfg['roles']
    results = []

    for item in questions[:n_samples]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        for role in roles:
            orch._get_or_create_agent(role).system_prompt = prompt_cfg.get(role, OPTIMAL_QA_PROMPT)
        t0 = time.time()
        try:
            responses = orch.run_round(q, roles)
            resp = list(responses.values())[-1] if responses else ''
        except Exception as e:
            resp = f'[ERROR: {e}]'
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split()), 'nodes': len(graph), 'pred': resp[:200]})
    return results

def aggregate(results):
    acc = sum(r['score'] for r in results) / len(results) if results else 0
    avg_lat = sum(r['latency'] for r in results) / len(results) if results else 0
    avg_tokens = sum(r.get('tokens_in', 0) for r in results) / len(results) if results else 0
    return {'accuracy': round(acc, 4), 'avg_latency': round(avg_lat, 2), 'avg_tokens': round(avg_tokens, 1), 'n': len(results)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=['TriviaQA','StrategyQA','PopQA','KodCode','BigCodeBench','PDDL'])
    parser.add_argument('--methods', nargs='*', default=['NoMemory','TextMemory','KG-LMF','KG-LMF-no-graph','KG-LMF-no-role','KG-LMF-no-tts','KG-LMF-no-compression','KG-LMF-no-evolution'])
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()

    print('='*60)
    print('KG-LMF COMPREHENSIVE BENCHMARK')
    print(f'Datasets: {args.datasets}')
    print(f'Methods: {args.methods}')
    print(f'N per dataset: {args.n_samples}')
    print('='*60)

    all_results = load_results()

    for ds_name in args.datasets:
        print(f'\n### DATASET: {ds_name} ###')
        data = download_real_dataset(ds_name, n=args.n_samples)
        if not data:
            print(f'  No data for {ds_name}, skipping')
            continue
        print(f'  Loaded {len(data)} samples')

        for method in args.methods:
            key = f'{ds_name}/{method}'
            if key in all_results:
                print(f'  [{method}] Already done, skipping')
                continue

            api = DeepSeekClient(max_tokens=512)
            print(f'  [{method}] Running...', end=' ', flush=True)

            if method == 'NoMemory':
                res = run_no_memory(api, ds_name, data, args.n_samples)
            elif method == 'TextMemory':
                res = run_text_memory(api, ds_name, data, args.n_samples)
            elif method == 'KG-LMF':
                res = run_kg_lmf(api, ds_name, data, args.n_samples)
            elif method.startswith('KG-LMF-'):
                abl = method.replace('KG-LMF-', '')
                res = run_kg_lmf(api, ds_name, data, args.n_samples, ablation=abl)
            else:
                print('UNKNOWN')
                continue

            agg = aggregate(res)
            agg['method'] = method
            agg['dataset'] = ds_name
            all_results[key] = agg
            save_results(all_results)
            a = agg['accuracy']
            l = agg['avg_latency']
            t = agg['avg_tokens']
            print(f'acc={a:.4f}, lat={l:.1f}s, tok={t:.0f}')

    # Summary
    sep = '=' * 60
    print('')
    print(sep)
    n_res = len(all_results)
    print(f'COMPREHENSIVE RESULTS ({n_res} experiments)')
    print(sep)
    for k in sorted(all_results.keys()):
        r = all_results[k]
        acc = r['accuracy']
        lat = r['avg_latency']
        n = r['n']
        print(f'  {k:40s} acc={acc:.4f}  lat={lat:.1f}s  n={n}')
    print(f'\nSaved to {RESULTS_FILE}')

if __name__ == '__main__':
    main()
