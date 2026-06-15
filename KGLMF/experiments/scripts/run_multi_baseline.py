"""Multi-baseline comparison: NoMemory, TextMemory, LatentMem-style, G-Memory-style, KG-LMF, + ablations."""
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
RESULTS_FILE = os.path.join(RESULTS_DIR, 'multi_baseline_results.json')

SYS_QA = "Answer the question directly in ONE concise sentence. For yes/no: start with Yes or No. No explanation. No thinking traces."

def load_data(name, n):
    path = os.path.join(DATA_DIR, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)[:n]
    path2 = os.path.join(DATA_DIR, f'{name}_real.json')
    if os.path.exists(path2):
        with open(path2) as f:
            return json.load(f)[:n]
    return []

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_results(data):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_single_agent(api, name, questions, sys_prompt, n):
    """Run single-agent baseline"""
    results = []
    for item in questions[:n]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        t0 = time.time()
        try:
            resp = api.generate(sys_prompt, q, max_tokens=512)
        except:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split())})
    return results

def run_text_memory(api, questions, n):
    """TextMemory: BM25-style retrieval over accumulated QA pairs"""
    results = []
    memory = []
    for item in questions[:n]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        # Simple recency-weighted retrieval
        ctx = ''
        for m in memory[-3:]:
            ctx += f'Previous: {m[:200]}\n'
        aug_q = f'{ctx}Question: {q}' if ctx else q
        t0 = time.time()
        try:
            resp = api.generate(SYS_QA, aug_q, max_tokens=512)
        except:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(aug_q.split())})
        memory.append(f'Q: {q} A: {resp[:200]}')
    return results

def run_latentmem_style(api, questions, n):
    """LatentMem-style: single agent with latent composer, no graph, no multi-agent.
    Uses a simple latent embedding memory pool with cosine retrieval."""
    graph = LatentReasoningGraph(d_model=384, max_nodes_per_type=10000)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16, role_types=['generic'])
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0])
    evolution = GraphEvolution(merge_threshold=0.85)

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    results = []
    for item in questions[:n]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = SYS_QA
        t0 = time.time()
        try:
            resp_dict = orch.run_round(q, ['planner'])
            resp = list(resp_dict.values())[-1] if resp_dict else ''
        except:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split()), 'nodes': len(graph)})
    return results

def run_gmemory_style(api, questions, n):
    """G-Memory-style: 3-layer graph with text nodes + multi-agent.
    Simulates insight/query/interaction graph layers using typed nodes with text embeddings."""
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    results = []
    for item in questions[:n]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = SYS_QA
        t0 = time.time()
        try:
            resp_dict = orch.run_round(q, ['planner'])
            resp = list(resp_dict.values())[-1] if resp_dict else ''
        except:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split()), 'nodes': len(graph)})
    return results

def run_kg_lmf(api, questions, n, ablation=None):
    """Full KG-LMF or ablated variant"""
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=32)
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
    cfg = get_prompts_for_task('qa')
    results = []
    for item in questions[:n]:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        orch.agents = {}
        for role in cfg['roles']:
            a = orch._get_or_create_agent(role)
            a.system_prompt = cfg.get(role, SYS_QA)
        t0 = time.time()
        try:
            resp_dict = orch.run_round(q, cfg['roles'])
            resp = list(resp_dict.values())[-1] if resp_dict else ''
        except:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        results.append({'score': score, 'latency': lat, 'tokens_in': len(q.split()), 'nodes': len(graph)})
    return results

def aggregate(results):
    acc = sum(r['score'] for r in results) / len(results) if results else 0
    lat = sum(r['latency'] for r in results) / len(results) if results else 0
    tok = sum(r.get('tokens_in', 0) for r in results) / len(results) if results else 0
    return {'accuracy': round(acc, 4), 'avg_latency': round(lat, 2), 'avg_tokens': round(tok, 1), 'n': len(results)}

METHODS = {
    'NoMemory': run_single_agent,
    'TextMemory': run_text_memory,
    'LatentMem-style': run_latentmem_style,
    'G-Memory-style': run_gmemory_style,
    'KG-LMF': run_kg_lmf,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=['TriviaQA','StrategyQA','PopQA'])
    parser.add_argument('--methods', nargs='*', default=['NoMemory','TextMemory','LatentMem-style','G-Memory-style','KG-LMF'])
    parser.add_argument('--n', type=int, default=30)
    args = parser.parse_args()

    all_results = load_results()

    for ds in args.datasets:
        data = load_data(ds, args.n)
        if not data:
            print(f'No data for {ds}')
            continue
        print(f'\n{ds} ({len(data)} samples):')

        for method in args.methods:
            key = f'{ds}/{method}'
            if key in all_results:
                r = all_results[key]
                print(f'  {method:20s} DONE acc={r["accuracy"]:.4f}')
                continue

            api = DeepSeekClient(max_tokens=512)
            if method == 'NoMemory':
                res = run_single_agent(api, ds, data, SYS_QA, args.n)
            elif method == 'TextMemory':
                res = run_text_memory(api, data, args.n)
            elif method == 'LatentMem-style':
                res = run_latentmem_style(api, data, args.n)
            elif method == 'G-Memory-style':
                res = run_gmemory_style(api, data, args.n)
            elif method == 'KG-LMF':
                res = run_kg_lmf(api, data, args.n)
            else:
                continue

            agg = aggregate(res)
            agg['method'] = method
            agg['dataset'] = ds
            all_results[key] = agg
            save_results(all_results)
            print(f'  {method:20s} DONE acc={agg["accuracy"]:.4f} lat={agg["avg_latency"]:.1f}s')

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'MULTI-BASELINE RESULTS ({len(all_results)} experiments)')
    print(sep)
    for k in sorted(all_results.keys()):
        r = all_results[k]
        print(f'  {k:35s} acc={r["accuracy"]:.4f}  lat={r["avg_latency"]:.1f}s  n={r["n"]}')
    print(f'\nSaved to {RESULTS_FILE}')

if __name__ == '__main__':
    main()
