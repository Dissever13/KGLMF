"""Run Qwen3.5:9b on ALL benchmark datasets with 81 samples."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.api.ollama_client import OllamaClient
from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

SYS_QA = "Answer directly in one sentence. No explanation."
SYS_CODE = "Write correct working Python code. Output ONLY the code."

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT = os.path.join(RESULTS_DIR, 'qwen_full_results.json')

def load_data(name, n):
    for ext in ['_real.json', '.json']:
        p = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)[:n]
    return []

def run_nomemory(api, questions, is_code=False):
    sys_p = SYS_CODE if is_code else SYS_QA
    correct, lats = 0, []
    for item in questions:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        t0 = time.time()
        try: r = api.generate(sys_p, q, max_tokens=512)
        except: r = ''
        lat = time.time() - t0
        s = 1.0 if gt.lower() in r.lower() else 0.0
        correct += s; lats.append(lat)
    n = len(questions)
    return {'accuracy': round(correct/n, 4) if n else 0, 'avg_latency': round(sum(lats)/n, 2) if n else 0, 'n': n}

def run_kg_lmf(api, questions, is_code=False):
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0])
    evolution = GraphEvolution(merge_threshold=0.85)
    orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector, evolution=evolution, graph=graph, api_client=api, d_model=384)

    correct, lats = 0, []
    for item in questions:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        orch.agents = {}
        a = orch._get_or_create_agent('critic')
        a.system_prompt = SYS_CODE if is_code else SYS_QA
        t0 = time.time()
        for retry in range(2):
            try:
                resp = orch.run_round(q, ['critic'])
                r = list(resp.values())[-1] if resp else ''
                break
            except: r = ''; time.sleep(1)
        lat = time.time() - t0
        s = 1.0 if gt.lower() in r.lower() else 0.0
        correct += s; lats.append(lat)
        time.sleep(0.2)
    n = len(questions)
    return {'accuracy': round(correct/n, 4) if n else 0, 'avg_latency': round(sum(lats)/n, 2) if n else 0, 'n': n, 'nodes': len(graph)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=81)
    parser.add_argument('--datasets', nargs='*', default=['TriviaQA','StrategyQA','PopQA','KodCode','BigCodeBench','PDDL','HotpotQA'])
    args = parser.parse_args()

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f: results = json.load(f)

    api = OllamaClient(model='qwen3.5:9b', max_tokens=4096)
    test = api.generate(SYS_QA, 'Hi')
    if not test:
        print('Qwen3.5 unavailable!')
        return
    print('Qwen3.5:9b connected')

    for ds in args.datasets:
        data = load_data(ds, args.n)
        if not data:
            print(f'{ds}: No data')
            continue
        dtype = data[0].get('type', 'qa') if data else 'qa'
        is_code = dtype == 'code'
        print(f'\n{ds} ({len(data)} samples, type={dtype}):')

        for method, fn in [('NoMemory', run_nomemory), ('KG-LMF', run_kg_lmf)]:
            key = f'{ds}/Qwen-{method}'
            if key in results and results[key].get('n', 0) >= args.n:
                r = results[key]
                print(f'  {method}: DONE acc={r["accuracy"]:.4f}')
                continue
            print(f'  {method}...', end=' ', flush=True)
            r = fn(api, data, is_code)
            r['method'] = method; r['dataset'] = ds; r['backbone'] = 'qwen3.5:9b'
            results[key] = r
            with open(OUTPUT, 'w') as f: json.dump(results, f, indent=2)
            print(f'acc={r["accuracy"]:.4f} lat={r["avg_latency"]:.1f}s')

    print('\n' + '='*70)
    print('QWEN3.5 FULL RESULTS')
    print('='*70)
    for ds in args.datasets:
        nm = results.get(f'{ds}/Qwen-NoMemory', {}).get('accuracy', 0)
        kg = results.get(f'{ds}/Qwen-KG-LMF', {}).get('accuracy', 0)
        delta = kg - nm
        best = 'KG' if delta > 0 else 'NM' if delta < 0 else '='
        print(f'  {ds:15s} NM={nm:.4f} KG={kg:.4f} Δ={delta:+.4f} [{best}]')
    print(f'\nSaved to {OUTPUT}')

if __name__ == '__main__':
    main()
