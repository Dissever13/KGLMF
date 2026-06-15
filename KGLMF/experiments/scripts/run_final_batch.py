"""Final batch: more Qwen samples + more multi-turn + more baselines"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.client import DeepSeekClient
from src.api.ollama_client import OllamaClient
from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SYS = "Answer directly in one sentence. No explanation."

def load_data(name, n):
    for ext in ['_real.json', '.json']:
        path = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)[:n]
    return []

def run_single(api, questions):
    correct, lats = 0, []
    for item in questions:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        t0 = time.time()
        try: resp = api.generate(SYS, q, max_tokens=512)
        except: resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        correct += score; lats.append(lat)
    n = len(questions)
    return {'accuracy': round(correct/n, 4) if n else 0, 'avg_latency': round(sum(lats)/n, 2) if n else 0, 'n': n}

def run_kg_lmf(api, questions):
    correct, lats, nodes = 0, [], []
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0])
    evolution = GraphEvolution(merge_threshold=0.85)
    orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector, evolution=evolution, graph=graph, api_client=api, d_model=384)

    for item in questions:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        orch.agents = {}
        a = orch._get_or_create_agent('critic')
        a.system_prompt = SYS
        t0 = time.time()
        for retry in range(2):
            try:
                resp = orch.run_round(q, ['critic'])
                r = list(resp.values())[-1] if resp else ''
                break
            except:
                r = ''; time.sleep(1)
        lat = time.time() - t0
        score = 1.0 if gt.lower() in r.lower() else 0.0
        correct += score; lats.append(lat); nodes.append(len(graph))
    n = len(questions)
    return {'accuracy': round(correct/n, 4) if n else 0, 'avg_latency': round(sum(lats)/n, 2) if n else 0, 'n': n, 'final_nodes': nodes[-1] if nodes else 0}

# Expanded multi-turn: 15 topics x 5 turns = 75 questions
MULTI_TURN_EXTRA = [
    ("sports", [
        ("In what sport is a touchdown scored?", "football"),
        ("How many points is a touchdown worth in American football?", "6"),
        ("What is the line called that a team must cross for a touchdown?", "goal line"),
        ("How many players are on the field per team in American football?", "11"),
        ("What is the championship game of the NFL called?", "Super Bowl"),
    ]),
    ("economics", [
        ("What is the currency of the European Union?", "euro"),
        ("How many EU countries use the euro?", "20"),
        ("What is the central bank of the eurozone called?", "European Central Bank"),
        ("In what city is the ECB headquartered?", "Frankfurt"),
        ("What year was the euro introduced as a physical currency?", "2002"),
    ]),
    ("physics", [
        ("What is the SI unit of force?", "newton"),
        ("Who formulated the three laws of motion?", "Newton"),
        ("What is the acceleration due to gravity on Earth approximately?", "9.8"),
        ("What is the speed of light in vacuum?", "299792458"),
        ("What theory describes gravity as curvature of spacetime?", "general relativity"),
    ]),
    ("medicine", [
        ("What vitamin is produced when skin is exposed to sunlight?", "D"),
        ("What disease is caused by vitamin D deficiency in children?", "rickets"),
        ("What organ stores vitamin D in the body?", "liver"),
        ("What is the recommended daily intake of vitamin D in IU?", "600"),
        ("What mineral does vitamin D help absorb?", "calcium"),
    ]),
    ("technology", [
        ("What company developed the iPhone?", "Apple"),
        ("Who was the CEO of Apple when the iPhone was launched?", "Steve Jobs"),
        ("In what year was the first iPhone released?", "2007"),
        ("What operating system runs on iPhones?", "iOS"),
        ("What programming language is primarily used for iOS apps?", "Swift"),
    ]),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_qwen', type=int, default=50)
    parser.add_argument('--skip_qwen', action='store_true')
    parser.add_argument('--skip_multiturn', action='store_true')
    args = parser.parse_args()

    all_results = {}

    # 1. Qwen3.5: NoMemory + KG-LMF on 50 samples x 3 datasets
    if not args.skip_qwen:
        print("="*60)
        print("QWEN3.5 BACKBONE (N=" + str(args.n_qwen) + ")")
        print("="*60)
        try:
            qwen_api = OllamaClient(model='qwen3.5:9b', max_tokens=4096)
            test = qwen_api.generate(SYS, 'Say hi')
            print("Qwen connected: " + str(bool(test)))
        except Exception as e:
            print("Qwen error: " + str(e))
            qwen_api = None

        if qwen_api:
            for ds in ['TriviaQA', 'StrategyQA', 'PopQA']:
                data = load_data(ds, args.n_qwen)
                if not data: continue
                print("\n" + ds + " (" + str(len(data)) + " samples):")

                # NoMemory
                key_nm = ds + '/Qwen-NoMemory'
                print("  NoMemory...", end=' ', flush=True)
                r = run_single(qwen_api, data)
                r['method'] = 'NoMemory'; r['dataset'] = ds; r['backbone'] = 'Qwen3.5-9B'
                all_results[key_nm] = r
                print("acc=" + str(r['accuracy']) + " lat=" + str(r['avg_latency']) + "s")

                # KG-LMF
                key_kg = ds + '/Qwen-KG-LMF'
                print("  KG-LMF...", end=' ', flush=True)
                r = run_kg_lmf(qwen_api, data)
                r['method'] = 'KG-LMF'; r['dataset'] = ds; r['backbone'] = 'Qwen3.5-9B'
                all_results[key_kg] = r
                print("acc=" + str(r['accuracy']) + " lat=" + str(r['avg_latency']) + "s nodes=" + str(r.get('final_nodes', '?')))

                # Save incrementally
                with open(os.path.join(RESULTS_DIR, 'qwen_extended.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)

    # 2. DeepSeek: Multi-turn extended (15 topics x 5 turns)
    if not args.skip_multiturn:
        print("\n" + "="*60)
        print("MULTI-TURN EXTENDED (DeepSeek V4 Pro)")
        print("="*60)
        ds_api = DeepSeekClient(max_tokens=512)

        all_topics = MULTI_TURN_EXTRA  # 5 extra topics
        for topic_name, turns in all_topics:
            print("\n" + topic_name + ":")

            # KG-LMF
            kg_c, kg_l, kg_nodes = 0, [], []
            graph = LatentReasoningGraph(d_model=384)
            composer = MemoryComposer(d_model=384, n_learnable_queries=16)
            router = GraphRouter(d_model=384)
            tts = AdaptiveTTSController(budget_min=4, budget_max=16)
            injector = LatentInjector(d_model=384, injection_layers=[0])
            evolution = GraphEvolution(merge_threshold=0.85)
            orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector, evolution=evolution, graph=graph, api_client=ds_api, d_model=384)

            for turn, (q, gt) in enumerate(turns):
                orch.agents = {}
                a = orch._get_or_create_agent('critic')
                a.system_prompt = SYS
                t0 = time.time()
                try:
                    resp = orch.run_round(q, ['critic'])
                    r = list(resp.values())[-1] if resp else ''
                except:
                    r = ''
                lat = time.time() - t0
                score = 1.0 if gt.lower() in r.lower() else 0.0
                kg_c += score; kg_l.append(lat); kg_nodes.append(len(graph))

            kg_acc = kg_c / len(turns)
            print("  KG-LMF: acc=" + str(round(kg_acc, 4)) + " nodes=" + str(kg_nodes))

            # NoMemory
            nm_c = 0
            for turn, (q, gt) in enumerate(turns):
                t0 = time.time()
                try: r = ds_api.generate(SYS, q, max_tokens=256)
                except: r = ''
                lat = time.time() - t0
                score = 1.0 if gt.lower() in r.lower() else 0.0
                nm_c += score
            nm_acc = nm_c / len(turns)
            print("  NoMemory: acc=" + str(round(nm_acc, 4)))

    print("\nDone!")
    print("Results in " + RESULTS_DIR)

if __name__ == '__main__':
    main()
