"""Multi-turn v2: HotpotQA + referential chain + cross-task + PDDL fix"""
import sys, os, time, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.api.client import DeepSeekClient
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
random.seed(42)

def load_data(name, n):
    for ext in ['_real.json', '.json']:
        p = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)[:n]
    return []

# ================================================================
# 1. HOTPOTQA MULTI-HOP (true multi-step reasoning)
# ================================================================
def run_hotpotqa(api):
    """Run on HotpotQA: each question requires multi-hop reasoning across facts"""
    data = load_data('HotpotQA', 50)
    if not data:
        print('No HotpotQA data')
        return None, None

    # KG-LMF with graph memory
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=32)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)
    orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector,
                       evolution=evolution, graph=graph, api_client=api, d_model=384)

    kg_c, kg_l, kg_nodes = 0, [], []
    nm_c, nm_l = 0, []

    for i, item in enumerate(data):
        q = item['question']
        gt = str(item['answer'])
        q_type = item.get('type', 'multihop')

        # KG-LMF
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = "You are a reasoning specialist. For multi-hop questions, think through the steps and give the final answer. Answer concisely."
        t0 = time.time()
        try:
            resp = orch.run_round(q, ['planner'])
            kg_r = list(resp.values())[-1] if resp else ''
        except:
            kg_r = ''
        lat = time.time() - t0
        kg_s = 1.0 if gt.lower() in kg_r.lower() else 0.0
        kg_c += kg_s; kg_l.append(lat); kg_nodes.append(len(graph))

        # NoMemory
        t0 = time.time()
        try:
            nm_r = api.generate(SYS, q, max_tokens=512)
        except:
            nm_r = ''
        nm_lat = time.time() - t0
        nm_s = 1.0 if gt.lower() in nm_r.lower() else 0.0
        nm_c += nm_s; nm_l.append(nm_lat)

        if (i+1) % 10 == 0:
            kg_a = kg_c / (i+1); nm_a = nm_c / (i+1)
            print(f'  [{i+1}/{len(data)}] KG={kg_a:.3f} NM={nm_a:.3f} nodes={kg_nodes[-1]}')

    kg_acc = kg_c / len(data); nm_acc = nm_c / len(data)
    kg_lat = sum(kg_l)/len(kg_l); nm_lat = sum(nm_l)/len(nm_l)
    print(f'  HotpotQA: KG-LMF={kg_acc:.4f} (lat={kg_lat:.1f}s) NoMemory={nm_acc:.4f} (lat={nm_lat:.1f}s)')

    return {'KG-LMF': {'accuracy': round(kg_acc,4), 'avg_latency': round(kg_lat,2), 'n': len(data), 'final_nodes': kg_nodes[-1] if kg_nodes else 0},
            'NoMemory': {'accuracy': round(nm_acc,4), 'avg_latency': round(nm_lat,2), 'n': len(data)}}

# ================================================================
# 2. PDDL FIX: pre-populate graph with planning templates
# ================================================================
def run_pddl_with_templates(api):
    """Run PDDL with pre-populated graph templates to enable memory-based planning"""
    templates = [
        ("(pick-up A)\n(put-down A)", "blocksworld basic"),
        ("(stack A B)\n(unstack A B)", "blocksworld stack"),
        ("(load-truck P T)\n(drive-truck T A B)\n(unload-truck P T)", "logistics truck"),
        ("(fly-airplane P X Y)\n(load-airplane C P)\n(unload-airplane C P)", "logistics air"),
        ("(pick B R1)\n(move R1 R2)\n(drop B R2)", "gripper move"),
    ]

    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)
    orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector,
                       evolution=evolution, graph=graph, api_client=api, d_model=384)

    # Pre-populate graph with planning templates
    for template, desc in templates:
        emb = api.get_embedding(template)
        graph.add_node(emb, type('NT', (), {'ROLE': 0, 'TRAJECTORY': 1, 'CONTEXT': 2, 'CAPSULE': 3}).CAPSULE, role='planner', metadata={'template': desc})

    data = load_data('PDDL', 30)
    if not data:
        print('No PDDL data')
        return None

    correct = 0
    for item in data:
        q = item.get('prompt', item.get('question', ''))
        gt = str(item.get('answer', ''))
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = "Generate a valid PDDL plan. Use standard PDDL syntax with :action, :parameters, :precondition, :effect."
        try:
            resp = orch.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except:
            r = ''
        score = 1.0 if gt.lower() in r.lower() else 0.0
        correct += score

    acc = correct / len(data) if data else 0
    print(f'  PDDL with templates: acc={acc:.4f} ({correct}/{len(data)})')
    return {'PDDL_templates': {'accuracy': round(acc,4), 'n': len(data), 'nodes': len(graph)}}

# ================================================================
# 3. CROSS-TASK TRANSFER
# ================================================================
def run_cross_task(api):
    """Train graph on TriviaQA, test on PopQA (both QA but different knowledge domains)"""
    trivia = load_data('TriviaQA', 30)
    popqa = load_data('PopQA', 30)
    strategyqa = load_data('StrategyQA', 30)

    if not trivia or not popqa:
        print('No cross-task data')
        return None

    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)
    orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector,
                       evolution=evolution, graph=graph, api_client=api, d_model=384)

    # Phase 1: Train graph on TriviaQA
    print('  Phase 1: Training graph on TriviaQA...')
    for item in trivia:
        q = item['question']; gt = str(item['answer'])
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = SYS
        try:
            resp = orch.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except:
            r = ''
        # Write to graph
        emb = api.get_embedding(r[:200] if r else q)
        graph.add_node(emb, type('NT', (), {'ROLE': 0, 'TRAJECTORY': 1, 'CONTEXT': 2, 'CAPSULE': 3}).CAPSULE, role='planner')
    trivia_nodes = len(graph)
    print(f'    Graph after TriviaQA: {trivia_nodes} nodes')

    # Phase 2: Test on PopQA with pre-trained graph
    print('  Phase 2: Testing on PopQA (cross-task)...')
    correct = 0
    for item in popqa:
        q = item['question']; gt = str(item['answer'])
        orch.agents = {}
        a = orch._get_or_create_agent('planner')
        a.system_prompt = SYS
        try:
            resp = orch.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except:
            r = ''
        score = 1.0 if gt.lower() in r.lower() else 0.0
        correct += score
    cross_acc = correct / len(popqa)
    print(f'    Cross-task (TriviaQA->PopQA): acc={cross_acc:.4f}')

    # Baseline: PopQA from scratch
    graph2 = LatentReasoningGraph(d_model=384)
    composer2 = MemoryComposer(d_model=384, n_learnable_queries=16)
    router2 = GraphRouter(d_model=384)
    tts2 = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector2 = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution2 = GraphEvolution(merge_threshold=0.85)
    orch2 = Orchestrator(composer=composer2, router=router2, tts=tts2, injector=injector2,
                        evolution=evolution2, graph=graph2, api_client=api, d_model=384)

    correct2 = 0
    for item in popqa:
        q = item['question']; gt = str(item['answer'])
        orch2.agents = {}
        a = orch2._get_or_create_agent('planner')
        a.system_prompt = SYS
        try:
            resp = orch2.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except:
            r = ''
        score = 1.0 if gt.lower() in r.lower() else 0.0
        correct2 += score
    scratch_acc = correct2 / len(popqa)
    print(f'    From scratch (PopQA only): acc={scratch_acc:.4f}')

    # Also test TriviaQA -> StrategyQA
    graph3 = LatentReasoningGraph(d_model=384)
    composer3 = MemoryComposer(d_model=384, n_learnable_queries=16)
    router3 = GraphRouter(d_model=384)
    tts3 = AdaptiveTTSController(budget_min=4, budget_max=16)
    injector3 = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution3 = GraphEvolution(merge_threshold=0.85)
    orch3 = Orchestrator(composer=composer3, router=router3, tts=tts3, injector=injector3,
                        evolution=evolution3, graph=graph3, api_client=api, d_model=384)
    # Pre-train on trivia
    for item in trivia:
        q = item['question']; gt = str(item['answer'])
        orch3.agents = {}
        a = orch3._get_or_create_agent('planner'); a.system_prompt = SYS
        try:
            resp = orch3.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except: r = ''
        emb = api.get_embedding(r[:200] if r else q)
        graph3.add_node(emb, type('NT', (), {'ROLE': 0, 'TRAJECTORY': 1, 'CONTEXT': 2, 'CAPSULE': 3}).CAPSULE, role='planner')

    # Test on StrategyQA
    correct3 = 0
    for item in strategyqa:
        q = item['question']; gt = str(item['answer'])
        orch3.agents = {}
        a = orch3._get_or_create_agent('planner'); a.system_prompt = SYS
        try:
            resp = orch3.run_round(q, ['planner'])
            r = list(resp.values())[-1] if resp else ''
        except: r = ''
        score = 1.0 if gt.lower() in r.lower() else 0.0
        correct3 += score
    cross_sqa = correct3 / len(strategyqa)
    print(f'    Cross-task (TriviaQA->StrategyQA): acc={cross_sqa:.4f}')

    return {
        'cross_task': {
            'train': 'TriviaQA (30)', 'test': 'PopQA (30)',
            'cross_acc': round(cross_acc,4), 'scratch_acc': round(scratch_acc,4),
            'improvement': round(cross_acc - scratch_acc, 4)
        },
        'cross_task2': {
            'train': 'TriviaQA (30)', 'test': 'StrategyQA (30)',
            'cross_acc': round(cross_sqa,4)
        }
    }

# ================================================================
# MAIN
# ================================================================
def main():
    api = DeepSeekClient(max_tokens=1024)
    all_results = {}

    out_path = os.path.join(RESULTS_DIR, 'multiturn_v2_results.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    # 1. HotpotQA multi-hop
    if 'HotpotQA' not in all_results:
        print('='*60)
        print('1. HOTPOTQA MULTI-HOP (50 samples)')
        print('='*60)
        hotpot = run_hotpotqa(api)
        if hotpot:
            all_results['HotpotQA'] = hotpot
            with open(out_path, 'w') as f: json.dump(all_results, f, indent=2)

    # 2. PDDL with templates
    if 'PDDL_templates' not in all_results:
        print('\n' + '='*60)
        print('2. PDDL WITH TEMPLATES')
        print('='*60)
        pddl = run_pddl_with_templates(api)
        if pddl:
            all_results.update(pddl)
            with open(out_path, 'w') as f: json.dump(all_results, f, indent=2)

    # 3. Cross-task transfer
    if 'cross_task' not in all_results:
        print('\n' + '='*60)
        print('3. CROSS-TASK TRANSFER')
        print('='*60)
        cross = run_cross_task(api)
        if cross:
            all_results.update(cross)
            with open(out_path, 'w') as f: json.dump(all_results, f, indent=2)

    # Summary
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    for k, v in sorted(all_results.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    print(f'  {k}/{kk}: acc={vv.get("accuracy","?")}')
                else:
                    print(f'  {k}: {vv}')
    print(f'\nSaved to {out_path}')

if __name__ == '__main__':
    main()
