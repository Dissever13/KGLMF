"""CoQA multi-turn experiment: true conversational QA with referential dependencies"""
import sys, os, time, json
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
OUTPUT = os.path.join(RESULTS_DIR, 'coqa_multiturn_results.json')

SYS = "Answer the question based on the conversation context. Answer concisely in one sentence."

def load_coqa(n_conversations=30):
    path = os.path.join(DATA_DIR, 'CoQA_real.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)[:n_conversations]

def run_coqa_kg_lmf(api, conversations):
    """KG-LMF: graph memory accumulates across conversation turns"""
    all_results = []
    total_turns = 0
    total_correct = 0
    total_lat = 0

    for conv in conversations:
        graph = LatentReasoningGraph(d_model=384)
        composer = MemoryComposer(d_model=384, n_learnable_queries=16)
        router = GraphRouter(d_model=384)
        tts = AdaptiveTTSController(budget_min=4, budget_max=32)
        injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
        evolution = GraphEvolution(merge_threshold=0.85)
        orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector,
                           evolution=evolution, graph=graph, api_client=api, d_model=384)

        correct = 0
        lats = []
        nodes_history = []

        for turn_num, turn in enumerate(conv['turns']):
            q = turn['question']
            gt = str(turn['answer'])
            orch.agents = {}
            a = orch._get_or_create_agent('planner')
            a.system_prompt = SYS
            t0 = time.time()
            try:
                resp = orch.run_round(q, ['planner'])
                r = list(resp.values())[-1] if resp else ''
            except:
                r = ''
            lat = time.time() - t0
            score = 1.0 if gt.lower() in r.lower() else 0.0
            correct += score
            lats.append(lat)
            nodes_history.append(len(graph))

        acc = correct / len(conv['turns'])
        avg_lat = sum(lats) / len(lats)
        all_results.append({
            'id': conv['id'],
            'num_turns': len(conv['turns']),
            'accuracy': round(acc, 4),
            'avg_latency': round(avg_lat, 2),
            'final_nodes': nodes_history[-1] if nodes_history else 0,
            'max_nodes': max(nodes_history) if nodes_history else 0,
        })
        total_turns += len(conv['turns'])
        total_correct += correct
        total_lat += sum(lats)

    avg_acc = total_correct / total_turns if total_turns else 0
    avg_lat = total_lat / total_turns if total_turns else 0
    print(f'  KG-LMF: {len(conversations)} convs, {total_turns} turns, acc={avg_acc:.4f}, lat={avg_lat:.1f}s')

    return {
        'num_conversations': len(conversations),
        'total_turns': total_turns,
        'accuracy': round(avg_acc, 4),
        'avg_latency': round(avg_lat, 2),
        'per_conversation': all_results
    }

def run_coqa_nomemory(api, conversations):
    """NoMemory: no context accumulation across turns"""
    total_turns = 0
    total_correct = 0
    total_lat = 0

    for conv in conversations:
        history = []  # Only current conversation context, no persistent memory
        for turn_num, turn in enumerate(conv['turns']):
            q = turn['question']
            gt = str(turn['answer'])
            # Build prompt with conversation history (like a normal chatbot)
            context = '\n'.join(history[-5:])
            full_q = f'Context:\n{context}\n\nQuestion: {q}' if context else q
            t0 = time.time()
            try:
                r = api.generate(SYS, full_q, max_tokens=256)
            except:
                r = ''
            lat = time.time() - t0
            score = 1.0 if gt.lower() in r.lower() else 0.0
            total_correct += score
            total_lat += lat
            history.append(f'Q: {q}\nA: {r[:200]}')
            total_turns += 1

    avg_acc = total_correct / total_turns if total_turns else 0
    avg_lat = total_lat / total_turns if total_turns else 0
    print(f'  NoMemory: {len(conversations)} convs, {total_turns} turns, acc={avg_acc:.4f}, lat={avg_lat:.1f}s')

    return {
        'num_conversations': len(conversations),
        'total_turns': total_turns,
        'accuracy': round(avg_acc, 4),
        'avg_latency': round(avg_lat, 2),
    }

def run_coqa_textmemory(api, conversations):
    """TextMemory: accumulate QA pairs in text memory, retrieve for each turn"""
    total_turns = 0
    total_correct = 0
    total_lat = 0
    memory = []

    for conv in conversations:
        for turn_num, turn in enumerate(conv['turns']):
            q = turn['question']
            gt = str(turn['answer'])
            # Retrieve from memory
            ctx = '\n'.join(memory[-5:])
            full_q = f'Memory:\n{ctx}\n\nQuestion: {q}' if ctx else q
            t0 = time.time()
            try:
                r = api.generate(SYS, full_q, max_tokens=256)
            except:
                r = ''
            lat = time.time() - t0
            score = 1.0 if gt.lower() in r.lower() else 0.0
            total_correct += score
            total_lat += lat
            memory.append(f'Q: {q}\nA: {r[:200]}')
            total_turns += 1

    avg_acc = total_correct / total_turns if total_turns else 0
    avg_lat = total_lat / total_turns if total_turns else 0
    print(f'  TextMemory: {len(conversations)} convs, {total_turns} turns, acc={avg_acc:.4f}, lat={avg_lat:.1f}s')

    return {
        'num_conversations': len(conversations),
        'total_turns': total_turns,
        'accuracy': round(avg_acc, 4),
        'avg_latency': round(avg_lat, 2),
    }

def main():
    conversations = load_coqa(30)
    if not conversations:
        print('No CoQA data!')
        return

    total_turns = sum(c['num_turns'] for c in conversations)
    print('='*60)
    print(f'CoQA MULTI-TURN EXPERIMENT')
    print(f'{len(conversations)} conversations, {total_turns} total turns')
    print(f'Avg turns/conversation: {total_turns/len(conversations):.1f}')
    print('='*60)

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            results = json.load(f)

    api = DeepSeekClient(max_tokens=512)

    # KG-LMF: graph memory across turns
    if 'KG-LMF' not in results:
        print('\n--- KG-LMF (graph memory) ---')
        results['KG-LMF'] = run_coqa_kg_lmf(api, conversations)
        with open(OUTPUT, 'w') as f:
            json.dump(results, f, indent=2)

    # NoMemory: only current conversation context
    if 'NoMemory' not in results:
        print('\n--- NoMemory (context only) ---')
        results['NoMemory'] = run_coqa_nomemory(api, conversations)
        with open(OUTPUT, 'w') as f:
            json.dump(results, f, indent=2)

    # TextMemory: accumulated text memory
    if 'TextMemory' not in results:
        print('\n--- TextMemory (text accumulation) ---')
        results['TextMemory'] = run_coqa_textmemory(api, conversations)
        with open(OUTPUT, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print('\n' + '='*60)
    print('CoQA MULTI-TURN RESULTS')
    print('='*60)
    for method in ['NoMemory', 'TextMemory', 'KG-LMF']:
        if method in results:
            r = results[method]
            acc = r['accuracy']
            turns = r['total_turns']
            print(f'  {method:15s}: acc={acc:.4f} ({turns} turns)')
    print(f'\nSaved to {OUTPUT}')

if __name__ == '__main__':
    main()
