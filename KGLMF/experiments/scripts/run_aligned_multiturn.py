"""Aligned multi-turn: same backbone, larger scale, graph evolution measurement."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.api.client import DeepSeekClient
from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT = os.path.join(RESULTS_DIR, 'aligned_multiturn.json')

SYS = "Answer directly in one sentence. No explanation."

# 10 topic chains x 5 turns each = 50 multi-turn questions
TOPICS = [
    ("geography", [
        ("What is the capital of France?", "Paris"),
        ("What river flows through Paris?", "Seine"),
        ("What famous Gothic cathedral is on an island in the Seine in Paris?", "Notre Dame"),
        ("What architectural style is Notre Dame Cathedral known for?", "Gothic"),
        ("In what century did Gothic architecture become dominant in France?", "12th"),
    ]),
    ("science", [
        ("What element has atomic number 1?", "Hydrogen"),
        ("What is the most common isotope of hydrogen?", "Protium"),
        ("What process in stars fuses hydrogen into helium?", "nuclear fusion"),
        ("What is the approximate minimum temperature for nuclear fusion in stars?", "15 million"),
        ("In what layer of the Sun does nuclear fusion occur?", "core"),
    ]),
    ("history", [
        ("Who was the first president of the United States?", "George Washington"),
        ("What war did George Washington lead the Continental Army in?", "Revolutionary War"),
        ("What year did the American Revolutionary War end?", "1783"),
        ("What treaty formally ended the American Revolutionary War?", "Treaty of Paris"),
        ("Which country was the primary opponent in that war?", "Great Britain"),
    ]),
    ("biology", [
        ("What organ pumps blood in the human body?", "heart"),
        ("How many chambers does the human heart have?", "four"),
        ("What valve separates the left atrium and left ventricle?", "mitral valve"),
        ("What condition occurs when the mitral valve does not close properly?", "mitral regurgitation"),
        ("What imaging technique diagnoses mitral valve problems?", "echocardiogram"),
    ]),
    ("astronomy", [
        ("What is the closest planet to the Sun?", "Mercury"),
        ("What is the largest planet in the solar system?", "Jupiter"),
        ("How many moons does Jupiter have approximately?", "95"),
        ("What is the name of Jupiter's largest moon?", "Ganymede"),
        ("Is Ganymede larger than the planet Mercury?", "yes"),
    ]),
    ("chemistry", [
        ("What is the chemical symbol for water?", "H2O"),
        ("What is the pH of pure water?", "7"),
        ("What ion makes a solution acidic?", "hydrogen"),
        ("What is the chemical formula for hydrochloric acid?", "HCl"),
        ("What is the pH of a 0.1M HCl solution approximately?", "1"),
    ]),
    ("literature", [
        ("Who wrote Pride and Prejudice?", "Jane Austen"),
        ("In what century was Pride and Prejudice published?", "19th"),
        ("What literary period is Jane Austen associated with?", "Regency"),
        ("Name another novel by Jane Austen.", "Emma"),
        ("What theme is central to most of Austen's novels?", "marriage"),
    ]),
    ("music", [
        ("Who composed the Ninth Symphony?", "Beethoven"),
        ("In what musical period did Beethoven compose?", "Classical"),
        ("Was Beethoven completely deaf when he composed the Ninth Symphony?", "yes"),
        ("What choral text is used in the final movement?", "Ode to Joy"),
        ("Who wrote the poem Ode to Joy?", "Schiller"),
    ]),
    ("computing", [
        ("Who is known as the father of computer science?", "Alan Turing"),
        ("What theoretical machine did Turing propose?", "Turing machine"),
        ("During which war did Turing help break the Enigma code?", "World War II"),
        ("What test did Turing propose for machine intelligence?", "Turing test"),
        ("In what year did Turing publish his seminal paper On Computable Numbers?", "1936"),
    ]),
    ("art", [
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("In what century was the Mona Lisa painted?", "16th"),
        ("What technique did da Vinci use for soft transitions between colors?", "sfumato"),
        ("In what museum is the Mona Lisa displayed?", "Louvre"),
        ("In what city is the Louvre museum located?", "Paris"),
    ]),
]

def run_multiturn_kg_lmf(api, topics):
    """Run KG-LMF on multi-turn topics, tracking graph evolution"""
    all_results = []
    for topic_name, turns in topics:
        graph = LatentReasoningGraph(d_model=384)
        composer = MemoryComposer(d_model=384, n_learnable_queries=16)
        router = GraphRouter(d_model=384)
        tts = AdaptiveTTSController(budget_min=4, budget_max=16)
        injector = LatentInjector(d_model=384, injection_layers=[0])
        evolution = GraphEvolution(merge_threshold=0.85)
        orch = Orchestrator(composer=composer, router=router, tts=tts,
                           injector=injector, evolution=evolution,
                           graph=graph, api_client=api, d_model=384)

        correct = 0
        latencies = []
        node_counts = []
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
            correct += score
            latencies.append(lat)
            node_counts.append(len(graph))

        acc = correct / len(turns)
        avg_lat = sum(latencies) / len(latencies)
        all_results.append({
            'topic': topic_name,
            'accuracy': round(acc, 4),
            'avg_latency': round(avg_lat, 2),
            'turns': len(turns),
            'final_nodes': node_counts[-1] if node_counts else 0,
            'node_growth': node_counts,
        })

    return all_results

def run_multiturn_nomemory(api, topics):
    """Run NoMemory on multi-turn topics"""
    all_results = []
    for topic_name, turns in topics:
        correct = 0
        latencies = []
        for turn, (q, gt) in enumerate(turns):
            t0 = time.time()
            try:
                r = api.generate(SYS, q, max_tokens=256)
            except:
                r = ''
            lat = time.time() - t0
            score = 1.0 if gt.lower() in r.lower() else 0.0
            correct += score
            latencies.append(lat)
        acc = correct / len(turns)
        avg_lat = sum(latencies) / len(latencies)
        all_results.append({
            'topic': topic_name,
            'accuracy': round(acc, 4),
            'avg_latency': round(avg_lat, 2),
            'turns': len(turns),
        })
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_deepseek', action='store_true')
    args = parser.parse_args()

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            results = json.load(f)

    if not args.skip_deepseek:
        print('=' * 60)
        print('MULTI-TURN EXPERIMENTS (DeepSeek V4 Pro)')
        print('=' * 60)
        api = DeepSeekClient(max_tokens=512)

        if 'KG-LMF_multiturn' not in results:
            print('Running KG-LMF multi-turn (10 topics x 5 turns)...')
            kg_results = run_multiturn_kg_lmf(api, TOPICS)
            results['KG-LMF_multiturn'] = kg_results
            with open(OUTPUT, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            kg_results = results['KG-LMF_multiturn']

        if 'NoMemory_multiturn' not in results:
            print('Running NoMemory multi-turn (10 topics x 5 turns)...')
            nm_results = run_multiturn_nomemory(api, TOPICS)
            results['NoMemory_multiturn'] = nm_results
            with open(OUTPUT, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            nm_results = results['NoMemory_multiturn']

        # Print summary
        kg_accs = [r['accuracy'] for r in kg_results]
        nm_accs = [r['accuracy'] for r in nm_results]
        kg_avg = sum(kg_accs) / len(kg_accs)
        nm_avg = sum(nm_accs) / len(nm_accs)
        kg_nodes = [r.get('final_nodes', 0) for r in kg_results]

        print('')
        print('%-15s %-12s %-12s %-10s' % ('Topic', 'KG-LMF', 'NoMemory', 'Nodes'))
        print('-' * 50)
        for i, (kg, nm) in enumerate(zip(kg_results, nm_results)):
            print('%-15s %-12s %-12s %-10s' % (
                kg['topic'], str(kg['accuracy']), str(nm['accuracy']),
                str(kg.get('final_nodes', '?'))))

        print('-' * 50)
        print('%-15s %-12s %-12s' % ('AVERAGE', str(round(kg_avg, 4)), str(round(nm_avg, 4))))
        print('')
        print('KG-LMF avg nodes/topic: ' + str(round(sum(kg_nodes) / len(kg_nodes), 1)))

    print('Saved to ' + OUTPUT)

if __name__ == '__main__':
    main()
