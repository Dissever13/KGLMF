"""Multi-turn experiments + second backbone comparison using Ollama Qwen3.5"""
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SYS_QA = "Answer directly in one sentence. No explanation."

# Multi-turn QA pairs (self-contained, graph memory accumulates supporting knowledge)
MULTI_TURN_TASKS = [
    {  # Geography: each turn self-contained but shares topic domain
        "topic": "geography",
        "turns": [
            ("What is the capital of France?", "Paris"),
            ("What is the name of the river that flows through Paris?", "Seine"),
            ("What famous Gothic cathedral is located on an island in the Seine River in Paris?", "Notre Dame"),
            ("What architectural style is Notre Dame Cathedral in Paris known for?", "Gothic"),
            ("In what century did Gothic architecture become dominant in France?", "12th"),
        ]
    },
    {
        "topic": "science",
        "turns": [
            ("What element has atomic number 1?", "Hydrogen"),
            ("What is the most common isotope of hydrogen called?", "Protium"),
            ("What process in stars fuses hydrogen into helium?", "nuclear fusion"),
            ("What is the approximate minimum temperature in Celsius needed for nuclear fusion in stars?", "15 million"),
            ("In what layer of the Sun does nuclear fusion occur?", "core"),
        ]
    },
    {
        "topic": "history",
        "turns": [
            ("Who was the first president of the United States?", "George Washington"),
            ("What war did George Washington lead the Continental Army in?", "Revolutionary War"),
            ("What year did the American Revolutionary War end?", "1783"),
            ("What treaty formally ended the American Revolutionary War?", "Treaty of Paris"),
            ("Which country was the primary opponent of the United States in the Revolutionary War?", "Great Britain"),
        ]
    },
    {
        "topic": "biology",
        "turns": [
            ("What organ pumps blood in the human body?", "heart"),
            ("How many chambers does the human heart have?", "four"),
            ("What is the name of the valve between the left atrium and left ventricle of the heart?", "mitral valve"),
            ("What condition occurs when the mitral valve does not close properly?", "mitral regurgitation"),
            ("What imaging technique is commonly used to diagnose mitral valve problems?", "echocardiogram"),
        ]
    },
]

# Single-turn tasks for backbone comparison
QA_PAIRS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In which year did the Titanic sink?", "1912"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("What element has atomic number 6?", "Carbon"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the longest river in the world?", "Nile"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the speed of light in m/s?", "299792458"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many continents are there?", "7"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the currency of Japan?", "yen"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
]

def run_multiturn(api, model_name, tasks):
    """Run multi-turn experiments showing graph evolution benefit"""
    results = {}
    for task in tasks:
        topic = task["topic"]
        print(f"\n  Multi-turn: {topic}")

        # KG-LMF with memory
        graph = LatentReasoningGraph(d_model=384)
        composer = MemoryComposer(d_model=384, n_learnable_queries=16)
        router = GraphRouter(d_model=384)
        tts = AdaptiveTTSController(budget_min=4, budget_max=16)
        injector = LatentInjector(d_model=384, injection_layers=[0])
        evolution = GraphEvolution(merge_threshold=0.85)
        orch = Orchestrator(composer=composer, router=router, tts=tts, injector=injector, evolution=evolution, graph=graph, api_client=api, d_model=384)

        kg_scores = []
        nm_scores = []
        kg_lats = []
        nm_lats = []

        for turn, (q, gt) in enumerate(task["turns"]):
            # KG-LMF
            orch.agents = {}
            t0 = time.time()
            try:
                resp = orch.run_round(q, ["critic"])
                kg_resp = list(resp.values())[-1] if resp else ""
            except:
                kg_resp = ""
            kg_lat = time.time() - t0
            kg_score = 1.0 if gt.lower() in kg_resp.lower() else 0.0
            kg_scores.append(kg_score)
            kg_lats.append(kg_lat)

            # NoMemory (same API, no graph)
            t0 = time.time()
            try:
                nm_resp = api.generate(SYS_QA, q, max_tokens=256)
            except:
                nm_resp = ""
            nm_lat = time.time() - t0
            nm_score = 1.0 if gt.lower() in nm_resp.lower() else 0.0
            nm_scores.append(nm_score)
            nm_lats.append(nm_lat)

            graph_nodes = len(graph)
            status = "OK" if kg_score else "NO"
            nm_status = "OK" if nm_score else "NO"
            kg_clean = kg_resp[:60].encode('ascii', errors='replace').decode('ascii')
            nm_clean = nm_resp[:60].encode('ascii', errors='replace').decode('ascii')
            print(f"    T{turn+1}: KG[{status}] NM[{nm_status}] nodes={graph_nodes}")

        kg_acc = sum(kg_scores) / len(kg_scores)
        nm_acc = sum(nm_scores) / len(nm_scores)
        kg_avg_lat = sum(kg_lats) / len(kg_lats)
        nm_avg_lat = sum(nm_lats) / len(nm_lats)

        print(f"    KG-LMF: {kg_acc:.2f} (lat {kg_avg_lat:.1f}s) | NoMemory: {nm_acc:.2f} (lat {nm_avg_lat:.1f}s)")
        results[topic] = {"KG-LMF_acc": kg_acc, "NoMemory_acc": nm_acc, "KG-LMF_lat": kg_avg_lat, "NoMemory_lat": nm_avg_lat, "turns": len(task["turns"])}

    return results

def run_backbone_comparison(api1, name1, api2, name2, questions):
    """Compare two backbones on single-turn QA"""
    results = {}
    for name, api in [(name1, api1), (name2, api2)]:
        correct = 0
        latencies = []
        for q, gt in questions:
            t0 = time.time()
            try:
                resp = api.generate(SYS_QA, q, max_tokens=256)
            except:
                resp = ""
            lat = time.time() - t0
            score = 1.0 if gt.lower() in resp.lower() else 0.0
            correct += score
            latencies.append(lat)
        results[name] = {"accuracy": correct / len(questions), "avg_latency": sum(latencies) / len(latencies)}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_ollama', action='store_true')
    parser.add_argument('--skip_deepseek', action='store_true')
    args = parser.parse_args()

    output = {}

    # 1. Multi-turn experiments (DeepSeek)
    if not args.skip_deepseek:
        print("="*60)
        print("MULTI-TURN EXPERIMENTS (DeepSeek V4 Pro)")
        print("="*60)
        api = DeepSeekClient(max_tokens=512)
        mt_results = run_multiturn(api, "DeepSeek-V4-Pro", MULTI_TURN_TASKS)
        output["multi_turn_deepseek"] = mt_results

        # Summary
        kg_accs = [v["KG-LMF_acc"] for v in mt_results.values()]
        nm_accs = [v["NoMemory_acc"] for v in mt_results.values()]
        print(f"\n  Multi-turn Average: KG-LMF={sum(kg_accs)/len(kg_accs):.2f}, NoMemory={sum(nm_accs)/len(nm_accs):.2f}")

    # 2. Backbone comparison (DeepSeek vs Ollama Qwen3.5)
    print("\n" + "="*60)
    print("BACKBONE COMPARISON")
    print("="*60)

    if not args.skip_deepseek:
        print("  Testing DeepSeek V4 Pro...")
        ds_api = DeepSeekClient(max_tokens=256)
    else:
        ds_api = None

    if not args.skip_ollama:
        print("  Testing Ollama Qwen3.5:9b...")
        try:
            qwen_api = OllamaClient(model="qwen3.5:9b", max_tokens=256)
            test_resp = qwen_api.generate(SYS_QA, "Say hello")
            print(f"  Qwen test: {test_resp[:50]}")
        except Exception as e:
            print(f"  Qwen ERROR: {e}")
            qwen_api = None
    else:
        qwen_api = None

    if ds_api and qwen_api:
        print("\n  Running backbone comparison (15 QA pairs)...")
        bb_results = run_backbone_comparison(ds_api, "DeepSeek-V4-Pro", qwen_api, "Qwen3.5-9B", QA_PAIRS)
        output["backbone_comparison"] = bb_results
        for name, r in bb_results.items():
            print(f"  {name}: acc={r['accuracy']:.4f}, lat={r['avg_latency']:.2f}s")
    elif ds_api:
        print("  Only DeepSeek available, skipping backbone comparison")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'multiturn_backbone_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
