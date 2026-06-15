"""KG-LMF Fast Experiment Runner — runs multiple datasets with actual evaluation"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.client import DeepSeekClient
from src.graph.graph import LatentReasoningGraph
from src.graph.node import NodeType
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator

# Test questions per dataset type (curated for quick validation)
DATASETS = {
    "TriviaQA": {
        "questions": [
            ("What is the capital of France?", "Paris"),
            ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
            ("What is the chemical symbol for gold?", "Au"),
            ("In which year did the Titanic sink?", "1912"),
            ("What is the largest ocean on Earth?", "Pacific"),
            ("Who painted the Sistine Chapel ceiling?", "Michelangelo"),
            ("What element has the atomic number 6?", "Carbon"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("What is the longest river in the world?", "Nile"),
            ("Who discovered penicillin?", "Alexander Fleming"),
        ],
        "metric": "em",
        "roles": ["planner", "critic"],
    },
    "StrategyQA": {
        "questions": [
            ("Can a fish drown in water?", "yes"),
            ("Is it possible to sneeze while sleeping?", "no"),
            ("Would a mirror work in a completely dark room?", "no"),
            ("Can sound travel through a vacuum?", "no"),
            ("Is a tomato a fruit?", "yes"),
        ],
        "metric": "em",
        "roles": ["planner", "coder", "critic"],
    },
    "PopQA": {
        "questions": [
            ("What is the capital of Australia?", "Canberra"),
            ("In which country is the city of Marrakech?", "Morocco"),
            ("What language is spoken in Brazil?", "Portuguese"),
            ("Who was the first president of the United States?", "George Washington"),
            ("What is the currency of Japan?", "Yen"),
        ],
        "metric": "em",
        "roles": ["planner", "critic"],
    },
    "KodCode": {
        "questions": [
            ("Write a Python function to compute factorial of n.", "def factorial"),
            ("Write a function to check if a string is a palindrome.", "def is_palindrome"),
            ("Write code to find the maximum element in a list.", "def find_max"),
        ],
        "metric": "contains",
        "roles": ["planner", "coder", "critic"],
    },
}

SYSTEM_PROMPTS = {
    "planner": "You are a strategic planner. Give direct, factual answers. Do NOT explain your reasoning. Answer ONLY what is asked.",
    "coder": "You are an expert programmer. Write only the requested code. No explanations.",
    "critic": "You are a reviewer. Give direct, concise feedback. Answer the question directly.",
}

def run_benchmark(api, ds_name, ds_config, n_samples=None):
    questions = ds_config["questions"]
    if n_samples:
        questions = questions[:n_samples]

    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=32)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    orch.agents = {}  # start fresh

    correct = 0
    latencies = []
    roles = ds_config["roles"]
    metric = ds_config["metric"]

    print(f"  Dataset: {ds_name} ({len(questions)} questions, roles: {roles})")

    for i, (question, ground_truth) in enumerate(questions):
        start = time.time()
        try:
            results = orch.run_round(question, roles)
            final = list(results.values())[-1] if results else ""
        except Exception as e:
            final = f"[ERROR: {e}]"

        lat = time.time() - start
        latencies.append(lat)

        if metric == "em":
            score = 1.0 if ground_truth.lower() in final.lower() else 0.0
        elif metric == "contains":
            score = 1.0 if ground_truth.lower() in final.lower() else 0.0
        else:
            score = 0.0
        correct += score

        status = "OK" if score > 0 else "MISS"
        pred_clean = final[:80].replace('\n', ' ').replace('\r', ' ')
        pred_ascii = pred_clean.encode('ascii', errors='replace').decode('ascii')
        gt_ascii = ground_truth[:50].encode('ascii', errors='replace').decode('ascii')
        print(f"    [{status}] Q{i+1}: gt=\"{gt_ascii}\" "
              f"-> pred=\"{pred_ascii}\" "
              f"({lat:.1f}s)")

    acc = correct / len(questions) if questions else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print(f"  => {ds_name}: accuracy={acc:.2f}, avg_latency={avg_lat:.1f}s, "
          f"nodes={len(graph)}")
    return {
        "dataset": ds_name,
        "accuracy": round(acc, 4),
        "n_samples": len(questions),
        "avg_latency": round(avg_lat, 2),
        "nodes": len(graph),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=0,
                       help="Max samples per dataset (0=all)")
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Specific datasets to run")
    parser.add_argument("--output", default="experiments/results/experiment_results.json")
    args = parser.parse_args()

    api = DeepSeekClient(max_tokens=1024)
    print(f"KG-LMF Experiment Runner")
    print(f"Model: {api.model}")
    print(f"Datasets: {args.datasets or list(DATASETS.keys())}")
    print(f"="*60)

    all_results = []
    ds_names = args.datasets or DATASETS.keys()

    for ds_name in ds_names:
        if ds_name not in DATASETS:
            print(f"  Unknown dataset: {ds_name}, skipping")
            continue

        api = DeepSeekClient(max_tokens=1024)
        result = run_benchmark(api, ds_name, DATASETS[ds_name],
                             n_samples=args.n_samples or None)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['dataset']:15s}: acc={r['accuracy']:.4f}  "
              f"lat={r['avg_latency']:.1f}s  nodes={r['nodes']}")

    if all_results:
        avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
        print(f"  {'AVERAGE':15s}: acc={avg_acc:.4f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": api.model, "results": all_results, "avg_accuracy": avg_acc},
                  f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
