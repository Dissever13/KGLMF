"""Parameter sensitivity sweep for KG-LMF on StrategyQA."""
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

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset(name, n):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)[:n]
    return [{"id": f"Q{i}", "question": f"Test {i}", "answer": "yes", "type": "reasoning"} for i in range(n)]

def run_kg_lmf_with_params(api, questions, budget_min, budget_max, scaling_factor, n_queries, n_samples=30):
    """Run KG-LMF with specific hyperparameters"""
    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=n_queries)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=budget_min, budget_max=budget_max, scaling_factor=scaling_factor)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    orch.agents = {}

    correct = 0
    latencies = []
    samples = questions[:n_samples]

    for i, item in enumerate(samples):
        q = item.get("question") or item.get("prompt", "")
        gt = str(item.get("answer", ""))
        t0 = time.time()
        try:
            responses = orch.run_round(q, ["planner", "critic"])
            resp = list(responses.values())[-1] if responses else ""
        except Exception as e:
            resp = f"[ERROR: {e}]"
        lat = time.time() - t0
        latencies.append(lat)
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        correct += score

    acc = correct / len(samples) if samples else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return {"accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2), "n": len(samples)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--output", default="experiments/results/param_sweep.json")
    parser.add_argument("--sweep", default="all", choices=["all", "budget", "k", "compress"])
    args = parser.parse_args()

    api = DeepSeekClient(max_tokens=512)
    questions = load_dataset(args.dataset, args.n_samples)

    sweeps = []

    if args.sweep in ("all", "budget"):
        print("="*60)
        print("SWEEP 1: TTS Budget Range [B_min, B_max]")
        print("="*60)
        budget_pairs = [(2, 16), (4, 32), (4, 64), (8, 64), (16, 128)]
        for bmin, bmax in budget_pairs:
            print(f"  Budget [{bmin}, {bmax}]...")
            result = run_kg_lmf_with_params(api, questions, bmin, bmax, 2.0, 16, args.n_samples)
            result["param"] = f"budget_[{bmin},{bmax}]"
            result["bmin"] = bmin
            result["bmax"] = bmax
            sweeps.append(result)
            print(f"    => acc={result['accuracy']:.4f}, lat={result['avg_latency']:.1f}s")

    if args.sweep in ("all", "k"):
        print("\n" + "="*60)
        print("SWEEP 2: Learnable Compression Queries (n_queries)")
        print("="*60)
        for nq in [4, 8, 16, 32]:
            print(f"  n_queries={nq}...")
            result = run_kg_lmf_with_params(api, questions, 4, 32, 2.0, nq, args.n_samples)
            result["param"] = f"n_queries_{nq}"
            result["n_queries"] = nq
            sweeps.append(result)
            print(f"    => acc={result['accuracy']:.4f}, lat={result['avg_latency']:.1f}s")

    if args.sweep in ("all", "compress"):
        print("\n" + "="*60)
        print("SWEEP 3: Scaling Factor (lambda)")
        print("="*60)
        for sf in [0.5, 1.0, 2.0, 4.0, 8.0]:
            print(f"  scaling_factor={sf}...")
            result = run_kg_lmf_with_params(api, questions, 4, 32, sf, 16, args.n_samples)
            result["param"] = f"scaling_{sf}"
            result["scaling_factor"] = sf
            sweeps.append(result)
            print(f"    => acc={result['accuracy']:.4f}, lat={result['avg_latency']:.1f}s")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', args.output)
    all_results = {"dataset": args.dataset, "n_samples": args.n_samples, "sweeps": sweeps}
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP SUMMARY ({args.dataset}, n={args.n_samples})")
    print(f"{'='*70}")
    print(f"{'Parameter':<30s} {'Accuracy':>8s} {'Latency':>8s}")
    print("-"*50)
    for s in sweeps:
        print(f"{s['param']:<30s} {s['accuracy']:>8.4f} {s['avg_latency']:>7.1f}s")
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
