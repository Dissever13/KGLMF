"""KG-LMF Ablation Study Runner"""
import sys, os, yaml, argparse, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.graph.graph import LatentReasoningGraph
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator
from src.api.client import DeepSeekClient
from src.eval.evaluator import Evaluator
from src.eval.datasets import load_triviaqa, load_strategyqa
from src.eval.metrics import exact_match

def build_orchestrator_ablated(config: dict, ablation: dict):
    d_model = config["model"]["d_model"]
    api = DeepSeekClient(model=config["model"]["backbone"])

    modify = ablation["modify"]

    if modify == "disable_graph_edges":
        graph = LatentReasoningGraph(d_model=d_model)
    else:
        graph = LatentReasoningGraph(d_model=d_model)

    if modify == "remove_role_projection":
        composer = MemoryComposer(
            d_model=d_model, n_layers=config["composer"]["n_layers"],
            n_heads=config["composer"]["n_heads"],
            n_learnable_queries=config["composer"]["n_learnable_queries"],
            role_types=["generic"])
    else:
        composer = MemoryComposer(
            d_model=d_model, n_layers=config["composer"]["n_layers"],
            n_heads=config["composer"]["n_heads"],
            n_learnable_queries=config["composer"]["n_learnable_queries"])

    if modify == "fix_budget_k16":
        tts = AdaptiveTTSController(budget_min=16, budget_max=16)
    else:
        tts = AdaptiveTTSController(
            budget_min=config["tts"]["budget_min"],
            budget_max=config["tts"]["budget_max"])

    router = GraphRouter(d_model=d_model,
        temperature=config["router"]["temperature"])

    if modify == "use_text_memory":
        injector = None
    else:
        injector = LatentInjector(d_model=d_model,
            injection_layers=config["injection"]["layers"])

    if modify == "disable_evolution":
        evolution = None
    else:
        evolution = GraphEvolution(
            merge_threshold=config["evolution"]["merge_threshold"],
            prune_percentile=config["evolution"]["prune_percentile"])

    return Orchestrator(
        composer=composer, router=router, tts=tts,
        injector=injector, evolution=evolution,
        graph=graph, api_client=api, d_model=d_model)

def run_ablation(ablation: dict, config: dict, datasets: dict, output_dir: str):
    name = ablation["name"]
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"  {ablation['description']}")
    print(f"{'='*60}")

    orchestrator = build_orchestrator_ablated(config, ablation)
    evaluator = Evaluator(orchestrator, output_dir=output_dir)

    results = {}
    for ds_name, (data, metric_fn, roles) in datasets.items():
        result = evaluator.evaluate_dataset(ds_name, data, metric_fn, roles)
        results[ds_name] = result["accuracy"]

    avg_acc = sum(results.values()) / len(results) if results else 0.0
    print(f"  Average accuracy: {avg_acc:.4f}")
    return {"ablation": name, "avg_accuracy": avg_acc, "per_dataset": results}

def main():
    parser = argparse.ArgumentParser(description="KG-LMF Ablation Studies")
    parser.add_argument("--config", default="experiments/configs/ablation.yaml")
    parser.add_argument("--main_config", default="experiments/configs/main.yaml")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), '..', '..')
    with open(os.path.join(base, args.config)) as f:
        ablation_config = yaml.safe_load(f)
    with open(os.path.join(base, args.main_config)) as f:
        main_config = yaml.safe_load(f)

    eval_datasets = {
        "TriviaQA": (load_triviaqa(args.max_samples), exact_match, ["planner", "critic"]),
        "StrategyQA": (load_strategyqa(args.max_samples), exact_match, ["planner", "coder", "critic"]),
    }

    all_results = []
    for ablation in ablation_config["ablations"]:
        result = run_ablation(ablation, main_config, eval_datasets, args.output_dir)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    full_kg_lmf_acc = None
    for r in all_results:
        print(f"  {r['ablation']:30s}: {r['avg_accuracy']:.4f}")
        if r['ablation'] == 'full_kg_lmf':
            full_kg_lmf_acc = r['avg_accuracy']

    if full_kg_lmf_acc:
        print(f"\n  Ablation impact (drop from full KG-LMF):")
        for r in all_results:
            if r['ablation'] != 'full_kg_lmf':
                delta = full_kg_lmf_acc - r['avg_accuracy']
                print(f"    {r['ablation']:28s}: -{delta:.4f}")

    outpath = os.path.join(args.output_dir, "ablation_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outpath}")

if __name__ == "__main__":
    main()
