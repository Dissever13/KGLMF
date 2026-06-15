"""KG-LMF Main Experiment Runner"""
import sys, os, yaml, argparse, json
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

def build_orchestrator(config: dict):
    d_model = config["model"]["d_model"]
    api = DeepSeekClient(model=config["model"]["backbone"])
    graph = LatentReasoningGraph(d_model=d_model)
    composer = MemoryComposer(
        d_model=d_model,
        n_layers=config["composer"]["n_layers"],
        n_heads=config["composer"]["n_heads"],
        n_learnable_queries=config["composer"]["n_learnable_queries"])
    router = GraphRouter(d_model=d_model,
        temperature=config["router"]["temperature"])
    tts = AdaptiveTTSController(
        budget_min=config["tts"]["budget_min"],
        budget_max=config["tts"]["budget_max"],
        scaling_factor=config["tts"]["scaling_factor"])
    injector = LatentInjector(d_model=d_model,
        injection_layers=config["injection"]["layers"])
    evolution = GraphEvolution(
        merge_threshold=config["evolution"]["merge_threshold"],
        prune_percentile=config["evolution"]["prune_percentile"])

    return Orchestrator(
        composer=composer, router=router, tts=tts,
        injector=injector, evolution=evolution,
        graph=graph, api_client=api, d_model=d_model)

def main():
    parser = argparse.ArgumentParser(description="KG-LMF Main Experiments")
    parser.add_argument("--config", default="experiments/configs/main.yaml")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--datasets", nargs="*",
                       default=["TriviaQA", "StrategyQA", "PopQA",
                               "KodCode", "BigCodeBench", "PDDL"])
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Building KG-LMF orchestrator...")
    print(f"  Backbone: {config['model']['backbone']}")
    print(f"  d_model: {config['model']['d_model']}")
    orchestrator = build_orchestrator(config)

    evaluator = Evaluator(orchestrator, output_dir=args.output_dir)
    evaluator.run_all_main_experiments(max_samples=args.max_samples)

    summary_path = os.path.join(args.output_dir, "main_results.json")
    with open(summary_path) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
