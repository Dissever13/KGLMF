import time
import json
import os
from typing import List, Dict, Callable
from .metrics import exact_match, f1_score_fn, pass_at_k, planning_success_rate
from .datasets import (load_triviaqa, load_strategyqa, load_popqa,
                        load_kodcode, load_bigcodebench, load_pddl)

class Evaluator:
    def __init__(self, orchestrator, output_dir: str = "experiments/results"):
        self.orchestrator = orchestrator
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_dataset(self, name: str, data: List[Dict],
                        metric_fn: Callable, role_sequence: List[str] = None,
                        metric_key: str = "answer") -> Dict:
        if role_sequence is None:
            role_sequence = ["planner", "coder", "critic"]

        correct = 0
        total = 0
        latencies = []
        predictions = []

        for idx, item in enumerate(data):
            query = item.get("question") or item.get("prompt", "")
            ground_truth = item.get(metric_key, item.get("answer", ""))

            start = time.time()
            try:
                responses = self.orchestrator.run_round(query, role_sequence)
                final_response = list(responses.values())[-1] if responses else ""
            except Exception as e:
                final_response = f"[ERROR: {e}]"
                print(f"  Error on {item.get('id', idx)}: {e}")

            latency = time.time() - start
            latencies.append(latency)

            if metric_fn == pass_at_k:
                score = metric_fn(final_response, item.get("test_cases", []))
            elif metric_fn == planning_success_rate:
                score = metric_fn(final_response, item.get("valid_actions", []))
            else:
                score = metric_fn(final_response, str(ground_truth))

            correct += score
            total += 1
            predictions.append({
                "id": item.get("id", str(idx)), "query": query,
                "prediction": final_response[:500],
                "ground_truth": str(ground_truth)[:500],
                "score": score, "latency": latency,
            })

            if (idx + 1) % 20 == 0:
                print(f"    [{name}] {idx+1}/{len(data)} — "
                      f"Acc: {correct/total:.4f}, "
                      f"Avg Lat: {sum(latencies)/len(latencies):.3f}s")

        result = {
            "dataset": name,
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total, "correct": correct,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "graph_stats": self.orchestrator.get_graph_stats(),
        }
        self.results[name] = result

        with open(os.path.join(self.output_dir, f"{name}_predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        return result

    def run_all_main_experiments(self, max_samples: int = 200):
        print("=" * 60)
        print("KG-LMF Main Experiments")
        print("=" * 60)

        datasets = {
            "TriviaQA": (load_triviaqa(max_samples), exact_match, ["planner", "critic"]),
            "StrategyQA": (load_strategyqa(max_samples), exact_match, ["planner", "coder", "critic"]),
            "PopQA": (load_popqa(max_samples), exact_match, ["planner", "critic"]),
            "KodCode": (load_kodcode(max(100, max_samples//2)), pass_at_k, ["planner", "coder", "critic"]),
            "BigCodeBench": (load_bigcodebench(max(100, max_samples//2)), pass_at_k, ["planner", "coder", "critic"]),
            "PDDL": (load_pddl(max(100, max_samples//2)), planning_success_rate, ["planner", "coder"]),
        }

        for name, (data, metric_fn, roles) in datasets.items():
            print(f"\n{'─'*40}")
            print(f"  Dataset: {name} ({len(data)} samples, roles: {roles})")
            result = self.evaluate_dataset(name, data, metric_fn, roles)
            print(f"  => Accuracy: {result['accuracy']:.4f}, "
                  f"Latency: {result['avg_latency']:.3f}s, "
                  f"Nodes: {result['graph_stats']['total_nodes']}")

        self._save_results("main_results.json")
        self._print_summary()

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        for name, r in self.results.items():
            print(f"  {name:15s}: {r['accuracy']:.4f}  "
                  f"(latency: {r['avg_latency']:.3f}s)")

    def _save_results(self, filename: str):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filepath}")
