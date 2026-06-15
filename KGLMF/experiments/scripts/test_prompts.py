"""Test KG-LMF with optimized prompts on representative samples."""
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

# Test questions from each benchmark
TESTS = {
    "TriviaQA": [
        ("What is the capital of France?", "Paris"),
        ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("In which year did the Titanic sink?", "1912"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What element has atomic number 6?", "Carbon"),
        ("Which planet is known as the Red Planet?", "Mars"),
        ("What is the longest river in the world?", "Nile"),
        ("Who discovered penicillin?", "Alexander Fleming"),
    ],
    "StrategyQA": [
        ("Can a fish drown in water? Answer yes or no.", "yes"),
        ("Is a tomato botanically a fruit? Answer yes or no.", "yes"),
        ("Would a mirror work in a completely dark room? Answer yes or no.", "no"),
        ("Can sound travel through a vacuum? Answer yes or no.", "no"),
        ("Is a banana a berry? Answer yes or no.", "yes"),
    ],
    "PopQA": [
        ("What language is spoken in Brazil?", "Portuguese"),
        ("What is the currency of Japan?", "yen"),
        ("In which country is Marrakech?", "Morocco"),
        ("What is the capital of Australia?", "Canberra"),
        ("Who was the first US president?", "George Washington"),
    ],
}

def run_test(api, dataset, questions):
    """Run KG-LMF with task-appropriate prompts"""
    from src.agents.prompts import get_prompts_for_task
    prompt_config = get_prompts_for_task(
        "code" if dataset == "KodCode" else
        "reasoning" if dataset == "StrategyQA" else
        "qa"
    )

    graph = LatentReasoningGraph(d_model=384)
    composer = MemoryComposer(d_model=384, n_learnable_queries=16)
    router = GraphRouter(d_model=384)
    tts = AdaptiveTTSController(budget_min=4, budget_max=32)
    injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
    evolution = GraphEvolution(merge_threshold=0.85)

    orch = Orchestrator(composer=composer, router=router, tts=tts,
                       injector=injector, evolution=evolution,
                       graph=graph, api_client=api, d_model=384)
    orch.agents = {}

    correct = 0
    latencies = []
    roles = prompt_config["roles"]

    for i, (question, gt) in enumerate(questions):
        # Use task-optimized prompts
        for role in roles:
            orch._get_or_create_agent(role).system_prompt = prompt_config[role]

        t0 = time.time()
        try:
            results = orch.run_round(question, roles)
            resp = list(results.values())[-1] if results else ""
        except Exception as e:
            resp = f"[ERROR: {e}]"
        lat = time.time() - t0
        latencies.append(lat)

        score = 1.0 if gt.lower() in resp.lower() else 0.0
        correct += score

        status = "OK" if score else "NO"
        clean = resp[:120].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
        print(f"  [{status}] Q{i+1}: gt={gt} -> {clean} ({lat:.1f}s)")

    acc = correct / len(questions) if questions else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return acc, avg_lat

def main():
    api = DeepSeekClient(max_tokens=1024)
    print("="*60)
    print("KG-LMF Optimized Prompt Test")
    print("="*60)

    for ds, questions in TESTS.items():
        print(f"\n{ds} ({len(questions)} questions):")
        acc, lat = run_test(api, ds, questions)
        # Compare with old results
        old = {"TriviaQA": 0.4653, "StrategyQA": 0.9901, "PopQA": 0.9406}
        print(f"  Accuracy: {acc:.2f} (old: {old.get(ds, 0):.2f})")
        print(f"  Latency: {lat:.1f}s")

if __name__ == "__main__":
    main()
