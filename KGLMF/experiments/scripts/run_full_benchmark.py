"""KG-LMF Full Benchmark Runner — 101 samples × 6 datasets × 7 methods"""
import sys, os, time, json, argparse, random
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

SYS_QA = "Answer the question directly and factually in ONE concise sentence. For yes/no questions, start with Yes or No. Do NOT explain reasoning. Do NOT output internal thinking. Just give the correct answer."
SYS_CODE = "You are an expert programmer. Write correct, working Python code. Output ONLY the code. No explanations. No comments."
SYS_PLAN = "You are a planning expert. Generate a valid PDDL plan with proper syntax. Output ONLY the plan."

def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    print(f"  WARNING: {path} not found, using fallback")
    return _fallback_dataset(name, 101)

def _fallback_dataset(name, n):
    qas = [("What is the capital of France?", "Paris"),
           ("Who wrote Romeo and Juliet?", "William Shakespeare"),
           ("What is the largest ocean?", "Pacific"),
           ("What year did WWII end?", "1945"),
           ("What element has atomic number 1?", "Hydrogen")]
    return [{"id": f"{name}_{i%5}", "question": qas[i%5][0],
             "prompt": qas[i%5][0], "answer": qas[i%5][1],
             "type": "qa"} for i in range(n)]

# ============================================================
# BASELINE IMPLEMENTATIONS
# ============================================================

class BaselineRunner:
    def __init__(self, api, method_name):
        self.api = api
        self.method_name = method_name

    def run(self, dataset_name, data, max_samples=None):
        raise NotImplementedError

class NoMemoryBaseline(BaselineRunner):
    """Single-agent LLM call, no memory"""
    def run(self, ds_name, data, max_samples=None):
        results = []
        samples = data[:max_samples] if max_samples else data
        for i, item in enumerate(samples):
            q = item.get("question") or item.get("prompt", "")
            gt = str(item.get("answer", ""))
            dtype = item.get("type", "qa")
            sys_prompt = SYS_CODE if dtype == "code" else (SYS_PLAN if dtype == "planning" else SYS_QA)
            t0 = time.time()
            try:
                resp = self.api.generate(sys_prompt, q, max_tokens=1024)
            except Exception as e:
                resp = f"[ERROR: {e}]"
            lat = time.time() - t0
            score = 1.0 if gt.lower() in resp.lower() else 0.0
            results.append({"id": item["id"], "score": score, "latency": lat, "pred": resp[:200]})
            if (i+1) % 20 == 0:
                acc = sum(r["score"] for r in results) / len(results)
                print(f"    [{self.method_name}] {ds_name}: {i+1}/{len(samples)} acc={acc:.3f}")
        return self._aggregate(results)

    def _aggregate(self, results):
        acc = sum(r["score"] for r in results) / len(results) if results else 0
        avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0
        return {"accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2), "n": len(results)}

class TextMemoryBaseline(BaselineRunner):
    """Text memory with BM25 retrieval + LLM"""
    def run(self, ds_name, data, max_samples=None):
        from rank_bm25 import BM25Okapi
        results = []
        memory_bank = []
        memory_texts = []
        samples = data[:max_samples] if max_samples else data

        for i, item in enumerate(samples):
            q = item.get("question") or item.get("prompt", "")
            gt = str(item.get("answer", ""))
            dtype = item.get("type", "qa")
            sys_prompt = SYS_CODE if dtype == "code" else SYS_QA

            # Retrieve from memory
            context = ""
            if memory_texts:
                tokenized_corpus = [t.split() for t in memory_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = q.split()
                scores = bm25.get_scores(tokenized_query)
                top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:5]
                context = "\n".join([memory_texts[idx][:200] for idx in top_idx])

            augmented_q = f"Context from memory:\n{context}\n\nQuestion: {q}" if context else q

            t0 = time.time()
            try:
                resp = self.api.generate(sys_prompt, augmented_q, max_tokens=1024)
            except Exception as e:
                resp = f"[ERROR: {e}]"
            lat = time.time() - t0

            score = 1.0 if gt.lower() in resp.lower() else 0.0
            results.append({"id": item["id"], "score": score, "latency": lat, "pred": resp[:200]})

            # Add to memory bank
            memory_texts.append(f"Q: {q}\nA: {resp[:300]}\nGT: {gt}")
            memory_bank.append({"q": q, "a": resp, "gt": gt})

            if (i+1) % 20 == 0:
                acc = sum(r["score"] for r in results) / len(results)
                print(f"    [{self.method_name}] {ds_name}: {i+1}/{len(samples)} acc={acc:.3f}")

        return self._aggregate(results)

    def _aggregate(self, results):
        acc = sum(r["score"] for r in results) / len(results) if results else 0
        avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0
        return {"accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2), "n": len(results)}

class KGLMFBaseline(BaselineRunner):
    """Full KG-LMF framework"""
    def __init__(self, api, method_name, ablation=None):
        super().__init__(api, method_name)
        self.ablation = ablation  # None = full; str = what to disable

    def run(self, ds_name, data, max_samples=None):
        graph = LatentReasoningGraph(d_model=384)
        composer = MemoryComposer(d_model=384, n_learnable_queries=16)
        router = GraphRouter(d_model=384)
        tts = AdaptiveTTSController(budget_min=4, budget_max=32)
        injector = LatentInjector(d_model=384, injection_layers=[0, 8, 16])
        evolution = GraphEvolution(merge_threshold=0.85)

        # Apply ablation
        if self.ablation == "no_graph":
            graph = LatentReasoningGraph(d_model=384, max_nodes_per_type=10000)
        elif self.ablation == "no_role":
            composer = MemoryComposer(d_model=384, n_learnable_queries=16, role_types=["generic"])
        elif self.ablation == "no_tts":
            tts = AdaptiveTTSController(budget_min=16, budget_max=16)
        elif self.ablation == "no_evolution":
            evolution = None

        orch = Orchestrator(composer=composer, router=router, tts=tts,
                           injector=injector, evolution=evolution,
                           graph=graph, api_client=self.api, d_model=384)
        orch.agents = {}

        results = []
        samples = data[:max_samples] if max_samples else data
        dtype = "qa"
        roles = ["planner", "critic"]

        for i, item in enumerate(samples):
            q = item.get("question") or item.get("prompt", "")
            gt = str(item.get("answer", ""))
            dtype = item.get("type", "qa")
            if dtype == "code":
                roles = ["planner", "coder", "critic"]
            elif dtype == "planning":
                roles = ["planner", "coder"]
            else:
                roles = ["planner", "critic"]

            t0 = time.time()
            try:
                responses = orch.run_round(q, roles)
                resp = list(responses.values())[-1] if responses else ""
            except Exception as e:
                resp = f"[ERROR: {e}]"
            lat = time.time() - t0

            score = 1.0 if gt.lower() in resp.lower() else 0.0
            results.append({"id": item["id"], "score": score, "latency": lat, "pred": resp[:200]})

            if (i+1) % 20 == 0:
                acc = sum(r["score"] for r in results) / len(results)
                print(f"    [{self.method_name}] {ds_name}: {i+1}/{len(samples)} acc={acc:.3f}")

        acc = sum(r["score"] for r in results) / len(results) if results else 0
        avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0
        return {"accuracy": round(acc, 4), "avg_latency": round(avg_lat, 2), "n": len(results), "nodes": len(graph)}

# ============================================================
# MAIN RUNNER
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KG-LMF Full Benchmark")
    parser.add_argument("--datasets", nargs="*", default=["TriviaQA", "StrategyQA", "PopQA", "KodCode", "BigCodeBench", "PDDL"])
    parser.add_argument("--methods", nargs="*", default=["NoMemory", "TextMemory", "KG-LMF"])
    parser.add_argument("--max_samples", type=int, default=101)
    parser.add_argument("--output", default="experiments/results/full_benchmark.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing results if resuming
    all_results = {}
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', args.output)
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    api = DeepSeekClient(max_tokens=1024)
    print("="*70)
    print("KG-LMF FULL BENCHMARK")
    print(f"Model: {api.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Samples per dataset: {args.max_samples}")
    print(f"Total API calls: ~{len(args.datasets) * len(args.methods) * args.max_samples}")
    print("="*70)

    for ds_name in args.datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name}")
        print(f"{'#'*70}")

        data = load_dataset(ds_name)
        print(f"  Loaded {len(data)} samples")

        for method_name in args.methods:
            key = f"{ds_name}/{method_name}"
            if key in all_results:
                print(f"  [{method_name}] Already completed, skipping")
                continue

            print(f"\n  --- {method_name} ---")

            if method_name == "NoMemory":
                runner = NoMemoryBaseline(DeepSeekClient(max_tokens=1024), method_name)
            elif method_name == "TextMemory":
                runner = TextMemoryBaseline(DeepSeekClient(max_tokens=1024), method_name)
            elif method_name == "KG-LMF":
                runner = KGLMFBaseline(DeepSeekClient(max_tokens=1024), method_name)
            elif method_name == "KG-LMF-no-graph":
                runner = KGLMFBaseline(DeepSeekClient(max_tokens=1024), method_name, ablation="no_graph")
            elif method_name == "KG-LMF-no-role":
                runner = KGLMFBaseline(DeepSeekClient(max_tokens=1024), method_name, ablation="no_role")
            elif method_name == "KG-LMF-no-tts":
                runner = KGLMFBaseline(DeepSeekClient(max_tokens=1024), method_name, ablation="no_tts")
            else:
                print(f"    Unknown method: {method_name}, skipping")
                continue

            result = runner.run(ds_name, data, args.max_samples)
            result["method"] = method_name
            result["dataset"] = ds_name
            all_results[key] = result

            print(f"    => {method_name}: acc={result['accuracy']:.4f}, "
                  f"lat={result['avg_latency']:.1f}s")

            # Save incrementally
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    summary = {}
    for key, r in sorted(all_results.items()):
        ds, method = key.split("/")
        print(f"  {ds:15s} | {method:20s} | acc={r['accuracy']:.4f} | lat={r.get('avg_latency',0):.1f}s")
        if method not in summary:
            summary[method] = []
        summary[method].append(r["accuracy"])

    print(f"\n  Average by method:")
    for method, accs in sorted(summary.items()):
        avg = sum(accs) / len(accs)
        print(f"    {method:20s}: {avg:.4f}")

    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
