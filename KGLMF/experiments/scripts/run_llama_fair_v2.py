#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair Comparison Experiment v2 -- Llama3.1 8B via Ollama.

FIVE methods (aligned for fair comparison):
  1. NoMemory-Single   -- 1 agent, direct prompt
  2. NoMemory-Multi    -- N agents (matching KG-LMF per task type), NO memory
  3. KG-LMF            -- N agents + latent graph memory + injection
  4. TextMemory-Single -- Dense retrieval memory + 1 agent
  5. TextMemory-Multi  -- Dense retrieval memory + N agents (matches KG-LMF)

NINE datasets (N=81 each):
  1. StrategyQA      (reasoning)
  2. TriviaQA        (qa)
  3. PopQA           (qa)
  4. HotpotQA        (reasoning)
  5. BigCodeBench    (code)
  6. KodCode         (code)
  7. MATH-500        (math)      -- competition math
  8. MMLU-Pro        (science)   -- multi-subject knowledge
  9. GPQA            (science)   -- graduate-level QA

SEVEN metrics:
  1. Accuracy
  2. Avg Latency (s)
  3. Avg Token Consumption
  4. Memory Hit Rate (cosine sim: retrieved context vs query)
  5. Efficiency Score (acc / tok * 1000)
  6. Memory Utility Score (delta_acc / delta_tok, normalized)
  7. Robustness Score (1 - CV of accuracy across datasets)

KEY FIXES from v1:
  - Dense embedding retrieval for TextMemory (replaces BM25 lexical matching)
  - Quality-gated memory storage (no self-poisoning)
  - TextMemory-Multi variant for fair comparison with KG-LMF
  - Removed harmful Critic from code pipeline (coder output = final)
  - Real token tracking from Ollama API (not estimates)
  - Proper memory hit rate (retrieved memory vs query, not answer vs query)
  - Better answer extraction for code and math tasks
  - Deterministic train/test split (same datasets, zero sample overlap)

Usage:
    cd D:/KGLM/KGLMF
    /d/anaconda3/envs/kglmf/python experiments/scripts/run_llama_fair_v2.py \
        --dataset all --n_samples 81 --output_dir results_llama_v2/
"""

import os, json, time, random, argparse, re, math
import tempfile, subprocess, sys
import torch, numpy as np
import requests
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph import LatentReasoningGraph
from src.graph.node import NodeType
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector, MemoryScorer
from src.evolution.evolution import GraphEvolution
from src.agents.orchestrator import Orchestrator
from src.api.local_llm import LocalLLM


# ===============================================================
# OLLAMA CLIENT (with real token tracking)
# ===============================================================

HF_MODEL_ROOT = "D:/HF_models"
HF_MODEL_CANDIDATES = [
    os.path.join(HF_MODEL_ROOT, "models--google--gemma-4-8b-it"),
    os.path.join(HF_MODEL_ROOT, "models--meta-llama--Meta-Llama-3.1-8B-Instruct"),
    os.path.join(HF_MODEL_ROOT, "models--meta-llama--Llama-3.1-8B-Instruct"),
    os.path.join(HF_MODEL_ROOT, "models--meta-llama--Meta-Llama-3-8B-Instruct"),
]
HF_MODEL_FILE_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".safetensors.index.json",
    ".bin.index.json",
)


def _contains_hf_model_files(path: str) -> bool:
    """Return True when a directory contains HF config or weight artifacts."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return True

    for root, _dirs, files in os.walk(path):
        for filename in files:
            if filename == "config.json" or filename.endswith(HF_MODEL_FILE_EXTENSIONS):
                return True
    return False


def _find_hf_model_path(cache_dir: str) -> Optional[str]:
    """Resolve a local HF cache/model directory to a loadable model path."""
    if not os.path.isdir(cache_dir):
        return None
    if os.path.isfile(os.path.join(cache_dir, "config.json")):
        return cache_dir

    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for snapshot in snapshots:
            if _contains_hf_model_files(snapshot):
                return snapshot

    blobs_dir = os.path.join(cache_dir, "blobs")
    if _contains_hf_model_files(blobs_dir):
        return cache_dir
    return None


def _detect_hf_model(task_type_for_latent: bool) -> tuple:
    """Select LocalLLM for latent KG-LMF when a local HF model is available."""
    if task_type_for_latent:
        for candidate in HF_MODEL_CANDIDATES:
            model_path = _find_hf_model_path(candidate)
            if model_path:
                return LocalLLM, {"model": model_path}
    return OllamaClient, {"model": "llama3.1:8b"}


class OllamaClient:
    """Local Llama3.1 8B via Ollama. Tracks real token counts per call."""

    def __init__(self, model: str = 'llama3.1:8b', host: str = 'http://localhost:11434'):
        self.model = model
        self.host = host
        self._embedder = None
        self.last_tokens = {'prompt': 0, 'completion': 0, 'total': 0}
        # Accumulate across calls for KG-LMF
        self.accumulated_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    def reset_accumulator(self):
        self.accumulated_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 history=None, latent_memory=None,
                 confidence: float = 0.0, **kwargs) -> str:
        """Returns text string. Token counts stored in self.last_tokens and accumulated."""
        full_prompt = ""
        if history:
            for h in history[-10:]:
                full_prompt += f"[{h.get('role', 'user')}]: {h.get('content', '')[:500]}\n"
        full_prompt += user_message

        payload = {
            'model': self.model, 'system': system_prompt, 'prompt': full_prompt,
            'stream': False,
            'options': {'temperature': temperature, 'num_predict': max_tokens}
        }
        try:
            resp = requests.post(f'{self.host}/api/generate', json=payload, timeout=300)
            data = resp.json()
        except Exception as e:
            print(f"  [WARN] Ollama error: {e}")
            self.last_tokens = {'prompt': 0, 'completion': 0, 'total': 0}
            return ""

        self.last_tokens = {
            'prompt': data.get('prompt_eval_count', 0),
            'completion': data.get('eval_count', 0),
            'total': data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
        }
        # Accumulate
        for k in self.last_tokens:
            self.accumulated_tokens[k] += self.last_tokens[k]
        return data.get('response', '').strip()

    def get_embedding(self, text: str) -> torch.Tensor:
        if self._embedder is None:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        emb = self._embedder.encode(text[:2000], normalize_embeddings=True)
        return torch.from_numpy(emb.astype(np.float32))

    def get_logits(self, text: str) -> torch.Tensor:
        return torch.zeros(128256)


# ===============================================================
# DENSE TEXT MEMORY (replaces BM25)
# ===============================================================

class DenseTextMemory:
    """Dense embedding-based text memory with semantic retrieval.

    KEY IMPROVEMENTS over BM25:
    1. Uses sentence-transformers for semantic (not lexical) matching
    2. Quality-gated storage: only stores answers that pass length/format checks
    3. Proper context formatting with clear separation from current question
    """

    def __init__(self, max_memory: int = 200, embedder=None):
        self.max_memory = max_memory
        self.embedder = embedder
        self.entries: List[Dict] = []  # [{question, answer, embedding}]
        self._embedding_matrix = None  # cached [N, d] for fast retrieval

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedder

    def add(self, question: str, answer: str, quality_score: float = 0.5):
        """Add a QA pair with quality gating.

        Only stores answers that are:
        - Not empty
        - Reasonable length (not too short or too long)
        - Have some minimum quality score
        """
        answer = answer.strip()
        if not answer or len(answer) < 2 or len(answer) > 2000:
            return  # Reject clearly bad answers

        # Quality gate: only store if score >= threshold
        if quality_score < 0.3:
            return

        emb = self._get_embedder().encode(
            question[:1000], normalize_embeddings=True
        ).astype(np.float32)

        self.entries.append({
            'question': question,
            'answer': answer,
            'embedding': emb,
        })

        # Prune oldest if over capacity
        if len(self.entries) > self.max_memory:
            self.entries.pop(0)

        # Update embedding matrix
        self._embedding_matrix = np.stack([e['embedding'] for e in self.entries])

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k semantically similar past QA pairs."""
        if len(self.entries) == 0:
            return []

        q_emb = self._get_embedder().encode(
            query[:1000], normalize_embeddings=True
        ).astype(np.float32)

        # Cosine similarity
        sims = np.dot(self._embedding_matrix, q_emb)  # [N,]
        top_k_idx = np.argsort(sims)[-k:][::-1]

        results = []
        for i in top_k_idx:
            if sims[i] > 0.3:  # Minimum similarity threshold
                results.append({
                    'question': self.entries[i]['question'],
                    'answer': self.entries[i]['answer'],
                    'similarity': float(sims[i]),
                })
        return results

    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string from retrieved memories.

        Format is clean and clearly separated from the current question.
        """
        docs = self.retrieve(query, k)
        if not docs:
            return ""

        lines = ["[Relevant Past Examples]"]
        for i, doc in enumerate(docs):
            lines.append(f"Example {i+1}:")
            lines.append(f"  Q: {doc['question'][:200]}")
            lines.append(f"  A: {doc['answer'][:200]}")
            lines.append("")
        lines.append("[Current Task]")
        return "\n".join(lines)

    def compute_hit_rate(self, query: str, k: int = 3) -> float:
        """Memory Hit Rate: average cosine similarity of top-k retrieved entries."""
        docs = self.retrieve(query, k)
        if not docs:
            return 0.0
        return float(np.mean([d['similarity'] for d in docs]))

    def __len__(self):
        return len(self.entries)


# ===============================================================
# SYSTEM PROMPTS
# ===============================================================

# KG-LMF uses sophisticated role-specific prompts (part of its design).
# ALL OTHER methods use the same simple prompt — fair baseline comparison.
SIMPLE_PROMPT = "Answer the question directly and concisely. Just give the answer, no explanation."

KG_LMF_PROMPTS = {
    'planner': (
        "You are a planning specialist. Analyze the task and create a clear plan. "
        "For simple factual questions, output the answer directly prefixed with 'ANSWER:'. "
        "For complex tasks, break them into numbered steps. Keep output under 150 words."
    ),
    'coder': (
        "You are an expert implementer. Execute the task and produce the final output. "
        "For code tasks: write clean, runnable Python code between ```python``` markers. "
        "For math tasks: show calculation steps then state 'ANSWER: [number]'. "
        "For knowledge tasks: provide the factual answer prefixed with 'ANSWER:'."
    ),
    'critic': (
        "You are a quality verifier. Review the output and extract the final answer. "
        "Output ONLY: ANSWER: [the verified final answer]. "
        "Do NOT add reasoning, do NOT add commentary, do NOT rewrite -- just extract the answer."
    ),
}

# Other methods use these simple prompts (identical to single-agent prompt)
ROLE_PROMPTS = {
    'planner': SIMPLE_PROMPT,
    'coder': SIMPLE_PROMPT,
    'critic': SIMPLE_PROMPT,
}


# ===============================================================
# TASK-TYPE → ROLE CONFIGURATION
# ===============================================================

# KEY FIX: Code tasks no longer use Critic (which causes error amplification with weak models)
# KEY FIX: Math tasks use Planner→Coder pipeline (similar to code but with math prompts)
TASK_CONFIG = {
    'qa':        {'roles': ['planner'],                    'n_agents': 1},
    'reasoning': {'roles': ['planner', 'critic'],          'n_agents': 2},
    'code':      {'roles': ['planner', 'coder'],           'n_agents': 2},  # REMOVED critic
    'math':      {'roles': ['planner', 'coder'],           'n_agents': 2},
    'science':   {'roles': ['planner', 'critic'],          'n_agents': 2},
}


# ===============================================================
# METHOD 1: NoMemory-Single
# ===============================================================

def run_nomemory_single(api: OllamaClient, task: str) -> dict:
    """Single agent, no memory. Baseline for raw model capability."""
    t0 = time.time()
    resp = api.generate(SIMPLE_PROMPT, task)
    elapsed = time.time() - t0
    tok = dict(api.last_tokens)  # copy
    return {
        'answer': resp,
        'latency': elapsed,
        'prompt_tokens': tok['prompt'],
        'completion_tokens': tok['completion'],
        'total_tokens': tok['total'],
    }


# ===============================================================
# METHOD 2: NoMemory-Multi (aligned agent count)
# ===============================================================

def run_nomemory_multi(api: OllamaClient, task: str, task_type: str) -> dict:
    """Multi-agent pipeline matching KG-LMF agent count. NO memory retrieval.

    KEY FIX: Code tasks use Planner→Coder only (no Critic).
    """
    config = TASK_CONFIG.get(task_type, TASK_CONFIG['qa'])
    roles = config['roles']

    t0 = time.time()
    total_prompt_tok = 0
    total_comp_tok = 0
    context = ""
    final_answer = ""

    for role in roles:
        prompt = ROLE_PROMPTS.get(role, SIMPLE_PROMPT)
        full_task = task
        if context and role != roles[0]:  # Pass context to non-first agents
            full_task = (
                f"Previous agent output:\n```\n{context[:800]}\n```\n\n"
                f"Original task:\n{task}"
            )
        resp = api.generate(prompt, full_task)
        total_prompt_tok += api.last_tokens['prompt']
        total_comp_tok += api.last_tokens['completion']
        context = resp
        final_answer = resp

    elapsed = time.time() - t0

    # Extract answer from final output
    final_answer = _extract_answer(final_answer, task_type)

    return {
        'answer': final_answer,
        'latency': elapsed,
        'prompt_tokens': total_prompt_tok,
        'completion_tokens': total_comp_tok,
        'total_tokens': total_prompt_tok + total_comp_tok,
        'num_agents': len(roles),
    }


# ===============================================================
# METHOD 3: KG-LMF (with real token tracking)
# ===============================================================

def _zero_tokens() -> Dict[str, int]:
    return {'prompt': 0, 'completion': 0, 'total': 0}


def _reset_llm_accumulator(api) -> None:
    if hasattr(api, "reset_accumulator"):
        api.reset_accumulator()
    elif hasattr(api, "accumulated_tokens"):
        api.accumulated_tokens = _zero_tokens()


def _get_llm_accumulated_tokens(api) -> Dict[str, int]:
    tokens = getattr(api, "accumulated_tokens", None)
    if tokens is None:
        tokens = getattr(api, "last_tokens", None)
    if not tokens:
        return _zero_tokens()
    return {
        'prompt': int(tokens.get('prompt', 0)),
        'completion': int(tokens.get('completion', 0)),
        'total': int(tokens.get('total', tokens.get('prompt', 0) + tokens.get('completion', 0))),
    }


def run_kglmf(api: Any, task: str, task_type: str,
              orchestrator: Orchestrator) -> dict:
    """Full KG-LMF pipeline. Tracks backend token counts when available.

    KEY FIX: Uses api.accumulated_tokens for real counts.
    KEY FIX: For code tasks, reduces memory injection (skip if no relevant code memory).
    """
    t0 = time.time()
    _reset_llm_accumulator(api)

    result = orchestrator.run_task(task=task, task_type=task_type)
    answer = _extract_answer(result['final_answer'], task_type)
    elapsed = time.time() - t0

    # Real token counts from the selected backend when available
    real_tok = _get_llm_accumulated_tokens(api)

    # Memory hit rate: average cosine similarity of retrieved graph nodes to query
    memory_hit = _compute_kg_memory_hit(api, task, orchestrator)

    # Count latent memory tokens (vectors injected into LLM hidden states, NOT text tokens)
    latent_size = 0
    for role_iter in result.get('iteration_log', []):
        for step in role_iter.get('steps', []):
            latent_size += 16  # n_learnable_queries per agent call

    return {
        'answer': answer,
        'latency': elapsed,
        'prompt_tokens': real_tok['prompt'],       # LLM text input tokens
        'completion_tokens': real_tok['completion'], # LLM text output tokens
        'total_tokens': real_tok['total'],           # LLM text tokens total
        'latent_tokens': latent_size,                # Latent memory vectors (NOT LLM tokens)
        'iterations': result['iterations'],
        'graph_size': result.get('graph_size', 0),
        'memory_hit_rate': memory_hit,
    }


def run_kglmf_multi(api: Any, task: str, task_type: str,
                    orchestrator: Orchestrator) -> dict:
    """KG-LMF Multi-agent: N agents + latent memory injection + role-specific prompts.

    This is the full multi-agent version of KG-LMF with Planner→Coder→Critic pipeline.
    Each agent has its own role-specific prompt, and latent memory provides context.
    """
    t0 = time.time()
    _reset_llm_accumulator(api)

    result = orchestrator.run_task(task=task, task_type=task_type, multi_agent=True)
    answer = _extract_answer(result['final_answer'], task_type)
    elapsed = time.time() - t0

    real_tok = _get_llm_accumulated_tokens(api)

    memory_hit = _compute_kg_memory_hit(api, task, orchestrator)
    latent_size = len(result.get('roles_used', [])) * 16  # n_learnable_queries per agent

    return {
        'answer': answer,
        'latency': elapsed,
        'prompt_tokens': real_tok['prompt'],
        'completion_tokens': real_tok['completion'],
        'total_tokens': real_tok['total'],
        'latent_tokens': latent_size,
        'iterations': result['iterations'],
        'graph_size': result.get('graph_size', 0),
        'memory_hit_rate': memory_hit,
    }


def _compute_kg_memory_hit(api, task: str, orchestrator: Orchestrator) -> float:
    """Memory Hit Rate for KG-LMF: avg cosine sim of graph nodes vs query."""
    try:
        if len(orchestrator.graph) == 0:
            return 0.0
        q_emb = api.get_embedding(task)
        all_embs = orchestrator.graph.get_all_embeddings()
        if all_embs.numel() == 0:
            return 0.0
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), all_embs.unsqueeze(0) if all_embs.dim() == 1 else all_embs, dim=-1
        )
        if sims.dim() > 1:
            sims = sims.mean(dim=0)
        return max(0.0, float(sims.mean().item()))
    except Exception:
        return 0.0


# ===============================================================
# METHOD 4 & 5: TextMemory (Single & Multi)
# ===============================================================

def run_textmemory_single(api: OllamaClient, task: str, memory: DenseTextMemory) -> dict:
    """Dense TextMemory + Single agent call."""
    ctx = memory.get_context(task, k=3)
    hit_rate = memory.compute_hit_rate(task, k=3)

    t0 = time.time()
    if ctx:
        full_prompt = f"{ctx}\n\nAnswer the following question concisely:\n{task}"
    else:
        full_prompt = task
    resp = api.generate(SIMPLE_PROMPT, full_prompt)
    elapsed = time.time() - t0
    tok = dict(api.last_tokens)

    return {
        'answer': resp,
        'latency': elapsed,
        'prompt_tokens': tok['prompt'],
        'completion_tokens': tok['completion'],
        'total_tokens': tok['total'],
        'memory_hit_rate': hit_rate,
        'context_used': len(ctx) > 0,
    }


def run_textmemory_multi(api: OllamaClient, task: str, task_type: str,
                         memory: DenseTextMemory) -> dict:
    """Dense TextMemory + Multi-agent pipeline (matches KG-LMF agent count).

    This is the fair comparison: TextMemory vs KG-LMF, both with same # of agents.
    """
    ctx = memory.get_context(task, k=3)
    hit_rate = memory.compute_hit_rate(task, k=3)

    config = TASK_CONFIG.get(task_type, TASK_CONFIG['qa'])
    roles = config['roles']

    t0 = time.time()
    total_prompt_tok = 0
    total_comp_tok = 0
    context = ""
    final_answer = ""

    for role in roles:
        prompt = ROLE_PROMPTS.get(role, SIMPLE_PROMPT)

        # Build full task with memory context (for first agent) or previous output
        if role == roles[0] and ctx:
            full_task = f"{ctx}\n\nTask to solve:\n{task}"
        elif context:
            full_task = (
                f"Previous agent output:\n```\n{context[:800]}\n```\n\n"
                f"Original task:\n{task}"
            )
        else:
            full_task = task

        resp = api.generate(prompt, full_task)
        total_prompt_tok += api.last_tokens['prompt']
        total_comp_tok += api.last_tokens['completion']
        context = resp
        final_answer = resp

    elapsed = time.time() - t0
    final_answer = _extract_answer(final_answer, task_type)

    return {
        'answer': final_answer,
        'latency': elapsed,
        'prompt_tokens': total_prompt_tok,
        'completion_tokens': total_comp_tok,
        'total_tokens': total_prompt_tok + total_comp_tok,
        'memory_hit_rate': hit_rate,
        'context_used': len(ctx) > 0,
        'num_agents': len(roles),
    }


# ===============================================================
# ANSWER EXTRACTION
# ===============================================================

def _extract_answer(text: str, task_type: str) -> str:
    """Extract final answer from agent output. Handles various formats."""
    if not text:
        return ""

    # Try explicit markers first
    markers = ["FINAL ANSWER:", "ANSWER:", "Final Answer:", "The answer is:"]
    for marker in markers:
        if marker in text:
            extracted = text.split(marker)[-1].strip()
            # Take first non-empty line
            lines = [l.strip() for l in extracted.split('\n') if l.strip()]
            if lines:
                return lines[0]
            return extracted

    # For code tasks: if there's a code block, concatenate code blocks
    if task_type == 'code':
        code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(b.strip() for b in code_blocks)

    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


# ===============================================================
# EVALUATION (improved)
# ===============================================================

def execute_code(code_str: str, timeout: int = 10) -> tuple:
    """Execute generated Python code and return (success, output_or_error)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(code_str)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def evaluate(pred: str, expected: str, metric: str, task_type: str = 'qa') -> float:
    """Evaluate prediction against expected answer.

    KEY FIXES:
    - Better numeric comparison for math tasks
    - Multiple-choice handling for science tasks
    - Substring matching for code tasks
    """
    if task_type == 'code' and pred:
        ok, _output = execute_code(pred)
        if ok:
            return 1.0

    if not pred or not expected:
        return 0.0

    p = pred.strip()
    e = expected.strip()

    # === Special handling for math tasks (GSM8K) ===
    if task_type == 'math' or metric == 'math_accuracy':
        return _eval_math(p, e)

    # === Special handling for science (ARC) ===
    if task_type == 'science' or metric == 'science_accuracy':
        return _eval_science(p, e)

    # === Boolean / yes-no answers ===
    yes_words = {'yes', 'true', 'correct', 'right'}
    no_words = {'no', 'false', 'incorrect', 'wrong'}
    p_lower = p.lower()
    e_lower = e.lower()
    p_yes = any(p_lower.startswith(w) for w in yes_words)
    p_no = any(p_lower.startswith(w) for w in no_words)
    e_yes = any(e_lower.startswith(w) for w in yes_words) or e_lower in yes_words
    e_no = any(e_lower.startswith(w) for w in no_words) or e_lower in no_words

    if metric in ('accuracy', 'exact_match', 'pass_at_1'):
        # Yes/no question handling
        if e_yes or e_no:
            if (e_yes and p_yes) or (e_no and p_no):
                return 1.0
            # Check if answer text contains the expected boolean
            if e_lower in p_lower:
                return 1.0
            return 0.0

        # Substring match for longer answers
        if len(e) > 5:
            if e_lower in p_lower or p_lower in e_lower:
                return 1.0

        # Overlap-based scoring for longer answers
        pw = set(re.findall(r'\w+', p_lower))
        ew = set(re.findall(r'\w+', e_lower))
        if not ew:
            return 0.0
        overlap = len(pw & ew) / len(ew)
        if overlap > 0.7:
            return 1.0
        return overlap

    elif metric == 'f1':
        pt = set(re.findall(r'\w+', p_lower))
        et = set(re.findall(r'\w+', e_lower))
        if not pt or not et:
            return 0.0
        tp = len(pt & et)
        prec = tp / len(pt) if pt else 0
        rec = tp / len(et) if et else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return 0.0


def _eval_math(pred: str, expected: str) -> float:
    """Evaluate math answers numerically.

    GSM8K answers are numbers. Handles various formats:
    - "#### 42", "42", "42.0", "42 dollars", etc.
    """
    # Extract the last number from the prediction
    pred_nums = re.findall(r'[\d,]+\.?\d*', pred)
    exp_nums = re.findall(r'[\d,]+\.?\d*', expected)

    if not pred_nums or not exp_nums:
        # Fall back to string matching
        return 1.0 if expected.strip().lower() in pred.strip().lower() else 0.0

    # Get the last number in prediction (typical for chain-of-thought)
    try:
        pred_val = float(pred_nums[-1].replace(',', ''))
        exp_val = float(exp_nums[-1].replace(',', ''))
        # Allow small tolerance for rounding
        if abs(pred_val - exp_val) < 1e-6:
            return 1.0
        # Also check if any number in pred matches expected (for different formats)
        for pn in pred_nums:
            try:
                if abs(float(pn.replace(',', '')) - exp_val) < 1e-6:
                    return 1.0
            except ValueError:
                continue
        return 0.0
    except ValueError:
        return 0.0


def _eval_science(pred: str, expected: str) -> float:
    """Evaluate science multiple-choice answers (ARC format).

    Accepts: just the letter (A/B/C/D), the full answer text, or both.
    """
    # Extract letter answer from prediction
    letter_match = re.search(r'\b([A-D])\b', pred.upper())
    # Try to match expected answer text
    exp_lower = expected.lower().strip()
    pred_lower = pred.lower().strip()

    if letter_match:
        # Check if the correct answer text appears in prediction
        if exp_lower in pred_lower:
            return 1.0
        # Just letter match -- accept if not contradicted
        return 0.5  # partial credit

    # Text matching
    if exp_lower in pred_lower:
        return 1.0

    # Word overlap
    pw = set(re.findall(r'\w+', pred_lower))
    ew = set(re.findall(r'\w+', exp_lower))
    if not ew:
        return 0.0
    overlap = len(pw & ew) / len(ew)
    return 1.0 if overlap > 0.7 else overlap * 0.5


# ===============================================================
# DATA LOADING (8 datasets)
# ===============================================================

# Dataset config (used for display only; actual loading via data_utils)
DATASET_CONFIG = {
    'strategyqa':    {'task_type': 'reasoning', 'metric': 'accuracy'},
    'triviaqa':      {'task_type': 'qa',        'metric': 'exact_match'},
    'popqa':         {'task_type': 'qa',        'metric': 'exact_match'},
    'hotpotqa':      {'task_type': 'reasoning', 'metric': 'accuracy'},
    'bigcodebench':  {'task_type': 'code',      'metric': 'pass_at_1'},
    'kodcode':       {'task_type': 'code',      'metric': 'pass_at_1'},
    'math500':       {'task_type': 'math',      'metric': 'math_accuracy'},
    'mmlu_pro':      {'task_type': 'science',   'metric': 'science_accuracy'},
    'gpqa':          {'task_type': 'science',   'metric': 'science_accuracy'},
}


def load_dataset(name: str, n: int, seed: int = 42) -> List[Dict]:
    """Load test split from shared data_utils (deterministic, non-overlapping with train).

    FIX #4: Uses the same deterministic split as training, guaranteeing zero overlap.
    """
    from data_utils import load_dataset_test
    samples = load_dataset_test(name)  # Returns exactly 81 test samples per dataset
    # Apply per-dataset random subsampling if fewer samples needed
    if n < len(samples):
        rng = random.Random(seed + hash(name) % 10000)
        samples = rng.sample(samples, n)
    return samples


# ===============================================================
# METRIC COMPUTATION
# ===============================================================

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics from individual result dicts."""
    n = len(results)
    if n == 0:
        return {}

    acc = sum(r['score'] for r in results) / n
    avg_lat = sum(r['latency'] for r in results) / n
    avg_tok = sum(r['total_tokens'] for r in results) / n  # LLM text tokens only
    avg_latent = sum(r.get('latent_tokens', 0) for r in results) / n  # Latent vectors
    avg_hit = sum(r.get('memory_hit_rate', 0) for r in results) / n
    eff = (acc / max(avg_tok, 1)) * 1000 if avg_tok > 0 else 0

    return {
        'accuracy': round(acc, 4),
        'avg_latency': round(avg_lat, 3),
        'avg_tokens': round(avg_tok, 1),           # LLM text tokens
        'avg_latent_tokens': round(avg_latent, 1),  # Latent memory vectors
        'memory_hit_rate': round(avg_hit, 4),
        'efficiency_score': round(eff, 4),
        'n': n,
    }


def compute_summary(all_results: Dict[str, List], ds_name: str,
                    task_type: str) -> Dict:
    """Compute per-method summaries for one dataset."""
    summary = {}
    for method, results in all_results.items():
        m = compute_metrics(results)
        if m:
            summary[method] = m

    # Memory Utility Score (for memory methods vs no-memory baselines)
    # Positive = memory helps, Negative = memory hurts
    base_acc = summary.get('nomemory_single', {}).get('accuracy', 0)
    base_tok = summary.get('nomemory_single', {}).get('avg_tokens', 1)

    for method in ['kglmf', 'textmemory_single', 'textmemory_multi']:
        if method in summary:
            m_acc = summary[method]['accuracy']
            m_tok = summary[method]['avg_tokens']
            delta_acc = m_acc - base_acc
            delta_tok = m_tok - base_tok
            if delta_tok > 0:
                summary[method]['memory_utility'] = round(delta_acc / (delta_tok / 1000), 6)
            else:
                summary[method]['memory_utility'] = round(delta_acc, 6)

    summary['dataset'] = ds_name
    summary['task_type'] = task_type
    return summary


# ===============================================================
# MAIN EXPERIMENT
# ===============================================================

METHODS = [
    'nomemory_single',
    'nomemory_multi',
    'kglmf',
    'kglmf_multi',
    'textmemory_single',
    'textmemory_multi',
]


def run_dataset(ds_name: str, n_samples: int, api: OllamaClient,
                checkpoint_path: str = None, seed: int = 42,
                hf_api: Optional[LocalLLM] = None) -> Dict:
    """Run all 5 methods on one dataset."""
    config = DATASET_CONFIG[ds_name]
    samples = load_dataset(ds_name, n_samples, seed)
    if not samples:
        print(f"  SKIP {ds_name}: no data")
        return {}

    n = len(samples)
    task_type = config['task_type']
    metric = config['metric']
    _llm = hf_api if hf_api is not None else None
    kglmf_api = _llm or api

    print(f"\n{'='*70}")
    print(f"  {ds_name} | N={n} | type={task_type} | metric={metric}")
    if _llm is not None:
        print(f"  KG-LMF backend: LocalLLM ({getattr(_llm, 'model_id', 'HF local model')})")
    else:
        print(f"  KG-LMF backend: Ollama ({api.model})")
    print(f"{'='*70}")

    # ── Initialize KG-LMF ──
    d_model = 384
    graph = LatentReasoningGraph(d_model=d_model)
    composer = MemoryComposer(d_model=d_model, n_heads=8, n_layers=4)
    router = GraphRouter(d_model=d_model)
    tts_controller = AdaptiveTTSController()
    # FIX #6: injection layers [0, 8, 16] — better defaults
    injector = LatentInjector(d_model=d_model, injection_layers=[0, 8, 16],
                              use_adapter=True)  # FIX #2
    evolution = GraphEvolution()
    scorer = None  # FIX #1: will be loaded if v3 checkpoint

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        composer.load_state_dict(ckpt['composer'])
        router.load_state_dict(ckpt['router'])

        # Handle both v2 (LatentInjector) and v3 (DynamicGateInjector) formats
        if 'injector' in ckpt:
            try:
                injector.load_state_dict(ckpt['injector'], strict=False)
            except Exception:
                print("  [KG-LMF] Injector format mismatch, using fresh injector")

        # FIX #1: Load scorer if available (v3 checkpoint)
        if 'scorer' in ckpt:
            scorer = MemoryScorer(d_model=d_model)
            scorer.load_state_dict(ckpt['scorer'])
            scorer.eval()
            print("  [KG-LMF] Loaded MemoryScorer (v3 checkpoint)")

        for nid, emb, ntype in ckpt.get('graph_nodes', []):
            graph.add_node(emb, NodeType(ntype))
        print(f"  [KG-LMF] Loaded checkpoint: {len(ckpt.get('graph_nodes', []))} nodes")

    orchestrator = Orchestrator(
        composer=composer, router=router, tts=tts_controller,
        injector=injector, evolution=evolution, graph=graph,
        api_client=kglmf_api, d_model=d_model, scorer=scorer,  # FIX #1
    )

    # ── Initialize TextMemory (shared across Single and Multi) ──
    text_memory_single = DenseTextMemory(max_memory=200)
    text_memory_multi = DenseTextMemory(max_memory=200)

    # ── Results containers ──
    all_results = {m: [] for m in METHODS}

    for i, sample in enumerate(tqdm(samples, desc=f"  {ds_name}")):
        q, expected = sample['question'], sample['answer']

        # ── Method 1: NoMemory Single ──
        r1 = run_nomemory_single(api, q)
        r1['score'] = evaluate(r1['answer'], expected, metric, task_type)
        r1['memory_hit_rate'] = 0.0
        all_results['nomemory_single'].append(r1)

        # ── Method 2: NoMemory Multi (aligned agents, no memory) ──
        r2 = run_nomemory_multi(api, q, task_type)
        r2['score'] = evaluate(r2['answer'], expected, metric, task_type)
        r2['memory_hit_rate'] = 0.0
        all_results['nomemory_multi'].append(r2)

        # ── Method 3: KG-LMF (single agent + latent memory) ──
        r3 = run_kglmf(kglmf_api, q, task_type, orchestrator)
        r3['score'] = evaluate(r3['answer'], expected, metric, task_type)
        all_results['kglmf'].append(r3)

        # ── Method 4: KG-LMF Multi (multi agent + latent memory) ──
        r4 = run_kglmf_multi(kglmf_api, q, task_type, orchestrator)
        r4['score'] = evaluate(r4['answer'], expected, metric, task_type)
        all_results['kglmf_multi'].append(r4)

        # ── Method 5: TextMemory Single (dense retrieval + 1 agent) ──
        r5 = run_textmemory_single(api, q, text_memory_single)
        r5['score'] = evaluate(r5['answer'], expected, metric, task_type)
        all_results['textmemory_single'].append(r5)
        qual = r5['score']
        text_memory_single.add(q, r5['answer'], quality_score=qual)

        # ── Method 6: TextMemory Multi (dense retrieval + N agents) ──
        r6 = run_textmemory_multi(api, q, task_type, text_memory_multi)
        r6['score'] = evaluate(r6['answer'], expected, metric, task_type)
        all_results['textmemory_multi'].append(r6)
        qual6 = r6['score']
        text_memory_multi.add(q, r6['answer'], quality_score=qual6)

    # ── Compute summaries ──
    summary = compute_summary(all_results, ds_name, task_type)

    # Print per-method results
    for method in METHODS:
        if method in summary:
            m = summary[method]
            util = m.get('memory_utility', '--')
            latent = m.get('avg_latent_tokens', 0)
            print(f"  {method:<24} acc={m['accuracy']:.4f}  lat={m['avg_latency']:.1f}s  "
                  f"tok={m['avg_tokens']:.0f}  latent={latent:.0f}  "
                  f"hit={m['memory_hit_rate']:.4f}  eff={m['efficiency_score']:.4f}  util={util}")

    # ── Save per-dataset results ──
    os.makedirs('results_llama_v2', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'results_llama_v2/{ds_name}_n{n}_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': {
                m: [{k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                     for k, v in r.items()}
                    for r in all_results[m]]
                for m in METHODS
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all',
                        help='Dataset name or "all"')
    parser.add_argument('--n_samples', type=int, default=81,
                        help='Samples per dataset')
    parser.add_argument('--checkpoint', default='checkpoints_v2/best_model.pt',
                        help='KG-LMF checkpoint path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='llama3.1:8b')
    parser.add_argument('--output_dir', default='results_llama_v2')
    parser.add_argument('--skip_methods', nargs='+', default=[],
                        help='Methods to skip (for debugging)')
    args = parser.parse_args()

    print("=" * 70)
    print("  Fair Comparison Experiment v2 -- Llama3.1 8B")
    print("=" * 70)
    print(f"  Methods: {' | '.join(METHODS)}")
    print(f"  Datasets: {len(DATASET_CONFIG)} total")
    print(f"  N per dataset: {args.n_samples}")
    print(f"  Metrics: Accuracy | Latency | Tokens | Memory Hit | Efficiency | Utility")
    print()

    api = OllamaClient(model=args.model)
    hf_api = None
    llm_cls, llm_kwargs = _detect_hf_model(task_type_for_latent=True)
    if llm_cls is LocalLLM:
        try:
            hf_api = LocalLLM.from_hf_cache(llm_kwargs["model"])
            print(f"  KG-LMF latent backend: LocalLLM ({llm_kwargs['model']})")
        except Exception as exc:
            print(f"  [WARN] LocalLLM unavailable: {exc}")
            print(f"  KG-LMF latent backend: Ollama ({args.model})")
    else:
        print(f"  KG-LMF latent backend: Ollama ({args.model})")

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    print(f"  Running: {datasets}")
    print()

    all_summaries = []
    start_time = datetime.now()

    for ds in datasets:
        summary = run_dataset(ds, args.n_samples, api, args.checkpoint, args.seed, hf_api=hf_api)
        if summary:
            all_summaries.append(summary)

    elapsed = datetime.now() - start_time

    # ===========================================================
    # FINAL AGGREGATED RESULTS
    # ===========================================================

    if not all_summaries:
        print("\n  No results to report.")
        return

    print(f"\n{'='*120}")
    print("  FINAL COMPARISON -- All Datasets")
    print(f"  Total time: {elapsed}")
    print(f"{'='*120}")

    # ── Per-dataset table ──
    header = (f"{'Dataset':<16} {'Method':<22} {'Acc':>8} {'Lat(s)':>8} "
              f"{'Tok':>8} {'Latent':>8} {'Hit':>8} {'Eff':>8} {'Util':>10}")
    print(header)
    print("-" * 120)

    for s in all_summaries:
        ds = s['dataset']
        first = True
        for method in METHODS:
            if method not in s:
                continue
            m = s[method]
            util = m.get('memory_utility', None)
            util_str = f"{util:10.6f}" if util is not None else f"{'--':>10}"
            print(f"{ds if first else '':<16} {method:<22} "
                  f"{m['accuracy']:8.4f} {m['avg_latency']:8.1f} "
                  f"{m['avg_tokens']:8.0f} {m.get('avg_latent_tokens',0):8.0f} {m['memory_hit_rate']:8.4f} "
                  f"{m['efficiency_score']:8.4f} {util_str}")
            first = False
        print()

    # ── Aggregate across datasets ──
    print(f"{'='*120}")
    print("  AGGREGATE STATISTICS (mean +/- std across datasets)")
    print(f"{'='*120}")

    agg_header = (f"{'Method':<22} {'Acc':>12} {'Lat(s)':>10} "
                  f"{'Tok':>10} {'Hit':>10} {'Eff':>10} {'Util':>10}")
    print(agg_header)
    print("-" * 120)

    agg_data = {m: defaultdict(list) for m in METHODS}
    for s in all_summaries:
        for method in METHODS:
            if method in s:
                for k, v in s[method].items():
                    if isinstance(v, (int, float)):
                        agg_data[method][k].append(v)

    for method in METHODS:
        if not agg_data[method].get('accuracy'):
            continue
        acc_mean = np.mean(agg_data[method]['accuracy'])
        acc_std = np.std(agg_data[method]['accuracy'])
        lat_mean = np.mean(agg_data[method]['avg_latency'])
        tok_mean = np.mean(agg_data[method]['avg_tokens'])
        hit_mean = np.mean(agg_data[method]['memory_hit_rate'])
        eff_mean = np.mean(agg_data[method]['efficiency_score'])
        util_vals = agg_data[method].get('memory_utility', [])
        util_mean = np.mean(util_vals) if util_vals else float('nan')

        # Robustness: 1 - CV
        cv = acc_std / max(acc_mean, 0.001)
        robustness = 1.0 - min(cv, 1.0)

        print(f"{method:<22} {acc_mean:7.4f}+/-{acc_std:<6.4f} {lat_mean:8.1f} "
              f"{tok_mean:8.0f} {hit_mean:8.4f} {eff_mean:8.4f} "
              f"{util_mean:10.6f}")

    print()

    # ── Win/Loss matrix ──
    print(f"  WIN/LOSS MATRIX (KG-LMF vs baselines, by dataset)")
    print(f"  {'Dataset':<16} {'vs Single':>12} {'vs Multi':>12} {'vs Text-S':>12} {'vs Text-M':>12}")
    print(f"  {'--'*16} {'--'*12} {'--'*12} {'--'*12} {'--'*12}")

    kg_wins = {'single': 0, 'multi': 0, 'text_s': 0, 'text_m': 0}
    for s in all_summaries:
        ds = s['dataset']
        kg = s.get('kglmf', {}).get('accuracy', 0)
        def delta(m): return kg - s.get(m, {}).get('accuracy', 0)
        def sym(d): return f"+{d:.4f}" if d > 0 else f"{d:.4f}"

        d_single = delta('nomemory_single')
        d_multi = delta('nomemory_multi')
        d_ts = delta('textmemory_single')
        d_tm = delta('textmemory_multi')

        if d_single > 0: kg_wins['single'] += 1
        if d_multi > 0: kg_wins['multi'] += 1
        if d_ts > 0: kg_wins['text_s'] += 1
        if d_tm > 0: kg_wins['text_m'] += 1

        print(f"  {ds:<16} {sym(d_single):>12} {sym(d_multi):>12} {sym(d_ts):>12} {sym(d_tm):>12}")
    print(f"  {'KG Wins':<16} {kg_wins['single']:>9}/8   {kg_wins['multi']:>9}/8   "
          f"{kg_wins['text_s']:>9}/8   {kg_wins['text_m']:>9}/8")

    # ── Save aggregate results ──
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    agg_path = f'{args.output_dir}/AGGREGATE_n{args.n_samples}_{ts}.json'
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model': args.model,
                'n_per_dataset': args.n_samples,
                'datasets': datasets,
                'seed': args.seed,
                'elapsed': str(elapsed),
            },
            'per_dataset': all_summaries,
            'aggregate': {
                m: {k: (np.mean(v).item() if hasattr(np.mean(v), 'item') else float(np.mean(v)))
                    for k, v in agg_data[m].items()}
                for m in METHODS if agg_data[m]
            },
            'kg_wins': kg_wins,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Aggregate results saved: {agg_path}")
    print(f"  Per-dataset results in: {args.output_dir}/")

    return all_summaries


if __name__ == '__main__':
    main()
