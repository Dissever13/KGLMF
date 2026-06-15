"""
Real baseline implementations for KG-LMF evaluation.
Replaces previous "style" approximations with faithful implementations.

Baselines:
  1. TextMemoryBM25 - Real BM25 retrieval with tokenized memory corpus
  2. ChainOfThought - Multi-step reasoning via explicit CoT prompting
  3. MultiAgentNoMemory - Multi-agent collaboration without memory augmentation
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
import re


# ═══════════════════════════════════════════════════════════
# 1. REAL BM25 TEXT MEMORY
# ═══════════════════════════════════════════════════════════

class TextMemoryBM25:
    """Real BM25-based text memory with tokenized retrieval.

    Stores past interactions as tokenized documents. On each query,
    retrieves the most relevant past interactions using BM25 scoring
    and prepends them to the prompt as context.
    """

    def __init__(self, max_memory: int = 100):
        self.max_memory = max_memory
        self.documents: List[str] = []          # raw text documents
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        return re.findall(r'\w+', text.lower())

    def add(self, query: str, response: str):
        """Add a query-response pair to memory."""
        doc = f"Q: {query}\nA: {response}"
        self.documents.append(doc)
        self.tokenized_corpus.append(self._tokenize(doc))

        # Prune old memories if over capacity
        if len(self.documents) > self.max_memory:
            self.documents.pop(0)
            self.tokenized_corpus.pop(0)

        # Rebuild BM25 index
        if len(self.tokenized_corpus) > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k most relevant past interactions."""
        if self.bm25 is None or len(self.documents) == 0:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k_idx if scores[i] > 0]

    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string from retrieved memories."""
        docs = self.retrieve(query, k)
        if not docs:
            return ""
        return "Relevant past interactions:\n" + "\n---\n".join(docs) + "\n\n"

    def __len__(self):
        return len(self.documents)


# ═══════════════════════════════════════════════════════════
# 2. CHAIN-OF-THOUGHT BASELINE
# ═══════════════════════════════════════════════════════════

class ChainOfThought:
    """Chain-of-Thought reasoning baseline with structured prompting.

    Implements the standard CoT approach: ask the model to think step-by-step
    before providing the final answer. No memory augmentation.
    """

    COT_SYSTEM_PROMPT = (
        "You are a careful reasoning assistant. For each question, follow these steps:\n"
        "1. Identify the key facts and knowledge needed\n"
        "2. Break down the reasoning into clear logical steps\n"
        "3. Verify each step before moving to the next\n"
        "4. Provide the final answer clearly\n\n"
        "IMPORTANT: Always show your step-by-step reasoning, then end with "
        "FINAL ANSWER: [your answer]"
    )

    QA_COT_PROMPT = (
        "Answer the following question. Think step by step:\n"
        "1. What type of question is this?\n"
        "2. What facts do I need to recall?\n"
        "3. What is the correct answer?\n\n"
        "End with: ANSWER: [your answer]"
    )

    CODE_COT_PROMPT = (
        "Solve this coding problem step by step:\n"
        "1. Understand the requirements and constraints\n"
        "2. Design the algorithm/approach\n"
        "3. Write the implementation\n"
        "4. Verify with test cases\n\n"
        "Provide the final code between ```python and ``` markers."
    )

    @staticmethod
    def build_prompt(task: str, task_type: str = "qa") -> Tuple[str, str]:
        """Build system + user prompts for CoT reasoning."""
        if task_type == "code":
            return ChainOfThought.COT_SYSTEM_PROMPT, (
                f"{ChainOfThought.CODE_COT_PROMPT}\n\nTask:\n{task}"
            )
        elif task_type == "reasoning":
            return ChainOfThought.COT_SYSTEM_PROMPT, (
                f"Reason step by step about this question:\n\n{task}"
            )
        else:  # qa, planning
            return ChainOfThought.COT_SYSTEM_PROMPT, (
                f"{ChainOfThought.QA_COT_PROMPT}\n\nQuestion: {task}"
            )

    @staticmethod
    def extract_answer(response: str) -> str:
        """Extract final answer from CoT response."""
        for marker in ["FINAL ANSWER:", "ANSWER:", "Final Answer:"]:
            if marker in response:
                return response.split(marker)[-1].strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[-1] if lines else response.strip()


# ═══════════════════════════════════════════════════════════
# 3. MULTI-AGENT WITHOUT MEMORY
# ═══════════════════════════════════════════════════════════

class MultiAgentNoMemory:
    """Multi-agent collaboration baseline WITHOUT memory augmentation.

    Uses the same Planner-Coder-Critic pipeline as KG-LMF but without:
    - Latent reasoning graph
    - Memory retrieval/composition
    - Graph evolution
    - Adaptive TTS

    This provides a clean ablation: the multi-agent structure alone vs + memory.
    """

    MA_SYSTEM_PROMPTS = {
        "planner": (
            "You are a planning specialist. Break down the task into clear sub-steps. "
            "For simple questions, answer directly. Output format:\n"
            "PLAN: [numbered steps] or ANSWER: [direct answer]"
        ),
        "coder": (
            "You are an implementation specialist. Execute the plan provided. "
            "For coding tasks, write clean Python code. For non-code tasks, provide "
            "the answer directly. Output format:\n"
            "EXECUTION: [step results]\nFINAL ANSWER: [answer]"
        ),
        "critic": (
            "You are a quality assurance specialist. Review the answer against the "
            "original question. If correct, restate it. If not, provide the corrected "
            "answer. Output format:\n"
            "VERDICT: [PASS/FAIL]\nFINAL ANSWER: [verified answer]"
        ),
    }

    @staticmethod
    def get_prompts(task_type: str = "qa") -> Dict:
        """Get role configurations for different task types."""
        if task_type == "qa":
            return {"roles": ["planner"], "max_iterations": 1}
        elif task_type == "reasoning":
            return {"roles": ["planner", "critic"], "max_iterations": 2}
        elif task_type == "code":
            return {"roles": ["planner", "coder", "critic"], "max_iterations": 3}
        else:
            return {"roles": ["planner", "coder"], "max_iterations": 2}


# ═══════════════════════════════════════════════════════════
# BASELINE RUNNER
# ═══════════════════════════════════════════════════════════

def run_nomemory(api, question: str) -> str:
    """Simple single-call baseline with no memory."""
    return api.generate(
        "Answer the question directly and concisely. Output only the answer.",
        question
    ).strip()


def run_textmemory(api, question: str, memory: TextMemoryBM25, k: int = 3) -> str:
    """TextMemory baseline with BM25 context retrieval."""
    context = memory.get_context(question, k)
    if context:
        prompt = f"{context}Based on the above context and your knowledge, answer:\n{question}"
    else:
        prompt = question
    response = api.generate(
        "Answer the question directly and concisely. Output only the answer.",
        prompt
    ).strip()
    return response


def run_cot(api, question: str, task_type: str = "qa") -> str:
    """Chain-of-Thought baseline."""
    system, user = ChainOfThought.build_prompt(question, task_type)
    response = api.generate(system, user).strip()
    return ChainOfThought.extract_answer(response)


def run_ma_nomemory(api, question: str, task_type: str = "qa") -> str:
    """Multi-agent without memory baseline."""
    config = MultiAgentNoMemory.get_prompts(task_type)
    prompts = MultiAgentNoMemory.MA_SYSTEM_PROMPTS
    roles = config['roles']
    max_iter = config['max_iterations']

    answer = ""
    context = ""
    for iteration in range(max_iter):
        for role in roles:
            prompt = prompts.get(role, "")
            full_task = question
            if context and role != "planner":
                full_task = f"Previous output:\n{context}\n\nTask: {question}"
            response = api.generate(prompt, full_task).strip()
            context = response

            if "FINAL ANSWER:" in response:
                answer = response.split("FINAL ANSWER:")[-1].strip()
            elif "ANSWER:" in response:
                answer = response.split("ANSWER:")[-1].strip()

        if answer:
            break

    return answer or context


def evaluate_all_baselines(api, samples: List[Dict], eval_fn) -> Dict:
    """Run all baselines on the same samples and return comparative results."""
    memory = TextMemoryBM25(max_memory=200)
    results = {
        'nomemory': [],
        'textmemory': [],
        'cot': [],
        'ma_nomemory': [],
    }

    for sample in samples:
        q = sample['question']
        task_type = sample.get('task_type', 'qa')
        expected = sample['answer']

        # NoMemory
        pred_nm = run_nomemory(api, q)
        results['nomemory'].append({
            'predicted': pred_nm, 'score': eval_fn(pred_nm, expected)
        })

        # TextMemory
        pred_tm = run_textmemory(api, q, memory)
        results['textmemory'].append({
            'predicted': pred_tm, 'score': eval_fn(pred_tm, expected)
        })

        # Chain-of-Thought
        pred_cot = run_cot(api, q, task_type)
        results['cot'].append({
            'predicted': pred_cot, 'score': eval_fn(pred_cot, expected)
        })

        # Multi-agent NoMemory
        pred_ma = run_ma_nomemory(api, q, task_type)
        results['ma_nomemory'].append({
            'predicted': pred_ma, 'score': eval_fn(pred_ma, expected)
        })

        # Update memory
        memory.add(q, pred_nm)

    return results
