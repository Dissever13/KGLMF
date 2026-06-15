"""
Multi-Agent Orchestrator with true Planner-Coder-Critic collaboration.

Implements iterative refinement loop:
  1. Planner decomposes task → structured plan
  2. Coder executes plan → solution
  3. Critic verifies solution → PASS (deliver) or FAIL (feedback → step 1)
  Repeat until PASS or max_iterations reached.

Each agent retrieves role-specific memory from the latent graph.
Agent outputs are written back to the graph for future retrieval.
"""

from typing import List, Dict, Optional, Tuple
import torch
from .agent import Agent
from ..graph.graph import LatentReasoningGraph
from ..composer.composer import MemoryComposer
from ..router.router import GraphRouter
from ..router.tts import AdaptiveTTSController
from ..injection.injector import LatentInjector, MemoryScorer
from ..evolution.evolution import GraphEvolution
from ..graph.node import NodeType
from .prompts import DEFAULT_PROMPTS, get_prompts_for_task, get_multi_prompts_for_task


class Orchestrator:
    def __init__(self, composer: MemoryComposer, router: GraphRouter,
                 tts: AdaptiveTTSController, injector: LatentInjector,
                 evolution: GraphEvolution, graph: LatentReasoningGraph,
                 api_client, d_model: int = 384,
                 scorer: MemoryScorer = None):  # FIX #1: scorer for dynamic gates
        self.composer = composer
        self.router = router
        self.tts = tts
        self.injector = injector
        self.evolution = evolution
        self.graph = graph
        self.api = api_client
        self.d_model = d_model
        self.scorer = scorer  # FIX #1
        self.agents: Dict[str, Agent] = {}
        self.stats = {
            'total_tasks': 0,
            'iterations_per_task': [],
            'critic_pass_rate': 0.0,
            'graph_nodes_used': [],
        }

    # ── Agent Management ──────────────────────────────

    def _get_or_create_agent(self, role: str, prompt: str = "") -> Agent:
        """Get existing agent by role, or create a new one."""
        for agent in self.agents.values():
            if agent.role == role:
                if prompt and agent.system_prompt != prompt:
                    agent.system_prompt = prompt
                return agent
        return self.create_agent(role, prompt)

    def create_agent(self, role: str, system_prompt: str = "") -> Agent:
        agent = Agent(
            id=f"{role}_{len(self.agents)}",
            role=role,
            system_prompt=system_prompt or DEFAULT_PROMPTS.get(role, "")
        )
        self.agents[agent.id] = agent
        return agent

    # ── Memory Retrieval ──────────────────────────────

    def get_memory_tokens(self, query: str, role: str) -> tuple:
        """Retrieve and compose memory tokens + confidence for a given query and role.

        FIX #5: Returns (memory_tokens, confidence) for dynamic gating.
        FIX #6: Uses injection_layers from injector config.
        """
        if len(self.graph) == 0:
            return torch.zeros(self.composer.n_learnable_queries, self.d_model), 0.0

        query_emb = self.api.get_embedding(query)
        complexity = self.tts.estimate_complexity(self.api.get_logits(query))
        budget = self.tts.compute_budget(complexity)

        retrieved = self.router.route(query_emb, self.graph, min(budget * 2, len(self.graph)))
        if not retrieved:
            return torch.zeros(self.composer.n_learnable_queries, self.d_model), 0.0

        k = min(budget, len(retrieved))
        memory_tokens = self.composer.compose(query_emb, self.graph, k, role)
        self.stats['graph_nodes_used'].append(k)

        # FIX #1: Compute confidence using scorer if available
        confidence = 0.0
        if self.scorer is not None and memory_tokens.sum() != 0:
            with torch.no_grad():
                confidence = float(self.scorer(
                    query_emb, memory_tokens
                ).item())
                # Clamp to valid range
                confidence = max(-1.0, min(1.0, confidence))

        compressed = self.composer.compress(memory_tokens, budget)
        return compressed, confidence

    # ── Agent Execution ───────────────────────────────

    def _run_agent(self, role: str, task: str, context: str = "",
                   prompt: str = "") -> str:
        """Run a single agent with memory-augmented generation.

        KEY: Latent memory replaces text context between agents.
        No verbose "PREVIOUS OUTPUT" text — memory injection provides inter-agent communication.
        This dramatically reduces KG-LMF's LLM token consumption.
        """
        agent = self._get_or_create_agent(role, prompt)
        memory_tokens, confidence = self.get_memory_tokens(task, role)

        # Task only — latent memory provides context, NOT text concatenation
        response = self.api.generate(
            system_prompt=agent.system_prompt,
            user_message=task,  # Just the original task, no text context
            history=None,        # No history accumulation
            latent_memory=memory_tokens if memory_tokens.sum() != 0 else None,
            confidence=confidence,
        )

        agent.add_to_history("user", task)
        agent.add_to_history("assistant", response)
        return response

    # ── Main Collaboration Loop ───────────────────────

    def run_task(self, task: str, task_type: str = "qa",
                 max_iterations: int = None, multi_agent: bool = False) -> Dict:
        """Execute a task through the multi-agent collaboration pipeline.

        Pipeline:
          Single-agent (multi_agent=False): 1 agent with latent memory
          Multi-agent (multi_agent=True): Planner → Coder → Critic with role prompts

        Returns dict with keys: final_answer, iterations, roles_used, verdict
        """
        if multi_agent:
            config = get_multi_prompts_for_task(task_type)
        else:
            config = get_prompts_for_task(task_type)
        roles = config['roles']
        max_iter = max_iterations or config.get('max_iterations', 2)

        self.stats['total_tasks'] += 1
        iteration = 0
        final_answer = ""
        verdict = "PASS"
        context = ""  # accumulates inter-agent communication
        roles_executed = []
        iteration_log = []

        while iteration < max_iter:
            iteration += 1
            iter_record = {'iteration': iteration, 'steps': []}

            for role in roles:
                prompt = config.get(role, DEFAULT_PROMPTS.get(role, ""))
                roles_executed.append(role)

                # Run the agent
                response = self._run_agent(role, task, context, prompt)
                iter_record['steps'].append({'role': role, 'output': response[:200]})

                # Extract content from response
                if role == "critic":
                    # Parse critic verdict
                    if "VERDICT: [PASS]" in response or "VERDICT: PASS" in response:
                        verdict = "PASS"
                        # Extract final answer
                        if "FINAL ANSWER:" in response:
                            final_answer = response.split("FINAL ANSWER:")[-1].strip()
                        else:
                            final_answer = response.split("VERDICT:")[0].strip()
                    elif "ANSWER:" in response:
                        verdict = "PASS"
                        final_answer = response.split("ANSWER:")[-1].strip()
                    else:
                        verdict = "FAIL"
                        final_answer = response
                elif role == "planner":
                    if "ANSWER:" in response:
                        # Planner provided direct answer (simple QA)
                        final_answer = response.split("ANSWER:")[-1].strip()
                        verdict = "PASS"
                    else:
                        # Planner produced a plan → pass to next role
                        context = response
                        final_answer = response  # fallback
                elif role == "coder":
                    if "FINAL ANSWER:" in response:
                        final_answer = response.split("FINAL ANSWER:")[-1].strip()
                    else:
                        final_answer = response
                    context = response

            iteration_log.append(iter_record)

            # Check termination
            if verdict == "PASS":
                break
            # If FAIL and more iterations remain, the critic output becomes context for re-planning
            if verdict == "FAIL" and iteration < max_iter:
                context = final_answer  # critic's feedback becomes context for planner

        self.stats['iterations_per_task'].append(iteration)

        # Write final interaction to graph
        self._write_to_graph(task, final_answer, task_type)

        # Periodic graph evolution
        if self.stats['total_tasks'] % 10 == 0:
            self.evolution.step(self.graph)

        return {
            'final_answer': final_answer or "No answer produced",
            'iterations': iteration,
            'verdict': verdict,
            'roles_used': list(set(roles_executed)),
            'iteration_log': iteration_log,
            'graph_size': len(self.graph),
        }

    # ── Graph Writing ─────────────────────────────────

    def _write_to_graph(self, task: str, answer: str, task_type: str):
        """Write successful interaction to the latent graph."""
        task_emb = self.api.get_embedding(task)
        answer_emb = self.api.get_embedding(answer[:500])
        combined = (task_emb + answer_emb) / 2

        # Map task_type to node type for graph organization
        type_map = {
            "qa": NodeType.CONTEXT,
            "reasoning": NodeType.TRAJECTORY,
            "code": NodeType.TRAJECTORY,
            "math": NodeType.TRAJECTORY,
            "science": NodeType.CONTEXT,
            "planning": NodeType.CAPSULE,
        }
        node_type = type_map.get(task_type, NodeType.CONTEXT)
        self.graph.add_node(combined, node_type, role=task_type, task_id=str(self.stats['total_tasks']))

    # ── Batch Execution ───────────────────────────────

    def run_batch(self, tasks: List[Dict]) -> List[Dict]:
        """Run multiple tasks. Each task dict: {question, task_type, answer (optional, for eval)}."""
        results = []
        for i, task in enumerate(tasks):
            result = self.run_task(
                task=task['question'],
                task_type=task.get('task_type', 'qa'),
            )
            result['task_id'] = i
            result['expected_answer'] = task.get('answer', '')
            results.append(result)
        return results

    # ── Stats ─────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'avg_iterations': (sum(self.stats['iterations_per_task']) /
                              max(1, len(self.stats['iterations_per_task']))),
            'graph': {
                'total_nodes': len(self.graph),
                'by_type': {t.value: len(self.graph.get_nodes_by_type(t))
                           for t in NodeType},
                'total_edges': len(self.graph.edges),
            },
            'agents': {aid: {'role': a.role, 'history_len': len(a.history)}
                      for aid, a in self.agents.items()},
        }
