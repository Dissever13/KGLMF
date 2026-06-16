import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from ..graph.graph import LatentReasoningGraph
from ..graph.node import NodeType
from ..graph.ops import GraphOps

class MemoryComposer(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 32,
                 n_layers: int = 4, n_learnable_queries: int = 16,
                 role_types: List[str] = None):
        super().__init__()
        self.d_model = d_model
        self.n_learnable_queries = n_learnable_queries
        self.role_types = role_types or ["planner", "coder", "critic"]

        self.pool_queries = nn.Parameter(
            torch.randn(1, n_learnable_queries, d_model) * 0.02)
        self.role_to_idx = {r: i for i, r in enumerate(self.role_types)}
        self.role_embeddings = nn.Embedding(len(self.role_types), d_model)
        nn.init.normal_(self.role_embeddings.weight, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.role_proj = nn.Linear(d_model, d_model, bias=False)
        self.graph_ops = GraphOps()

    def write(self, role: str, capability_ctx: torch.Tensor,
              trajectory: torch.Tensor, knowledge: torch.Tensor,
              graph: LatentReasoningGraph, task_id: str = "") -> str:
        combined = torch.stack([
            capability_ctx, trajectory, knowledge
        ], dim=0).unsqueeze(0)
        latent_tokens = F.scaled_dot_product_attention(
            self.pool_queries, combined, combined
        ).squeeze(0)
        node_type = self._role_to_node_type(role)
        return graph.add_node(latent_tokens, node_type, role=role, task_id=task_id)

    def compose(self, query: torch.Tensor, graph: LatentReasoningGraph,
                k: int, role: str = "planner") -> torch.Tensor:
        # Ensure query on same device as composer weights
        device = next(self.role_embeddings.parameters()).device
        query = query.to(device)

        role_idx = self.role_to_idx.get(role, 0)
        role_emb = self.role_embeddings(
            torch.tensor([role_idx], device=device))
        role_query = query + self.role_proj(role_emb).squeeze(0)

        node_ids = graph.get_node_ids()
        if not node_ids:
            return torch.zeros(self.n_learnable_queries, self.d_model,
                              device=device)

        all_embeddings = graph.get_all_embeddings().to(device)
        scores = self.graph_ops.scoring_function(role_query, all_embeddings)
        top_k = self.graph_ops.top_k_retrieval(scores, node_ids, k)

        retrieved_tokens = []
        for nid, _ in top_k:
            node = graph.get_node(nid)
            tokens = node.get_latent_tokens().to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            retrieved_tokens.append(tokens)

        if not retrieved_tokens:
            return torch.zeros(self.n_learnable_queries, self.d_model,
                              device=device)

        retrieved_embs = torch.cat(retrieved_tokens, dim=0).unsqueeze(0)

        # Ensure pool_queries on correct device
        pool_q = self.pool_queries.to(device)
        composed, _ = self.cross_attn(
            pool_q, retrieved_embs, retrieved_embs)
        return composed.squeeze(0)

    def compress(self, memories: torch.Tensor, budget: int) -> torch.Tensor:
        n_memories = memories.shape[0]
        if n_memories <= budget:
            return memories
        compress_queries = self.pool_queries[:, :budget, :]
        compressed = F.scaled_dot_product_attention(
            compress_queries, memories.unsqueeze(0), memories.unsqueeze(0))
        return compressed.squeeze(0)

    def _role_to_node_type(self, role: str) -> NodeType:
        return {"planner": NodeType.ROLE, "coder": NodeType.TRAJECTORY,
                "critic": NodeType.CONTEXT}.get(role, NodeType.CAPSULE)
