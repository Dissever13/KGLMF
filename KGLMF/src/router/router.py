import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from ..graph.graph import LatentReasoningGraph

class GraphRouter(nn.Module):
    def __init__(self, d_model: int = 384, temperature: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.score_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.score_proj.weight)

    def score_nodes(self, query: torch.Tensor,
                    graph: LatentReasoningGraph) -> torch.Tensor:
        # Ensure query is on the same device as the router weights
        device = next(self.score_proj.parameters()).device
        query = query.to(device)
        projected_query = self.score_proj(query)
        all_embs = graph.get_all_embeddings().to(device)
        if all_embs.shape[0] == 0:
            return torch.empty(0, device=device)
        query_norm = F.normalize(projected_query.unsqueeze(0), dim=-1)
        embs_norm = F.normalize(all_embs, dim=-1)
        logits = torch.matmul(query_norm, embs_norm.T).squeeze(0)
        return torch.softmax(logits / self.temperature, dim=-1)

    def route(self, query: torch.Tensor, graph: LatentReasoningGraph,
              k: int) -> List[Tuple[str, float, torch.Tensor]]:
        scores = self.score_nodes(query, graph)
        node_ids = graph.get_node_ids()
        if len(node_ids) == 0:
            return []
        k = min(k, len(node_ids))
        top_scores, top_idx = torch.topk(scores, k)
        results = []
        for j, idx in enumerate(top_idx):
            nid = node_ids[idx]
            node = graph.get_node(nid)
            node.access_count += 1
            results.append((nid, top_scores[j].item(), node.embedding))

        # FIX #4: Edge-aware neighbor expansion
        results = self._expand_neighbors(results, graph, k)

        return results

    def _expand_neighbors(self, results: list, graph: LatentReasoningGraph,
                          k: int) -> list:
        """FIX #4: Expand retrieved nodes with 1-hop graph neighbors.

        Graph edges encode meaningful relationships. Nodes connected by edges
        to top-scoring nodes are likely relevant even if not top-k by similarity.
        This makes the graph topology actually matter for retrieval.
        """
        seen_ids = {r[0] for r in results}
        expanded = list(results)

        for nid, score, emb in results:
            # Get neighbors via outgoing edges
            if nid in graph.edges:
                for neighbor_id, edge in graph.edges[nid].items():
                    if neighbor_id not in seen_ids and len(expanded) < k * 2:
                        neighbor = graph.get_node(neighbor_id)
                        if neighbor is not None:
                            # Neighbor score = edge_weight × source score
                            neighbor_score = edge.weight * score * 0.8
                            expanded.append((neighbor_id, neighbor_score, neighbor.embedding))
                            seen_ids.add(neighbor_id)

        # Sort by score, keep top results
        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded[:max(k, len(results))]
