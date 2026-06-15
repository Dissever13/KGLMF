import torch
import torch.nn.functional as F
import time as time_mod
from ..graph.graph import LatentReasoningGraph
from ..graph.node import NodeType

class GraphEvolution:
    def __init__(self, merge_threshold: float = 0.85,
                 prune_percentile: float = 10.0,
                 temporal_decay_beta: float = 0.01,
                 max_nodes_per_type: int = 500):
        self.merge_threshold = merge_threshold
        self.prune_percentile = prune_percentile
        self.temporal_decay_beta = temporal_decay_beta
        self.max_nodes_per_type = max_nodes_per_type

    def merge_similar_nodes(self, graph: LatentReasoningGraph):
        for node_type in NodeType:
            typed_nodes = graph.get_nodes_by_type(node_type)
            if len(typed_nodes) < 2:
                continue
            merged = set()
            for i, ni in enumerate(typed_nodes):
                if ni.id in merged:
                    continue
                for j in range(i + 1, len(typed_nodes)):
                    nj = typed_nodes[j]
                    if nj.id in merged:
                        continue
                    sim = F.cosine_similarity(
                        ni.embedding.unsqueeze(0), nj.embedding.unsqueeze(0)
                    ).item()
                    if sim > self.merge_threshold:
                        total_util = ni.utility + nj.utility
                        ni.embedding = (ni.embedding * ni.utility +
                                       nj.embedding * nj.utility) / max(total_util, 1e-8)
                        ni.utility = total_util
                        ni.access_count += nj.access_count
                        merged.add(nj.id)
            for nid in merged:
                graph.remove_node(nid)

    def prune_low_utility(self, graph: LatentReasoningGraph):
        now = time_mod.time()
        for node in graph.nodes.values():
            age = now - node.timestamp
            node.utility = (node.access_count *
                          torch.exp(torch.tensor(
                              -self.temporal_decay_beta * age)).item())

        for node_type in NodeType:
            typed_nodes = graph.get_nodes_by_type(node_type)
            if len(typed_nodes) <= 5:
                continue
            utilities = [n.utility for n in typed_nodes]
            threshold = torch.quantile(
                torch.tensor(utilities),
                self.prune_percentile / 100.0).item()
            for n in typed_nodes:
                if n.utility < threshold:
                    graph.remove_node(n.id)

    def expand_graph(self, graph: LatentReasoningGraph,
                    new_embedding: torch.Tensor, role: str,
                    task_id: str, edge_threshold: float = 0.5):
        role_to_type = {"planner": NodeType.ROLE, "coder": NodeType.TRAJECTORY,
                       "critic": NodeType.CONTEXT}
        node_type = role_to_type.get(role, NodeType.CAPSULE)
        new_id = graph.add_node(new_embedding, node_type, role=role, task_id=task_id)

        for nid, node in graph.nodes.items():
            if nid == new_id:
                continue
            sim = F.cosine_similarity(
                new_embedding.unsqueeze(0), node.embedding.unsqueeze(0)
            ).item()
            if sim > edge_threshold:
                graph.add_edge(new_id, nid, sim, "semantic")
        return new_id

    def step(self, graph: LatentReasoningGraph):
        self.merge_similar_nodes(graph)
        self.prune_low_utility(graph)
