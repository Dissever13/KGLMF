"""Main-experiment graph with empirical and semantic relations."""
import torch
from torch.nn import functional as F

from KGLMF.src.graph.temporal import TemporalGraph
from KGLMF.src.evolution.aligned import AlignedEvolution

def semantic_key(node):
    if node.get('write_sources'):
        value = sum(s['weight'] * semantic_key(s['record']) for s in node['write_sources'])
    elif node.get('capsule_records'):
        value = torch.stack([semantic_key(r) for r in node['capsule_records']]).mean(0)
    elif node.get('components') is not None:
        value = node['components']['representations'].detach().float().cpu().mean(0)
    else:
        value = node['tokens'].detach().float().cpu().mean(0)
    return F.normalize(value, dim=0)

class AlignedGraph(AlignedEvolution, TemporalGraph):
    semantic_neighbors = 3
    semantic_threshold = .3
    grace_episodes = 5

    def recompute_edges(self):
        super().recompute_edges()
        ids = list(self.nodes)
        if len(ids) < 2:
            return
        keys = torch.stack([semantic_key(self.nodes[n]) for n in ids])
        similarity = keys @ keys.T
        similarity.fill_diagonal_(-1)
        # Symmetric union-kNN. No forced bridges and no layout-generated edges.
        for i, a in enumerate(ids):
            ranked = sorted(range(len(ids)), key=lambda j: (-float(similarity[i, j]), ids[j]))
            for j in ranked[:min(self.semantic_neighbors, len(ids)-1)]:
                s = float(similarity[i, j])
                if s < self.semantic_threshold:
                    continue
                w = min(1., max(0., s - max(0., s-self.redundancy_threshold)))
                if w >= self.edge_threshold:
                    pair = tuple(sorted((a, ids[j])))
                    self.edges[pair] = max(self.edges.get(pair, 0.), w)
