"""Relevance scoring and graph propagation."""
import torch
from torch.nn import functional as F

from KGLMF.src.router.cost import count_active_edges

class GraphRouting:
    def router_parameters(self):
        return list(self.router.parameters())+list(self.route_roles.parameters())+list(self.route_role_projection.parameters())

    def cost_coefficients(self):
        """Fixed unit weights; stored with the model but never optimized."""
        return self.fixed_cost_weights

    def structural_cost(self, budget, active_nodes, active_edges):
        gamma = self.cost_coefficients()
        return budget+gamma[0]*active_nodes+gamma[1]*active_edges

    def route(self, query, graph, role="solver", k=4, excluded=(), domains=None, topology="graph", role_condition=True, capsules=True, hops=None, excluded_nodes=()):
        ids = graph.eligible(excluded,domains,capsules,excluded_nodes)
        if not ids:
            return dict(ids=[],initial=torch.empty(0,device=self.pool.device),scores=torch.empty(0,device=self.pool.device),
                chosen=torch.empty(0,dtype=torch.long,device=self.pool.device),active=0,active_edges=0,depth=0)
        q = query.to(self.pool.device).float()
        role_vec = self.route_role_projection(self.route_roles.weight[self.role_names.index(role)]) if role_condition else torch.zeros_like(q)
        keys = torch.stack([graph.nodes[n]["tokens"].mean(0) for n in ids]).to(q.device)
        initial = (keys@self.router(q+role_vec)/self.temperature).softmax(0)
        scores,depth = initial,0
        if topology not in {"graph","flat","shuffled"}:
            raise ValueError("Unknown topology")
        if topology != "flat":
            lookup = {nid:i for i,nid in enumerate(ids)}
            adjacency = torch.zeros((len(ids),len(ids)),device=q.device)
            for (a,b),weight in graph.edges.items():
                if a in lookup and b in lookup:
                    adjacency[lookup[a],lookup[b]] = weight
                    adjacency[lookup[b],lookup[a]] = weight
            isolated = adjacency.sum(1) == 0
            adjacency[isolated,isolated] = 1.
            if topology == "shuffled":
                perm = torch.randperm(len(ids),generator=torch.Generator().manual_seed(319)).to(q.device)
                adjacency = adjacency[perm][:,perm]
            transition = adjacency/adjacency.sum(1,keepdim=True)
            depth = self.hops if hops is None else hops
            for _ in range(depth):
                scores = self.alpha*(transition.T@scores)+(1-self.alpha)*initial
        chosen = scores.topk(min(k,len(ids))).indices
        return dict(ids=ids,initial=initial,scores=scores,chosen=chosen,active=len(ids),active_edges=count_active_edges(graph, ids) if depth > 0 else 0,depth=depth)


class CosineGraphRouting:
    def route(self, query, graph, role="solver", k=4, excluded=(), domains=None, topology="graph", role_condition=True, capsules=True, hops=None, excluded_nodes=()):
        ids = graph.eligible(excluded,domains,capsules,excluded_nodes)
        if not ids:
            return dict(ids=[],initial=torch.empty(0,device=self.pool.device),scores=torch.empty(0,device=self.pool.device),
                chosen=torch.empty(0,dtype=torch.long,device=self.pool.device),active=0,active_edges=0,depth=0)
        q = query.to(self.pool.device).float()
        role_vec = self.route_role_projection(self.route_roles.weight[self.role_names.index(role)]) if role_condition else torch.zeros_like(q)
        keys = torch.stack([graph.nodes[n]["tokens"].mean(0) for n in ids]).to(q.device)
        initial = (F.normalize(keys, dim=-1) @ F.normalize(self.router(q+role_vec), dim=0) / self.temperature).softmax(0)
        scores,depth = initial,0
        if topology not in {"graph","flat","shuffled"}:
            raise ValueError("Unknown topology")
        if topology != "flat":
            lookup = {nid:i for i,nid in enumerate(ids)}
            adjacency = torch.zeros((len(ids),len(ids)),device=q.device)
            for (a,b),weight in graph.edges.items():
                if a in lookup and b in lookup:
                    adjacency[lookup[a],lookup[b]] = weight
                    adjacency[lookup[b],lookup[a]] = weight
            isolated = adjacency.sum(1) == 0
            adjacency[isolated,isolated] = 1.
            if topology == "shuffled":
                perm = torch.randperm(len(ids),generator=torch.Generator().manual_seed(319)).to(q.device)
                adjacency = adjacency[perm][:,perm]
            transition = adjacency/adjacency.sum(1,keepdim=True)
            depth = self.hops if hops is None else hops
            for _ in range(depth):
                scores = self.alpha*(transition.T@scores)+(1-self.alpha)*initial
        chosen = scores.topk(min(k,len(ids))).indices
        return dict(ids=ids,initial=initial,scores=scores,chosen=chosen,active=len(ids),active_edges=count_active_edges(graph, ids) if depth > 0 else 0,depth=depth)
