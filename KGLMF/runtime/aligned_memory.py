"""v4: semantic initialization, residual alignment and sparse semantic edges."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_memory import TemporalMemory, TemporalGraph, compact_record
from .memory import KINDS, count_active_edges


class ResidualAlignment(nn.Module):
    def __init__(self, cma):
        super().__init__()
        d, h = cma.shape
        self.register_buffer('base', cma.detach().clone())
        self.residual = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, h))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return x @ self.base + self.residual(x)


class AlignedMemory(TemporalMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve encoder coordinates at cold start; all remain trainable.
        for layer in [self.router, self.key, self.value, self.write_key, self.write_value, *self.write_projections.values()]:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.gate_bias.fill_(math.log(.1/.9))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)

    def initialize_cma(self, *args, **kwargs):
        super().initialize_cma(*args, **kwargs)
        self.adapter = ResidualAlignment(self.cma).to(self.pool.device)

    def utility_score(self, query, composed):
        return self.scorer(torch.cat((query.to(self.pool.device).float(), composed.mean(0)))).squeeze().tanh()


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


class AlignedGraph(TemporalGraph):
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

    def maintain(self, episode, merge=True, prune=True, promote=False, **kwargs):
        if promote:
            raise ValueError('Capsules are composed from sources, never promoted from one node')
        self.recompute_edges()
        self.refresh_utilities(episode)
        merged, pruned = [], []
        ids = list(self.nodes)
        if merge:
            for i,a in enumerate(ids):
                if a not in self.nodes or self.nodes[a]['kind'] == 'role':
                    continue
                na = self.nodes[a]
                for b in ids[i+1:]:
                    if b not in self.nodes:
                        continue
                    nb = self.nodes[b]
                    wa, wb = na['utility'], nb['utility']
                    if na['kind'] != nb['kind'] or min(wa,wb) <= 0:
                        continue
                    if float(F.cosine_similarity(na['tokens'].mean(0),nb['tokens'].mean(0),dim=0)) <= self.merge_threshold:
                        continue
                    def sources(node):
                        return node.get('write_sources') or [dict(weight=1.,record=compact_record(node))]
                    left, right = sources(na), sources(nb)
                    na['tokens'] = (wa*na['tokens']+wb*nb['tokens'])/(wa+wb)
                    na['write_sources'] = [dict(weight=s['weight']*w/(wa+wb),record=s['record'])
                                           for branch,w in ((left,wa),(right,wb)) for s in branch]
                    na['components'] = None
                    if wb > wa:
                        na['role'],na['task'] = nb['role'],nb['task']
                    for field in ('source_ids','domains','constituents'):
                        if field in na or field in nb:
                            na[field] = sorted(set(na.get(field,[])+nb.get(field,[])))
                    na['events'].update(nb['events']); na['gains'].update(nb['gains'])
                    na['origins'] += nb['origins']; na['text'] += '\n'+nb['text']
                    na['utility'] = wa+wb
                    self._statistics(na)
                    self.remove(b)
                    for node in self.nodes.values():
                        if node.get('constituents'):
                            node['constituents'] = sorted({a if x == b else x for x in node['constituents']})
                    merged.append([a,b])
                    break
        self.recompute_edges(); self.refresh_utilities(episode)
        if prune:
            for nid,n in list(self.nodes.items()):
                decayed = n['utility'] * math.exp(-self.decay*max(0,episode-n['last_read']))
                if n['kind'] != 'role' and episode - n['episode'] >= self.grace_episodes and decayed < self.prune_threshold:
                    self.remove(nid); pruned.append(nid)
        for kind in KINDS:
            if kind == 'role':
                continue
            ranked=sorted((n for n in self.nodes if self.nodes[n]['kind']==kind),
                          key=lambda n:(self.nodes[n]['utility'] if prune else 0.,self.nodes[n]['creation']))
            for nid in ranked[:max(0,len(ranked)-self.per_type_capacity)]:
                self.remove(nid);pruned.append(nid)
        self.recompute_edges();self.refresh_utilities(episode)
        return dict(merged=len(merged),pruned=len(pruned),promoted=0,merge_events=merged,
                    prune_events=pruned,promotion_events=[])
