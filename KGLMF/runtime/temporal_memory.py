"""Revised typed writing, compositional Capsules and empirical temporal edges.

This module is selected explicitly by the temporal experiment, preserving old
checkpoint compatibility in the historical runners. No model runs on import.
"""
import copy
import math
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F

from .memory import MemoryGraph, LatentMemory, ROLES, KINDS


class TemporalMemory(LatentMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_projections = nn.ModuleDict({kind: nn.Linear(self.width, self.width)
                                               for kind in KINDS if kind != 'capsule'})

    def pool_blocks(self, values):
        values = values.to(self.pool.device).float()
        attention = (self.write_pool @ self.write_key(values).T / math.sqrt(self.width)).softmax(-1)
        return self.write_norm(attention @ self.write_value(values))

    def write(self, record):
        if not isinstance(record, dict) or record.get('kind') not in self.write_projections:
            raise ValueError('Typed writing requires a role/trajectory/context record')
        values = record['representations'].to(self.pool.device).float()
        return self.pool_blocks(self.write_projections[record['kind']](values))

    def reencode_node(self, node):
        if node.get('write_sources'):
            return sum(s['weight'] * self.reencode_node(s['record']) for s in node['write_sources'])
        if node.get('capsule_records'):
            return self.pool_blocks(torch.cat([self.reencode_node(r) for r in node['capsule_records']]))
        if node.get('components') is not None:
            return self.write(node['components'])
        return node['tokens'].to(self.pool.device)

    def forward(self, *args, **kwargs):
        prefix, meta = super().forward(*args, **kwargs)
        if not torch.is_grad_enabled() and meta['ids']:
            query, graph = args[:2]
            role = args[2] if len(args) > 2 else kwargs.get('role', 'solver')
            keep = {'k','excluded','domains','topology','role_condition','capsules','hops','excluded_nodes'}
            route = self.route(query, graph, role, **{k:v for k,v in kwargs.items() if k in keep})
            meta['initial_scores'] = dict(zip(route['ids'], route['initial'].detach().cpu().tolist()))
            meta['propagated_scores'] = dict(zip(route['ids'], route['scores'].detach().cpu().tolist()))
        return prefix, meta


def compact_record(node):
    """Retain re-encoding inputs without copying retrieval/event histories."""
    return {k:copy.deepcopy(node[k]) for k in ('kind','tokens','components','write_sources','capsule_records') if k in node}


class TemporalGraph(MemoryGraph):
    def __init__(self, *args, **kwargs):
        kwargs.pop('edge_coefficients', None)
        super().__init__(*args, **kwargs)
        self.pending_requests = []

    def connect(self, nid):
        # A newly written node has no observed joint retrieval yet.
        return None

    def raw_weight(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        shared = set(na['events']) & set(nb['events'])
        scored = [na['events'][event]['score'] for event in shared
                  if na['events'][event].get('score') is not None]
        success = sum(scored) / len(scored) if scored else 0.
        sim = float(F.cosine_similarity(na['tokens'].mean(0), nb['tokens'].mean(0), dim=0))
        return max(0., success * len(shared) - max(0., sim - self.redundancy_threshold))

    def recompute_edges(self):
        # Only pairs with a shared retrieval can have positive weight; this is
        # algebraically equivalent to enumerating every current pair in Eq. 2.
        from itertools import combinations
        events = defaultdict(list)
        for nid, node in self.nodes.items():
            for event in node['events']:
                events[event].append(nid)
        pairs = {tuple(sorted(pair)) for ids in events.values() for pair in combinations(ids, 2)}
        raw = {pair:self.raw_weight(*pair) for pair in pairs}
        maximum = max(1e-8, max(raw.values(), default=0.))
        self.edges = {pair:w/maximum for pair,w in raw.items() if w > 0 and w/maximum >= self.edge_threshold}

    def weight(self, a, b):
        return self.edges.get(tuple(sorted((a,b))), 0.)

    def read_scored(self, ids, domain, episode, event_id, source_ids, score):
        if not 0 <= score <= 1 or not math.isfinite(score):
            raise ValueError('A joint retrieval requires the recorded first-answer score')
        self.read(ids, domain, episode, event_id=event_id, source_ids=source_ids)
        for nid in set(ids):
            self.nodes[nid]['events'][event_id]['score'] = float(score)
        self.pending_requests.append((list(dict.fromkeys(ids)), domain, episode))

    def refresh_utilities(self, episode):
        for nid,n in self.nodes.items():
            n['utility'] = self.utility(nid, episode)

    def utility(self, nid, episode):
        n = self.nodes[nid]
        weights = [w for pair,w in self.edges.items() if nid in pair]
        mean_weight = sum(weights)/len(weights) if weights else 0.
        a,b,c = self.utility_coefficients
        return a*mean_weight+b*n['accesses']-c*max(0,episode-n['last_read'])

    def construct_capsules(self, memory, episode):
        self.recompute_edges()
        self.refresh_utilities(episode)
        constructed = []
        for ids, domain, _ in self.pending_requests:
            source = [n for n in ids if n in self.nodes and self.nodes[n]['kind'] != 'capsule'
                      and self.nodes[n]['utility'] > self.capsule_threshold]
            if len(source) < 2:
                continue
            records = [compact_record(self.nodes[n]) for n in source]
            with torch.no_grad():
                block = memory.pool_blocks(torch.cat([self.nodes[n]['tokens'] for n in source]))
            source_ids = sorted({sid for n in source for sid in self.nodes[n]['source_ids']})
            source_domains = sorted({d for n in source for d in self.nodes[n]['domains']})
            nid = self.add(block, source_ids, source_domains[0], '\n'.join(self.nodes[n]['text'] for n in source),
                           episode=episode, kind='capsule', role='solver', task=domain)
            self.nodes[nid].update(constituents=source, capsule_records=records, domains=source_domains)
            constructed.append(nid)
        self.pending_requests.clear()
        return constructed

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
                if n['kind'] != 'role' and episode > n['episode'] and decayed < self.prune_threshold:
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

    def snapshot(self):
        return dict(nodes=[dict(id=nid,type=n['kind'],constituents=n.get('constituents',[]) if n['kind']=='capsule' else [])
                           for nid,n in self.nodes.items()],
                    edges=[dict(source=a,target=b,weight=w) for (a,b),w in sorted(self.edges.items())])
