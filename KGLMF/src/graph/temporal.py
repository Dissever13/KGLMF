"""Scored co-retrieval edges and graph snapshots."""
import math
from collections import defaultdict
from torch.nn import functional as F

from KGLMF.src.graph.base import MemoryGraph
from KGLMF.src.evolution.temporal import TemporalEvolution

class TemporalGraph(TemporalEvolution, MemoryGraph):
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





    def snapshot(self):
        return dict(nodes=[dict(id=nid,type=n['kind'],constituents=n.get('constituents',[]) if n['kind']=='capsule' else [])
                           for nid,n in self.nodes.items()],
                    edges=[dict(source=a,target=b,weight=w) for (a,b),w in sorted(self.edges.items())])
