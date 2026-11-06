"""Typed graph storage, retrieval events and serialization."""
import copy
import math
from collections import Counter
from torch.nn import functional as F
from KGLMF.src.constants import KINDS

from KGLMF.src.evolution.base import GraphEvolution

class MemoryGraph(GraphEvolution):
    def __init__(self, width, max_nodes=2000, edge_threshold=.1, neighbors=None,
                 per_type_capacity=500, merge_threshold=.85, prune_threshold=.1,
                 capsule_threshold=.3, redundancy_threshold=.95, decay=.01,
                 edge_coefficients=None, utility_coefficients=None, write_tokens=16):
        self.width,self.max_nodes = width,max_nodes
        self.edge_threshold,self.neighbors = edge_threshold,neighbors
        self.per_type_capacity,self.write_tokens = per_type_capacity,write_tokens
        self.merge_threshold,self.prune_threshold = merge_threshold,prune_threshold
        self.capsule_threshold,self.redundancy_threshold,self.decay = capsule_threshold,redundancy_threshold,decay
        self.edge_coefficients = list(edge_coefficients or [.2]*5)
        self.utility_coefficients = list(utility_coefficients or [1/3]*3)
        if len(self.edge_coefficients) != 5 or len(self.utility_coefficients) != 3:
            raise ValueError("Five edge and three utility coefficients required")
        if any(x < 0 for x in self.edge_coefficients+self.utility_coefficients):
            raise ValueError("Coefficients must be nonnegative")
        if not math.isclose(sum(self.edge_coefficients),1.) or not math.isclose(sum(self.utility_coefficients),1.):
            raise ValueError("Coefficients must sum to one")
        self.nodes,self.edges,self.next_id = {},{},0

    def clone(self):
        return copy.deepcopy(self)

    def add(self, tokens, source_ids, domain, text="", episode=0, kind="trajectory", role="solver", components=None, task=None):
        tokens = tokens.detach().float().cpu()
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tokens.ndim != 2 or tokens.shape[1] != self.width or not source_ids or kind not in KINDS:
            raise ValueError("Typed node requires a token block and source provenance")
        if len(tokens) != self.write_tokens:
            tokens = F.adaptive_avg_pool1d(tokens.T[None],self.write_tokens)[0].T
        nid = f"n{self.next_id}"
        self.next_id += 1
        self.nodes[nid] = dict(tokens=tokens,source_ids=sorted(set(source_ids)),domains=[domain],
            task=domain if task is None else task,text=text,episode=episode,creation=self.next_id,kind=kind,role=role,
            reads={},events={},gains={},accesses=0,last_read=episode,utility=0.,components=components,
            origins=[dict(role=role,task=domain,kind=kind)])
        self.connect(nid)
        return nid

    def weight(self, a, b):
        na,nb = self.nodes[a],self.nodes[b]
        sim = float(F.cosine_similarity(na["tokens"].mean(0),nb["tokens"].mean(0),dim=0))
        role = 1. if na["role"] == nb["role"] else .3
        task = 1. if na["task"] == nb["task"] or "unspecified" in (na["task"],nb["task"]) else .2
        freq = min(len(set(na["events"]) & set(nb["events"]))/10,1.)
        red = max(0.,sim-self.redundancy_threshold)
        c = self.edge_coefficients
        return max(0.,c[0]*sim+c[1]*role+c[2]*task+c[3]*freq-c[4]*red)

    def connect(self, nid):
        self.edges = {pair:w for pair,w in self.edges.items() if nid not in pair}
        ranked = [(self.weight(nid,other),other) for other in self.nodes if other != nid]
        kept = sorted(((w,other) for w,other in ranked if w >= self.edge_threshold),reverse=True)
        if self.neighbors is not None:
            kept = kept[:self.neighbors]
        for w,other in kept:
            self.edges[tuple(sorted((nid,other)))] = w

    def recompute_edges(self):
        self.edges = {}
        ids = list(self.nodes)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                w = self.weight(a,b)
                if w >= self.edge_threshold:
                    self.edges[tuple(sorted((a,b)))] = w

    def eligible(self, excluded=(), domains=None, capsules=True, excluded_nodes=()):
        excluded,excluded_nodes = set(excluded),set(excluded_nodes)
        return [nid for nid,n in self.nodes.items() if nid not in excluded_nodes
            and not excluded.intersection(n["source_ids"])
            and (domains is None or n["kind"] == "role" or set(n["domains"]).issubset(set(domains)))
            and (capsules or n["kind"] != "capsule")]

    def read(self, ids, domain, episode, promote=False, event_id=None, source_ids=()):
        event_id = event_id or f"{episode}:{domain}:request"
        for nid in set(ids):
            n = self.nodes[nid]
            n["events"][event_id] = dict(domain=domain,episode=episode,source_ids=list(source_ids))
            self._statistics(n)
        if promote:
            self.promote()

    @staticmethod
    def _statistics(node):
        node["reads"] = dict(Counter(e["domain"] for e in node["events"].values()))
        node["accesses"] = len(node["events"])
        node["last_read"] = max([node["episode"]]+[e["episode"] for e in node["events"].values()])

    def record_gain(self, nid, event_id, delta, source_ids=(), domain=None):
        if not -1 <= delta <= 1 or not math.isfinite(delta):
            raise ValueError("Paired per-node gain must lie in [-1,1]")
        self.nodes[nid]["gains"][f"{event_id}:{nid}"] = dict(delta=float(delta),source_ids=list(source_ids),domain=domain)




    def remove(self, nid):
        del self.nodes[nid]
        self.edges = {pair:w for pair,w in self.edges.items() if nid not in pair}


    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_state_dict(cls, state):
        graph = cls(state["width"])
        graph.__dict__.update(copy.deepcopy(state))
        if any(n not in graph.nodes for pair in graph.edges for n in pair):
            raise ValueError("Dangling serialized edge")
        if any(n["kind"] not in KINDS or n["tokens"].shape != (graph.write_tokens,graph.width) for n in graph.nodes.values()):
            raise ValueError("Invalid node type/block in checkpoint")
        return graph

    def stats(self):
        degrees = Counter(x for pair in self.edges for x in pair)
        return dict(nodes=len(self.nodes),edges=len(self.edges),types=dict(Counter(n["kind"] for n in self.nodes.values())),
            mean_degree=sum(degrees.values())/max(1,len(self.nodes)),stored_vectors=sum(len(n["tokens"]) for n in self.nodes.values()),
            paired_gain_observations=sum(len(n["gains"]) for n in self.nodes.values()))
