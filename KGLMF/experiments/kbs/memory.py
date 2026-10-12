"""Equations 1--13: typed blocks, hard routing and continuous prefix memory.

Router/scorer calibration is separate from likelihood. Composition uses no
routing score prior, and NLL cannot train discrete selections.
"""
import copy
import math
import time
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F

ROLES = ("planner", "solver", "critic")
KINDS = ("role", "trajectory", "context", "capsule")
ANCHORS = "yes no true false one two three first second answer question because therefore however think know find use make good bad big small high low same different correct wrong positive negative before after start end create change result".split()


class MemoryGraph:
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

    @staticmethod
    def candidacy(node):
        count = node["accesses"]
        specificity = sum((v/count)**2 for v in node["reads"].values()) if count else 1.
        gain = sum(g["delta"] for g in node["gains"].values())/len(node["gains"]) if node["gains"] else 0.
        return count*gain/(specificity+1e-8)

    def promote(self):
        promoted = []
        for nid,n in self.nodes.items():
            if n["kind"] in {"trajectory","context"} and n["gains"] and self.candidacy(n) > self.capsule_threshold:
                n["kind"] = "capsule"
                promoted.append(nid)
        return promoted

    def utility(self, nid, episode):
        n = self.nodes[nid]
        weights = [w for pair,w in self.edges.items() if nid in pair]
        mean_weight = sum(weights)/len(weights) if weights else 0.
        age = max(0,episode-n["last_read"])
        a,b,c = self.utility_coefficients
        return (a*mean_weight+b*n["accesses"]-c*age)*math.exp(-self.decay*age)

    def remove(self, nid):
        del self.nodes[nid]
        self.edges = {pair:w for pair,w in self.edges.items() if nid not in pair}

    def maintain(self, episode, merge=True, decay=None, merge_threshold=None, promote=True, prune=True):
        if decay is not None:
            self.decay = decay
        threshold = self.merge_threshold if merge_threshold is None else merge_threshold
        promoted = self.promote() if promote else []
        self.recompute_edges()
        for nid,n in self.nodes.items():
            n["utility"] = self.utility(nid,episode)
        merged,pruned = [],[]
        if merge:
            ids = list(self.nodes)
            for i,a in enumerate(ids):
                if a not in self.nodes:
                    continue
                na = self.nodes[a]
                for b in ids[i+1:]:
                    if b not in self.nodes:
                        continue
                    nb = self.nodes[b]
                    wa,wb = na["utility"],nb["utility"]
                    if na["kind"] != nb["kind"] or min(wa,wb) <= 0:
                        continue
                    sim = float(F.cosine_similarity(na["tokens"].mean(0),nb["tokens"].mean(0),dim=0))
                    if sim <= threshold:
                        continue
                    na["tokens"] = (wa*na["tokens"]+wb*nb["tokens"])/(wa+wb)
                    if wb > wa:
                        na["role"],na["task"] = nb["role"],nb["task"]
                    na["source_ids"] = sorted(set(na["source_ids"]+nb["source_ids"]))
                    na["domains"] = sorted(set(na["domains"]+nb["domains"]))
                    na["events"].update(nb["events"])
                    na["gains"].update(nb["gains"])
                    na["origins"] += nb["origins"]
                    na["text"] += "\n"+nb["text"]
                    na["components"] = None
                    na["utility"] = wa+wb
                    self._statistics(na)
                    self.remove(b)
                    merged.append([a,b])
                    break
        self.recompute_edges()
        for nid,n in self.nodes.items():
            n["utility"] = self.utility(nid,episode)
        if prune:
            for nid,n in list(self.nodes.items()):
                if n["utility"] < self.prune_threshold:
                    self.remove(nid)
                    pruned.append(nid)
        for kind in KINDS:
            ranked = sorted((nid for nid in self.nodes if self.nodes[nid]["kind"] == kind),
                key=lambda nid:(self.nodes[nid]["utility"],self.nodes[nid]["creation"]))
            for nid in ranked[:max(0,len(ranked)-self.per_type_capacity)]:
                self.remove(nid)
                pruned.append(nid)
        while len(self.nodes) > self.max_nodes:
            nid = min(self.nodes,key=lambda k:(self.nodes[k]["utility"],self.nodes[k]["creation"]))
            self.remove(nid)
            pruned.append(nid)
        return dict(merged=len(merged),pruned=len(pruned),promoted=len(promoted),
            merge_events=merged,prune_events=pruned,promotion_events=promoted)

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


class LatentMemory(nn.Module):
    def __init__(self, width, hidden, max_tokens=32, temperature=.1, hops=4, alpha=.5, write_tokens=16, dropout=.1):
        super().__init__()
        if min(width,hidden,max_tokens,write_tokens) < 1 or not 0 < alpha < 1 or temperature <= 0:
            raise ValueError("Invalid dimensions/controller")
        self.width,self.hidden,self.max_tokens = width,hidden,max_tokens
        self.temperature,self.hops,self.alpha,self.write_tokens = temperature,hops,alpha,write_tokens
        self.router = nn.Linear(width,width,bias=False)
        nn.init.xavier_uniform_(self.router.weight)
        self.route_roles = nn.Embedding(len(ROLES),width)
        self.route_role_projection = nn.Linear(width,width,bias=False)
        self.roles = nn.Embedding(len(ROLES),width)
        nn.init.normal_(self.route_roles.weight,std=.02)
        nn.init.normal_(self.roles.weight,std=.02)
        self.role_projection = nn.Linear(width,width,bias=False)
        self.pool = nn.Parameter(torch.randn(max_tokens,width)*.02)
        self.write_pool = nn.Parameter(torch.randn(write_tokens,width)*.02)
        self.write_projections = nn.ModuleList([nn.Linear(width,width) for _ in range(4)])
        self.key,self.value = nn.Linear(width,width,bias=False),nn.Linear(width,width,bias=False)
        self.write_key,self.write_value = nn.Linear(width,width,bias=False),nn.Linear(width,width,bias=False)
        self.norm,self.write_norm = nn.LayerNorm(width),nn.LayerNorm(width)
        self.adapter = nn.Sequential(nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden))
        self.scorer = nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,1))
        self.gate_bias,self.gate_scale = nn.Parameter(torch.tensor(0.)),nn.Parameter(torch.tensor(.3))
        self.cost_logits = nn.Parameter(torch.zeros(2))
        self.register_buffer("cma",torch.zeros(width,hidden))
        self.register_buffer("input_norm",torch.tensor(1.))
        self.register_buffer("cma_ready",torch.tensor(False))
        self.calibration_frozen = False

    def initialize_cma(self, x, y, input_norm, ridge=1e-5):
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 1 or x.shape[1] != self.width or y.shape[1] != self.hidden:
            raise ValueError("Paired anchors require [L,d]/[L,h] shapes")
        x,y = x.double().cpu(),y.double().cpu()
        m = torch.linalg.solve(x.T@x+ridge*torch.eye(self.width,dtype=torch.float64),x.T@y).float()
        with torch.no_grad():
            self.cma.copy_(m.to(self.cma.device))
            self.adapter[0].weight.copy_(m.T.to(self.cma.device))
            self.adapter[0].bias.zero_()
            self.input_norm.fill_(float(input_norm))
            self.cma_ready.fill_(True)

    def router_parameters(self):
        return list(self.router.parameters())+list(self.route_roles.parameters())+list(self.route_role_projection.parameters())

    def cost_coefficients(self):
        """Positive bounded weights with fixed total scale, initialized at 1/1."""
        return .1+1.8*self.cost_logits.softmax(0)

    def structural_cost(self, budget, active_nodes, graph_depth):
        gamma = self.cost_coefficients()
        return budget+gamma[0]*active_nodes+gamma[1]*graph_depth

    def freeze_calibration(self):
        for p in self.router_parameters()+list(self.scorer.parameters()):
            p.requires_grad_(False)
            p.grad = None
        self.calibration_frozen = True

    def write(self, components):
        if len(components) != 4:
            raise ValueError("Write requires role, context, trajectory, knowledge components")
        parts = [projection(part.to(self.pool.device).float().reshape(-1,self.width))
                 for part,projection in zip(components,self.write_projections)]
        values = torch.cat(parts)
        attn = (self.write_pool@self.write_key(values).T/math.sqrt(self.width)).softmax(-1)
        return self.write_norm(attn@self.write_value(values))

    def route(self, query, graph, role="solver", k=4, excluded=(), domains=None, topology="graph", role_condition=True, capsules=True, hops=None, excluded_nodes=()):
        ids = graph.eligible(excluded,domains,capsules,excluded_nodes)
        if not ids:
            return dict(ids=[],initial=torch.empty(0,device=self.pool.device),scores=torch.empty(0,device=self.pool.device),
                chosen=torch.empty(0,dtype=torch.long,device=self.pool.device),active=0,depth=0)
        q = query.to(self.pool.device).float()
        role_vec = self.route_role_projection(self.route_roles.weight[ROLES.index(role)]) if role_condition else torch.zeros_like(q)
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
        return dict(ids=ids,initial=initial,scores=scores,chosen=chosen,active=len(ids),depth=depth)

    def compose(self, graph, ids, role, budget, role_condition=True, reencode=False):
        blocks = []
        for nid in ids:
            node = graph.nodes[nid]
            blocks.append(self.write(node["components"]) if reencode and node.get("components") is not None else node["tokens"].to(self.pool.device))
        tokens = torch.cat(blocks)
        role_vec = self.role_projection(self.roles.weight[ROLES.index(role)]) if role_condition else torch.zeros(self.width,device=tokens.device)
        queries = self.pool[:budget]+role_vec
        attn = (queries@self.key(tokens).T/math.sqrt(self.width)).softmax(-1)
        return self.norm(attn@self.value(tokens))

    def utility_score(self, query, composed):
        return self.scorer(torch.cat((query.to(self.pool.device).float(),composed.mean(0)))).squeeze().clamp(-1,1)

    def forward(self, query, graph, role="solver", k=4, budget=16, excluded=(), domains=None, topology="graph", role_condition=True, capsules=True, hops=None, selected_ids=None, excluded_nodes=(), reencode=False, gate="dynamic", adapter="trained"):
        if not 1 <= budget <= self.max_tokens or k < 1:
            raise ValueError("K/B outside supported range")
        measure = not torch.is_grad_enabled()
        def synchronize():
            if measure and self.pool.is_cuda:
                torch.cuda.synchronize(self.pool.device)
        synchronize()
        started = time.perf_counter()
        routing = self.route(query,graph,role,k,excluded,domains,topology,role_condition,capsules,hops,excluded_nodes)
        synchronize()
        routing_seconds = time.perf_counter()-started
        chosen = routing["chosen"]
        ids = [routing["ids"][i] for i in chosen.detach().cpu().tolist()] if selected_ids is None else list(selected_ids)
        allowed = set(graph.eligible(excluded,domains,capsules,excluded_nodes))
        if any(n not in allowed for n in ids):
            raise ValueError("Selected node violates retrieval access boundary")
        if not ids:
            return None,dict(ids=[],budget=0,weights=[],active_nodes=routing["active"],graph_depth=routing["depth"],gate=0.,routing_seconds=routing_seconds,synthesis_seconds=0.)
        started = time.perf_counter()
        composed = self.compose(graph,ids,role,budget,role_condition,reencode)
        if adapter == "cma":
            if not bool(self.cma_ready):
                raise ValueError("CMA anchors not initialized")
            aligned = composed@self.cma
        elif adapter == "trained":
            aligned = self.adapter(composed)
        else:
            raise ValueError("Unknown adapter condition")
        aligned = aligned*self.input_norm/aligned.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        ca = self.utility_score(query,composed)
        if gate == "dynamic":
            strength = torch.sigmoid(self.gate_bias+self.gate_scale*ca)
        elif gate == "fixed":
            strength = ca.new_tensor(.5)
        else:
            raise ValueError("Unknown gate condition")
        weights = [float(routing["scores"][routing["ids"].index(n)].detach()) for n in ids]
        synchronize()
        synthesis_seconds = time.perf_counter()-started
        return strength*aligned,dict(ids=ids,budget=budget,weights=weights,
            active_nodes=routing["active"],graph_depth=routing["depth"],
            structural_cost=float(self.structural_cost(budget,routing["active"],routing["depth"]).detach()),
            gate=float(strength.detach()),utility_score=float(ca.detach()),routing_seconds=routing_seconds,synthesis_seconds=synthesis_seconds)


def refresh_graph(memory, graph):
    with torch.no_grad():
        for node in graph.nodes.values():
            if node.get("components") is not None:
                node["tokens"] = memory.write(node["components"]).detach().float().cpu()
    graph.recompute_edges()
