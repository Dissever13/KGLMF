"""Compose, align and gate the continuous memory prefix."""
import time
import torch
from KGLMF.src.constants import STRUCTURAL_COST_VERSION


class MemoryInjection:
    def utility_score(self, query, composed):
        return self.scorer(torch.cat((query.to(self.pool.device).float(),composed.mean(0)))).squeeze().clamp(-1,1)

    def forward(self, query, graph, role="solver", k=4, budget=16, excluded=(), domains=None, topology="graph", role_condition=True, capsules=True, hops=None, selected_ids=None, excluded_nodes=(), reencode=False, gate="dynamic", adapter="trained", composition_role_condition=None):
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
            return None,dict(ids=[],budget=0,weights=[],active_nodes=routing["active"],active_edges=routing["active_edges"],graph_depth=routing["depth"],cost_version=STRUCTURAL_COST_VERSION,gate=0.,routing_seconds=routing_seconds,synthesis_seconds=0.)
        started = time.perf_counter()
        composed = self.compose(graph,ids,role,budget,role_condition if composition_role_condition is None else composition_role_condition,reencode)
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
            active_nodes=routing["active"],active_edges=routing["active_edges"],graph_depth=routing["depth"],cost_version=STRUCTURAL_COST_VERSION,
            structural_cost=float(self.structural_cost(budget,routing["active"],active_edges=routing["active_edges"]).detach()),
            gate=float(strength.detach()),utility_score=float(ca.detach()),routing_seconds=routing_seconds,synthesis_seconds=synthesis_seconds)
