"""Memory writing and role-conditioned composition."""
import math
import torch

from KGLMF.src.evolution.refresh import encoded_node

class MemoryComposition:
    def write(self, components):
        if len(components) != 4:
            raise ValueError("Write requires role, context, trajectory, knowledge components")
        parts = [projection(part.to(self.pool.device).float().reshape(-1,self.width))
                 for part,projection in zip(components,self.write_projections)]
        values = torch.cat(parts)
        attn = (self.write_pool@self.write_key(values).T/math.sqrt(self.width)).softmax(-1)
        return self.write_norm(attn@self.write_value(values))

    def compose(self, graph, ids, role, budget, role_condition=True, reencode=False):
        blocks = []
        for nid in ids:
            node = graph.nodes[nid]
            blocks.append(encoded_node(self,node) if reencode else node["tokens"].to(self.pool.device))
        tokens = torch.cat(blocks)
        role_vec = self.role_projection(self.roles.weight[self.role_names.index(role)]) if role_condition else torch.zeros(self.width,device=tokens.device)
        queries = self.pool[:budget]+role_vec
        attn = (queries@self.key(tokens).T/math.sqrt(self.width)).softmax(-1)
        return self.norm(attn@self.value(tokens))
