"""Typed node writing and compositional Capsule encoding."""
import math
import torch
from torch import nn
from KGLMF.src.constants import KINDS

from KGLMF.src.composer.base import LatentMemory

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
