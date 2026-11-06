"""Memory parameters shared by writing, routing, composition and injection."""
import torch
from torch import nn
from KGLMF.src.constants import ROLES

from KGLMF.src.router.scoring import GraphRouting
from KGLMF.src.composer.attention import MemoryComposition
from KGLMF.src.injection.alignment import MemoryAlignment
from KGLMF.src.injection.prefix import MemoryInjection

class LatentMemory(GraphRouting, MemoryComposition, MemoryAlignment, MemoryInjection, nn.Module):
    def __init__(self, width, hidden, max_tokens=32, temperature=.1, hops=4, alpha=.5, write_tokens=16, dropout=.1, role_names=None):
        super().__init__()
        if min(width,hidden,max_tokens,write_tokens) < 1 or not 0 < alpha < 1 or temperature <= 0:
            raise ValueError("Invalid dimensions/controller")
        self.width,self.hidden,self.max_tokens = width,hidden,max_tokens
        self.role_names = tuple(role_names or ROLES)
        if not self.role_names or len(set(self.role_names)) != len(self.role_names):
            raise ValueError("Role names must be nonempty and unique")
        self.temperature,self.hops,self.alpha,self.write_tokens = temperature,hops,alpha,write_tokens
        self.router = nn.Linear(width,width,bias=False)
        nn.init.xavier_uniform_(self.router.weight)
        self.route_roles = nn.Embedding(len(self.role_names),width)
        self.route_role_projection = nn.Linear(width,width,bias=False)
        self.roles = nn.Embedding(len(self.role_names),width)
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
        self.register_buffer("fixed_cost_weights", torch.ones(2))
        self.register_buffer("cma",torch.zeros(width,hidden))
        self.register_buffer("input_norm",torch.tensor(1.))
        self.register_buffer("cma_ready",torch.tensor(False))
        self.calibration_frozen = False





    def freeze_calibration(self):
        for p in self.router_parameters()+list(self.scorer.parameters()):
            p.requires_grad_(False)
            p.grad = None
        self.calibration_frozen = True
