"""Cross-modal alignment and its trainable residual."""
import torch
from torch import nn


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

class MemoryAlignment:
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

class ResidualMemoryAlignment:
    def initialize_cma(self, *args, **kwargs):
        super().initialize_cma(*args, **kwargs)
        self.adapter = ResidualAlignment(self.cma).to(self.pool.device)

    def utility_score(self, query, composed):
        return self.scorer(torch.cat((query.to(self.pool.device).float(), composed.mean(0)))).squeeze().tanh()
