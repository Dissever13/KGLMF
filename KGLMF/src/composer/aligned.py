"""The memory module used by the main experiment."""
import math
import torch
from torch import nn

from KGLMF.src.composer.typed import TemporalMemory
from KGLMF.src.router.scoring import CosineGraphRouting
from KGLMF.src.injection.alignment import ResidualMemoryAlignment

class AlignedMemory(CosineGraphRouting, ResidualMemoryAlignment, TemporalMemory):
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
