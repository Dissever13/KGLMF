"""
Latent Memory Injector with dynamic gating support.

FIX #5: Dynamic gates — input-dependent, modulated by scorer confidence.
        gate = sigmoid(base + confidence * scale)
        - When scorer predicts memory helps: gate opens wide
        - When scorer predicts memory hurts: gate closes
        - Backward compatible: confidence=None uses static gate behavior

FIX #2: Embedding Adapter — 2-layer residual MLP for proper space alignment
        Replaces single Linear projection for better MiniLM→LLM mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EmbeddingAdapter(nn.Module):
    """2-layer residual adapter for aligning embedding spaces."""
    def __init__(self, d_model: int = 384, hidden_dim: int = 512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class LatentInjector(nn.Module):
    """Memory injection with optional dynamic gating.

    FIX #5: When confidence is provided, gates are modulated:
        effective_gate = sigmoid(base_param + confidence * confidence_scale)
    """

    def __init__(self, d_model: int = 384,
                 injection_layers: List[int] = None,
                 use_adapter: bool = True):
        super().__init__()
        self.d_model = d_model
        self.injection_layers = injection_layers or [0, 8, 16]  # FIX #6
        self.use_adapter = use_adapter

        # Per-layer gate base parameters
        self.gate_base = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.0))  # scalar, σ(0)=0.5
            for layer in self.injection_layers
        })

        # Per-layer confidence scale (FIX #5: dynamic gating)
        self.gate_conf_scale = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.3))
            for layer in self.injection_layers
        })

        # Memory projection (FIX #2: adapter or linear)
        if use_adapter:
            self.memory_proj = nn.ModuleDict({
                str(layer): EmbeddingAdapter(d_model, hidden_dim=512)
                for layer in self.injection_layers
            })
        else:
            self.memory_proj = nn.ModuleDict({
                str(layer): nn.Linear(d_model, d_model, bias=False)
                for layer in self.injection_layers
            })

        # Initialize
        for proj in self.memory_proj.values():
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)

    def _compute_gate(self, layer: int,
                      confidence: Optional[float] = None) -> torch.Tensor:
        """Compute gate value. Dynamic if confidence provided, static otherwise.

        FIX #5: gate = sigmoid(base + confidence * scale)
        """
        key = str(layer)
        if key not in self.gate_base:
            return torch.tensor(0.5)

        base = self.gate_base[key]

        if confidence is not None:
            # Dynamic gate: modulated by scorer confidence
            scale = self.gate_conf_scale[key]
            conf_tensor = torch.tensor(float(confidence), device=base.device)
            return torch.sigmoid(base + scale * conf_tensor)
        else:
            # Static gate: backward compatible
            return torch.sigmoid(base)

    def inject(self, hidden_states: torch.Tensor,
               memory_tokens: torch.Tensor,
               layer_idx: int,
               confidence: Optional[float] = None) -> torch.Tensor:
        """Inject memory tokens into hidden states via cross-attention.

        Args:
            hidden_states: LLM hidden states [batch, seq, d_model] or [seq, d_model]
            memory_tokens: Composed memory tokens [n_queries, d_model]
            layer_idx: Which transformer layer is being injected
            confidence: Optional scorer confidence for dynamic gating (FIX #5)

        Returns:
            Modified hidden states
        """
        layer_key = str(layer_idx)
        if layer_key not in self.memory_proj:
            return hidden_states

        # Project memory to LLM-compatible space (FIX #2: adapter)
        projected_memory = self.memory_proj[layer_key](memory_tokens)

        # Compute gate (FIX #5: dynamic)
        gate = self._compute_gate(layer_idx, confidence)

        # Cross-attention: hidden states attend to projected memory
        if hidden_states.dim() == 3:
            # [batch, seq, d_model]
            attn_output = F.scaled_dot_product_attention(
                hidden_states,
                projected_memory.unsqueeze(0).expand(hidden_states.shape[0], -1, -1),
                projected_memory.unsqueeze(0).expand(hidden_states.shape[0], -1, -1))
        elif hidden_states.dim() == 2:
            # [seq, d_model]
            attn_output = F.scaled_dot_product_attention(
                hidden_states.unsqueeze(0),
                projected_memory.unsqueeze(0),
                projected_memory.unsqueeze(0)).squeeze(0)
        else:
            return hidden_states

        return hidden_states + gate * attn_output

    def prepend_memory(self, memory_tokens: torch.Tensor,
                       input_embeds: torch.Tensor) -> torch.Tensor:
        """Prepend memory tokens to input embeddings (soft prompt style)."""
        batch_size = input_embeds.shape[0]
        if memory_tokens.dim() == 2:
            memory_tokens = memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([memory_tokens, input_embeds], dim=1)

    def get_gate_values(self, confidence: Optional[float] = None) -> dict:
        """Get current gate values for monitoring."""
        return {
            str(l): self._compute_gate(l, confidence).item()
            for l in self.injection_layers
        }


# FIX #1: MemoryScorer — differentiable proxy for LLM feedback
class MemoryScorer(nn.Module):
    """Predicts how much memory tokens will help the LLM (accuracy_delta).

    Trained on LLM feedback data. Provides differentiable gradient for
    composer training — the key innovation that makes end-to-end training
    possible without backprop through the LLM.
    """

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # [-1, 1] range for accuracy delta
        )

    def forward(self, query_emb: torch.Tensor,
                memory_tokens: torch.Tensor) -> torch.Tensor:
        mem_pooled = memory_tokens.mean(dim=0)
        combined = torch.cat([query_emb.to(mem_pooled.device), mem_pooled])
        return self.net(combined)
