from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch

class NodeType(Enum):
    ROLE = "role"
    TRAJECTORY = "trajectory"
    CONTEXT = "context"
    CAPSULE = "capsule"

@dataclass
class Node:
    id: str
    node_type: NodeType
    embedding: torch.Tensor
    latent_tokens: Optional[torch.Tensor] = None
    role: str = "generic"
    task_id: str = ""
    timestamp: float = 0.0
    utility: float = 1.0
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latent_tokens is None:
            self.latent_tokens = self.embedding.unsqueeze(0)
        elif self.latent_tokens.dim() == 1:
            self.latent_tokens = self.latent_tokens.unsqueeze(0)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "node_type": self.node_type.value,
            "role": self.role, "task_id": self.task_id,
            "timestamp": self.timestamp, "utility": self.utility,
            "access_count": self.access_count, "metadata": self.metadata,
        }

    def get_latent_tokens(self) -> torch.Tensor:
        return self.latent_tokens if self.latent_tokens is not None else self.embedding.unsqueeze(0)

    @property
    def dim(self) -> int:
        return self.embedding.shape[0]
