from dataclasses import dataclass, field
from enum import Enum
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
    role: str = "generic"
    task_id: str = ""
    timestamp: float = 0.0
    utility: float = 1.0
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "node_type": self.node_type.value,
            "role": self.role, "task_id": self.task_id,
            "timestamp": self.timestamp, "utility": self.utility,
            "access_count": self.access_count, "metadata": self.metadata,
        }

    @property
    def dim(self) -> int:
        return self.embedding.shape[0]
