"""Edge dataclass for the Latent Reasoning Graph.

Represents a weighted, typed connection between two graph nodes.
"""

from dataclasses import dataclass


@dataclass
class Edge:
    """A directed edge connecting two nodes in the reasoning graph.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        weight: Edge weight computed by GraphOps.compute_edge_weight.
        edge_type: Semantic category of the edge (e.g. "semantic", "causal", "temporal").
        co_occurrence: Number of times the source-target pair co-occurred in reasoning traces.
    """

    source_id: str
    target_id: str
    weight: float = 1.0
    edge_type: str = "semantic"
    co_occurrence: int = 0

    def to_dict(self) -> dict:
        """Serialize the edge to a plain dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "edge_type": self.edge_type,
            "co_occurrence": self.co_occurrence,
        }
