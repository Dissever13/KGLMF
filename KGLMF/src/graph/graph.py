"""Latent Reasoning Graph -- the core knowledge graph for KG-LMF.

Maintains a dynamic graph of Node objects and weighted Edge connections,
supporting typed node storage, retrieval, pruning, and embedding aggregation.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from .edge import Edge
from .node import Node, NodeType


class LatentReasoningGraph:
    """A dynamic, typed knowledge graph for latent reasoning.

    Nodes are stored by auto-incrementing string IDs and organised by their
    NodeType.  Edges are stored keyed by (source_id, target_id) tuples.  The
    graph enforces a per-type node budget and exposes internal pruning so that
    memory footprint stays bounded over long-running reasoning sessions.

    Parameters
    ----------
    d_model : int
        Dimensionality of node embeddings (default 4096, matching typical
        large-model hidden sizes).
    max_nodes_per_type : int
        Soft cap on the number of nodes allowed for any single NodeType.
        Callers should invoke :meth:`_prune_by_type` to enforce the cap.
    """

    def __init__(
        self,
        d_model: int = 384,
        max_nodes_per_type: int = 500,
    ) -> None:
        self.d_model = d_model
        self.max_nodes_per_type = max_nodes_per_type

        self._next_id: int = 0

        # Core storage -----------------------------------------------------------
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        embedding: torch.Tensor,
        node_type: NodeType,
        role: str = "generic",
        task_id: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Create a new node and insert it into the graph.

        Parameters
        ----------
        embedding : torch.Tensor
            Node embedding vector of shape (d_model,).
        node_type : NodeType
            Semantic type of the node (ROLE, TRAJECTORY, CONTEXT, CAPSULE).
        role : str
            Role label for role-typed edges (e.g. "planner", "executor").
        task_id : str
            Identifier of the task that produced this node.
        metadata : dict or None
            Optional free-form metadata dictionary.

        Returns
        -------
        str
            The auto-generated node ID (``"n_0"``, ``"n_1"``, ...).
        """
        node_id = f"n_{self._next_id}"
        self._next_id += 1

        if embedding.shape[0] != self.d_model:
            raise ValueError(
                f"Embedding dim {embedding.shape[0]} != d_model {self.d_model}"
            )

        node = Node(
            id=node_id,
            node_type=node_type,
            embedding=embedding.detach().clone(),
            role=role,
            task_id=task_id,
            timestamp=time.time(),
            metadata=metadata if metadata is not None else {},
        )
        self.nodes[node_id] = node
        return node_id

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src: str,
        tgt: str,
        weight: float = 1.0,
        edge_type: str = "semantic",
    ) -> None:
        """Add or overwrite a directed edge between two existing nodes.

        Parameters
        ----------
        src : str
            Source node ID.
        tgt : str
            Target node ID.
        weight : float
            Edge weight (typically from :meth:`GraphOps.compute_edge_weight`).
        edge_type : str
            Semantic category label for the edge.
        """
        if src not in self.nodes:
            raise KeyError(f"Source node '{src}' not found")
        if tgt not in self.nodes:
            raise KeyError(f"Target node '{tgt}' not found")

        # Bump co-occurrence if edge already exists, otherwise create fresh.
        key = (src, tgt)
        existing = self.edges.get(key)
        if existing is not None:
            existing.weight = weight
            existing.edge_type = edge_type
            existing.co_occurrence += 1
        else:
            self.edges[key] = Edge(
                source_id=src,
                target_id=tgt,
                weight=weight,
                edge_type=edge_type,
                co_occurrence=1,
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Node]:
        """Return the Node with *node_id* or ``None``."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Return all nodes matching *node_type*."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_all_embeddings(self) -> torch.Tensor:
        """Stack every node embedding into a (num_nodes, d_model) tensor.

        Returns an empty ``(0, d_model)`` tensor when the graph has no nodes.
        """
        if not self.nodes:
            return torch.empty((0, self.d_model))
        return torch.stack([n.embedding for n in self.nodes.values()], dim=0)

    def get_node_ids(self) -> List[str]:
        """Return all node IDs currently in the graph."""
        return list(self.nodes.keys())

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all edges incident to it.

        Returns ``True`` if the node existed, ``False`` otherwise.
        """
        if node_id not in self.nodes:
            return False

        del self.nodes[node_id]

        # Purge every edge that references the removed node.
        stale = [
            (s, t)
            for (s, t) in self.edges
            if s == node_id or t == node_id
        ]
        for key in stale:
            del self.edges[key]

        return True

    def _prune_by_type(
        self,
        node_type: NodeType,
        keep_ratio: float = 0.5,
    ) -> int:
        """Remove the lowest-utility nodes of *node_type*.

        Retains the top ``keep_ratio`` fraction, sorted by ``utility``
        descending.  At least one node of the type is always kept when any
        exist.

        Parameters
        ----------
        node_type : NodeType
            The node type whose population should be thinned.
        keep_ratio : float
            Fraction of nodes to retain (0.0 < keep_ratio <= 1.0).

        Returns
        -------
        int
            Number of nodes removed.
        """
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

        candidates = [
            (nid, node)
            for nid, node in self.nodes.items()
            if node.node_type == node_type
        ]
        if not candidates:
            return 0

        candidates.sort(key=lambda item: item[1].utility, reverse=True)

        keep_count = max(1, int(len(candidates) * keep_ratio))
        to_remove = candidates[keep_count:]

        for nid, _node in to_remove:
            self.remove_node(nid)

        return len(to_remove)
