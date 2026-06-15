"""Graph operations for the KG-LMF reasoning graph.

Provides edge-weight computation, differentiable retrieval scoring, and
top-k selection over the LatentReasoningGraph.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .edge import Edge
from .graph import LatentReasoningGraph
from .node import Node, NodeType


class GraphOps:
    """Differentiable graph operations for the Latent Reasoning Graph.

    Responsibilities
    ----------------
    * Compute scalar edge weights from node-pair features via a weighted
      five-term formula.
    * Score a query embedding against the full node set with
      temperature-scaled softmax.
    * Retrieve top-*k* node IDs ranked by score.

    Parameters
    ----------
    alpha : list[float] or None
        Five-element weight vector ``[a1, a2, a3, a4, a5]`` for the edge
        scoring formula.  When ``None`` the default ``[1.0, 0.5, 0.3, 0.2,
        0.1]`` is used.
    """

    def __init__(self, alpha: Optional[List[float]] = None) -> None:
        if alpha is None:
            self.alpha: List[float] = [1.0, 0.5, 0.3, 0.2, 0.1]
        else:
            if len(alpha) != 5:
                raise ValueError(
                    f"alpha must contain exactly 5 elements, got {len(alpha)}"
                )
            self.alpha = list(alpha)

    # ------------------------------------------------------------------
    # Edge weight computation
    # ------------------------------------------------------------------

    def compute_edge_weight(
        self,
        vi: torch.Tensor,
        vj: torch.Tensor,
        role_i: str,
        role_j: str,
        co_occur: int,
        task_id_i: str = "",
        task_id_j: str = "",
    ) -> float:
        r"""Score a candidate edge between two node embeddings.

        The weight is computed as::

            W = a1 * cos_sim
              + a2 * role_sim
              + a3 * task_sim
              + a4 * min(co_occur/10, 1)
              - a5 * max(0, cos_sim - 0.95)

        where ``role_sim = 1.0`` when the two role labels match (``0.3``
        otherwise) and ``task_sim`` is ``1.0`` for matching task IDs,
        ``0.2`` for differing non-empty task IDs, and ``1.0`` when either
        (or both) task ID is the empty string (intra-task default).

        The final term is a *redundancy penalty* that down-weights edges
        between nearly identical embeddings, discouraging degenerate
        clusters.

        Parameters
        ----------
        vi : torch.Tensor
            Source node embedding (d_model,).
        vj : torch.Tensor
            Target node embedding (d_model,).
        role_i : str
            Role label of the source node.
        role_j : str
            Role label of the target node.
        co_occur : int
            Number of times this node pair has co-occurred in reasoning traces.
        task_id_i : str
            Task identifier of the source node.
        task_id_j : str
            Task identifier of the target node.

        Returns
        -------
        float
            Scalar edge weight, clamped to :math:`\ge 0`.
        """
        a1, a2, a3, a4, a5 = self.alpha

        # --- cosine similarity ---
        cos_sim: float = float(
            F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0), dim=-1)
            .detach()
            .cpu()
            .item()
        )

        # --- role similarity ---
        role_sim: float = 1.0 if role_i == role_j else 0.3

        # --- task similarity ---
        if task_id_i and task_id_j:
            task_sim = 1.0 if task_id_i == task_id_j else 0.2
        else:
            task_sim = 1.0  # intra-task default

        # --- co-occurrence term ---
        co_occur_term: float = min(co_occur / 10.0, 1.0)

        # --- redundancy penalty ---
        redundancy_penalty: float = max(0.0, cos_sim - 0.95)

        weight = (
            a1 * cos_sim
            + a2 * role_sim
            + a3 * task_sim
            + a4 * co_occur_term
            - a5 * redundancy_penalty
        )

        return max(0.0, weight)

    # ------------------------------------------------------------------
    # Retrieval scoring
    # ------------------------------------------------------------------

    def scoring_function(
        self,
        query: torch.Tensor,
        nodes_emb: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Compute a differentiable retrieval distribution over nodes.

        The query is compared against every node embedding via cosine
        similarity.  Raw similarities are then passed through a
        temperature-scaled softmax to produce a probability vector.

        Parameters
        ----------
        query : torch.Tensor
            Query embedding of shape ``(d_model,)``.
        nodes_emb : torch.Tensor
            Stacked node embeddings of shape ``(N, d_model)``.  May be
            empty ``(0, d_model)``.
        temperature : float
            Softmax temperature (> 0).  Lower values produce sharper
            (more selective) distributions.

        Returns
        -------
        torch.Tensor
            Softmax scores of shape ``(N,)``, or an empty ``(0,)`` tensor
            when *nodes_emb* has zero rows.
        """
        N = nodes_emb.shape[0]
        if N == 0:
            return torch.empty(0, device=query.device)

        # L2-normalise for cosine similarity.
        query_norm = F.normalize(query.unsqueeze(0), p=2, dim=-1)   # (1, D)
        nodes_norm = F.normalize(nodes_emb, p=2, dim=-1)            # (N, D)

        similarities = (query_norm @ nodes_norm.T).squeeze(0)       # (N,)

        return F.softmax(similarities / temperature, dim=0)

    # ------------------------------------------------------------------
    # Top-k retrieval
    # ------------------------------------------------------------------

    def top_k_retrieval(
        self,
        scores: torch.Tensor,
        node_ids: List[str],
        k: int,
    ) -> List[Tuple[str, float]]:
        """Select the *k* highest-scoring node IDs.

        Parameters
        ----------
        scores : torch.Tensor
            Score vector of shape ``(N,)`` (as returned by
            :meth:`scoring_function`).
        node_ids : list[str]
            Node ID strings aligned with *scores* (``len(node_ids) == N``).
        k : int
            Number of top entries to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(node_id, score)`` pairs ordered by descending score.
            Returns an empty list when *k* <= 0 or *node_ids* is empty.
        """
        if k <= 0 or len(node_ids) == 0:
            return []

        k = min(k, len(node_ids))

        topk_scores, topk_indices = torch.topk(scores, k=k, dim=0)

        return [
            (node_ids[idx.item()], score.item())
            for idx, score in zip(topk_indices, topk_scores)
        ]
