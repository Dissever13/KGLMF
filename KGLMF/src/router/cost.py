"""Count the graph edges used by retrieval."""
def count_active_edges(graph, node_ids):
    """Count positive undirected edges in the eligible propagation subgraph.

    Each edge contributes once, regardless of the propagation round count.
    Self-transitions used to normalize isolated nodes are not graph edges.
    """
    eligible = set(node_ids)
    return len({frozenset((a, b)) for (a, b), weight in graph.edges.items()
                if a != b and a in eligible and b in eligible and weight > 0})
