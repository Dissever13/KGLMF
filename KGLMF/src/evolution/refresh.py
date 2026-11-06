"""Refresh stored latent representations after writer updates."""
import torch


def encoded_node(memory, node):
    if hasattr(memory, 'reencode_node'):
        return memory.reencode_node(node)
    if node.get("write_sources"):
        return sum(s["weight"]*(memory.write(s["components"]) if s.get("components") is not None
                   else s["tokens"].to(memory.pool.device)) for s in node["write_sources"])
    if node.get("components") is not None:
        return memory.write(node["components"])
    return node["tokens"].to(memory.pool.device)

def refresh_graph(memory, graph):
    with torch.no_grad():
        for node in graph.nodes.values():
            if node.get("components") is not None or node.get("write_sources") or node.get("capsule_records"):
                node["tokens"] = encoded_node(memory,node).detach().float().cpu()
    graph.recompute_edges()
