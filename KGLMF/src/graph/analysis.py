"""Graph diagnostics on saved states, including degree-preserving random controls.

No embeddings are re-encoded and no language model is run.
"""
import argparse
import random
from collections import deque
from KGLMF.src.data.protocol import write_json


def summarize_graph(nodes, edges):
    adjacency={n:set() for n in nodes}
    for a,b in edges:
        if a!=b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    clustering={}
    for n,neighbors in adjacency.items():
        k=len(neighbors)
        triangles=sum(len(adjacency[v]&neighbors) for v in neighbors)
        clustering[n]=triangles/(k*(k-1)) if k>1 else 0.
    reachable_pairs=distance_sum=0
    for n in nodes:
        dist={n:0}
        queue=deque([n])
        while queue:
            u=queue.popleft()
            for v in adjacency[u]:
                if v not in dist:
                    dist[v]=dist[u]+1
                    queue.append(v)
        reachable_pairs+=len(dist)-1
        distance_sum+=sum(dist.values())
    types={}
    for kind in sorted({node["kind"] for node in nodes.values()}):
        selected=[n for n in nodes if nodes[n]["kind"]==kind]
        types[kind]={"count":len(selected),"mean_degree":sum(len(adjacency[n]) for n in selected)/len(selected),
                     "mean_clustering":sum(clustering[n] for n in selected)/len(selected)}
    return dict(nodes=len(nodes),edges=len(edges),mean_clustering=sum(clustering.values())/max(1,len(nodes)),
                reachable_ordered_pairs=reachable_pairs,
                mean_distance_reachable=distance_sum/reachable_pairs if reachable_pairs else None,
                reachability_fraction=reachable_pairs/max(1,len(nodes)*(len(nodes)-1)),by_type=types)


def randomized_edges(edges, seed):
    rng=random.Random(seed)
    edge_set={tuple(sorted(pair)) for pair in edges}
    swaps=0
    for _ in range(20*len(edge_set)):
        if len(edge_set)<2:
            break
        (a,b),(c,d)=rng.sample(sorted(edge_set),2)
        if rng.random()<.5:
            c,d=d,c
        x,y=tuple(sorted((a,d))),tuple(sorted((c,b)))
        if len({a,b,c,d})<4 or x in edge_set or y in edge_set:
            continue
        edge_set.remove(tuple(sorted((a,b))))
        edge_set.remove(tuple(sorted((c,d))))
        edge_set.update([x,y])
        swaps+=1
    return edge_set,swaps


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--random-controls",type=int,default=10)
    a=p.parse_args()
    import torch
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    graph=state["engine"]["graph"] if "engine" in state else state.get("graph",state)
    controls=[]
    for seed in range(a.random_controls):
        edges,swaps=randomized_edges(graph["edges"],seed)
        controls.append(dict(seed=seed,successful_swaps=swaps,**summarize_graph(graph["nodes"],edges)))
    write_json(a.output,dict(observed=summarize_graph(graph["nodes"],graph["edges"]),random_controls=controls,
                            interpretation="Descriptive topology only; inspect connectivity and successful swaps. No automatic small-world claim."))
