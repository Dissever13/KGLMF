"""Node utility, merging and pruning primitives."""
import math
from torch.nn import functional as F
from KGLMF.src.constants import KINDS


class GraphEvolution:
    @staticmethod
    def candidacy(node):
        count = node["accesses"]
        specificity = sum((v/count)**2 for v in node["reads"].values()) if count else 1.
        gain = sum(g["delta"] for g in node["gains"].values())/len(node["gains"]) if node["gains"] else 0.
        return count*gain/(specificity+1e-8)

    def promote(self):
        promoted = []
        for nid,n in self.nodes.items():
            if n["kind"] in {"trajectory","context"} and n["gains"] and self.candidacy(n) > self.capsule_threshold:
                n["kind"] = "capsule"
                promoted.append(nid)
        return promoted

    def utility(self, nid, episode):
        n = self.nodes[nid]
        weights = [w for pair,w in self.edges.items() if nid in pair]
        mean_weight = sum(weights)/len(weights) if weights else 0.
        age = max(0,episode-n["last_read"])
        a,b,c = self.utility_coefficients
        return (a*mean_weight+b*n["accesses"]-c*age)*math.exp(-self.decay*age)

    def maintain(self, episode, merge=True, decay=None, merge_threshold=None, promote=True, prune=True):
        if decay is not None:
            self.decay = decay
        threshold = self.merge_threshold if merge_threshold is None else merge_threshold
        promoted = self.promote() if promote else []
        self.recompute_edges()
        for nid,n in self.nodes.items():
            n["utility"] = self.utility(nid,episode)
        merged,pruned = [],[]
        if merge:
            ids = list(self.nodes)
            for i,a in enumerate(ids):
                if a not in self.nodes:
                    continue
                na = self.nodes[a]
                for b in ids[i+1:]:
                    if b not in self.nodes:
                        continue
                    nb = self.nodes[b]
                    wa,wb = na["utility"],nb["utility"]
                    if na["kind"] != nb["kind"] or min(wa,wb) <= 0:
                        continue
                    sim = float(F.cosine_similarity(na["tokens"].mean(0),nb["tokens"].mean(0),dim=0))
                    if sim <= threshold:
                        continue
                    # Preserve the write records and the Eq.13 mixture when the
                    # writer changes after a later online likelihood update.
                    def sources(node):
                        return node.get("write_sources") or [dict(weight=1.,components=node.get("components"),tokens=node["tokens"].clone())]
                    left,right = sources(na),sources(nb)
                    na["tokens"] = (wa*na["tokens"]+wb*nb["tokens"])/(wa+wb)
                    na["write_sources"] = [dict(s,weight=s["weight"]*w/(wa+wb))
                        for branch,w in ((left,wa),(right,wb)) for s in branch]
                    if wb > wa:
                        na["role"],na["task"] = nb["role"],nb["task"]
                    na["source_ids"] = sorted(set(na["source_ids"]+nb["source_ids"]))
                    na["domains"] = sorted(set(na["domains"]+nb["domains"]))
                    na["events"].update(nb["events"])
                    na["gains"].update(nb["gains"])
                    na["origins"] += nb["origins"]
                    na["text"] += "\n"+nb["text"]
                    na["components"] = None
                    na["utility"] = wa+wb
                    self._statistics(na)
                    self.remove(b)
                    merged.append([a,b])
                    break
        self.recompute_edges()
        for nid,n in self.nodes.items():
            n["utility"] = self.utility(nid,episode)
        if prune:
            for nid,n in list(self.nodes.items()):
                if n["utility"] < self.prune_threshold:
                    self.remove(nid)
                    pruned.append(nid)
        for kind in KINDS:
            ranked = sorted((nid for nid in self.nodes if self.nodes[nid]["kind"] == kind),
                key=lambda nid:(self.nodes[nid]["utility"],self.nodes[nid]["creation"]))
            for nid in ranked[:max(0,len(ranked)-self.per_type_capacity)]:
                self.remove(nid)
                pruned.append(nid)
        while len(self.nodes) > self.max_nodes:
            nid = min(self.nodes,key=lambda k:(self.nodes[k]["utility"],self.nodes[k]["creation"]))
            self.remove(nid)
            pruned.append(nid)
        return dict(merged=len(merged),pruned=len(pruned),promoted=len(promoted),
            merge_events=merged,prune_events=pruned,promotion_events=promoted)
