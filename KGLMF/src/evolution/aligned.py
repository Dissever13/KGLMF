"""Main-experiment maintenance with protection for newly written nodes."""
import math
from torch.nn import functional as F
from KGLMF.src.constants import KINDS

from KGLMF.src.graph.records import compact_record

class AlignedEvolution:
    def maintain(self, episode, merge=True, prune=True, promote=False, **kwargs):
        if promote:
            raise ValueError('Capsules are composed from sources, never promoted from one node')
        self.recompute_edges()
        self.refresh_utilities(episode)
        merged, pruned = [], []
        ids = list(self.nodes)
        if merge:
            for i,a in enumerate(ids):
                if a not in self.nodes or self.nodes[a]['kind'] == 'role':
                    continue
                na = self.nodes[a]
                for b in ids[i+1:]:
                    if b not in self.nodes:
                        continue
                    nb = self.nodes[b]
                    wa, wb = na['utility'], nb['utility']
                    if na['kind'] != nb['kind'] or min(wa,wb) <= 0:
                        continue
                    if float(F.cosine_similarity(na['tokens'].mean(0),nb['tokens'].mean(0),dim=0)) <= self.merge_threshold:
                        continue
                    def sources(node):
                        return node.get('write_sources') or [dict(weight=1.,record=compact_record(node))]
                    left, right = sources(na), sources(nb)
                    na['tokens'] = (wa*na['tokens']+wb*nb['tokens'])/(wa+wb)
                    na['write_sources'] = [dict(weight=s['weight']*w/(wa+wb),record=s['record'])
                                           for branch,w in ((left,wa),(right,wb)) for s in branch]
                    na['components'] = None
                    if wb > wa:
                        na['role'],na['task'] = nb['role'],nb['task']
                    for field in ('source_ids','domains','constituents'):
                        if field in na or field in nb:
                            na[field] = sorted(set(na.get(field,[])+nb.get(field,[])))
                    na['events'].update(nb['events']); na['gains'].update(nb['gains'])
                    na['origins'] += nb['origins']; na['text'] += '\n'+nb['text']
                    na['utility'] = wa+wb
                    self._statistics(na)
                    self.remove(b)
                    for node in self.nodes.values():
                        if node.get('constituents'):
                            node['constituents'] = sorted({a if x == b else x for x in node['constituents']})
                    merged.append([a,b])
                    break
        self.recompute_edges(); self.refresh_utilities(episode)
        if prune:
            for nid,n in list(self.nodes.items()):
                decayed = n['utility'] * math.exp(-self.decay*max(0,episode-n['last_read']))
                if n['kind'] != 'role' and episode - n['episode'] >= self.grace_episodes and decayed < self.prune_threshold:
                    self.remove(nid); pruned.append(nid)
        for kind in KINDS:
            if kind == 'role':
                continue
            ranked=sorted((n for n in self.nodes if self.nodes[n]['kind']==kind),
                          key=lambda n:(self.nodes[n]['utility'] if prune else 0.,self.nodes[n]['creation']))
            for nid in ranked[:max(0,len(ranked)-self.per_type_capacity)]:
                self.remove(nid);pruned.append(nid)
        self.recompute_edges();self.refresh_utilities(episode)
        return dict(merged=len(merged),pruned=len(pruned),promoted=0,merge_events=merged,
                    prune_events=pruned,promotion_events=[])
