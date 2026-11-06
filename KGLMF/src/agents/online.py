"""Predict -> persist -> score/feedback -> adapt -> write -> next problem.

HF models are loaded only by an explicit run command. Test backends may be
injected for CPU verification of ordering, gradients and checkpoint recovery.
"""
import copy
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from KGLMF.src.graph.base import MemoryGraph
from KGLMF.src.constants import ROLES, ANCHORS, STRUCTURAL_COST_VERSION
from KGLMF.src.evolution.refresh import refresh_graph
from KGLMF.src.data.protocol import digest, read_json, write_json, provenance_ids
from KGLMF.src.data.streams import VERSION, validate_manifest, validate_config, prediction_view
from KGLMF.src.execution.runner import seed_all, make_memory, graph_options, prompt_for, question_text, training_target, experience_components
from KGLMF.src.training.optimization import TaskScorer


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    write_json(temp, value)
    os.replace(temp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def records(path):
    """Repair only an interrupted final line in an owned run journal."""
    path = Path(path)
    if not path.exists():
        return []
    lines, parsed = path.read_bytes().splitlines(keepends=True), []
    for index, line in enumerate(lines):
        try:
            parsed.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError('Corrupt run journal before its final line')
            path.write_bytes(b''.join(lines[:index]))
    return parsed


def rng_state():
    return dict(python=random.getstate(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def train_only(memory, parameters):
    flags = [(p, p.requires_grad) for p in memory.parameters()]
    allowed = {id(p) for p in parameters}
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in allowed)
            p.grad = None
        yield
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)


def total_cost(calls):
    costs = {}
    for call in calls:
        for name, value in call['cost'].items():
            if isinstance(value, (int, float)):
                costs[name] = costs.get(name, 0.) + value
    costs['all_text_tokens'] = costs.get('prompt_tokens', 0) + costs.get('output_tokens', 0) + costs.get('probe_tokens', 0)
    return costs


def normalize_internal(text):
    try:
        start, end = text.index('{'), text.rindex('}') + 1
        value = json.loads(text[start:end])
        if type(value.get('accepted')) is not bool:
            raise ValueError('Boolean acceptance required')
        correction = value.get('corrected_response')
        if correction is not None and not isinstance(correction, str):
            raise ValueError('String correction required')
        return dict(accepted=value['accepted'], corrected_response=correction, status='parsed')
    except (ValueError, TypeError):
        return dict(accepted=None, corrected_response=None, status='unavailable_invalid_json')


class OnlineEngine:
    memory_factory = staticmethod(make_memory)
    graph_type = MemoryGraph

    def __init__(self, backend, config, condition, seed, scorer=None, source_graph=None):
        self.backend, self.config, self.condition, self.seed = backend, config, condition, seed
        self.scorer = scorer or TaskScorer(config.get('scoring', {}))
        seed_all(seed)
        self.memory = self.memory_factory(backend, config['memory'])
        x, y, norm, _ = backend.lexical_anchors(ANCHORS)
        self.memory.initialize_cma(x, y, norm)
        self.memory.eval()
        self.graph = self.graph_type(backend.width, **graph_options(config['memory']))
        if condition.get('storage') == 'flat':
            self.graph.edge_threshold = 2.  # No possible Eq.2 edge; metadata is bookkeeping only.
        with torch.no_grad():
            for role in self.initial_roles():
                public = dict(question='', raw={}, domain='unspecified', stream_domain=getattr(self, 'active_domain', 'reasoning'))
                parts = self.record_components(public, role, '', kind='role')
                self.graph.add(self.memory.write(parts), ['role:' + role], 'unspecified', role,
                               kind='role', role=role, components=parts)
        if source_graph is not None:
            self.import_sources(source_graph)
        auxiliary = self.memory.router_parameters() + list(self.memory.scorer.parameters())
        excluded = {id(p) for p in auxiliary}
        if condition.get('adapter') == 'cma':
            excluded.update(id(p) for p in self.memory.adapter.parameters())
        if condition.get('gate') == 'fixed':
            excluded.update((id(self.memory.gate_bias), id(self.memory.gate_scale)))
        self.likelihood_parameters = [p for p in self.memory.parameters() if id(p) not in excluded]
        self.auxiliary_parameters = auxiliary
        lr = config['training']['learning_rate']
        self.aux_optimizer = torch.optim.Adam(auxiliary, lr=lr)
        self.nll_optimizer = torch.optim.Adam(self.likelihood_parameters, lr=lr)
        self.history_max_entropy = None
        self.episode = 0
        self.completed_ids = []
        self.parameter_version = 0
        self.source_ids = sorted({sid for n in self.graph.nodes.values() if n['kind'] != 'role' for sid in n['source_ids']})

    def initial_roles(self):
        return list(ROLES)

    def execution_stages(self, public):
        roles = ['solver'] if self.condition['agents'] == 1 else list(ROLES)
        return [[role] for role in roles]

    def is_solver_request(self, role):
        return role == 'solver'

    def request_key(self, call):
        return call.get('request_key', call['role'])

    def import_sources(self, graph):
        self.backend.synchronize()
        started = time.perf_counter()
        for node in graph.nodes.values():
            if node['kind'] == 'role':
                continue
            copied = copy.deepcopy(node)
            copied['source_episode'] = copied['episode']
            copied['episode'] = copied['last_read'] = 0
            for event in copied['events'].values():
                event['source_episode'], event['episode'] = event['episode'], 0
            copied['creation'] = self.graph.next_id
            if not self.condition.get('capsules', True) and copied['kind'] == 'capsule':
                copied['kind'] = copied['origins'][0]['kind']
            self.graph.nodes[f'n{self.graph.next_id}'] = copied
            self.graph.next_id += 1
        refresh_graph(self.memory, self.graph)
        self.cap_experiences()
        self.backend.synchronize()
        self.source_import_seconds = time.perf_counter() - started

    def cap_experiences(self):
        cap = self.condition.get('experience_cap')
        if cap is None:
            return
        ids = [n for n in self.graph.nodes if self.graph.nodes[n]['kind'] != 'role']
        # Shared upper capacity, not a claim of identical occupancy at cold start.
        for nid in sorted(ids, key=lambda n: digest(self.graph.nodes[n]['source_ids']))[cap:]:
            self.graph.remove(nid)
        self.graph.recompute_edges()

    def vocabulary_size(self):
        return int(self.backend.model.get_input_embeddings().weight.shape[0])

    def build_prompt(self, public, role, history='', memory=''):
        return prompt_for(public, role, history, memory)

    def record_components(self, public, role, response, kind, knowledge=None):
        return experience_components(self.backend, public, role, response, kind, knowledge)

    def record_retrieval(self, call, row, prediction):
        self.graph.read(call['retrieval']['ids'], row['dataset'], self.episode,
                        event_id=f'{row["id"]}:{self.request_key(call)}', source_ids=provenance_ids(row))

    def maintain_graph(self, full, family):
        c = self.condition
        return self.graph.maintain(self.episode, merge=full and family == 'latent' and c.get('merge', True),
                                  promote=full and family == 'latent' and c.get('capsules', True),
                                  prune=full and family == 'latent' and c.get('prune', True))

    def budget(self, entropy, sid, role):
        c = self.condition
        if c.get('budget_policy', 'fixed') == 'fixed':
            return c.get('budget', 16)
        if c['budget_policy'] == 'random':
            return random.Random(f'{self.seed}:{sid}:{role}:budget').choice(c.get('budgets', list(range(12, 33))))
        if c['budget_policy'] != 'entropy' or entropy is None or not math.isfinite(entropy) or entropy < 0:
            raise ValueError('Invalid causal entropy budget')
        lo, hi = self.config['controller']['min_tokens'], self.config['memory']['max_tokens']
        scale = (hi - lo) / max(self.history_max_entropy if self.history_max_entropy is not None else math.log(self.vocabulary_size()), 1e-8)
        return min(hi, lo + math.floor(scale * entropy))

    def prefix(self, public, role, budget, selected=None, reencode=False):
        c = self.condition
        query = self.backend.encode(question_text(public))
        return self.memory(query, self.graph, role, k=c.get('k', 4), budget=budget,
                           excluded=provenance_ids(public), selected_ids=selected, reencode=reencode,
                           topology=c.get('topology', 'graph'), role_condition=c.get('router_role_condition', c.get('role_condition', True)),
                           composition_role_condition=c.get('composition_role_condition', c.get('role_condition', True)),
                           excluded_nodes=[n for n in self.graph.nodes if self.graph.nodes[n]['kind'] == 'role'] if c.get('storage') == 'flat' else (),
                           hops=c.get('hops'), gate=c.get('gate', 'dynamic'), adapter=c.get('adapter', 'trained'))

    @torch.no_grad()
    def predict(self, public):
        started = time.perf_counter()
        self.memory.eval()
        history, calls = [], []
        c, family = self.condition, self.condition['family']
        stages = self.execution_stages(public)
        occurrences = {}
        experienced = any(n['kind'] != 'role' for n in self.graph.nodes.values())
        for stage in stages:
            pending_history = []
            for role in stage:
                occurrences[role] = occurrences.get(role, 0) + 1
                request_key = role if occurrences[role] == 1 else f'{role}#{occurrences[role]}'
                prompt = self.build_prompt(public, role, '\n'.join(history))
                prefix, meta = None, dict(ids=[], budget=0, active_nodes=0, active_edges=0, graph_depth=0)
                entropy, probe = None, dict(probe_tokens=0, probe_seconds=0.)
                if family == 'latent' and c.get('budget_policy') == 'entropy':
                    entropy, probe = self.backend.entropy(prompt)
                allocated = self.budget(entropy, public['id'], role) if family == 'latent' else 0
                self.backend.synchronize()
                retrieval_start = time.perf_counter()
                if experienced and family == 'latent':
                    prefix, meta = self.prefix(public, role, allocated)
                elif experienced and family == 'text':
                    q = self.backend.encode(question_text(public)).cpu()
                    ids = self.graph.eligible(excluded=provenance_ids(public))
                    ids = sorted(ids, key=lambda n: float(F.cosine_similarity(q, self.graph.nodes[n]['tokens'].mean(0), dim=0)), reverse=True)[:c.get('k', 4)]
                    text = '\n\n'.join(self.graph.nodes[n]['text'] for n in ids)
                    token_ids = self.backend.tokenizer.encode(text, add_special_tokens=False)[:c.get('text_memory_tokens', 512)]
                    prompt = self.build_prompt(public, role, '\n'.join(history), self.backend.tokenizer.decode(token_ids))
                    meta.update(ids=ids, text_memory_tokens=len(token_ids))
                self.backend.synchronize()
                retrieval_seconds = time.perf_counter() - retrieval_start
                generation_seed = self.seed + int(digest([public['id'], request_key, 'prediction'])[:8], 16)
                seed_all(generation_seed)
                response, cost = self.backend.generate(prompt, prefix)
                cost.update(probe, memory_seconds=retrieval_seconds,
                            communication_tokens=sum(len(self.backend.tokenizer.encode(h, add_special_tokens=False)) for h in history))
                for name in ('text_memory_tokens', 'structural_cost', 'routing_seconds', 'synthesis_seconds'):
                    cost[name] = meta.get(name, 0.)
                calls.append(dict(role=role, request_key=request_key, prompt=prompt, response=response, retrieval=meta, entropy=entropy,
                                  allocated_budget=allocated, generation_seed=generation_seed, cost=cost))
                pending_history.append(f'{role}: ' + self.communication_text(response))
            history.extend(pending_history)
        self.backend.synchronize()
        cost = total_cost(calls)
        cost['prediction_seconds'] = time.perf_counter() - started
        return dict(id=public['id'], dataset=public['dataset'], domain=public['domain'], stream_domain=public['stream_domain'],
                    metric=public['metric'], episode=self.episode + 1, phase='online', prediction=response,
                    prediction_sha256=digest(response), calls=calls, cost=cost, parameter_version=self.parameter_version,
                    graph_before=self.graph.stats(), score=None, status='answer_recorded_before_feedback')

    def communication_text(self, response):
        ids = self.backend.tokenizer.encode(response, add_special_tokens=False)
        return self.backend.tokenizer.decode(ids[:self.condition.get('communication_tokens', 256)])

    @torch.no_grad()
    def internal_feedback(self, public, call):
        prompt = ('Evaluate the response for the specified agent role using only the problem and response. '
                  'Return JSON with accepted (boolean) and corrected_response (string or null). '
                  'Keep a correction in the same role and output format.\n'
                  f'Role: {call["role"]}\nProblem: {question_text(public)}\nResponse: {call["response"]}')
        seed_all(self.seed + int(digest([public['id'], self.request_key(call), 'internal'])[:8], 16))
        response, cost = self.backend.generate(prompt)
        return dict(normalize_internal(response), raw_response=response, cost=cost, source='internal_critic')

    @torch.no_grad()
    def collect_feedback(self, row, prediction):
        """Called only after prediction is durable. Probes never change state."""
        started = time.perf_counter()
        public, gathered, probe_calls, scored = prediction_view(row), [], [], {}
        def grade(response):
            key = digest(response)
            if key not in scored:
                scored[key] = self.scorer.score(response, row, required=True)
            if scored[key]['score'] is None or not math.isfinite(float(scored[key]['score'])) or not 0 <= float(scored[key]['score']) <= 1:
                raise ValueError('Missing/invalid paired verdict; never substitute zero')
            return scored[key]
        # Reuse the already recorded main verdict without executing it again.
        scored[digest(prediction['prediction'])] = prediction['verdict']
        self.memory.eval()
        for call in prediction['calls']:
            internal = self.internal_feedback(public, call)
            probe_calls.append(dict(cost=internal['cost']))
            with_score = grade(call['response'])
            target, origin = None, None
            if self.is_solver_request(call['role']) and row['domain'] != 'code':
                target, origin = training_target(row), 'benchmark_reference'
            elif self.is_solver_request(call['role']) and row['domain'] == 'code':
                candidates = [(call['response'], 'verified_generated_solver'),
                              (prediction['prediction'], 'verified_generated_final'),
                              (internal.get('corrected_response'), 'verified_generated_correction')]
                for response, source in candidates:
                    if response and grade(response)['score'] == 1:
                        target, origin = response, source
                        break
            elif internal['accepted'] is True and with_score['score'] == 1:
                target, origin = call['response'], 'internally_accepted_verified_response'
            paired, minus = None, {}
            ids = call['retrieval']['ids']
            if ids and self.condition['family'] == 'latent':
                branches = [('none', [])]
                if self.config.get('per_node_feedback_probes', True):
                    branches += [(nid, [n for n in ids if n != nid]) for nid in ids]
                for name, selected in branches:
                    prefix = self.prefix(public, call['role'], call['retrieval']['budget'], selected)[0] if selected else None
                    seed_all(call['generation_seed'])
                    response, cost = self.backend.generate(call['prompt'], prefix)
                    probe_calls.append(dict(cost=cost))
                    verdict = grade(response)
                    gain = float(with_score['score'] - verdict['score'])
                    record = dict(prediction=response, prediction_sha256=digest(response), verdict=verdict, cost=cost,
                                  delta=gain, feedback_source='benchmark_task_score')
                    if name == 'none':
                        paired = record
                    else:
                        minus[name] = record
            gathered.append(dict(role=call['role'], internal=internal, verdict=with_score, target=target,
                                 target_origin=origin, pair=paired, per_node_pairs=minus))
        self.backend.synchronize()
        costs = total_cost(probe_calls)
        costs['feedback_seconds'] = time.perf_counter() - started
        return dict(requests=gathered, cost=costs, source='internal_and_post_answer_benchmark',
                    graph_parameter_version=prediction['parameter_version'], graded_outputs=scored)

    def adapt(self, row, prediction, feedback):
        started = time.perf_counter()
        c, cfg, public = self.condition, self.config['training'], prediction_view(row)
        family = c['family']
        updates = []
        if family == 'latent':
            for call, f in zip(prediction['calls'], feedback['requests']):
                ids, pair = call['retrieval']['ids'], f['pair']
                log = dict(role=call['role'], router_update=False, likelihood_update=False, target_origin=f['target_origin'])
                query = self.backend.encode(question_text(public))
                if ids and pair is not None:
                    self.memory.eval()
                    with train_only(self.memory, self.auxiliary_parameters):
                        routing = self.memory.route(query, self.graph, call['role'], k=c.get('k', 4),
                            excluded=provenance_ids(public), topology=c.get('topology', 'graph'),
                            role_condition=c.get('router_role_condition', c.get('role_condition', True)), hops=c.get('hops'),
                            excluded_nodes=[n for n in self.graph.nodes if self.graph.nodes[n]['kind'] == 'role'] if c.get('storage') == 'flat' else ())
                        indices = [routing['ids'].index(n) for n in ids]
                        mass_loss = (routing['initial'][indices].sum() - (1 + pair['delta']) / 2) ** 2
                        with torch.no_grad():
                            composed = self.memory.compose(self.graph, ids, call['role'], call['retrieval']['budget'],
                                                           c.get('composition_role_condition', c.get('role_condition', True)))
                        scorer_loss = (self.memory.utility_score(query, composed) - pair['delta']) ** 2
                        self.aux_optimizer.zero_grad(set_to_none=True)
                        (mass_loss + scorer_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.auxiliary_parameters, cfg['gradient_clip'])
                        log.update(router_update=True, routing_loss=float(mass_loss.detach()), scorer_loss=float(scorer_loss.detach()),
                                   router_gradient=sum(float(p.grad.abs().sum()) for p in self.memory.router_parameters() if p.grad is not None))
                        self.aux_optimizer.step()
                    self.parameter_version += 1
                if ids and f['target'] is not None:
                    self.memory.train()
                    seed_all(self.seed + int(digest([row['id'], self.request_key(call), 'likelihood'])[:8], 16))
                    with train_only(self.memory, self.likelihood_parameters):
                        prefix, _ = self.prefix(public, call['role'], call['retrieval']['budget'], ids, reencode=True)
                        nll = self.backend.loss(call['prompt'], f['target'], prefix)
                        regularizer = sum(p.square().sum() for p in self.likelihood_parameters)
                        cost = self.memory.structural_cost(call['retrieval']['budget'], call['retrieval']['active_nodes'], active_edges=call['retrieval']['active_edges'])
                        self.nll_optimizer.zero_grad(set_to_none=True)
                        (nll + cfg['l2'] * regularizer + cfg['cost_weight'] * cost).backward()
                        if not torch.isfinite(nll) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.likelihood_parameters):
                            raise RuntimeError('Nonfinite online likelihood gradient')
                        if any(p.grad is not None for p in self.auxiliary_parameters):
                            raise RuntimeError('NLL must not update Router/scorer')
                        torch.nn.utils.clip_grad_norm_(self.likelihood_parameters, cfg['gradient_clip'])
                        log.update(likelihood_update=True, nll=float(nll.detach()),
                                   write_gradient=float(self.memory.write_pool.grad.abs().sum()) if self.memory.write_pool.grad is not None else 0.,
                                   prefix_gradient_valid=prefix.requires_grad)
                        self.nll_optimizer.step()
                    self.parameter_version += 1
                updates.append(log)
            for module in (self.backend.model, self.backend.encoder):
                if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                    raise RuntimeError('Backbone and external encoder must remain frozen')
            self.memory.eval()
            refresh_graph(self.memory, self.graph)
        self.episode += 1
        for call, f in zip(prediction['calls'], feedback['requests']):
            self.record_retrieval(call, row, prediction)
            for nid, pair in f['per_node_pairs'].items():
                self.graph.record_gain(nid, f'{row["id"]}:{self.request_key(call)}', pair['delta'], provenance_ids(row), row['dataset'])
                self.graph.nodes[nid]['gains'][f'{row["id"]}:{self.request_key(call)}:{nid}'].update(
                    feedback_source=pair['feedback_source'], parameter_version=prediction['parameter_version'])
        write = family != 'none' and c.get('scope') != 'source' and c.get('evolution', 'full') != 'static'
        if write:
            with torch.no_grad():
                for index, (call, f) in enumerate(zip(prediction['calls'], feedback['requests'])):
                    allowed = dict(internal_acceptance=f['internal']['accepted'], benchmark_score=f['verdict']['score'])
                    if row['domain'] != 'code':
                        allowed['reference_answer'] = row['answers'][0]
                    text = f'Question: {question_text(public)}\nGenerated response: {call["response"]}\nFeedback: ' + json.dumps(allowed)
                    for kind in (['context', 'trajectory'] if index == 0 else ['trajectory']):
                        node_text = 'Question: ' + question_text(public) if kind == 'context' else text
                        components = self.record_components(public, call['role'], call['response'], kind,
                                                            knowledge=json.dumps(allowed))
                        tokens = self.memory.write(components) if family == 'latent' else self.backend.encode(node_text).repeat(self.graph.write_tokens, 1)
                        nid = self.graph.add(tokens, provenance_ids(row), row['stream_domain'], node_text, self.episode,
                                             kind, call['role'], components if family == 'latent' else None, row['dataset'])
                        self.graph.nodes[nid].update(feedback_source=feedback['source'], parameter_version=self.parameter_version)
        full = c.get('evolution', 'full') == 'full' and c.get('storage') != 'flat'
        maintenance = self.maintain_graph(full, family)
        self.cap_experiences()
        values = [call['entropy'] for call in prediction['calls'] if call['entropy'] is not None]
        if values:
            observed = max(values)
            self.history_max_entropy = observed if self.history_max_entropy is None else max(self.history_max_entropy, observed)
        self.completed_ids.append(row['id'])
        self.backend.synchronize()
        return dict(updates=updates, maintenance=maintenance, graph_after=self.graph.stats(),
                    parameter_version_after=self.parameter_version, entropy_history_after=self.history_max_entropy,
                    adaptation_seconds=time.perf_counter() - started, wrote_target_experience=write)

    def state_dict(self):
        return dict(memory={k: v.detach().cpu() for k, v in self.memory.state_dict().items()}, graph=self.graph.state_dict(),
                    aux_optimizer=self.aux_optimizer.state_dict(), nll_optimizer=self.nll_optimizer.state_dict(),
                    history_max_entropy=self.history_max_entropy, episode=self.episode, completed_ids=self.completed_ids,
                    parameter_version=self.parameter_version, source_ids=self.source_ids, rng=rng_state(),
                    structural_cost_version=STRUCTURAL_COST_VERSION)

    def load_state(self, state):
        if state.get('structural_cost_version') != STRUCTURAL_COST_VERSION:
            raise ValueError('Structural cost version changed; use the original frozen source for historical checkpoints')
        self.memory.load_state_dict(state['memory'], strict=True)
        self.graph = type(self.graph).from_state_dict(state['graph'])
        self.aux_optimizer.load_state_dict(state['aux_optimizer'])
        self.nll_optimizer.load_state_dict(state['nll_optimizer'])
        for key in ('history_max_entropy', 'episode', 'completed_ids', 'parameter_version', 'source_ids'):
            setattr(self, key, state[key])
        restore_rng(state['rng'])
