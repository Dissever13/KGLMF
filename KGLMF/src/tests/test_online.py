"""CPU-only causal/gradient checks. No HF weights, datasets or code execution."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from KGLMF.src.data.streams import prepare_stream, prediction_view, example_config, validate_manifest, validate_config, VERSION
from KGLMF.src.agents.online import OnlineEngine, records
from KGLMF.src.execution.stream import run_stream, run, output_lock
from KGLMF.src.graph.base import MemoryGraph
from KGLMF.src.evolution.refresh import encoded_node, refresh_graph
from KGLMF.src.data.protocol import digest, write_json, DATASETS
from KGLMF.src.evaluation.reporting import aggregate


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8', errors='replace')


class FrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding, self.head = nn.Embedding(20, 12), nn.Linear(12, 3)
        g = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g))
        self.requires_grad_(False)
    def get_input_embeddings(self):
        return self.embedding


class ToyBackend:
    device, width, hidden = 'cpu', 8, 12
    resolved = {'model': 'CPU_synthetic_no_benchmark'}
    tokenizer = Tokenizer()
    def __init__(self):
        self.model = FrozenModel()
        self.encoder = nn.Linear(8, 8).requires_grad_(False)
        self.prompts, self.loss_targets = [], []
        self.fail_code = False
    def encode(self, text):
        g = torch.Generator().manual_seed(int(digest(text)[:8], 16))
        return nn.functional.normalize(torch.randn(8, generator=g), dim=0)
    def lexical_anchors(self, anchors):
        g = torch.Generator().manual_seed(4)
        x = torch.randn(24, 8, generator=g)
        return x, x @ torch.randn(8, 12, generator=g), 1., anchors[:24]
    def synchronize(self):
        pass
    def generate(self, prompt, prefix=None):
        self.prompts.append(prompt)
        response = json.dumps({'accepted': True, 'corrected_response': None}) if prompt.startswith('Evaluate the response') else 'FINAL: yes'
        if 'CODE_TEST' in prompt and not prompt.startswith('Evaluate the response'):
            response = '```python\ndef f(): return 0\n```' if self.fail_code else '```python\ndef f(): return 1\n```'
        return response, dict(prompt_tokens=10, output_tokens=3, latent_positions=0 if prefix is None else len(prefix), generation_seconds=.001)
    def entropy(self, prompt):
        return .3, dict(probe_tokens=7, probe_seconds=.001)
    def loss(self, prompt, target, prefix):
        self.loss_targets.append(target)
        return nn.functional.cross_entropy(self.model.head(prefix.mean(0))[None], torch.tensor([1]))


class Scorer:
    def __init__(self, journal=None, code_fail=False):
        self.journal, self.code_fail = journal, code_fail
    def score(self, response, row, required=True):
        if self.journal is not None:
            assert any(r['id'] == row['id'] for r in records(self.journal)), 'oracle used before durable answer'
        score = int('return 1' in response) if row['domain'] == 'code' else 1
        return dict(score=float(0 if self.code_fail else score), status='synthetic_scorer', prediction_sha256=digest(response))


def config():
    c = example_config('synthetic')
    c['backend']['device'] = 'cpu'
    c['memory'].update(prune_threshold=-100., merge_threshold=1.1)
    return c


def row(i, dataset='strategyqa'):
    code = dataset == 'kodcode'
    return dict(id=f'id{i}', group_id=f'group{i}', dataset=dataset, domain='code' if code else 'reasoning',
                stream_domain='code' if code else 'reasoning', metric='external_code' if code else 'boolean',
                question=f'CODE_TEST {i}' if code else f'Synthetic question {i}',
                answers=[] if code else ['True'], raw={'test': 'HIDDEN_TEST', 'answer': 'SECRET_IMPLEMENTATION'}, source_revision='toy')


def identity(rows, name='kg_multi'):
    return dict(condition=name, seed=42, manifest_sha256='synthetic', stream_key=name + '/reasoning', order=[r['id'] for r in rows])


def source_graph(backend):
    graph = MemoryGraph(8, prune_threshold=-100., merge_threshold=1.1)
    for i in range(6):
        parts = [backend.encode(f'source {i} part {j}') for j in range(4)]
        graph.add(torch.randn(16, 8), [f'source{i}'], 'qa', f'source {i}', components=parts, task='triviaqa')
    return graph


def test_stream_groups_share_category_and_have_no_offline_split(tmp_path):
    for dataset, file in [('strategyqa', 'strategyqa_real.json'), ('hotpotqa', 'HotpotQA_real.json')]:
        write_json(tmp_path / file, [dict(id=f'{dataset}{i}', question=f'{dataset} q{i}', answer=True, source_revision='toy') for i in range(3)])
    m = prepare_stream(tmp_path, tmp_path / 'stream.json', ['strategyqa', 'hotpotqa'])
    assert list(m['streams']) == ['reasoning'] and len(m['streams']['reasoning']) == 6
    assert {r['dataset'] for r in m['samples']} == {'strategyqa', 'hotpotqa'}
    assert all('split' not in r for r in m['samples'])
    assert m['feedback'] == 'internal_and_post_answer_benchmark'
    validate_manifest(m, verify_files=True)


def test_single_problem_and_derivative_groups_are_valid_online(tmp_path):
    write_json(tmp_path / 'strategyqa_real.json', [dict(id='a', question='q', answer=False, source_revision='toy'),
               dict(id='b', derivative_of='a', question='q variant', answer=False, source_revision='toy')])
    m = prepare_stream(tmp_path, tmp_path / 'm.json', ['strategyqa'])
    assert len(m['samples']) == 1 and len(m['duplicates_removed']) == 1
    assert m['samples'][0]['answers'] == ['False']


def test_prediction_view_strips_gold_and_hidden_code_fields():
    public = prediction_view(row(1, 'kodcode'))
    assert 'answers' not in public and 'HIDDEN_TEST' not in json.dumps(public)
    assert 'SECRET_IMPLEMENTATION' not in json.dumps(public)


def test_durable_answer_precedes_feedback_and_cold_start_skips_nll(tmp_path):
    backend, c = ToyBackend(), config()
    engine = OnlineEngine(backend, c, c['suites']['main'][-1], 42, Scorer(tmp_path / 'answers.jsonl'))
    before = copy.deepcopy(engine.memory.state_dict())
    run_stream(engine, [row(0)], tmp_path, identity([row(0)]))
    result = records(tmp_path / 'results.jsonl')[0]
    assert all(call['retrieval']['ids'] == [] and call['retrieval']['budget'] == 0 for call in result['calls'])
    assert not any(u['likelihood_update'] or u['router_update'] for u in result['updates'])
    assert all(torch.equal(before[k], engine.memory.state_dict()[k]) for k in before)
    assert engine.graph.stats()['types']['trajectory'] == 3
    assert records(tmp_path / 'answers.jsonl')[0]['score'] is None
    assert engine.history_max_entropy == .3


def test_router_writer_adapter_and_cost_update_after_completed_problem(tmp_path):
    backend, c = ToyBackend(), config()
    engine = OnlineEngine(backend, c, c['suites']['main'][-1], 42, Scorer())
    before_router = engine.memory.router.weight.detach().clone()
    rows = [row(i) for i in range(3)]
    run_stream(engine, rows, tmp_path, identity(rows))
    result = records(tmp_path / 'results.jsonl')[1]
    assert any(u['router_update'] and u['router_gradient'] > 0 for u in result['updates'])
    assert any(u['likelihood_update'] and u['write_gradient'] > 0 and u['prefix_gradient_valid'] for u in result['updates'])
    assert not torch.equal(before_router, engine.memory.router.weight)
    assert all(p.grad is None and not p.requires_grad for module in (backend.model, backend.encoder) for p in module.parameters())
    gamma = engine.memory.cost_coefficients()
    assert torch.all(gamma >= .1) and torch.all(gamma <= 1.9) and torch.allclose(gamma.sum(), torch.tensor(2.))
    assert all('SECRET_IMPLEMENTATION' not in p and 'HIDDEN_TEST' not in p for p in backend.prompts)
    assert result['cost']['total_text_tokens_including_adaptation'] > result['cost']['all_text_tokens']


def test_failed_code_and_scalar_verdict_do_not_create_positive_target(tmp_path):
    backend, c = ToyBackend(), config()
    backend.fail_code = True
    engine = OnlineEngine(backend, c, c['suites']['main'][-1], 42, Scorer(code_fail=True), source_graph(backend))
    rows = [row(0, 'kodcode')]
    run_stream(engine, rows, tmp_path, identity(rows))
    f = records(tmp_path / 'feedback.jsonl')[0]['feedback']
    assert all(request['target'] is None for request in f['requests'])
    assert not backend.loss_targets
    assert all('SECRET_IMPLEMENTATION' not in n['text'] and 'HIDDEN_TEST' not in n['text'] for n in engine.graph.nodes.values())


def test_verified_generated_code_is_target_not_reference_implementation(tmp_path):
    backend, c = ToyBackend(), config()
    engine = OnlineEngine(backend, c, c['suites']['main'][-1], 42, Scorer(), source_graph(backend))
    rows = [row(0, 'kodcode')]
    run_stream(engine, rows, tmp_path, identity(rows))
    requests = records(tmp_path / 'feedback.jsonl')[0]['feedback']['requests']
    solver = next(f for f in requests if f['role'] == 'solver')
    assert solver['target_origin'] == 'verified_generated_solver' and 'return 1' in solver['target']
    assert all('SECRET_IMPLEMENTATION' not in t for t in backend.loss_targets)


def test_source_target_full_share_parameters_and_isolate_experience_writes(tmp_path):
    backend, c = ToyBackend(), config()
    source = source_graph(backend)
    engines = [OnlineEngine(backend, c, condition, 42, Scorer(), source if condition['scope'] != 'target' else None)
               for condition in c['suites']['transfer']]
    initial = engines[0].memory.state_dict()
    assert all(all(torch.equal(initial[k], e.memory.state_dict()[k]) for k in initial) for e in engines[1:])
    rows = [row(0)]
    for e in engines:
        run_stream(e, rows, tmp_path / e.condition['name'], identity(rows, e.condition['name']))
    assert all('id0' not in n['source_ids'] for n in engines[0].graph.nodes.values())
    assert engines[0].parameter_version > 0  # source-only adapts parameters, not target experience records
    assert any('id0' in n['source_ids'] for n in engines[1].graph.nodes.values())
    assert engines[1].source_ids == []
    assert any('source0' in n['source_ids'] for n in engines[2].graph.nodes.values())


def test_resume_reuses_answer_feedback_and_does_not_double_update(tmp_path):
    c, rows = config(), [row(i) for i in range(3)]
    baseline = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    run_stream(baseline, rows, tmp_path / 'baseline', identity(rows))
    interrupted = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    original = interrupted.adapt
    def fail_after_update(r, prediction, feedback):
        result = original(r, prediction, feedback)
        if r['id'] == 'id1':
            raise RuntimeError('synthetic interruption after in-memory update')
        return result
    interrupted.adapt = fail_after_update
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        run_stream(interrupted, rows, tmp_path / 'resume', identity(rows))
    resumed = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    predicted = []
    original_predict = resumed.predict
    def track(public):
        predicted.append(public['id'])
        return original_predict(public)
    resumed.predict = track
    run_stream(resumed, rows, tmp_path / 'resume', identity(rows))
    assert predicted == ['id2']
    assert len(records(tmp_path / 'resume/answers.jsonl')) == 3
    assert len(records(tmp_path / 'resume/results.jsonl')) == 3
    assert resumed.completed_ids == baseline.completed_ids
    assert all(torch.equal(baseline.memory.state_dict()[k], resumed.memory.state_dict()[k]) for k in baseline.memory.state_dict())
    assert resumed.graph.stats() == baseline.graph.stats()


def test_resume_rejects_changed_order_or_condition(tmp_path):
    c, rows = config(), [row(0)]
    engine = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    run_stream(engine, rows, tmp_path, identity(rows))
    with pytest.raises(ValueError, match='does not match'):
        run_stream(engine, rows, tmp_path, dict(identity(rows), seed=43))


def test_merged_blocks_preserve_weighted_reencoding_and_writer_gradient():
    c, backend = config(), ToyBackend()
    engine = OnlineEngine(backend, c, c['suites']['main'][-1], 42, Scorer())
    graph = MemoryGraph(8, merge_threshold=-1., prune_threshold=-100.)
    parts = [[backend.encode(f'{i}:{j}') for j in range(4)] for i in range(2)]
    for i, components in enumerate(parts):
        graph.add(engine.memory.write(components), [str(i)], 'qa', components=components)
    assert graph.maintain(0, prune=False)['merged'] == 1
    node = next(iter(graph.nodes.values()))
    assert node['components'] is None and len(node['write_sources']) == 2
    with torch.no_grad():
        engine.memory.write_projections[0].bias.add_(torch.arange(8) * .2)
    expected = sum(s['weight'] * engine.memory.write(s['components']) for s in node['write_sources'])
    assert torch.allclose(encoded_node(engine.memory, node), expected)
    encoded_node(engine.memory, node).square().sum().backward()
    assert engine.memory.write_pool.grad is not None
    refresh_graph(engine.memory, graph)
    assert torch.allclose(node['tokens'], expected.detach())


def test_condition_and_category_reset_and_online_aggregation(tmp_path):
    write_json(tmp_path / 'strategyqa_real.json', [dict(id=str(i), question=f'question {i}', answer=True, source_revision='toy') for i in range(3)])
    m = prepare_stream(tmp_path, tmp_path / 'm.json', ['strategyqa'])
    c = config()
    write_json(tmp_path / 'c.json', c)
    run(tmp_path / 'c.json', tmp_path / 'm.json', 'main', tmp_path / 'run', 42,
        ['none_single', 'kg_multi'], backend=ToyBackend(), scorer=Scorer())
    report = aggregate([tmp_path / 'run'], tmp_path / 'report.json')
    assert report['version'] == VERSION and len(report['paired_comparisons']) == 1
    assert all(records(tmp_path / 'run' / (name + '.jsonl'))[0]['graph_before']['types'] == {'role': 3}
               for name in ('none_single', 'kg_multi'))
    assert validate_config(c)['feedback'] == 'internal_and_post_answer_benchmark'
    from KGLMF.src.evaluation.online_analysis import curves
    time_report = curves([tmp_path / 'run'], tmp_path / 'curves.json', window=2)
    assert len(time_report['curves']) == 4
    kg = [r for r in time_report['curves'] if r['condition'] == 'kg_multi']
    assert [(r['episode_start'], r['episode_end']) for r in kg] == [(1, 2), (3, 3)]
    assert kg[0]['statistics']['cumulative_performance']['mean'] == 100.


def test_code_stream_needs_tests_but_not_reference_solution(tmp_path):
    write_json(tmp_path / 'kodcode_real.json', [dict(task_id='toy/1', question='CODE_TEST', test='assert f() == 1', source_revision='toy')])
    m = prepare_stream(tmp_path, tmp_path / 'm.json', ['kodcode'])
    assert m['samples'][0]['answers'] == []
    assert m['streams']['code'] == [m['samples'][0]['id']]


def test_output_lock_rejects_concurrent_writer_and_releases(tmp_path):
    with output_lock(tmp_path / 'run'):
        with pytest.raises(RuntimeError, match='Another process'):
            with output_lock(tmp_path / 'run'):
                pytest.fail('Second writer acquired the same output')
    with output_lock(tmp_path / 'run'):
        pass


def test_resume_repairs_committed_checkpoint_before_result_append(tmp_path, monkeypatch):
    import KGLMF.src.execution.stream as runner
    c, rows = config(), [row(0), row(1)]
    engine = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    original = runner.append_record
    def break_result(path, record):
        if Path(path).name == 'results.jsonl':
            raise RuntimeError('interruption after committed checkpoint')
        return original(path, record)
    monkeypatch.setattr(runner, 'append_record', break_result)
    with pytest.raises(RuntimeError, match='committed checkpoint'):
        run_stream(engine, rows, tmp_path, identity(rows))
    monkeypatch.setattr(runner, 'append_record', original)
    resumed = OnlineEngine(ToyBackend(), c, c['suites']['main'][-1], 42, Scorer())
    run_stream(resumed, rows, tmp_path, identity(rows))
    assert resumed.completed_ids == ['id0', 'id1']
    assert [r['id'] for r in records(tmp_path / 'results.jsonl')] == ['id0', 'id1']
    assert len(records(tmp_path / 'answers.jsonl')) == 2


@pytest.mark.parametrize('condition', [c['name'] for c in config()['suites']['ablation']])
def test_all_ablation_paths_have_cold_start_and_executable_gradients(tmp_path, condition):
    c, backend = config(), ToyBackend()
    selected = next(cnd for cnd in c['suites']['ablation'] if cnd['name'] == condition)
    engine = OnlineEngine(backend, c, selected, 42, Scorer())
    run_stream(engine, [row(0), row(1)], tmp_path, identity([row(0), row(1)], condition))
    out = records(tmp_path / 'results.jsonl')
    assert all(call['retrieval']['ids'] == [] for call in out[0]['calls'])
    assert any(u['likelihood_update'] for u in out[1]['updates'])
    if condition == 'flat_pool':
        assert not engine.graph.edges
        assert all(engine.graph.nodes[n]['kind'] != 'role' for call in out[1]['calls'] for n in call['retrieval']['ids'])
    if condition == 'no_capsule':
        assert engine.graph.stats()['types'].get('capsule', 0) == 0


def all_benchmark_manifest(tmp_path):
    for name, (filename, domain, metric) in DATASETS.items():
        item = dict(id=name, question=f'{name} original problem', answer='A' if metric == 'choice' else 'yes', source_revision='CPU_toy')
        if metric == 'choice':
            item['options'] = ['first', 'second']
        if domain == 'code':
            item.update(task_id=f'{name}/test', test='assert f() == 1', question=f'CODE_TEST {name}')
        write_json(tmp_path / filename, [item])
    return prepare_stream(tmp_path, tmp_path / 'all9.json')


def test_transfer_runner_builds_sources_and_three_causal_target_streams(tmp_path):
    m, c = all_benchmark_manifest(tmp_path), config()
    write_json(tmp_path / 'c.json', c)
    run(tmp_path / 'c.json', tmp_path / 'all9.json', 'transfer', tmp_path / 'run', 42,
        backend=ToyBackend(), scorer=Scorer())
    lookup = {r['id']: r for r in m['samples']}
    for condition in c['suites']['transfer']:
        rows = records(tmp_path / 'run' / (condition['name'] + '.jsonl'))
        assert len(rows) == 4
        for direction, names in c['transfer_directions'].items():
            target_ids = {r['id'] for r in m['samples'] if r['dataset'] in names}
            selected = [r for r in rows if r['stream_key'].startswith(direction)]
            assert {r['id'] for r in selected} == target_ids
            checkpoint = torch.load(tmp_path / 'run/states' / direction / 'target' / condition['name'] / 'state.pt', weights_only=True)
            nodes = checkpoint['engine']['graph']['nodes'].values()
            stored_ids = {sid for n in nodes if n['kind'] != 'role' for sid in n['source_ids'] if sid in lookup}
            if condition['scope'] == 'source':
                assert not stored_ids & target_ids
            elif condition['scope'] == 'target':
                assert stored_ids <= target_ids and stored_ids
            else:
                assert stored_ids & target_ids and stored_ids - target_ids
            if condition['scope'] == 'target':
                assert all(not call['retrieval']['ids'] for call in selected[0]['calls'])


def test_online_external_import_checks_history_not_just_prediction_ids(tmp_path):
    from KGLMF.src.evaluation.online_external import import_online_external, REQUIRED_COST
    write_json(tmp_path / 'strategyqa_real.json', [dict(id=str(i), question=f'question {i}', answer=True, source_revision='toy') for i in range(2)])
    m = prepare_stream(tmp_path, tmp_path / 'm.json', ['strategyqa'])
    ids = m['streams']['reasoning']
    data = dict(method='original_method', implementation_url='https://example.org/original', implementation_revision='immutable-test-revision',
                manifest_sha256=m['sha256'], reset_boundary='task_category', feedback='internal_and_post_answer_benchmark',
                initial_benchmark_experience_ids=[], seed=42, backend=ToyBackend.resolved, records=[])
    for i, sid in enumerate(ids):
        data['records'].append(dict(id=sid, prediction='FINAL: yes', prediction_sha256=digest('FINAL: yes'),
                                   answer_recorded_before_feedback=True, available_feedback_ids=ids[:i],
                                   retrieved_source_ids=ids[:i], cost={k: 0. for k in REQUIRED_COST}))
    write_json(tmp_path / 'external.json', data)
    import_online_external(tmp_path / 'external.json', tmp_path / 'm.json', tmp_path / 'valid', scorer=Scorer())
    assert len(records(tmp_path / 'valid/original_method.jsonl')) == 2
    data['records'][0]['retrieved_source_ids'] = [ids[1]]
    write_json(tmp_path / 'external.json', data)
    with pytest.raises(ValueError, match='future'):
        import_online_external(tmp_path / 'external.json', tmp_path / 'm.json', tmp_path / 'leak', scorer=Scorer())
