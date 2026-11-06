"""Release checks: isolated benchmark streams and strict resume bindings."""
import copy
import json
import shutil

import pytest

from .. import benchmark
from ..protocol import write_json, digest
from ..online_protocol import prepare_stream
from ..prompt_resources import domain_prompts, role_templates
from .test_paper_engine import settings


def prepare(tmp_path, datasets=('strategyqa',)):
    data = tmp_path/'data'
    data.mkdir()
    for name, filename in [('strategyqa', 'strategyqa_real.json'), ('hotpotqa', 'HotpotQA_real.json')]:
        write_json(data/filename, [dict(id=f'{name}-{i}', question=f'{name} question {i}?',
                                     answer='yes', source_revision='synthetic-test-v1') for i in range(3)])
    manifest = tmp_path/'manifest.json'
    prepare_stream(data, manifest, datasets=list(datasets), seed=42)
    c = settings(); c['condition'] = c['suites']['main'][-1]
    config = tmp_path/'config.json'; write_json(config, c)
    return config, manifest


def test_prompt_resources_are_independent_copies():
    x = role_templates(); x.clear()
    y = domain_prompts(); y.clear()
    assert role_templates()['domains'] and domain_prompts()


def test_rejects_shared_state_between_benchmarks_in_same_domain(tmp_path):
    config, manifest = prepare(tmp_path, ('strategyqa', 'hotpotqa'))
    with pytest.raises(ValueError, match='exactly one benchmark'):
        benchmark.resolve(config, manifest, 42)


def test_seed_does_not_change_fixed_order_and_resume_rejects_changes(tmp_path):
    config, manifest = prepare(tmp_path)
    c, condition, rows, identity = benchmark.resolve(config, manifest, 42)
    _, _, other, other_identity = benchmark.resolve(config, manifest, 43)
    assert [r['id'] for r in rows] == [r['id'] for r in other]
    assert identity['reset_boundary'] == 'benchmark' and identity != other_identity
    out = tmp_path/'run'; snapshot = out/'source_snapshot'; snapshot.mkdir(parents=True)
    write_json(out/'identity.json', identity)
    for name in identity['source_sha256']:
        shutil.copyfile(benchmark.PACKAGE/name, snapshot/name)
    benchmark.verify_existing(out, identity)
    with pytest.raises(ValueError, match='Changed source/config'):
        benchmark.verify_existing(out, other_identity)
    first = next(snapshot.iterdir()); first.write_text('tampered')
    with pytest.raises(ValueError, match='Source snapshot mismatch'):
        benchmark.verify_existing(out, identity)


def test_summary_counts_first_answers_and_separates_feedback(tmp_path):
    write_json(tmp_path/'identity.json', dict(stream_key='strategyqa', condition='kg_multi', seed=42))
    write_json(tmp_path/'status.json', dict(status='complete', total=2))
    rows = [dict(score=x, cost=dict(all_text_tokens=10, total_text_tokens_including_adaptation=17,
                                  prediction_seconds=2, post_answer_seconds=3)) for x in (1., 0.)]
    (tmp_path/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    result = benchmark.summarize(tmp_path)
    assert (result['score_percent'], result['correct']) == (50., 1)
    assert result['inference_text_tokens'] == 20 and result['feedback_text_tokens'] == 14
    assert result['total_latency_seconds'] == 10 and 'no SD' in result['uncertainty']


def test_portable_run_starts_cold_and_completed_resume_loads_no_model(tmp_path, monkeypatch):
    from .. import aligned_backend, aligned_engine
    from ..online import records
    from .test_aligned import RoleBackend, AnswerOnlyScorer
    real_engine = aligned_engine.AlignedEngine
    engines = []
    monkeypatch.setattr(aligned_backend, 'AlignedBackend', lambda cfg: RoleBackend())
    def make_engine(backend, config, condition, seed):
        engine = real_engine(backend, config, condition, seed, AnswerOnlyScorer())
        assert engine.episode == 0 and all(n['kind'] == 'role' for n in engine.graph.nodes.values())
        engines.append(engine)
        return engine
    monkeypatch.setattr(aligned_engine, 'AlignedEngine', make_engine)
    for name in ('strategyqa', 'hotpotqa'):
        root = tmp_path/name; root.mkdir()
        config, manifest = prepare(root, (name,))
        result = benchmark.run(config, manifest, 42, root/'run')
        assert result['n'] == 3
        assert not records(root/'run/answers.jsonl')[0]['calls'][0]['retrieval']['ids']
        assert benchmark.run(config, manifest, 42, root/'run', resume=True) == result
    assert len(engines) == 2
