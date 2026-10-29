"""Synthetic checks of role dependencies and the paper ablation boundaries."""
import copy
from pathlib import Path
import pytest
from .test_temporal import Backend
from .test_online import config, row, Scorer, identity
from ..paper_engine import PaperEngine, ROLE_IDS, ablation_conditions
from ..protocol import read_json
from ..online_protocol import prediction_view
from ..online_runner import run_stream
from ..online import records


def settings(domain='reasoning'):
    c=config()
    paper=Path(__file__).resolve().parents[3]/'paper/kbs_revision_2026-10-09'
    c.update(active_domain=domain,total_episodes=3,per_node_feedback_probes=False,
             role_templates=read_json(paper/'domain_role_templates.json'),
             domain_prompts=read_json(paper/'domain_prompts.json'))
    return c


@pytest.mark.parametrize('domain,count', [('reasoning',3),('qa',3),('code',5),('science_math',4)])
def test_domain_roles_have_distinct_embeddings_and_correct_anchors(domain,count):
    c=settings(domain);e=PaperEngine(Backend(),c,c['suites']['main'][-1],42,Scorer())
    assert e.memory.roles.num_embeddings==e.memory.route_roles.num_embeddings==count
    assert len(e.graph.nodes)==count
    for node,role in zip(e.graph.nodes.values(),ROLE_IDS[domain]):
        assert node['role']==role and node['domains']==[domain]
        assert e.instruction(role) == c['role_templates']['domains'][domain]['roles'][ROLE_IDS[domain].index(role)]['instruction']


def test_code_reviews_are_independent_and_programmer_revises():
    c=settings('code');e=PaperEngine(Backend(),c,c['suites']['main'][-1],42,Scorer())
    result=e.predict(prediction_view(row(0,'kodcode')))
    assert [x['role'] for x in result['calls']]==['analyst','designer','programmer','reviewer','tester','programmer']
    review,tester,final=result['calls'][3:]
    assert '\nreviewer:' not in tester['prompt']
    assert '\ntester:' not in review['prompt']
    assert '\nreviewer:' in final['prompt'] and '\ntester:' in final['prompt']
    assert result['calls'][2]['generation_seed'] != final['generation_seed']
    assert len({x['request_key'] for x in result['calls']})==6


def test_flat_ablation_preserves_capsules_and_maintenance(tmp_path):
    c=settings();full=c['suites']['main'][-1];flat=next(x for x in ablation_conditions(full) if x['name']=='flat_latent')
    assert flat['storage']=='typed' and flat['capsules'] and flat['merge'] and flat['prune']
    e=PaperEngine(Backend(),c,flat,42,Scorer(tmp_path/'answers.jsonl'))
    rows=[row(i) for i in range(3)]
    run_stream(e,rows,tmp_path,identity(rows,name='flat_latent'))
    results=records(tmp_path/'results.jsonl')
    assert sum(r['maintenance']['constructed'] for r in results)>0
    assert e.graph.edges  # Empirical edges remain for utilities; never used by flat routing.
    for result in results:
        for call in result['calls']:
            assert call['retrieval']['graph_depth']==0


def test_no_role_removes_both_role_conditions():
    c=settings();condition=next(x for x in ablation_conditions(c['suites']['main'][-1]) if x['name']=='no_role')
    assert not condition['router_role_condition'] and not condition['composition_role_condition']
    assert condition['agents']==3 and condition['capsules']


def test_wrong_team_checkpoint_rejected():
    c=settings();e=PaperEngine(Backend(),c,c['suites']['main'][-1],42,Scorer())
    state=e.state_dict();bad=copy.deepcopy(c)
    bad['role_templates']['domains']['reasoning']['roles'][0]['instruction']='Changed instruction'
    other=PaperEngine(Backend(),bad,bad['suites']['main'][-1],42,Scorer())
    with pytest.raises(ValueError,match='schedule mismatch'):other.load_state(state)


def test_capsules_use_source_domain_not_hardcoded_reasoning(tmp_path):
    c=settings('science_math');e=PaperEngine(Backend(),c,c['suites']['main'][-1],42,Scorer())
    rows=[dict(row(i),stream_domain='science_math') for i in range(3)]
    run_stream(e,rows,tmp_path,identity(rows))
    caps=[n for n in e.graph.nodes.values() if n['kind']=='capsule']
    assert caps and all(n['domains']==['science_math'] for n in caps)
