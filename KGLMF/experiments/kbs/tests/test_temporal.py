"""Synthetic software checks only; outputs must never be used as paper results."""
import copy
from pathlib import Path
import pytest
import torch
from KGLMF.experiments.kbs.temporal_memory import TemporalGraph, TemporalMemory
from KGLMF.experiments.kbs.temporal_engine import TemporalEngine
from KGLMF.experiments.kbs.memory import encoded_node, refresh_graph
from KGLMF.experiments.kbs.online_runner import run_stream
from KGLMF.experiments.kbs.online import records
from KGLMF.experiments.kbs.protocol import read_json, score_answer
from .test_online import ToyBackend, Scorer, config, row, identity


def node(g,kind='trajectory',episode=0,vector=None):
    return g.add(torch.randn(16,g.width) if vector is None else vector,['source'+str(g.next_id)],
                 'reasoning',kind=kind,episode=episode)


def test_empirical_edge_normalization_uses_counts_and_first_answer_scores():
    g=TemporalGraph(8,edge_threshold=0.,redundancy_threshold=1.)
    a,b,c=[node(g) for _ in range(3)]
    g.recompute_edges();assert not g.edges
    g.read_scored([a,b],'x',1,'r1',[],1.)
    g.read_scored([a,b],'x',2,'r2',[],0.)
    g.read_scored([a,c],'x',2,'r3',[],.25)
    g.recompute_edges()
    assert g.edges[tuple(sorted((a,b)))]==pytest.approx(1.)
    assert g.edges[tuple(sorted((a,c)))]==pytest.approx(.25)
    g.remove(b);g.recompute_edges()
    assert g.edges[tuple(sorted((a,c)))]==pytest.approx(1.)


def test_capsules_combine_partial_types_and_reuse_sources():
    torch.manual_seed(4)
    m=TemporalMemory(8,12);g=TemporalGraph(8,merge_threshold=1.1)
    a,b=node(g,'context'),node(g,'trajectory')
    g.read_scored([a,b],'x',1,'r1',[],1.)
    first=g.construct_capsules(m,1)
    assert len(first)==1 and g.nodes[first[0]]['constituents']==[a,b]
    assert g.nodes[a]['kind']=='context' and g.nodes[b]['kind']=='trajectory'
    g.read_scored([a,b],'x',2,'r2',[],1.)
    second=g.construct_capsules(m,2)
    assert len(second)==1 and first!=second
    assert g.nodes[second[0]]['tokens'].shape==(16,8)


def test_role_anchor_and_new_node_get_retrieval_opportunity():
    g=TemporalGraph(8)
    anchor=node(g,'role');fresh=node(g,episode=1)
    g.maintain(1,merge=False);assert anchor in g.nodes and fresh in g.nodes
    g.maintain(2,merge=False);assert anchor in g.nodes and fresh not in g.nodes


def test_typed_write_and_capsule_refresh_keep_gradients():
    m=TemporalMemory(8,12);g=TemporalGraph(8)
    components={'kind':'context','representations':torch.randn(9,8)}
    tokens=m.write(components);tokens.square().sum().backward()
    assert m.write_projections['context'].weight.grad is not None
    assert m.write_projections['trajectory'].weight.grad is None
    a=g.add(tokens,['a'],'reasoning',kind='context',components=components)
    b=node(g,'trajectory');g.read_scored([a,b],'x',1,'r',[],1.)
    cap=g.construct_capsules(m,1)[0]
    before=g.nodes[cap]['tokens'].clone()
    with torch.no_grad():m.write_value.weight[0].add_(.1)
    refresh_graph(m,g)
    assert not torch.allclose(before,g.nodes[cap]['tokens'])
    assert encoded_node(m,g.nodes[cap]).requires_grad


def test_merge_unions_events_without_double_counting():
    g=TemporalGraph(8,merge_threshold=.8,prune_threshold=-100)
    vec=torch.randn(16,8);a=node(g,vector=vec);b=node(g,vector=vec.clone());c=node(g,'context')
    g.read_scored([a,b,c],'x',1,'same',[],1.)
    result=g.maintain(1,prune=False)
    assert result['merged']==1
    surviving=a if a in g.nodes else b
    assert g.nodes[surviving]['accesses']==1
    assert set(g.nodes[surviving]['events'])=={'same'}


class Backend(ToyBackend):
    def encode_record(self,text):
        return torch.stack([self.encode(text+str(i)) for i in range(4)])


def test_temporal_engine_journal_resume_and_preanswer_trace(tmp_path):
    c=config();c.update(total_episodes=3,per_node_feedback_probes=False)
    c['domain_prompts']=read_json(Path(__file__).resolve().parents[3]/'paper'/'kbs_revision_2026-10-09'/'domain_prompts.json')
    rows=[row(i) for i in range(3)]
    backend=Backend();condition=c['suites']['main'][-1]
    e=TemporalEngine(backend,c,condition,42,Scorer(tmp_path/'answers.jsonl'))
    calls=[]
    run_stream(e,rows,tmp_path,identity(rows),observer=lambda eng,r:calls.append(eng.episode))
    result=records(tmp_path/'results.jsonl')
    assert calls==[1,2,3]
    assert not result[0]['calls'][0]['retrieval']['ids']
    assert any(u['likelihood_update'] for u in result[1]['updates'])
    assert sum(r['maintenance']['constructed'] for r in result)>0
    assert all('initial_scores' in call['retrieval'] for call in result[1]['calls'])
    assert all('per_node_pairs' in f and not f['per_node_pairs'] for r in records(tmp_path/'feedback.jsonl') for f in r['feedback']['requests'])
    other=TemporalEngine(Backend(),c,condition,42,Scorer())
    run_stream(other,rows,tmp_path,identity(rows))
    assert other.episode==3 and len(records(tmp_path/'answers.jsonl'))==3


@pytest.mark.parametrize('text', ['FINAL: no', '**FINAL:** **no**', 'Final Answer: no', '### FINAL: `no`'])
def test_explicit_final_formats_are_equivalent(text):
    assert score_answer(text,{'metric':'boolean','answers':['false']})['score']==1


def test_scoring_does_not_guess_from_prose_or_explanations():
    for text in ['I considered FINAL: no but disagree.', 'FINAL: nowhere', 'FINAL: no, or perhaps yes']:
        assert score_answer(text,{'metric':'boolean','answers':['false']})['score']==0


def test_temporal_communication_preserves_answer_and_budget():
    engine=object.__new__(TemporalEngine)
    engine.backend=Backend();engine.condition={'communication_tokens':100}
    response='Reasoning: '+('long reasoning '*100)+'\nFINAL: yes'
    message=engine.communication_text(response)
    assert message.startswith('Reasoning:') and message.endswith('FINAL: yes')
    assert len(engine.backend.tokenizer.encode(message,add_special_tokens=False))<=100
    assert engine.communication_text('FINAL: yes')=='FINAL: yes'


def test_feedback_and_answer_survive_trajectory_encoding_budget():
    engine=object.__new__(TemporalEngine)
    engine.backend=Backend();engine.condition={'communication_tokens':100};engine.record_cache={}
    captured=[]
    engine.backend.encode_record=lambda text: captured.append(text) or torch.zeros(2,8)
    engine.record_components(row(1),'solver','reason '*600+'\nFINAL: no','trajectory','verified_feedback')
    assert captured[0].index('verified_feedback')<captured[0].index('Interaction:')
    assert captured[0].endswith('FINAL: no')


def test_temporal_prompt_ends_with_metric_contract_without_gold():
    engine=object.__new__(TemporalEngine)
    engine.prompts=read_json(Path(__file__).resolve().parents[3]/'paper'/'kbs_revision_2026-10-09'/'domain_prompts.json')
    from KGLMF.experiments.kbs.online_protocol import prediction_view
    prompt=engine.build_prompt(prediction_view(row(1)),'critic','A previous proposal')
    assert prompt.index('Previous agents:')<prompt.index('Your response format:')
    assert 'exactly yes or no' in prompt and 'SECRET_IMPLEMENTATION' not in prompt
