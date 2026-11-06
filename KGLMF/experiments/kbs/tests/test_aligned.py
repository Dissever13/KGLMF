"""Role-feedback, gradient, evidence and graph invariants, not benchmark evidence."""
import copy
import json
import pytest
import torch
from .test_temporal import Backend
from .test_paper_engine import settings
from .test_online import row, identity, Scorer
from ..aligned_engine import AlignedEngine
from ..aligned_memory import AlignedMemory, AlignedGraph
from ..role_feedback import parse_role_quality, routing_advantage_loss, role_training_target, review_consistency
from ..online_runner import run_stream
from ..online import records
from ..aligned_data import evidence_text
from ..online_protocol import prediction_view
from ..aligned_backend import target_only_nll


class RoleBackend(Backend):
    def generate(self, prompt, prefix=None):
        if prompt.startswith('Evaluate the candidate'):
            return json.dumps({'criteria':[2,2,2]}), dict(prompt_tokens=10, output_tokens=8)
        if 'Your response format:\nReturn PLAN:' in prompt:
            return 'PLAN: Compare the stated quantities.', dict(prompt_tokens=10, output_tokens=8)
        return super().generate(prompt, prefix)


class AnswerOnlyScorer(Scorer):
    def score(self, response, row, required=True):
        assert not response.startswith('PLAN:'), 'Planner must never be graded as final answer'
        return super().score(response, row, required)


def test_missing_quality_not_zero_and_strict_schema():
    assert parse_role_quality('garbage')['score'] is None
    assert parse_role_quality('{"criteria":[true,2,2]}')['score'] is None
    assert parse_role_quality('{"criteria":[0,0,0]}')['score'] == 0
    assert parse_role_quality('{"criteria":[2,2,1]}')['accepted']


def test_two_line_reviews_are_not_penalized_for_missing_explanation():
    assert review_consistency('VERDICT: ACCEPT\nFINAL: yes',1)[0]
    assert not review_consistency('VERDICT: ACCEPT\nFINAL: yes',0)[0]
    assert review_consistency('VERDICT: REVISE\nFINAL: yes',0)[0]
    assert review_consistency('VERDICT: REVISE\nFINAL: yes',1)[0]
    assert not review_consistency('FINAL: yes',1)[0]
    with pytest.raises(ValueError):review_consistency('VERDICT: ACCEPT',None)


@pytest.mark.parametrize('seed', [0, 1, 42, 43])
def test_tail_logits_match_masked_causal_loss_and_gradients(seed):
    logits=torch.randn(1,9,11,generator=torch.Generator().manual_seed(seed),requires_grad=True)
    gold=torch.tensor([[2,4,6]])
    labels=torch.full((1,9),-100,dtype=torch.long);labels[:,-3:]=gold
    full=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1))
    tail=target_only_nll(logits[:,-4:],gold)
    # Different reduction shapes can differ by one float32 rounding step.
    torch.testing.assert_close(full,tail,rtol=1e-6,atol=1e-7)
    a=torch.autograd.grad(full,logits,retain_graph=True)[0]
    b=torch.autograd.grad(tail,logits)[0]
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)


def test_public_evidence_does_not_expose_gold_support_labels():
    from .test_online import Tokenizer
    context={'title':['A','B'], 'sentences':[['Evidence one.'],['Evidence two.']]}
    text,metadata=evidence_text(context,Tokenizer(),budget=30)
    r=row(1);r['question']+='\nProvided documents:\n'+text
    r['raw'].update(context=context,supporting_facts='SECRET_SUPPORT',facts='SECRET_FACTS')
    public=prediction_view(r)
    assert 'Evidence' in public['question'] and metadata['retained_tokens']==30
    assert 'SECRET' not in str(public)


def test_role_training_targets_do_not_overwrite_role_with_answer():
    accepted={'accepted':True}
    assert role_training_target('planner','PLAN: inspect',accepted,None,'FINAL: yes')[0]=='PLAN: inspect'
    assert role_training_target('critic','VERDICT: REVISE\nFINAL: yes',{'accepted':False},1,'FINAL: yes')[0] is None
    assert role_training_target('solver','FINAL: no',accepted,0,'FINAL: yes')[0]=='FINAL: yes'


@pytest.mark.parametrize('delta,direction', [(1,-1),(-1,1),(0,0)])
def test_signed_router_gradient(delta,direction):
    mass=torch.tensor(.4,requires_grad=True)
    routing_advantage_loss(mass,delta).backward()
    assert torch.sign(mass.grad).item()==direction


def test_entire_adapter_starts_at_cma_not_just_first_layer():
    m=AlignedMemory(8,12);x=torch.randn(24,8);y=torch.randn(24,12)
    m.initialize_cma(x,y,1.)
    z=torch.randn(5,8)
    assert torch.equal(m.adapter(z),z@m.cma)
    m.adapter(z).square().mean().backward()
    assert m.adapter.residual[-1].weight.grad.abs().sum()>0


def test_semantic_edges_are_real_sparse_and_no_forced_connectivity():
    g=AlignedGraph(8)
    def add(key):
        parts=dict(kind='context',representations=key.repeat(16,1))
        return g.add(key.repeat(16,1),[str(g.next_id)],'reasoning',kind='context',components=parts)
    a=add(torch.tensor([1.,0,0,0,0,0,0,0]));b=add(torch.tensor([1.,.1,0,0,0,0,0,0]))
    c=add(torch.tensor([-1.,0,0,0,0,0,0,0]))
    g.recompute_edges()
    assert tuple(sorted((a,b))) in g.edges
    assert not any(c in pair for pair in g.edges)
    assert all(0<w<=1 for w in g.edges.values())


def test_new_node_has_five_episode_retrieval_opportunities():
    g=AlignedGraph(8);n=g.add(torch.randn(16,8),['x'],'reasoning',episode=1)
    g.maintain(5,merge=False);assert n in g.nodes
    g.maintain(6,merge=False);assert n not in g.nodes


def test_role_feedback_durable_and_resume(tmp_path):
    cfg=settings();e=AlignedEngine(RoleBackend(),cfg,cfg['suites']['main'][-1],42,AnswerOnlyScorer(tmp_path/'answers.jsonl'))
    rows=[row(i) for i in range(3)]
    run_stream(e,rows,tmp_path,identity(rows))
    fb=records(tmp_path/'feedback.jsonl')
    assert all(f['feedback']['requests'][0]['verdict']['task_verdict'] is None for f in fb)
    assert all(f['feedback']['requests'][0]['verdict']['score']==1 for f in fb)
    assert all(f['feedback']['requests'][0]['target'].startswith('PLAN:') for f in fb)
    assert e.backend.loss_targets and any(x.startswith('PLAN:') for x in e.backend.loss_targets)
    other=AlignedEngine(RoleBackend(),cfg,cfg['suites']['main'][-1],42,AnswerOnlyScorer())
    run_stream(other,rows,tmp_path,identity(rows))
    assert other.completed_ids==e.completed_ids
    assert all(torch.equal(a,b) for a,b in zip(other.memory.parameters(),e.memory.parameters()))
