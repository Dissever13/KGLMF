"""Meaningful synthetic checks; never download/load a real encoder/backbone."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from KGLMF.src.composer.base import LatentMemory
from KGLMF.src.graph.base import MemoryGraph
from KGLMF.src.evolution.refresh import refresh_graph
from KGLMF.src.data.protocol import prepare_manifest, write_json, digest, provenance_ids, VERSION
from KGLMF.src.execution.runner import choose_budget, build_graph, make_memory
from KGLMF.src.training.optimization import round_robin, feedback_pairs, TaskScorer
from KGLMF.src.config.examples import example_config
from KGLMF.src.tests.test_protocol import FakeBackend


def written_graph(module):
    graph = MemoryGraph(module.width,edge_threshold=0.)
    for i in range(8):
        components = [torch.randn(module.width) for _ in range(4)]
        graph.add(module.write(components),[f"s{i}"],"math" if i%2 else "science",components=components)
    return graph


def test_writer_and_all_32_composition_queries_receive_nll_gradients():
    torch.manual_seed(42)
    module = LatentMemory(8,12)
    graph = written_graph(module)
    module.freeze_calibration()
    frozen = nn.Linear(12,5).requires_grad_(False)
    prefix,meta = module(torch.randn(8),graph,budget=32,reencode=True)
    nn.functional.cross_entropy(frozen(prefix),torch.arange(32)%5).backward()
    assert all(g["tokens"].shape == (16,8) for g in graph.nodes.values())
    assert prefix.shape == (32,12) and torch.unique(prefix.detach(),dim=0).shape[0] == 32
    assert (module.pool.grad.abs().sum(1)>0).all() and module.write_pool.grad.abs().sum()>0
    assert module.router.weight.grad is None


def test_auxiliary_score_mass_updates_router_without_topk_gradient_claim():
    module = LatentMemory(8,12)
    graph = written_graph(module)
    routing = module.route(torch.randn(8),graph)
    chosen = routing["chosen"].detach()
    ((routing["initial"][chosen].sum()-.5)**2).backward()
    assert module.router.weight.grad is not None and module.router.weight.grad.abs().sum()>0
    assert module.pool.grad is None and all(p.grad is None for p in module.adapter.parameters())


def test_fixed_cost_weights_survive_optimization_and_checkpoint_round_trip():
    module = LatentMemory(8,12)
    assert torch.equal(module.cost_coefficients(),torch.ones(2))
    assert "fixed_cost_weights" in dict(module.named_buffers())
    assert "fixed_cost_weights" not in dict(module.named_parameters())
    assert not hasattr(module,"cost_logits")
    optimizer = torch.optim.Adam(module.parameters(),lr=.01)
    before = module.pool.detach().clone()
    cost = module.structural_cost(16,80,active_edges=4)
    assert cost.item() == 100. and not cost.requires_grad
    optimizer.zero_grad()
    (module.pool.square().sum()+.001*cost).backward()
    assert module.fixed_cost_weights.grad is None
    optimizer.step()
    assert not torch.equal(module.pool,before)
    assert torch.equal(module.cost_coefficients(),torch.ones(2))
    assert module.structural_cost(16,80,active_edges=4).item() == 100.
    restored = LatentMemory(8,12)
    restored.load_state_dict(module.state_dict())
    assert torch.equal(restored.cost_coefficients(),torch.ones(2))



def test_cma_paired_coordinates_and_norm_and_fixed_gate():
    torch.manual_seed(4)
    module = LatentMemory(8,12,dropout=0.)
    x = torch.randn(24,8)
    truth = torch.randn(8,12)
    module.initialize_cma(x,x@truth,3.)
    assert torch.allclose(module.cma,truth,atol=1e-4)
    assert torch.allclose(module.adapter[0].weight,truth.T,atol=1e-4)
    graph = written_graph(module)
    prefix,meta = module(torch.randn(8),graph,gate="fixed",adapter="cma")
    assert torch.allclose(prefix.norm(dim=-1),torch.full((16,),1.5),atol=1e-5)
    assert meta["gate"] == .5


def test_role_changes_synthesis_even_with_identical_selected_nodes():
    module = LatentMemory(8,12,dropout=0.).eval()
    graph = written_graph(module)
    q = torch.randn(8)
    ids = list(graph.nodes)[:4]
    a,_ = module(q,graph,"planner",selected_ids=ids,gate="fixed")
    b,_ = module(q,graph,"critic",selected_ids=ids,gate="fixed")
    assert not torch.allclose(a,b)
    a,_ = module(q,graph,"planner",selected_ids=ids,gate="fixed",role_condition=False)
    b,_ = module(q,graph,"critic",selected_ids=ids,gate="fixed",role_condition=False)
    assert torch.allclose(a,b)


def test_row_normalized_graph_uses_transpose_and_only_isolated_self_loops():
    module = LatentMemory(4,6,hops=1,alpha=.5)
    graph = MemoryGraph(4)
    for i in range(3):
        graph.add(torch.randn(16,4),[str(i)],"qa")
    graph.edges = {("n0","n1"):2.,("n0","n2"):1.}
    routing = module.route(torch.randn(4),graph,role_condition=False)
    a = torch.tensor([[0.,2.,1.],[2.,0.,0.],[1.,0.,0.]])
    expected = .5*(a/a.sum(1,keepdim=True)).T@routing["initial"]+.5*routing["initial"]
    assert torch.allclose(expected,routing["scores"]) and routing["depth"] == 1
    assert torch.allclose(routing["scores"].sum(),torch.tensor(1.))


def test_merged_block_key_events_and_pair_observations_are_preserved():
    graph = MemoryGraph(4,prune_threshold=-100.)
    block = torch.randn(16,4)
    a = graph.add(block,["a"],"qa")
    b = graph.add(block+.001,["b"],"science")
    for node in (a,b):
        graph.read([node],"qa",0,event_id="same-request")
        graph.record_gain(node,"same-request",.05 if node==a else .1,["train-source"])
    result = graph.maintain(0,promote=False,prune=False)
    assert result["merged"] == 1
    n = next(iter(graph.nodes.values()))
    assert n["accesses"] == 1 and len(n["gains"]) == 2 and n["source_ids"] == ["a","b"]
    assert n["tokens"].shape == (16,4) and len(n["origins"]) == 2


def test_each_type_has_its_own_hard_capacity():
    graph = MemoryGraph(4,per_type_capacity=2,max_nodes=8)
    for kind in ("role","context","trajectory","capsule"):
        for i in range(4):
            graph.add(torch.randn(16,4),[kind+str(i)],"qa",kind=kind)
    graph.maintain(0,merge=False,promote=False,prune=False)
    assert graph.stats()["types"] == dict(role=2,context=2,trajectory=2,capsule=2)
    assert all(n in graph.nodes for pair in graph.edges for n in pair)


def test_group_split_precedes_caps_and_training_seeds_do_not_change_manifest(tmp_path):
    rows = [{"id":str(i),"question":f"q{i}","answer":"yes","source_revision":"test"} for i in range(40)]
    for i in range(1,5):
        rows[i]["derivative_of"] = "0"
    write_json(tmp_path/"strategyqa_real.json",rows)
    m = prepare_manifest(tmp_path,tmp_path/"manifest.json",["strategyqa"])
    grouped = [r for r in m["samples"] if r["raw"]["id"] in {"0","1","2","3","4"}]
    assert len({r["group_id"] for r in grouped}) == 1 and len({r["split"] for r in grouped}) == 1
    assert len(m["samples"]) == 40 and m["limits"] is None
    again = prepare_manifest(tmp_path,tmp_path/"again.json",["strategyqa"])
    assert m["sha256"] == again["sha256"]


def test_controller_exact_floor_and_cap_rejects_uncalibrated_values():
    condition = dict(budget_policy="entropy",min_tokens=12,max_tokens=32,entropy_scale=2.)
    assert [choose_budget(condition,e) for e in (0.,1.49,1.5,30.)] == [12,14,15,32]
    with pytest.raises(ValueError):
        choose_budget(dict(budget_policy="entropy"),1.)


def test_source_group_exclusion_applies_to_merged_memories():
    module = LatentMemory(8,12)
    graph = written_graph(module)
    graph.nodes["n0"]["source_ids"].append("group:x")
    _,meta = module(torch.randn(8),graph,excluded=["different-query","group:x"])
    assert "n0" not in meta["ids"]


def test_typed_initial_graph_contains_only_training_sources():
    config = example_config("fake")
    backend = FakeBackend()
    module = make_memory(backend,config["memory"])
    rows = [dict(id=digest(i)[:24],group_id="group:"+str(i),split="train",domain="qa",dataset="toy",metric="em",raw={},question=f"q{i}",answers=["gold"]) for i in range(8)]
    graph = build_graph(backend,rows,config["memory"],module)
    assert set(graph.stats()["types"]) == {"role","context","trajectory"}
    assert all(n["tokens"].shape == (16,8) for n in graph.nodes.values())
    with pytest.raises(ValueError):
        build_graph(backend,[dict(rows[0],split="test")],config["memory"],module)


def test_round_robin_interleaves_domains_and_missing_external_score_fails():
    rows = [dict(dataset="a",id=str(i)) for i in range(3)]+[dict(dataset="b",id="b")]
    assert [r["dataset"] for r in round_robin(rows)] == ["a","b","a","a"]
    with pytest.raises(ValueError,match="isolated"):
        TaskScorer({}).score("code",dict(metric="external_code",answers=["x"]))


@pytest.mark.parametrize("freeze_adapter",[False,True])
def test_complete_staged_pipeline_with_tiny_frozen_backend(tmp_path,monkeypatch,freeze_adapter):
    from KGLMF.src.training.optimization import initialize, collect_feedback, train, finalize
    from KGLMF.src.execution.runner import calibrate, evaluate
    import KGLMF.src.backends.huggingface as backend_module
    class TinyBackend(FakeBackend):
        def __init__(self,*args,**kwargs):
            self.model = nn.Linear(12,3).requires_grad_(False)
            self.encoder = nn.Linear(8,8).requires_grad_(False)
        def lexical_anchors(self,anchors):
            x = torch.randn(24,8)
            return x,x@torch.randn(8,12),1.,anchors[:24]
        def loss(self,prompt,target,prefix):
            return nn.functional.cross_entropy(self.model(prefix.mean(0))[None],torch.tensor([1]))
    monkeypatch.setattr(backend_module,"HFBackend",TinyBackend)
    write_json(tmp_path/"strategyqa_real.json",[{"id":str(i),"question":f"q{i}","answer":False,"source_revision":"toy"} for i in range(10)])
    manifest = prepare_manifest(tmp_path,tmp_path/"manifest.json",["strategyqa"])
    config = example_config("synthetic")
    config["backend"]["device"] = "cpu"
    config["training"]["epochs"] = 1
    write_json(tmp_path/"config.json",config)
    calibrate(tmp_path/"config.json",tmp_path/"manifest.json",tmp_path/"calibrated.json")
    initialize(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"initial.pt",42)
    collect_feedback(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"initial.pt",tmp_path/"feedback",42,"cma" if freeze_adapter else "trained")
    train(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"candidate.pt",42,
          initial=tmp_path/"initial.pt",feedback=tmp_path/"feedback/pairs.jsonl",freeze_adapter=freeze_adapter)
    collect_feedback(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"candidate.pt",tmp_path/"graph_feedback",42)
    finalize(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"candidate.pt",tmp_path/"graph_feedback/pairs.jsonl",tmp_path/"trained.pt",42)
    state = torch.load(tmp_path/"trained.pt",weights_only=True)
    assert state["stage"] == "trained" and state["gradient_checked"]
    assert state["memory"]["cma_ready"].item() and state["dev_score"] == 1.
    evaluate(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"trained.pt","main",tmp_path/"run",42,["kg_multi"])
    assert json.loads((tmp_path/"run/run.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError,match="initialized"):
        evaluate(tmp_path/"calibrated.json",tmp_path/"manifest.json",tmp_path/"initial.pt","main",tmp_path/"bad",42,["kg_multi"])


def test_embedding_scale_is_equal_for_text_and_continuous_prefix():
    from KGLMF.src.backends.huggingface import HFBackend
    class ScaledEmbedding(nn.Embedding):
        embed_scale = 3.
        def forward(self,ids):
            return super().forward(ids)*self.embed_scale
    embedding = ScaledEmbedding(20,12).requires_grad_(False)
    backend = HFBackend.__new__(HFBackend)
    backend.model = SimpleNamespace(get_input_embeddings=lambda:embedding)
    prefix = embedding.weight[1:3].clone().requires_grad_(True)
    result = backend.embeddings(torch.tensor([[1,2]]),prefix)
    assert torch.allclose(result[:,:2],result[:,2:])
    result.sum().backward()
    assert torch.allclose(prefix.grad,torch.full_like(prefix,3.))


def test_duplicate_native_ids_remain_linked_to_their_derivatives(tmp_path):
    rows = [dict(id=str(i),question=f"q{i}",answer=True,source_revision="test") for i in range(20)]
    rows.append(dict(id="alternate",question="q0",answer=True,source_revision="test"))
    rows[1]["derivative_of"] = "alternate"
    write_json(tmp_path/"strategyqa_real.json",rows)
    manifest = prepare_manifest(tmp_path,tmp_path/"manifest.json",["strategyqa"])
    by_q = {r["question"]:r for r in manifest["samples"]}
    assert by_q["q0"]["group_id"] == by_q["q1"]["group_id"] and len(manifest["duplicates_removed"]) == 1


def test_same_question_with_reordered_options_is_grouped_but_not_discarded(tmp_path):
    rows = [dict(id=str(i),question=f"q{i}",answer="A",choices=["a","b"],source_revision="test") for i in range(20)]
    rows.append(dict(id="reordered",question="q0",answer="B",choices=["b","a"],source_revision="test"))
    write_json(tmp_path/"mmlu_pro_real.json",rows)
    manifest = prepare_manifest(tmp_path,tmp_path/"manifest.json",["mmlu_pro"])
    pair = [r for r in manifest["samples"] if r["question"] == "q0"]
    assert len(pair) == 2 and len({r["group_id"] for r in pair}) == 1 and len({r["split"] for r in pair}) == 1
