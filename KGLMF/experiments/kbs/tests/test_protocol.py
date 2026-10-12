import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from KGLMF.experiments.kbs.memory import LatentMemory, MemoryGraph
from KGLMF.experiments.kbs.protocol import (digest, prepare_manifest, validate_manifest,
    score_answer, choice_gold, write_json, bootstrap_mean, VERSION)
from KGLMF.experiments.kbs.backend import HFBackend, completion_ids
from KGLMF.experiments.kbs.runner import choose_budget, restrict_graph, predict
from KGLMF.experiments.kbs.reporting import import_scores
from KGLMF.experiments.kbs.recover_sources import convert


def graph_fixture():
    torch.manual_seed(2)
    graph = MemoryGraph(8,max_nodes=8,edge_threshold=0.,neighbors=2)
    for i in range(6):
        graph.add(torch.randn(2,8),[f"s{i}"],"science" if i%2 else "math",text=str(i))
    return graph


def test_nll_composer_adapter_gradient_and_frozen_routing_backbone():
    graph = graph_fixture()
    module = LatentMemory(8,12,max_tokens=32)
    module.freeze_calibration()
    frozen = nn.Linear(12,5).requires_grad_(False)
    prefix,meta = module(torch.randn(8),graph,k=4,budget=32,excluded=["s0"])
    assert "n0" not in meta["ids"] and prefix.shape == (32,12)
    loss = nn.functional.cross_entropy(frozen(prefix),torch.arange(32)%5)
    loss.backward()
    for parameter in (module.pool,module.adapter[0].weight,module.roles.weight,module.gate_bias):
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
        assert torch.isfinite(parameter.grad).all()
    assert all(p.grad is None for p in frozen.parameters())
    assert module.router.weight.grad is None and all(p.grad is None for p in module.scorer.parameters())


def test_graph_changes_selection_and_flat_keeps_same_composer():
    module = LatentMemory(8,12,alpha=.95,hops=1)
    graph = graph_fixture()
    with torch.no_grad():
        module.router.weight.copy_(torch.eye(8))
    q = graph.nodes["n0"]["tokens"].mean(0)
    # Strong bridge redistributes mass while preserving node/adapter values.
    graph.edges = {("n0","n5"):100.}
    _,flat = module(q,graph,k=3,role_condition=False,topology="flat")
    _,linked = module(q,graph,k=3,role_condition=False,topology="graph")
    assert flat["weights"] != linked["weights"]


def test_graph_roundtrip_merge_provenance_cap():
    graph = graph_fixture()
    graph.add(graph.nodes["n0"]["tokens"],["new"],"qa",episode=1)
    graph.read(["n0","n6"],"math",0,event_id="pair")
    result = graph.maintain(0,prune=False)
    assert result["merged"]>=1
    assert {"s0","new"}.issubset(graph.nodes["n0"]["source_ids"])
    assert "n0" not in graph.eligible(excluded=["new"])
    restored = MemoryGraph.from_state_dict(graph.state_dict())
    assert restored.edges == graph.edges and restored.stats() == graph.stats()
    restored.max_nodes=2
    restored.maintain(0,merge=False,prune=False)
    assert len(restored.nodes)==2 and len(graph.nodes)>2
    assert all(a in restored.nodes and b in restored.nodes for a,b in restored.edges)


def test_capsule_requires_measured_per_node_gain_not_read_diversity():
    graph=graph_fixture()
    graph.read(["n0"],"math",1,True)
    graph.read(["n0"],"math",2,True)
    assert graph.nodes["n0"]["kind"]=="trajectory"
    graph.read(["n0"],"science",3,True)
    assert graph.nodes["n0"]["kind"]=="trajectory"
    graph.record_gain("n0","paired-train",.3,["training-query"],"science")
    graph.promote()
    assert graph.nodes["n0"]["kind"]=="capsule"


def test_transfer_filters_real_nodes_and_does_not_mutate_initial():
    graph = graph_fixture()
    source = restrict_graph(graph,{"scope":"source","matched_capacity":2},"math")
    assert len(source.nodes)==2 and all("math" not in n["domains"] for n in source.nodes.values())
    assert len(graph.nodes)==6


def test_manifest_deduplicates_and_preserves_false(tmp_path):
    rows = [{"question":f"q {i}","answer":False} for i in range(20)]
    rows.append({"question":"  Q 0  ","answer":False})
    write_json(tmp_path/"strategyqa_real.json",rows)
    m = prepare_manifest(tmp_path,tmp_path/"manifest.json",["strategyqa"])
    assert len(m["duplicates_removed"])==1 and len(m["samples"])==20
    assert all(r["answers"] == ["False"] for r in m["samples"])
    validate_manifest(m)
    m["samples"][0]["question"]="changed"
    with pytest.raises(ValueError,match="hash"):
        validate_manifest(m)


def test_scoring_no_substring_or_execution_proxy():
    assert score_answer("FINAL: no",{"metric":"boolean","answers":["false"]})["score"]==1
    assert score_answer("FINAL: nowhere",{"metric":"boolean","answers":["false"]})["score"]==0
    assert score_answer("Paris and London",{"metric":"em","answers":["Paris"]})["score"]==0
    assert score_answer("print(1)",{"metric":"external_code","answers":["x"]})["score"] is None
    assert choice_gold("Q",["-1.5"],{"choices":["1.5","-1.5"]})==["B"]
    with pytest.raises(ValueError):
        choice_gold("Q Options: (A) 10 (B) 12",["11"],{})


def test_budget_no_repetition_and_cost_separation():
    m=LatentMemory(8,12,max_tokens=32)
    graph=graph_fixture()
    assert m(torch.randn(8),graph,budget=32)[0].shape[0]==32
    with pytest.raises(ValueError):
        m(torch.randn(8),graph,budget=33)
    assert choose_budget({"budget_policy":"entropy","min_tokens":12,"max_tokens":32,"entropy_scale":10},.3)==15


def test_output_token_accounting():
    seq=torch.tensor([[10,11,12,20,21]])
    assert completion_ids(seq,3,False).tolist()==[20,21]
    assert completion_ids(torch.tensor([[20,21]]),3,True).tolist()==[20,21]


def test_gemma_ple_preserves_continuous_gradient_without_nearest_ids():
    backend=HFBackend.__new__(HFBackend)
    embedding=nn.Embedding(20,12).requires_grad_(False)
    backend.model=SimpleNamespace(get_input_embeddings=lambda:embedding)
    backend.ple_model=SimpleNamespace(get_per_layer_inputs=lambda ids, _:torch.ones(1,ids.shape[1],2,3))
    prefix=torch.randn(4,12,requires_grad=True)
    result=backend.continuous_inputs(torch.tensor([[1,2,3]]),prefix)
    assert result["per_layer_inputs"].shape==(1,7,2,3)
    assert result["per_layer_inputs"][:,:4].count_nonzero()==0
    result["inputs_embeds"].sum().backward()
    assert prefix.grad.abs().sum()>0


def test_full_loss_keeps_prefix_gradient():
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(20,12)
            self.head=nn.Linear(12,20)
        def get_input_embeddings(self):
            return self.emb
        def forward(self,inputs_embeds,attention_mask,labels,use_cache):
            logits=self.head(inputs_embeds.cumsum(1))
            return SimpleNamespace(loss=nn.functional.cross_entropy(logits[:,:-1].reshape(-1,20),labels[:,1:].reshape(-1),ignore_index=-100))
    backend=HFBackend.__new__(HFBackend)
    backend.model=TinyLM().requires_grad_(False)
    backend.device="cpu"
    backend.ple_model=None
    backend.max_new=8
    backend.prompt_ids=lambda prompt:torch.tensor([[1,2]])
    class Batch:
        input_ids=torch.tensor([[3,4]])
        def to(self,device): return self
    backend.tokenizer=lambda *a,**k:Batch()
    prefix=torch.randn(3,12,requires_grad=True)
    backend.loss("question","answer",prefix).backward()
    assert prefix.grad.abs().sum()>0 and all(p.grad is None for p in backend.model.parameters())


def test_official_score_hash_binding(tmp_path):
    row=dict(id="a",prediction="code",prediction_sha256=digest("code"),score=None,status="requires_official_harness")
    p,s,o=tmp_path/"pred.jsonl",tmp_path/"score.jsonl",tmp_path/"out.jsonl"
    p.write_text(json.dumps(row)+"\n",encoding="utf-8")
    v=dict(id="a",prediction_sha256=digest("different"),score=1,harness="test",revision="abc")
    s.write_text(json.dumps(v)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="Missing"):
        import_scores(p,s,o)
    v["prediction_sha256"]=digest("code")
    s.write_text(json.dumps(v)+"\n",encoding="utf-8")
    import_scores(p,s,o)
    assert json.loads(o.read_text())["score"]==1


def test_recover_gpqa_does_not_lose_correct_option():
    row={"Question":"q","Correct Answer":"11","Incorrect Answer 1":"10","Incorrect Answer 2":"12","Incorrect Answer 3":"13"}
    out=convert("gpqa",[row],"revision")[0]
    assert out["choices"][ord(out["answer"])-65]=="11"


def test_bootstrap_is_seed_unit():
    result=bootstrap_mean([40,42,44])
    assert result["mean"]==42 and result["std"]==2 and result["resampling_unit"]=="training_seed"


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False): return [ord(c)%128 for c in text]
    def decode(self,ids,**kwargs): return "".join(chr(i) for i in ids)


class FakeBackend:
    device="cpu"
    width=8
    hidden=12
    resolved={"model":"synthetic"}
    tokenizer=FakeTokenizer()
    def __init__(self,*args,**kwargs): pass
    def encode(self,text):
        g=torch.Generator().manual_seed(int(digest(text)[:8],16))
        return torch.nn.functional.normalize(torch.randn(8,generator=g),dim=0)
    def synchronize(self): pass
    def generate(self,prompt,prefix=None):
        return "FINAL: no",dict(prompt_tokens=10,output_tokens=3,latent_positions=0 if prefix is None else len(prefix),generation_seconds=.01)
    def entropy(self,prompt): return .3,dict(probe_tokens=7,probe_seconds=.002)


def test_prediction_updates_use_no_gold_and_counts_all_calls():
    graph=graph_fixture()
    graph.prune_threshold=-100.  # Keep writes observable in this provenance test.
    model=LatentMemory(8,12,max_tokens=32)
    row=dict(id="fresh",question="q",answers=["secret_gold"],domain="qa",dataset="toy",metric="em",raw={})
    condition=dict(family="latent",agents=3,k=4,budget_policy="entropy",entropy_scale=10,evolution="full")
    before=graph.state_dict()
    result=predict(FakeBackend(),model,graph,row,condition,42,1,False)
    assert len(graph.nodes)==len(before["nodes"])
    assert result["prediction"]=="FINAL: no" and result["cost"]["probe_tokens"]==21
    assert result["cost"]["output_tokens"]==9 and result["cost"]["all_text_tokens"]==60
    predict(FakeBackend(),model,graph,row,condition,42,1,True)
    new_nodes=[n for n in graph.nodes.values() if "fresh" in n["source_ids"]]
    assert new_nodes and all("secret_gold" not in n["text"] and "FINAL: no" in n["text"] for n in new_nodes)


def test_runner_roundtrip_and_paired_aggregation_with_toy_backend(tmp_path,monkeypatch):
    from KGLMF.experiments.kbs.create_configs import example_config
    from KGLMF.experiments.kbs.runner import evaluate
    from KGLMF.experiments.kbs.reporting import aggregate
    import KGLMF.experiments.kbs.backend as backend_module
    monkeypatch.setattr(backend_module,"HFBackend",FakeBackend)
    write_json(tmp_path/"strategyqa_real.json",[{"question":f"q{i}","answer":False,"source_revision":"toy"} for i in range(20)])
    manifest=prepare_manifest(tmp_path,tmp_path/"manifest.json",["strategyqa"])
    config=example_config("synthetic")
    config["backend"]["device"]="cpu"
    for suite in config["suites"].values():
        for condition in suite:
            condition["budget_policy"]="fixed"
    write_json(tmp_path/"config.json",config)
    graph=MemoryGraph(8)
    for row in manifest["samples"]:
        if row["split"]=="train":
            graph.add(FakeBackend().encode(row["question"]),[row["id"]],row["domain"],"reference")
    from KGLMF.experiments.kbs.runner import make_memory
    module=make_memory(FakeBackend(),config["memory"])
    checkpoint=tmp_path/"toy.pt"
    torch.save(dict(version=VERSION,stage="trained",manifest_sha256=manifest["sha256"],seed=42,
                    backend_config=config["backend"],memory_config=config["memory"],training_config={},
                    memory=module.state_dict(),graph=graph.state_dict(),selected_epoch=0),checkpoint)
    evaluate(tmp_path/"config.json",tmp_path/"manifest.json",checkpoint,"main",tmp_path/"run",42,
             selected=["none_single","kg_multi"])
    result=aggregate([tmp_path/"run"],tmp_path/"summary.json")
    assert len(result["summaries"])==2 and len(result["paired_comparisons"])==1
    assert all(r["performance"]["mean"]==100 for r in result["summaries"])
    with pytest.raises(ValueError,match="Duplicate"):
        aggregate([tmp_path/"run",tmp_path/"run"],tmp_path/"bad.json")


def test_bigcode_hash_export_import(tmp_path):
    from KGLMF.experiments.kbs.harness import bigcode_export,bigcode_import
    task=dict(id="x",dataset="bigcodebench",prediction="```python\ndef task_func(): return 1\n```",
              prediction_sha256="predhash",raw={"task_id":"BigCodeBench/1","test":"test","entry_point":"task_func"})
    write_json(tmp_path/"export.json",[task])
    bigcode_export(tmp_path/"export.json",tmp_path/"samples.jsonl")
    sample=json.loads((tmp_path/"samples.jsonl").read_text())
    write_json(tmp_path/"results.json",{"eval":{"BigCodeBench/1":[dict(sample,status="pass")]}})
    bigcode_import(tmp_path/"results.json",tmp_path/"samples.jsonl.mapping.json",tmp_path/"verdicts.jsonl","abc")
    assert json.loads((tmp_path/"verdicts.jsonl").read_text())["score"]==1


def test_degree_preserving_random_graph():
    from collections import Counter
    from KGLMF.experiments.kbs.graph_analysis import randomized_edges,summarize_graph
    edges={tuple(sorted((str(i),str((i+1)%12)))) for i in range(12)}
    shuffled,swaps=randomized_edges(edges,42)
    assert swaps>0 and Counter(v for e in edges for v in e)==Counter(v for e in shuffled for v in e)
    nodes={str(i):{"kind":"trajectory"} for i in range(12)}
    stats=summarize_graph(nodes,edges)
    assert stats["reachability_fraction"]==1 and stats["mean_clustering"]==0
