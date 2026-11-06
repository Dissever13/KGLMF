"""Explicit initialization, paired feedback and the two learned stages.

No model is loaded on import. Code verdicts require an explicitly configured
isolated evaluator; ungraded items never become zero rewards.
"""
import copy
import json
import os
import random
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from KGLMF.src.constants import ANCHORS, ROLES
from KGLMF.src.evolution.refresh import refresh_graph
from KGLMF.src.data.protocol import VERSION, digest, read_json, write_json, validate_manifest, validate_sources, validate_config, provenance_ids, score_answer


class TaskScorer:
    """Evaluator-only interface. Gold/tests never appear in generation prompts."""
    def __init__(self, config):
        self.config = config

    def score(self, prediction, row, required=True):
        result = score_answer(prediction,row)
        if result["score"] is not None:
            return result
        command = self.config.get("command")
        if not command:
            if not required:
                return result
            raise ValueError("Configure scoring.command as an argv list for an isolated code/math evaluator")
        if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):
            raise ValueError("Evaluator command must be an argv list; shell execution is disabled")
        request = dict(id=row["id"],dataset=row["dataset"],metric=row["metric"],prediction=prediction,
                       answers=row["answers"],raw=row["raw"],prediction_sha256=digest(prediction))
        proc = subprocess.run(command,input=json.dumps(request,ensure_ascii=False),text=True,
                              encoding="utf-8",env=dict(os.environ,PYTHONIOENCODING="utf-8"),
                              capture_output=True,timeout=self.config.get("timeout_seconds",120),check=True,shell=False)
        verdict = json.loads(proc.stdout)
        if verdict.get("id") != row["id"] or verdict.get("prediction_sha256") != digest(prediction):
            raise ValueError("Evaluator response is not bound to the exact prediction")
        if verdict.get("score") not in (0,1) or not verdict.get("harness") or not verdict.get("revision"):
            raise ValueError("Evaluator must return binary task score and versioned harness")
        return dict(result,score=float(verdict["score"]),status="external_harness_scored",
                    harness=verdict["harness"],harness_revision=verdict["revision"])


def round_robin(rows, seed=42):
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"],[]).append(row)
    for name,items in groups.items():
        random.Random(f"{seed}:{name}").shuffle(items)
    result = []
    for i in range(max(map(len,groups.values()),default=0)):
        for name in sorted(groups):
            if i < len(groups[name]):
                result.append(groups[name][i])
    return result


def initialize(config_path, manifest_path, output, seed):
    from KGLMF.src.backends.huggingface import HFBackend
    from KGLMF.src.execution.runner import seed_all, make_memory, build_graph
    if Path(output).exists():
        raise FileExistsError(output)
    config = validate_config(read_json(config_path))
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    seed_all(seed)
    backend = HFBackend(config["backend"])
    memory = make_memory(backend,config["memory"])
    x,y,norm,retained = backend.lexical_anchors(ANCHORS)
    memory.initialize_cma(x,y,norm)
    rows = [r for r in manifest["samples"] if r["split"] == "train"]
    graph = build_graph(backend,rows,config["memory"],memory)
    state = dict(version=VERSION,stage="initialized",seed=seed,manifest_sha256=manifest["sha256"],
        backend_config=config["backend"],resolved_backend=backend.resolved,memory_config=config["memory"],
        training_config=config["training"],memory={k:v.detach().cpu() for k,v in memory.state_dict().items()},
        graph=graph.state_dict(),selected_epoch=-1,anchors=retained,train_ids=[r["id"] for r in rows])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,output)
    return str(output)


def collect_feedback(config_path, manifest_path, initial, output, seed, adapter=None):
    """Training-only with/no-memory and with/without-each-selected-node pairs."""
    from KGLMF.src.backends.huggingface import HFBackend
    from KGLMF.src.execution.runner import seed_all, load_checkpoint, prompt_for, question_text
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = validate_config(read_json(config_path))
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    seed_all(seed)
    backend = HFBackend(config["backend"])
    memory,graph,state = load_checkpoint(initial,config,manifest,backend,allow_initialized=True)
    if state["seed"] != seed or state["stage"] not in {"initialized","likelihood_trained"}:
        raise ValueError("Feedback requires matching initialization or selected likelihood checkpoint")
    selected_adapter = "cma" if state.get("training_config",{}).get("freeze_adapter") else "trained"
    if state["stage"] == "initialized":
        selected_adapter = adapter or "trained"
    elif adapter and adapter != selected_adapter:
        raise ValueError("Feedback adapter must match the selected likelihood condition")
    scorer = TaskScorer(config.get("scoring",{}))
    output.mkdir(parents=True)
    write_json(output/"feedback.json",dict(version=VERSION,status="running",seed=seed,
        manifest_sha256=manifest["sha256"],initial_sha256=__import__("hashlib").sha256(Path(initial).read_bytes()).hexdigest(),
        protocol="training_only_fixed_retrieved_set_pairs",checkpoint_stage=state["stage"],adapter=selected_adapter))
    with (output/"pairs.jsonl").open("w",encoding="utf-8") as stream,torch.no_grad():
        for row in round_robin([r for r in manifest["samples"] if r["split"] == "train"],seed):
            for role in ROLES:
                prompt = prompt_for(row,role)
                query = backend.encode(question_text(row))
                prefix,meta = memory(query,graph,role,k=config["training"]["k"],budget=config["training"]["budget"],excluded=provenance_ids(row),adapter=selected_adapter)
                if prefix is None:
                    raise ValueError("Empty training memory after group exclusion")
                pair_seed = seed+int(digest([row["id"],role])[:8],16)
                branches = [("with",None,prefix),("none",None,None)]
                for nid in meta["ids"]:
                    remaining = [n for n in meta["ids"] if n != nid]
                    minus,_ = memory(query,graph,role,k=config["training"]["k"],budget=config["training"]["budget"],
                                     excluded=provenance_ids(row),selected_ids=remaining,adapter=selected_adapter)
                    branches.append(("without_node",nid,minus))
                for branch,nid,branch_prefix in branches:
                    seed_all(pair_seed)
                    prediction,cost = backend.generate(prompt,branch_prefix)
                    score = scorer.score(prediction,row,required=False)
                    score["extracted_answer"] = score.pop("prediction")
                    record = dict(version=VERSION,id=row["id"],group_id=row["group_id"],dataset=row["dataset"],metric=row["metric"],
                        role=role,pair=digest([row["id"],role]),branch=branch,removed_node=nid,selected_ids=meta["ids"],
                        budget=meta["budget"],prediction=prediction,prediction_sha256=digest(prediction),cost=cost,
                        seed=pair_seed,**score)
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                    stream.flush()
    metadata = read_json(output/"feedback.json")
    metadata["status"] = "complete"
    metadata["pairs_sha256"] = __import__("hashlib").sha256((output/"pairs.jsonl").read_bytes()).hexdigest()
    write_json(output/"feedback.json",metadata)
    return str(output)


def score_feedback(config_path, manifest_path, feedback, output):
    """Evaluate exported pairs without regeneration; normally run by the author."""
    if Path(output).exists():
        raise FileExistsError(output)
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    rows = {r["id"]:r for r in manifest["samples"] if r["split"] == "train"}
    scorer = TaskScorer(read_json(config_path).get("scoring",{}))
    records = []
    for line in Path(feedback).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] not in rows:
            raise ValueError("Non-training query in feedback")
        if record["score"] is None:
            verdict = scorer.score(record["prediction"],rows[record["id"]])
            verdict["extracted_answer"] = verdict.pop("prediction")
            record.update(verdict)
        records.append(record)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    metadata = read_json(Path(feedback).parent/"feedback.json")
    if metadata.get("pairs_sha256") != __import__("hashlib").sha256(Path(feedback).read_bytes()).hexdigest():
        raise ValueError("Raw paired feedback changed after collection")
    metadata["scored_pairs_sha256"] = __import__("hashlib").sha256(Path(output).read_bytes()).hexdigest()
    write_json(Path(output).parent/"feedback.json",metadata)
    return str(output)


def feedback_pairs(path, manifest, graph):
    train = {r["id"]:r for r in manifest["samples"] if r["split"] == "train"}
    pairs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("version") != VERSION or record["id"] not in train or record.get("group_id") != train[record["id"]]["group_id"]:
            raise ValueError("Feedback source/version does not match training manifest")
        if record.get("score") is None or not 0 <= record["score"] <= 1 or digest(record["prediction"]) != record["prediction_sha256"]:
            raise ValueError("All feedback predictions must have valid task scores and hashes")
        row = train[record["id"]]
        if any(n not in graph.eligible(provenance_ids(row)) for n in record["selected_ids"]):
            raise ValueError("Feedback retrieves its own group or invalid node")
        key = record["pair"]
        pair = pairs.setdefault(key,{"row":row,"records":{}})
        branch = (record["branch"],record.get("removed_node"))
        if branch in pair["records"]:
            raise ValueError("Duplicate feedback branch")
        pair["records"][branch] = record
    expected_pairs = {digest([r["id"],role]) for r in train.values() for role in ROLES}
    if set(pairs) != expected_pairs:
        raise ValueError("Feedback must cover every training query and all three roles")
    for pair in pairs.values():
        records = pair["records"]
        if ("with",None) not in records or ("none",None) not in records:
            raise ValueError("Missing paired memory/no-memory branch")
        with_mem = records[("with",None)]
        expected = {("with",None),("none",None)} | {("without_node",n) for n in with_mem["selected_ids"]}
        if set(records) != expected or any((r["id"],r["role"],r["seed"],r["budget"],r["selected_ids"]) !=
            (with_mem["id"],with_mem["role"],with_mem["seed"],with_mem["budget"],with_mem["selected_ids"]) for r in records.values()):
            raise ValueError("Incomplete or mismatched per-node counterfactual pairs")
    return list(pairs.values())


def mean_development_score(backend, memory, graph, rows, cfg, scorer, fixed_adapter=False):
    from KGLMF.src.execution.runner import prompt_for, question_text, seed_all
    by_dataset = {}
    with torch.no_grad():
        for row in rows:
            seed_all(int(row["id"][:8],16))
            prefix,_ = memory(backend.encode(question_text(row)),graph,"solver",k=cfg["k"],budget=cfg["budget"],
                              adapter="cma" if fixed_adapter else "trained")
            prediction,_ = backend.generate(prompt_for(row,"solver"),prefix)
            by_dataset.setdefault(row["dataset"],[]).append(scorer.score(prediction,row)["score"])
    if not by_dataset:
        raise ValueError("Empty development partition")
    return sum(sum(v)/len(v) for v in by_dataset.values())/len(by_dataset)


def train(config_path, manifest_path, output, seed, freeze_adapter=False, initial=None, feedback=None):
    from KGLMF.src.backends.huggingface import HFBackend
    from KGLMF.src.execution.runner import seed_all, load_checkpoint, prompt_for, question_text, training_target
    if not initial or not feedback:
        raise ValueError("train requires --initial and --feedback from initialize/feedback/score-feedback")
    if Path(output).exists():
        raise FileExistsError(output)
    config = validate_config(read_json(config_path))
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    seed_all(seed)
    backend = HFBackend(config["backend"])
    memory,graph,state = load_checkpoint(initial,config,manifest,backend,allow_initialized=True)
    if state["seed"] != seed or state["stage"] != "initialized":
        raise ValueError("Training requires matching initialized checkpoint")
    pairs = feedback_pairs(feedback,manifest,graph)
    metadata = read_json(Path(feedback).parent/"feedback.json")
    if metadata.get("status") != "complete" or metadata.get("manifest_sha256") != manifest["sha256"] or metadata.get("initial_sha256") != __import__("hashlib").sha256(Path(initial).read_bytes()).hexdigest():
        raise ValueError("Feedback metadata/initial checkpoint mismatch")
    if __import__("hashlib").sha256(Path(feedback).read_bytes()).hexdigest() not in {metadata.get("pairs_sha256"),metadata.get("scored_pairs_sha256")}:
        raise ValueError("Feedback file was not collected/scored by this protocol")
    if metadata.get("adapter") != ("cma" if freeze_adapter else "trained"):
        raise ValueError("Initial feedback must match the trained/CMA-only adapter condition")
    cfg = dict(config["training"],freeze_adapter=freeze_adapter)
    dev = [r for r in manifest["samples"] if r["split"] == "dev"]
    scorer = TaskScorer(config.get("scoring",{}))
    # Check availability before any gradient update, rather than failing at epoch end.
    if any(r["metric"].startswith("external_") for r in dev) and not scorer.config.get("command"):
        raise ValueError("Development code/math task-score selection requires scoring.command")
    trace = []
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.with_suffix(".training.json"),dict(version=VERSION,status="running",trace=[]))
    router_params = memory.router_parameters()
    scorer_params = list(memory.scorer.parameters())
    optimizer = torch.optim.Adam(router_params+scorer_params,lr=cfg["learning_rate"])
    best,calibrated = -float("inf"),None
    for epoch in range(cfg["epochs"]):
        memory.eval()  # Dropout disabled for paired-score calibration.
        losses = []
        for pair in pairs:
            row = pair["row"]
            with_mem = pair["records"][("with",None)]
            delta = with_mem["score"]-pair["records"][("none",None)]["score"]
            query = backend.encode(question_text(row))
            routing = memory.route(query,graph,with_mem["role"],k=cfg["k"],excluded=provenance_ids(row))
            indices = [routing["ids"].index(n) for n in with_mem["selected_ids"]]
            mass_loss = (routing["initial"][indices].sum()-(1+delta)/2)**2
            with torch.no_grad():
                composed = memory.compose(graph,with_mem["selected_ids"],with_mem["role"],with_mem["budget"])
            scorer_loss = (memory.utility_score(query,composed)-delta)**2
            optimizer.zero_grad(set_to_none=True)
            (mass_loss+scorer_loss).backward()
            torch.nn.utils.clip_grad_norm_(router_params+scorer_params,cfg["gradient_clip"])
            optimizer.step()
            losses.append(float((mass_loss+scorer_loss).detach()))
        score = mean_development_score(backend,memory,graph,dev,cfg,scorer,freeze_adapter)
        trace.append(dict(stage="calibration",epoch=epoch,loss=sum(losses)/len(losses),dev_score=score))
        if score > best:
            best,calibrated = score,copy.deepcopy(memory.state_dict())
    memory.load_state_dict(calibrated)
    memory.freeze_calibration()
    if freeze_adapter:
        memory.adapter.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in memory.parameters() if p.requires_grad],lr=cfg["learning_rate"])
    best,gradient_checked = -float("inf"),False
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    train_rows = [r for r in manifest["samples"] if r["split"] == "train"]
    for epoch in range(cfg["epochs"]):
        memory.train()
        losses = []
        for index,row in enumerate(round_robin(train_rows,seed+epoch)):
            role = ROLES[(epoch+index)%len(ROLES)]
            budget = cfg["budgets"][(epoch+index)%len(cfg["budgets"])]
            query = backend.encode(question_text(row))
            prefix,meta = memory(query,graph,role,k=cfg["k"],budget=budget,excluded=provenance_ids(row),
                               reencode=True,adapter="cma" if freeze_adapter else "trained")
            if prefix is None:
                raise ValueError("Empty likelihood-training retrieval")
            nll = backend.loss(prompt_for(row,role),training_target(row),prefix)
            regularizer = sum(p.square().sum() for p in memory.parameters() if p.requires_grad)
            if not gradient_checked:
                watched = [memory.pool,memory.write_pool,memory.gate_bias,memory.roles.weight]
                if not freeze_adapter:
                    watched.append(memory.adapter[0].weight)
                checks = torch.autograd.grad(nll,watched,retain_graph=True,allow_unused=True)
                if any(g is None or not torch.isfinite(g).all() or float(g.abs().sum()) == 0 for g in checks):
                    raise RuntimeError("Broken continuous prefix/synthesizer/writer/gate gradient")
                if any(p.grad is not None for p in memory.router_parameters()+list(memory.scorer.parameters())):
                    raise RuntimeError("Likelihood unexpectedly trained calibrated Router/scorer")
                if any(p.grad is not None or p.requires_grad for module in (backend.model,backend.encoder) for p in module.parameters()):
                    raise RuntimeError("Backbone/encoder must remain frozen")
                gradient_checked = True
            # Metadata is detached for JSON reporting; use the live coefficients
            # here so Eq.9 updates their logits without differentiating counts.
            structural_cost = memory.structural_cost(meta["budget"],meta["active_nodes"],active_edges=meta["active_edges"])
            loss = nll+cfg["l2"]*regularizer+cfg["cost_weight"]*structural_cost
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in memory.parameters() if p.requires_grad],cfg["gradient_clip"])
            optimizer.step()
            losses.append(float(nll.detach()))
        memory.eval()
        refresh_graph(memory,graph)
        score = mean_development_score(backend,memory,graph,dev,cfg,scorer,freeze_adapter)
        trace.append(dict(stage="likelihood",epoch=epoch,nll=sum(losses)/len(losses),dev_score=score))
        if score > best:
            best = score
            torch.save(dict(state,stage="likelihood_trained",memory={k:v.detach().cpu() for k,v in memory.state_dict().items()},
                graph=graph.state_dict(),training_config=cfg,selected_epoch=epoch,dev_score=best,
                gradient_checked=gradient_checked,
                feedback_sha256=__import__("hashlib").sha256(Path(feedback).read_bytes()).hexdigest()),output)
        write_json(output.with_suffix(".training.json"),dict(version=VERSION,trace=trace,gradient_checked=gradient_checked,
            status="running",train_ids=[r["id"] for r in train_rows],dev_ids=[r["id"] for r in dev]))
    log = read_json(output.with_suffix(".training.json"))
    log["status"] = "complete"
    write_json(output.with_suffix(".training.json"),log)
    return str(output)


def finalize(config_path, manifest_path, checkpoint, feedback, output, seed):
    """Use the selected trained model's own paired gains for Capsule/maintenance."""
    from KGLMF.src.backends.huggingface import HFBackend
    from KGLMF.src.execution.runner import seed_all, load_checkpoint
    if Path(output).exists():
        raise FileExistsError(output)
    config = validate_config(read_json(config_path))
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    seed_all(seed)
    backend = HFBackend(config["backend"])
    memory,graph,state = load_checkpoint(checkpoint,config,manifest,backend,allow_initialized=True)
    if state["stage"] != "likelihood_trained" or state["seed"] != seed:
        raise ValueError("Finalization requires the matching selected likelihood checkpoint")
    metadata = read_json(Path(feedback).parent/"feedback.json")
    feedback_hash = __import__("hashlib").sha256(Path(feedback).read_bytes()).hexdigest()
    checkpoint_hash = __import__("hashlib").sha256(Path(checkpoint).read_bytes()).hexdigest()
    if metadata.get("status") != "complete" or metadata.get("initial_sha256") != checkpoint_hash or metadata.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Final graph gains must be measured with this selected model, not initial/random-prefix gains")
    if feedback_hash not in {metadata.get("pairs_sha256"),metadata.get("scored_pairs_sha256")}:
        raise ValueError("Unregistered paired feedback file")
    pairs = feedback_pairs(feedback,manifest,graph)
    for pair in pairs:
        with_mem = pair["records"][("with",None)]
        event = "train:"+with_mem["pair"]
        graph.read(with_mem["selected_ids"],pair["row"]["domain"],0,event_id=event,source_ids=provenance_ids(pair["row"]))
        for nid in with_mem["selected_ids"]:
            minus = pair["records"][("without_node",nid)]
            graph.record_gain(nid,event,with_mem["score"]-minus["score"],provenance_ids(pair["row"]),pair["row"]["domain"])
    maintenance = graph.maintain(0)
    if not graph.nodes:
        raise ValueError("Maintenance removed all memory; inspect feedback/calibrated utility settings")
    memory.eval()
    dev = [r for r in manifest["samples"] if r["split"] == "dev"]
    dev_score = mean_development_score(backend,memory,graph,dev,state["training_config"],TaskScorer(config.get("scoring",{})),state["training_config"].get("freeze_adapter",False))
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state,stage="trained",graph=graph.state_dict(),graph_feedback_sha256=feedback_hash,
                    selected_checkpoint_sha256=checkpoint_hash,maintenance=maintenance,post_maintenance_dev_score=dev_score),output)
    write_json(Path(output).with_suffix(".training.json"),dict(version=VERSION,status="complete",stage="finalized",seed=seed,
        selected_epoch=state["selected_epoch"],selected_model_dev_score=state["dev_score"],post_maintenance_dev_score=dev_score,
        graph_feedback_sha256=feedback_hash,maintenance=maintenance))
    return str(output)


def score_predictions(config_path, manifest_path, predictions, output):
    if Path(output).exists():
        raise FileExistsError(output)
    manifest = validate_sources(validate_manifest(read_json(manifest_path)))
    rows = {r["id"]:r for r in manifest["samples"] if r["split"] == "test"}
    scorer = TaskScorer(read_json(config_path).get("scoring",{}))
    result = []
    for line in Path(predictions).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] not in rows or record.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("Evaluation predictions/manifest mismatch")
        if record["score"] is None:
            verdict = scorer.score(record["prediction"],rows[record["id"]])
            record.update({k:verdict[k] for k in ("score","status","harness","harness_revision")})
        result.append(record)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in result),encoding="utf-8")
    return str(output)


def export_memory(checkpoint, manifest_path, output):
    """CPU-only text/provenance export for identical baseline information access."""
    if Path(output).exists():
        raise FileExistsError(output)
    manifest = validate_manifest(read_json(manifest_path))
    state = torch.load(checkpoint,map_location="cpu",weights_only=True)
    if state.get("version") != VERSION or state.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Checkpoint/manifest mismatch")
    train_ids = [r["id"] for r in manifest["samples"] if r["split"] == "train"]
    permitted = {sid for r in manifest["samples"] if r["split"] == "train" for sid in provenance_ids(r)} | {"role:"+role for role in ROLES}
    if any(not set(n["source_ids"]).issubset(permitted) for n in state["graph"]["nodes"].values()):
        raise ValueError("Non-training source in exported initial memory")
    write_json(output,dict(version=VERSION,manifest_sha256=manifest["sha256"],training_source_ids=train_ids,
        experiences=[{k:n[k] for k in ("source_ids","domains","task","text","kind","role")} for n in state["graph"]["nodes"].values()]))
    return str(output)
