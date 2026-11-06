"""Train/evaluate only on explicit CLI calls. No downloads or runs on import."""
import copy
import json
import math
import platform
import random
import time
from pathlib import Path

import torch

from KGLMF.src.composer.base import LatentMemory
from KGLMF.src.graph.base import MemoryGraph
from KGLMF.src.constants import ROLES
from KGLMF.src.data.protocol import VERSION, digest, read_json, write_json, validate_manifest, validate_sources, validate_config, provenance_ids, score_answer


INSTRUCTIONS = {
    "planner": "Analyze the problem and suggest a solution. End with FINAL: followed by your best answer.",
    "solver": "Solve the problem. End with FINAL: followed only by the answer (a letter for multiple choice; yes/no for boolean questions).",
    "critic": "Check the proposed solutions, correct errors, and give the final solution. End with FINAL: followed only by the answer (a letter for multiple choice; yes/no for boolean questions).",
}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def question_text(row):
    text = row["question"]
    # Some source questions already contain choices; only append structured ones.
    choices = row.get("raw", {}).get("choices", row.get("raw", {}).get("options"))
    if isinstance(choices, dict) and "text" in choices:
        labels = choices.get("label", list("ABCDEFGHIJ")[:len(choices["text"])])
        text += "\n" + "\n".join(f"{a}. {b}" for a,b in zip(labels,choices["text"]))
    elif isinstance(choices, list):
        text += "\n" + "\n".join(f"{chr(65+i)}. {value}" for i,value in enumerate(choices))
    return text


def prompt_for(row, role, history="", memory=""):
    suffix = " Return the complete code in a fenced python block." if row["domain"] == "code" else ""
    if row["domain"] == "math":
        suffix = r" Write the final mathematical answer in \boxed{...}."
    return (INSTRUCTIONS[role]+suffix+"\nQuestion:\n"+question_text(row)+
            ("\nRetrieved training experience:\n"+memory if memory else "")+
            ("\nPrevious agents:\n"+history if history else ""))


def training_target(row):
    answer = row["answers"][0]
    if row["domain"] == "code":
        return "```python\n"+answer+"\n```"
    if row["domain"] == "math":
        return "FINAL: \\boxed{"+answer+"}"
    if row["metric"] == "boolean":
        answer = {"false":"no","true":"yes","0":"no","1":"yes"}.get(answer.lower(),answer)
    return "FINAL: "+answer


def experience_components(backend, row, role, response, kind="trajectory", knowledge=None):
    # Only permitted training labels or completed label-free responses reach here.
    zero = torch.zeros(backend.width)
    return [backend.encode(INSTRUCTIONS[role]).cpu(),backend.encode(question_text(row)).cpu(),
            backend.encode(response).cpu() if kind == "trajectory" else zero,
            backend.encode(response if knowledge is None else knowledge).cpu() if kind == "trajectory" else zero]


def graph_options(config):
    keys = {"max_nodes","edge_threshold","neighbors","per_type_capacity","merge_threshold","prune_threshold",
            "capsule_threshold","redundancy_threshold","decay","edge_coefficients","utility_coefficients","write_tokens"}
    return {k:v for k,v in config.items() if k in keys}


def build_graph(backend, rows, memory_config, memory):
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("Only training examples can populate initial memory")
    graph = MemoryGraph(backend.width,**graph_options(memory_config))
    memory.eval()
    for role in ROLES:
        vector = backend.encode(INSTRUCTIONS[role]).cpu()
        components = [vector,torch.zeros_like(vector),torch.zeros_like(vector),torch.zeros_like(vector)]
        with torch.no_grad():
            tokens = memory.write(components)
        graph.add(tokens,["role:"+role],"unspecified",INSTRUCTIONS[role],kind="role",role=role,components=components)
    from KGLMF.src.training.optimization import round_robin
    # Populate at most one type capacity of paired context/trajectory sources.
    # Selection uses source IDs/seed only, without feedback or evaluation access.
    population = round_robin(rows,42)[:memory_config.get("initial_examples",graph.per_type_capacity)]
    for row in population:
        q, a = question_text(row), row["answers"][0]
        role = ROLES[int(row["id"][:8],16)%len(ROLES)]
        seed_all(42+int(row["id"][:8],16))
        interaction,_ = backend.generate(prompt_for(row,role))
        for kind in ("context","trajectory"):
            components = experience_components(backend,row,role,interaction,kind,knowledge=a)
            with torch.no_grad():
                tokens = memory.write(components)
            text = f"Question: {q}"+(f"\nInteraction: {interaction}\nVerified training answer: {a}" if kind == "trajectory" else "")
            graph.add(tokens,provenance_ids(row),row["domain"],text,kind=kind,role=role,components=components,task=row["dataset"])
    # Initial capacity selection is label-independent; no gain/pruning is invented.
    graph.maintain(0,merge=False,promote=False,prune=False)
    return graph


def make_memory(backend, config):
    allowed = {"max_tokens","temperature","hops","alpha","write_tokens","dropout"}
    return LatentMemory(backend.width,backend.hidden,**{k:v for k,v in config.items() if k in allowed}).to(backend.device)


def load_checkpoint(path, config, manifest, backend, allow_initialized=False):
    # Load only checkpoints created locally/by a trusted source: torch serialization.
    state = torch.load(path,map_location="cpu",weights_only=True)
    if state.get("version") != VERSION or state.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Legacy checkpoint or manifest mismatch; train KBS checkpoint first")
    if state.get("stage") != "trained" and not (allow_initialized and state.get("stage") in {"initialized","likelihood_trained"}):
        raise ValueError("Incomplete or initialized checkpoint cannot be used for evaluation")
    if state.get("stage") in {"trained","likelihood_trained"}:
        log_path = Path(path).with_suffix(".training.json")
        if log_path.exists() and read_json(log_path).get("status") != "complete":
            raise ValueError("Interrupted training checkpoint cannot be used for formal evaluation")
    for key in ("model","encoder","revision","encoder_revision","dtype"):
        if state["backend_config"].get(key) != config["backend"].get(key):
            raise ValueError(f"Checkpoint backend mismatch: {key}")
    if state["memory_config"] != config["memory"]:
        raise ValueError("Checkpoint memory configuration mismatch")
    memory = make_memory(backend,config["memory"])
    memory.load_state_dict(state["memory"],strict=True)
    if state.get("stage") in {"trained","likelihood_trained"}:
        memory.freeze_calibration()
    memory.eval()
    graph = MemoryGraph.from_state_dict(state["graph"])
    permitted = {sid for r in manifest["samples"] if r["split"] == "train" for sid in provenance_ids(r)} | {"role:"+role for role in ROLES}
    if any(not set(n["source_ids"]).issubset(permitted) for n in graph.nodes.values()):
        raise ValueError("Non-training provenance in initial memory")
    if any(not set(event.get("source_ids",[])).issubset(permitted) for n in graph.nodes.values()
           for event in list(n["events"].values())+list(n["gains"].values())):
        raise ValueError("Non-training feedback provenance in initial memory")
    return memory, graph, state


def train(config_path, manifest_path, output, seed, freeze_adapter=False, initial=None, feedback=None):
    from KGLMF.src.training.optimization import train as staged_train
    return staged_train(config_path,manifest_path,output,seed,freeze_adapter,initial,feedback)


def calibrate(config_path, manifest_path, output):
    """Development-only maximum next-token entropy, Eq.8 (natural log)."""
    if Path(output).exists():
        raise FileExistsError(output)
    config,manifest = validate_config(read_json(config_path)),validate_sources(validate_manifest(read_json(manifest_path)))
    from KGLMF.src.backends.huggingface import HFBackend
    backend = HFBackend(config["backend"])
    values,ids = [],[]
    for row in manifest["samples"]:
        if row["split"] == "dev":
            ids.append(row["id"])
            for role in ROLES:
                values.append(backend.entropy(prompt_for(row,role))[0])
    if not values:
        raise ValueError("Empty development set for controller calibration")
    bmin,bmax = config["controller"]["min_tokens"],config["memory"]["max_tokens"]
    scale = (bmax-bmin)/max(max(values),1e-8)
    for suite in config["suites"].values():
        for condition in suite:
            if condition.get("budget_policy") == "entropy":
                condition.update(min_tokens=bmin,max_tokens=bmax,entropy_scale=scale)
    config["entropy_calibration"] = dict(manifest_sha256=manifest["sha256"],dev_ids=ids,
        maximum_entropy=max(values),scale=scale,roles=list(ROLES),context="role prompts without memory or history")
    write_json(output,config)
    return str(output)


def choose_budget(condition, entropy=None, rng=None):
    policy = condition.get("budget_policy","fixed")
    if policy == "fixed":
        return condition.get("budget",16)
    if policy == "random":
        return rng.choice(condition.get("budgets",[12,16,24,32]))
    if policy == "entropy":
        if entropy is None or not math.isfinite(entropy) or entropy < 0 or "entropy_scale" not in condition:
            raise ValueError("Calibrated nonnegative entropy/scale required")
        scale = condition["entropy_scale"]
        lo,hi = condition.get("min_tokens",12),condition.get("max_tokens",32)
        if not math.isfinite(scale) or scale < 0 or not 1 <= lo <= hi:
            raise ValueError("Invalid entropy controller")
        return min(hi,lo+math.floor(scale*entropy))
    raise ValueError(policy)


def restrict_graph(graph, condition, target_domain):
    scope = condition.get("scope","full")
    if scope == "full" and not condition.get("matched_capacity"):
        return graph
    copy_graph = graph.clone()
    for nid,node in list(copy_graph.nodes.items()):
        domains = set(node["domains"])
        keep = node["kind"] == "role" or scope == "full" or (scope == "target" and domains == {target_domain}) or (scope == "source" and target_domain not in domains)
        if not keep:
            copy_graph.remove(nid)
    capacity = condition.get("matched_capacity")
    if capacity:
        # Identical cap and deterministic subsampling; log actual size, reject scarcity.
        if len(copy_graph.nodes) < capacity:
            raise ValueError(f"Scope {scope} has fewer than {capacity} nodes; lower shared transfer cap")
        roles = {n for n in copy_graph.nodes if copy_graph.nodes[n]["kind"] == "role"}
        if capacity < len(roles):
            raise ValueError("Transfer capacity must retain common role profiles")
        keep = roles | set(sorted((n for n in copy_graph.nodes if n not in roles),key=lambda n:digest(copy_graph.nodes[n]["source_ids"]))[:capacity-len(roles)])
        for nid in list(copy_graph.nodes):
            if nid not in keep:
                copy_graph.remove(nid)
    return copy_graph


def predict(backend, memory, graph, row, condition, seed, episode=0, allow_updates=False):
    started = time.perf_counter()
    history, calls, used = [], [], []
    rng = random.Random(f"{seed}:{row['id']}")
    roles = ["solver"] if condition.get("agents",3) == 1 else list(ROLES)
    if condition.get("agents",3) not in {1,3}:
        raise ValueError("This protocol defines one or three agents")
    for role in roles:
        prompt = prompt_for(row,role,"\n".join(history))
        prefix,meta,probe = None,{"ids":[],"budget":0}, {"probe_tokens":0,"probe_seconds":0.}
        family = condition.get("family","latent")
        retrieval_started = time.perf_counter()
        if family != "none":
            backend.synchronize()
            encode_started = time.perf_counter()
            query = backend.encode(question_text(row))
            backend.synchronize()
            encoding_seconds = time.perf_counter()-encode_started
            if family == "text":
                candidates = graph.eligible(excluded=provenance_ids(row))
                ranked = sorted(candidates,key=lambda n:float(torch.nn.functional.cosine_similarity(
                    query.cpu(),graph.nodes[n]["tokens"].mean(0),dim=0)),reverse=True)
                ids = ranked[:condition.get("k",4)]
                text = "\n\n".join(graph.nodes[n]["text"] for n in ids)
                cap = condition.get("text_memory_tokens",512)
                token_ids = backend.tokenizer.encode(text,add_special_tokens=False)[:cap]
                prompt = prompt_for(row,role,"\n".join(history),backend.tokenizer.decode(token_ids))
                meta = {"ids":ids,"budget":0,"text_memory_tokens":len(token_ids)}
            elif family == "latent":
                entropy = None
                if condition.get("budget_policy") == "entropy":
                    entropy,probe = backend.entropy(prompt)
                budget = choose_budget(condition,entropy,rng)
                prefix,meta = memory(query,graph,role,k=condition.get("k",4),budget=budget,
                                     excluded=provenance_ids(row),topology=condition.get("topology","graph"),
                                     role_condition=condition.get("role_condition",True),
                                     capsules=True,hops=condition.get("hops"),gate=condition.get("gate","dynamic"),
                                     adapter=condition.get("adapter","trained"))
                if condition.get("adapter") == "random" and prefix is not None:
                    raise ValueError("Random adapter requires separately trained checkpoint; use no-memory/flat controls")
                meta["entropy"] = entropy
            else:
                raise ValueError(family)
        backend.synchronize()
        memory_seconds = time.perf_counter()-retrieval_started-probe["probe_seconds"]
        generation_seed = seed+int(digest([row["id"],role])[:8],16)
        seed_all(generation_seed)
        response,cost = backend.generate(prompt,prefix)
        cost["memory_seconds"] = max(0.,memory_seconds)
        cost["communication_tokens"] = sum(len(backend.tokenizer.encode(h,add_special_tokens=False)) for h in history)
        cost["text_memory_tokens"] = meta.get("text_memory_tokens",0)
        cost["structural_cost"] = meta.get("structural_cost",0)
        cost["encoding_seconds"] = encoding_seconds if family != "none" else 0.
        cost["routing_seconds"] = meta.get("routing_seconds",0.)
        cost["synthesis_seconds"] = meta.get("synthesis_seconds",0.)
        # All methods share the same communication truncation limit.
        history_ids = backend.tokenizer.encode(response,add_special_tokens=False)[:condition.get("communication_tokens",256)]
        history.append(f"{role}: "+backend.tokenizer.decode(history_ids))
        used.extend(meta["ids"])
        calls.append(dict(role=role,generation_seed=generation_seed,retrieval=meta,cost={**cost,**probe},response=response))
    backend.synchronize()
    prediction_seconds = time.perf_counter()-started
    score = score_answer(response,row)
    maintenance = {"merged":0,"pruned":0}
    update_start = time.perf_counter()
    if allow_updates:
        # Prediction is complete before any write. Gold and scores never enter memory.
        for call in calls:
            graph.read(call["retrieval"]["ids"],row["domain"],episode,event_id=f"{row['id']}:{call['role']}",source_ids=provenance_ids(row))
        if condition.get("evolution","static") != "static":
            components = experience_components(backend,row,roles[-1],response)
            with torch.no_grad():
                tokens = memory.write(components)
            graph.add(tokens,provenance_ids(row),row["domain"],f"Question: {question_text(row)}\nGenerated answer: {response}",
                      episode=episode,role=roles[-1],components=components,task=row["dataset"])
            full = condition.get("evolution") == "full"
            maintenance = graph.maintain(episode,merge=full,promote=full and condition.get("capsules",True),prune=full)
    backend.synchronize()
    totals = {key:sum(c["cost"].get(key,0) for c in calls) for key in
              ["prompt_tokens","output_tokens","latent_positions","probe_tokens","probe_seconds","generation_seconds",
               "memory_seconds","communication_tokens","text_memory_tokens","structural_cost",
               "encoding_seconds","routing_seconds","synthesis_seconds"]}
    totals.update(prediction_seconds=prediction_seconds,update_seconds=time.perf_counter()-update_start)
    totals["all_text_tokens"] = totals["prompt_tokens"]+totals["output_tokens"]+totals["probe_tokens"]
    score["extracted_answer"] = score.pop("prediction")
    return dict(id=row["id"],dataset=row["dataset"],domain=row["domain"],metric=row["metric"],
                prediction=response,prediction_sha256=digest(response),**score,
                calls=calls,cost=totals,graph=graph.stats(),maintenance=maintenance,
                failure_category=None,episode=episode)


def evaluate(config_path, manifest_path, checkpoint, suite, output, seed, selected=None):
    config,manifest = validate_config(read_json(config_path)),validate_sources(validate_manifest(read_json(manifest_path)))
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new output directory; never mix/reuse partial runs")
    conditions = config["suites"][suite]
    if selected:
        conditions = [c for c in conditions if c["name"] in selected]
        if {c["name"] for c in conditions} != set(selected):
            raise ValueError("Unknown selected condition")
    if any(c.get("family","latent") == "latent" and c.get("budget_policy") == "entropy" for c in conditions):
        calibration = config.get("entropy_calibration",{})
        dev_ids = {r["id"] for r in manifest["samples"] if r["split"] == "dev"}
        if calibration.get("manifest_sha256") != manifest["sha256"] or not set(calibration.get("dev_ids",[])).issubset(dev_ids) or not calibration.get("dev_ids"):
            raise ValueError("Run dev-only calibrate first; placeholder entropy thresholds cannot be evaluated")
    from KGLMF.src.backends.huggingface import HFBackend
    seed_all(seed)
    backend = HFBackend(config["backend"])
    memory,initial,state = load_checkpoint(checkpoint,config,manifest,backend)
    if state["seed"] != seed:
        raise ValueError("Evaluation seed must match independently trained checkpoint seed")
    if state["training_config"].get("freeze_adapter"):
        conditions = [dict(c,name=c["name"]+"_fixed_adapter",adapter="cma") for c in conditions]
    test_rows = [r for r in manifest["samples"] if r["split"] == "test"]
    domains = config.get("suite_domains",{}).get(suite)
    if domains:
        test_rows = [r for r in test_rows if r["domain"] in domains]
    out.mkdir(parents=True)
    run_meta = dict(version=VERSION,manifest_sha256=manifest["sha256"],config=config,config_sha256=digest(config),
                    suite=suite,seed=seed,backend=backend.resolved,checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_sha256=__import__("hashlib").sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    selected_epoch=state["selected_epoch"],platform=platform.platform(),status="running",
                    test_ids=[r["id"] for r in test_rows],conditions=conditions)
    write_json(out/"run.json",run_meta)
    write_json(out/"initial_graph.json",dict(stats=initial.stats(),nodes={nid:{k:n[k] for k in
        ("source_ids","domains","task","kind","role","accesses","reads","gains","origins")} for nid,n in initial.nodes.items()}))
    # Warmup is declared, identical across conditions, excluded from timing.
    backend.generate("Reply with the word ready.")
    with torch.no_grad():
        for condition in conditions:
            graph = initial.clone()
            if not condition.get("capsules",True):
                # Disable Capsule designation/promotion while retaining all sources.
                for node in graph.nodes.values():
                    if node["kind"] == "capsule":
                        node["kind"] = node["origins"][0]["kind"]
                graph.recompute_edges()
            rows = list(test_rows)
            probe_rows = []
            if suite == "stream":
                # Fixed unconsumed retention set; remaining episodes ordered by domain.
                random.Random(seed).shuffle(rows)
                count = max(1,len(rows)//5)
                probe_rows,rows = rows[:count],rows[count:]
                rows.sort(key=lambda r:r["domain"])
            path = out/(condition["name"]+".jsonl")
            with path.open("w",encoding="utf-8") as stream:
                def emit(record,phase):
                    record.update(condition=condition["name"],seed=seed,phase=phase,
                                  manifest_sha256=manifest["sha256"],version=VERSION)
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                    stream.flush()
                for index,row in enumerate(rows):
                    if suite == "stream" and index % config.get("probe_interval",20) == 0:
                        for probe in probe_rows:
                            emit(predict(backend,memory,graph.clone(),probe,condition,seed,index,False),"retention")
                    # Static tests start from identical graph state on every example.
                    current = graph if suite == "stream" else restrict_graph(graph,condition,row["domain"]).clone()
                    seed_all(seed+int(row["id"][:8],16))
                    record = predict(backend,memory,current,row,condition,seed,index+1,
                                     suite == "stream" and condition.get("evolution") != "static")
                    emit(record,"stream" if suite == "stream" else "test")
                if suite == "stream":
                    for probe in probe_rows:
                        emit(predict(backend,memory,graph.clone(),probe,condition,seed,len(rows),False),"retention")
                    torch.save(graph.state_dict(),out/(condition["name"]+".graph.pt"))
    run_meta["status"] = "complete"
    write_json(out/"run.json",run_meta)
    return str(out)
