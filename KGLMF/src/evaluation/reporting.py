"""Pairing-aware summaries and official-harness score import; no fabricated cells."""
from collections import defaultdict
import json
from pathlib import Path

from KGLMF.src.data.protocol import VERSION, bootstrap_mean, digest, read_json, write_json


def read_records(paths):
    records,seen,metadata,backends = [],set(),[],{}
    manifest, run_version = None, None
    for directory in paths:
        directory = Path(directory)
        run = read_json(directory/"run.json")
        if run["status"] != "complete":
            raise ValueError(f"Incomplete run: {directory}")
        if run["version"] not in {VERSION, "kbs-online-v3"}:
            raise ValueError("Legacy run format rejected")
        if run_version is not None and run_version != run["version"]:
            raise ValueError("Cannot pool offline and chronological online protocols")
        run_version = run["version"]
        if manifest is not None and manifest != run["manifest_sha256"]:
            raise ValueError("Cannot combine different split manifests")
        manifest = run["manifest_sha256"]
        model = run["backend"]["model"]
        signature = {k:run["backend"].get(k) for k in ["model_commit","encoder_commit","transformers_version","torch_version","dtype","batch_size",
                     "quantization","hardware","max_input_tokens","max_new_tokens","temperature","top_p","top_k","repetition_penalty","input_embedding_scale","truncation_side"]}
        if model in backends and backends[model] != signature:
            raise ValueError("Incompatible backend/hardware/decoding for pooled performance-efficiency comparison")
        backends[model] = signature
        metadata.append(run)
        for condition in run["conditions"]:
            original = directory/(condition["name"]+".jsonl")
            scored = directory/(condition["name"]+".scored.jsonl")
            path = scored if scored.exists() else original
            if scored.exists():
                raw_rows = [json.loads(x) for x in original.read_text(encoding="utf-8").splitlines()]
                new_rows = [json.loads(x) for x in scored.read_text(encoding="utf-8").splitlines()]
                if len(raw_rows) != len(new_rows) or any(
                    any(a[k] != b[k] for k in a if k not in {"score","status"})
                    for a,b in zip(raw_rows,new_rows)):
                    raise ValueError("Scored file changes original prediction/provenance/cost records")
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = (run["backend"]["model"],run["suite"],row["condition"],row["seed"],row["id"],row["phase"],row["episode"])
                if key in seen:
                    raise ValueError("Duplicate run/observation; do not include copies")
                seen.add(key)
                row["model"] = run["backend"]["model"]
                row["suite"] = run["suite"]
                records.append(row)
    return records,metadata


def aggregate(paths, output, reference="kg_multi"):
    records,metadata = read_records(paths)
    groups = defaultdict(list)
    for row in records:
        # Retention is a curve; never pool repeated probes into pseudo-replications.
        point = row["episode"] if row["phase"] == "retention" else None
        groups[(row["model"],row["suite"],row["dataset"],row["condition"],row["phase"],point)].append(row)
    summaries,pairs = [],[]
    for key,rows in sorted(groups.items(),key=lambda item:str(item[0])):
        model,suite,dataset,condition,phase,episode = key
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        complete = all(r["score"] is not None for r in rows)
        result = dict(model=model,suite=suite,dataset=dataset,condition=condition,phase=phase,
                      episode=episode,n_seeds=len(by_seed),n_examples_by_seed={s:len(v) for s,v in by_seed.items()},
                      pending_scores=sum(r["score"] is None for r in rows))
        if any(r['cost'].keys() != rows[0]['cost'].keys() for r in rows):
            raise ValueError('Cost schema changed within a condition; do not silently omit components')
        if complete:
            result["performance"] = bootstrap_mean([100*sum(r["score"] for r in v)/len(v) for s,v in sorted(by_seed.items())])
        result["cost"] = {cost:bootstrap_mean([sum(r["cost"][cost] for r in v)/len(v) for s,v in sorted(by_seed.items())])
                          for cost in rows[0]["cost"]}
        if len(by_seed)<3:
            result["warning"] = "Fewer than three independent training seeds; no reliable seed uncertainty estimate"
        summaries.append(result)
        reference_rows = groups.get((model,suite,dataset,reference,phase,episode))
        if condition == reference or not reference_rows or not complete or any(r["score"] is None for r in reference_rows):
            continue
        lhs = {(r["seed"],r["id"]):r for r in rows}
        rhs = {(r["seed"],r["id"]):r for r in reference_rows}
        if lhs.keys() != rhs.keys():
            raise ValueError(f"Unpaired examples/seeds: {condition} vs {reference}")
        differences = defaultdict(list)
        for (seed,sid),r in lhs.items():
            differences[seed].append(100*(rhs[(seed,sid)]["score"]-r["score"]))
        pairs.append(dict(model=model,suite=suite,dataset=dataset,phase=phase,episode=episode,
                          contrast=f"{reference} minus {condition}",
                          paired_seed_difference=bootstrap_mean([sum(v)/len(v) for s,v in sorted(differences.items())])))
    report = dict(version=metadata[0]["version"],summaries=summaries,paired_comparisons=pairs,
                  uncertainty="Sample SD across independently trained seeds; percentile bootstrap of seed means/differences. Not an example-level CI or a significance claim.",
                  runs=[{k:r[k] for k in ["seed","suite","manifest_sha256","checkpoint_sha256"]} for r in metadata])
    write_json(output,report)
    return report


def import_scores(predictions, scores, output):
    """External harness JSONL: id, prediction_sha256, score, harness, revision.

    Executing generated programs belongs to a sandboxed official harness, not
    this process. Exact output hashes bind a verdict to the actual generation.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    rows = [json.loads(x) for x in Path(predictions).read_text(encoding="utf-8").splitlines()]
    score_paths = [scores] if isinstance(scores,(str,Path)) else scores
    verdicts = [json.loads(x) for path in score_paths for x in Path(path).read_text(encoding="utf-8").splitlines()]
    lookup = {}
    for v in verdicts:
        key = (v["id"],v["prediction_sha256"])
        if key in lookup or v.get("score") not in (0,1) or not v.get("harness") or not v.get("revision"):
            raise ValueError("Invalid/duplicate official harness score")
        lookup[key] = v
    used = set()
    for row in rows:
        if row["score"] is not None:
            continue
        key = (row["id"],digest(row["prediction"]))
        if key not in lookup:
            raise ValueError(f"Missing verdict for {row['id']} / generation hash")
        verdict = lookup[key]
        row.update(score=float(verdict["score"]),status="official_harness_scored",harness=verdict["harness"],harness_revision=verdict["revision"])
        used.add(key)
    if used != lookup.keys():
        raise ValueError("Unused verdicts; check dataset/condition/seed")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")


def export_harness(predictions, manifest, output):
    from KGLMF.src.data.protocol import validate_manifest
    sources = {r["id"]:r for r in validate_manifest(read_json(manifest))["samples"]}
    rows = [json.loads(x) for x in Path(predictions).read_text(encoding="utf-8").splitlines()]
    pending = []
    for row in rows:
        if row["score"] is None:
            source = sources[row["id"]]
            pending.append({k:row[k] for k in ["id","dataset","metric","prediction","prediction_sha256"]} |
                           {"source_id":source["source_id"],"raw":source["raw"],"answers":source["answers"]})
    write_json(output,pending)


def failure_sheet(paths, output):
    rows,_ = read_records(paths)
    # The labels are for human review. Never infer a causal failure from a score alone.
    failed = [{k:r[k] for k in ["id","dataset","condition","seed","prediction","score","calls","cost","phase","episode"]}
              | {"failure_category":None,"reviewer_notes":"","allowed_categories":["harmful_retrieval","missing_memory","role_routing","synthesis","budget","base_reasoning","other"]}
              for r in rows if r["score"] == 0]
    write_json(output,failed)
