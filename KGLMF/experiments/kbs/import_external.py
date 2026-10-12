"""Import records from a verified original external implementation.

Input JSON contains method, implementation_url, implementation_revision,
training_source_ids, backend (same resolved model fields), seed, and records.
Records contain id, prediction, cost. Cost must include every component in the
internal schema. No prompt-only approximation is assigned a paper's method name.
"""
import argparse
from pathlib import Path
import json
import math
from .protocol import VERSION, read_json, validate_manifest, digest, score_answer, write_json


def import_external(source, manifest_path, output):
    data=read_json(source)
    manifest=validate_manifest(read_json(manifest_path))
    if not data.get("implementation_url") or not data.get("implementation_revision"):
        raise ValueError("Original implementation URL and immutable revision required")
    train={r["id"] for r in manifest["samples"] if r["split"]=="train"}
    if set(data["training_source_ids"]) != train:
        raise ValueError("External method must receive exactly the same training information")
    test={r["id"]:r for r in manifest["samples"] if r["split"]=="test"}
    if len(data["records"]) != len(test) or {r["id"] for r in data["records"]} != test.keys():
        raise ValueError("External prediction IDs must match the manifest test set exactly")
    out=Path(output)
    if out.exists():
        raise FileExistsError(out)
    required_cost={"prompt_tokens","output_tokens","latent_positions","probe_tokens","probe_seconds","generation_seconds","prediction_seconds","update_seconds","all_text_tokens","memory_seconds","communication_tokens","text_memory_tokens","structural_cost","encoding_seconds","routing_seconds","synthesis_seconds"}
    records=[]
    for index,r in enumerate(data["records"]):
        if set(r["cost"]) != required_cost or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in r["cost"].values()):
            raise ValueError("External cost accounting incomplete; unknown is not zero")
        row=test[r["id"]]
        score=score_answer(r["prediction"],row)
        score["extracted_answer"]=score.pop("prediction")
        records.append(dict(r,**score,version=VERSION,manifest_sha256=manifest["sha256"],
                            dataset=row["dataset"],domain=row["domain"],metric=row["metric"],
                            condition=data["method"],seed=data["seed"],phase="test",episode=index+1,
                            prediction_sha256=digest(r["prediction"]),calls=r.get("calls",[])))
    name=data["method"]
    if not name.replace("_","").isalnum():
        raise ValueError("Method must be a filename-safe identifier")
    out.mkdir(parents=True)
    (out/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    write_json(out/"run.json",dict(version=VERSION,status="complete",manifest_sha256=manifest["sha256"],
        suite="main",seed=data["seed"],backend=data["backend"],conditions=[{"name":name}],
        checkpoint_sha256=data["implementation_revision"],implementation_url=data["implementation_url"],
        external_information_access_verified=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ["source","manifest","output"]:
        p.add_argument("--"+field,required=True)
    a=p.parse_args()
    import_external(a.source,a.manifest,a.output)
