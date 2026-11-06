"""Adapters for post-generation scoring. Nothing executes on import.

math: runs Math-Verify only when explicitly invoked.
bigcode-export: writes samples for the official isolated evaluator.
bigcode-import: verifies its exact evaluated solutions and imports verdicts.
"""
import argparse
import json
import re
from pathlib import Path
from importlib.metadata import version
from KGLMF.src.data.protocol import read_json, write_json, digest


def write_lines(path, rows):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")


def score_math(export_path, output):
    from math_verify import parse,verify
    records=[]
    for r in read_json(export_path):
        if r["metric"] != "external_math":
            continue
        gold=r["answers"][0]
        parsed_gold=parse(gold if "$" in gold else "$"+gold+"$")
        if not parsed_gold:
            raise ValueError(f"Unparseable reference {r['id']}; repair source or evaluator configuration")
        parsed_answer=parse(r["prediction"])
        records.append(dict(id=r["id"],prediction_sha256=r["prediction_sha256"],
                            score=int(bool(parsed_answer) and verify(parsed_gold,parsed_answer)),
                            harness="huggingface/Math-Verify",revision=version("math-verify")))
    write_lines(output,records)


def bigcode_export(export_path, output):
    records,mapping=[],[]
    seen=set()
    for r in read_json(export_path):
        if r["dataset"] != "bigcodebench":
            continue
        raw=r["raw"]
        task=raw.get("task_id")
        if not task or not raw.get("test") or not raw.get("entry_point"):
            raise ValueError("Original BigCodeBench task ID/tests/entry_point required")
        if task in seen:
            raise ValueError("Export one condition/phase/episode at a time: repeated task ID")
        seen.add(task)
        blocks=re.findall(r"```(?:python|py)?\s*\n(.*?)```",r["prediction"],flags=re.S)
        solution=blocks[-1].strip() if blocks else re.sub(r"^FINAL:\s*","",r["prediction"].strip(),flags=re.I)
        records.append(dict(task_id=task,solution=solution))
        mapping.append(dict(task_id=task,id=r["id"],prediction_sha256=r["prediction_sha256"],solution_sha256=digest(solution)))
    write_lines(output,records)
    write_json(str(output)+".mapping.json",mapping)
    Path(str(output)+".task_ids.txt").write_text(",".join(r["task_id"] for r in records),encoding="utf-8")


def bigcode_import(results_path, mapping_path, output, revision):
    results=read_json(results_path)["eval"]
    mapping=read_json(mapping_path)
    records=[]
    if set(results) != {r["task_id"] for r in mapping}:
        raise ValueError("Official result task set differs from the exported subset")
    for r in mapping:
        outputs=results[r["task_id"]]
        if len(outputs)!=1:
            raise ValueError("Expected exactly one generation per task for pass@1")
        result=outputs[0]
        if digest(result["solution"]) != r["solution_sha256"]:
            raise ValueError("Evaluated solution differs; use --calibrated False and the exact exported samples")
        if result["status"] not in {"pass","fail","failed","timeout"}:
            raise ValueError(f"Unrecognized harness status: {result['status']}")
        records.append(dict(id=r["id"],prediction_sha256=r["prediction_sha256"],
                            score=int(result["status"]=="pass"),harness="bigcode-project/bigcodebench",revision=revision))
    write_lines(output,records)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("math","bigcode-export"):
        p=sub.add_parser(name)
        p.add_argument("--input",required=True)
        p.add_argument("--output",required=True)
    p=sub.add_parser("bigcode-import")
    for name in ("results","mapping","output","revision"):
        p.add_argument("--"+name,required=True)
    a=parser.parse_args()
    if a.command=="math":
        score_math(a.input,a.output)
    elif a.command=="bigcode-export":
        bigcode_export(a.input,a.output)
    else:
        bigcode_import(a.results,a.mapping,a.output,a.revision)
