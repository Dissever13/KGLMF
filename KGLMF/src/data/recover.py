"""Normalize ORIGINAL official JSON exports without evaluating any model.

Examples (download/export source data separately with its license):
python -m KGLMF.src.data.recover --dataset gpqa --input original.json --output gpqa_real.json --revision COMMIT

This never attempts to reconstruct missing answer options/tests from legacy Q/A.
The manifest generator intentionally labels the later resplit a LOCAL protocol.
"""
import argparse
import json
import random
from KGLMF.src.data.protocol import digest, read_json, write_json


def convert(dataset, rows, revision):
    output = []
    for index,row in enumerate(rows):
        if dataset in {"strategyqa","triviaqa","hotpotqa"}:
            result = dict(row,question=row["question"],answer=row["answer"])
            if "_id" in row:
                result["id"] = row["_id"]
            if dataset == "triviaqa" and isinstance(result["answer"],dict):
                result["aliases"] = result["answer"].get("aliases",[])+result["answer"].get("normalized_aliases",[])
        elif dataset == "popqa":
            answers = row.get("possible_answers",row.get("answer"))
            if isinstance(answers,str) and answers.startswith("["):
                answers = json.loads(answers)
            result = dict(row,question=row["question"],answer=answers)
        elif dataset == "math500":
            result = dict(row,question=row["problem"],answer=row["answer"])
        elif dataset == "gpqa":
            q = row["Question"]
            correct = row["Correct Answer"]
            choices = [correct]+[row[f"Incorrect Answer {i}"] for i in (1,2,3)]
            if len(set(choices)) != 4:
                raise ValueError("Duplicate GPQA options in original source")
            random.Random(digest(q)).shuffle(choices)
            result = dict(question=q,choices=choices,answer=chr(65+choices.index(correct)))
        elif dataset == "mmlu_pro":
            result = dict(question=row["question"],choices=row["options"],answer=row["answer"],category=row.get("category"))
        elif dataset == "bigcodebench":
            result = dict(question=row["instruct_prompt"],answer=row["canonical_solution"],
                          task_id=row["task_id"],test=row["test"],entry_point=row["entry_point"],
                          complete_prompt=row["complete_prompt"],code_prompt=row["code_prompt"])
        elif dataset == "kodcode":
            # Keep original schema explicit; do not infer that arbitrary Q/A is KodCode.
            result = dict(question=row["question"],answer=row["solution"],
                          task_id=str(row["question_id"]),test=row["test"])
        else:
            raise ValueError("Unknown dataset schema: "+dataset)
        # Preserve native identifiers, metadata and derivative relationships.
        result = dict(row,**result)
        result.setdefault("id",row.get("question_id",row.get("_id",result.get("task_id",f"{dataset}:{index}"))))
        result.update(source_revision=revision,source_index=index,original=row)
        output.append(result)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    from KGLMF.src.data.protocol import DATASETS
    p.add_argument("--dataset",choices=list(DATASETS),required=True)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--revision",required=True)
    args = p.parse_args()
    from pathlib import Path
    if Path(args.output).exists():
        p.error("Refusing overwrite; use a separate recovered-data directory")
    write_json(args.output,convert(args.dataset,read_json(args.input),args.revision))
