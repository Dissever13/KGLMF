"""Audit local exports before a costly run; no model loading or data edits."""
import argparse
from collections import Counter
from pathlib import Path
from KGLMF.src.data.protocol import DATASETS, read_json, write_json, choice_gold, question_key


def audit(data_dir):
    report = {}
    for name,(filename,domain,metric) in DATASETS.items():
        path = Path(data_dir)/filename
        rows = read_json(path)
        failures=[]
        for index,row in enumerate(rows):
            if metric == "choice":
                try:
                    choice_gold(row["question"],[str(row["answer"])],row)
                except ValueError:
                    failures.append(index)
        report[name] = dict(rows=len(rows),duplicate_questions=len(rows)-len({question_key(r["question"]) for r in rows}),
                            ambiguous_or_missing_choice_indices=failures,
                            code_records_missing_id_or_tests=sum(not (r.get("task_id") and r.get("test")) for r in rows) if domain == "code" else 0,
                            source_recovery_required=bool(failures) or domain == "code" and any(not (r.get("task_id") and r.get("test")) for r in rows))
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=audit(a.data_dir)
    write_json(a.output,result)
    for name,v in result.items():
        print(name,v["rows"],"choice_issues",len(v["ambiguous_or_missing_choice_indices"]),"code_missing",v["code_records_missing_id_or_tests"])
