"""Read-only source audit. No model loading."""
from pathlib import Path
from .protocol import DATASETS, read_json, choice_gold

if __name__ == "__main__":
    for name in ("gpqa","mmlu_pro"):
        failures = []
        for i,row in enumerate(read_json(Path("KGLMF/experiments/data")/DATASETS[name][0])):
            try:
                choice_gold(row["question"],[str(row["answer"])],row)
            except ValueError:
                failures.append((i,row))
        print(name,"ambiguous choices",len(failures),"examples",failures[:2])
