"""Summarize a completed main experiment."""
import argparse
import json
from KGLMF.src.execution.benchmark import summarize

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.output), indent=2))
