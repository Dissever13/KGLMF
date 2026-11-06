"""Freeze one benchmark's deduplicated input sequence."""
import argparse
from KGLMF.runtime.online_protocol import prepare_stream
from KGLMF.runtime.protocol import DATASETS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    prepare_stream(args.data_dir, args.output, [args.dataset], args.seed, args.limit)

if __name__ == '__main__':
    main()
