"""Write main-experiment settings for local model paths."""
import argparse
from copy import deepcopy
from KGLMF.runtime.default_config import DEFAULT_CONFIG
from KGLMF.runtime.protocol import write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--encoder', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    c = deepcopy(DEFAULT_CONFIG)
    c['backend'].update(model=args.model, encoder=args.encoder)
    write_json(args.output, c)

if __name__ == '__main__':
    main()
