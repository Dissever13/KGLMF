"""Explicit online-v3 commands. prepare/check never load language models."""
import argparse
import json

from KGLMF.src.data.streams import VERSION, prepare_stream, validate_manifest, validate_config
from KGLMF.src.data.protocol import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description='KG-LMF chronological online evaluation; no offline pretraining stage')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--seed', type=int, default=42, help='Manifest order seed, shared across conditions')
    p.add_argument('--limit', type=int, help='Maximum retained problems per benchmark; subset results must be labelled')
    p = sub.add_parser('check')
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output')
    p = sub.add_parser('run')
    for name in ('config', 'manifest', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--suite', choices=['main', 'ablation', 'budget', 'sensitivity', 'transfer', 'stream'], required=True)
    p.add_argument('--conditions', nargs='+')
    p.add_argument('--resume', action='store_true')
    for command in ('summarize', 'failures', 'curves'):
        p = sub.add_parser(command)
        p.add_argument('--runs', nargs='+', required=True)
        p.add_argument('--output', required=True)
        if command == 'summarize':
            p.add_argument('--reference', default='kg_multi')
        if command == 'curves':
            p.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_stream(args.data_dir, args.output, args.datasets, args.seed, args.limit)
        print(json.dumps(dict(version=VERSION, sha256=manifest['sha256'], counts=manifest['counts'],
                              streams={k: len(v) for k, v in manifest['streams'].items()}), indent=2))
    elif args.command == 'check':
        config = validate_config(read_json(args.config))
        manifest = validate_manifest(read_json(args.manifest), verify_files=True)
        result = dict(status='configuration_checked_no_models_loaded', version=VERSION,
                      counts=manifest['counts'], streams={k: len(v) for k, v in manifest['streams'].items()},
                      reset_boundary=manifest['reset_boundary'], feedback=config['feedback'],
                      suites={k: [c['name'] for c in v] for k, v in config['suites'].items()},
                      requirements=['Set exact local model paths/revisions', 'Provide versioned source fields',
                                    'Configure synchronous isolated code/math scorer before run'])
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == 'run':
        from KGLMF.src.execution.stream import run
        print(run(args.config, args.manifest, args.suite, args.output, args.seed, args.conditions, args.resume))
    else:
        from KGLMF.src.evaluation.reporting import aggregate, failure_sheet
        if args.command == 'summarize':
            aggregate(args.runs, args.output, args.reference)
        elif args.command == 'failures':
            failure_sheet(args.runs, args.output)
        else:
            from KGLMF.src.evaluation.online_analysis import curves
            curves(args.runs, args.output, args.window)


if __name__ == '__main__':
    main()
