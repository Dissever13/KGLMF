"""Portable entry point for one benchmark, condition and seed per output directory.

This wrapper uses the existing aligned engine and durable online runner. It does
not recreate historical mixed-stream experiments or supply benchmark data.
"""
import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

from KGLMF.src.data.protocol import digest, read_json
from KGLMF.src.data.streams import validate_manifest
from KGLMF.src.agents.online import atomic_json, records
from KGLMF.src.execution.stream import output_lock, run_stream
from KGLMF.src.agents.prompts import domain_prompts, role_templates
from KGLMF.src.agents.orchestration import ROLE_IDS

PACKAGE = Path(__file__).resolve().parents[1]


def source_signature():
    return {p.relative_to(PACKAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE.rglob('*.py'))}


def resolve(config_path, manifest_path, seed):
    config = copy.deepcopy(read_json(config_path))
    manifest = validate_manifest(read_json(manifest_path), verify_files=True)
    datasets = {r['dataset'] for r in manifest['samples']}
    if len(datasets) != 1 or len(manifest['streams']) != 1:
        raise ValueError('Supply a manifest containing exactly one benchmark')
    domain, ids = next(iter(manifest['streams'].items()))
    lookup = {r['id']: r for r in manifest['samples']}
    rows = [lookup[sid] for sid in ids]
    config.update(active_domain=domain, total_episodes=len(rows),
                  domain_prompts=domain_prompts(), role_templates=role_templates(),
                  per_node_feedback_probes=False)
    condition = config.pop('condition')
    condition['agents'] = len(ROLE_IDS[domain])
    from KGLMF.src.agents.aligned import REVISION
    identity = dict(method_revision=REVISION, condition=condition['name'],
                    seed=seed, stream_key=next(iter(datasets)),
                    reset_boundary='benchmark', manifest_sha256=manifest['sha256'],
                    config_sha256=digest(config), condition_sha256=digest(condition),
                    source_sha256=source_signature())
    return config, condition, rows, identity


def verify_existing(output, identity):
    output = Path(output)
    if read_json(output/'identity.json') != identity:
        raise ValueError('Changed source/config/manifest/seed; use a new run directory')
    for name, expected in identity['source_sha256'].items():
        actual = hashlib.sha256((output/'source_snapshot'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Source snapshot mismatch: '+name)


def summarize(output):
    output = Path(output)
    identity = read_json(output/'identity.json')
    rows = records(output/'results.jsonl')
    status = read_json(output/'status.json')
    if status['status'] != 'complete' or len(rows) != status['total']:
        raise ValueError('Cannot summarize an incomplete benchmark')
    costs = [r['cost'] for r in rows]
    result = dict(dataset=identity['stream_key'], condition=identity['condition'],
                  seed=identity['seed'], n=len(rows),
                  score_percent=100*sum(r['score'] for r in rows)/len(rows),
                  correct=sum(r['score'] == 1 for r in rows),
                  inference_text_tokens=sum(c['all_text_tokens'] for c in costs),
                  feedback_text_tokens=sum(c['total_text_tokens_including_adaptation']-
                                           c['all_text_tokens'] for c in costs),
                  total_latency_seconds=sum(c['prediction_seconds']+c['post_answer_seconds']
                                            for c in costs),
                  uncertainty='single seed; no SD or confidence interval')
    atomic_json(output/'summary.json', result)
    return result


def run(config_path, manifest_path, seed, output, resume=False):
    from KGLMF.src.backends.aligned import AlignedBackend
    from KGLMF.src.agents.aligned import AlignedEngine
    config, condition, rows, identity = resolve(config_path, manifest_path, seed)
    output = Path(output)
    with output_lock(output):
        output.mkdir(parents=True, exist_ok=True)
        if (output/'identity.json').exists():
            if not resume:
                raise ValueError('Existing run requires --resume')
            verify_existing(output, identity)
            if read_json(output/'status.json')['status'] == 'complete':
                return summarize(output)
        else:
            if resume:
                raise ValueError('No existing run to resume')
            if any(p.name != '.run.lock' for p in output.iterdir()):
                raise ValueError('Use an empty output directory')
            snapshot = output/'source_snapshot'
            snapshot.mkdir()
            for name in identity['source_sha256']:
                (snapshot/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PACKAGE/name, snapshot/name)
            atomic_json(output/'identity.json', identity)
            atomic_json(output/'resolved_config.json', config)
            atomic_json(output/'manifest.json', read_json(manifest_path))
        status = dict(status='loading_model', total=len(rows), updated_at_unix=time.time())
        atomic_json(output/'status.json', status)
        try:
            backend = AlignedBackend(config['backend'])
            resolved = digest(backend.resolved)
            binding = output/'backend.json'
            if binding.exists() and read_json(binding)['sha256'] != resolved:
                raise ValueError('Resolved backbone or encoder changed')
            atomic_json(binding, dict(sha256=resolved, resolved=backend.resolved))
            identity = dict(identity, backend_sha256=resolved)
            engine = AlignedEngine(backend, config, condition, seed)
            def observe(current, record):
                status.update(status='running', completed=current.episode,
                              updated_at_unix=time.time())
                atomic_json(output/'status.json', status)
                print(json.dumps(dict(episode=current.episode, total=len(rows),
                                      score=record['score'])), flush=True)
            run_stream(engine, rows, output, identity, observer=observe)
            status.update(status='complete', updated_at_unix=time.time())
            atomic_json(output/'status.json', status)
        except BaseException as exc:
            status.update(status='interrupted', error_type=type(exc).__name__, error=str(exc))
            atomic_json(output/'status.json', status)
            raise
    return summarize(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'run', 'summarize'])
    p.add_argument('--config')
    p.add_argument('--manifest')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.action == 'summarize':
        if not args.output: p.error('--output is required')
        result = summarize(args.output)
    else:
        if not args.config or not args.manifest: p.error('--config and --manifest are required')
        if args.action == 'check':
            config, condition, rows, identity = resolve(args.config, args.manifest, args.seed)
            result = dict(status='validated_without_loading_model', n=len(rows), identity=identity)
        else:
            if not args.output: p.error('--output is required')
            result = run(args.config, args.manifest, args.seed, args.output, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
