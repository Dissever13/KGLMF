"""Causal problem streams; no offline split, model import, or automatic run."""
import copy
import hashlib
from collections import Counter
from pathlib import Path

from .protocol import (DATASETS, digest, read_json, write_json, first_present,
                       question_key, input_key, grouping_aliases, choice_gold,
                       validate_sources, validate_config as validate_legacy_config)

VERSION = 'kbs-online-v3'
STREAMS = {
    'reasoning': ('strategyqa', 'hotpotqa'),
    'qa': ('triviaqa', 'popqa'),
    'code': ('kodcode', 'bigcodebench'),
    'science_math': ('math500', 'mmlu_pro', 'gpqa'),
}
STREAM_FOR = {name: group for group, names in STREAMS.items() for name in names}


def prepare_stream(data_dir, output, datasets=None, seed=42, limit=None):
    if Path(output).exists():
        raise FileExistsError('Manifest exists; use a new output')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    names = sorted(datasets or DATASETS)
    if not names or set(names) - DATASETS.keys():
        raise ValueError('Unknown/empty benchmark selection')
    rows, files = [], {}
    for name in names:
        filename, domain, metric = DATASETS[name]
        path = Path(data_dir) / filename
        raw = read_json(path)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f'{path}: nonempty JSON array required')
        files[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for index, item in enumerate(raw):
            q = str(first_present(item, ['question', 'prompt', 'input']))
            if not question_key(q):
                raise ValueError(f'{name}:{index}: empty question')
            if domain == 'code':
                aliases = []  # No reference implementation is needed by the learner.
            else:
                answer = first_present(item, ['answer', 'target', 'output', 'answer_key'])
                if isinstance(answer, dict):
                    answer = first_present(answer, ['aliases', 'value', 'text'])
                aliases = answer if isinstance(answer, list) else [answer]
                aliases = [str(a) for a in aliases]
                extra = item.get('aliases', [])
                aliases += [str(a) for a in extra] if isinstance(extra, list) else [str(extra)]
                if metric == 'choice':
                    aliases = choice_gold(q, aliases, item)
            key = input_key(q, item, domain)
            rows.append(dict(id=digest([name, key, item.get('id', item.get('task_id')), index])[:24],
                             dataset=name, domain=domain, stream_domain=STREAM_FOR[name], metric=metric,
                             question=q, answers=list(dict.fromkeys(aliases)), raw=item,
                             source_index=index, source_id=item.get('id', item.get('task_id')),
                             source_revision=item.get('source_revision')))
    # Union every known duplicate/derivative before selecting one representative.
    parent = {r['id']: r['id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    aliases = {}
    for r in rows:
        for alias in grouping_aliases(r['dataset'], r['raw'], r['question'], r['domain']):
            if alias in aliases:
                parent[find(r['id'])] = find(aliases[alias])
            else:
                aliases[alias] = r['id']
    groups = {}
    for r in rows:
        groups.setdefault(find(r['id']), []).append(r)
    selected, removed, capped, counts = [], [], [], Counter()
    for members in sorted(groups.values(), key=lambda g: digest([seed, sorted(r['id'] for r in g)])):
        members = sorted(members, key=lambda r: digest([seed, 'representative', r['id']]))
        chosen = members[0]
        gid = 'group:' + digest(sorted(r['id'] for r in members))[:24]
        chosen['group_id'] = gid
        removed.extend(dict(id=r['id'], dataset=r['dataset'], kept_id=chosen['id'], group_id=gid) for r in members[1:])
        if limit is not None and counts[chosen['dataset']] >= limit:
            capped.append(chosen['id'])
            continue
        counts[chosen['dataset']] += 1
        selected.append(chosen)
    order = {group: [r['id'] for r in sorted(selected, key=lambda r: digest([seed, 'order', r['id']]))
                     if r['stream_domain'] == group] for group in STREAMS}
    order = {k: v for k, v in order.items() if v}
    manifest = dict(version=VERSION, protocol='deduplicated_chronological_streams', order_seed=seed,
                    requested_datasets=names, limit_per_benchmark=limit, source_files=files,
                    samples=selected, streams=order, counts=dict(counts), duplicates_removed=removed,
                    capped_ids=capped, reset_boundary='task_category', feedback='internal_and_post_answer_benchmark')
    manifest['sha256'] = digest(manifest)
    validate_manifest(manifest)
    validate_sources(manifest)
    write_json(output, manifest)
    return manifest


def validate_manifest(m, verify_files=False):
    if m.get('version') != VERSION or m.get('reset_boundary') != 'task_category':
        raise ValueError('An online-v3 task-category stream manifest is required')
    if digest({k: v for k, v in m.items() if k != 'sha256'}) != m.get('sha256'):
        raise ValueError('Manifest hash mismatch')
    rows = m['samples']
    if not rows or len({r['id'] for r in rows}) != len(rows) or len({r['group_id'] for r in rows}) != len(rows):
        raise ValueError('Empty stream or repeated example/derivative group')
    lookup = {r['id']: r for r in rows}
    ordered = []
    for group, ids in m['streams'].items():
        if group not in STREAMS or not ids:
            raise ValueError('Invalid task-category stream')
        for sid in ids:
            if sid not in lookup or lookup[sid]['dataset'] not in STREAMS[group] or lookup[sid]['stream_domain'] != group:
                raise ValueError('Stream membership mismatch')
        ordered += ids
    if Counter(ordered) != Counter(lookup.keys()):
        raise ValueError('Every problem must occur exactly once in its stream')
    if any(r['metric'] != DATASETS[r['dataset']][2] or r['domain'] != DATASETS[r['dataset']][1]
           or (r['domain'] != 'code' and not r['answers']) for r in rows):
        raise ValueError('Invalid metric/source answers')
    validate_sources(m)
    if verify_files:
        for record in m['source_files'].values():
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Source file changed after manifest preparation')
    return m


def prediction_view(row):
    """Physically exclude reference answers, implementations, tests and scores."""
    view = {k: row[k] for k in ('id', 'dataset', 'domain', 'stream_domain', 'metric', 'question', 'group_id')}
    view['raw'] = {k: copy.deepcopy(row['raw'][k]) for k in ('choices', 'options', 'task_id') if k in row['raw']}
    return view


def validate_config(config):
    if config.get('protocol') != VERSION:
        raise ValueError('Use kbs-online-v3 config; offline checkpoints/configs are a different protocol')
    check = copy.deepcopy(config)
    check['protocol'] = 'kbs-v2'
    validate_legacy_config(check)
    if config['controller'].get('rule') != 'completed_history_max_entropy':
        raise ValueError('Online entropy must use completed history only')
    if config.get('feedback') != 'internal_and_post_answer_benchmark':
        raise ValueError('All benchmarks must use the declared two feedback sources')
    if config['training']['epochs'] != 1 or config['training']['batch_size'] != 1:
        raise ValueError('Online scoring visits each question once; one update per eligible request')
    return config


def example_config(model):
    from .create_configs import example_config as old_config
    c = old_config(model)
    c.update(protocol=VERSION, feedback='internal_and_post_answer_benchmark')
    c['controller']['rule'] = 'completed_history_max_entropy'
    c['training']['epochs'] = 1
    c['memory'].pop('initial_examples', None)
    c.pop('suite_domains', None)
    c.pop('probe_interval', None)
    c['suites']['stream'] = [dict(c['suites']['main'][-1], name='evolve', evolution='full'),
                            dict(c['suites']['main'][-1], name='append', evolution='append', capsules=False),
                            dict(c['suites']['main'][-1], name='no_write', evolution='static')]
    c['suites']['ablation'].append(dict(c['suites']['main'][-1], name='fixed_adapter', adapter='cma'))
    for name, changes in [('flat_pool', dict(storage='flat', topology='flat', capsules=False)),
                          ('no_role_retrieval', dict(router_role_condition=False)),
                          ('no_role_composition', dict(composition_role_condition=False)),
                          ('no_merge', dict(merge=False)), ('no_prune', dict(prune=False))]:
        c['suites']['ablation'].append(dict(c['suites']['main'][-1], name=name, **changes))
    c['suites']['sensitivity'] = [dict(c['suites']['main'][-1], name=f'k_{k}', k=k, budget_policy='fixed', budget=16) for k in (1, 2, 4, 8)]
    c['suites']['sensitivity'] += [dict(c['suites']['main'][-1], name=f'b_{b}', k=4, budget_policy='fixed', budget=b) for b in (4, 8, 16, 32)]
    c['suites']['transfer'] = [dict(c['suites']['main'][-1], name=name, scope=scope, experience_cap=64)
                             for name, scope in [('source_only', 'source'), ('target_only', 'target'), ('full', 'full')]]
    c['transfer_directions'] = {'non_science_to_science': ['mmlu_pro', 'gpqa'],
                                'non_reasoning_to_reasoning': ['strategyqa', 'hotpotqa']}
    for suite in c['suites'].values():
        for condition in suite:
            if condition['family'] == 'text':
                condition.update(evolution='append', capsules=False)
    return c
