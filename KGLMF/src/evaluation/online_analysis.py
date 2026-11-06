"""Read-only chronological performance, budget and maintenance curves."""
from collections import defaultdict

from KGLMF.src.data.protocol import bootstrap_mean, write_json
from KGLMF.src.evaluation.reporting import read_records


def curves(paths, output, window=50):
    if window < 1:
        raise ValueError('Positive episode window required')
    rows, metadata = read_records(paths)
    if any(m['version'] != 'kbs-online-v3' for m in metadata):
        raise ValueError('Chronological curves require online-v3 runs')
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row['model'], row['suite'], row['condition'], row['stream_key'])
        groups[key][row['seed']].append(row)
    output_rows = []
    for key, by_seed in sorted(groups.items()):
        model, suite, condition, stream = key
        by_seed = {s: sorted(v, key=lambda r: r['episode']) for s, v in by_seed.items()}
        orders = [[r['id'] for r in v] for v in by_seed.values()]
        if any(ids != orders[0] for ids in orders):
            raise ValueError('Seed curves require the same chronological problem order')
        if any([r['episode'] for r in v] != list(range(1, len(v) + 1)) for v in by_seed.values()):
            raise ValueError('An episode was omitted or repeated')
        for start in range(0, len(orders[0]), window):
            values = defaultdict(list)
            end = min(start + window, len(orders[0]))
            for seed, run in sorted(by_seed.items()):
                block, prefix, last = run[start:end], run[:end], run[end - 1]
                values['window_performance'].append(100 * sum(r['score'] for r in block) / len(block))
                values['cumulative_performance'].append(100 * sum(r['score'] for r in prefix) / len(prefix))
                values['nodes'].append(last['graph_after']['nodes'])
                values['edges'].append(last['graph_after']['edges'])
                for kind in ('role', 'context', 'trajectory', 'capsule'):
                    values[kind + '_nodes'].append(last['graph_after']['types'].get(kind, 0))
                for event in ('promoted', 'merged', 'pruned'):
                    values[event].append(sum(r['maintenance'].get(event, 0) for r in block))
                calls = [c for r in block for c in r['calls']]
                values['mean_allocated_budget'].append(sum(c.get('allocated_budget', 0) for c in calls) / max(1, len(calls)))
                values['mean_used_budget'].append(sum(c.get('retrieval', {}).get('budget', 0) for c in calls) / max(1, len(calls)))
                values['likelihood_updates'].append(sum(u['likelihood_update'] for r in block for u in r['updates']))
                values['router_updates'].append(sum(u['router_update'] for r in block for u in r['updates']))
                for cost in ('all_text_tokens', 'total_text_tokens_including_adaptation', 'prediction_seconds', 'post_answer_seconds'):
                    values[cost].append(sum(r['cost'][cost] for r in block) / len(block))
            output_rows.append(dict(model=model, suite=suite, condition=condition, stream_key=stream,
                                    episode_start=start + 1, episode_end=end, examples_per_seed=end - start,
                                    statistics={name: bootstrap_mean(v) for name, v in values.items()}))
    result = dict(version='kbs-online-v3', window=window, curves=output_rows,
                  interpretation='Window and cumulative first-answer scores; seed-level uncertainty. No retention or causal graph claim.')
    write_json(output, result)
    return result
