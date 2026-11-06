"""Audit/import original-method online predictions; never run an approximation.

History declarations permit consistency checks, not proof of an external run's
causality. Keep its original answer/update journals for an independent audit.
"""
import math
import json
from pathlib import Path

from KGLMF.src.data.streams import VERSION, validate_manifest
from KGLMF.src.data.protocol import read_json, write_json, digest
from KGLMF.src.training.optimization import TaskScorer


REQUIRED_COST = {
    'prompt_tokens', 'output_tokens', 'probe_tokens', 'latent_positions',
    'communication_tokens', 'text_memory_tokens', 'all_text_tokens',
    'generation_seconds', 'probe_seconds', 'memory_seconds', 'prediction_seconds',
    'post_answer_seconds', 'total_text_tokens_including_adaptation',
    'training_write_maintenance_seconds',
}


def import_online_external(source, manifest_path, output, scoring_config=None, scorer=None):
    data = read_json(source)
    manifest = validate_manifest(read_json(manifest_path), verify_files=True)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if not data.get('implementation_url') or not data.get('implementation_revision'):
        raise ValueError('Original implementation URL and immutable revision required')
    if data.get('manifest_sha256') != manifest['sha256']:
        raise ValueError('External run used a different sample manifest/order')
    if (data.get('reset_boundary') != 'task_category'
            or data.get('feedback') != 'internal_and_post_answer_benchmark'
            or data.get('initial_benchmark_experience_ids') != []):
        raise ValueError('External run must declare cold start, category resets and matched feedback access')
    name = data['method']
    if not name.replace('_', '').isalnum():
        raise ValueError('Method must be a filename-safe identifier')
    if not isinstance(data.get('seed'), int) or not data.get('backend', {}).get('model'):
        raise ValueError('Run seed and resolved backend metadata required')
    lookup = {r['id']: r for r in manifest['samples']}
    incoming = data['records']
    if len(incoming) != len(lookup) or {r['id'] for r in incoming} != lookup.keys():
        raise ValueError('External predictions must cover every declared problem exactly once')
    scorer = scorer or TaskScorer(scoring_config or {})
    completed = {group: [] for group in manifest['streams']}
    records = []
    for r in incoming:
        row = lookup[r['id']]
        group = row['stream_domain']
        past = completed[group]
        if r['id'] != manifest['streams'][group][len(past)]:
            raise ValueError('External problem order differs from the declared category stream')
        if (r.get('answer_recorded_before_feedback') is not True
                or r.get('available_feedback_ids') != past
                or not set(r.get('retrieved_source_ids', [])).issubset(past)):
            raise ValueError('Current/future or cross-category experience access detected')
        if digest(r['prediction']) != r.get('prediction_sha256'):
            raise ValueError('Prediction hash mismatch')
        cost = r['cost']
        if not REQUIRED_COST.issubset(cost) or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in cost.values()):
            raise ValueError('Complete finite nonnegative cost accounting required; unknown is not zero')
        if not math.isclose(cost['all_text_tokens'], sum(cost[k] for k in ('prompt_tokens', 'output_tokens', 'probe_tokens'))):
            raise ValueError('First-answer token accounting inconsistent')
        if cost['total_text_tokens_including_adaptation'] < cost['all_text_tokens']:
            raise ValueError('Adaptation-inclusive tokens cannot omit first-answer tokens')
        if records and records[0]['cost'].keys() != cost.keys():
            raise ValueError('External records must use one consistent cost schema')
        verdict = scorer.score(r['prediction'], row, required=True)
        if verdict['score'] is None or not 0 <= float(verdict['score']) <= 1:
            raise ValueError('A valid task score is required')
        records.append(dict(r, verdict=verdict, score=float(verdict['score']), version=VERSION,
                            manifest_sha256=manifest['sha256'], dataset=row['dataset'], domain=row['domain'],
                            stream_domain=group, metric=row['metric'], condition=name, seed=data['seed'],
                            phase='online', episode=len(past) + 1, status='external_online_regraded',
                            calls=r.get('calls', [])))
        past.append(row['id'])
    out.mkdir(parents=True)
    (out / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write_json(out / 'run.json', dict(version=VERSION, status='complete', manifest_sha256=manifest['sha256'],
               suite='main', seed=data['seed'], backend=data['backend'], conditions=[{'name': name}],
               checkpoint_sha256=data['implementation_revision'], implementation_url=data['implementation_url'],
               external_information_access='declarations_checked_original_journals_required_for_audit'))
    return str(out)
