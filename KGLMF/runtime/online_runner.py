"""Durable answer journals and atomic post-episode online checkpoints."""
import copy
import hashlib
import time
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from .online import OnlineEngine, atomic_json, atomic_checkpoint, append_record, records
from .online_protocol import VERSION, validate_manifest, validate_config, prediction_view
from .protocol import digest, read_json
from .memory import MemoryGraph


def run_stream(engine, rows, directory, identity, result_path=None, observer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'state.pt'
    answers_path, feedback_path = directory / 'answers.jsonl', directory / 'feedback.jsonl'
    result_path = Path(result_path or directory / 'results.jsonl')
    complete = {r['id']: r for r in records(result_path) if r.get('stream_key') == identity['stream_key']}
    state = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('version') != VERSION or state.get('identity') != identity:
            raise ValueError('Checkpoint does not match this online stream/config/condition')
        engine.load_state(state['engine'])
        if engine.completed_ids != [r['id'] for r in rows[:engine.episode]]:
            raise ValueError('Checkpoint is not a prefix of the declared problem order')
        if state.get('last_record') and state['last_record']['id'] not in complete:
            append_record(result_path, state['last_record'])
            complete[state['last_record']['id']] = state['last_record']
    else:
        if records(answers_path) or records(feedback_path) or complete:
            raise ValueError('Run journals without their causal initial checkpoint')
        state = dict(version=VERSION, identity=identity, engine=engine.state_dict(), last_record=None)
        atomic_checkpoint(checkpoint, state)
    answers = {r['id']: r for r in records(answers_path)}
    feedback = {r['id']: r for r in records(feedback_path)}
    if observer is not None and state.get('last_record'):
        observer(engine, state['last_record'])
    for row in rows[engine.episode:]:
        if row['id'] in engine.completed_ids:
            raise ValueError('Cannot score an already completed problem again')
        # Checkpoint hash binds a prediction to the exact pre-update state.
        before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        prediction = answers.get(row['id'])
        if prediction is None:
            prediction = engine.predict(prediction_view(row))
            prediction.update(pre_update_checkpoint_sha256=before_hash, identity_sha256=digest(identity))
            append_record(answers_path, prediction)  # fsync BEFORE oracle/feedback access
            answers[row['id']] = prediction
        elif prediction.get('pre_update_checkpoint_sha256') != before_hash or prediction.get('identity_sha256') != digest(identity):
            raise ValueError('Recorded answer is not bound to this pre-update checkpoint')
        released = feedback.get(row['id'])
        if released is None:
            scored_start = time.perf_counter()
            verdict = engine.scorer.score(prediction['prediction'], row, required=True)
            if verdict['score'] is None or not 0 <= float(verdict['score']) <= 1:
                raise ValueError('A valid task score is required before advancing the online stream')
            scored_prediction = dict(prediction, verdict=verdict, score=float(verdict['score']))
            score_seconds = time.perf_counter() - scored_start
            batch = engine.collect_feedback(row, scored_prediction)
            released = dict(id=row['id'], prediction_sha256=prediction['prediction_sha256'],
                            pre_update_checkpoint_sha256=before_hash, verdict=verdict, feedback=batch,
                            scoring_seconds=score_seconds)
            append_record(feedback_path, released)
            feedback[row['id']] = released
        if released['prediction_sha256'] != prediction['prediction_sha256'] or released['pre_update_checkpoint_sha256'] != before_hash:
            raise ValueError('Feedback is not bound to the recorded prediction/state')
        # Replaying a failed adaptation starts from the durable pre-update state;
        # gradients/writes can commit only with the single atomic checkpoint.
        engine.load_state(state['engine'])
        scored_prediction = dict(prediction, verdict=released['verdict'], score=float(released['verdict']['score']))
        adapted = engine.adapt(row, scored_prediction, released['feedback'])
        result = copy.deepcopy(scored_prediction)
        result.update(adapted, version=VERSION, condition=identity['condition'], seed=identity['seed'],
                      manifest_sha256=identity['manifest_sha256'], stream_key=identity['stream_key'],
                      status='completed_post_answer_update', feedback_source=released['feedback']['source'])
        result['cost']['scoring_seconds'] = released['scoring_seconds']
        result['cost']['training_write_maintenance_seconds'] = adapted['adaptation_seconds']
        for name, value in released['feedback']['cost'].items():
            result['cost']['feedback_' + name] = value
        result['cost']['post_answer_seconds'] = (released['scoring_seconds'] + released['feedback']['cost']['feedback_seconds'] + adapted['adaptation_seconds'])
        result['cost']['total_text_tokens_including_adaptation'] = result['cost']['all_text_tokens'] + released['feedback']['cost']['all_text_tokens']
        state = dict(version=VERSION, identity=identity, engine=engine.state_dict(), last_record=result)
        atomic_checkpoint(checkpoint, state)
        append_record(result_path, result)
        complete[row['id']] = result
        if observer is not None:
            observer(engine, result)
    return engine.graph.clone()


def combine_sources(graphs, width, memory_config):
    from .runner import graph_options
    combined = MemoryGraph(width, **graph_options(memory_config))
    for graph in graphs:
        for node in graph.nodes.values():
            if node['kind'] != 'role':
                combined.nodes[f'n{combined.next_id}'] = copy.deepcopy(node)
                combined.next_id += 1
    combined.recompute_edges()
    return combined


@contextmanager
def output_lock(output):
    path = Path(output).resolve()
    locks = path.parent / '.online_locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (digest(str(path)) + '.lock')).open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another process is using this output directory') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(config_path, manifest_path, suite, output, seed, selected=None, resume=False, backend=None, scorer=None):
    with output_lock(output):
        return _run(config_path, manifest_path, suite, output, seed, selected, resume, backend, scorer)


def _run(config_path, manifest_path, suite, output, seed, selected=None, resume=False, backend=None, scorer=None):
    config = validate_config(read_json(config_path))
    manifest = validate_manifest(read_json(manifest_path), verify_files=True)
    if suite not in config['suites']:
        raise ValueError('Unknown suite')
    conditions = config['suites'][suite]
    if selected:
        conditions = [c for c in conditions if c['name'] in selected]
        if {c['name'] for c in conditions} != set(selected):
            raise ValueError('Unknown selected condition')
    identity = dict(version=VERSION, manifest_sha256=manifest['sha256'], config_sha256=digest(config),
                    suite=suite, seed=seed, conditions=conditions)
    out = Path(output)
    if out.exists():
        if not resume:
            raise FileExistsError('Output exists; use --resume or a new output directory')
        prior = read_json(out / 'run.json')
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume config/manifest/conditions/seed mismatch')
        if prior['status'] == 'complete':
            return str(out)
    else:
        if resume:
            raise FileNotFoundError('Resume output does not exist')
        out.mkdir(parents=True)
        atomic_json(out / 'run.json', dict(identity, status='initializing'))
    if backend is None:
        from .backend import HFBackend
        backend = HFBackend(config['backend'])
    if resume and prior.get('backend') is not None and prior['backend'] != backend.resolved:
        raise ValueError('Resume backbone/encoder/runtime signature changed')
    meta = dict(identity, status='running', backend=backend.resolved,
                checkpoint_sha256=digest([config['backend'], seed, 'cold_memory_modules_CMA']),
                config=config, reset_boundary='task_category', source_build_cost={}, stream_setup={})
    atomic_json(out / 'run.json', meta)
    lookup = {r['id']: r for r in manifest['samples']}
    streams = {name: [lookup[sid] for sid in ids] for name, ids in manifest['streams'].items()}
    def execute(condition, key, rows, source=None, results=None):
        backend.synchronize()
        started = time.perf_counter()
        engine = OnlineEngine(backend, config, condition, seed, scorer, source)
        backend.synchronize()
        meta['stream_setup'][key] = dict(initialization_seconds_this_invocation=time.perf_counter() - started,
            source_import_seconds_this_invocation=getattr(engine, 'source_import_seconds', 0.),
            resumed_checkpoint=(out / 'states' / key / 'state.pt').exists(),
            accounting='Source import is included in initialization; do not add it twice')
        stream_identity = dict(identity, stream_key=key, condition=condition['name'], order=[r['id'] for r in rows],
                               backend_sha256=digest(backend.resolved))
        return run_stream(engine, rows, out / 'states' / key, stream_identity, results)
    try:
        if suite == 'transfer':
            for direction, target_names in config['transfer_directions'].items():
                source_graphs, target_rows = [], []
                source_started = time.perf_counter()
                builder = dict(config['suites']['main'][-1], name='source_build')
                for domain, rows in streams.items():
                    source_rows = [r for r in rows if r['dataset'] not in target_names]
                    target_rows += [r for r in rows if r['dataset'] in target_names]
                    if source_rows:
                        source_graphs.append(execute(builder, f'{direction}/source/{domain}', source_rows))
                if not target_rows or not source_graphs:
                    raise ValueError(f'{direction}: both source and target problems are required')
                source = combine_sources(source_graphs, backend.width, config['memory'])
                meta['source_build_cost'][direction] = dict(elapsed_seconds_this_invocation=time.perf_counter() - source_started,
                    source_problems=sum(len([r for r in rows if r['dataset'] not in target_names]) for rows in streams.values()),
                    graph=source.stats(), accounting='source journals contain all first-answer and adaptation costs')
                for condition in conditions:
                    imported = None if condition['scope'] == 'target' else source
                    execute(condition, f'{direction}/target/{condition["name"]}', target_rows, imported,
                            out / (condition['name'] + '.jsonl'))
        else:
            for condition in conditions:
                for domain, rows in streams.items():
                    execute(condition, f'{condition["name"]}/{domain}', rows, results=out / (condition['name'] + '.jsonl'))
    except BaseException as exc:
        meta.update(status='interrupted', error_type=type(exc).__name__, error=str(exc))
        atomic_json(out / 'run.json', meta)
        raise
    meta['status'] = 'complete'
    atomic_json(out / 'run.json', meta)
    return str(out)
