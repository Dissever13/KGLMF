"""Versioned graph rerun and author-requested diagnostic ablations.

No results from the small ablation pilot are publication results. Importing this
module does not load a model, alter a run, or launch inference.
"""
import argparse
import copy
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import torch
from .aligned_engine import AlignedEngine as PaperEngine, REVISION
from .paper_engine import ROLE_IDS, ablation_conditions
from .aligned_data import prepare_data
from .protocol import read_json, write_json, digest
from .online_protocol import prepare_stream, validate_manifest
from .online import atomic_json, records
from .online_runner import run_stream, output_lock

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]


def signature():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.glob('*.py'))}


def prepare(output):
    from transformers import AutoTokenizer
    out=Path(output)
    if out.exists() and any(p.is_file() for p in out.rglob('*')): raise FileExistsError('Use a new independent directory')
    base=PACKAGE/'runs/paper_alignment_2026-10-17_v3/config.json'
    config=read_json(base)
    out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(config['backend']['model'],local_files_only=True)
    raw=Path('D:/KGLM/docs/paper_alignment_2026-10-18_v4/raw_sources')
    prepare_data(out,raw,tokenizer)
    config.update(method_revision=REVISION, seeds=[42,43,44], repetition_count=1,
                  purpose='independent_method_validation_ablation_author_report_only', per_node_feedback_probes=False,
                  order_seed=20261018, window=50)
    config['backend'].update(max_input_tokens=4096,max_new_tokens=768)
    config['training'].update(router_learning_rate=1e-4,synthesizer_learning_rate=5e-5,
                             adapter_learning_rate=1e-5,warmup_episodes=20)
    config['data_input']='Hotpot all supplied distractor paragraphs (up to ten), 1800 tokens; StrategyQA and MATH question-only'
    config['graph_revision']=dict(semantic_neighbors=3,semantic_threshold=.3,grace_episodes=5)
    config['role_feedback']='role-specific rubric for intermediate roles; benchmark answers; critic/verifier verified answer and decision consistency; unavailable never zero'
    full=copy.deepcopy(config['suites']['graph_analysis'][0])
    full.update(storage='typed',topology='graph',capsules=True,merge=True,prune=True)
    none=copy.deepcopy(config['suites']['graph_analysis'][1])
    jobs=[]
    for suite in ('ablation','temporal'):
        for seed in config['seeds']:
            for domain in (('science_math','reasoning') if suite=='ablation' else ('reasoning',)):
                conditions=ablation_conditions(full) if suite=='ablation' else [full,none]
                for condition in conditions:
                    condition=copy.deepcopy(condition);condition['agents']=len(ROLE_IDS[domain])
                    jobs.append(dict(key=f'{suite}/seed_{seed}/{domain}/{condition["name"]}',suite=suite,
                                     seed=seed,domain=domain,condition=condition,total=51 if suite=='ablation' else 302))
    config['jobs']=jobs
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='prepared',method_revision=REVISION,jobs_total=len(jobs)))
    print(json.dumps(dict(status='prepared',jobs=len(jobs),answers=sum(j['total'] for j in jobs))),flush=True)


def resolve_job(out,config,job):
    manifest=validate_manifest(read_json(out/job['suite']/'manifest.json'),verify_files=True)
    lookup={r['id']:r for r in manifest['samples']}
    rows=[lookup[sid] for sid in manifest['streams'][job['domain']]]
    if len(rows)!=job['total']:
        raise ValueError('Manifest differs from declared job size')
    c=copy.deepcopy(config)
    c.update(active_domain=job['domain'],total_episodes=len(rows))
    return manifest,rows,c


def run(output,resume=False):
    from .aligned_backend import AlignedBackend as HFBackend
    out=Path(output);config=read_json(out/'config.json')
    if config['method_revision']!=REVISION:
        raise ValueError('Wrong method revision')
    with output_lock(out):
        status=read_json(out/'status.json')
        if status['status']=='complete':
            return summarize(out)
        if status['status']!='prepared' and not resume:
            raise ValueError('Existing experiment requires --resume')
        code=signature()
        if status.get('source_sha256') and status['source_sha256']!=code:
            raise ValueError('Source changed; never mix method versions')
        snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for name,sha in code.items():
            saved=snapshot/name
            if saved.exists() and hashlib.sha256(saved.read_bytes()).hexdigest()!=sha:
                raise ValueError('Source snapshot mismatch')
            if not saved.exists():shutil.copyfile(PACKAGE/name,saved)
        status.update(status='loading_model',source_sha256=code)
        atomic_json(out/'status.json',status)
        try:
            backend=HFBackend(config['backend'])
            original=backend.lexical_anchors;cache={}
            def anchors(words):
                key=tuple(words)
                if key not in cache:cache[key]=original(words)
                return cache[key]
            backend.lexical_anchors=anchors
            status['backend']=backend.resolved
            for index,job in enumerate(config['jobs']):
                directory=out/job['key'];result_path=directory/'results.jsonl'
                manifest,rows,c=resolve_job(out,config,job)
                engine=PaperEngine(backend,c,job['condition'],job['seed'])
                identity=dict(method_revision=REVISION,condition=job['condition']['name'],seed=job['seed'],
                              repeat_id=0,stream_key=job['key'],manifest_sha256=manifest['sha256'],
                              config_sha256=digest(c),source_sha256=code,
                              backend_sha256=digest(backend.resolved),team_signature=engine.team_signature)
                positions={math.ceil(len(rows)*x) for x in (.25,.5,1.)}
                def observe(current,record):
                    if job['suite']=='temporal' and job['condition']['name']=='kg_multi' and current.episode in positions:
                        atomic_json(out/'temporal'/f'seed_{job["seed"]}'/'snapshots'/f'{current.episode:04d}.json',
                                    dict(current.graph.snapshot(),episode=current.episode,run=[job['seed'],0],phase='post_update'))
                    status.update(status='running',job=job['key'],job_index=index+1,
                                  jobs_total=len(config['jobs']),completed=current.episode,total=len(rows),
                                  last_score=record['score'],updated_at_unix=time.time())
                    atomic_json(out/'status.json',status)
                    print(json.dumps({'job':job['key'],'episode':current.episode,'total':len(rows),
                                      'score':record['score'],'graph':current.graph.stats(),
                                      'updates':record['updates']}),flush=True)
                run_stream(engine,rows,directory,identity,result_path,observer=observe)
                atomic_json(directory/'completed.json',dict(status='complete',identity=identity,n=len(rows)))
                del engine
                if torch.cuda.is_available():torch.cuda.empty_cache()
                if index == 35:
                    summarize_ablation(out)
            summarize(out)
            status.update(status='complete',updated_at_unix=time.time())
            atomic_json(out/'status.json',status)
        except BaseException as exc:
            status.update(status='interrupted',error_type=type(exc).__name__,error=str(exc),updated_at_unix=time.time())
            atomic_json(out/'status.json',status)
            raise


def job_results(out,job):
    directory=out/job['key'];rows=records(directory/'results.jsonl')
    if len(rows)!=job['total'] or not (directory/'completed.json').exists():
        raise ValueError('Incomplete job: '+job['key'])
    return rows


def export_temporal(output):
    out=Path(output);config=read_json(out/'config.json');streams=[];request=None
    for job in config['jobs']:
        if job['suite']!='temporal':continue
        rows=job_results(out,job)
        name=job['condition']['name']
        streams.append(dict(condition='KG-LMF' if name=='kg_multi' else 'NoMem-M',training_seed=job['seed'],repeat_id=0,
                            episodes=[dict(episode=r['episode'],id=r['id'],dataset=r['dataset'],score=r['score'],
                                           answer_recorded_before_feedback=True) for r in rows]))
        if name=='kg_multi' and job['seed']==42:
            traces=[r['request_trace'] for r in rows if r.get('request_trace')]
            if traces:request=traces[0]
    snapshots=[read_json(p) for p in sorted((out/'temporal/seed_42/snapshots').glob('*.json'))]
    if len(snapshots)!=3 or request is None:raise ValueError('Missing real graph trace')
    write_json(out/'temporal/figure_data.json',dict(protocol='kglmf-temporal-2026-10-15',model='Qwen3.5-4B',
        domain='reasoning',sampling_design='three_seeds_one_run_each',total_episodes=302,representative_run=[42,0],
        runs=streams,snapshots=snapshots,request=request,final_window_policy='merge_short_remainder'))


def summary_rows(out,suite):
    config=read_json(out/'config.json');result=[]
    for job in config['jobs']:
        if job['suite']!=suite:continue
        rows=job_results(out,job)
        result.append(dict(job=job['key'],condition=job['condition']['name'],domain=job['domain'],seed=job['seed'],
             n=len(rows),score_percent=100*statistics.mean(r['score'] for r in rows),
             prediction_seconds=statistics.mean(r['cost']['prediction_seconds'] for r in rows),
             latency_seconds=statistics.mean(r['cost']['prediction_seconds']+r['cost']['post_answer_seconds'] for r in rows),
             missing_role_scores=sum(u.get('feedback_unavailable',False) for r in rows for u in r['updates']),
             router_updates=sum(u['router_update'] for r in rows for u in r['updates']),
             likelihood_updates=sum(u['likelihood_update'] for r in rows for u in r['updates'])))
    groups={}
    for r in result:groups.setdefault((r['domain'],r['condition']),[]).append(r)
    aggregate=[]
    for (domain,condition),values in groups.items():
        assert sorted(v['seed'] for v in values)==[42,43,44]
        aggregate.append(dict(domain=domain,condition=condition,
             mean_percent=statistics.mean(v['score_percent'] for v in values),
             sd_percentage_points=statistics.stdev(v['score_percent'] for v in values),
             seeds=[v['seed'] for v in values],latency_seconds=statistics.mean(v['latency_seconds'] for v in values)))
    return dict(results=result,aggregate=aggregate,uncertainty='sample SD across three seeds; one run per seed',method_revision=REVISION)


def summarize_ablation(output):
    out=Path(output);summary=summary_rows(out,'ablation')
    summary['purpose']='AUTHOR VALIDATION ONLY; DO NOT INSERT AS FINAL PAPER ABLATIONS'
    write_json(out/'ablation_summary.json',summary)
    return summary


def summarize(output):
    out=Path(output);summarize_ablation(out)
    summary=summary_rows(out,'temporal');write_json(out/'temporal_summary.json',summary)
    export_temporal(out)
    print(json.dumps(summary['aggregate']),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','summarize','export'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    elif args.action=='run':run(args.output,args.resume)
    elif args.action=='export':export_temporal(args.output)
    else:summarize(args.output)

if __name__=='__main__':main()
