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
from .paper_engine import PaperEngine, REVISION, ROLE_IDS, ablation_conditions
from .protocol import read_json, write_json, digest
from .online_protocol import prepare_stream, validate_manifest
from .online import atomic_json, records
from .online_runner import run_stream, output_lock

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]


def signature():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.glob('*.py'))}


def prepare(output):
    out=Path(output)
    if out.exists():
        raise FileExistsError('Use a new, declared run directory')
    old=PACKAGE/'runs/temporal_2026-10-16_v2'
    config=read_json(old/'config.json')
    template=read_json(PROJECT/'paper/kbs_revision_2026-10-09/domain_role_templates.json')
    out.mkdir(parents=True)
    temporal=out/'temporal';temporal.mkdir()
    # Exact same examples, IDs, order and source-file hashes as the completed v2.
    shutil.copyfile(old/'manifest.json',temporal/'manifest.json')
    pilot=out/'ablation';pilot.mkdir()
    data=pilot/'data';data.mkdir()
    for filename in ('math500_real.json','HotpotQA_real.json'):
        source=PROJECT/'experiments/data'/filename
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        rows=[dict(row,source_revision='local-export-sha256:'+sha,
                   id=f'{filename}:{i}',source_index=i) for i,row in enumerate(read_json(source))]
        write_json(data/filename,rows)
    manifest=prepare_stream(data,pilot/'manifest.json',datasets=['math500','hotpotqa'],seed=42,limit=30)
    assert manifest['counts']=={'hotpotqa':30,'math500':30}
    config.update(method_revision=REVISION,role_templates=template,
                  purpose='graph_rerun_and_diagnostic_ablation_not_final_paper_results',
                  seed=42,repetition_count=1,per_node_feedback_probes=False)
    full=copy.deepcopy(config['suites']['graph_analysis'][0])
    full.update(storage='typed',topology='graph',capsules=True,merge=True,prune=True)
    none=copy.deepcopy(config['suites']['graph_analysis'][1])
    jobs=[]
    for condition in (full,none):
        jobs.append(dict(key='temporal/'+condition['name'],suite='temporal',
                         domain='reasoning',condition=condition,total=300))
    for domain in ('science_math','reasoning'):
        for condition in ablation_conditions(full):
            condition['agents'] = len(ROLE_IDS[domain])
            jobs.append(dict(key='ablation/'+domain+'/'+condition['name'],suite='ablation',
                             domain=domain,condition=condition,total=30))
    config['jobs']=sorted(jobs,key=lambda j: j['suite']!='ablation')
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='prepared',method_revision=REVISION,jobs_total=len(jobs)))
    print(json.dumps({'status':'prepared','temporal_examples_per_condition':300,
                      'ablation_examples_per_domain_condition':30,'jobs':len(jobs)}),flush=True)


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
    from .backend import HFBackend
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
                engine=PaperEngine(backend,c,job['condition'],42)
                identity=dict(method_revision=REVISION,condition=job['condition']['name'],seed=42,
                              repeat_id=0,stream_key=job['key'],manifest_sha256=manifest['sha256'],
                              config_sha256=digest(c),source_sha256=code,
                              backend_sha256=digest(backend.resolved),team_signature=engine.team_signature)
                positions={math.ceil(len(rows)*x) for x in (.25,.5,1.)}
                def observe(current,record):
                    if job['suite']=='temporal' and job['condition']['name']=='kg_multi' and current.episode in positions:
                        atomic_json(out/'temporal/snapshots'/f'{current.episode:04d}.json',
                                    dict(current.graph.snapshot(),episode=current.episode,run=[42,0],phase='post_update'))
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
                if job['key']=='temporal/none_multi':
                    export_temporal(out)
            summarize(out)
            status.update(status='complete',updated_at_unix=time.time())
            atomic_json(out/'status.json',status)
        except BaseException as exc:
            status.update(status='interrupted',error_type=type(exc).__name__,error=str(exc),updated_at_unix=time.time())
            atomic_json(out/'status.json',status)
            raise


def export_temporal(output):
    out=Path(output);streams=[];request=None
    for name in ('kg_multi','none_multi'):
        rows=records(out/'temporal'/name/'results.jsonl')
        if len(rows)!=300:raise ValueError('Incomplete temporal run')
        streams.append(dict(condition='KG-LMF' if name=='kg_multi' else 'NoMem-M',training_seed=42,repeat_id=0,
                            episodes=[dict(episode=r['episode'],id=r['id'],dataset=r['dataset'],score=r['score'],
                                           answer_recorded_before_feedback=True) for r in rows]))
        if name=='kg_multi':
            traces=[r['request_trace'] for r in rows if r.get('request_trace')]
            if traces:request=traces[0]
    snapshots=[read_json(p) for p in sorted((out/'temporal/snapshots').glob('*.json'))]
    if len(snapshots)!=3 or request is None:
        raise ValueError('No substitute for missing real graph traces')
    write_json(out/'temporal/figure_data.json',dict(protocol='kglmf-temporal-2026-10-15',model='Qwen3.5-4B',
               domain='reasoning',sampling_design='single_run',total_episodes=300,representative_run=[42,0],
               runs=streams,snapshots=snapshots,request=request))


def summarize(output):
    out=Path(output);config=read_json(out/'config.json');summary=[]
    for job in config['jobs']:
        directory=out/job['key'];rows=records(directory/'results.jsonl')
        if len(rows)!=job['total'] or not (directory/'completed.json').exists():
            raise ValueError('Cannot summarize incomplete jobs as final results')
        summary.append(dict(job=job['key'],n=len(rows),score_percent=100*statistics.mean(r['score'] for r in rows),
                            prediction_seconds=statistics.mean(r['cost']['prediction_seconds'] for r in rows),
                            latency_seconds=statistics.mean(r['cost']['prediction_seconds']+r['cost']['post_answer_seconds'] for r in rows),
                            text_tokens=statistics.mean(r['cost']['total_text_tokens_including_adaptation'] for r in rows),
                            latent_positions=statistics.mean(r['cost'].get('latent_positions',0)+r['cost'].get('feedback_latent_positions',0) for r in rows)))
    write_json(out/'summary.json',dict(purpose='DIAGNOSTIC ONLY — NOT FINAL PAPER ABLATION RESULTS',
               seed=42,repetitions=1,uncertainty=None,method_revision=REVISION,results=summary))
    export_temporal(out)
    print(json.dumps(summary),flush=True)
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
