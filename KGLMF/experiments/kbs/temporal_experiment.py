"""Run only the requested online graph experiment; one mixed reasoning stream.

prepare creates an auditable local-export manifest. run uses a real frozen HF
backbone and online memory training. export reads completed journals only.
"""
from .prompt_resources import domain_prompts, role_templates
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from .protocol import digest, read_json, write_json
from .online_protocol import prepare_stream, validate_manifest
from .online import atomic_json, records
from .online_runner import run_stream, output_lock
from .temporal_engine import TemporalEngine

ROOT=Path(__file__).resolve().parents[2]
REVISION='typed-compositional-temporal-2026-10-16-v2'


def prepare(output, per_dataset_limit=150):
    out=Path(output)
    if out.exists():
        raise FileExistsError('Use a new directory for a new declared experiment')
    out.mkdir(parents=True)
    data=out/'data';data.mkdir()
    for name in ('HotpotQA_real.json','strategyqa_real.json'):
        source=ROOT/'experiments'/'data'/name
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        rows=read_json(source)
        enriched=[dict(r,source_revision='local-export-sha256:'+sha,id=f'{name}:{i}',source_index=i) for i,r in enumerate(rows)]
        write_json(data/name,enriched)
    manifest=prepare_stream(data,out/'manifest.json',datasets=['strategyqa','hotpotqa'],seed=42,limit=per_dataset_limit)
    if manifest['counts']!={'hotpotqa':per_dataset_limit,'strategyqa':per_dataset_limit}:
        raise ValueError('Both benchmarks must supply the declared number of distinct problems')
    config=read_json(Path(__file__).parent/'configs'/'qwen.json')
    config['seeds']=[42]
    config['backend'].update(encoder='D:/HF_models/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
                             enable_thinking=False)
    config['memory'].pop('edge_coefficients',None)
    config.update(method_revision=REVISION,total_episodes=2*per_dataset_limit,per_node_feedback_probes=False,
                  repetition_count=1,order_seed=42,window=50,
                  domain_prompts=domain_prompts())
    full=next(c for c in config['suites']['main'] if c['name']=='kg_multi')
    none=next(c for c in config['suites']['main'] if c['name']=='none_multi')
    config['suites']={'graph_analysis':[full,none]}
    write_json(out/'config.json',config)
    atomic_json(out/'status.json',dict(status='prepared',method_revision=REVISION,counts=manifest['counts']))
    print(json.dumps({'status':'prepared','counts':manifest['counts'],'order_seed':42,'repetitions':1}),flush=True)


def source_signature():
    names=['temporal_memory.py','temporal_engine.py','temporal_experiment.py','online.py','online_runner.py','memory.py','backend.py',
           'protocol.py','online_protocol.py','runner.py','training.py']
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}


def run(output, resume=False):
    out=Path(output)
    config=read_json(out/'config.json');manifest=validate_manifest(read_json(out/'manifest.json'),verify_files=True)
    if config.get('method_revision')!=REVISION:
        raise ValueError('Wrong experiment method revision')
    from .backend import HFBackend
    with output_lock(out):
        status=read_json(out/'status.json')
        if status['status']=='complete':
            return export(out)
        if status['status']!='prepared' and not resume:
            raise ValueError('Use --resume for this existing run')
        code=source_signature()
        if status.get('source_sha256') and status['source_sha256']!=code:
            raise ValueError('Method code changed during a run; do not mix versions')
        status.update(status='loading_model',source_sha256=code)
        atomic_json(out/'status.json',status)
        try:
            backend=HFBackend(config['backend'])
            # Paired lexical anchors and fixed queries do not change across conditions.
            original_anchors=backend.lexical_anchors
            anchor_cache={}
            def anchors(words):
                key=tuple(words)
                if key not in anchor_cache:anchor_cache[key]=original_anchors(words)
                return anchor_cache[key]
            backend.lexical_anchors=anchors
            backend.resolved['enable_thinking']=False
            status.update(status='running',backend=backend.resolved)
            atomic_json(out/'status.json',status)
            lookup={r['id']:r for r in manifest['samples']}
            rows=[lookup[sid] for sid in manifest['streams']['reasoning']]
            total=len(rows);positions=[math.ceil(total*x) for x in (.25,.5,1.)]
            for condition in config['suites']['graph_analysis']:
                name=condition['name'];start=time.perf_counter()
                engine=TemporalEngine(backend,config,condition,42)
                identity=dict(method_revision=REVISION,condition=name,seed=42,repeat_id=0,
                              stream_key=name+'/reasoning',manifest_sha256=manifest['sha256'],
                              config_sha256=digest(config),source_sha256=code,backend_sha256=digest(backend.resolved))
                def observe(current,record):
                    step=current.episode
                    if name=='kg_multi' and step in positions:
                        snapshot=dict(current.graph.snapshot(),episode=step,run=[42,0],phase='post_update')
                        atomic_json(out/'snapshots'/f'{step:04d}.json',snapshot)
                    status.update(status='running',condition=name,completed=step,total=total,
                                  last_score=record['score'],graph=current.graph.stats(),elapsed_condition_seconds=time.perf_counter()-start)
                    atomic_json(out/'status.json',status)
                    print(json.dumps({'condition':name,'episode':step,'total':total,'score':record['score'],
                                      'nodes':len(current.graph.nodes),'edges':len(current.graph.edges),
                                      'updates':record.get('updates',[]),'seconds':round(record['cost']['prediction_seconds']+record['cost']['post_answer_seconds'],2)}),flush=True)
                run_stream(engine,rows,out/'states'/name,identity,out/f'{name}.jsonl',observer=observe)
                del engine
                if torch.cuda.is_available():torch.cuda.empty_cache()
            status.update(status='complete')
            atomic_json(out/'status.json',status)
            export(out)
        except BaseException as exc:
            status.update(status='interrupted',error_type=type(exc).__name__,error=str(exc))
            atomic_json(out/'status.json',status)
            raise


def export(output):
    out=Path(output);config=read_json(out/'config.json')
    runs=[];request=None
    for condition in config['suites']['graph_analysis']:
        rows=records(out/f'{condition["name"]}.jsonl')
        if len(rows)!=config['total_episodes']:
            raise ValueError('Never export an incomplete stream as a completed experiment')
        runs.append(dict(condition='KG-LMF' if condition['name']=='kg_multi' else 'NoMem-M',training_seed=42,repeat_id=0,
                         episodes=[dict(episode=r['episode'],id=r['id'],dataset=r['dataset'],score=r['score'],
                                        answer_recorded_before_feedback=True) for r in rows]))
        traces=[r['request_trace'] for r in rows if r.get('request_trace')]
        if condition['name']=='kg_multi' and traces:
            request=traces[0]
    snapshots=[read_json(p) for p in sorted((out/'snapshots').glob('*.json'))]
    if len(snapshots)!=3 or request is None:
        raise ValueError('Missing true snapshots or an eligible pre-answer request; no fallback graph is invented')
    data=dict(protocol='kglmf-temporal-2026-10-15',model='Qwen3.5-4B',domain='reasoning',
              sampling_design='single_run',total_episodes=config['total_episodes'],representative_run=[42,0],
              runs=runs,snapshots=snapshots,request=request)
    atomic_json(out/'figure_data.json',data)
    print('Exported completed first-answer streams, snapshots and original routing scores.',flush=True)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','export'])
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--per-dataset-limit',type=int,default=150)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.per_dataset_limit)
    elif a.action=='run':run(a.output,a.resume)
    else:export(a.output)
