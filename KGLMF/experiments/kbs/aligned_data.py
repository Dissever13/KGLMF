"""Fixed held-out manifests with label-independent Hotpot evidence packing."""
import copy
import hashlib
from pathlib import Path
from .protocol import read_json, write_json, digest, question_key
from .online_protocol import prepare_stream

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]


def evidence_text(context, tokenizer, budget=1800):
    if isinstance(context, dict):
        context = list(zip(context['title'], context['sentences']))
    if not isinstance(context, list) or not 1 <= len(context) <= 10:
        raise ValueError('Hotpot distractor requires its supplied public paragraphs (up to ten)')
    blocks = ['['+str(title)+']\n'+''.join(sentences) for title,sentences in context]
    tokens = [tokenizer.encode(block, add_special_tokens=False) for block in blocks]
    # Water-fill equally, then return spare capacity to paragraphs that need it.
    kept = [0]*len(tokens)
    remaining = budget
    while remaining and any(n < len(t) for n,t in zip(kept,tokens)):
        for i,t in enumerate(tokens):
            if remaining and kept[i] < len(t):
                kept[i] += 1
                remaining -= 1
    text = '\n\n'.join(tokenizer.decode(t[:n], skip_special_tokens=True) for t,n in zip(tokens,kept))
    return text, dict(paragraphs=len(context), original_tokens=sum(map(len,tokens)), retained_tokens=sum(kept),
                      truncated_paragraphs=sum(n<len(t) for n,t in zip(kept,tokens)), allocation=kept,
                      rule='equal_share_with_spare_redistribution_in_original_order', token_budget=budget)


def prepare_data(out, raw_dir, tokenizer):
    out, raw_dir = Path(out), Path(raw_dir)
    used = set()
    old_manifests = [PACKAGE/'runs/temporal_2026-10-15/manifest.json',
                     PACKAGE/'runs/temporal_2026-10-16_v2/manifest.json',
                     PACKAGE/'runs/paper_alignment_2026-10-17_v3/ablation/manifest.json']
    for path in old_manifests:
        old = read_json(path)
        used.update(question_key(r['question']) for r in old['samples'])
    hp = read_json(raw_dir/'hotpot_validation.json')
    strategy = read_json(raw_dir/'strategy_train.json') + read_json(raw_dir/'strategy_dev.json')
    math = read_json(PROJECT/'experiments/data/math500_real.json')
    pools = {'hotpotqa':hp, 'strategyqa':strategy, 'math500':math}
    chosen = {}
    for name,items in pools.items():
        unique = {}
        for i,item in enumerate(items):
            q = item.get('question', item.get('problem'))
            key = question_key(q)
            if key in used or key in unique:
                continue
            unique[key] = dict(item, question=q, source_index=i)
        ranked = sorted(unique.values(), key=lambda r: digest([20261018, name, question_key(r['question'])]))
        required = 202 if name=='hotpotqa' else 151 if name=='strategyqa' else 51
        if len(ranked) < required:
            raise ValueError('Insufficient previously unused samples: '+name)
        chosen[name] = ranked[:required]
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.iterdir() if p.is_file()}
    source_hashes['math500_real.json'] = hashlib.sha256((PROJECT/'experiments/data/math500_real.json').read_bytes()).hexdigest()
    protocols = []
    manifests = {}
    for suite in ('ablation', 'temporal'):
        data = out/suite/'data';data.mkdir(parents=True,exist_ok=True)
        subsets = {'hotpotqa':chosen['hotpotqa'][:51], 'math500':chosen['math500']} if suite=='ablation' else {
            'hotpotqa':chosen['hotpotqa'][51:], 'strategyqa':chosen['strategyqa']}
        for name,items in subsets.items():
            rows = []
            for item in items:
                row = copy.deepcopy(item)
                native_id = row.get('id',row.get('qid',row.get('unique_id',str(row['source_index']))))
                row['id'] = str(native_id)
                row['original_question'] = row['question']
                row['source_revision'] = 'frozen-source-sha256:'+digest(source_hashes)
                if name=='hotpotqa':
                    packed,stats = evidence_text(row['context'],tokenizer)
                    row['question'] += '\n\nProvided documents:\n'+packed
                    row['input_protocol'] = 'budgeted_distractor_all_supplied_paragraphs'
                    protocols.append(dict(suite=suite,id=row['id'],**stats))
                else:
                    row['input_protocol'] = 'question_only'
                if name=='math500' and 'answer' not in row:
                    raise ValueError('MATH reference missing')
                # Metadata remains private; only the packed public question crosses prediction_view.
                rows.append(row)
            filename={'hotpotqa':'HotpotQA_real.json','strategyqa':'strategyqa_real.json','math500':'math500_real.json'}[name]
            write_json(data/filename,rows)
        manifests[suite] = prepare_stream(data,out/suite/'manifest.json',datasets=list(subsets),seed=20261018)
    questions = [r['raw']['original_question'] for m in manifests.values() for r in m['samples']]
    assert len({question_key(q) for q in questions}) == 404
    assert not used.intersection(question_key(q) for q in questions)
    write_json(out/'data_protocol.json',dict(excludes_previous_graph_and_pilot_questions=True,
        selection='sha256 order without answer or difficulty filtering',selection_seed=20261018,
        prior_manifests=[str(p) for p in old_manifests],source_hashes=source_hashes,
        original_source='HotpotQA official HF distractor validation; StrategyQA author train/dev split; local MATH-500 export',
        independent_scope='disjoint from recorded v1/v2/v3 graph and pilot streams; not a claim about backbone pretraining',
        hotpot=protocols,feedback='references and rubric feedback after durable first answer',
        sampling_design='three_training_and_generation_seeds_fixed_problem_order_one_run_each'))
    return manifests
