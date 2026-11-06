"""Current paper method for the explicitly requested graph-analysis experiment."""
import copy
import torch
from KGLMF.src.agents.online import OnlineEngine, total_cost
from KGLMF.src.execution.runner import question_text
from KGLMF.src.data.protocol import digest, provenance_ids
from KGLMF.src.composer.typed import TemporalMemory
from KGLMF.src.graph.temporal import TemporalGraph


def make_temporal_memory(backend, config):
    keys={'max_tokens','temperature','hops','alpha','write_tokens','dropout'}
    return TemporalMemory(backend.width,backend.hidden,**{k:v for k,v in config.items() if k in keys}).to(backend.device)


class TemporalEngine(OnlineEngine):
    memory_factory=staticmethod(make_temporal_memory)
    graph_type=TemporalGraph

    def __init__(self, backend, config, condition, seed, scorer=None, source_graph=None):
        if source_graph is not None:
            raise ValueError('Temporal graph analysis starts without experience')
        self.prompts=config['domain_prompts']
        self.record_cache={}
        self.trace_recorded=False
        super().__init__(backend,config,condition,seed,scorer)

    def instruction(self, role, domain='reasoning'):
        return next(r['instruction'] for r in self.prompts['domains'][domain]['roles'] if r['id']==role)

    def build_prompt(self, public, role, history='', memory=''):
        domain=public.get('stream_domain','reasoning')
        output=self.prompts['format'][role]
        if domain=='reasoning':
            answer=('exactly yes or no' if public['metric']=='boolean' else
                    'only the short answer entity or phrase, without explanation')
            if role=='planner':
                output='Return PLAN: and at most three short steps (60 words maximum).'
            elif role=='solver':
                output=f'Give at most 80 words of reasoning, then a separate line FINAL: followed by {answer}.'
            else:
                output=f'Review the preceding solution. Return only two lines: VERDICT: ACCEPT or REVISE, then FINAL: followed by {answer}.'
            output+=' Use plain text without Markdown formatting.'
        return (self.prompts['common']+'\n'+self.instruction(role,domain)+
                '\nQuestion:\n'+question_text(public)+
                ('\nPrior experience:\n'+memory if memory else '')+
                ('\nPrevious agents:\n'+history if history else '')+
                '\nYour response format:\n'+output)

    def communication_text(self, response):
        # Preserve the conclusion as well as the opening reasoning within the
        # same communication budget in both experimental conditions.
        tokenizer=self.backend.tokenizer
        ids=tokenizer.encode(response,add_special_tokens=False)
        budget=self.condition.get('communication_tokens',256)
        if len(ids)<=budget:
            return response
        marker=tokenizer.encode('\n[... omitted ...]\n',add_special_tokens=False)
        if budget<=len(marker)+2:
            return tokenizer.decode(ids[-budget:])
        remaining=budget-len(marker)
        head=remaining//2
        return tokenizer.decode(ids[:head]+marker+ids[-(remaining-head):])

    def record_components(self, public, role, response, kind, knowledge=None):
        if kind=='role':
            text=self.instruction(role,public.get('stream_domain','reasoning'))
        elif kind=='context':
            text=question_text(public)
        else:
            # Put the completed-episode feedback before a potentially long
            # interaction so the frozen encoder's token cap cannot drop it.
            text=f'Role: {role}\nFeedback: {knowledge or ""}\nProblem: {question_text(public)}\nInteraction: {self.communication_text(response)}'
        key=digest([kind,text])
        if key not in self.record_cache:
            self.record_cache[key]=dict(kind=kind,representations=self.backend.encode_record(text))
        return self.record_cache[key]

    def record_retrieval(self, call, row, prediction):
        self.graph.read_scored(call['retrieval']['ids'],row['dataset'],self.episode,
                               f'{row["id"]}:{self.request_key(call)}',provenance_ids(row),float(prediction['score']))

    def maintain_graph(self, full, family):
        created=[]
        if full and family=='latent' and self.condition.get('capsules',True):
            created=self.graph.construct_capsules(self.memory,self.episode)
        result=self.graph.maintain(self.episode,merge=full and family=='latent' and self.condition.get('merge',True),
                                   prune=full and family=='latent' and self.condition.get('prune',True))
        result.update(constructed=len(created),construction_events=created)
        self.graph.pending_requests.clear()
        return result

    @torch.no_grad()
    def predict(self, public):
        prediction=super().predict(public)
        if self.condition['family']=='latent' and not self.trace_recorded and self.episode+1>self.config['total_episodes']/2:
            for call in prediction['calls']:
                selected=call['retrieval']['ids']
                eligible=any(self.graph.nodes[n]['kind']=='capsule' and
                             any(parent in self.graph.nodes for parent in self.graph.nodes[n].get('constituents',[])) for n in selected)
                if eligible:
                    prediction['request_trace']=dict(self.graph.snapshot(),
                        request_id=f'{public["id"]}:{self.request_key(call)}',episode=self.episode+1,phase='pre_answer',
                        selection_rule='first_eligible_after_midpoint',selected_ids=selected,
                        initial_scores=call['retrieval']['initial_scores'],propagated_scores=call['retrieval']['propagated_scores'])
                    break
        return prediction

    def collect_feedback(self, row, prediction):
        if self.condition['family']=='none':
            return dict(requests=[dict(role=c['role'],internal={'accepted':None},verdict=prediction['verdict'],
                                      target=None,target_origin=None,pair=None,per_node_pairs={}) for c in prediction['calls']],
                        cost=dict(total_cost([]),feedback_seconds=0.),source='internal_and_post_answer_benchmark',
                        graph_parameter_version=prediction['parameter_version'],graded_outputs={})
        return super().collect_feedback(row,prediction)

    def adapt(self, row, prediction, feedback):
        result=super().adapt(row,prediction,feedback)
        self.trace_recorded=self.trace_recorded or 'request_trace' in prediction
        # Frozen-encoder cache is an optimization, not part of learned state.
        self.record_cache.clear()
        return result

    def state_dict(self):
        return dict(super().state_dict(),trace_recorded=self.trace_recorded,
                    method_revision='typed-compositional-temporal-2026-10-16-v2')

    def load_state(self,state):
        if state.get('method_revision')!='typed-compositional-temporal-2026-10-16-v2':
            raise ValueError('Cannot import an old promotion/edge checkpoint')
        super().load_state(state)
        self.trace_recorded=state['trace_recorded']
