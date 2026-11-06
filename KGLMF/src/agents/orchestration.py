"""Domain-specific execution of the typed, compositional paper method.

Old experiment snapshots remain immutable. New runs must use a new directory.
Agents within one stage receive the same preceding history; GPU calls are
serialized without introducing messages between independent stage members.
"""
import copy
from KGLMF.src.agents.temporal import TemporalEngine
from KGLMF.src.composer.typed import TemporalMemory
from KGLMF.src.agents.online import OnlineEngine
from KGLMF.src.data.protocol import digest

REVISION = 'paper-aligned-domain-teams-2026-10-17-v3'
ROLE_IDS = {
    'reasoning': ['planner', 'solver', 'critic'],
    'qa': ['analyst', 'answerer', 'verifier'],
    'code': ['analyst', 'designer', 'programmer', 'reviewer', 'tester'],
    'science_math': ['analyst', 'solver', 'verifier', 'integrator'],
}
STAGES = {
    'reasoning': [['planner'], ['solver'], ['critic']],
    'qa': [['analyst'], ['answerer'], ['verifier']],
    'code': [['analyst'], ['designer'], ['programmer'], ['reviewer', 'tester'], ['programmer']],
    'science_math': [['analyst'], ['solver'], ['verifier'], ['integrator']],
}
PRIMARY = {'reasoning': 'solver', 'qa': 'answerer', 'code': 'programmer', 'science_math': 'solver'}


def make_memory(backend, config):
    keys = {'max_tokens', 'temperature', 'hops', 'alpha', 'write_tokens', 'dropout', 'role_names'}
    return TemporalMemory(backend.width, backend.hidden,
                          **{k: v for k, v in config.items() if k in keys}).to(backend.device)


class PaperEngine(TemporalEngine):
    memory_factory = staticmethod(make_memory)

    def __init__(self, backend, config, condition, seed, scorer=None, source_graph=None):
        config = copy.deepcopy(config)
        self.active_domain = config['active_domain']
        if self.active_domain not in ROLE_IDS:
            raise ValueError('Unknown task domain')
        self.catalog = config['role_templates']
        self.team = self.catalog['domains'][self.active_domain]['roles']
        self.role_ids = ROLE_IDS[self.active_domain]
        if len(self.team) != len(self.role_ids):
            raise ValueError('Role catalog and executable team disagree')
        config['memory']['role_names'] = self.role_ids
        condition = copy.deepcopy(condition)
        if condition['agents'] != 1:
            condition['agents'] = len(self.role_ids)
        self.team_signature = digest([self.active_domain, self.team, STAGES[self.active_domain]])
        super().__init__(backend, config, condition, seed, scorer, source_graph)
        for node in self.graph.nodes.values():
            if node['kind'] == 'role':
                node['domains'] = [self.active_domain]
                node['task'] = self.active_domain

    def initial_roles(self):
        return list(self.role_ids)

    def instruction(self, role, domain=None):
        if domain is not None and domain != self.active_domain:
            raise ValueError('A domain stream must reset its team and memory')
        return self.team[self.role_ids.index(role)]['instruction']

    def execution_stages(self, public):
        if public['stream_domain'] != self.active_domain:
            raise ValueError('Mixed-domain state reuse is not allowed')
        return ([[PRIMARY[self.active_domain]]] if self.condition['agents'] == 1
                else copy.deepcopy(STAGES[self.active_domain]))

    def is_solver_request(self, role):
        return role == PRIMARY[self.active_domain]

    def build_prompt(self, public, role, history='', memory=''):
        from KGLMF.src.execution.runner import question_text
        domain = self.active_domain
        name = self.team[self.role_ids.index(role)]['name']
        if public['metric'] == 'boolean':
            final = 'FINAL: followed by exactly yes or no.'
        elif public['metric'] == 'choice':
            final = 'FINAL: followed by only the chosen option letter.'
        elif public['domain'] == 'math':
            final = r'FINAL: followed by the mathematical answer in \boxed{...}.'
        else:
            final = 'FINAL: followed by only the short answer entity or phrase, without explanation.'
        if domain == 'code':
            if role == 'programmer':
                contract = ('Return a complete Python solution in one fenced python block. '
                            'If review or test-design messages are present, revise the candidate accordingly.')
            elif role == 'tester':
                contract = ('Propose concise tests from the public specification. '
                            'Do not claim tests were executed and do not provide the final program.')
            else:
                contract = 'Give at most 100 words of role-specific analysis; do not provide the final program.'
        elif role in ('planner', 'analyst'):
            contract = 'Return PLAN: and at most three short steps (60 words maximum).'
        elif role in ('critic', 'verifier'):
            contract = 'Return only two lines: VERDICT: ACCEPT or REVISE, then ' + final
        elif role == 'integrator':
            contract = 'Resolve the preceding proposals and return only ' + final
        else:
            contract = 'Give at most 80 words of reasoning, then a separate line ' + final
        common = self.prompts['common']
        return (common + '\nRole: ' + name + '\n' + self.instruction(role) +
                '\nQuestion:\n' + question_text(public) +
                ('\nPrior experience:\n' + memory if memory else '') +
                ('\nPrevious agents:\n' + history if history else '') +
                '\nYour response format:\n' + contract + ' Use plain text except for requested code or mathematics.')

    def state_dict(self):
        state = super().state_dict()
        state.update(method_revision=REVISION, team_signature=self.team_signature,
                     active_domain=self.active_domain)
        return state

    def load_state(self, state):
        if (state.get('method_revision') != REVISION or
                state.get('team_signature') != self.team_signature or
                state.get('active_domain') != self.active_domain):
            raise ValueError('Checkpoint method, domain or role schedule mismatch')
        OnlineEngine.load_state(self, state)
        self.trace_recorded = state['trace_recorded']


def ablation_conditions(full):
    """Exactly the five changes in the manuscript, preserving all other controls."""
    full = copy.deepcopy(full)
    full.update(name='kg_multi', storage='typed', topology='graph', capsules=True,
                merge=True, prune=True, role_condition=True)
    variants = [('fixed_alignment', {'adapter': 'cma'}),
                ('flat_latent', {'topology': 'flat'}),
                ('no_role', {'role_condition': False, 'router_role_condition': False,
                             'composition_role_condition': False}),
                ('no_capsules', {'capsules': False}),
                ('no_maintenance', {'merge': False, 'prune': False})]
    return [full] + [dict(full, **dict(changes, name=name)) for name, changes in variants]
