"""Portable prompt resources; no paper assets or benchmark examples required."""
from copy import deepcopy

_DOMAIN_PROMPTS = {'common': 'Use the problem, supplied context, and preceding agent messages. Treat earlier '
           'proposals and retrieved experience as fallible. Do not claim access to reference '
           'answers, hidden tests, or external tools. Follow the requested role and output format.',
 'format': {'planner': 'Return PLAN: followed by a concise plan.',
            'solver': 'End with FINAL: followed by the answer.',
            'critic': 'Return VERDICT: ACCEPT or REVISE for the proposed solution, then end with '
                      'FINAL: followed by the complete final answer.'},
 'domains': {'reasoning': {'title': 'Reasoning',
                           'benchmarks': ['StrategyQA', 'HotpotQA'],
                           'roles': [{'id': 'planner',
                                      'name': 'Reasoning planner',
                                      'instruction': 'Identify the facts, intermediate questions, '
                                                     'and logical dependencies needed to answer '
                                                     'the question. Distinguish supplied evidence '
                                                     'from assumptions and propose a concise '
                                                     'reasoning plan.'},
                                     {'id': 'solver',
                                      'name': 'Reasoner',
                                      'instruction': 'Use the plan and available evidence to solve '
                                                     'the question. Resolve the intermediate '
                                                     'questions and check that the conclusion '
                                                     'follows. Give yes or no for a boolean '
                                                     'question, or a short answer for a factual '
                                                     'question.'},
                                     {'id': 'critic',
                                      'name': 'Logic critic',
                                      'instruction': 'Check the proposed answer for missing '
                                                     'reasoning steps, contradictions, and '
                                                     'unsupported facts. Correct any identifiable '
                                                     'error and provide the final answer to the '
                                                     'original question, preserving the required '
                                                     'answer format.'}]},
             'qa': {'title': 'Knowledge QA',
                    'benchmarks': ['TriviaQA', 'PopQA'],
                    'roles': [{'id': 'planner',
                               'name': 'Question analyst',
                               'instruction': 'Identify the queried entity, relation, and '
                                              'disambiguating constraints. List the facts needed '
                                              'to resolve the question, separating known facts '
                                              'from uncertain assumptions.'},
                              {'id': 'solver',
                               'name': 'Answerer',
                               'instruction': 'Resolve the requested fact from the question, '
                                              "available context, and analyst's plan. Avoid adding "
                                              'unsupported details. Give the shortest unambiguous '
                                              'answer, using a conventional entity name where '
                                              'appropriate.'},
                              {'id': 'critic',
                               'name': 'Fact critic',
                               'instruction': 'Check entity identity, relation direction, dates, '
                                              'and other constraints against the proposed answer. '
                                              'Correct mismatches where possible. Return a concise '
                                              'final answer without appending an explanation to '
                                              'the answer string.'}]},
             'code': {'title': 'Code generation',
                      'benchmarks': ['KodCode', 'BigCodeBench'],
                      'roles': [{'id': 'planner',
                                 'name': 'Software planner',
                                 'instruction': 'Identify the required function signature, input '
                                                'and output constraints, allowed libraries, and '
                                                'edge cases. Propose an algorithm and relevant '
                                                'checks without assuming access to hidden tests.'},
                                {'id': 'solver',
                                 'name': 'Programmer',
                                 'instruction': 'Implement a complete Python solution from the '
                                                'specification and plan. Preserve the required '
                                                'interface and handle identified edge cases. Put '
                                                'the complete program in one fenced python block '
                                                'after FINAL:.'},
                                {'id': 'critic',
                                 'name': 'Code critic',
                                 'instruction': "Review the program's logic, API use, boundary "
                                                'cases, and compliance with the requested '
                                                'interface. Correct identifiable defects by '
                                                'inspection; do not claim unperformed execution. '
                                                'Put the complete final program in one fenced '
                                                'python block after FINAL:.'}]},
             'science_math': {'title': 'Science and mathematics',
                              'benchmarks': ['MATH-500', 'MMLU-Pro', 'GPQA'],
                              'roles': [{'id': 'planner',
                                         'name': 'Analytical planner',
                                         'instruction': 'Identify the governing concepts, '
                                                        'assumptions, equations, and relevant '
                                                        'constraints. For a choice question, '
                                                        'identify what distinguishes the options. '
                                                        'Propose a concise solution plan.'},
                                        {'id': 'solver',
                                         'name': 'Analytical solver',
                                         'instruction': 'Apply the plan to derive the answer and '
                                                        'check intermediate calculations or '
                                                        'scientific implications. For multiple '
                                                        'choice, end with the option identifier; '
                                                        'for mathematics, end with the final '
                                                        'expression in a boxed form.'},
                                        {'id': 'critic',
                                         'name': 'Verification critic',
                                         'instruction': 'Check assumptions, algebra, units, '
                                                        'logical consistency, and option '
                                                        'selection. Correct identifiable errors '
                                                        'and provide the complete final answer. '
                                                        'Use only the option identifier for '
                                                        'multiple choice or a boxed expression for '
                                                        'mathematics.'}]}}}

_DOMAIN_ROLE_TEMPLATES = {'kind': 'domain_role_prompt_templates',
 'execution_schedule': None,
 'scope': 'Role descriptions only; the frozen online reasoning run retains its recorded prompts '
          'and schedule.',
 'domains': {'reasoning': {'title': 'Reasoning',
                           'benchmarks': 'StrategyQA, HotpotQA',
                           'roles': [{'name': 'Planner',
                                      'instruction': 'Identify the key facts and decompose the '
                                                     'question into necessary reasoning steps.'},
                                     {'name': 'Reasoner',
                                      'instruction': 'Derive an answer from available evidence. '
                                                     'State any unresolved assumptions.'},
                                     {'name': 'Critic',
                                      'instruction': 'Check the inference, correct contradictions, '
                                                     'and return a concise answer.'}]},
             'qa': {'title': 'Knowledge QA',
                    'benchmarks': 'TriviaQA, PopQA',
                    'roles': [{'name': 'Analyst',
                               'instruction': 'Identify the target entity, relation, and '
                                              'disambiguating constraints.'},
                              {'name': 'Answerer',
                               'instruction': 'Resolve the requested fact and give the shortest '
                                              'unambiguous answer.'},
                              {'name': 'Verifier',
                               'instruction': 'Check entity identity and factual consistency; '
                                              'correct unsupported answers.'}]},
             'code': {'title': 'Code generation',
                      'benchmarks': 'KodCode, BigCodeBench',
                      'roles': [{'name': 'Analyst',
                                 'instruction': 'Extract the required interface, input-output '
                                                'constraints, and allowed libraries.'},
                                {'name': 'Designer',
                                 'instruction': 'Propose an algorithm and identify its complexity '
                                                'and boundary cases.'},
                                {'name': 'Programmer',
                                 'instruction': 'Implement a complete Python solution that '
                                                'satisfies the specification.'},
                                {'name': 'Reviewer',
                                 'instruction': 'Inspect correctness, API use, and edge cases; '
                                                'propose concrete fixes.'},
                                {'name': 'Tester',
                                 'instruction': 'Design checks from the specification and public '
                                                'examples. Distinguish expected from executed '
                                                'results.'}]},
             'science_math': {'title': 'Science and mathematics',
                              'benchmarks': 'MATH-500, MMLU-Pro, GPQA',
                              'roles': [{'name': 'Analyst',
                                         'instruction': 'Identify relevant principles, '
                                                        'assumptions, equations, and constraints.'},
                                        {'name': 'Solver',
                                         'instruction': 'Derive the answer and make the essential '
                                                        'calculations explicit.'},
                                        {'name': 'Verifier',
                                         'instruction': 'Check the derivation independently, '
                                                        'including units, logic, and limiting '
                                                        'cases.'},
                                        {'name': 'Integrator',
                                         'instruction': 'Resolve disagreements and return the '
                                                        'final option or mathematical '
                                                        'expression.'}]}}}

def domain_prompts():
    return deepcopy(_DOMAIN_PROMPTS)


def role_templates():
    return deepcopy(_DOMAIN_ROLE_TEMPLATES)
