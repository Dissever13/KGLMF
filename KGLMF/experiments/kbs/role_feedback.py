"""Role-local feedback after durable first answers; unavailable is not zero."""
import json
import re

RUBRICS = {
    'planner': ('relevant decomposition', 'logically feasible steps', 'concise actionable plan'),
    'analyst': ('correct interpretation of the specification', 'identification of relevant constraints', 'actionable analysis'),
    'designer': ('algorithmic correctness', 'coverage of constraints and edge cases', 'implementable design'),
    'reviewer': ('identification of actual defects', 'validity of suggested corrections', 'coverage of specification'),
    'tester': ('tests follow the public specification', 'useful boundary and failure coverage', 'correct expected outcomes without claiming execution'),
    'solver': ('valid reasoning', 'use of available evidence', 'requested answer format'),
    'answerer': ('valid reasoning', 'use of available evidence', 'requested answer format'),
    'programmer': ('specification compliance', 'algorithmic correctness', 'complete executable solution'),
    'critic': ('sound review of the preceding proposal', 'justified acceptance or correction', 'requested verdict and answer format'),
    'verifier': ('sound verification of the preceding proposal', 'justified acceptance or correction', 'requested verdict and answer format'),
    'integrator': ('resolution of disagreements', 'evidence-consistent synthesis', 'requested final answer format'),
}
ANSWER_ROLES = {'solver', 'answerer', 'programmer', 'critic', 'verifier', 'integrator'}


def review_consistency(response, previous_score):
    if previous_score is None:
        raise ValueError('A review requires its preceding answer verdict')
    match = re.search(r'(?im)^\s*VERDICT:\s*(ACCEPT|REVISE)\s*$', response)
    decision = match[1].upper() if match else None
    return bool(decision and (previous_score == 1 or decision == 'REVISE')), decision


def parse_role_quality(text):
    try:
        value = json.loads(text[text.index('{'):text.rindex('}') + 1])
        scores = value['criteria']
        if not isinstance(scores, list) or len(scores) != 3 or any(type(s) is not int or s not in (0, 1, 2) for s in scores):
            raise ValueError('Three integer rubric scores required')
        quality = sum(scores) / 6
        return dict(score=quality, criteria=scores, accepted=quality >= 5/6,
                    corrected_response=None, status='parsed', source='role_rubric_internal')
    except (KeyError, ValueError, TypeError):
        return dict(score=None, criteria=None, accepted=None, corrected_response=None,
                    status='unavailable_invalid_json', source='role_rubric_internal')


def routing_advantage_loss(mass, delta):
    """Signed, bounded-reward binary objective; zero advantage has zero gradient."""
    mass = mass.clamp(1e-6, 1-1e-6)
    return -max(delta, 0.) * mass.log() - max(-delta, 0.) * (1-mass).log()


def role_training_target(role, response, internal, verified, reference=None):
    # Never teach a planner/reviewer to emit the benchmark answer instead of its role.
    if role not in ANSWER_ROLES:
        return (response, 'role_rubric_accepted') if internal['accepted'] is True else (None, None)
    if role in {'solver', 'answerer', 'integrator'} and reference is not None:
        return reference, 'post_answer_benchmark_reference'
    # A critic's verdict is not verified by exact-match answer scoring alone.
    if verified == 1 and internal['accepted'] is True:
        return response, ('verified_review_response' if internal.get('status')=='verified_review'
                          else 'role_rubric_and_benchmark_verified')
    return None, None
