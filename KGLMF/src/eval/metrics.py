from typing import List, Dict
import re

def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == str(ground_truth).strip().lower() else 0.0

def f1_score_fn(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(str(ground_truth).lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    intersection = pred_tokens & truth_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def pass_at_k(code: str, test_cases: List[Dict], k: int = 1) -> float:
    import subprocess, tempfile, os
    if not test_cases:
        return 0.0
    passed = 0
    tmpf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    try:
        tmpf.write(code)
        tmpf.flush()
        fname = tmpf.name
        tmpf.close()
        for tc in test_cases:
            try:
                result = subprocess.run(
                    ['python', fname],
                    input=str(tc.get('input', '')),
                    capture_output=True, text=True, timeout=30)
                expected = str(tc.get('expected', tc.get('output', '')))
                if result.stdout.strip() == expected.strip():
                    passed += 1
            except:
                pass
    finally:
        try:
            os.unlink(tmpf.name)
        except:
            pass
    return passed / len(test_cases)

def planning_success_rate(plan: str, valid_actions: List[str]) -> float:
    plan_actions = set(re.findall(r'\((\w+(?:-\w+)*)', plan))
    valid = set(valid_actions)
    if not valid:
        return 0.0
    return len(plan_actions & valid) / len(valid)
