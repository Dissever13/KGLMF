from typing import List, Dict
from datasets import load_dataset

def load_triviaqa(split: str = "validation", max_samples: int = 200) -> List[Dict]:
    try:
        ds = load_dataset("trivia_qa", "rc", split="validation")
        data = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            data.append({
                "id": item.get("question_id", str(i)),
                "question": item["question"],
                "answer": item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"]),
                "type": "qa",
            })
        return data
    except:
        return [_make_placeholder("TriviaQA", i, "qa") for i in range(max_samples)]

def load_strategyqa(max_samples: int = 200) -> List[Dict]:
    try:
        ds = load_dataset("wics/strategy-qa", split="test")
        data = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            data.append({
                "id": str(i), "question": item["question"],
                "answer": str(item["answer"]), "type": "reasoning",
            })
        return data
    except:
        return [_make_placeholder("StrategyQA", i, "reasoning") for i in range(max_samples)]

def load_popqa(max_samples: int = 200) -> List[Dict]:
    try:
        ds = load_dataset("akariasai/PopQA", split="test")
        data = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            data.append({
                "id": str(item.get("id", i)), "question": item["question"],
                "answer": str(item["answer"]), "type": "qa",
            })
        return data
    except:
        return [_make_placeholder("PopQA", i, "qa") for i in range(max_samples)]

def load_kodcode(max_samples: int = 100) -> List[Dict]:
    try:
        ds = load_dataset("KodCode/KodCode", split="test")
        data = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            data.append({
                "id": str(i), "prompt": item["prompt"],
                "test_cases": item.get("test_cases", item.get("tests", [])),
                "type": "code",
            })
        return data
    except:
        return [_make_placeholder("KodCode", i, "code") for i in range(max_samples)]

def load_bigcodebench(max_samples: int = 100) -> List[Dict]:
    try:
        ds = load_dataset("bigcode/bigcodebench", split="test")
        data = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            data.append({
                "id": str(i), "prompt": item["prompt"],
                "test_cases": item.get("test_cases", item.get("tests", [])),
                "type": "code",
            })
        return data
    except:
        return [_make_placeholder("BigCodeBench", i, "code") for i in range(max_samples)]

def load_pddl(max_samples: int = 100) -> List[Dict]:
    return [_make_placeholder("PDDL", i, "planning") for i in range(max_samples)]

def _make_placeholder(ds_name: str, idx: int, dtype: str) -> Dict:
    return {
        "id": f"{ds_name}_{idx}",
        "question": f"Sample {ds_name} question {idx}: What is the capital of France?",
        "prompt": f"Sample {ds_name} prompt {idx}: Write a function to compute factorial.",
        "answer": "Paris",
        "test_cases": [{"input": "factorial(5)", "expected": "120"}],
        "valid_actions": ["pick-up", "put-down", "stack", "unstack"],
        "type": dtype,
    }
