"""
Shared data loading with deterministic train/test split.

Guarantees zero overlap between training and testing:
  - All samples are shuffled with master seed
  - First 101 samples → training
  - Next 81 samples → testing
  - Works for all 9 datasets

FIX #4 (REVISED): Training and test use the SAME datasets but DIFFERENT,
deterministically non-overlapping sample splits. This is standard ML practice
and ensures the composer learns domain-relevant patterns.
"""

import os, json, random
from typing import List, Dict

# All 9 benchmark datasets
DATASET_CONFIG = {
    'strategyqa':    {'task_type': 'reasoning', 'metric': 'accuracy',
                      'file': 'strategyqa_real.json'},
    'triviaqa':      {'task_type': 'qa',        'metric': 'exact_match',
                      'file': 'trivia_qa_real.json'},
    'popqa':         {'task_type': 'qa',        'metric': 'exact_match',
                      'file': 'akariasai_PopQA_real.json'},
    'hotpotqa':      {'task_type': 'reasoning', 'metric': 'accuracy',
                      'file': 'HotpotQA_real.json'},
    'bigcodebench':  {'task_type': 'code',      'metric': 'pass_at_1',
                      'file': 'bigcodebench_real.json'},
    'kodcode':       {'task_type': 'code',      'metric': 'pass_at_1',
                      'file': 'kodcode_real.json'},
    'math500':       {'task_type': 'math',      'metric': 'math_accuracy',
                      'file': 'math500_real.json'},
    'mmlu_pro':      {'task_type': 'science',   'metric': 'science_accuracy',
                      'file': 'mmlu_pro_real.json'},
    'gpqa':          {'task_type': 'science',   'metric': 'science_accuracy',
                      'file': 'gpqa_real.json'},
}

# Master seed for reproducible splits
MASTER_SEED = 42
TRAIN_SIZE = 101
TEST_SIZE = 81

# Path resolution
def _get_data_dir():
    """Find data directory relative to this file or project root."""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'data'),
        os.path.join(os.path.dirname(__file__), '..', 'data'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'data'),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return os.path.join(os.path.dirname(__file__), '..', 'data')

DATA_DIR = _get_data_dir()


def load_split(dataset_name: str, split: str = 'train') -> List[Dict]:
    """Load a deterministic, non-overlapping split of a dataset.

    Args:
        dataset_name: key in DATASET_CONFIG
        split: 'train' (first 101 samples) or 'test' (next 81 samples)

    Returns:
        List of {question, answer, task_type, metric} dicts

    Guarantees:
        - Zero overlap between train and test splits
        - Reproducible with fixed master seed
        - Same splits every time for a given dataset+split
    """
    config = DATASET_CONFIG[dataset_name]
    fpath = os.path.join(DATA_DIR, config['file'])
    if not os.path.exists(fpath):
        print(f"  [WARN] Data file not found: {fpath}")
        return []

    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Parse raw data into uniform format
    all_samples = []
    for item in data:
        q = item.get('question') or item.get('prompt') or item.get('input', '')
        a = item.get('answer') or item.get('target') or item.get('output', '')
        if not a and 'answer_key' in item:
            a = item['answer_key']
        if q and str(a):
            all_samples.append({
                'question': str(q),
                'answer': str(a),
                'task_type': config['task_type'],
                'metric': config['metric'],
            })

    # Deterministic shuffle
    rng = random.Random(MASTER_SEED)
    rng.shuffle(all_samples)

    # Deterministic split — prioritize test size (81), use remainder for train
    total_needed = TRAIN_SIZE + TEST_SIZE  # 182
    if len(all_samples) < total_needed:
        # Test gets priority: 81, or 50% if very small
        n_test = min(TEST_SIZE, max(40, len(all_samples) // 2))
        n_train = min(TRAIN_SIZE, len(all_samples) - n_test)
    else:
        n_train = TRAIN_SIZE
        n_test = TEST_SIZE

    if split == 'train':
        return all_samples[:n_train]
    elif split == 'test':
        return all_samples[n_train:n_train + n_test]
    else:
        raise ValueError(f"Unknown split: {split}")


def load_all_train() -> List[Dict]:
    """Load training splits from ALL 9 datasets."""
    all_samples = []
    for ds_name in DATASET_CONFIG:
        samples = load_split(ds_name, 'train')
        for s in samples:
            s['source'] = ds_name
        all_samples.extend(samples)
        print(f"  {ds_name}: {len(samples)} train samples")
    print(f"  Total training: {len(all_samples)} samples "
          f"(from {len(DATASET_CONFIG)} datasets, non-overlapping with test)")
    return all_samples


def load_dataset_test(dataset_name: str) -> List[Dict]:
    """Load test split for one dataset (for experiment script)."""
    return load_split(dataset_name, 'test')
