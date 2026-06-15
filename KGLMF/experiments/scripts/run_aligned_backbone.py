"""Aligned backbone comparison: same datasets, same samples, both backbones."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.api.client import DeepSeekClient
from src.api.ollama_client import OllamaClient

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT = os.path.join(RESULTS_DIR, 'aligned_backbone.json')

SYS = "Answer directly in one sentence. No explanation. Just the answer."

def load_data(name, n):
    for ext in ['_real.json', '.json']:
        path = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)[:n]
    return []

def run_baseline(api, questions):
    correct = 0
    latencies = []
    for item in questions:
        q = item.get('question') or item.get('prompt', '')
        gt = str(item.get('answer', ''))
        t0 = time.time()
        try:
            resp = api.generate(SYS, q, max_tokens=512)
        except Exception as e:
            resp = ''
        lat = time.time() - t0
        score = 1.0 if gt.lower() in resp.lower() else 0.0
        correct += score
        latencies.append(lat)
    n = len(questions)
    acc = round(correct / n, 4) if n else 0
    avg_lat = round(sum(latencies) / n, 2) if n else 0
    return {'accuracy': acc, 'avg_latency': avg_lat, 'n': n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--datasets', nargs='*', default=['TriviaQA','StrategyQA','PopQA'])
    parser.add_argument('--backbones', nargs='*', default=['deepseek','qwen'])
    args = parser.parse_args()

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            results = json.load(f)

    backbones = {}
    if 'deepseek' in args.backbones:
        backbones['DeepSeek-V4-Pro'] = DeepSeekClient(max_tokens=512)
    if 'qwen' in args.backbones:
        try:
            qwen = OllamaClient(model='qwen3.5:9b', max_tokens=4096)
            test = qwen.generate(SYS, 'Say hi')
            if test:
                backbones['Qwen3.5-9B'] = qwen
                print('Qwen test OK')
            else:
                print('Qwen returned empty, skipping')
        except Exception as e:
            print('Qwen unavailable: ' + str(e))

    print('Backbones: ' + str(list(backbones.keys())))
    print('Datasets: ' + str(args.datasets) + ', N=' + str(args.n))
    print('=' * 60)

    for ds_name in args.datasets:
        data = load_data(ds_name, args.n)
        if not data:
            print('  No data for ' + ds_name)
            continue
        dtype = data[0].get('type', 'qa') if data else 'qa'
        print('')
        print(ds_name + ' (' + str(len(data)) + ' samples, type=' + dtype + '):')

        for bb_name, api in backbones.items():
            key = ds_name + '/' + bb_name
            if key in results and results[key].get('n', 0) >= args.n:
                r = results[key]
                print('  ' + bb_name + ' DONE acc=' + str(r['accuracy']) + ' lat=' + str(r['avg_latency']) + 's')
                continue

            print('  ' + bb_name + ' running...', end=' ', flush=True)
            r = run_baseline(api, data)
            r['method'] = 'NoMemory'
            r['dataset'] = ds_name
            r['backbone'] = bb_name
            results[key] = r
            with open(OUTPUT, 'w') as f:
                json.dump(results, f, indent=2)
            print('acc=' + str(r['accuracy']) + ' lat=' + str(r['avg_latency']) + 's')

    # Summary
    sep = '=' * 70
    print('')
    print(sep)
    print('BACKBONE COMPARISON (' + str(len(results)) + ' experiments)')
    print(sep)
    header = '%-15s %-20s %-20s' % ('Dataset', 'DeepSeek-V4-Pro', 'Qwen3.5-9B')
    print(header)
    print('-' * 55)
    for ds in args.datasets:
        ds_key = ds + '/DeepSeek-V4-Pro'
        qw_key = ds + '/Qwen3.5-9B'
        ds_acc = results.get(ds_key, {}).get('accuracy', 0)
        qw_acc = results.get(qw_key, {}).get('accuracy', 0)
        print('%-15s %-20s %-20s' % (ds, str(ds_acc), str(qw_acc)))

    print('')
    print('Saved to ' + OUTPUT)

if __name__ == '__main__':
    main()
