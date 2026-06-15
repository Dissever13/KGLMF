#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical significance tests for KG-LMF experiment results.

Computes:
  1. Bootstrap 95% confidence intervals for accuracy
  2. Paired t-test between methods (per-sample scores)
  3. McNemar's test for binary correct/incorrect outcomes
  4. Effect size (Cohen's d) between methods
  5. Wilcoxon signed-rank test (non-parametric alternative)

Usage:
    /d/WorkSoftware/python D:/KGLM/KGLMF/experiments/scripts/statistical_tests.py \
        --baseline results/triviaqa_nomemory_n81.json \
        --treatment results/triviaqa_kglmf_n81.json \
        --output stats/
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats
from datetime import datetime


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000,
                 ci_level: float = 0.95, seed: int = 42) -> dict:
    """Compute bootstrap confidence interval for mean accuracy.

    Args:
        scores: binary (0/1) per-sample scores
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level (default 0.95)

    Returns dict with mean, ci_lower, ci_upper, std_err
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        means[i] = scores[idx].mean()

    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(means, 100 * alpha)
    ci_upper = np.percentile(means, 100 * (1 - alpha))

    return {
        'mean': float(scores.mean()),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci_level,
        'std_err': float(scores.std(ddof=1) / np.sqrt(n)),
        'n_samples': n,
    }


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired t-test between two methods on the same samples.

    H0: mean(scores_a - scores_b) = 0 (no difference)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    diff = scores_a - scores_b
    return {
        'test': 'paired t-test',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std(ddof=1)),
        'n_pairs': len(scores_a),
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test for paired binary outcomes.

    Compares discordant pairs: (correct_a=1, correct_b=0) vs (correct_a=0, correct_b=1)
    """
    both_correct = np.sum((correct_a == 1) & (correct_b == 1))
    both_wrong = np.sum((correct_a == 0) & (correct_b == 0))
    a_only = np.sum((correct_a == 1) & (correct_b == 0))
    b_only = np.sum((correct_a == 0) & (correct_b == 1))

    # McNemar chi-squared with continuity correction
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {'test': "McNemar's test", 'p_value': 1.0, 'significant_05': False,
                'note': 'No discordant pairs'}

    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_discordant
    p_value = 1 - stats.chi2.cdf(chi2, 1)

    return {
        'test': "McNemar's test",
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'both_correct': int(both_correct),
        'both_wrong': int(both_wrong),
        'a_only_correct': int(a_only),
        'b_only_correct': int(b_only),
        'n_discordant': int(n_discordant),
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Cohen's d effect size for paired samples."""
    diff = scores_a - scores_b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

    # Interpret
    if abs(d) < 0.2:
        magnitude = 'negligible'
    elif abs(d) < 0.5:
        magnitude = 'small'
    elif abs(d) < 0.8:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return {
        'cohens_d': float(d),
        'magnitude': magnitude,
        'interpretation': f'{magnitude} effect size (d={d:.3f})',
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    diff = scores_a - scores_b
    non_zero = diff[diff != 0]

    if len(non_zero) == 0:
        return {'test': 'Wilcoxon signed-rank', 'p_value': 1.0,
                'significant_05': False, 'note': 'All differences are zero'}

    statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
    return {
        'test': 'Wilcoxon signed-rank',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'n_nonzero_diffs': len(non_zero),
    }


def compare_methods(baseline_file: str, treatment_file: str) -> dict:
    """Run full statistical comparison between two methods.

    Args:
        baseline_file: JSON results file for baseline method
        treatment_file: JSON results file for treatment (KG-LMF)

    Returns comprehensive report dict.
    """
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    with open(treatment_file, 'r', encoding='utf-8') as f:
        treatment = json.load(f)

    base_scores = np.array([r['score'] for r in baseline['results']])
    treat_scores = np.array([r['score'] for r in treatment['results']])

    # Ensure same length
    n = min(len(base_scores), len(treat_scores))
    base_scores = base_scores[:n]
    treat_scores = treat_scores[:n]

    # Binary for McNemar
    base_binary = (base_scores >= 0.5).astype(int)
    treat_binary = (treat_scores >= 0.5).astype(int)

    report = {
        'baseline': {
            'file': os.path.basename(baseline_file),
            'accuracy': float(base_scores.mean()),
            'n': len(base_scores),
        },
        'treatment': {
            'file': os.path.basename(treatment_file),
            'accuracy': float(treat_scores.mean()),
            'n': len(treat_scores),
        },
        'delta_accuracy': float(treat_scores.mean() - base_scores.mean()),
        'bootstrap_baseline': bootstrap_ci(base_scores),
        'bootstrap_treatment': bootstrap_ci(treat_scores),
        'paired_ttest': paired_ttest(treat_scores, base_scores),
        'mcnemar': mcnemar_test(treat_binary, base_binary),
        'cohens_d': cohens_d(treat_scores, base_scores),
        'wilcoxon': wilcoxon_test(treat_scores, base_scores),
        'timestamp': datetime.now().isoformat(),
    }
    return report


def print_report(report: dict):
    """Pretty-print statistical report."""
    b = report['baseline']
    t = report['treatment']

    print(f"\n{'='*70}")
    print(f"STATISTICAL COMPARISON REPORT")
    print(f"{'='*70}")
    print(f"Baseline:  {b['file']}  (acc={b['accuracy']:.4f}, n={b['n']})")
    print(f"Treatment: {t['file']}  (acc={t['accuracy']:.4f}, n={t['n']})")
    print(f"Delta:     {report['delta_accuracy']:+.4f} ({report['delta_accuracy']/b['accuracy']*100:+.1f}% relative)")
    print()

    # Bootstrap CIs
    b_ci = report['bootstrap_baseline']
    t_ci = report['bootstrap_treatment']
    print(f"Bootstrap 95% CI:")
    print(f"  Baseline:  [{b_ci['ci_lower']:.4f}, {b_ci['ci_upper']:.4f}]")
    print(f"  Treatment: [{t_ci['ci_upper']:.4f}, {t_ci['ci_upper']:.4f}]")
    # Check CI overlap
    overlap = b_ci['ci_upper'] >= t_ci['ci_lower']
    print(f"  {'NO OVERLAP -- significant difference' if not overlap else 'Overlap -- may not be significant'}")
    print()

    # Paired t-test
    tt = report['paired_ttest']
    print(f"Paired t-test: t={tt['t_statistic']:.3f}, p={tt['p_value']:.4f}")
    print(f"  Significant at 0.05: {'YES' if tt['significant_05'] else 'NO'}")
    print(f"  Significant at 0.01: {'YES' if tt['significant_01'] else 'NO'}")
    print(f"  Mean diff: {tt['mean_diff']:+.4f} (std={tt['std_diff']:.4f})")
    print()

    # Effect size
    cd = report['cohens_d']
    print(f"Cohen's d: {cd['cohens_d']:.3f} ({cd['magnitude']})")
    print()

    # McNemar
    mc = report['mcnemar']
    print(f"McNemar's test: chi2={mc.get('chi2_statistic', 'N/A')}, p={mc['p_value']:.4f}")
    print(f"  Discordant pairs: {mc['n_discordant']}")
    print(f"  Both correct: {mc['both_correct']}, Both wrong: {mc['both_wrong']}")
    print(f"  Treatment only correct: {mc['a_only_correct']}, "
          f"Baseline only correct: {mc['b_only_correct']}")
    print()

    # Wilcoxon
    wx = report['wilcoxon']
    print(f"Wilcoxon signed-rank: W={wx.get('statistic', 'N/A')}, p={wx['p_value']:.4f}")
    print(f"  Significant at 0.05: {'YES' if wx['significant_05'] else 'NO'}")
    print()

    # Verdict
    print(f"{'='*70}")
    sig_count = sum([
        tt['significant_05'],
        mc['significant_05'],
        wx['significant_05'],
        not overlap,
    ])
    if sig_count >= 3:
        verdict = "STATISTICALLY SIGNIFICANT — treatment reliably outperforms baseline"
    elif sig_count >= 2:
        verdict = "LIKELY SIGNIFICANT — multiple tests indicate a real difference"
    elif sig_count >= 1:
        verdict = "WEAK EVIDENCE — one test suggests difference, need more data"
    else:
        verdict = "NOT SIGNIFICANT — no reliable difference detected"
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True,
                       help='JSON results for baseline method (e.g., NoMemory)')
    parser.add_argument('--treatment', type=str, required=True,
                       help='JSON results for treatment method (e.g., KG-LMF)')
    parser.add_argument('--output_dir', type=str, default='stats/')
    args = parser.parse_args()

    report = compare_methods(args.baseline, args.treatment)
    print_report(report)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.baseline))[0]
    treat_name = os.path.splitext(os.path.basename(args.treatment))[0]
    out_path = os.path.join(args.output_dir,
        f"stats_{base_name}_vs_{treat_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {out_path}")


if __name__ == '__main__':
    main()
