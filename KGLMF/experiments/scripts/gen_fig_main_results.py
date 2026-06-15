"""
Generate fig_main_results.pdf — Main Results Bar Chart.

Usage:
    python gen_fig_main_results.py

Modify the DATA dict below to change values. No external dependencies beyond matplotlib + numpy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ============================================================
# >>>>> MODIFY VALUES HERE <<<<<
# ============================================================
# Each dataset: list of scores in the same order as METHODS
METHODS = ['NoMemory', 'TextMemory', 'KG-LMF']

DATA = {
    'TriviaQA':  [95.05, 96.04, 97.10],
    'StrategyQA': [96.04, 97.03, 99.01],
    'PopQA':      [97.03, 99.01, 95.05],
    'KodCode':    [90.38, 95.65, 94.06],
    'BigCodeBench':  [78.02, 80.20, 80.40],
    'PDDL':       [51.67,   68.32, 72.98],
}

# ============================================================
# Plotting
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ['#2E86AB', '#A23B72', '#F18F01']

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 6,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

datasets = list(DATA.keys())
x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, method in enumerate(METHODS):
    values = [DATA[d][i] for d in datasets]
    bars = ax.bar(x + i * width, values, width, label=method,
                  color=COLORS[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=5, fontweight='bold')

ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.annotate('KG-LMF\nBest', xy=(1 + width, DATA['StrategyQA'][2]),
#             xytext=(1 + width, 108), ha='center', fontsize=8, fontweight='bold',
#             color=COLORS[2], arrowprops=dict(arrowstyle='->', color=COLORS[2], lw=1.5))

ax.set_title('Main Results Across Six Benchmarks', fontweight='bold')
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'fig_main_results.pdf')
fig.savefig(out_path, format='pdf')
plt.close()
print(f'Saved: {out_path}')
