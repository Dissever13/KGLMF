
"""
Generate fig_efficiency.pdf — Accuracy vs Latency Trade-off Scatter
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Data
# ============================================================

POINTS = [
    ('NoMemory',   96.17, 2.6, 140),
    ('TextMemory', 97.36, 2.9, 180),
    ('KG-LMF',     98.50, 1.5, 240),
]

# ============================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'output'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [
    '#2E86AB',
    '#A23B72',
    '#F18F01'
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(6.4, 4.6))

# 手动标签偏移
offsets = { 'NoMemory': (0.08, -0.12), 'TextMemory': (-0.08, 0.13), 'KG-LMF': (0.14, 0.10), }

for i, (name, acc, lat, size) in enumerate(POINTS):

    color = COLORS[i]

    ax.scatter(
        lat,
        acc,
        s=size,
        color=color,
        alpha=0.90,
        edgecolors='white',
        linewidth=2,
        zorder=5
    )

    dx, dy = offsets[name]

    ax.text(
        lat + dx,
        acc + dy,
        name,
        fontsize=9,
        fontweight='bold',
        color=color
    )

# 坐标轴范围收紧
ax.set_xlim(1.2, 3.2)
ax.set_ylim(95.5, 99.2)

ax.set_xlabel('Average Latency (seconds)')
ax.set_ylabel('Average Accuracy (%)')

ax.set_title(
    'Accuracy vs. Latency Trade-off',
    fontweight='bold'
)

# 去掉图例
# ax.legend()

# 网格
ax.grid(
    linestyle='--',
    alpha=0.25
)

# 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out_path = os.path.join(
    OUTPUT_DIR,
    'fig_efficiency.pdf'
)

fig.savefig(
    out_path,
    format='pdf'
)

plt.close()

print(f'Saved: {out_path}')

