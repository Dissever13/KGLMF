"""
Reproduce fig_backbone_comparison.pdf — 3-Backbone accuracy heatmap.

Usage:
    python gen_fig_backbone_comparison.py

Modify DATA below to change cell values. Deltas auto-computed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# >>>>> MODIFY VALUES HERE <<<<<
# ============================================================
DATASETS = ['TriviaQA', 'StrategyQA', 'PopQA', 'KodCode', 'BigCodeB.', 'PDDL', 'HotpotQA']

# 3 backbone pairs, each: (NoMemory label, +KG-LMF label)
BACKBONE_GROUPS = [
    ('DeepSeek\nV4 Pro', 'DeepSeek\n+KG-LMF'),
    ('Llama3.1\n8B',    'Llama3.1\n+KG-LMF'),
    ('Qwen3.5\n9B',     'Qwen3.5\n+KG-LMF'),
]

# Group names displayed below the heatmap
BACKBONE_NAMES = ['DeepSeek V4 Pro', 'Llama3.1:8B', 'Qwen3.5:9B']

# 7 rows x 6 columns: [NoMem, +KG] repeated per backbone, row = dataset
DATA = [
    [ 95.13,  97.10,   77.36,  78.02,   85.24,  90.53],   # TriviaQA
    [ 96.15,  99.01,   84.00,  84.33,    85.10,  95.74],   # StrategyQA
    [ 93.77,  95.05,  0, 0,  0, 0],   # PopQA
    [90.26,  94.06,  0, 0,    0,   0],   # KodCode
    [ 73.49,  80.40,  0, 0,    0,   0],   # BigCodeB.
    [  65.22,   72.98,   49.17,  50.96,    0,   0],   # PDDL
    [ 56.01,  66.07,   26.58,  27.24,   50.51,  63.25],   # HotpotQA
]

# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

data = np.array(DATA)
n_rows, n_cols = data.shape

# Compute deltas: +KG - NoMem for each backbone pair
deltas = []
for bb in range(3):
    deltas.append(data[:, bb * 2 + 1] - data[:, bb * 2])

# --- Figure ---
fig, ax = plt.subplots(figsize=(9, 5.2))

# Heatmap — no gaps between cells
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100,
               interpolation='nearest')

# Cell values
for r in range(n_rows):
    for c in range(n_cols):
        val = data[r, c]
        color = 'white' if val <= 50 else 'black'
        ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)

# --- Title at VERY TOP ---
ax.text(n_cols / 2 - 0.5, -1.55,
        'Three-Backbone Comparison: NoMemory vs KG-LMF Accuracy (%)',
        ha='center', va='top', fontsize=11, fontweight='bold', color='#111111')

# --- Sub-column labels at TOP of heatmap ---
for c in range(n_cols):
    bb = c // 2          # which backbone group
    sub = c % 2          # 0=NoMem, 1=+KG
    label = BACKBONE_GROUPS[bb][sub]
    ax.text(c, -0.55, label, ha='center', va='bottom', fontsize=7.5,
            fontweight='bold', color='#333333')

# --- Row labels on the left ---
for r, ds in enumerate(DATASETS):
    ax.text(-0.55, r, ds, ha='right', va='center', fontsize=8.5, color='#333333')

# --- Delta annotations between sub-column pairs ---
for bb in range(3):
    for r in range(n_rows):
        d = deltas[bb][r]
        if d != 0:
            x_center = bb * 2 + 0.5
            ax.annotate(f'{d:+.1f}', xy=(x_center, r - 0.42),
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', color='#333333')

# --- Backbone group names at BOTTOM of heatmap ---
for bb, name in enumerate(BACKBONE_NAMES):
    ax.text(bb * 2 + 0.5, n_rows - 0.05, name, ha='center', va='top',
            fontsize=8.5, fontweight='bold', color='#333333')

# --- Color bar (smaller) ---
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015,
                     ticks=[0, 20, 40, 60, 80, 100],
                     shrink=0.70)
cbar.ax.tick_params(labelsize=7, width=0.5)
cbar.outline.set_linewidth(0.5)
cbar.set_label('Accuracy (%)', fontsize=8, labelpad=1)

# --- Clean up ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1.5, n_cols + 0.2)
ax.set_ylim(n_rows + 0.8, -1.8)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'fig_backbone_comparison.pdf')
fig.savefig(out_path, format='pdf', dpi=200)
plt.close()
print(f'Saved: {out_path}')
