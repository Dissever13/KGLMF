#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KG-LMF Architecture Diagram - High-quality vector PDF via matplotlib.
Clean ASCII-safe text. All boxes sized to fit. Arrows properly routed.

Usage:
    /d/WorkSoftware/python D:/KGLM/KGLMF/experiments/scripts/gen_architecture_fig.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionStyle, ArrowStyle
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FIG_W, FIG_H = 18.5, 10.5

# Colors
P1_BG = '#E3F2FD'; P1_BD = '#1565C0'
P2_BG = '#FFF3E0'; P2_BD = '#E65100'
P3_BG = '#E8F5E9'; P3_BD = '#2E7D32'
C_GRAPH  = '#42A5F5'; C_COMP = '#FF9800'; C_ROUT = '#FF7043'
C_INJECT = '#66BB6A'; C_EVOLVE = '#AB47BC'; C_LLM = '#78909C'
C_AGENT  = '#26C6DA'; C_TRAJ = '#80DEEA'; C_OUT = '#FAFAFA'
C_ARR    = '#455A64'; C_DARK = '#212121'; C_MED = '#616161'
C_LOOP   = '#8E24AA'; C_WHITE = '#FFFFFF'

# Layout
P1_X, P1_W = 0.4, 5.6
P2_X, P2_W = 6.3, 5.5
P3_X, P3_W = 12.1, 6.0
PHASE_Y, PHASE_H = 1.6, 7.8

RAD = 0.10
PY = 0.10
LH = 0.24


def bx(ax, x, y, w, h, fc, ec, lw=1.5, r=RAD, z=5, a=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=a)
    ax.add_patch(p)

def ar(ax, x1, y1, x2, y2, c=C_ARR, w=1.8, z=4, ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=ArrowStyle('->', head_length=7, head_width=5),
                        connectionstyle=ConnectionStyle("arc3,rad=0.0"),
                        color=c, linewidth=w, linestyle=ls, zorder=z)
    ax.add_patch(a)

def tb(ax, x, y, lines, bw, fc, ec, lw=1.5, r=RAD, z=6):
    """Draw text box. lines: [(text, size, color, bold), ...]"""
    n = len(lines)
    bh = n * LH + 2 * PY
    bx(ax, x, y, bw, bh, fc, ec, lw, r, z)
    sy = y + bh - PY - LH * 0.35
    for i, (txt, fs, tc, bd) in enumerate(lines):
        ty = sy - i * LH
        kw = dict(fontsize=fs, ha='center', va='center', color=tc, zorder=20)
        if bd: kw['fontweight'] = 'bold'
        ax.text(x + bw/2, ty, txt, **kw)
    return (x, y, bw, bh)


def draw():
    print("KG-LMF Architecture Diagram")
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.axis('off')

    # ---- TITLE ----
    ax.text(FIG_W/2, FIG_H - 0.45,
            'KG-LMF: Knowledge Graph-Driven Latent Memory Framework',
            fontsize=21, fontweight='bold', ha='center', color=C_DARK, zorder=100)
    ax.text(FIG_W/2, FIG_H - 0.95,
            'Multi-Agent Test-Time Scaling Architecture',
            fontsize=14, ha='center', color=C_MED, style='italic', zorder=100)

    # ---- PHASE BGs ----
    for px, pw, fc, ec, lb in [
        (P1_X, P1_W, P1_BG, P1_BD, 'Phase 1: Graph Construction'),
        (P2_X, P2_W, P2_BG, P2_BD, 'Phase 2: Retrieval & Composition'),
        (P3_X, P3_W, P3_BG, P3_BD, 'Phase 3: Injection & Evolution'),
    ]:
        bx(ax, px, PHASE_Y, pw, PHASE_H, fc, ec, 2.8, 0.2, 1, 0.5)
        ax.text(px + 0.25, PHASE_Y + PHASE_H - 0.32, lb, fontsize=11.5,
                fontweight='bold', color=ec, zorder=30)

    # ================================================
    # PHASE 1: Graph Construction
    # ================================================
    p1c = P1_X + P1_W/2

    # Agent Roles (top-left)
    aw = 2.2
    ax1 = P1_X + (P1_W/2 - aw)/2
    ay = PHASE_Y + PHASE_H - 2.0
    tb(ax, ax1, ay,
       [('Agent Roles', 9, C_WHITE, True),
        ('Planner, Coder, Critic', 7.5, C_WHITE, False)],
       aw, C_AGENT, '#00838F', 1.8)

    # Trajectories (top-right)
    tx1 = P1_X + P1_W/2 + (P1_W/2 - aw)/2
    tb(ax, tx1, ay,
       [('Interaction Trajectories', 9, '#004D40', True),
        ('Multi-turn agent dialogs & tool calls', 7.5, '#004D40', False)],
       aw, C_TRAJ, '#00838F', 1.8)

    # Arrows down to graph
    mid_y = ay - 0.4
    ar(ax, ax1 + aw/2, ay, ax1 + aw/2, mid_y, P1_BD, 2)
    ar(ax, tx1 + aw/2, ay, tx1 + aw/2, mid_y, P1_BD, 2)

    # Latent Reasoning Graph (center)
    gw = 4.8
    gx = p1c - gw/2
    gy = mid_y - 0.2 - 1.8
    tb(ax, gx, gy,
       [('Latent Reasoning Graph  G = (V, E, T)', 10, C_WHITE, True),
        ('Typed, weighted directed graph with 4 node types', 7.5, '#E3F2FD', False),
        ('Role  |  Trajectory  |  Context  |  Capsule', 7.5, '#E3F2FD', False),
        ('Each node: continuous embedding (d=384) via sentence-transformers', 7, '#BBDEFB', False),
        ('Edges encode: executes, follows, refines, contradicts, summarizes', 7, '#BBDEFB', False)],
       gw, C_GRAPH, P1_BD, 2.5)

    # ================================================
    # P1 -> P2 CONNECTOR
    # ================================================
    c1y = gy - 0.55
    ar(ax, P1_X + P1_W, c1y, P2_X, c1y, P1_BD, 2.8)
    ax.text(P1_X + P1_W + 0.28, c1y + 0.18, 'Encoded', fontsize=6.5, ha='center',
            color=P1_BD, fontweight='bold', zorder=30)
    ax.text(P1_X + P1_W + 0.28, c1y - 0.08, 'Graph Nodes', fontsize=6.5, ha='center',
            color=P1_BD, zorder=30)

    # ================================================
    # PHASE 2: Retrieval & Composition
    # ================================================
    p2c = P2_X + P2_W/2

    # Task Query
    qy = PHASE_Y + PHASE_H - 0.55
    ax.text(p2c, qy, 'Task Query  q', fontsize=10.5, fontweight='bold',
            ha='center', color=C_DARK, zorder=30)
    qay = qy - 0.18

    # Graph Router (left)
    rw = 2.4
    rx = P2_X + 0.3
    ry = qay - 0.35 - 1.15
    tb(ax, rx, ry,
       [('Graph Router', 9.5, C_WHITE, True),
        ('Score nodes: s(v; q) based on embedding similarity', 7.5, '#FFE0B2', False),
        ('Top-K retrieval of most relevant graph nodes', 7.5, '#FFE0B2', False),
        ('Adaptive budget: B = min(Bmax, Bmin + lambda*C(q))', 7, '#FFE0B2', False)],
       rw, C_ROUT, P2_BD, 1.8)

    # Adaptive TTS (right)
    ttx = P2_X + P2_W - 0.3 - rw
    tb(ax, ttx, ry,
       [('Adaptive Test-Time Scaling', 9.5, '#BF360C', True),
        ('Budget scales with query complexity C(q)', 7.5, '#BF360C', False),
        ('Save computation on simple tasks (use Bmin)', 7.5, '#BF360C', False),
        ('Allocate more resources for hard tasks (up to Bmax)', 7, '#BF360C', False)],
       rw, '#FFB74D', P2_BD, 1.8)

    # Arrows
    ar(ax, rx + rw/2, qay, rx + rw/2, ry + 1.15, C_ARR, 1.5)
    ar(ax, ttx + rw/2, qay, ttx + rw/2, ry + 1.15, C_ARR, 1.5)
    ar(ax, rx + rw, ry + 0.575, ttx, ry + 0.575, C_ARR, 1.5)

    # Memory Composer (center)
    cw = 4.8
    cx = p2c - cw/2
    cy = ry - 0.6 - 1.7
    tb(ax, cx, cy,
       [('Memory Composer: Role-Conditioned Composition', 10, C_WHITE, True),
        ('4-layer Transformer (d=384, 8 heads) + Learned Query Vectors + LoRA (r=16)', 7.5, '#FFE0B2', False),
        ('Role embedding via FiLM conditioning at composer layer 0', 7.5, '#FFE0B2', False),
        ('Cross-attention over retrieved nodes -> latent memory tokens m_t', 7.5, '#FFCC80', False)],
       cw, C_COMP, P2_BD, 2.5)

    ar(ax, rx + rw/2, ry, rx + rw/2, cy + 1.7, C_ARR, 1.5)
    ar(ax, ttx + rw/2, ry, ttx + rw/2, cy + 1.7, C_ARR, 1.5)

    # ================================================
    # P2 -> P3 CONNECTOR
    # ================================================
    c2y = cy - 0.55
    ar(ax, P2_X + P2_W, c2y, P3_X, c2y, P2_BD, 2.8)
    ax.text(P2_X + P2_W + 0.28, c2y + 0.18, 'Latent', fontsize=6.5, ha='center',
            color=P2_BD, fontweight='bold', zorder=30)
    ax.text(P2_X + P2_W + 0.28, c2y - 0.08, 'Tokens  m_t', fontsize=6.5, ha='center',
            color=P2_BD, zorder=30)

    # ================================================
    # PHASE 3: Injection & Evolution
    # ================================================
    p3c = P3_X + P3_W/2

    # Latent tokens label
    ty1 = PHASE_Y + PHASE_H - 0.55
    ax.text(p3c, ty1, 'Latent Memory Tokens  m_t', fontsize=10.5, fontweight='bold',
            ha='center', color=C_DARK, zorder=30)
    tay = ty1 - 0.18

    # Gated Injection (left)
    iw = 2.8
    ix = P3_X + 0.3
    iy = tay - 0.35 - 1.4
    tb(ax, ix, iy,
       [('Gated Latent Injection', 9.5, C_WHITE, True),
        ('Cross-attention at transformer layers {0, 4, 8}', 7.5, '#C8E6C9', False),
        ('Covers low-level, mid-level, and high-level features', 7.5, '#C8E6C9', False),
        ('Learned gate: gamma controls how much memory to use', 7.5, '#A5D6A7', False)],
       iw, C_INJECT, P3_BD, 1.8)

    # Backbone LLM (right)
    lw = 2.6
    lx = P3_X + P3_W - 0.3 - lw
    tb(ax, lx, iy,
       [('Frozen Backbone LLM', 9.5, C_WHITE, True),
        ('DeepSeek-V4, Qwen3.5, Llama3.1', 7.5, '#CFD8DC', False),
        ('No fine-tuning required', 7.5, '#B0BEC5', False),
        ('Plug-and-play: compatible with any Transformer', 7.5, '#90A4AE', False)],
       lw, C_LLM, '#455A64', 1.8)

    ar(ax, ix + iw/2, tay, ix + iw/2, iy + 1.4, C_ARR, 1.5)
    ar(ax, lx + lw/2, tay, lx + lw/2, iy + 1.4, C_ARR, 1.5)
    ar(ax, ix + iw, iy + 0.7, lx, iy + 0.7, C_ARR, 1.5)

    # Output
    ow = 5.2
    ox = p3c - ow/2
    oy = iy - 0.55 - 0.8
    tb(ax, ox, oy,
       [('Multi-Agent Response', 10, C_DARK, True),
        ('Planner + Coder + Critic -- augmented with retrieved latent context', 8, C_MED, False)],
       ow, C_OUT, '#9E9E9E', 1.8)

    ar(ax, ix + iw/2, iy, ix + iw/2, oy + 0.8, C_ARR, 1.5)
    ar(ax, lx + lw/2, iy, lx + lw/2, oy + 0.8, C_ARR, 1.5)

    # Online Graph Evolution
    ew = 2.8
    ex = ix
    ey = oy - 0.55 - 1.0
    tb(ax, ex, ey,
       [('Online Graph Evolution', 9.5, C_WHITE, True),
        ('Merge similar nodes  |  Expand with new interactions', 7.5, '#E1BEE7', False),
        ('Prune low-utility nodes  |  Utility U(n) updated via task feedback', 7.5, '#CE93D8', False)],
       ew, C_EVOLVE, '#6A1B9A', 1.8)

    ar(ax, ox + ow/2, oy, ox + ow/2, ey + 1.0, C_ARR, 1.5)

    # ================================================
    # FEEDBACK LOOP (Phase 3 -> Phase 1)
    # ================================================
    fby = ey - 0.6
    fsx = ex + ew/2
    fex = p1c

    ar(ax, fsx, ey, fsx, fby, C_LOOP, 2.5, ls='--')
    ar(ax, fsx, fby, fex, fby, C_LOOP, 2.5, ls='--')
    ar(ax, fex, fby, fex, gy, C_LOOP, 2.5, ls='--')

    ax.text(FIG_W/2, fby - 0.42,
            'Graph Update Feedback Loop -- utility-based online evolution (merge, expand, prune)',
            fontsize=8, ha='center', color=C_LOOP, fontweight='bold',
            style='italic', zorder=30)

    # ================================================
    # DATA FLOW BAR
    # ================================================
    flow_y = fby - 0.95
    bar_w = 16.8
    bar_x = (FIG_W - bar_w)/2
    bx(ax, bar_x, flow_y, bar_w, 0.55, C_WHITE, '#BDBDBD', 1, 0.08, 5)

    stages = ['Raw Interactions', 'Latent Node\nEmbeddings',
              'Role-Conditioned\nMemory Tokens', 'Hidden State\nInjection',
              'Agent Response\n+ Graph Update']
    n = len(stages)
    step = (bar_w - 1.2)/(n - 1)
    for i, s in enumerate(stages):
        sx = bar_x + 0.6 + i*step
        ax.text(sx, flow_y + 0.27, s, fontsize=6.5, ha='center', va='center',
                color=C_MED, zorder=25)
        if i < n - 1:
            ar(ax, sx + step*0.55, flow_y + 0.27, sx + step*0.85, flow_y + 0.27,
                '#BDBDBD', 1.2)

    # ================================================
    # LEGEND
    # ================================================
    lgy = flow_y - 0.7
    ax.text(0.4, lgy, 'Key Components:', fontsize=9, fontweight='bold',
            color=C_DARK, zorder=30)

    items = [(C_GRAPH, 'Latent Graph'), (C_COMP, 'Composer'),
             (C_ROUT, 'Router'), (C_INJECT, 'Injection'),
             (C_EVOLVE, 'Evolution'), (C_LLM, 'Backbone LLM'),
             (C_AGENT, 'Agent Roles')]
    for i, (clr, lbl) in enumerate(items):
        lx = 0.4 + i*2.5
        bx(ax, lx, lgy - 0.25, 0.42, 0.2, clr, '#999', 0.5, 0.03, 15)
        ax.text(lx + 0.52, lgy - 0.15, lbl, fontsize=7.5, va='center',
                color=C_DARK, zorder=30)

    # ================================================
    # SAVE
    # ================================================
    out = os.path.join(FIG_DIR, 'fig_architecture.pdf')
    fig.savefig(out, format='pdf', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out} ({os.path.getsize(out)/1024:.0f} KB)")
    print("Done!")


if __name__ == '__main__':
    draw()
