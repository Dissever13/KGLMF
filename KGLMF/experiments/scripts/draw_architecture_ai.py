#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KG-LMF Architecture Diagram — Clean Adobe Illustrator rendering.
Simple boxes, generous spacing, direct arrows. Biorender-inspired palette.

Usage:
    /d/WorkSoftware/python D:/KGLM/KGLMF/experiments/scripts/draw_architecture_ai.py
"""

import os, math, pythoncom, win32com.client

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Canvas
W, H = 1400, 820

def iy(y): return H - y

# ── Colors ──────────────────────────────────────────
P1_BG = (235, 245, 255); P1_BD = (21, 101, 192);  P1_TX = (21, 101, 192)
P2_BG = (255, 245, 235); P2_BD = (230, 81, 0);    P2_TX = (191, 54, 12)
P3_BG = (235, 248, 235); P3_BD = (46, 125, 50);   P3_TX = (46, 125, 50)

C_GRAPH  = (52, 152, 219);  C_COMP = (255, 152, 0)
C_ROUT   = (255, 112, 67);  C_INJECT = (76, 175, 80)
C_EVOLVE = (156, 39, 176);  C_LLM = (120, 144, 156)
C_AGENT  = (0, 188, 212);   C_TRAJ = (0, 150, 136)
C_ARR    = (84, 110, 122);  C_DASH = (156, 39, 176)
C_DARK   = (33, 33, 33);    C_MED = (97, 97, 97)
C_WHITE  = (255, 255, 255); C_LGRAY = (245, 245, 245)

FB = "Arial-BoldMT"
FR = "ArialMT"

# ── Layout ──────────────────────────────────────────
MARGIN  = 30
GAP     = 35
PW = (W - 2*MARGIN - 2*GAP) // 3   # ~423

P1_X = MARGIN                        # 30
P2_X = MARGIN + PW + GAP             # 488
P3_X = MARGIN + 2*(PW + GAP)         # 946

PHASE_TOP = 100
PHASE_H   = 600

# Box sizes
BOX_W_SMALL = 180   # small module boxes
BOX_W_WIDE  = PW - 40  # wide (phase-filling) boxes
BOX_H_SMALL = 68    # 2-line box
BOX_H_MED   = 95    # 3-line box
BOX_H_LARGE = 130   # 5-line box


class AI:
    def __init__(self):
        pythoncom.CoInitialize()
        self.app = win32com.client.Dispatch("Illustrator.Application")
        self.doc = self.app.Documents.Add()
        self.js(f"var d=app.activeDocument;d.artboards[0].artboardRect=[0,{H},{W},0];d.documentColorSpace=DocumentColorSpace.RGB;")

    def js(self, code):
        try: return self.app.DoJavaScript(code)
        except Exception as e: print(f"  JS: {str(e)[:100]}")

    # ── Shapes ───────────────────────────────────

    def rect(self, x, y, w, h, fill, stroke=None, sw=0, r=0):
        top = iy(y)
        s = f"d.pathItems.roundedRectangle({top},{x},{w},{h},{r},{r})" if r else f"d.pathItems.rectangle({top},{x},{w},{h})"
        st = ""
        if stroke and sw > 0:
            st = (f"var sc=new RGBColor();sc.red={stroke[0]};sc.green={stroke[1]};sc.blue={stroke[2]};"
                  f"s.strokeColor=sc;s.strokeWidth={sw};s.stroked=true;")
        else: st = "s.stroked=false;"
        self.js(f"var d=app.activeDocument;var s={s};var fc=new RGBColor();fc.red={fill[0]};fc.green={fill[1]};fc.blue={fill[2]};s.fillColor=fc;s.filled=true;{st}")

    def arrow(self, x1, y1, x2, y2, color=C_ARR, w=2, sz=9, dash=False):
        iy1, iy2 = iy(y1), iy(y2)
        ds = "L.strokeDashes=[6,4];" if dash else ""
        self.js(f"var d=app.activeDocument;var L=d.pathItems.add();L.setEntirePath([[{x1},{iy1}],[{x2},{iy2}]]);var sc=new RGBColor();sc.red={color[0]};sc.green={color[1]};sc.blue={color[2]};L.strokeColor=sc;L.strokeWidth={w};L.filled=false;L.stroked=true;{ds}")
        dx, dy = x2-x1, iy2-iy1
        L = math.sqrt(dx*dx+dy*dy)
        if L < 0.5: return
        ux, uy = dx/L, dy/L
        px, py = -uy*sz*0.5, ux*sz*0.5
        bx, by = x2-ux*sz, iy2-uy*sz
        self.js(f"var d=app.activeDocument;var h=d.pathItems.add();h.setEntirePath([[{x2},{iy2}],[{bx+px},{by+py}],[{bx-px},{by-py}]]);h.closed=true;var fc=new RGBColor();fc.red={color[0]};fc.green={color[1]};fc.blue={color[2]};h.fillColor=fc;h.filled=true;h.stroked=false;")

    # ── Text ─────────────────────────────────────

    def txt(self, s, x, y, size=12, font=FR, color=C_DARK, bold=False, center=False):
        il = iy(y)
        e = s.replace('\\','\\\\').replace("'","\\'")
        fj = f"try{{t.textRange.characterAttributes.textFont=textFonts.getByName('{font}');}}catch(e){{}}"
        ct = "t.textRange.characterAttributes.justification=Justification.CENTER;" if center else ""
        bd = "t.textRange.characterAttributes.strokeWeight=0.6;" if bold else ""
        self.js(f"var d=app.activeDocument;var t=d.textFrames.pointText([{x},{il}]);t.contents='{e}';t.textRange.characterAttributes.size={size};{fj}var fc=new RGBColor();fc.red={color[0]};fc.green={color[1]};fc.blue={color[2]};t.textRange.characterAttributes.fillColor=fc;{ct}{bd}")

    def ctxt(self, s, cx, cy, size=12, font=FR, color=C_DARK, bold=False):
        self.txt(s, cx, cy, size, font, color, bold, center=True)

    # ── Box with text ────────────────────────────

    def box(self, x, y, w, h, lines, bg, border, bsw=1.8, r=10):
        """Draw rounded box with centered text lines inside.
        lines: [(text, size, font, color, bold), ...]
        Text lines are spaced evenly from top to bottom within the box.
        """
        self.rect(x, y, w, h, bg, border, bsw, r)
        n = len(lines)
        # Text area: leave padding at top/bottom
        pad_top = 14
        pad_bot = 10
        txt_h = h - pad_top - pad_bot
        if n == 1:
            spacing = 0
            first_y = y + h/2 + 2
        elif n == 2:
            spacing = txt_h * 0.55
            first_y = y + h - pad_top - 4
        elif n == 3:
            spacing = txt_h * 0.42
            first_y = y + h - pad_top - 2
        elif n == 4:
            spacing = txt_h * 0.32
            first_y = y + h - pad_top - 2
        else:
            spacing = txt_h * 0.26
            first_y = y + h - pad_top

        for i, (txt, sz, fnt, clr, bld) in enumerate(lines):
            ly = first_y - i * spacing
            self.txt(txt, x + w/2, ly, sz, fnt, clr, bld, center=True)


    # ── Export ───────────────────────────────────
    def save_pdf(self, path):
        abs_path = os.path.abspath(path).replace('\\', '/')
        self.js(f"var d=app.activeDocument;var f=new File('{abs_path}');var o=new PDFSaveOptions();d.saveAs(f,o);")
        if os.path.exists(abs_path):
            print(f"  PDF: {abs_path} ({os.path.getsize(abs_path)//1024} KB)")
        else:
            print(f"  FAIL: {abs_path}")

    def save_ai(self, path):
        abs_path = os.path.abspath(path).replace('\\', '/')
        self.doc.SaveAs(abs_path)
        print(f"  AI:  {abs_path}")


# ═══════════════════════════════════════════════════════════
def draw():
    print("=" * 60)
    print("KG-LMF Architecture Diagram — Clean Illustrator Render")
    print("=" * 60)
    ai = AI()

    # ── TITLE ─────────────────────────────────────
    ai.ctxt("KG-LMF: Knowledge Graph-Driven Latent Memory Framework for Multi-Agent Test-Time Scaling",
            W//2, 30, 22, FB, C_DARK)
    ai.ctxt("Architecture Overview", W//2, 62, 14, FR, C_MED)

    # ── PHASE BACKGROUNDS ─────────────────────────
    for px, bg, bd, label in [
        (P1_X, P1_BG, P1_BD, "Phase 1:  Graph Construction"),
        (P2_X, P2_BG, P2_BD, "Phase 2:  Retrieval & Composition"),
        (P3_X, P3_BG, P3_BD, "Phase 3:  Injection & Evolution"),
    ]:
        ai.rect(px, PHASE_TOP, PW, PHASE_H, bg, bd, 2.5, 18)
        ai.txt(label, px + 15, PHASE_TOP + 18, 12, FB, bd)

    # ════════════════════════════════════════════════
    # PHASE 1: Graph Construction
    # ════════════════════════════════════════════════
    p1c = P1_X + PW//2

    # --- Agent Roles (top-left) ---
    a_w, a_h = BOX_W_SMALL, BOX_H_SMALL
    a_x = P1_X + (PW//2 - a_w)//2
    a_y = PHASE_TOP + PHASE_H - 170
    ai.box(a_x, a_y, a_w, a_h,
           [("Agent Roles", 10, FB, C_WHITE, True),
            ("Planner | Coder | Critic", 8, FR, C_WHITE, False)],
           C_AGENT, (0, 131, 143), 1.5, 8)

    # --- Trajectories (top-right) ---
    t_x = P1_X + PW//2 + (PW//2 - a_w)//2
    ai.box(t_x, a_y, a_w, a_h,
           [("Interaction Trajectories", 10, FB, (0, 77, 64), True),
            ("Multi-turn agent dialogs & tool calls", 7.5, FR, (0, 77, 64), False)],
           C_TRAJ, (0, 131, 143), 1.5, 8)

    # --- Arrows down to graph ---
    mid = a_y - 40
    ai.arrow(a_x + a_w//2, a_y, a_x + a_w//2, mid, P1_BD, 2, 8)
    ai.arrow(t_x + a_w//2, a_y, t_x + a_w//2, mid, P1_BD, 2, 8)

    # --- Latent Reasoning Graph (center) ---
    g_w, g_h = BOX_W_WIDE, BOX_H_LARGE
    g_x = p1c - g_w//2
    g_y = mid - 20 - g_h
    ai.box(g_x, g_y, g_w, g_h,
           [("Latent Reasoning Graph   G = (V, E, T)", 11, FB, C_WHITE, True),
            ("Typed, weighted directed graph with 4 node types", 8, FR, (210, 235, 255), False),
            ("Role  |  Trajectory  |  Context  |  Capsule", 8, FR, (210, 235, 255), False),
            ("Each node: continuous embedding (d=384) via sentence-transformers", 7.5, FR, (190, 220, 245), False),
            ("Edges: executes, follows, refines, contradicts, summarizes", 7, FR, (180, 210, 240), False)],
           C_GRAPH, P1_BD, 2.5, 12)

    # ════════════════════════════════════════════════
    # P1 -> P2 CONNECTOR
    # ════════════════════════════════════════════════
    c1y = g_y - 50
    ai.arrow(P1_X + PW, c1y, P2_X, c1y, P1_BD, 2.8, 10)
    ai.ctxt("Encoded Graph Nodes", P1_X + PW + GAP//2, c1y - 14, 7, FR, P1_TX)

    # ════════════════════════════════════════════════
    # PHASE 2: Retrieval & Composition
    # ════════════════════════════════════════════════
    p2c = P2_X + PW//2

    # --- Task Query ---
    qy = PHASE_TOP + PHASE_H - 48
    ai.ctxt("Task Query  q", p2c, qy, 12, FB, C_DARK)
    qay = qy - 18

    # --- Graph Router (left) ---
    r_w, r_h = BOX_W_SMALL, BOX_H_MED
    r_x = P2_X + 15
    r_y = qay - 15 - r_h
    ai.box(r_x, r_y, r_w, r_h,
           [("Graph Router", 10, FB, C_WHITE, True),
            ("Score: s(v; q) via embedding", 7.5, FR, (255, 224, 200), False),
            ("Top-K retrieval of relevant", 7.5, FR, (255, 224, 200), False),
            ("graph nodes for composition", 7.5, FR, (255, 224, 200), False)],
           C_ROUT, P2_BD, 1.5, 8)

    # --- Adaptive TTS (right) ---
    tt_x = P2_X + PW - r_w - 15
    ai.box(tt_x, r_y, r_w, r_h,
           [("Adaptive TTS", 10, FB, (191, 54, 12), True),
            ("Budget: B = min(Bmax,", 7.5, FR, (191, 54, 12), False),
            ("Bmin + lambda * C(q))", 7.5, FR, (191, 54, 12), False),
            ("Auto-scale compute to query", 7.5, FR, (191, 54, 12), False)],
           (255, 183, 77), P2_BD, 1.5, 8)

    # --- Arrows ---
    ai.arrow(r_x + r_w//2, qay, r_x + r_w//2, r_y + r_h, C_ARR, 1.6, 7)
    ai.arrow(tt_x + r_w//2, qay, tt_x + r_w//2, r_y + r_h, C_ARR, 1.6, 7)
    ai.arrow(r_x + r_w, r_y + r_h//2, tt_x, r_y + r_h//2, C_ARR, 1.6, 7)

    # --- Memory Composer (center, bottom) ---
    c_w, c_h = BOX_W_WIDE, BOX_H_LARGE
    c_x = p2c - c_w//2
    c_y = r_y - 35 - c_h
    ai.box(c_x, c_y, c_w, c_h,
           [("Memory Composer: Role-Conditioned Composition", 11, FB, C_WHITE, True),
            ("4-layer Transformer (d=384, 8 heads) + Learned Query Vectors + LoRA (r=16)", 8, FR, (255, 224, 200), False),
            ("Role embedding via FiLM at layer 0  |  Cross-attention over retrieved nodes", 8, FR, (255, 224, 200), False),
            ("Output: latent memory tokens  m_t = sum_i alpha_i * v_i", 7.5, FR, (255, 200, 170), False)],
           C_COMP, P2_BD, 2.5, 12)

    ai.arrow(r_x + r_w//2, r_y, r_x + r_w//2, c_y + c_h, C_ARR, 1.6, 7)
    ai.arrow(tt_x + r_w//2, r_y, tt_x + r_w//2, c_y + c_h, C_ARR, 1.6, 7)

    # ════════════════════════════════════════════════
    # P2 -> P3 CONNECTOR
    # ════════════════════════════════════════════════
    c2y = c_y - 50
    ai.arrow(P2_X + PW, c2y, P3_X, c2y, P2_BD, 2.8, 10)
    ai.ctxt("Latent Memory Tokens  m_t", P2_X + PW + GAP//2, c2y - 14, 7, FR, P2_TX)

    # ════════════════════════════════════════════════
    # PHASE 3: Injection & Evolution
    # ════════════════════════════════════════════════
    p3c = P3_X + PW//2

    # --- Latent Tokens ---
    ty1 = PHASE_TOP + PHASE_H - 48
    ai.ctxt("Latent Memory Tokens  m_t", p3c, ty1, 12, FB, C_DARK)
    tay = ty1 - 18

    # --- Gated Injection (left) ---
    i_w, i_h = BOX_W_SMALL, BOX_H_MED
    i_x = P3_X + 15
    i_y = tay - 15 - i_h
    ai.box(i_x, i_y, i_w, i_h,
           [("Gated Latent Injection", 10, FB, C_WHITE, True),
            ("Cross-attention at layers", 7.5, FR, (200, 235, 200), False),
            ("{0, 4, 8} (low, mid, high)", 7.5, FR, (200, 235, 200), False),
            ("Learned gate gamma controls", 7.5, FR, (180, 220, 180), False),
            ("memory integration strength", 7, FR, (180, 220, 180), False)],
           C_INJECT, P3_BD, 1.5, 8)

    # --- Backbone LLM (right) ---
    l_w, l_h = BOX_W_SMALL, i_h
    ll_x = P3_X + PW - l_w - 15
    ai.box(ll_x, i_y, l_w, l_h,
           [("Frozen Backbone LLM", 10, FB, C_WHITE, True),
            ("DeepSeek-V4", 8, FR, (210, 216, 220), False),
            ("Qwen3.5:9B", 8, FR, (210, 216, 220), False),
            ("Llama3.1:8B", 8, FR, (210, 216, 220), False),
            ("No fine-tuning needed", 7.5, FR, (200, 206, 210), False)],
           C_LLM, (55, 71, 79), 1.5, 8)

    ai.arrow(i_x + i_w//2, tay, i_x + i_w//2, i_y + i_h, C_ARR, 1.6, 7)
    ai.arrow(ll_x + l_w//2, tay, ll_x + l_w//2, i_y + i_h, C_ARR, 1.6, 7)
    ai.arrow(i_x + i_w, i_y + i_h//2, ll_x, i_y + i_h//2, C_ARR, 1.6, 7)

    # --- Output ---
    o_w, o_h = BOX_W_WIDE, BOX_H_SMALL
    o_x = p3c - o_w//2
    o_y = i_y - 30 - o_h
    ai.box(o_x, o_y, o_w, o_h,
           [("Multi-Agent Response", 10, FB, C_DARK, True),
            ("Planner + Coder + Critic -- augmented with retrieved latent context", 8, FR, C_MED, False)],
           C_LGRAY, (158, 158, 158), 1.2, 8)

    ai.arrow(i_x + i_w//2, i_y, i_x + i_w//2, o_y + o_h, C_ARR, 1.6, 7)
    ai.arrow(ll_x + l_w//2, i_y, ll_x + l_w//2, o_y + o_h, C_ARR, 1.6, 7)

    # --- Online Graph Evolution ---
    e_w, e_h = BOX_W_SMALL + 20, BOX_H_MED
    e_x = i_x
    e_y = o_y - 30 - e_h
    ai.box(e_x, e_y, e_w, e_h,
           [("Online Graph Evolution", 10, FB, C_WHITE, True),
            ("Merge similar nodes", 7.5, FR, (225, 210, 240), False),
            ("Expand with new interactions", 7.5, FR, (225, 210, 240), False),
            ("Prune low-utility nodes", 7.5, FR, (225, 210, 240), False)],
           C_EVOLVE, (106, 27, 154), 1.5, 8)

    ai.arrow(o_x + o_w//2, o_y, o_x + o_w//2, e_y + e_h, C_ARR, 1.6, 7)

    # ════════════════════════════════════════════════
    # FEEDBACK LOOP
    # ════════════════════════════════════════════════
    fby = e_y - 40
    fsx = e_x + e_w//2
    fex = p1c

    ai.arrow(fsx, e_y, fsx, fby, C_DASH, 2.2, 9, dash=True)
    ai.arrow(fsx, fby, fex, fby, C_DASH, 2.2, 9, dash=True)
    ai.arrow(fex, fby, fex, g_y, C_DASH, 2.2, 9, dash=True)

    ai.ctxt("Feedback: utility-based merge, expand, prune  ->  continuous graph improvement",
            W//2, fby - 18, 8, FR, C_DASH)

    # ════════════════════════════════════════════════
    # DATA FLOW BAR
    # ════════════════════════════════════════════════
    flow_y = fby - 50
    bar_x = 50
    bar_w = W - 100
    ai.rect(bar_x, flow_y, bar_w, 40, C_WHITE, (189, 189, 189), 0.8, 6)

    stages = ["Raw Interactions", "Latent Node Embeddings", "Role-Conditioned Tokens",
              "Hidden State Injection", "Agent Response + Graph Update"]
    step = (bar_w - 120) // (len(stages) - 1)
    for i, s in enumerate(stages):
        ai.ctxt(s, bar_x + 60 + i*step, flow_y + 10, 7, FR, C_MED)
        if i < len(stages) - 1:
            ai.arrow(bar_x + 60 + i*step + 70, flow_y + 10,
                     bar_x + 60 + (i+1)*step - 70, flow_y + 10,
                     (189, 189, 189), 1, 5)

    # ════════════════════════════════════════════════
    # LEGEND
    # ════════════════════════════════════════════════
    lgy = flow_y - 50
    ai.txt("Components:", 30, lgy, 9, FB, C_DARK)

    items = [(C_GRAPH,"Latent Graph"),(C_COMP,"Composer"),(C_ROUT,"Router"),
             (C_INJECT,"Injection"),(C_EVOLVE,"Evolution"),(C_LLM,"Backbone LLM"),
             (C_AGENT,"Agent Roles")]
    for i, (clr, lbl) in enumerate(items):
        lx = 30 + i * 180
        ai.rect(lx, lgy - 22, 40, 16, clr, (130,130,130), 0.5, 2)
        ai.txt(lbl, lx + 46, lgy - 19, 7, FR, C_MED)

    # ════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════
    print("\nExporting...")
    ai_p  = os.path.join(FIG_DIR, "fig_architecture_ai.ai")
    pdf_p = os.path.join(FIG_DIR, "fig_architecture_ai.pdf")
    main_p = os.path.join(FIG_DIR, "fig_architecture.pdf")

    ai.save_ai(ai_p)
    ai.save_pdf(pdf_p)
    ai.save_pdf(main_p)

    print(f"\nDone!")
    print(f"  AI:   {ai_p}")
    print(f"  PDF:  {main_p}")


if __name__ == "__main__":
    draw()
