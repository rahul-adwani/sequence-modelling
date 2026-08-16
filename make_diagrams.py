"""Hand-drawn-style diagrams and equation plates for Article 2.

    python make_diagrams.py

Two kinds of output, both intended as references for redrawing by hand:

* `d*.png`  schematics. They carry an idea, not a measurement, so the sketchy
            line is deliberate: it marks them as explanation rather than result.
* `e*.png`  equation plates, set in the notation of Part 1 (a^t_x, w_ab,
            delta^t_x, J for the cost) so nothing is silently renamed between
            articles.

Kept separate from `make_figures.py`, where every pixel is computed from measured
data and nothing may be drawn by hand.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
except ImportError:  # pragma: no cover
    print("matplotlib not installed; diagrams skipped", file=sys.stderr)
    sys.exit(0)

from foundations.harness import FIG_DIR

warnings.filterwarnings("ignore")

INK, ACCENT, GREY = "#2f2f2f", "#c0392b", "#9aa0a6"
FILL, FILL2, FILL3 = "#eceaf5", "#fdeee6", "#e6f2ec"


def box(ax, x, y, label="", w=2.0, h=1.3, fc=FILL, fs=11, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc=fc, ec=INK, lw=lw, zorder=2))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=3)


def arr(ax, p1, p2, label="", rad=0.0, color=INK, lw=1.6, dy=0.30, fs=10,
        style="-|>", ls="solid"):
    ax.add_patch(FancyArrowPatch(p1, p2, lw=lw, arrowstyle=style, linestyle=ls,
                                 mutation_scale=14, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + dy, label,
                ha="center", fontsize=fs, color=color, zorder=4)


def canvas(w, h, xlim, ylim, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=14, color=INK)
    return fig, ax


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=140, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  {name}")


# ============================================================ diagrams

def d0_two_assumptions():
    """S0: what Part 1 assumed, and how a sequence breaks both assumptions."""
    fig, ax = canvas(10, 4.0, (0, 20), (0, 8),
                     "Part 1 assumed a fixed shape and fixed feature slots")
    for i in range(4):
        box(ax, 1.0, 6.2 - i * 1.35, "", w=1.0, h=0.9, fs=9)
    ax.text(1.5, 7.6, "4 features", ha="center", fontsize=10)
    ax.text(1.5, 0.5, "always exactly 4\nand order means nothing",
            ha="center", fontsize=9.5, color=GREY)
    box(ax, 3.6, 3.4, "network", w=2.4, h=2.6)
    arr(ax, (2.1, 4.6), (3.6, 4.6))
    arr(ax, (6.0, 4.6), (7.0, 4.6))
    box(ax, 7.0, 4.1, "y", w=0.9, h=0.9)

    ax.text(10.2, 7.5, "a sentence breaks both", fontsize=11.5, color=ACCENT)
    for i, wd in enumerate(["the", "dog", "ate", "it"]):
        box(ax, 10.2 + i * 2.2, 4.8, wd, w=1.8, h=1.0, fc=FILL2, fs=10)
    ax.text(14.0, 3.9, "length varies from sentence to sentence",
            ha="center", fontsize=9.5, color=ACCENT)
    ax.text(14.0, 3.0, "and the order carries the meaning",
            ha="center", fontsize=9.5, color=ACCENT)
    ax.text(14.0, 1.9, "'dog ate it'  vs  'it ate dog'",
            ha="center", fontsize=10, color=INK)
    save(fig, "d0_two_assumptions.png")


def d1_output_layer():
    """S1: only the top of the network changed."""
    fig, ax = canvas(9, 4.4, (0, 18), (0, 9),
                     "only the output layer changed")
    for side, x0, fc in (("one output", 1.0, FILL), ("K outputs", 10.0, FILL2)):
        ax.text(x0 + 2.0, 8.2, side, ha="center", fontsize=11.5)
        for i in range(3):
            box(ax, x0 + i * 1.35, 4.6, "", w=1.0, h=0.9, fs=9)
        ax.text(x0 + 2.0, 3.9, "shared body", ha="center", fontsize=9.5,
                color=GREY)
    box(ax, 2.0, 6.4, "", w=1.0, h=0.9, fc=FILL)
    ax.text(2.5, 7.6, r"$\sigma$", ha="center", fontsize=13)
    for i in range(4):
        box(ax, 10.0 + i * 1.35, 6.4, "", w=1.0, h=0.9, fc=FILL2)
    ax.text(12.7, 7.6, "softmax", ha="center", fontsize=11)
    for x0 in (1.0, 10.0):
        arr(ax, (x0 + 2.0, 5.5), (x0 + 2.0, 6.4))
    arr(ax, (2.5, 2.6), (2.5, 4.4), color=ACCENT, lw=2.0)
    arr(ax, (12.7, 2.6), (12.7, 4.4), color=ACCENT, lw=2.0)
    ax.text(7.6, 1.9, r"the same backward term:   $\delta^{L} = a^{L} - y$",
            ha="center", fontsize=12.5, color=ACCENT)
    ax.text(7.6, 0.8, "so the hidden-layer recursion keeps its form; only the "
            "width of the vector handed in at the top changes",
            ha="center", fontsize=10, color=GREY)
    save(fig, "d1_output_layer.png")


def d2_rnn():
    """S2: the folded and unrolled recurrent network. The main diagram."""
    fig, ax = canvas(12, 5.0, (0, 24), (0, 10),
                     "the recurrent network: folded, and unrolled")
    box(ax, 1.2, 4.4, "", w=2.2, h=1.6)
    ax.text(2.3, 5.2, r"$a^{t}$", ha="center", va="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((3.4, 5.9), (3.4, 4.5), lw=1.6,
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=INK, connectionstyle="arc3,rad=-1.6"))
    ax.text(5.2, 5.2, r"$w$", fontsize=12)
    arr(ax, (2.3, 3.2), (2.3, 4.4), r"$u$", dy=-0.1, fs=11)
    arr(ax, (2.3, 6.0), (2.3, 7.2), r"$v$", dy=-0.1, fs=11)
    ax.text(2.3, 2.7, r"$x^{t}$", ha="center", fontsize=11)
    ax.text(2.3, 7.6, r"$y^{t}$", ha="center", fontsize=11)
    ax.text(2.3, 1.6, "folded", ha="center", fontsize=10.5, color=GREY)

    ax.plot([7.2, 7.2], [1.2, 8.8], color=GREY, lw=1.0, ls=(0, (4, 4)))

    for i, wd in enumerate(["the", "dog", "ate"]):
        x = 9.0 + i * 4.4
        box(ax, x, 4.4, "", w=2.4, h=1.6)
        ax.text(x + 1.2, 5.2, rf"$a^{{{i}}}$", ha="center", va="center",
                fontsize=12)
        arr(ax, (x + 1.2, 3.2), (x + 1.2, 4.4), r"$u$", dy=-0.1, fs=11)
        ax.text(x + 1.2, 2.7, wd, ha="center", fontsize=10.5, style="italic")
        arr(ax, (x + 1.2, 6.0), (x + 1.2, 7.2), r"$v$", dy=-0.1, fs=11)
        ax.text(x + 1.2, 7.6, rf"$y^{{{i}}}$", ha="center", fontsize=11)
        if i < 2:
            arr(ax, (x + 2.4, 5.2), (x + 4.4, 5.2), r"$w$", fs=12)
    ax.text(15.6, 1.6, "unrolled", ha="center", fontsize=10.5, color=GREY)

    # The annotation that is the whole idea. Circles sit exactly on the two w
    # labels; no connecting arc, because any arc between them collides with the
    # output labels and the caption already makes the link.
    for cx in (12.6, 17.0):
        ax.add_patch(plt.Circle((cx, 5.5), 0.52, fill=False, ec=ACCENT, lw=1.8,
                                zorder=5))
    ax.text(14.8, 8.6, "the same matrix, every step", ha="center",
            fontsize=11.5, color=ACCENT)
    ax.text(13.8, 0.6, r"no layer superscript on $w$: there is only one",
            ha="center", fontsize=10, color=ACCENT)
    save(fig, "d2_rnn.png")


def d3_vanishing():
    """S3: forward flows fine, backward tapers to nothing."""
    fig, ax = canvas(11, 4.6, (0, 22), (0, 9),
                     "the forward pass is fine; the gradient is not")
    xs = [1.4 + i * 3.9 for i in range(5)]
    for i, x in enumerate(xs):
        box(ax, x, 4.8, "", w=2.2, h=1.4)
        ax.text(x + 1.1, 5.5, rf"$a^{{{i+1}}}$", ha="center", va="center",
                fontsize=11)
        if i < 4:
            arr(ax, (x + 2.2, 5.5), (x + 3.9, 5.5), lw=1.8)
    ax.text(11.0, 7.2, "forward: every arrow the same weight", ha="center",
            fontsize=10.5, color=GREY)

    widths = [0.4, 0.8, 1.6, 3.2]
    for i in range(4):
        x1, x2 = xs[4 - i] + 0.2, xs[3 - i] + 2.0
        arr(ax, (x1, 3.6), (x2, 3.6), color=ACCENT, lw=widths[i])
    ax.text(11.0, 2.5, r"backward: multiplied by $w^{T}\,\mathrm{diag}(1-a^2)$"
            " at every step", ha="center", fontsize=10.5, color=ACCENT)
    ax.text(11.0, 1.4, "about 0.35 per step, so one millionth after 13 steps",
            ha="center", fontsize=10.5, color=ACCENT)
    ax.text(21.4, 3.6, "loss", ha="right", fontsize=10.5, color=ACCENT)
    save(fig, "d3_vanishing.png")


def d4_lstm():
    """S4: the cell state runs straight across the top, uninterrupted."""
    fig, ax = canvas(11, 5.2, (0, 22), (0, 10),
                     "the LSTM: a straight line, with plumbing hanging off it")
    ax.plot([1.0, 21.0], [8.0, 8.0], color=ACCENT, lw=3.0, zorder=1)
    ax.text(0.9, 8.6, r"$c^{t-1}$", ha="left", fontsize=12, color=ACCENT)
    ax.text(21.1, 8.6, r"$c^{t}$", ha="right", fontsize=12, color=ACCENT)
    for x, sym in ((7.0, r"$\times$"), (12.0, "+")):
        ax.add_patch(plt.Circle((x, 8.0), 0.62, fc="white", ec=INK, lw=1.6,
                                zorder=3))
        ax.text(x, 8.0, sym, ha="center", va="center", fontsize=15, zorder=4)

    gates = [(4.2, r"$f^{t}$", "forget", FILL), (9.2, r"$i^{t}$", "input", FILL),
             (12.0, r"$g^{t}$", "candidate", FILL3),
             (16.5, r"$o^{t}$", "output", FILL)]
    for x, sym, name, fc in gates:
        box(ax, x - 0.9, 3.6, sym, w=1.8, h=1.2, fc=fc, fs=12)
        ax.text(x, 3.1, name, ha="center", fontsize=9.5, color=GREY)
    arr(ax, (4.2, 4.8), (7.0, 7.4))
    # i and g multiply each other FIRST, and only their product reaches the line.
    # c_t = f (*) c_{t-1} + i (*) g, so the addition on the line takes one input,
    # not two.
    ax.add_patch(plt.Circle((10.6, 5.9), 0.52, fc="white", ec=INK, lw=1.6,
                            zorder=3))
    ax.text(10.6, 5.9, r"$\times$", ha="center", va="center", fontsize=14,
            zorder=4)
    arr(ax, (9.2, 4.8), (10.35, 5.45))
    arr(ax, (12.0, 4.8), (10.9, 5.45))
    arr(ax, (10.9, 6.35), (11.75, 7.45))
    ax.plot([1.2, 18.0], [2.2, 2.2], color=INK, lw=1.4)
    for x, *_ in gates:
        arr(ax, (x, 2.2), (x, 3.6), lw=1.2)
    ax.text(9.6, 1.5, r"every gate reads $[\,a^{t-1},\ x^{t}\,]$",
            ha="center", fontsize=10.5)

    arr(ax, (18.4, 8.0), (18.4, 5.6), style="-|>")
    ax.text(19.4, 6.8, r"$\tanh$", fontsize=11)
    arr(ax, (16.5, 4.8), (18.2, 5.3))
    box(ax, 17.6, 4.4, "", w=1.6, h=1.0, fc=FILL2)
    arr(ax, (18.4, 4.4), (18.4, 3.2))
    ax.text(18.4, 2.7, r"$a^{t}$", ha="center", fontsize=12)
    ax.text(11.0, 9.5, "nothing on this line passes through a weight matrix",
            ha="center", fontsize=11, color=ACCENT)
    save(fig, "d4_lstm.png")


def d5_bottleneck():
    """S5: everything the decoder knows arrives through one narrow vector."""
    fig, ax = canvas(11, 4.8, (0, 22), (0, 9),
                     "one fixed-length vector, however long the input")
    for i, wd in enumerate(["the", "dog", "ate"]):
        box(ax, 0.8 + i * 2.8, 4.8, "", w=2.0, h=1.3)
        ax.text(1.8 + i * 2.8, 4.2, wd, ha="center", fontsize=10.5,
                style="italic")
        arr(ax, (1.8 + i * 2.8, 3.9), (1.8 + i * 2.8, 4.8))
        if i < 2:
            arr(ax, (2.8 + i * 2.8, 5.45), (3.6 + i * 2.8, 5.45))
    ax.text(4.0, 7.0, "encoder", ha="center", fontsize=10.5, color=GREY)

    box(ax, 9.9, 4.9, r"$c$", w=1.1, h=1.1, fc=FILL2, fs=13)
    arr(ax, (8.4, 5.45), (9.9, 5.45), color=ACCENT, lw=2.2)
    arr(ax, (11.0, 5.45), (12.6, 5.45), color=ACCENT, lw=2.2)
    ax.text(10.45, 6.6, "one vector", ha="center", fontsize=10, color=ACCENT)

    for i, (wd, fed) in enumerate([("le", "<go>"), ("chien", "le"),
                                   ("mange", "chien")]):
        x = 12.6 + i * 3.0
        box(ax, x, 4.8, "", w=2.0, h=1.3, fc=FILL2)
        arr(ax, (x + 1.0, 6.1), (x + 1.0, 7.0))
        ax.text(x + 1.0, 7.3, wd, ha="center", fontsize=10.5, style="italic")
        arr(ax, (x + 1.0, 3.4), (x + 1.0, 4.8))
        ax.text(x + 1.0, 3.0, rf"$c$, {fed}", ha="center", fontsize=9.5,
                color=ACCENT)
        if i < 2:
            arr(ax, (x + 2.0, 5.45), (x + 3.0, 5.45))
    ax.text(16.6, 1.9, "every decoder step reads the same $c$",
            ha="center", fontsize=10.5, color=ACCENT)
    ax.text(16.6, 1.0, "a four-word source and a forty-word source get the "
            "same width", ha="center", fontsize=9.5, color=GREY)
    save(fig, "d5_bottleneck.png")


def d6_attention():
    """S6: keep every encoder state; weight them per output step."""
    fig, ax = canvas(11, 4.8, (0, 22), (0, 9),
                     "attention: score every encoder state, then average them")
    weights = [0.06, 0.71, 0.15, 0.08]
    for i, wd in enumerate(["the", "dog", "ate", "it"]):
        x = 1.2 + i * 3.4
        box(ax, x, 6.2, "", w=2.0, h=1.2)
        ax.text(x + 1.0, 5.7, wd, ha="center", fontsize=10.5, style="italic")
        ax.text(x + 1.0, 7.9, rf"$h_{{{i}}}$", ha="center", fontsize=11)
        arr(ax, (x + 1.0, 6.2), (15.4, 3.4), rad=0.18, color=ACCENT,
            lw=0.5 + 5.0 * weights[i], style="-")
        ax.text(x + 1.0, 4.9, f"{weights[i]:.2f}", ha="center", fontsize=9.5,
                color=ACCENT)
    ax.text(7.6, 8.7, "all encoder states kept, and scored directly",
            ha="center", fontsize=10.5, color=GREY)
    box(ax, 14.4, 2.1, "", w=2.0, h=1.3, fc=FILL2)
    ax.text(15.4, 1.5, "output step $i$", ha="center", fontsize=10)
    ax.text(19.6, 3.6, "line weight $=\\ a_{ij}$", fontsize=10.5, color=ACCENT)
    ax.text(19.6, 2.7, "the weights sum to 1", fontsize=9.5, color=GREY)
    ax.text(19.6, 1.9, "so they are readable", fontsize=9.5, color=GREY)
    save(fig, "d6_attention.png")


def d7_self_attention():
    """S7: one hop versus a chain of hops."""
    fig, ax = canvas(11, 4.6, (0, 22), (0, 9),
                     "one hop, not a chain of them")
    for i in range(6):
        box(ax, 1.0 + i * 3.4, 6.0, "", w=1.8, h=1.0, fc=FILL2)
    arr(ax, (1.9, 7.0), (18.9, 7.0), rad=-0.32, color=ACCENT, lw=2.2)
    ax.text(10.4, 8.6, "self-attention: one operation, any distance",
            ha="center", fontsize=11, color=ACCENT)

    for i in range(6):
        box(ax, 1.0 + i * 3.4, 2.6, "", w=1.8, h=1.0, fc="#f2f2f2")
        if i < 5:
            arr(ax, (2.8 + i * 3.4, 3.1), (4.4 + i * 3.4, 3.1), color=GREY,
                lw=1.4)
    ax.text(10.4, 1.5, "recurrence: one hop per step, and a gradient that "
            "multiplies at each", ha="center", fontsize=11, color=GREY)
    save(fig, "d7_self_attention.png")


def d8_causal_mask():
    """S7: the triangle that makes a decoder-only model."""
    fig, ax = canvas(6.4, 5.2, (-1.4, 7), (-1.6, 7),
                     "the causal mask")
    n = 6
    for i in range(n):
        for j in range(n):
            allowed = j <= i
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   fc=FILL if allowed else "white",
                                   ec=INK, lw=1.0))
            if not allowed:
                ax.plot([j + .18, j + .82], [n - 1 - i + .18, n - 1 - i + .82],
                        color=GREY, lw=1.0)
    ax.text(3.0, 6.4, "key position $j$", ha="center", fontsize=10.5)
    ax.text(-0.5, 3.0, "query\nposition $i$", ha="center", va="center",
            fontsize=10.5)
    ax.text(1.4, -0.9, "may attend", fontsize=10, color=INK)
    ax.text(4.6, -0.9, r"$-\infty$", fontsize=11, color=GREY)
    ax.text(3.0, -1.5, "so the weight is exactly zero, not merely small",
            ha="center", fontsize=9.5, color=GREY)
    save(fig, "d8_causal_mask.png")


def d2b_parameter_count():
    """S2: where 60 parameters come from, drawn as connections rather than told."""
    # Both source columns sit at the same x, stacked vertically, so neither set
    # of connections passes through the other's circles. Drawing the input to the
    # left of the previous state made the u-lines cross that column and read as
    # though the input fed it.
    fig, ax = canvas(10, 5.6, (0, 20), (-1.2, 12.4),
                     "where the 60 parameters come from, at one step")
    xsrc, xcur = 3.4, 14.6
    y_in = [10.4 - i * 1.15 for i in range(4)]
    y_prev = [5.2 - i * 1.15 for i in range(6)]
    y_cur = [9.6 - i * 1.55 for i in range(6)]

    for y in y_in:
        ax.add_patch(plt.Circle((xsrc, y), 0.42, fc=FILL2, ec=INK, lw=1.4,
                                zorder=3))
    for y in y_prev:
        ax.add_patch(plt.Circle((xsrc, y), 0.42, fc=FILL, ec=INK, lw=1.4,
                                zorder=3))
    for y in y_cur:
        ax.add_patch(plt.Circle((xcur, y), 0.42, fc=FILL3, ec=INK, lw=1.4,
                                zorder=3))
    for y in y_in:
        for yc in y_cur:
            ax.plot([xsrc + .42, xcur - .42], [y, yc], color=ACCENT, lw=0.35,
                    zorder=1, alpha=.8)
    for y in y_prev:
        for yc in y_cur:
            ax.plot([xsrc + .42, xcur - .42], [y, yc], color=INK, lw=0.35,
                    zorder=1, alpha=.4)

    ax.text(xsrc - 1.4, 8.9, "input\n4 values", ha="right", fontsize=10.5,
            color=ACCENT)
    ax.text(xsrc - 1.4, 2.6, "previous state\n6 units", ha="right",
            fontsize=10.5)
    ax.text(xcur + 1.4, 6.0, "current state\n6 units", ha="left", fontsize=10.5)
    ax.text(9.0, 11.6, r"$u$:  $4 \times 6 = 24$", ha="center", fontsize=12.5,
            color=ACCENT)
    ax.text(9.0, -0.3, r"$w$:  $6 \times 6 = 36$", ha="center", fontsize=12.5)
    ax.text(9.0, -1.0, r"$24 + 36 = 60$ weights for one step", ha="center",
            fontsize=11, color=GREY)
    save(fig, "d2b_parameter_count.png")


# ============================================================ equation plates

PLATES = {
    "e1_output_and_loss": ("the output layer and the loss", [
        (r"$p_k \;=\; \dfrac{e^{z_k}}{\sum_j e^{z_j}}$",
         "softmax: K scores become K probabilities summing to one"),
        (r"$J \;=\; -\sum_k y_k \, \log p_k$",
         "cross-entropy for K classes"),
        (r"$\dfrac{\partial J}{\partial z} \;=\; p - y$",
         "the gradient at the output layer: what was predicted, minus what happened"),
        (r"$J^{t} \;=\; -\log p_{x^{t}}$",
         "against a one-hot target every term vanishes but one"),
        (r"$J \;=\; \dfrac{1}{T}\sum_{t=1}^{T} J^{t}$",
         "for a sequence: the mean loss across steps"),
    ]),
    "e2_rnn": ("the recurrent network", [
        (r"$a^{t} \;=\; \tanh\!\left(w\,a^{t-1} \;+\; u\,x^{t} \;+\; b\right)$",
         "forward: the same w and u at every step"),
        (r"$y^{t} \;=\; v\,a^{t}$", "the output at step t"),
        (r"$\dfrac{\partial J^{t}}{\partial w} \;=\; \sum_{k}"
         r"\dfrac{\partial J^{t}}{\partial y^{t}}"
         r"\dfrac{\partial y^{t}}{\partial a^{t}}"
         r"\dfrac{\partial a^{t}}{\partial a^{k}}"
         r"\dfrac{\partial a^{k}}{\partial w}$",
         "backpropagation through time: a sum over every step at which w acted"),
        (r"$\delta^{k-1} \;=\; w^{T}\,\mathrm{diag}\!\left(1-(a^{k-1})^{2}\right)"
         r"\,\delta^{k}$",
         "the backward recurrence, in the notation of Part 1"),
    ]),
    "e3_vanishing": ("why the gradient dies", [
        (r"$\dfrac{\partial a^{t}}{\partial a^{k}} \;=\;"
         r"\prod_{j=k+1}^{t} w^{T}\,\mathrm{diag}\!\left(1-(a^{j})^{2}\right)$",
         "one factor per step of distance: a product, not a single quantity"),
        (r"$\left\|\dfrac{\partial a^{t}}{\partial a^{k}}\right\| \;\leq\;"
         r"\left(\gamma\,\sigma_{\max}(w)\right)^{\,t-k}$",
         "geometric in the distance, with ratio set by the largest singular value"),
        (r"$d^{*} \;=\; \dfrac{\log \varepsilon}{\log r}$",
         "the effective horizon: where the surviving gradient falls below epsilon"),
    ]),
    "e4_lstm": ("gating", [
        (r"$c^{t} \;=\; f^{t}\odot c^{t-1} \;+\; i^{t}\odot g^{t}$",
         "keep some of what you had, add some of what is new"),
        (r"$a^{t} \;=\; o^{t}\odot \tanh\!\left(c^{t}\right)$",
         "the hidden state is a gated view of the cell state"),
        (r"$\dfrac{\partial c^{t}}{\partial c^{t-1}} \;=\;"
         r"\mathrm{diag}\!\left(f^{t}\right)$",
         "elementwise, with no weight matrix: the geometric series is broken"),
        (r"$\dfrac{\partial c^{T}}{\partial c^{k}} \;=\;"
         r"\prod_{j=k+1}^{T}\mathrm{diag}\!\left(f^{j}\right)$",
         "a product of gates the network chose, not of a fixed matrix"),
    ]),
    "e5_bottleneck": ("the bottleneck", [
        (r"$c \;=\; a^{L}_{\mathrm{enc}}$",
         "one fixed-width vector, whatever the source length"),
        (r"$d \cdot 32 \;\geq\; L \cdot \log_{2} V$",
         "a counting bound: the state must distinguish every possible source"),
    ]),
    "e6_attention": ("attention, as Bahdanau introduced it", [
        (r"$e_{ij} = v_a^{T} \tanh(W_a\, s^{\,i-1} + U_a\, h^{\,j})$",
         "the alignment model: a small network scoring decoder state i-1 "
         "against encoder state j. Its parameters are learned like any other"),
        (r"$\alpha_{ij} = \dfrac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})}$",
         "softmax over source positions: the same object as the output "
         "layer, now ranging over j"),
        (r"$c^{\,i} = \sum_{j} \alpha_{ij}\, h^{\,j}$",
         "a fresh context for output step i, recomputed at every step"),
        (r"$s^{\,i} = f(s^{\,i-1},\ y^{\,i-1},\ c^{\,i})$",
         "and it enters the decoder recurrence alongside the previous word"),
    ]),
    "e7_transformer": ("self-attention", [
        (r"$Q = X W_q,\quad K = X W_k,\quad V = X W_v$",
         "three views of the same sequence"),
        (r"$\mathrm{Attention}(Q,K,V) \;=\;"
         r"\mathrm{softmax}\!\left(\dfrac{QK^{T}}{\sqrt{d}} + M\right)V$",
         "M is zero where attention is allowed and minus infinity where it is not"),
        (r"$PE_{t,2i} = \sin\!\left(\dfrac{t}{10000^{2i/d}}\right),\quad"
         r"PE_{t,2i+1} = \cos\!\left(\dfrac{t}{10000^{2i/d}}\right)$",
         "position added to the data, because the mechanism cannot see order"),
    ]),
}


def equation_plate(name: str, title: str, items) -> None:
    n = len(items)
    fig, ax = plt.subplots(figsize=(9.5, 1.15 + 1.5 * n))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, title, fontsize=13, color=INK, va="top")
    y = 0.90
    step = 0.86 / n
    for eq, gloss in items:
        ax.text(0.03, y - step * 0.30, eq, fontsize=17, color=INK, va="center")
        ax.text(0.03, y - step * 0.68, gloss, fontsize=10, color=GREY,
                va="center")
        y -= step
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.35)
    plt.close(fig)
    print(f"  {name}.png")


def main() -> int:
    print("schematics (hand-drawn style)")
    with plt.xkcd(scale=1.0, length=110, randomness=2):
        for fn in (d0_two_assumptions, d1_output_layer, d2_rnn,
                   d2b_parameter_count, d3_vanishing,
                   d4_lstm, d5_bottleneck, d6_attention, d7_self_attention,
                   d8_causal_mask):
            fn()
    print("equation plates (Part 1 notation)")
    for name, (title, items) in PLATES.items():
        equation_plate(name, title, items)
    print(f"done -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
