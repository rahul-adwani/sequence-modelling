"""Generate the Article 3 figures from the same code the sections run.

    python make_figures.py

Every figure recomputes its data from the section modules rather than reading a
number out of a log, so a figure cannot drift from the claim it illustrates. If a
section changes, the figures change with it.

Design notes, because they are constraints rather than taste:

* **At most three series per chart, always slots 1-3 in fixed order.** The
  reference palette documents that its first three slots clear the all-pairs
  colour-vision floors in both light and dark modes; beyond three they do not, so
  three is the cap rather than a preference. Hues are assigned to entities in a
  fixed order and never cycled.
* **No dual-axis charts.** Two measures on different scales get two stacked panels
  sharing an x-axis, never two y-scales on one plot.
* **Direct labels on every series.** Partly for readability, partly because the
  aqua slot sits below 3:1 contrast on a light surface, and the palette's relief
  rule requires visible labels wherever it is used.
* **Light mode only.** These are PNGs for a repository and a blog post, not a
  themed web page, so a single committed look is the honest choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    print("matplotlib not installed; figures skipped "
          "(every number in logs/ is produced without it)", file=sys.stderr)
    sys.exit(0)

from foundations.harness import FIG_DIR, entropy_bits, kl_bits
from foundations.s01_output_and_loss import (DelayedSource, window_model,
                                           window_predict)
from foundations.s08_tokenization import (BPE, generate_text, lexicon, rot13,
                                          take_words, zipf_weights)
from foundations.s09_llm_as_function import VOCAB, fitted
from foundations.s10_decoding import (constant_rate_tokens, exact_values,
                                      greedy_values, stationary_over_states,
                                      temper, top_p)
from foundations.s11_kv_cache import SPECS, Config, attention_share
from foundations.s12_cost import (DEVICES, SemanticCache, decode_intensity,
                                  intent_queries)

# ---------------------------------------------------------------- style tokens

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # slots 1-3, fixed order

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": "#d8d6cf",
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": "#e8e6df",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "figure.dpi": 150,
})


def _frame(ax, title: str = "", sub: str = "", ylabel: str = "",
           xlabel: str = "") -> None:
    """Recessive axes: horizontal grid only, no top or right spine."""
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        # The subtitle is drawn at y = 1.02 in axes coords, so the title has to
        # be lifted clear of it rather than sitting at the default pad.
        ax.set_title(title, loc="left", color=INK, pad=22 if sub else 8)
    if sub:
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=8.5,
                color=INK_2, va="bottom")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path.cwd())}")


# --------------------------------------------------------------------- figures

def fig_window_capacity() -> None:
    """S1.7a: two distinct failure modes of a fixed-context window."""
    src = DelayedSource()
    test = src.sample(4_000, seed=72)
    max_w = src.lag + 2
    series = {}
    for label, n, seed in (("15,000 training tokens", 15_000, 71),
                           ("240,000 training tokens", 240_000, 73)):
        train = src.sample(n, seed=seed)
        ys = []
        for w in range(max_w + 1):
            table = window_model(train, w, src.vocab)
            ys.append(float(np.mean([
                kl_bits(src.true_dist(test[:t]),
                        window_predict(table, test[:t], w, src.vocab))
                for t in range(max_w + 1, 1_400)])))
        series[label] = ys

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = list(range(max_w + 1))
    for i, (label, ys) in enumerate(series.items()):
        ax.plot(xs, ys, color=SERIES[i], marker="o", label=label, zorder=3)
        ax.annotate(label.split()[0] + " tokens", (xs[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=SERIES[i], fontsize=8.5, va="center")

    ax.axvline(src.lag, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0,
               zorder=1)
    # Captions in axes-fraction coordinates so they cannot drift onto the data
    # or off the left edge when the y-range changes.
    ax.annotate(f"dependency\nat lag {src.lag}", xy=(0.635, 0.44),
                xycoords="axes fraction", color=INK_2, fontsize=8.5,
                ha="center")
    ax.annotate("structural failure:\nthe information is absent\nfrom the input",
                xy=(0.235, 0.60), xycoords="axes fraction", color=INK_2,
                fontsize=8.5, ha="center")
    ax.annotate("statistical failure:\ncontexts grow as |V|ⁿ,\ndata per context falls",
                xy=(0.79, 0.68), xycoords="axes fraction", color=INK_2,
                fontsize=8.5, ha="center")

    _frame(ax, "Widening a fixed context window is self-defeating",
           "divergence from the true next-token distribution, by window size",
           "mean D_KL from truth (bits)", "window n (tokens conditioned on)")
    ax.set_xlim(-0.35, max_w + 1.9)
    ax.set_ylim(-0.12, 2.05)
    _save(fig, "f1_window_capacity.png")


def fig_bpe_compression() -> None:
    """S8.4: what the merges buy, and the knee where they stop buying it."""
    lex = lexicon(400, seed=31)
    weights = zipf_weights(len(lex))
    train = generate_text(18_000, seed=32, lex=lex, weights=weights)
    held = generate_text(4_000, seed=33, lex=lex, weights=weights)
    n_words = len(held.split())

    bpe = BPE(train, n_merges=350)
    grid = list(range(0, 351, 25))
    tpw = [bpe.truncated(m).n_tokens(held) / n_words for m in grid]
    gain = [(tpw[i] - tpw[i + 1]) / (grid[i + 1] - grid[i])
            for i in range(len(grid) - 1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                   gridspec_kw={"hspace": 0.28})
    ax1.plot(grid, tpw, color=SERIES[0], marker="o", zorder=3)
    _frame(ax1, "Merges buy sequence length, and then stop buying it",
           "byte-pair encoding on a corpus with an open vocabulary",
           "tokens per word\n(held out)")
    ax1.annotate(f"{tpw[0]:.2f} at 0 merges\n(characters)", (grid[0], tpw[0]),
                 xytext=(10, -4), textcoords="offset points", color=INK_2,
                 fontsize=8.5, va="top")
    ax1.annotate(f"{tpw[-1]:.2f} at {grid[-1]} merges", (grid[-1], tpw[-1]),
                 xytext=(-6, 12), textcoords="offset points", color=SERIES[0],
                 fontsize=8.5, ha="right")

    ax2.plot(grid[1:], gain, color=SERIES[1], marker="o", zorder=3)
    _frame(ax2, "", "", "tokens per word saved\nby each further merge",
           f"merges performed (vocabulary = {len(bpe.base)} characters "
           f"+ this many)")
    ax2.set_yscale("log")
    ax2.annotate("the first merges take pairs that occur everywhere;\n"
                 "the last take pairs that occur in a few words",
                 xy=(0.30, 0.62), xycoords="axes fraction", color=INK_2,
                 fontsize=8.5)
    _save(fig, "f5_bpe_compression.png")


def fig_script_cost() -> None:
    """S8.6: the same content, in two scripts, under two training mixes."""
    lex = lexicon(400, seed=31)
    weights = zipf_weights(len(lex))
    base = generate_text(8_000, seed=34, lex=lex, weights=weights)
    n = len(base.split())
    corpora = {
        "trained on 90% script A": (take_words(base, int(0.9 * n)) + " "
                                    + rot13(take_words(base, int(0.1 * n)))),
        "trained on an even mix": (take_words(base, n // 2) + " "
                                   + rot13(take_words(base, n // 2))),
    }
    eval_a = " ".join(generate_text(1_200, seed=35, lex=lex,
                                    weights=weights).split())
    eval_b = rot13(eval_a)

    labels, counts_a, counts_b = [], [], []
    for label, corpus in corpora.items():
        tok = BPE(corpus, n_merges=220)
        labels.append(label)
        counts_a.append(tok.n_tokens(eval_a))
        counts_b.append(tok.n_tokens(eval_b))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, counts_a, width=0.36, color=SERIES[0], zorder=3)
    ax.bar(x + 0.19, counts_b, width=0.36, color=SERIES[1], zorder=3)
    for i in range(len(labels)):
        ax.annotate(f"{counts_a[i]:,}", (i - 0.19, counts_a[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=SERIES[0])
        ax.annotate(f"{counts_b[i]:,}", (i + 0.19, counts_b[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=SERIES[1])
        ax.annotate(f"x{counts_b[i] / counts_a[i]:.2f}",
                    (i + 0.19, counts_b[i]), xytext=(0, 18),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.annotate("script A", (0 - 0.19, 0), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=SURFACE, fontweight="bold")
    ax.annotate("script B", (0 + 0.19, 0), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=SURFACE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _frame(ax, "The same text costs more in the script the tokenizer saw less of",
           f"identical content, {len(eval_a):,} characters either way; "
           f"script B is script A under a character substitution",
           "tokens for the same content", "")
    ax.set_ylim(0, max(counts_b) * 1.22)
    _save(fig, "f12_script_cost.png")


def fig_greedy_regret() -> None:
    """S10.6: two measures on different scales, so two panels rather than two axes."""
    lm = _decoding_model()
    logP = np.log2(np.maximum(lm.dist_tensor(), np.finfo(np.float64).tiny))
    horizons = [2, 3, 4, 5, 6, 8, 10, 12]
    frac, mean_r = [], []
    for T in horizons:
        regret = exact_values(logP, T) - greedy_values(logP, T)
        frac.append(100 * float(np.mean(regret > 1e-9)))
        mean_r.append(float(regret.mean()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                   gridspec_kw={"hspace": 0.28})
    ax1.plot(horizons, frac, color=SERIES[0], marker="o", zorder=3)
    _frame(ax1, "Greedy decoding degrades as the horizon lengthens",
           "measured at every one of the 576 start contexts, not a chosen one",
           "% of contexts where\ngreedy is suboptimal")
    ax1.set_ylim(0, 100)
    ax1.annotate(f"{frac[-1]:.0f}% at T={horizons[-1]}",
                 (horizons[-1], frac[-1]), xytext=(-8, -16),
                 textcoords="offset points", color=SERIES[0], fontsize=8.5,
                 ha="right")

    ax2.plot(horizons, mean_r, color=SERIES[1], marker="o", zorder=3)
    _frame(ax2, "", "", "mean regret (bits)", "horizon T (tokens decoded)")
    ax2.annotate(f"{mean_r[-1]:.2f} bits", (horizons[-1], mean_r[-1]),
                 xytext=(-4, 10), textcoords="offset points",
                 color=SERIES[1], fontsize=8.5, ha="right")
    _save(fig, "f2_greedy_regret.png")


def _decoding_model():
    """The model S9 fitted, not a hand-set approximation to it.

    Guessing the mixing weights here rather than taking the ones the dev split
    selected produced figures of a different model from the one the sections
    make claims about: nucleus sizes of 10 to 18 against a measured 3 to 6, and
    a speculative acceptance rate of 0.48 against a measured 0.30. Every figure
    in this file is supposed to recompute its data from the section modules, and
    that is the whole reason why.
    """
    return fitted()["model"]


def fig_temperature() -> None:
    """S10.2: the reweighting, drawn, and the ordering visibly surviving it."""
    P = _decoding_model().dist_tensor()
    u, v = VOCAB.index("the"), VOCAB.index("model")
    p = P[u, v]
    order = np.argsort(-p)[:8]
    temps = [(0.5, "T = 0.5, sharpened"), (1.0, "T = 1.0, as the model gave it"),
             (2.0, "T = 2.0, flattened")]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(order))
    for i, (T, label) in enumerate(temps):
        ax.bar(x + (i - 1) * 0.27, temper(p, T)[order], width=0.25,
               color=SERIES[i], zorder=3)
        # Axes-fraction coordinates, so the key sits in the empty upper right
        # rather than tracking a bar height that changes with the model.
        ax.annotate(label, xy=(0.42, 0.90 - 0.075 * i),
                    xycoords="axes fraction", color=SERIES[i], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([VOCAB[j] for j in order], fontsize=8.5, rotation=30,
                       ha="right")
    _frame(ax, "Temperature moves the mass and leaves the order alone",
           "the eight most probable tokens at one context, in that order at "
           "every temperature",
           "probability", "")
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylim(0, float(temper(p, temps[0][0])[order].max()) * 1.30)
    _save(fig, "f9_temperature.png")


def fig_truncation() -> None:
    """S10.3 and S10.4: a fixed k is blind to entropy, a nucleus is not."""
    P = _decoding_model().dist_tensor()
    n = P.shape[0]
    ent, keep3, nuc = [], [], []
    for a in range(n):
        for b in range(n):
            ent.append(entropy_bits(P[a, b]))
            keep3.append(float(np.sort(P[a, b])[::-1][:3].sum()))
            nuc.append(top_p(P[a, b], 0.9)[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.8), sharex=True,
                                   gridspec_kw={"hspace": 0.30})
    ax1.scatter(ent, keep3, s=9, color=SERIES[0], alpha=0.55, zorder=3,
                linewidths=0)
    _frame(ax1, "Top-k keeps a count; top-p keeps a share",
           f"every one of the {n * n} contexts of the model, one dot each",
           "mass kept by k = 3")
    # The numbers are read off the data rather than typed in, so this caption
    # cannot outlive the measurement it describes.
    ax1.annotate(f"the same k=3 keeps {100 * max(keep3):.0f}% of the mass at "
                 f"the most confident context\nand {100 * min(keep3):.0f}% at "
                 f"the least, and cannot tell the difference",
                 xy=(0.04, 0.10), xycoords="axes fraction", color=INK_2,
                 fontsize=8.5)

    ax2.scatter(ent, nuc, s=9, color=SERIES[1], alpha=0.55, zorder=3,
                linewidths=0)
    _frame(ax2, "", "", "nucleus size at p = 0.9",
           "entropy of the model's distribution at that context (bits)")
    ax2.annotate("the nucleus is an adaptive k, read off\n"
                 "the distribution the model just produced",
                 xy=(0.04, 0.74), xycoords="axes fraction", color=INK_2,
                 fontsize=8.5)
    _save(fig, "f10_truncation.png")


def fig_speculative() -> None:
    """S10.9: throughput against draft length, and the ceiling it approaches."""
    P = _decoding_model().dist_tensor()
    Q = fitted()["base"].with_lambdas((0.30, 0.70, 0.0)).dist_tensor()
    alpha = np.minimum(P, Q).sum(axis=2)
    pi = stationary_over_states(P)
    m = float((pi * alpha).sum())

    ks = list(range(0, 13))
    at_mean = [constant_rate_tokens(m, k) for k in ks]
    per_ctx = [float((pi * np.array([[constant_rate_tokens(float(alpha[a, b]), k)
                                      for b in range(P.shape[0])]
                                     for a in range(P.shape[0])])).sum())
               for k in ks]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ks, per_ctx, color=SERIES[0], marker="o", zorder=4)
    ax.plot(ks, at_mean, color=SERIES[1], marker="o", zorder=3)
    ax.axhline(1 / (1 - m), color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0,
               zorder=1)
    ax.annotate(f"ceiling 1/(1-a) = {1 / (1 - m):.2f}", (ks[0], 1 / (1 - m)),
                xytext=(4, 5), textcoords="offset points", color=INK_2,
                fontsize=8.5)
    ax.annotate("each context at its own\nacceptance rate", (ks[-1], per_ctx[-1]),
                xytext=(-6, 14), textcoords="offset points", color=SERIES[0],
                fontsize=8.5, ha="right")
    ax.annotate(f"one rate everywhere,\na = {m:.2f}", (ks[-1], at_mean[-1]),
                xytext=(-6, -20), textcoords="offset points", color=SERIES[1],
                fontsize=8.5, ha="right")
    _frame(ax, "A longer draft stops paying almost immediately",
           f"tokens produced per call to the target model, at a measured "
           f"acceptance rate of {m:.2f}",
           "tokens per verification", "draft length k")
    ax.annotate(f"the two curves differ by at most "
                f"{max(a - b for a, b in zip(per_ctx, at_mean)):.3f} tokens "
                f"here:\nJensen's correction is real and, at this draft "
                f"quality, small",
                xy=(0.30, 0.20), xycoords="axes fraction", color=INK_2,
                fontsize=8.5)
    ax.set_ylim(0.9, max(per_ctx) * 1.18)
    _save(fig, "f11_speculative.png")


def fig_prefill_composition() -> None:
    """S11.3: 'attention is quadratic' only dominates past T ~ 2d + d_ff."""
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, (label, cfg) in enumerate([
            ("d_model 4096, d_ff 14336", Config(d_model=4096, d_ff=14336)),
            ("d_model 768, d_ff 3072", Config(d_model=768, d_ff=3072))]):
        Ts = np.logspace(np.log10(64), np.log10(2 ** 19), 200)
        share = [100 * attention_share(cfg, int(t)) for t in Ts]
        ax.plot(Ts, share, color=SERIES[i], zorder=3)
        cross = 2 * cfg.d_model + cfg.d_ff
        ax.plot([cross], [50], marker="o", color=SERIES[i], zorder=4)
        # Each label goes on the side of its crossing point where that curve is
        # not: the wide model's curve is above 50% to the right, the narrow one's
        # is below 50% to the left, so the labels sit in genuinely empty space.
        side = dict(xytext=(12, -26), ha="left") if i == 0 \
            else dict(xytext=(-12, 16), ha="right")
        ax.annotate(f"{label}\n50% at T = {cross:,}", (cross, 50),
                    textcoords="offset points", color=SERIES[i], fontsize=8.5,
                    **side)

    ax.axhline(50, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.set_xscale("log")
    _frame(ax, "Attention is not the dominant prefill cost until the prompt is long",
           "prefill = 4Td² + 2T²d + 2Td·d_ff; the quadratic term's share",
           "attention share of prefill (%)", "prompt length T (tokens, log scale)")
    ax.set_ylim(0, 100)
    _save(fig, "f3_prefill_composition.png")


def fig_kv_memory() -> None:
    """S11.4: where the KV cache overtakes the weights."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    Ts = np.logspace(np.log10(512), np.log10(2 ** 19), 200)
    picks = [SPECS[0], SPECS[1], SPECS[2]]      # MHA, GQA, MQA at 8B
    for i, spec in enumerate(picks):
        gb = [spec.kv_bytes_per_token() * t / 2 ** 30 for t in Ts]
        ax.plot(Ts, gb, color=SERIES[i], zorder=3)
        label = spec.name.split(", ")[1]
        ax.annotate(label, (Ts[-1], gb[-1]), xytext=(6, 0),
                    textcoords="offset points", color=SERIES[i], fontsize=8.5,
                    va="center")

    w = picks[0].weight_bytes() / 2 ** 30
    ax.axhline(w, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.annotate(f"model weights, {w:.0f} GiB (fp16)", (Ts[0], w),
                xytext=(2, 6), textcoords="offset points", color=INK_2,
                fontsize=8.5)
    for i, spec in enumerate(picks):
        t_star = spec.crossover_tokens(1)
        if Ts[0] <= t_star <= Ts[-1]:
            ax.plot([t_star], [w], marker="o", color=SERIES[i], zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _frame(ax, "The KV cache overtakes the weights inside advertised context windows",
           "8B-class model, fp16, batch 1; dots mark where cache size equals weights",
           "KV cache size (GiB, log scale)", "context length T (tokens, log scale)")
    ax.set_xlim(Ts[0], Ts[-1] * 3.2)
    _save(fig, "f4_kv_memory.png")


def fig_roofline() -> None:
    """S12.2: prefill sits above the ridge, decode far below it."""
    dev = DEVICES[0]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    I = np.logspace(-1, 4, 400)
    attain = [dev.attainable_tflops(i) for i in I]
    ax.plot(I, attain, color=INK_2, linewidth=1.6, zorder=3)
    ax.axvline(dev.balance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0,
               zorder=1)
    ax.annotate(f"ridge point\n{dev.balance:.0f} FLOP/byte", (dev.balance, 3.0),
                xytext=(8, 0), textcoords="offset points", color=INK_2,
                fontsize=8.5)

    pts = [("decode, batch 1", decode_intensity(SPECS[1], 1, 2_048), SERIES[0]),
           ("decode, batch 32", decode_intensity(SPECS[1], 32, 2_048), SERIES[1]),
           ("prefill, T = 1,024", 1_161.0, SERIES[2])]
    for label, x, colour in pts:
        y = dev.attainable_tflops(x)
        ax.plot([x], [y], marker="o", markersize=8, color=colour, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{label}\n{100 * y / dev.tflops:.1f}% of peak", (x, y),
                    xytext=(9, -4), textcoords="offset points", color=colour,
                    fontsize=8.5, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    _frame(ax, "Decoding cannot use the arithmetic a GPU has",
           "roofline: attainable = min(peak, intensity × bandwidth)",
           "attainable TFLOP/s (log scale)",
           "arithmetic intensity (FLOP per byte, log scale)")
    _save(fig, "f6_roofline.png")


def fig_cache_tradeoff() -> None:
    """S12.4: the frontier a similarity threshold actually moves along."""
    queries = intent_queries()
    taus = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    hit, false_hit = [], []
    for tau in taus:
        c = SemanticCache(tau)
        for v, i in queries:
            c.query(v, i)
        hit.append(100 * c.hits / len(queries))
        false_hit.append(100 * c.false_hits / len(queries))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(false_hit, hit, color=SERIES[0], marker="o", zorder=3)
    # Selective labels with hand-placed offsets. The upper-left of this frontier
    # is dense -- five thresholds sit within three points of each other on the
    # hit-rate axis -- so a uniform offset guarantees collisions.
    placement = {0.40: (10, -2, "left"), 0.60: (0, -15, "center"),
                 0.70: (14, 11, "left"), 0.80: (14, -15, "left"),
                 0.85: (12, 0, "left"), 0.90: (12, 0, "left")}
    for tau, x, y in zip(taus, false_hit, hit):
        if tau in placement:
            dx, dy, ha = placement[tau]
            ax.annotate(f"τ = {tau:.2f}", (x, y), xytext=(dx, dy),
                        textcoords="offset points", color=INK_2, fontsize=8.5,
                        ha=ha)
    ax.annotate("tightening to τ = 0.80 removes almost every\n"
                "wrong answer and costs 3 points of hit rate;\n"
                "past 0.85 the cache stops working",
                xy=(0.36, 0.30), xycoords="axes fraction", color=INK_2,
                fontsize=8.5)
    _frame(ax, "A semantic cache buys hit rate with wrong answers",
           "each point is one similarity threshold; 24 intents, 600 queries",
           "cache hit rate (%)", "queries answered from the wrong entry (%)")
    ax.set_xlim(-3, max(false_hit) + 14)
    ax.set_ylim(0, 108)
    _save(fig, "f7_cache_tradeoff.png")


def fig_cost_inversion() -> None:
    """S12.7: the ranking the token bill reverses."""
    configs = [("cheap,\nweaker", 0.15, 5_400, 0.55),
               ("cheap,\nmore steps", 0.15, 9_200, 0.72),
               ("mid", 0.60, 4_800, 0.86),
               ("strong,\nfewer steps", 3.00, 3_100, 0.94)]
    names = [c[0] for c in configs]
    tokens_only = [c[2] * c[1] / 1e6 for c in configs]
    with_esc = [t + (1 - c[3]) * 2.00 for t, c in zip(tokens_only, configs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.9),
                                   gridspec_kw={"wspace": 0.34})
    x = np.arange(len(configs))
    for ax, vals, title, colour in (
            (ax1, tokens_only, "Billed: token cost per attempt", SERIES[0]),
            (ax2, with_esc, "Paid: total cost per task", SERIES[1])):
        ax.bar(x, vals, width=0.62, color=colour, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        _frame(ax, title, "", "$ per task", "")
        best = int(np.argmin(vals))
        for i, v in enumerate(vals):
            ax.annotate(f"${v:.4f}" if v < 0.01 else f"${v:.3f}", (i, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8,
                        color=INK if i == best else INK_2,
                        fontweight="bold" if i == best else "normal")
        ax.set_ylim(0, max(vals) * 1.22)
    ax2.text(0.0, -0.30, "adding $2.00 for a human to pick up each failure",
             transform=ax2.transAxes, fontsize=8.5, color=INK_2)
    _save(fig, "f8_cost_inversion.png")


def main() -> int:
    print("generating Article 3 figures")
    for fn in (fig_window_capacity, fig_bpe_compression, fig_script_cost,
               fig_temperature, fig_truncation, fig_speculative,
               fig_greedy_regret, fig_prefill_composition, fig_kv_memory,
               fig_roofline, fig_cache_tradeoff, fig_cost_inversion):
        fn()
    print(f"done: {len(list(FIG_DIR.glob('*.png')))} figures in "
          f"{FIG_DIR.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
