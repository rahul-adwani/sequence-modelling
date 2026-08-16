"""S3. Why the gradient dies over distance.

S2 ended on an observation rather than an explanation: in a 12-step sequence, the
last three steps supplied 72% of the gradient and the first three supplied 1.4%.
This section explains it, and the explanation is short enough to state in one
line before deriving it.

Look again at the third factor in the backpropagation-through-time sum:

    dJ_t/dW = sum_k (dJ_t/dy_t)(dy_t/ds_t) (ds_t/ds_k) (ds_k/dW)
                                            ^^^^^^^^^

`ds_t/ds_k` is not one quantity. It is the effect of a nudge at step k on the
state at step t, and to get there the nudge has to pass through every step in
between. So it is a *product*, one factor per step of distance:

    ds_t/ds_k = prod_{j=k+1..t} ds_j/ds_{j-1},    ds_j/ds_{j-1} = diag(1 - s_j^2) W

Multiply a number by itself d times and you get a geometric sequence. Multiply a
*matrix* by itself d times and the same thing happens to its magnitude, governed
by its largest singular value. Vanishing gradients are not a quirk of training
recurrent networks; they are a geometric series, and this section measures its
ratio.

The crucial detail -- the one that separates an RNN from a deep feedforward
network -- is that this is the *same* matrix every time. A 50-layer feedforward
network multiplies 50 different matrices going backwards, and they can and do
compensate for one another. An RNN unrolled 50 steps multiplies one matrix by
itself 50 times. There is nothing to compensate.
"""
from __future__ import annotations

import numpy as np

from .harness import Section, rng
from .s02_recurrence import RNN, sequence

D_IN, D_HIDDEN, D_OUT = 4, 16, 3


def decay_profile(sigma: float, T: int = 60, seed: int = 4,
                  activation: str = "tanh", xseed: int | None = None
                  ) -> list[float]:
    """||dJ/ds_k|| against distance travelled, for a W with this spectral radius.

    Normalised so element 0 is 1.0, because the question is what the *journey*
    costs, not how large the gradient was when it set off. `xseed` varies the
    input sequence while holding the weights fixed, which is what 3.8 needs to
    separate the spread across inputs from the spread across initialisations.
    """
    net = RNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT, sigma_max=sigma,
              activation=activation, seed=seed)
    xs = sequence(T, D_IN, seed=seed + 100 if xseed is None else xseed)
    target = rng(seed + 200).normal(size=D_OUT)
    norms = net.state_gradient_norms(xs, target)
    return [n / norms[0] for n in norms]


def fitted_ratio(profile: list[float], lo: int = 5, hi: int = 40) -> float:
    """The per-step multiplier, recovered by fitting a line to log ||grad||.

    A geometric sequence is a straight line in log space, and its slope is the
    log of the common ratio. Fitting over a middle window avoids the first few
    steps, where the initial state and the output layer still dominate.
    """
    d = np.arange(lo, hi + 1)
    y = np.log(np.maximum(np.asarray(profile[lo:hi + 1]), 1e-300))
    slope = float(np.polyfit(d, y, 1)[0])
    return float(np.exp(slope))


def log_linear_r2(profile: list[float], lo: int = 5, hi: int = 40) -> float:
    """How well log ||grad|| is a straight line in distance.

    This is the direct test of the section's central claim. "Geometric sequence"
    means exactly "straight line in log space", so the coefficient of
    determination of that fit is the statistic that decides it -- rather than
    spot-checking whether one measured drop matches ratio^k, which conflates the
    fit's quality with the particular points chosen.
    """
    d = np.arange(lo, hi + 1)
    y = np.log(np.maximum(np.asarray(profile[lo:hi + 1]), 1e-300))
    slope, intercept = np.polyfit(d, y, 1)
    resid = y - (slope * d + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def horizon(ratio: float, threshold: float = 1e-6) -> float:
    """How far back the gradient travels before falling below `threshold`.

        ratio^d = threshold   =>   d = log(threshold) / log(ratio)

    Undefined for ratio >= 1, where the gradient grows instead of shrinking.
    """
    if ratio >= 1.0:
        return float("inf")
    return float(np.log(threshold) / np.log(ratio))


def clip(g: np.ndarray, max_norm: float) -> np.ndarray:
    """Gradient clipping: rescale to `max_norm` if it is larger, else leave it.

    Note what this can and cannot touch. It is a ceiling, so it bounds a gradient
    that has grown too large. It has no floor, so a gradient that has shrunk to
    1e-12 passes through completely unchanged.
    """
    n = float(np.linalg.norm(g))
    return g * (max_norm / n) if n > max_norm else g


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s03", "Why the gradient dies over distance",
                "the product hiding inside backpropagation through time").open()

    # ------------------------------------------------------- 3.1 the product
    s.h("3.1  The factor that is really a product")
    s.eq("ds_t/ds_k = prod_{j=k+1..t} diag(1 - s_j^2) W",
         "one factor per step of distance between the nudge and the loss")
    s.eq("||ds_t/ds_k||_2 <= (gamma * sigma_max(W))^(t-k)",
         "a sufficient upper bound, from submultiplicativity of the 2-norm: "
         "gamma bounds |tanh'| and sigma_max(W) bounds ||W||_2. It bounds growth; "
         "it does not by itself determine it")
    s.say()
    s.say("  Two words worth pinning down, because everything below turns on "
          "them.")
    s.say()
    s.say("    singular value  -- how much a matrix can stretch a vector. The "
          "largest\n                       one, sigma_max, is the worst case: no "
          "vector is stretched\n                       by more than this factor.")
    s.say("    spectral radius -- used loosely here for that same largest "
          "singular value,\n                       the quantity that decides "
          "whether repeated multiplication\n                       grows or "
          "shrinks a vector.")
    s.say()
    s.say("  If sigma_max < 1, each step shrinks the gradient by a roughly "
          "constant\n  factor. Over d steps that is that factor to the power d. "
          "That is all\n  the vanishing gradient problem is.")

    # ------------------------------------------- 3.2 the folk criterion is wrong
    s.h("3.2  The usual explanation names the wrong quantity")
    s.say("  The standard telling goes: W is drawn from a normal distribution so "
          "its\n  entries are mostly below 1, tanh' is below 1, and we are "
          "therefore\n  multiplying a lot of small numbers together.")
    s.say()
    s.say("  The conclusion is right and the criterion is not. What controls a "
          "repeated\n  matrix product is the largest singular value, and that is "
          "not the size of\n  a typical entry -- it is a property of the matrix "
          "as a whole.")
    rows = []
    for d in (4, 16, 64, 256):
        for name, sd in (("N(0, 1)", 1.0), ("N(0, 1/d)", None)):
            scale = sd if sd is not None else 1 / np.sqrt(d)
            W = rng(3).normal(0, scale, (d, d))
            rows.append([d, name, round(float(np.abs(W).mean()), 3),
                         f"{float((np.abs(W) < 1).mean()):.0%}",
                         round(float(np.linalg.norm(W, 2)), 2)])
    s.table(["width d", "initialisation", "mean |W_ij|", "entries below 1",
             "sigma_max(W)"], rows, align="rlrrr")

    plain = [r for r in rows if r[1] == "N(0, 1)"]
    scaled = [r for r in rows if r[1] == "N(0, 1/d)"]
    s.check("the share of entries below 1 barely moves with width while "
            "sigma_max grows without bound",
            max(abs(float(r[3].rstrip('%')) - 68) for r in plain) < 8
            and plain[-1][4] > 6 * plain[0][4],
            f"about 68% of entries are below 1 at every width, yet sigma_max "
            f"goes {plain[0][4]} -> {plain[-1][4]} from d=4 to d=256")
    s.check("for standard normal entries sigma_max grows as about 2*sqrt(d)",
            all(abs(r[4] / (2 * np.sqrt(r[0])) - 1) < 0.2 for r in plain[1:]),
            "; ".join(f"d={r[0]}: {r[4]} against 2*sqrt(d) = "
                      f"{2 * np.sqrt(r[0]):.1f}" for r in plain[1:]))
    s.check("even the modern 1/sqrt(d) scaling leaves sigma_max near 2, not "
            "below 1",
            all(1.5 < r[4] < 2.5 for r in scaled[1:]),
            "sigma_max is " + ", ".join(str(r[4]) for r in scaled[1:])
            + " -- every entry is below 1 and the matrix still amplifies")
    s.note("so the matrix factor typically *grows* the gradient rather than "
           "shrinking it, and the contraction in a vanilla RNN comes from the "
           "other term: tanh' is at most 1 and well below it once units "
           "saturate. That matters because it tells you what to fix and what "
           "not to. It is also why orthogonal initialisation, which sets every "
           "singular value to exactly 1, is the standard choice for recurrent "
           "weights -- it removes the matrix from the argument entirely and "
           "leaves the activation as the only remaining factor. The sections "
           "below therefore control sigma_max directly rather than trusting a "
           "distribution to supply it.")

    # ------------------------------------------------------ 3.3 measure it
    s.h("3.3  Measuring the ratio")
    s.say("  Build the same network with W rescaled to different sigma_max, run "
          "a\n  60-step sequence, and watch the gradient shrink on its way back.")
    rows = []
    for sigma in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        prof = decay_profile(sigma)
        net = RNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT,
                  sigma_max=sigma, seed=4)
        st, _ = net.forward(sequence(60, D_IN, seed=104))
        mean_fp = float(np.mean([np.mean(1 - x ** 2) for x in st[1:]]))
        rows.append([sigma, round(fitted_ratio(prof), 4), round(mean_fp, 3),
                     f"{prof[10]:.2e}", f"{prof[25]:.2e}", f"{prof[50]:.2e}"])
    s.table(["sigma_max(W)", "fitted per-step ratio", "mean tanh'",
             "grad at d=10", "d=25", "d=50"], rows, align="rrrrrr")

    r2s = [log_linear_r2(decay_profile(r[0])) for r in rows]
    s.table(["sigma_max(W)", "R^2 of the log-linear fit"],
            [[r[0], round(v, 5)] for r, v in zip(rows, r2s)], align="rr")
    s.check("in the contracting regime the decay is a straight line in log "
            "space, which is precisely what 'geometric' means",
            all(v > 0.99 for r, v in zip(rows, r2s) if r[0] <= 2.0),
            "R^2 is " + ", ".join(f"{v:.4f}" for r, v in zip(rows, r2s)
                                  if r[0] <= 2.0)
            + " for sigma_max up to 2")
    s.check("but the geometric description breaks down as the crossover is "
            "approached",
            r2s[-1] < 0.5 and r2s[0] > 0.99,
            f"R^2 falls from {r2s[0]:.4f} at sigma_max = {rows[0][0]} to "
            f"{r2s[-1]:.4f} at {rows[-1][0]}, because near the crossover the "
            "per-step factor stops being constant: units saturate by different "
            "amounts at different points in the sequence, so no single ratio "
            "describes the whole journey")
    s.note("that limitation is worth stating precisely, because it is where the "
           "geometric-series picture earns its keep and where it does not. In "
           "the regime that actually matters -- gradients contracting, which "
           "3.3 shows is the default for a saturating network -- the decay is "
           "geometric to four decimal places, and the horizon calculation in "
           "3.5 follows from it. Near the explosion crossover the picture is "
           "genuinely a worse model of what is happening.")
    s.check("with a saturating activation the ratio stays below 1 far past "
            "sigma_max = 1",
            rows[1][1] < 1.0 and rows[3][1] < 1.0,
            f"sigma_max = 1 gives a ratio of {rows[1][1]:.3f} and sigma_max = "
            f"{rows[3][0]} still only {rows[3][1]:.3f}; the crossover is near "
            f"sigma_max = {rows[-1][0]}, not 1")
    s.check("tanh is self-limiting: raising sigma_max lowers the activation "
            "derivative",
            all(rows[i][2] > rows[i + 1][2] for i in range(len(rows) - 1)),
            "mean tanh' falls " + " > ".join(f"{r[2]:.3f}" for r in rows)
            + " as sigma_max rises, because larger pre-activations saturate more "
            "units")
    s.note("that last row is the interesting one, and it is not what the usual "
           "telling predicts. A saturating activation fights explosion: push "
           "sigma_max up and the pre-activations grow, which drives units into "
           "the flat regions of tanh, which drops tanh' and pulls the product "
           "back down. The two failures are therefore not mirror images around "
           "sigma_max = 1. Vanishing is the default and the crossover into "
           "genuine explosion sits far out. This is why exploding gradients are "
           "associated with unbounded activations and very large weights, while "
           "the saturating recurrent networks of the era mostly just went quiet.")

    # ------------------------------------------------ 3.3 tanh makes it worse
    s.h("3.4  The activation makes it worse, never better")
    s.eq("tanh'(z) = 1 - tanh(z)^2 = 1 - s^2,   which is 1 only at s = 0",
         "so the diag factor is at most 1 and is strictly less whenever a unit "
         "is doing anything")
    rows = []
    for sigma in (0.9, 0.95, 1.0):
        prof = decay_profile(sigma)
        ratio = fitted_ratio(prof)
        rows.append([sigma, round(ratio, 4), round(ratio / sigma, 4)])
    s.table(["sigma_max(W)", "measured ratio", "measured / bound"], rows,
            align="rrr")
    s.check("the measured decay is strictly faster than the sigma_max bound",
            all(r[2] < 1.0 for r in rows),
            "measured/bound is " + ", ".join(f"{r[2]:.3f}" for r in rows)
            + " -- the bound assumes tanh' = 1 everywhere, which happens only "
            "at exactly zero input")
    s.check("even setting sigma_max to exactly 1 does not stop the decay",
            rows[-1][1] < 1.0,
            f"at sigma_max = 1 the measured ratio is still {rows[-1][1]:.3f}, "
            "because the saturating activation contributes a factor below 1 at "
            "every step")
    s.note("this is why tuning the initialisation does not rescue a vanilla RNN. "
           "You can set sigma_max to 1 and the gradient still dies, because the "
           "activation supplies its own contraction. Pushing sigma_max above 1 "
           "to compensate buys explosion instead -- and the two failures are not "
           "symmetric, which 3.6 is about.")

    # ----------------------------------------------------- 3.4 the horizon
    s.h("3.5  How far back the network can actually learn")
    s.eq("d* = log(threshold) / log(ratio)",
         "the distance at which the surviving gradient falls below a threshold "
         "-- here one millionth of where it started")
    rows = []
    for sigma in (0.5, 0.7, 0.9, 0.95, 0.99):
        r = fitted_ratio(decay_profile(sigma))
        rows.append([sigma, round(r, 4), round(horizon(r), 1)])
    s.table(["sigma_max(W)", "measured ratio", "effective horizon (steps)"],
            rows, align="rrr")
    s.check("the effective memory horizon is a handful of steps unless the "
            "ratio is pushed against 1",
            rows[0][2] < 20 and rows[-1][2] > rows[0][2],
            f"{rows[0][2]:.0f} steps at sigma={rows[0][0]}, rising to "
            f"{rows[-1][2]:.0f} at sigma={rows[-1][0]}")
    s.note("the horizon depends on the ratio only through a logarithm, so buying "
           "reach by nudging sigma_max towards 1 is a bad trade: it costs "
           "stability quickly and buys distance slowly. A network that needs to "
           "connect a pronoun to a name forty words earlier is being asked for a "
           "horizon these numbers do not offer.")

    # ------------------------------------------------------- 3.5 clipping
    s.h("3.6  Clipping fixes one failure and cannot touch the other")
    s.say("  Gradient clipping rescales the gradient when its norm exceeds a "
          "ceiling.\n  It is the standard answer to exploding gradients, and it "
          "works. Ask what\n  it does to the other case.")
    rows = []
    # sigma_max = 6 rather than something just above 1, because 3.3 showed the
    # activation keeps the product contracting well past 1. The exploding regime
    # has to be entered on the evidence, not on the assumption.
    for sigma, label in ((6.0, "exploding"), (0.7, "vanishing")):
        prof = decay_profile(sigma)
        raw = prof[40]
        assert raw > 0
        clipped = float(np.linalg.norm(clip(np.array([raw]), max_norm=1.0)))
        rows.append([label, sigma, f"{raw:.3e}", f"{clipped:.3e}",
                     "bounded" if clipped < raw else "unchanged"])
    s.table(["regime", "sigma_max", "gradient at d=40", "after clipping at 1.0",
             "effect"], rows, align="lrrrl")
    s.check("clipping bounds an exploding gradient",
            float(rows[0][3]) < float(rows[0][2]),
            f"{rows[0][2]} clipped to {rows[0][3]}")
    s.check("clipping does nothing whatever to a vanishing one",
            float(rows[1][3]) == float(rows[1][2]),
            f"{rows[1][2]} passes through unchanged -- clipping is a ceiling, "
            "and the problem here is a floor")
    s.note("the asymmetry is the point. Explosion is a scale problem and a scale "
           "fix solves it. Vanishing is an information problem: by the time the "
           "gradient has been multiplied down to 1e-13 the direction it was "
           "carrying is gone into rounding error, and no rescaling recovers a "
           "signal that is no longer there. This is why exploding gradients are "
           "a footnote in practice and vanishing gradients changed the "
           "architecture.")

    # ----------------------------------------------- 3.6 the distinction to keep
    s.h("3.7  It can represent the dependency; it cannot learn it")
    s.say("  Worth separating two things that get run together.")
    s.say()
    s.say("    can the architecture express the answer?   yes -- a suitable W "
          "exists\n    can gradient descent find it?             no -- the "
          "gradient that would\n                                              "
          "point towards it arrives as noise")
    net_a = RNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT, sigma_max=0.7,
                seed=9)
    xs = sequence(50, D_IN, seed=11)
    target = rng(12).normal(size=D_OUT)
    base = net_a.loss(xs, target)
    perturbed = RNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT,
                    sigma_max=0.7, seed=9)
    perturbed.U[:, 0] += 0.5                       # change how step 0's input enters
    s.kv("loss before perturbing the input weights", round(base, 6))
    s.kv("loss after", round(perturbed.loss(xs, target), 6))
    prof = decay_profile(0.7)
    s.kv("gradient reaching step 0, relative", f"{prof[-1]:.2e}")
    s.check("the forward pass still carries early information, while the "
            "backward pass no longer carries the gradient for it",
            abs(perturbed.loss(xs, target) - base) > 1e-6 and prof[-1] < 1e-8,
            f"changing how the first input enters moves the loss by "
            f"{abs(perturbed.loss(xs, target) - base):.4f}, so the dependency is "
            f"expressible; but the gradient arriving there is {prof[-1]:.1e} of "
            "its starting magnitude, so training cannot exploit it")
    s.note("this distinction decides what the fix has to be. The problem is not "
           "that the model is too small or the state too narrow -- it is that "
           "the path the gradient travels multiplies it down. So the repair must "
           "change the *path*, not the capacity. S4 is exactly that: a route "
           "from s_t back to s_{t-1} that adds instead of multiplying.")

    # ------------------------------------------- 3.8 so where does explosion live
    s.h("3.8  So where do exploding gradients actually come from?")
    s.say("  3.3 makes this a fair question. If tanh damps its own gradient hard "
          "enough\n  that sigma_max = 3 still contracts, why is gradient clipping "
          "in every\n  recurrent training loop ever written? Two answers, and "
          "the second is the\n  one that bites.")

    s.say()
    s.say("  (a) A non-saturating activation removes the brake entirely.")
    s.eq("relu'(z) = 1 wherever z > 0",
         "exactly 1, not 'at most 1' -- there is no shrinking factor to "
         "counteract sigma_max")
    rows = []
    for sigma in (0.8, 1.2, 2.0, 3.0):
        t_prof = decay_profile(sigma)
        r_prof = decay_profile(sigma, activation="relu")
        rows.append([sigma, f"{t_prof[40]:.2e}", f"{r_prof[40]:.2e}",
                     f"{r_prof[40] / max(t_prof[40], 1e-300):.1e}x"])
    s.table(["sigma_max(W)", "tanh, gradient at d=40", "ReLU, same",
             "ReLU / tanh"], rows, align="rrrr")
    s.check("a non-saturating activation reaches explosion where a saturating "
            "one is still contracting",
            float(rows[-1][2]) > 1.0 and float(rows[-1][1]) < 1.0,
            f"at sigma_max = {rows[-1][0]} ReLU gives {rows[-1][2]} while tanh "
            f"gives {rows[-1][1]} -- {rows[-1][3]} apart, because tanh's "
            "derivative falls as its input grows and ReLU's does not")

    s.say()
    s.say("  (b) The real answer: gradient size is a distribution, not a number.")
    s.say("      Everything above reports one input sequence. Run many, and the "
          "spread\n      is the finding.")
    vals = np.array([decay_profile(2.5, seed=4, xseed=1_000 + i)[40]
                     for i in range(400)])
    s.table(["statistic", "gradient at d=40"],
            [["median", f"{np.median(vals):.3e}"],
             ["mean", f"{vals.mean():.3e}"],
             ["99th percentile", f"{np.percentile(vals, 99):.3e}"],
             ["maximum", f"{vals.max():.3e}"]], align="lr")
    s.kv("max / median", f"{vals.max() / np.median(vals):.0f}x")
    s.kv("mean / median", f"{vals.mean() / np.median(vals):.1f}x")
    s.check("the distribution of gradient magnitudes is heavy-tailed across "
            "inputs",
            vals.mean() > 3 * np.median(vals)
            and vals.max() > 100 * np.median(vals),
            f"over 400 input sequences at sigma_max = 2.5 the mean is "
            f"{vals.mean() / np.median(vals):.1f}x the median and the worst "
            f"sequence is {vals.max() / np.median(vals):.0f}x it -- a mean well "
            "above the median is the signature of a tail")
    s.note("that is where exploding gradients come from, and why they are so "
           "disruptive relative to how rare they are. The typical sequence "
           "contracts, so training looks healthy for hundreds of updates. Then "
           "one batch happens to drive the units into their linear region, where "
           "tanh' is near 1 and sigma_max is unopposed, the gradient arrives "
           "hundreds of times larger than usual, and a single step throws the "
           "weights somewhere useless. Explosion is not a regime you are in, it "
           "is an outlier you eventually meet.")
    s.say()
    s.note("read that way, clipping stops looking like a hack. It is not "
           "correcting a systematic bias -- 3.6 showed it does nothing to the "
           "median case -- it is a bound on the tail, and bounding the tail is "
           "exactly the right response to a heavy-tailed distribution. It also "
           "explains why clipping is nearly free: it fires on a small fraction "
           "of batches and leaves the rest untouched. And it explains the last "
           "piece of the puzzle, which is that neither clipping nor "
           "initialisation touches the vanishing problem at all. That one needs "
           "a different path through time, which is S4.")

    s.close()
    return s


if __name__ == "__main__":
    run()
