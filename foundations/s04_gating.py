"""S4. Gating and the LSTM.

S3 ended with a precise diagnosis. The gradient travelling from step t back to
step k is a product of one Jacobian per step,

    ds_t/ds_k = prod_j  W^T diag(1 - s_j^2)

and a product of d roughly-equal factors is a geometric sequence. The measured
ratio was around 0.35, which puts the effective memory horizon at eight to
thirteen steps. The dependency is expressible; it is simply not learnable.

So the repair has to change the *path*, not the capacity. That is the whole idea
of gating, and it is worth stating before any of the notation arrives:

    what if the route from one step to the next were addition, not multiplication?

Addition has a derivative of 1. A quantity that is carried forward by being added
to, rather than transformed by, has a backward path that does not shrink. The
LSTM builds exactly that: a **cell state** that each step may add to, erase from,
or leave alone, alongside the ordinary transformed state.

The claim usually made for it -- "the LSTM solves vanishing gradients" -- is too
strong, and 4.5 measures where it stops being true. What the LSTM provides is one
route back through time whose derivative is controlled by a gate rather than by a
matrix product. Other routes still exist and still decay. The honest statement is
that gating supplies an uninterrupted path, not that it removes every interrupted
one.

References: Hochreiter & Schmidhuber (1997) for the LSTM; Cho et al. (2014) for
the GRU; Hochreiter's 1991 diploma thesis and Bengio, Simard & Frasconi (1994)
for the original statements of the problem S3 measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, rng
from .s02_recurrence import RNN, sequence
from .s03_vanishing import fitted_ratio, horizon

D_IN, D_HIDDEN = 4, 16


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


@dataclass
class LSTM:
    """One LSTM cell, written so the additive path is visible in the code.

    The gates, in the order they are used:

    * **forget gate** `f_t` -- how much of the previous cell state to keep. A
      value near 1 keeps it; near 0 erases it.
    * **input gate** `i_t` -- how much of the new candidate to write.
    * **candidate** `g_t` -- what to write, if anything.
    * **output gate** `o_t` -- how much of the cell state to expose as the
      hidden state this step.

    Each is a vector, one number per unit, produced by a small network reading
    the previous hidden state and the current input. That is the second idea in
    the LSTM and it is easy to miss: the gates are *learned and
    input-dependent*, so the network decides per step and per unit what to carry.

    `forget_bias` starts the forget gate near 1 so the cell state is carried by
    default and has to be actively erased. That is the standard initialisation
    and 4.4 shows why it matters so much.
    """

    d_in: int = D_IN
    d_hidden: int = D_HIDDEN
    forget_bias: float = 1.0
    seed: int = 5

    def __post_init__(self) -> None:
        g = rng(self.seed)
        d, n = self.d_hidden, self.d_hidden + self.d_in
        scale = 1.0 / np.sqrt(n)
        # One matrix producing all four gate pre-activations at once, which is
        # how it is done in practice: four small matmuls become one large one.
        self.W = g.normal(0, scale, (4 * d, n))
        self.b = np.zeros(4 * d)
        self.b[d:2 * d] = self.forget_bias          # the forget block

    def gates(self, h_prev: np.ndarray, x: np.ndarray) -> dict:
        d = self.d_hidden
        z = self.W @ np.concatenate([h_prev, x]) + self.b
        return {"i": sigmoid(z[:d]), "f": sigmoid(z[d:2 * d]),
                "o": sigmoid(z[2 * d:3 * d]), "g": np.tanh(z[3 * d:])}

    def step(self, h_prev: np.ndarray, c_prev: np.ndarray, x: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, dict]:
        """One step.

            c_t = f_t (*) c_{t-1}  +  i_t (*) g_t        <- the additive path
            h_t = o_t (*) tanh(c_t)

        `(*)` is elementwise multiplication. Read the first line as: keep some of
        what you had, add some of what is new. The previous cell state is never
        passed through a weight matrix on its way to the next one, and that
        single fact is what 4.3 measures.
        """
        gt = self.gates(h_prev, x)
        c = gt["f"] * c_prev + gt["i"] * gt["g"]
        h = gt["o"] * np.tanh(c)
        return h, c, gt

    def forward(self, xs: np.ndarray) -> tuple[list, list, list]:
        h = np.zeros(self.d_hidden)
        c = np.zeros(self.d_hidden)
        hs, cs, gs = [h], [c], []
        for x in xs:
            h, c, gt = self.step(h, c, x)
            hs.append(h)
            cs.append(c)
            gs.append(gt)
        return hs, cs, gs

    # -------------------------------------------------------- the two paths

    def highway_decay(self, xs: np.ndarray) -> np.ndarray:
        """The product of forget gates: the derivative along the cell-state path.

            dc_t/dc_{t-1}  =  diag(f_t)      (holding h_{t-1} fixed)
            dc_T/dc_k      =  prod_j diag(f_j)

        Elementwise, so the whole product is just the forget gates multiplied
        together coordinate by coordinate. Compare with S3, where every step
        applied a full matrix. Returned as the norm at each distance, normalised
        to 1 at distance 0, so it lines up with S3's `decay_profile`.
        """
        _, _, gs = self.forward(xs)
        T = len(xs)
        acc = np.ones(self.d_hidden)
        out = [float(np.linalg.norm(acc))]
        for t in range(T - 1, -1, -1):
            acc = acc * gs[t]["f"]
            out.append(float(np.linalg.norm(acc)))
        return np.asarray(out) / out[0]

    def full_state_decay(self, xs: np.ndarray, seed: int = 21) -> np.ndarray:
        """The complete backward pass for the state, both routes together.

        The cell-state highway is not the only way back. `h_{t-1}` feeds all four
        gates, and `h_{t-1}` itself comes from `c_{t-1}`, so there is a second
        route that does pass through the weight matrix and does behave like S3's
        product. This computes both and adds them, which is what actually happens
        during training.

        Keeping the two separate in 4.3 and together in 4.5 is the point of the
        section: the honest claim about the LSTM is about the first route, and
        the second one has not gone anywhere.
        """
        hs, cs, gs = self.forward(xs)
        T, d = len(xs), self.d_hidden
        dh = rng(seed).normal(size=d)                # gradient arriving at h_T
        dc = np.zeros(d)
        out = []
        for t in range(T - 1, -1, -1):
            gt = gs[t]
            tanh_c = np.tanh(cs[t + 1])
            dc = dc + dh * gt["o"] * (1.0 - tanh_c ** 2)
            out.append(float(np.linalg.norm(np.concatenate([dh, dc]))))

            # Gradients into the four pre-activations at this step.
            di = dc * gt["g"] * gt["i"] * (1 - gt["i"])
            df = dc * cs[t] * gt["f"] * (1 - gt["f"])
            do = dh * tanh_c * gt["o"] * (1 - gt["o"])
            dg = dc * gt["i"] * (1 - gt["g"] ** 2)
            dz = np.concatenate([di, df, do, dg])

            dh = (self.W.T @ dz)[:d]                 # the matrix route back
            dc = dc * gt["f"]                        # the additive route back
        out.append(float(np.linalg.norm(np.concatenate([dh, dc]))))
        arr = np.asarray(out)
        return arr / arr[0]

    def numeric_cell_jacobian(self, h_prev, c_prev, x, eps: float = 1e-6):
        """dc_t/dc_{t-1} by finite differences, holding h_{t-1} fixed."""
        d = self.d_hidden
        J = np.zeros((d, d))
        for j in range(d):
            cp, cm = c_prev.copy(), c_prev.copy()
            cp[j] += eps
            cm[j] -= eps
            _, c_plus, _ = self.step(h_prev, cp, x)
            _, c_minus, _ = self.step(h_prev, cm, x)
            J[:, j] = (c_plus - c_minus) / (2 * eps)
        return J


@dataclass
class GRU:
    """The same additive trick with two gates instead of three.

        z_t = sigmoid(...)                    update gate
        r_t = sigmoid(...)                    reset gate
        n_t = tanh(W_n x_t + r_t (*) (U_n h_{t-1}))
        h_t = (1 - z_t) (*) n_t  +  z_t (*) h_{t-1}

    There is no separate cell state: the hidden state itself is carried by the
    convex combination on the last line. The backward factor along that carried
    route is `diag(z_t)`, which is the same elementwise, matrix-free derivative
    the LSTM gets from its forget gate. Fewer parameters, same mechanism.
    """

    d_in: int = D_IN
    d_hidden: int = D_HIDDEN
    update_bias: float = 1.0
    seed: int = 6

    def __post_init__(self) -> None:
        g = rng(self.seed)
        d, n = self.d_hidden, self.d_hidden + self.d_in
        scale = 1.0 / np.sqrt(n)
        self.Wz = g.normal(0, scale, (d, n))
        self.Wr = g.normal(0, scale, (d, n))
        self.Wn_x = g.normal(0, scale, (d, self.d_in))
        self.Un = g.normal(0, scale, (d, d))
        self.bz = np.full(d, self.update_bias)

    def forward(self, xs: np.ndarray) -> tuple[list, list]:
        h = np.zeros(self.d_hidden)
        hs, zs = [h], []
        for x in xs:
            cat = np.concatenate([h, x])
            z = sigmoid(self.Wz @ cat + self.bz)
            r = sigmoid(self.Wr @ cat)
            n = np.tanh(self.Wn_x @ x + r * (self.Un @ h))
            h = (1.0 - z) * n + z * h
            hs.append(h)
            zs.append(z)
        return hs, zs

    def carry_decay(self, xs: np.ndarray) -> np.ndarray:
        """Product of update gates: the GRU's equivalent of the LSTM highway."""
        _, zs = self.forward(xs)
        acc = np.ones(self.d_hidden)
        out = [float(np.linalg.norm(acc))]
        for t in range(len(xs) - 1, -1, -1):
            acc = acc * zs[t]
            out.append(float(np.linalg.norm(acc)))
        return np.asarray(out) / out[0]


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s04", "Gating and the LSTM",
                "replacing a matrix product with an addition").open()

    xs = sequence(60, D_IN, seed=104)

    # -------------------------------------------------------- 4.1 the idea
    s.h("4.1  The idea, before the notation")
    s.say("  S3's problem was a product. Travelling d steps back multiplied the")
    s.say("  gradient by d matrices, and a product of d similar factors is a")
    s.say("  geometric sequence -- it goes to zero, or it does not, and mostly it")
    s.say("  goes to zero.")
    s.say()
    s.say("  Addition has a derivative of 1. So: build a quantity that each step")
    s.say("  *adds to* rather than *transforms*, and its backward path stops")
    s.say("  shrinking. That quantity is the cell state, and the machinery around")
    s.say("  it exists to decide how much to add, how much to keep, and how much")
    s.say("  to reveal.")
    s.eq("c_t = f_t (*) c_{t-1} + i_t (*) g_t",
         "keep some of what you had, add some of what is new; (*) is "
         "elementwise multiplication")
    s.eq("h_t = o_t (*) tanh(c_t)",
         "the hidden state is a gated view of the cell state, not the cell state "
         "itself")

    # ------------------------------------------------------ 4.2 the components
    s.h("4.2  What each gate is")
    net = LSTM()
    hs, cs, gs = net.forward(xs)
    s.table(["symbol", "name", "what it decides", "typical value here"],
            [["f_t", "forget gate", "how much of c_{t-1} survives",
              round(float(np.mean([g["f"].mean() for g in gs])), 3)],
             ["i_t", "input gate", "how much of the candidate is written",
              round(float(np.mean([g["i"].mean() for g in gs])), 3)],
             ["g_t", "candidate", "what would be written",
              round(float(np.mean([g["g"].mean() for g in gs])), 3)],
             ["o_t", "output gate", "how much of c_t is exposed as h_t",
              round(float(np.mean([g["o"].mean() for g in gs])), 3)]],
            align="llll")
    s.kv("hidden width d", net.d_hidden)
    s.kv("parameters", net.W.size + net.b.size)
    s.kv("forget-gate bias at initialisation", net.forget_bias)
    s.check("every gate is a vector of values in (0, 1), decided per unit and "
            "per step",
            all(np.all((g["f"] > 0) & (g["f"] < 1)) for g in gs)
            and all(np.all((g["i"] > 0) & (g["i"] < 1)) for g in gs),
            "the gates are outputs of a sigmoid reading h_{t-1} and x_t, so the "
            "network learns what to carry rather than being told")
    s.note("this is the part most summaries skip. The gates are not fixed decay "
           "constants; they are computed from the input at every step. An LSTM "
           "can hold one unit open for two hundred steps while cycling another "
           "every three, and which it does is learned")

    # ---------------------------------------------------- 4.3 the derivative
    s.h("4.3  The derivative along the cell state")
    s.eq("dc_t/dc_{t-1} = diag(f_t)",
         "holding h_{t-1} fixed: elementwise, and no weight matrix anywhere in it")
    h0, c0 = hs[10], cs[10]
    J_num = net.numeric_cell_jacobian(h0, c0, xs[10])
    f_t = net.gates(h0, xs[10])["f"]
    J_an = np.diag(f_t)
    s.kv("max |numeric - diag(f_t)|", f"{float(np.max(np.abs(J_num - J_an))):.3e}")
    s.kv("largest off-diagonal entry", f"{float(np.max(np.abs(J_num - np.diag(np.diag(J_num))))):.3e}")
    s.check("the cell-state Jacobian is exactly the diagonal matrix of forget "
            "gates",
            float(np.max(np.abs(J_num - J_an))) < 1e-8,
            f"finite differences agree with diag(f_t) to "
            f"{float(np.max(np.abs(J_num - J_an))):.1e}, and the off-diagonal "
            "entries are zero -- units do not mix along this path")
    s.note("compare with S3, where the same quantity was W^T diag(1 - s^2): a "
           "full matrix, applied once per step. Here it is a diagonal of numbers "
           "the network chose. Multiplying d diagonals together is still a "
           "product, but of quantities the network controls rather than of a "
           "fixed matrix's singular values")

    # ------------------------------------------------- 4.4 measure the horizon
    s.h("4.4  What that buys, measured against S3")
    vanilla = RNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=3, sigma_max=0.95,
                  seed=4)
    v_prof = vanilla.state_gradient_norms(xs, rng(204).normal(size=3))
    v_prof = np.asarray(v_prof) / v_prof[0]
    hw = net.highway_decay(xs)
    gru = GRU()
    gr = gru.carry_decay(xs)

    rows = []
    for d in (10, 25, 40, 55):
        rows.append([d, f"{v_prof[min(d, len(v_prof) - 1)]:.2e}",
                     f"{hw[min(d, len(hw) - 1)]:.2e}",
                     f"{gr[min(d, len(gr) - 1)]:.2e}"])
    s.table(["distance back", "vanilla RNN", "LSTM cell path", "GRU carry path"],
            rows, align="rrrr")
    s.kv("vanilla per-step ratio", round(fitted_ratio(list(v_prof)), 4))
    s.kv("LSTM highway per-step ratio", round(fitted_ratio(list(hw)), 4))
    s.kv("vanilla horizon (steps)", round(horizon(fitted_ratio(list(v_prof))), 1))
    s.kv("LSTM highway horizon (steps)",
         round(horizon(fitted_ratio(list(hw))), 1))
    s.check("the gated path decays enormously more slowly than the matrix path",
            hw[40] > 1e6 * v_prof[min(40, len(v_prof) - 1)],
            f"at distance 40 the vanilla gradient is "
            f"{v_prof[min(40, len(v_prof) - 1)]:.1e} of its starting size and "
            f"the LSTM cell path is {hw[40]:.1e}")
    s.check("the GRU's update gate does the same job as the LSTM's forget gate",
            gr[40] > 1e6 * v_prof[min(40, len(v_prof) - 1)],
            f"GRU carry path at distance 40 is {gr[40]:.1e}, the same mechanism "
            "with one fewer gate and no separate cell state")
    s.note("the forget-gate bias is doing real work here. Initialised at 1.0 the "
           "gate starts near 0.73 and the product decays slowly; initialised at "
           "0 it would start at 0.5 and the highway would halve the gradient "
           "every step, which is no better than the matrix it replaced. This is "
           "why that bias is a standard trick rather than a detail")

    # ------------------------------------------ 4.5 the honest limit
    s.h("4.5  Where 'the LSTM solves vanishing gradients' stops being true")
    s.say("  The cell state is not the only route back. h_{t-1} feeds all four")
    s.say("  gates, and h_{t-1} came from c_{t-1}, so there is a second path that")
    s.say("  does go through the weight matrix -- and behaves exactly like S3.")
    full = net.full_state_decay(xs)
    rows = []
    for d in (10, 25, 40, 55):
        rows.append([d, f"{v_prof[min(d, len(v_prof) - 1)]:.2e}",
                     f"{full[min(d, len(full) - 1)]:.2e}",
                     f"{hw[min(d, len(hw) - 1)]:.2e}"])
    s.table(["distance back", "vanilla RNN, full", "LSTM, full",
             "LSTM cell path (mechanism only)"], rows, align="rrrr")
    s.check("like for like, the LSTM's complete backward pass retains vastly "
            "more gradient than the vanilla network's",
            full[min(40, len(full) - 1)]
            > 1e4 * v_prof[min(40, len(v_prof) - 1)],
            f"at distance 40: LSTM {full[min(40, len(full) - 1)]:.2e} against "
            f"the RNN's {v_prof[min(40, len(v_prof) - 1)]:.2e}, both measured as "
            "complete backward passes")
    s.check("the two routes add rather than one attenuating the other",
            full[min(40, len(full) - 1)] > hw[min(40, len(hw) - 1)],
            f"the full pass ({full[min(40, len(full) - 1)]:.2e}) exceeds the "
            f"cell path alone ({hw[min(40, len(hw) - 1)]:.2e}); gradient arriving "
            "by the matrix route is added to what the highway carries, not "
            "subtracted from it. The two are not directly comparable in any case "
            "-- the highway figure is a pure Jacobian product, the full figure a "
            "gradient with a source term -- and the honest comparison is the one "
            "above, against the ungated network")
    s.note("so the accurate statement is not that gating removes the vanishing "
           "gradient. The h-route still contains a matrix product and still "
           "behaves like S3; what changed is that a second route now exists "
           "which does not, and gradient arriving by it is added to whatever the "
           "matrix route delivers. Gating supplies a path that does not vanish; "
           "it does not repair the ones that do. "
           "That distinction matters because it predicts the failure mode: an "
           "LSTM whose forget gates learn to close will vanish again, and the "
           "gradient reaching a closed gate cannot reopen it")

    # ------------------------------------------ 4.6 what gating does not fix
    s.h("4.6  Two things gating does not touch")
    s.say("  (a) c_t still cannot be computed before c_{t-1}. The sequential")
    s.say("      chain is exactly as long as it was.")
    s.say("  (b) The state is still a fixed-width vector. Everything the network")
    s.say("      knows about an arbitrarily long past is compressed into d")
    s.say("      numbers.")
    vocab = 50_257
    rows = [[d, f"{32 * d:,}", f"{32 * d / np.log2(vocab):,.0f}"]
            for d in (256, 512, 1024)]
    s.table(["state width d", "bits available (fp32)",
             "tokens distinguishable"], rows, align="rrr")
    s.check("a fixed-width state cannot losslessly retain an arbitrarily long "
            "past, however good its gates",
            True,
            "distinguishing all |V|^T prefixes needs T log2|V| bits; d floats "
            "hold at most 32d, so d=512 bounds lossless recall near 1,049 "
            "tokens regardless of architecture")
    s.note("(a) is answered in S7 and (b) in S5 and S6. Note they are different "
           "kinds of problem: (a) is about wall-clock time and (b) is about "
           "information. Gating solved a third kind -- a gradient problem -- and "
           "solving one says nothing about the other two")

    # ------------------------------------------------ 4.7 what it unlocked
    s.h("4.7  What this made possible")
    s.say("  The functional consequence is worth stating plainly, because it is")
    s.say("  the reason the LSTM dominated sequence modelling for roughly two")
    s.say("  decades.")
    s.say()
    s.say("    before  a sequence model could reliably use about ten steps of")
    s.say("            context, so anything needing longer-range structure -- a")
    s.say("            pronoun and its antecedent, an opening balance and a")
    s.say("            closing one -- was out of reach")
    s.say()
    s.say("    after   speech recognition, handwriting recognition and machine")
    s.say("            translation all became practical on real inputs; sequence")
    s.say("            models over long event histories became worth building")
    s.say()
    s.note("and the next problem is already visible in 4.6(b). Once you ask a "
           "sequence model to *produce* a sequence rather than a label -- to "
           "translate a sentence rather than classify it -- that fixed-width "
           "state has to carry the entire input across to the output. S5 is "
           "about what that costs")

    s.close()
    return s


if __name__ == "__main__":
    run()
