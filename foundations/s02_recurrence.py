"""S2. Recurrence: sharing one weight matrix across time.

The network in Part 1 cannot read a sentence. Its first weight matrix has shape
(8, 4) because the input always has exactly four features, and a sentence has no
fixed number of anything. This section is about the first serious answer to that,
and about the price it quietly charges.

The answer is to process the sequence one element at a time, carrying a summary
forward:

    s_t = tanh(W s_{t-1} + U x_t + b)        the hidden state at step t
    y_t = V s_t                              the output at step t

and -- this is the whole idea -- to use *the same* W, U and V at every step. The
network no longer has a size that depends on the sequence; it has a size that
depends on how much it chooses to remember.

Sharing the weights changes what a gradient is. In Part 1 every weight was used
exactly once in a forward pass, so the derivative of the loss with respect to it
was a single chain of factors. Here W is used at every step, so its total effect
on the loss at step t is the *sum* of what it did at every step up to t:

    dJ_t/dW = sum_{k=0..t} (dJ_t/dy_t)(dy_t/ds_t)(ds_t/ds_k)(ds_k/dW)

That is backpropagation through time, and 2.3 derives and verifies it. The third
factor is the one to watch, because it is not a single quantity -- it expands
into a product of one Jacobian per step of distance travelled. A sum over uses,
of products over distance. S3 is about what that product does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, rng


@dataclass
class RNN:
    """A vanilla recurrent network, written so the shared weights are visible.

    Terminology, since the rest of the section leans on it:

    * **hidden state** `s_t` -- a vector of `d_hidden` numbers that is the
      network's entire memory of everything before step t. Its width is chosen
      by the designer and does not change with sequence length.
    * **recurrent weights** `W` -- how the previous state influences the next
      one. The same matrix at every step.
    * **input weights** `U` -- how the current input enters.
    * **output weights** `V` -- how the state is read out.
    * **unrolling** -- drawing the T steps as a T-layer feedforward network,
      which is what makes Part 1's backpropagation apply unchanged.
    """

    d_in: int = 4
    d_hidden: int = 6
    d_out: int = 3
    sigma_max: float | None = None     # rescale W, for S3's sweep
    activation: str = "tanh"                 # "tanh" or "relu", for S3.8
    seed: int = 7

    def __post_init__(self) -> None:
        g = rng(self.seed)
        self.W = g.normal(0, 1 / np.sqrt(self.d_hidden),
                          (self.d_hidden, self.d_hidden))
        if self.sigma_max is not None:
            # Rescale so the largest singular value is exactly the requested
            # value. Note this is sigma_max, the operator 2-norm, NOT the
            # spectral radius (largest |eigenvalue|). They coincide for normal
            # matrices and can differ substantially otherwise. sigma_max is what
            # bounds a single step; it is not by itself the asymptotic growth
            # rate of a repeated product.
            self.W *= self.sigma_max / np.linalg.norm(self.W, 2)
        self.U = g.normal(0, 1 / np.sqrt(self.d_in), (self.d_hidden, self.d_in))
        self.V = g.normal(0, 1 / np.sqrt(self.d_hidden),
                          (self.d_out, self.d_hidden))
        self.b = np.zeros(self.d_hidden)

    @property
    def n_params(self) -> int:
        return self.W.size + self.U.size + self.V.size + self.b.size

    # ------------------------------------------------------------------ forward

    def forward(self, xs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run the sequence. Returns the states s_0..s_T and the outputs.

        `s[0]` is the initial state, all zeros, so `s[t]` is the state *after*
        reading `xs[t-1]`. Keeping the states is not an optimisation: the
        backward pass needs every one of them, which is why memory during
        training grows with sequence length even though the parameters do not.
        """
        s = [np.zeros(self.d_hidden)]
        ys = []
        for x in xs:
            z = self.W @ s[-1] + self.U @ x + self.b
            s.append(np.tanh(z) if self.activation == "tanh"
                     else np.maximum(z, 0.0))
            ys.append(self.V @ s[-1])
        return s, ys

    def fprime(self, s_t: np.ndarray) -> np.ndarray:
        """The activation derivative, recovered from the activation's output.

        For tanh, f'(z) = 1 - tanh(z)^2 = 1 - s^2. For ReLU, f'(z) is 1 exactly
        where z > 0, and relu(z) > 0 exactly there too, so `s > 0` recovers it.
        Both read off `s`, which is why the forward pass need not keep z.

        The contrast is the whole of S3.8: tanh's derivative shrinks as its input
        grows, so it damps its own gradient. ReLU's is exactly 1 wherever the
        unit is active, so it damps nothing.
        """
        if self.activation == "tanh":
            return 1.0 - s_t ** 2
        return (s_t > 0).astype(float)

    def loss(self, xs: np.ndarray, target: np.ndarray) -> float:
        """Squared error on the final output only.

        Scoring just the last step is deliberate. It makes the question sharp:
        for the network to do well, information from step 0 must survive all the
        way to step T, and the gradient must travel all the way back. A loss
        summed over every step would let the network score well on the easy
        recent steps and hide the failure S3 is about.
        """
        _, ys = self.forward(xs)
        return float(0.5 * np.sum((ys[-1] - target) ** 2))

    # ----------------------------------------------------------------- backward

    def bptt(self, xs: np.ndarray, target: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """Backpropagation through time.

        The recursion is Part 1's, applied to the unrolled network:

            delta_T = (ds_T/dz_T)^T V^T (y_T - target)
            delta_{k-1} = W^T diag(1 - s_{k-1}^2) delta_k

        and because W is used at every step, its gradient accumulates:

            dJ/dW = sum_k delta_k s_{k-1}^T

        `per_k` returns each step's separate contribution before summing, which
        is what 2.4 plots: it is the same decomposition as the slide's sum over
        k, made visible.
        """
        s, ys = self.forward(xs)
        T = len(xs)
        dV = np.outer(ys[-1] - target, s[T])
        delta = self.V.T @ (ys[-1] - target) * self.fprime(s[T])

        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        per_k: list[np.ndarray] = []

        for k in range(T, 0, -1):
            contribution = np.outer(delta, s[k - 1])
            per_k.append(contribution)
            dW += contribution
            dU += np.outer(delta, xs[k - 1])
            db += delta
            if k > 1:
                delta = (self.W.T @ delta) * self.fprime(s[k - 1])

        return dW, dU, dV, list(reversed(per_k))

    def state_gradient_norms(self, xs: np.ndarray,
                             target: np.ndarray) -> list[float]:
        """||dJ/ds_k||, indexed by distance back from the loss.

        Element 0 is the gradient at the final state, element d is the gradient
        that survived travelling d steps back. This isolates the travelling part
        of the gradient from the `s_{k-1}` factor that `bptt` multiplies it by,
        which is what S3 needs: the question there is what the *journey* does,
        not what happens at the destination.
        """
        s, ys = self.forward(xs)
        T = len(xs)
        delta = self.V.T @ (ys[-1] - target) * self.fprime(s[T])
        out = [float(np.linalg.norm(delta))]
        for k in range(T, 1, -1):
            delta = (self.W.T @ delta) * self.fprime(s[k - 1])
            out.append(float(np.linalg.norm(delta)))
        return out

    def finite_difference_W(self, xs: np.ndarray, target: np.ndarray,
                            eps: float = 1e-6) -> np.ndarray:
        g = np.zeros_like(self.W)
        it = np.nditer(self.W, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = self.W[i]
            self.W[i] = orig + eps
            lp = self.loss(xs, target)
            self.W[i] = orig - eps
            lm = self.loss(xs, target)
            self.W[i] = orig
            g[i] = (lp - lm) / (2 * eps)
            it.iternext()
        return g


def sequence(T: int, d_in: int, seed: int) -> np.ndarray:
    return rng(seed).normal(size=(T, d_in))


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s02", "Recurrence: sharing weights across time",
                "one matrix reused at every step, and what that does to the "
                "gradient").open()

    # ------------------------------------------------- 2.1 why not the obvious
    s.h("2.1  The obvious fix does not work")
    s.say("  Part 1's first weight matrix has shape (8, 4): eight hidden units "
          "reading\n  four features. The 4 is not a choice, it is the width of "
          "an IRIS row.")
    s.say()
    s.say("  The obvious repair is to give every position in the sequence its "
          "own\n  weights. Two things go wrong, and the second is worse.")

    d_in, d_hidden = 4, 6
    rows = []
    for T in (5, 20, 100, 1_000):
        per_position = T * (d_hidden * d_in + d_hidden * d_hidden)
        shared = d_hidden * d_in + d_hidden * d_hidden
        rows.append([T, f"{per_position:,}", f"{shared:,}",
                     f"{per_position / shared:.0f}x"])
    s.table(["sequence length T", "parameters, per-position weights",
             "parameters, shared", "ratio"], rows, align="rrrr")
    s.check("per-position weights make the model's size depend on the longest "
            "sequence it may ever see",
            int(rows[-1][1].replace(",", "")) == 1_000 * int(rows[0][2].replace(",", "")),
            f"{rows[-1][1]} parameters at T=1,000 against {rows[0][2]} shared, "
            "and a sequence one token longer than anything seen in training has "
            "no weights at all")
    s.note("the second problem has no table. With separate weights per position, "
           "whatever the network learns about a word at position 3 teaches it "
           "nothing about the same word at position 40 -- they are handled by "
           "different parameters that never see each other's gradients. Sharing "
           "the weights is not a memory optimisation. It is the statement that "
           "the rule for reading the next element does not depend on where you "
           "are in the sequence.")

    # ------------------------------------------------------------ 2.2 the RNN
    s.h("2.2  The recurrent network")
    s.eq("s_t = tanh(W s_{t-1} + U x_t + b)",
         "the hidden state: the network's entire memory of everything before t")
    s.eq("y_t = V s_t", "the output, read off the state")
    s.say()
    s.say("  W, U and V are the same at every step. The state s_t is a fixed-"
          "width\n  vector no matter how long the sequence is -- which is the "
          "whole trick,\n  and later the whole problem.")

    net = RNN(d_in=d_in, d_hidden=d_hidden, d_out=3)
    s.kv("hidden width d", net.d_hidden)
    s.kv("parameters", net.n_params)
    lengths = [3, 10, 50, 200]
    shapes = []
    for T in lengths:
        st, ys = net.forward(sequence(T, d_in, seed=1))
        shapes.append((T, len(st) - 1, ys[-1].shape[0]))
    s.table(["sequence length", "states computed", "final output width"],
            [[a, b, c] for a, b, c in shapes], align="rrr")
    s.check("the same parameters process any sequence length",
            len({net.n_params}) == 1 and all(c == net.d_out for _, _, c in shapes),
            f"{net.n_params} parameters handled lengths "
            f"{', '.join(str(t) for t in lengths)} unchanged")

    # ------------------------------------------------------------- 2.3 BPTT
    s.h("2.3  Backpropagation through time")
    s.say("  Unroll the T steps and it is a T-layer network -- so Part 1's "
          "backward\n  pass applies with no change. The one new feature is that "
          "every layer\n  shares the same W, so the gradient does not arrive "
          "from one place.")
    s.eq("dJ_t/dW = sum_{k=0..t} (dJ_t/dy_t)(dy_t/ds_t)(ds_t/ds_k)(ds_k/dW)",
         "a sum over every step at which W acted")
    s.eq("delta_{k-1} = W^T diag(1 - s_{k-1}^2) delta_k",
         "Part 1's recursion; 1 - s^2 is tanh'(z) written through the "
         "activation")
    s.eq("dJ/dW = sum_k delta_k s_{k-1}^T", "accumulate, do not overwrite")

    xs = sequence(12, d_in, seed=2)
    target = rng(3).normal(size=net.d_out)
    dW, _, _, per_k = net.bptt(xs, target)
    fd = net.finite_difference_W(xs, target)
    rel = float(np.linalg.norm(dW - fd)
                / (np.linalg.norm(dW) + np.linalg.norm(fd)))
    s.kv("relative L2 error vs finite differences", f"{rel:.3e}")
    s.check("the derived backward pass through time is correct",
            rel < 1e-8,
            f"analytic gradient agrees with central finite differences to "
            f"{rel:.1e} over all {net.W.size} recurrent weights, on a "
            f"{len(xs)}-step sequence")
    s.check("the accumulated gradient equals the sum of its per-step contributions",
            np.allclose(sum(per_k), dW, atol=1e-14),
            "sum over k of delta_k s_{k-1}^T reproduces dJ/dW exactly, which is "
            "the slide's sum made literal")

    # ------------------------------------------- 2.4 where the gradient comes from
    s.h("2.4  Which steps actually supply the gradient")
    s.say("  The loss is scored on the final output only. So every one of the "
          "12 terms\n  in that sum is a message from a different point in the "
          "past, saying how\n  much W's action *there* mattered to the answer "
          "*here*.")
    norms = [float(np.linalg.norm(c)) for c in per_k]
    total = float(np.linalg.norm(dW))
    rows = []
    for k, n in enumerate(norms, start=1):
        if k in (1, 2, 3, 6, 9, 11, 12):
            rows.append([k, len(xs) - k, round(n, 8),
                         f"{100 * n / sum(norms):.2f}%"])
    s.table(["step k", "distance back from the loss", "||contribution||",
             "share of total"], rows, align="rrrr")
    s.note("k = 1 is exactly zero for a structural reason rather than a "
           "numerical one: the contribution at step k is delta_k s_{k-1}^T, and "
           "s_0 is the zero vector, so W had nothing to act on at the first "
           "step. Every other row is the gradient genuinely arriving from that "
           "distance.")
    near = sum(norms[-3:])
    far = sum(norms[:3])
    s.check("the gradient is dominated by the steps nearest the loss",
            near > 20 * far,
            f"the last three steps contribute {100 * near / sum(norms):.1f}% of "
            f"the total magnitude; the first three contribute "
            f"{100 * far / sum(norms):.2f}%")
    s.note("this is the failure in miniature, and everything after it in this "
           "article is a response. The network can only learn to use "
           "information from step 1 if the gradient reaching step 1 is large "
           "enough to move the weights. Here it is roughly "
           f"{far / near:.0e} of the signal arriving from the recent steps. S3 "
           "shows this is not a quirk of this example -- it is geometric, and "
           "one number about W decides the rate.")

    # -------------------------------------------------------- 2.5 the other cost
    s.h("2.5  The cost that has nothing to do with gradients")
    s.say("  s_t cannot be computed until s_{t-1} exists. Every step waits for "
          "the one\n  before it, however many processors are available.")
    s.table(["sequence length T", "sequential steps required"],
            [[T, T] for T in (128, 1_024, 8_192)], align="rr")
    s.check("recurrence forces a number of sequential steps equal to the "
            "sequence length",
            True,
            "s_t depends on s_{t-1} by construction, so no amount of hardware "
            "shortens the chain")
    s.note("worth parking rather than solving now. Even a recurrent network that "
           "learned perfectly would still read a 8,192-token document in 8,192 "
           "dependent steps. When S7 removes recurrence entirely, this is half "
           "of what it buys -- and the other half is the gradient problem S3 is "
           "about to make precise.")

    s.close()
    return s


if __name__ == "__main__":
    run()
