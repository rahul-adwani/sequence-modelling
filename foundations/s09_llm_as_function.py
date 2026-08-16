"""S9. The baseline: a language model is a function, and that is the whole of it.

A language model is a map from a token sequence to a distribution over the next
token:

    f_theta : V^* -> Delta(V),    f_theta(x_1..x_t) = p(x_{t+1} | x_1..x_t)

S8 fixed the vocabulary `V`. This section is about the map itself, and about the
three properties of it that everything after here depends on. Each is
established by measurement rather than assertion:

1. The autoregressive factorisation is the chain rule, so it is exact. It is not
   a modelling approximation and nothing is lost by writing a joint distribution
   as a product of conditionals.
2. `f` is a pure function. Identical input gives bit-identical output, forever,
   with no dependence on what was asked before. The model is stateless; the
   context is the only channel; everything an agent "remembers" is something a
   program put back into the prompt.
3. "Context" means the conditioning argument, nothing more. Truncate it and the
   distribution changes by an amount you can measure in bits.

The output of `f` is a distribution, never a token. The rule that turns one into
the other is a separate object with its own failure modes, and it gets its own
section: S10.

To measure any of that honestly we need a model whose *true* distribution is
known, which no real LLM offers. So the object of study is a known stochastic
source, and the model is a trigram estimated from samples of it. The estimator
is elementary on purpose: everything demonstrated here is a property of the
functional form `V^* -> Delta(V)`, shared by a trigram and a frontier
transformer alike, and using a small one means the true distribution is
available to compare against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, entropy_bits, kl_bits, perplexity, rng

# A readable vocabulary, so the sampled sequences in the log look like language
# rather than integers. The words carry no meaning to the model: it sees indices.
VOCAB = [
    "a", "agent", "and", "call", "context", "each", "emits", "engine",
    "grows", "in", "loop", "model", "no", "one", "prompt", "reads",
    "state", "step", "stops", "the", "then", "token", "with", ".",
]
V = len(VOCAB)
ORDER = 2  # the true source is order-2, which the trigram can represent exactly


# --------------------------------------------------------------------- the source

@dataclass
class MarkovSource:
    """A known order-2 stochastic source over `V` tokens.

    Each of the V^2 contexts gets a distribution supported on `support` tokens,
    with weights drawn from a symmetric Dirichlet. Sparse support keeps the
    per-context entropy low enough that a few thousand tokens are informative,
    and Dirichlet weights keep the distributions from being uniform over that
    support, which would make the estimation problem degenerate.

    This class is the ground truth. Because `true_dist` is available for any
    context, quantities that are unmeasurable for a real LLM -- the divergence
    between the model and reality, the irreducible entropy of the process --
    become exactly computable here.
    """

    support: int = 5
    alpha: float = 0.6
    seed: int = 11

    def __post_init__(self) -> None:
        g = rng(self.seed)
        self.P = np.zeros((V, V, V), dtype=np.float64)
        for u in range(V):
            for v in range(V):
                idx = g.choice(V, size=self.support, replace=False)
                w = g.dirichlet(np.full(self.support, self.alpha))
                self.P[u, v, idx] = w
        # Row-normalise against Dirichlet round-off so every context is a
        # probability vector to machine precision, not to three decimals.
        self.P /= self.P.sum(axis=2, keepdims=True)

    def true_dist(self, u: int, v: int) -> np.ndarray:
        return self.P[u, v]

    def sample(self, n: int, seed: int, burn_in: int = 200) -> np.ndarray:
        """Draw a sequence. The burn-in discards the transient so that the
        context distribution in the returned sample is the chain's stationary
        one, rather than an artifact of where we happened to start."""
        g = rng(seed)
        u, v = 0, 1
        out = np.empty(n + burn_in, dtype=np.int64)
        for t in range(n + burn_in):
            w = g.choice(V, p=self.P[u, v])
            out[t] = w
            u, v = v, w
        return out[burn_in:]


# ---------------------------------------------------------------------- the model

class TrigramLM:
    """An interpolated trigram: the smallest honest instance of `V^* -> Delta(V)`.

        p_hat(w | u,v) = l3 * c(u,v,w)/c(u,v)
                       + l2 * c(v,w)/c(v)
                       + l1 * c(w)/N            with l1 + l2 + l3 = 1

    Interpolation is not decoration. The maximum-likelihood trigram assigns
    probability exactly zero to any triple absent from training, which sends
    held-out cross-entropy to infinity the first time an unseen triple appears.
    Mixing in the lower orders guarantees a strictly positive floor everywhere,
    and that is the same job that smoothing, backoff, and a softmax over a dense
    output layer all do: keep the support full.

    The mixing weights are fitted on a development split, never on the training
    counts -- fitted on training data the search would drive l3 to 1, because
    the unsmoothed model is by construction the best possible explanation of the
    data it memorised.
    """

    def __init__(self, tokens: np.ndarray, lambdas: tuple[float, float, float]
                 = (0.10, 0.30, 0.60)):
        self.l1, self.l2, self.l3 = lambdas
        assert abs(self.l1 + self.l2 + self.l3 - 1.0) < 1e-12

        self.c3 = np.zeros((V, V, V), dtype=np.float64)
        self.c2 = np.zeros((V, V), dtype=np.float64)
        self.c1 = np.zeros(V, dtype=np.float64)
        for t in range(len(tokens)):
            w = tokens[t]
            self.c1[w] += 1
            if t >= 1:
                self.c2[tokens[t - 1], w] += 1
            if t >= 2:
                self.c3[tokens[t - 2], tokens[t - 1], w] += 1

        self.n_tokens = len(tokens)
        self.uni = self.c1 / max(self.c1.sum(), 1.0)
        self._d2 = self.c2.sum(axis=1, keepdims=True)
        self._d3 = self.c3.sum(axis=2, keepdims=True)

    def with_lambdas(self, lambdas: tuple[float, float, float]) -> "TrigramLM":
        """Reuse the counts under new weights. Counting is the expensive part and
        it does not depend on the weights, so the grid search is nearly free."""
        clone = object.__new__(TrigramLM)
        clone.__dict__.update(self.__dict__)
        clone.l1, clone.l2, clone.l3 = lambdas
        return clone

    def dist(self, u: int, v: int) -> np.ndarray:
        """p_hat(. | u, v). A pure function of (u, v) and the frozen counts."""
        p3 = (self.c3[u, v] / self._d3[u, v, 0]) if self._d3[u, v, 0] > 0 \
            else np.zeros(V)
        p2 = (self.c2[v] / self._d2[v, 0]) if self._d2[v, 0] > 0 else np.zeros(V)
        return self.l3 * p3 + self.l2 * p2 + self.l1 * self.uni

    def dist_tensor(self) -> np.ndarray:
        """All V^2 conditionals at once, as P[u, v, w] = p_hat(w | u, v).

        The same arithmetic as `dist`, vectorised. S10.6 optimises over
        sequences from every start context, which is 576 evaluations per step;
        materialising the tensor once turns that search from minutes of Python
        into milliseconds of NumPy.
        """
        p3 = np.divide(self.c3, self._d3, out=np.zeros_like(self.c3),
                       where=self._d3 > 0)
        p2 = np.divide(self.c2, self._d2, out=np.zeros_like(self.c2),
                       where=self._d2 > 0)
        P = self.l3 * p3 + self.l2 * p2[None, :, :] + self.l1 * self.uni[None, None, :]
        return P / P.sum(axis=2, keepdims=True)

    def logits(self, u: int, v: int) -> np.ndarray:
        """Scores whose softmax is `dist`, for the temperature demonstrations.

        A real model produces logits and derives probabilities; this one is the
        other way round, so the logits are recovered as log p. The two are the
        same object up to an additive constant, which the softmax removes.
        """
        return np.log(np.maximum(self.dist(u, v), np.finfo(np.float64).tiny))

    # -------------------------------------------------------------- likelihood

    def sequence_logprob(self, tokens: np.ndarray, u0: int = 0,
                         v0: int = 1) -> float:
        """log2 p(x_1..x_T) by the chain rule, summed in log space.

            log p(x_1..x_T) = sum_t log p(x_t | x_{<t})

        Summed in logs rather than multiplied: a hundred tokens at p ~ 0.3 each
        underflows float64 well before the sequence ends.
        """
        u, v, total = u0, v0, 0.0
        for w in tokens:
            total += np.log2(max(self.dist(u, v)[w], np.finfo(np.float64).tiny))
            u, v = v, int(w)
        return float(total)

    def cross_entropy_bits(self, tokens: np.ndarray) -> float:
        """Held-out cross-entropy in bits per token: -1/T sum log2 p_hat(x_t|...)."""
        return -self.sequence_logprob(tokens) / len(tokens)


def words(seq) -> str:
    return " ".join(VOCAB[int(i)] for i in seq)


# ------------------------------------------------------------- the shared fit

_FITTED: dict | None = None


def fitted() -> dict:
    """The source, the splits, and the trigram fitted on the development split.

    S9 and S10 study the same model, so it is built once and shared. That is
    safe here for a specific reason rather than as a convenience: every input is
    seeded through `rng`, nothing reads the clock or the global NumPy state, and
    the fit is a pure function of the splits. Running the two sections in either
    order, or one without the other, gives identical numbers.
    """
    global _FITTED
    if _FITTED is not None:
        return _FITTED

    src = MarkovSource()
    train = src.sample(48_000, seed=101)
    dev = src.sample(6_000, seed=202)
    test = src.sample(6_000, seed=303)

    base = TrigramLM(train)
    grid = [(l1, l2, round(1 - l1 - l2, 4))
            for l1 in (0.02, 0.05, 0.10, 0.20)
            for l2 in (0.10, 0.20, 0.30, 0.40)
            if 1 - l1 - l2 > 0.2]
    scored = sorted(((base.with_lambdas(g).cross_entropy_bits(dev), g)
                     for g in grid), key=lambda t: t[0])

    _FITTED = {"src": src, "train": train, "dev": dev, "test": test,
               "base": base, "scored": scored, "lambdas": scored[0][1],
               "model": base.with_lambdas(scored[0][1])}
    return _FITTED


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s09", "The baseline: an LLM as a function",
                "f: V* -> Delta(V), and the three consequences that follow").open()

    f = fitted()
    src, train, dev, test = f["src"], f["train"], f["dev"], f["test"]

    s.h("2.1  The object under study")
    s.kv("vocabulary size |V|", V)
    s.kv("true source order", ORDER)
    s.kv("support per context", src.support, "of 24 tokens")
    s.kv("train / dev / test tokens", f"{len(train)} / {len(dev)} / {len(test)}")
    s.say()
    s.say(f"  sample:  {words(train[:22])}")
    s.note("a known order-2 source, so the true distribution is available for "
           "comparison; a real LLM never offers this")

    # ---------------------------------------------------------------- 2.2 fit
    s.h("2.2  Fitting the interpolated trigram")
    s.eq("p_hat(w|u,v) = l3*c(u,v,w)/c(u,v) + l2*c(v,w)/c(v) + l1*c(w)/N",
         "interpolation keeps the support full; an unsmoothed trigram assigns "
         "zero to unseen triples and held-out cross-entropy diverges")

    scored, best_lam, model = f["scored"], f["lambdas"], f["model"]

    s.table(["l1 (uni)", "l2 (bi)", "l3 (tri)", "dev bits/token", "dev PP"],
            [[g[0], g[1], g[2], round(ce, 4), round(perplexity(ce), 3)]
             for ce, g in scored[:4]] +
            [["...", "...", "...", "...", "..."],
             [scored[-1][1][0], scored[-1][1][1], scored[-1][1][2],
              round(scored[-1][0], 4), round(perplexity(scored[-1][0]), 3)]],
            align="rrrrr")
    s.kv("selected lambdas", best_lam)
    s.note("selected on dev, never on train: fitted on the training counts the "
           "search drives l3 -> 1, because the unsmoothed model is by "
           "construction the best explanation of what it memorised")

    train_ce = model.cross_entropy_bits(train[:6_000])
    test_ce = model.cross_entropy_bits(test)
    s.kv("train cross-entropy", round(train_ce, 4), "bits/token")
    s.kv("test  cross-entropy", round(test_ce, 4), "bits/token")
    s.kv("test perplexity", round(perplexity(test_ce), 3), "effective branches")

    # -------------------------------------------------- 2.3 the decomposition
    s.h("2.3  Where the loss actually goes")
    s.eq("H(p*, q) = H(p*) + D_KL(p* || q)",
         "cross-entropy splits into the entropy of the process, which no model "
         "can remove, and the divergence of the model from it, which is the "
         "only part training can touch")

    # Taken as expectations under the true conditionals at the contexts a
    # held-out sample actually visits. Under those expectations the identity is
    # algebraic and holds to machine precision; estimated instead from sampled
    # tokens it holds only to Monte Carlo error, which is the version below.
    h_true, kl_mean, ce_exact = 0.0, 0.0, 0.0
    for t in range(2, len(test)):
        u, v = int(test[t - 2]), int(test[t - 1])
        p, q = src.true_dist(u, v), model.dist(u, v)
        h_true += entropy_bits(p)
        kl_mean += kl_bits(p, q)
        ce_exact += float(-np.sum(p[p > 0] * np.log2(
            np.maximum(q[p > 0], np.finfo(np.float64).tiny))))
    n = len(test) - 2
    h_true, kl_mean, ce_exact = h_true / n, kl_mean / n, ce_exact / n

    s.kv("H(p*)   irreducible entropy", round(h_true, 4), "bits/token")
    s.kv("D_KL(p* || p_hat)  model gap", round(kl_mean, 4), "bits/token")
    s.kv("H(p*) + D_KL", round(h_true + kl_mean, 4), "bits/token")
    s.kv("H(p*, p_hat)  computed directly", round(ce_exact, 4), "bits/token")
    s.check("cross-entropy equals entropy plus divergence, to machine precision",
            abs((h_true + kl_mean) - ce_exact) < 1e-10,
            f"residual {abs((h_true + kl_mean) - ce_exact):.3e} bits/token")
    s.check("the irreducible term dominates the model's total loss",
            h_true > kl_mean,
            f"H(p*)={h_true:.3f} vs D_KL={kl_mean:.3f} bits/token")
    s.note("a perplexity number alone cannot tell you whether a model is weak "
           "or the process is noisy; only the split can, and the split needs a "
           "ground truth that production systems do not have")

    # ------------------------------------------------- 2.4 more data, less gap
    s.h("2.4  The divergence shrinks with data, the entropy does not")
    rows = []
    for n_tok in (750, 3_000, 12_000, 48_000):
        m = TrigramLM(src.sample(n_tok, seed=404)).with_lambdas(best_lam)
        kl = float(np.mean([kl_bits(src.true_dist(int(test[t - 2]), int(test[t - 1])),
                                   m.dist(int(test[t - 2]), int(test[t - 1])))
                            for t in range(2, 2_002)]))
        ce = m.cross_entropy_bits(test)
        rows.append([n_tok, round(kl, 4), round(ce, 4), round(perplexity(ce), 3)])
    s.table(["train tokens", "D_KL(p*||p_hat)", "test bits/token", "test PP"],
            rows, align="rrrr")
    kls = [r[1] for r in rows]
    s.check("divergence from the true process decreases monotonically with data",
            all(kls[i] > kls[i + 1] for i in range(len(kls) - 1)),
            " > ".join(f"{k:.4f}" for k in kls))

    # ------------------------------------------------------ 2.5 one prediction
    s.h("2.5  One forward pass, in full")
    u, v = VOCAB.index("the"), VOCAB.index("model")
    p_hat, p_star = model.dist(u, v), src.true_dist(u, v)
    order = np.argsort(-p_hat)[:6]
    s.say(f"  context = ({VOCAB[u]!r}, {VOCAB[v]!r})")
    s.table(["token", "p_hat", "p_true", "logit = log p_hat", "-log2 p (bits)"],
            [[VOCAB[i], round(float(p_hat[i]), 5), round(float(p_star[i]), 5),
              round(float(model.logits(u, v)[i]), 4),
              round(float(-np.log2(p_hat[i])), 3)] for i in order],
            align="lrrrr")
    s.kv("sum of all 24 probabilities", round(float(p_hat.sum()), 12))
    s.kv("H(p_hat) at this context", round(entropy_bits(p_hat), 4), "bits")
    s.check("the output is a normalised distribution over the whole vocabulary",
            abs(p_hat.sum() - 1.0) < 1e-12 and np.all(p_hat >= 0),
            f"|sum - 1| = {abs(p_hat.sum() - 1.0):.2e}, min p = {p_hat.min():.3e}")
    s.note("this vector is the model's entire output; a token is what a separate "
           "decoding rule does to it")

    # ----------------------------------------------------------- 2.6 purity
    s.h("2.6  The function is pure, therefore the model is stateless")
    a = model.dist(*(VOCAB.index("the"), VOCAB.index("model")))
    b = model.dist(*(VOCAB.index("a"), VOCAB.index("token")))
    a_again = model.dist(*(VOCAB.index("the"), VOCAB.index("model")))
    for _ in range(50):  # exercise it hard in between
        model.dist(int(rng(7).integers(V)), int(rng(8).integers(V)))
    a_after_load = model.dist(*(VOCAB.index("the"), VOCAB.index("model")))

    s.check("identical context gives bit-identical output",
            np.array_equal(a, a_again), "np.array_equal on float64, not allclose")
    s.check("intervening calls on other contexts leave the output unchanged",
            np.array_equal(a, a_after_load),
            "52 calls between the two evaluations, max abs diff "
            f"{np.max(np.abs(a - a_after_load)):.1e}")
    s.check("different contexts give different outputs",
            not np.array_equal(a, b),
            f"D_KL = {kl_bits(a, b):.3f} bits")
    s.say()
    s.note("this is the single most consequential fact in the repo. The model "
           "has no memory, no session and no accumulating state. A conversation "
           "is a string that a program rebuilds and resubmits on every turn. "
           "Memory, tools, planning and orchestration are therefore not model "
           "features; they are things written around a stateless function, and "
           "S13 onwards is about what it takes to write them.")

    # ---------------------------------------------------------- 2.7 context
    s.h("2.7  What 'context' means mechanically")
    s.say("  Truncate the conditioning set to the last n tokens and measure how "
          "far\n  the prediction moves from the full-context prediction.")
    rows = []
    for keep in (0, 1, 2):
        divs = []
        for t in range(2, 1_202):
            uu, vv = int(test[t - 2]), int(test[t - 1])
            full = model.dist(uu, vv)
            if keep == 0:
                trunc = model.uni
            elif keep == 1:
                trunc = model.c2[vv] / max(model._d2[vv, 0], 1.0)
            else:
                trunc = full
            divs.append(kl_bits(full, trunc))
        rows.append([keep, round(float(np.mean(divs)), 4),
                     round(float(np.max(divs)), 4)])
    s.table(["tokens kept", "mean D_KL vs full (bits)", "max D_KL"], rows,
            align="rrr")
    s.check("truncating below the true order of the process destroys information",
            rows[0][1] > rows[1][1] > rows[2][1] and rows[2][1] < 1e-12,
            f"keep 0: {rows[0][1]:.3f}, keep 1: {rows[1][1]:.3f}, "
            f"keep 2: {rows[2][1]:.1e} bits")
    s.note("the context window is a hard horizon, not a soft preference. Beyond "
           "the dependency length of the process it costs nothing to drop "
           "tokens; below it, information is gone and no decoding rule recovers "
           "it. Real text has dependencies thousands of tokens long, which is "
           "the entire argument for long windows -- and the reason S11's KV cache "
           "matters, since that horizon has to be paid for in memory.")

    s.say()
    s.note("where we are: the function is fixed, exact, stateless, and outputs a "
           "distribution rather than a token. Nothing above chose a token. Every "
           "system built on a model has to, and S10 is about the fact that the "
           "rule for doing it is a search with no tractable optimum")

    s.close()
    return s


if __name__ == "__main__":
    run()
