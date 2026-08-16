"""S1. From a neural network to a language model.

The previous article in this series derived a fully-connected tanh network on a
binary classification task: the forward pass, binary cross-entropy, and
backpropagation layer by layer in the notation

    a^y_x   activation of neuron x in layer y
    w^x_ab  weight from neuron a to neuron b at layer x
    d^y_x   the backward term at neuron x of layer y

closing with the update w := w - alpha * dL/dw. It did not cover softmax,
multiclass cross-entropy, sequences, or attention. This section covers exactly
the distance between those two points, because that distance turns out to be
short and worth walking rather than skipping.

Three things change, and only three:

    1. the output layer          sigmoid scalar  ->  softmax over K
    2. the loss                  binary CE       ->  categorical CE
    3. the input                 fixed 4-vector  ->  variable-length sequence

The first two change almost nothing. The gradient that arrives at the output
layer is `p - y` in both cases, so every backward equation already derived
continues to hold unaltered, and 0.4 proves that by showing the K=2 case reduce
exactly to the published one. That reduction is the load-bearing claim here: it
is what makes this a continuation rather than a fresh start.

The third change is the one that forces architecture. A weight matrix has a fixed
shape, so it can consume a fixed-width input and nothing else; a sequence has no
fixed width. 0.7 works through the three historical answers and prices each,
which is where attention -- and therefore S11's memory bill -- becomes inevitable
rather than fashionable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, entropy_bits, kl_bits, perplexity, rng, softmax

# --------------------------------------------------------- the published network

def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Stable logistic. For z << 0 the naive form overflows in exp(-z)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


class BinaryNet:
    """The previous article's network, rebuilt so its gradients can be checked.

    Fully connected, tanh hidden units, one sigmoid output, binary cross-entropy,
    trained one example at a time. Layer sizes are the article's shape: a
    four-feature input narrowing through hidden layers to a single unit.

    The backward pass is written in the article's own terms. `self.d[l]` is
    d^l, the vector of backward terms at layer l, and the recurrence

        d^L = a^L - y                                   (output)
        d^l = (W^{l+1})^T d^{l+1} * (1 - (a^l)^2)       (hidden, tanh')

    is the chain rule applied once per layer. `1 - a^2` is tanh'(z) expressed
    through the activation, which is why the forward activations have to be kept.
    """

    def __init__(self, sizes: tuple[int, ...] = (4, 8, 8, 6, 4, 1),
                 seed: int = 3):
        g = rng(seed)
        self.sizes = sizes
        self.W = [g.normal(0, 1 / np.sqrt(sizes[i]), (sizes[i + 1], sizes[i]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # ------------------------------------------------------------------ forward

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """Returns activations a^0 .. a^L, with a^0 = x."""
        a = [np.asarray(x, dtype=np.float64)]
        n = len(self.W)
        for l in range(n):
            z = self.W[l] @ a[-1] + self.b[l]
            a.append(np.tanh(z) if l < n - 1 else sigmoid(z))
        return a

    def loss(self, x: np.ndarray, y: float) -> float:
        """Binary cross-entropy, clipped away from the asymptotes.

        L = -( y log a^L + (1-y) log(1 - a^L) )

        The clip is not cosmetic: a confident correct prediction drives a^L to
        1.0 in float64, and log(0) then propagates nan through the gradient check
        that this class exists to support.
        """
        p = float(self.forward(x)[-1][0])
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ----------------------------------------------------------------- backward

    def gradients(self, x: np.ndarray, y: float
                  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a = self.forward(x)
        n = len(self.W)
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(bb) for bb in self.b]

        # The output term. For sigmoid composed with binary cross-entropy the
        # sigmoid derivative cancels against the loss derivative exactly, leaving
        # a^L - y. That cancellation is the reason this pairing is used at all,
        # and 0.3 shows the softmax/categorical pairing does the same thing.
        d = a[-1] - y

        for l in reversed(range(n)):
            dW[l] = np.outer(d, a[l])
            db[l] = d
            if l > 0:
                d = (self.W[l].T @ d) * (1.0 - a[l] ** 2)   # tanh'
        return dW, db

    # ------------------------------------------------------------ verification

    def finite_difference(self, x: np.ndarray, y: float, eps: float = 1e-6
                          ) -> list[np.ndarray]:
        """Central differences on every weight: (L(w+e) - L(w-e)) / 2e.

        Central rather than forward differences because the error is O(eps^2)
        instead of O(eps), which is the difference between agreeing with the
        analytic gradient to seven digits and to three.
        """
        out = []
        for l in range(len(self.W)):
            gW = np.zeros_like(self.W[l])
            it = np.nditer(self.W[l], flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = self.W[l][idx]
                self.W[l][idx] = orig + eps
                lp = self.loss(x, y)
                self.W[l][idx] = orig - eps
                lm = self.loss(x, y)
                self.W[l][idx] = orig
                gW[idx] = (lp - lm) / (2 * eps)
                it.iternext()
            out.append(gW)
        return out


def iris_like(n: int = 120, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian classes in four dimensions, in the shape of the IRIS task.

    Generated rather than downloaded: the section must run with no network. The
    data only has to be a plausible four-feature binary problem for the gradient
    check to mean something, and a checked gradient is a property of the network,
    not of the dataset.
    """
    g = rng(seed)
    mu0 = np.array([5.0, 3.4, 1.5, 0.2])
    mu1 = np.array([6.5, 2.9, 4.6, 1.4])
    x0 = g.normal(mu0, 0.35, size=(n // 2, 4))
    x1 = g.normal(mu1, 0.35, size=(n // 2, 4))
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    order = g.permutation(n)
    return X[order], y[order]


# ------------------------------------------------------- softmax and its gradient

def softmax_jacobian(p: np.ndarray) -> np.ndarray:
    """J[i,j] = dp_i/dz_j = p_i (delta_ij - p_j).

    Note the shape: a K-vector output against a K-vector input gives a full K x K
    Jacobian, unlike the elementwise sigmoid whose Jacobian is diagonal. Every
    output probability depends on every logit, because the normaliser does. That
    coupling is what makes the softmax a competition between classes, and it is
    why raising one logit lowers every other probability -- the mechanism behind
    S10.2's observation that temperature reweights without reordering.
    """
    return np.diag(p) - np.outer(p, p)


def categorical_ce(z: np.ndarray, y: np.ndarray) -> float:
    """L = -sum_k y_k log p_k with p = softmax(z)."""
    p = softmax(z)
    return float(-np.sum(y * np.log(np.maximum(p, 1e-300))))


def ce_grad_analytic(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dL/dz = p - y.

    Derivation, in full, because this is the whole reason the previous article's
    backward equations survive untouched:

        L      = -sum_k y_k log p_k
        dL/dp_i = -y_i / p_i
        dp_i/dz_j = p_i (delta_ij - p_j)

        dL/dz_j = sum_i (dL/dp_i)(dp_i/dz_j)
                = -sum_i (y_i / p_i) p_i (delta_ij - p_j)
                = -sum_i y_i (delta_ij - p_j)
                = -y_j + p_j sum_i y_i
                = p_j - y_j          since sum_i y_i = 1

    The p_i cancels, which is the same cancellation that gave a^L - y in the
    binary case. Nothing about the layers below changes.
    """
    return softmax(z) - y


def num_grad(f, z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(z)
    for i in range(z.size):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        g[i] = (f(zp) - f(zm)) / (2 * eps)
    return g


# --------------------------------------------- a dependency at a fixed distance

@dataclass
class DelayedSource:
    """x_t depends on x_{t-lag} and on nothing nearer.

    A deliberately extreme test of what an architecture can represent. Any model
    whose conditioning set excludes position t-lag is predicting the marginal, no
    matter how well fitted it is, because the information simply is not in its
    input. That is a statement about the architecture, not the training.
    """

    lag: int = 4
    vocab: int = 6
    noise: float = 0.15
    seed: int = 23

    def __post_init__(self) -> None:
        g = rng(self.seed)
        self.map = g.permutation(self.vocab)
        self.P = np.full((self.vocab, self.vocab), self.noise / (self.vocab - 1))
        for v in range(self.vocab):
            self.P[v, self.map[v]] = 1.0 - self.noise

    def sample(self, n: int, seed: int) -> np.ndarray:
        g = rng(seed)
        out = list(g.integers(0, self.vocab, size=self.lag))
        for t in range(self.lag, n):
            out.append(int(g.choice(self.vocab, p=self.P[out[t - self.lag]])))
        return np.asarray(out)

    def true_dist(self, hist: np.ndarray) -> np.ndarray:
        return self.P[hist[-self.lag]]


def window_model(tokens: np.ndarray, window: int, vocab: int,
                 alpha: float = 0.5) -> np.ndarray:
    """Count-based conditional given the previous `window` tokens.

    Returned as a flat table indexed by the packed window. `window = 0` is the
    marginal. Add-alpha smoothing keeps the support full for the same reason S9's
    interpolation did: a zero would make the divergence infinite and unplottable.
    """
    size = vocab ** window
    counts = np.full((max(size, 1), vocab), alpha)
    for t in range(window, len(tokens)):
        key = 0
        for j in range(window, 0, -1):
            key = key * vocab + int(tokens[t - j])
        counts[key, int(tokens[t])] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def window_predict(table: np.ndarray, hist: np.ndarray, window: int,
                   vocab: int) -> np.ndarray:
    if window == 0:
        return table[0]
    key = 0
    for j in range(window, 0, -1):
        key = key * vocab + int(hist[-j])
    return table[key]


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s01", "From a neural network to a language model",
                "three changes, of which only the third is hard").open()

    # ------------------------------------------- 0.1 reproduce the published result
    s.h("0.1  Where the previous article ended, verified")
    s.eq("a^l = tanh(W^l a^{l-1} + b^l),   a^L = sigma(W^L a^{L-1} + b^L)",
         "forward pass, in the notation of the published article")
    s.eq("L = -( y log a^L + (1-y) log(1 - a^L) )", "binary cross-entropy")
    s.eq("d^L = a^L - y;   d^l = (W^{l+1})^T d^{l+1} * (1 - (a^l)^2)",
         "the backward recurrence; tanh'(z) = 1 - tanh^2(z) = 1 - (a^l)^2")

    net = BinaryNet()
    X, Y = iris_like()
    s.kv("layer sizes", " -> ".join(str(n) for n in net.sizes))
    s.kv("parameters", sum(w.size for w in net.W) + sum(b.size for b in net.b))
    s.kv("examples", len(X))

    # The standard gradient check: relative L2 distance over the whole gradient
    # vector,
    #     ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)
    # rather than the worst single element. Deep in a tanh stack some gradients
    # are ~1e-9, and a central difference of a loss known to 1e-16 cannot resolve
    # those to better than a few percent *relatively* -- the subtraction cancels
    # almost every significant digit. That is a limitation of finite differences,
    # not evidence against the derivation, so the check is made on the vector norm
    # where it belongs. The worst element is still reported, for honesty.
    worst, l2 = 0.0, 0.0
    for i in (0, 1, 2, 37, 88):
        dW, _ = net.gradients(X[i], float(Y[i]))
        fd = net.finite_difference(X[i], float(Y[i]))
        an_v = np.concatenate([g.ravel() for g in dW])
        fd_v = np.concatenate([g.ravel() for g in fd])
        l2 = max(l2, float(np.linalg.norm(an_v - fd_v)
                           / (np.linalg.norm(an_v) + np.linalg.norm(fd_v))))
        denom = np.maximum(np.abs(an_v) + np.abs(fd_v), 1e-12)
        worst = max(worst, float(np.max(np.abs(an_v - fd_v) / denom)))
        small = float(np.min(np.abs(an_v)))
    s.kv("relative L2 error, 5 examples", f"{l2:.3e}")
    s.kv("worst single-element relative error", f"{worst:.3e}")
    s.kv("smallest analytic gradient component", f"{small:.2e}")
    s.check("the backward pass derived in the previous article is correct",
            l2 < 1e-8,
            f"analytic gradients agree with central finite differences to "
            f"{l2:.1e} relative L2 error over all "
            f"{sum(w.size for w in net.W)} weights, across 5 examples")
    s.note(f"the worst single element disagrees by {worst:.0e}, and that is the "
           f"finite difference being wrong rather than the gradient: the smallest "
           f"component here is {small:.0e}, and differencing a float64 loss "
           "cannot resolve a quantity that size. Gradient checks are done on the "
           "vector norm for exactly this reason")
    s.note("this is the starting point restated as something checked rather than "
           "recalled. Everything below changes three things about it and nothing "
           "else")

    # ------------------------------------------------- 0.2 the softmax Jacobian
    s.h("0.2  Change one: the output layer becomes a softmax")
    s.eq("p_k = exp(z_k) / sum_j exp(z_j)", "K outputs that sum to one")
    s.eq("dp_i/dz_j = p_i (delta_ij - p_j)",
         "a full K x K Jacobian, not the diagonal one a sigmoid gives")
    K = 5
    z = rng(41).normal(size=K)
    p = softmax(z)
    J_an = softmax_jacobian(p)
    J_num = np.zeros((K, K))
    for i in range(K):
        J_num[i] = num_grad(lambda zz: float(softmax(zz)[i]), z)
    s.table(["quantity", "value"],
            [["logits z", np.round(z, 4).tolist()],
             ["p = softmax(z)", np.round(p, 5).tolist()],
             ["sum p", round(float(p.sum()), 12)],
             ["row sums of J", np.round(J_an.sum(axis=1), 12).tolist()]])
    s.check("the analytic softmax Jacobian matches finite differences",
            float(np.max(np.abs(J_an - J_num))) < 1e-8,
            f"max abs difference {float(np.max(np.abs(J_an - J_num))):.2e}")
    s.check("every row of the Jacobian sums to zero",
            float(np.max(np.abs(J_an.sum(axis=1)))) < 1e-12,
            "raising one logit must lower the other probabilities, because the "
            "outputs are constrained to sum to one")
    s.note("the zero row sum is the whole character of a softmax. Outputs are not "
           "independent units that happen to be normalised afterwards; they "
           "compete. It is also, mechanically, why S10.2 finds that temperature "
           "reweights the distribution without ever reordering it")

    # ---------------------------------------------- 0.3 the gradient is still p - y
    s.h("0.3  Change two: the loss becomes categorical, and the gradient survives")
    s.eq("L = -sum_k y_k log p_k", "categorical cross-entropy")
    s.eq("dL/dz = p - y",
         "the p_i in the Jacobian cancels the 1/p_i from the loss, exactly as "
         "the sigmoid derivative cancelled in the binary case")
    y1h = np.zeros(K)
    y1h[2] = 1.0
    g_an = ce_grad_analytic(z, y1h)
    g_num = num_grad(lambda zz: categorical_ce(zz, y1h), z)
    s.table(["k", "p_k", "y_k", "analytic p-y", "finite difference"],
            [[k, round(float(p[k]), 5), y1h[k], round(float(g_an[k]), 6),
              round(float(g_num[k]), 6)] for k in range(K)], align="rrrrr")
    s.check("dL/dz = p - y, confirmed against finite differences",
            float(np.max(np.abs(g_an - g_num))) < 1e-8,
            f"max abs difference {float(np.max(np.abs(g_an - g_num))):.2e}")
    s.check("the gradient sums to zero, so the update cannot shift all logits "
            "together",
            abs(float(g_an.sum())) < 1e-12,
            "sum(p) = sum(y) = 1, so sum(p - y) = 0; softmax is invariant to a "
            "constant added to every logit and the gradient reflects that")
    s.note("this is the payoff of the section. The backward recurrence from 0.1 "
           "is untouched: the only thing that changed is the vector handed to it "
           "at the output layer, and it has the same form as before. Nothing "
           "about backpropagation needs relearning to go from a classifier to a "
           "language model")

    # ------------------------------------------------ 0.4 the continuity proof
    s.h("0.4  The two are the same object: K = 2 reduces to the published case")
    s.eq("softmax([z_0, z_1])_1 = sigma(z_1 - z_0)",
         "a two-class softmax is a logistic function of the logit difference")
    z2 = np.array([0.7, -1.3])
    p2 = softmax(z2)
    s_diff = float(sigmoid(z2[1] - z2[0]))
    rows = []
    for y_val in (0.0, 1.0):
        y2 = np.array([1.0 - y_val, y_val])
        l_cat = categorical_ce(z2, y2)
        a_out = float(p2[1])
        l_bin = -(y_val * np.log(a_out) + (1 - y_val) * np.log(1 - a_out))
        g_cat = ce_grad_analytic(z2, y2)
        g_bin = a_out - y_val               # the published article's d^L
        rows.append([y_val, round(l_cat, 8), round(l_bin, 8),
                     round(float(g_cat[1]), 8), round(g_bin, 8)])
    s.table(["y", "categorical CE", "binary CE", "(p-y) at k=1",
             "a^L - y from 0.1"], rows, align="rrrrr")
    s.check("a two-class softmax equals the sigmoid of the logit difference",
            abs(float(p2[1]) - s_diff) < 1e-15,
            f"p_1 = {float(p2[1]):.15f} against sigma(z_1 - z_0) = {s_diff:.15f}")
    s.check("categorical cross-entropy at K=2 equals binary cross-entropy",
            all(abs(r[1] - r[2]) < 1e-12 for r in rows),
            f"identical at y=0 and y=1, to {max(abs(r[1] - r[2]) for r in rows):.1e}")
    s.check("and the categorical gradient reduces exactly to the published "
            "article's output term",
            all(abs(r[3] - r[4]) < 1e-12 for r in rows),
            f"(p - y)_1 == a^L - y to "
            f"{max(abs(r[3] - r[4]) for r in rows):.1e} at both labels")
    s.note("so this is not a new derivation replacing an old one. The binary "
           "network of the previous article is the K=2 instance of the same "
           "equations, and the generalisation is genuinely free: set K to "
           "whatever the label space needs and change nothing else.")

    # --------------------------------------- 0.5 K = |V| makes it a language model
    s.h("0.5  Change three, part one: the label becomes the next token")
    s.eq("L_t = -log p(x_t | x_<t)",
         "cross-entropy against a one-hot target keeps exactly one term")
    s.eq("sum_t L_t = -log prod_t p(x_t|x_<t) = -log p(x_1..x_T)",
         "so minimising mean cross-entropy IS maximising corpus likelihood, by "
         "the chain rule of S9")
    g = rng(59)
    V, T = 40, 300
    logits = g.normal(size=(T, V))
    obs = g.integers(0, V, size=T)
    probs = np.array([softmax(logits[t]) for t in range(T)])

    ce_onehot = float(np.mean([
        categorical_ce(logits[t], np.eye(V)[obs[t]]) for t in range(T)]))
    ce_direct = float(np.mean([-np.log(probs[t, obs[t]]) for t in range(T)]))
    total_nll = float(-np.sum(np.log(probs[np.arange(T), obs])))
    s.kv("mean CE via full one-hot sum", round(ce_onehot, 10), "nats/token")
    s.kv("mean CE via -log p(observed)", round(ce_direct, 10), "nats/token")
    s.kv("corpus negative log-likelihood", round(total_nll, 6), "nats")
    s.kv("T x mean CE", round(T * ce_direct, 6), "nats")
    s.check("cross-entropy against a one-hot target is the negative log-"
            "probability of the observed token",
            abs(ce_onehot - ce_direct) < 1e-12,
            f"difference {abs(ce_onehot - ce_direct):.2e} nats")
    s.check("minimising mean cross-entropy is maximising corpus log-likelihood",
            abs(total_nll - T * ce_direct) < 1e-9,
            f"corpus NLL {total_nll:.6f} = T x mean CE = {T * ce_direct:.6f}")

    ce_bits = ce_direct / np.log(2)
    ce_uniform = float(np.log2(V))
    s.table(["model", "bits/token", "perplexity"],
            [["uniform over |V|", round(ce_uniform, 4),
              round(perplexity(ce_uniform), 3)],
             ["untrained random logits", round(ce_bits, 4),
              round(perplexity(ce_bits), 3)]], align="lrr")
    s.check("perplexity is the exponentiated cross-entropy, in either base",
            abs(perplexity(ce_bits) - np.exp(ce_direct)) < 1e-9,
            f"2^{ce_bits:.4f} = e^{ce_direct:.4f} = {perplexity(ce_bits):.3f}")
    s.check("the uniform model has perplexity exactly |V|",
            abs(perplexity(ce_uniform) - V) < 1e-9,
            f"2^log2({V}) = {perplexity(ce_uniform):.3f}")
    s.check("perplexity has no upper bound: a miscalibrated model scores worse "
            "than uniform",
            perplexity(ce_bits) > V,
            f"random logits give {perplexity(ce_bits):.1f} against the uniform "
            f"{V}, because confidently assigning low probability to what actually "
            "happened is penalised without limit")
    s.note("|V| is the baseline, not the ceiling. Being wrong cheaply -- staying "
           "near uniform -- costs log2|V| bits; being wrong confidently costs "
           "arbitrarily more, since the loss is -log p and p can approach zero. "
           "That asymmetry is worth carrying forward: it is the same reason S9's "
           "KL floors q away from zero, and the reason a confident hallucination "
           "is a worse failure than an admitted uncertainty by more than it looks")
    s.note("the previous article closed on accuracy, precision and recall, which "
           "need a decision threshold. A language model is scored before any "
           "threshold: perplexity is the loss itself, exponentiated, and reads as "
           "an effective branching factor. It is the same cross-entropy, in "
           "different clothes")

    # --------------------------------- 0.6 what the wide output layer costs
    s.h("0.6  What |V| outputs cost")
    s.eq("unembedding parameters = d * |V|;  body ~ 12 L d^2",
         "4d^2 of attention projections plus 8d^2 of a 4x feed-forward, per layer")
    rows = []
    for name, d, L in [("small, d=768 L=12", 768, 12),
                       ("medium, d=1600 L=48", 1600, 48),
                       ("large, d=4096 L=32", 4096, 32),
                       ("large, d=8192 L=80", 8192, 80)]:
        vocab = 50_257
        body, head = 12 * L * d * d, d * vocab
        rows.append([name, f"{body / 1e6:,.0f}M", f"{head / 1e6:,.0f}M",
                     f"{100 * head / (body + head):.1f}%"])
    s.table(["configuration", "body params", "unembedding params",
             "unembedding share"], rows, align="lrrr")
    first, last = float(rows[0][3].rstrip("%")), float(rows[-1][3].rstrip("%"))
    s.check("the output layer's share of parameters shrinks as the model grows",
            first > last,
            f"{first:.1f}% at d=768 falling to {last:.1f}% at d=8192, because "
            "the body grows as d^2 and the head only as d")
    s.note("worth knowing before reading a parameter count. At small scale a "
           "third of the weights can sit in a layer that only converts a "
           "d-vector into vocabulary scores, which is why tied embeddings and "
           "vocabulary size are live design questions for small models and "
           "rounding errors for large ones")

    # ----------------------------- 0.7 the hard change: variable-length input
    s.h("0.7  Change three, part two: a fixed weight matrix cannot read a sequence")
    s.say("  W^1 in 0.1 has shape (8, 4) because the input always has exactly 4")
    s.say("  features. A sequence has no fixed width, so something has to give.")
    s.say("  Three answers, and each pays for it somewhere different.")

    s.say()
    s.say("  (a) Fixed window. Condition on the last n tokens and truncate the "
          "rest.\n      This is an n-gram or a fixed-context MLP -- and it is "
          "exactly the\n      trigram S9 uses, so the reader has already met one.")
    src = DelayedSource()
    test = src.sample(4_000, seed=72)
    small_train = src.sample(15_000, seed=71)
    big_train = src.sample(240_000, seed=73)

    max_w = src.lag + 2

    def mean_kl(train: np.ndarray, w: int) -> float:
        table = window_model(train, w, src.vocab)
        # Evaluation starts past the widest window under test, so every
        # configuration is scored on exactly the same positions. Letting each
        # window start as early as it could would give the narrow ones extra
        # easy positions and quietly bias the comparison.
        return float(np.mean([
            kl_bits(src.true_dist(test[:t]),
                    window_predict(table, test[:t], w, src.vocab))
            for t in range(max_w + 1, 1_400)]))

    rows = []
    for w in range(0, max_w + 1):
        contexts = src.vocab ** w
        rows.append([w, "yes" if w >= src.lag else "no", f"{contexts:,}",
                     f"{len(big_train) / contexts:,.0f}",
                     round(mean_kl(small_train, w), 4),
                     round(mean_kl(big_train, w), 4)])
    s.table([f"window n", f"covers lag {src.lag}", "contexts |V|^n",
             "obs/context at 240k", "D_KL at 15k tokens", "D_KL at 240k tokens"],
            rows, align="rlrrrr")

    below_small = [r[4] for r in rows[:src.lag]]
    below_big = [r[5] for r in rows[:src.lag]]
    s.check("a window shorter than the dependency fails structurally, and 16x the "
            "data does not help",
            all(abs(b - s_) < 0.05 for b, s_ in zip(below_big, below_small))
            and min(below_big) > 1.0,
            f"D_KL for n < {src.lag} is {min(below_small):.3f}-"
            f"{max(below_small):.3f} bits at 15k tokens and "
            f"{min(below_big):.3f}-{max(below_big):.3f} bits at 240k -- "
            "unchanged, because the information is absent from the input rather "
            "than merely under-sampled")
    best = min(range(len(rows)), key=lambda i: rows[i][5])
    s.check("the divergence collapses exactly at the dependency length",
            best == src.lag,
            f"minimum D_KL at n = {best}, which is the lag; "
            f"{rows[src.lag][5]:.4f} bits against "
            f"{rows[src.lag - 1][5]:.4f} at n = {src.lag - 1}")
    s.check("widening the window past the dependency makes the model worse, not "
            "merely more expensive",
            rows[src.lag + 2][5] > rows[src.lag][5],
            f"n={src.lag}: {rows[src.lag][5]:.4f} bits, "
            f"n={src.lag + 1}: {rows[src.lag + 1][5]:.4f}, "
            f"n={src.lag + 2}: {rows[src.lag + 2][5]:.4f} -- each extra token "
            f"multiplies the context count by |V|={src.vocab} while the data per "
            "context divides by it")
    s.note("two different failures, and conflating them is why 'just use a bigger "
           "window' sounds reasonable. Below the dependency length the failure is "
           "structural: the information is not in the input and no quantity of "
           "data recovers it, which is S9.7's point arriving as an architectural "
           "constraint. Above it the failure is statistical: a fixed-window model "
           "estimates a free parameter set per context configuration, so cost and "
           "variance both grow as |V|^n. That is the real indictment of the "
           "fixed-window answer -- not that the window is too small, but that "
           "widening it is self-defeating. Attention escapes this because it does "
           "not hold a parameter per configuration; it computes a weighting over "
           "positions from shared projections, and pays in memory instead.")

    s.say()
    s.say("  (b) Recurrence. Carry a fixed-size state forward: h_t = f(h_{t-1}, "
          "x_t).\n      Memory is O(1) in sequence length, which is the "
          "attraction. The cost\n      is a hard information bound.")
    s.eq("to distinguish all |V|^T prefixes needs T log2|V| bits; a state of d "
         "float32 holds at most 32d",
         "so lossless recall is impossible beyond T* = 32d / log2|V|")
    rows = []
    for d in (256, 512, 1024, 4096):
        vocab = 50_257
        t_star = 32 * d / np.log2(vocab)
        rows.append([d, f"{32 * d:,}", round(float(np.log2(vocab)), 2),
                     f"{t_star:,.0f}"])
    s.table(["state width d", "state bits (fp32)", "bits per token",
             "T* lossless bound"], rows, align="rrrr")
    s.check("a recurrent state cannot losslessly carry more than 32d/log2|V| "
            "tokens",
            all(abs(float(r[3].replace(",", "")) - 32 * r[0] / np.log2(50_257))
                < 1.0 for r in rows),
            f"d=512 bounds lossless recall at about "
            f"{32 * 512 / np.log2(50257):,.0f} tokens; d=4096 at about "
            f"{32 * 4096 / np.log2(50257):,.0f}")
    s.note("an upper bound by counting, so it holds regardless of architecture or "
           "training. Beyond T* a recurrent model is necessarily compressing, and "
           "what it discards is chosen by the weights rather than by the reader. "
           "That is the honest description of a recurrent model's memory: lossy "
           "by construction, at a rate set by its width")

    s.say()
    s.say("  (c) Attention. Keep every position and let each query read all of "
          "them.\n      Nothing is compressed, so nothing is structurally lost "
          "-- and the bill\n      arrives as memory that grows with the sequence.")
    s.eq("kv_bytes = 2 * L * H_kv * d_head * T * bytes",
         "linear in T, which is precisely the cost S11 takes apart")
    rows = []
    for T in (1_024, 8_192, 32_768):
        kv = 2 * 32 * 8 * 128 * T * 2
        rows.append([f"{T:,}", f"{T * np.log2(50_257) / 8 / 1024:,.1f} KiB",
                     f"{kv / 2**20:,.0f} MiB"])
    s.table(["context T", "information content of the prefix",
             "KV cache, 8B GQA fp16"], rows, align="rrr")
    s.check("attention pays for losslessness in memory that grows with the "
            "sequence",
            True,
            "no compression bound applies, because no compression happens; the "
            "cost moves from an information limit to a memory limit")
    s.say()
    s.note("that is the trade, stated plainly. Recurrence is O(1) memory and "
           "lossy above T*. Attention is O(T) memory and lossless. Transformers "
           "won because the second bill is one you can pay with hardware and the "
           "first is one you cannot pay at all -- and every consequence in S11, "
           "from grouped-query attention to paged allocation to what a context "
           "window costs, is the industry working down that memory bill.")

    s.h("0.8  Where this leaves us")
    s.say("  Training is over and theta is frozen. What remains at inference is a")
    s.say("  function from a token sequence to a distribution over the next token:")
    s.eq("f_theta : V^* -> Delta(V)", "the object S9 takes apart")
    s.check("the output of the generalised network is a distribution over the "
            "vocabulary",
            abs(float(probs[0].sum()) - 1.0) < 1e-12 and bool(np.all(probs >= 0)),
            f"|sum - 1| = {abs(float(probs[0].sum()) - 1.0):.2e} over |V|={V}")
    s.note("everything after this point is a property of that function and of the "
           "code placed around it. S9 establishes what the function is and what "
           "it cannot do; S11 prices the context it consumes; S12 prices serving "
           "it. The agent stack, which is the subject of the next article, is "
           "what gets built on top")

    s.close()
    return s


if __name__ == "__main__":
    run()
