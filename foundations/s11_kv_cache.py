"""S11. Context, attention and the KV cache.

S9 established that the model is a stateless function of its context. This
section is about what that costs, because the cost is the reason every practical
constraint on agents exists: why context windows are finite, why the first token
is slow and the rest are fast, why "prompt caching" pays and why it stops paying
the moment you put a timestamp at the top of a system prompt.

The through-line is one algebraic fact. Under a causal mask, attention output at
position i depends only on keys and values at positions <= i. Appending a token
therefore cannot change any key or value already computed. The KV cache is not
an approximation, a heuristic, or a quality trade-off: it is that identity,
implemented. This section proves it to machine precision, then prices it.

Then the part that is a trade-off, and is routinely confused with the part that
is not: quantising the cache, or evicting from it with a sliding window, are
lossy. They change the function being computed. The section measures how much.

Everything runs in float64 NumPy on a randomly initialised two-layer
transformer. The weights are meaningless -- nothing here is trained, and nothing
here needs to be. Every claim is about the algebra of causal attention and the
arithmetic of its memory, both of which are identical in an untrained toy and a
frontier model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .harness import Section, entropy_bits, kl_bits, rng, softmax

# ------------------------------------------------------------------- FLOP counter

@dataclass
class Counter:
    """Multiply-accumulate counter.

    Wall-clock time is not the headline measurement in this section, on purpose.
    Timing a few hundred small NumPy calls on a memory-constrained laptop
    measures the machine's mood: the same call can differ by more than an order
    of magnitude depending on what else holds RAM at that moment. Counted
    multiply-accumulates are exact, hardware-independent and identical on any
    rerun, and the asymptotics -- which is what the section actually claims --
    are visible in them directly. One wall-clock ratio is reported at the end,
    clearly labelled as indicative.
    """

    macs: int = 0

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # An (m,k) @ (k,n) product is m*n*k multiply-accumulates.
        self.macs += int(np.prod(a.shape[:-1])) * a.shape[-1] * b.shape[-1]
        return a @ b

    def bmm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batched matmul over leading axes, as used for per-head attention."""
        self.macs += int(np.prod(a.shape[:-1])) * a.shape[-1] * b.shape[-1]
        return a @ b


# ------------------------------------------------------------------------ model

@dataclass
class Config:
    vocab: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_pos: int = 4096

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (x + 0.044715 * x ** 3)))


class Transformer:
    """A causal decoder, written out so the cache identity is visible.

    Untrained, float64, no batching. Those are all deliberate: float64 so that
    "identical" can mean 1e-15 rather than 1e-3, and no batch dimension so the
    indices in the attention expression are the indices in the equation.
    """

    def __init__(self, cfg: Config, seed: int = 5):
        self.cfg = cfg
        g = rng(seed)
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head

        def n(*shape, scale=None):
            # 1/sqrt(fan_in), the standard scaling. It matters here only because
            # it keeps activations off the softmax saturation floor; with
            # badly-scaled weights every attention row would be one-hot and 2.2's
            # comparison would be trivially satisfied for the wrong reason.
            s = scale if scale is not None else 1.0 / np.sqrt(shape[0])
            return g.normal(0.0, s, size=shape)

        self.embed = n(cfg.vocab, d, scale=1.0)
        self.pos = n(cfg.max_pos, d, scale=0.02)
        self.blocks = [{
            "wq": n(d, d), "wk": n(d, d), "wv": n(d, d), "wo": n(d, d),
            "w1": n(d, cfg.d_ff), "w2": n(cfg.d_ff, d),
        } for _ in range(cfg.n_layers)]
        self.unembed = n(d, cfg.vocab)
        self.scale = 1.0 / np.sqrt(dh)

    # ------------------------------------------------------------ full forward

    def forward(self, tokens: np.ndarray, ctr: Counter | None = None
                ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Score a whole sequence in one pass, returning logits and the cache.

            Q = X W_q,  K = X W_k,  V = X W_v
            A = softmax( Q K^T / sqrt(d_head) + M ) V,   M_ij = 0 if j<=i else -inf

        This is prefill. Every position is computed simultaneously, so the
        attention matrix is T x T per head and the cost is quadratic in T.
        """
        ctr = ctr or Counter()
        cfg = self.cfg
        T = len(tokens)
        x = self.embed[tokens] + self.pos[:T]

        # -inf above the diagonal. Position i may attend to j only for j <= i,
        # which is the entire reason a cache is possible: row i is a function of
        # the prefix up to i and of nothing after it.
        mask = np.triu(np.full((T, T), -np.inf), k=1)

        cache = []
        for blk in self.blocks:
            xn = layernorm(x)
            q = ctr.matmul(xn, blk["wq"])
            k = ctr.matmul(xn, blk["wk"])
            v = ctr.matmul(xn, blk["wv"])
            qh, kh, vh = (self._split(t) for t in (q, k, v))       # (H, T, dh)

            scores = ctr.bmm(qh, kh.transpose(0, 2, 1)) * self.scale
            probs = softmax(scores + mask[None, :, :], axis=-1)
            attn = self._merge(ctr.bmm(probs, vh))
            x = x + ctr.matmul(attn, blk["wo"])

            xn = layernorm(x)
            x = x + ctr.matmul(gelu(ctr.matmul(xn, blk["w1"])), blk["w2"])
            cache.append((kh, vh))

        return ctr.matmul(layernorm(x), self.unembed), cache

    # ---------------------------------------------------------- cached decode

    def step(self, token: int, pos: int,
             cache: list[tuple[np.ndarray, np.ndarray]],
             ctr: Counter | None = None,
             kv_transform=None) -> tuple[np.ndarray, list]:
        """One decode step for one new token, reusing the cache.

        The new token's query attends over the cached keys and values plus its
        own. No past key or value is recomputed, and by the causal mask none of
        them would change if they were. Cost is linear in the context length
        instead of quadratic, and there is no mask at all: a single query row
        attending to everything up to and including itself is already causal.

        `kv_transform` exists for 2.6. Passing a lossy function here -- int8
        quantisation, window eviction -- is how an approximate cache is
        distinguished, by measurement, from the exact one.
        """
        ctr = ctr or Counter()
        x = (self.embed[token] + self.pos[pos])[None, :]           # (1, d)
        new_cache = []
        for blk, (k_prev, v_prev) in zip(self.blocks, cache):
            xn = layernorm(x)
            q = self._split(ctr.matmul(xn, blk["wq"]))             # (H, 1, dh)
            k_new = self._split(ctr.matmul(xn, blk["wk"]))
            v_new = self._split(ctr.matmul(xn, blk["wv"]))

            k = np.concatenate([k_prev, k_new], axis=1)            # (H, t+1, dh)
            v = np.concatenate([v_prev, v_new], axis=1)
            k_used, v_used = (kv_transform(k, v) if kv_transform else (k, v))

            scores = ctr.bmm(q, k_used.transpose(0, 2, 1)) * self.scale
            probs = softmax(scores, axis=-1)
            attn = self._merge(ctr.bmm(probs, v_used))
            x = x + ctr.matmul(attn, blk["wo"])

            xn = layernorm(x)
            x = x + ctr.matmul(gelu(ctr.matmul(xn, blk["w1"])), blk["w2"])
            new_cache.append((k, v))                               # store exact

        return ctr.matmul(layernorm(x), self.unembed)[0], new_cache

    # -------------------------------------------------------------- head plumbing

    def _split(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        return x.reshape(T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 0, 2)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        H, T, dh = x.shape
        return x.transpose(1, 0, 2).reshape(T, H * dh)


# ------------------------------------------------------------- lossy KV variants

def int8_kv(k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-head int8 quantisation of the cache, then dequantisation.

        s = max|x| / 127,   x_q = round(x / s),   x_hat = s * x_q

    Four times smaller than fp32 and, unlike the exact cache, a different
    function. The scale is per head rather than per tensor because a single
    global scale is dominated by whichever head has the largest outlier, which
    is what makes naive KV quantisation look worse than it needs to.
    """
    out = []
    for t in (k, v):
        amax = np.max(np.abs(t), axis=(1, 2), keepdims=True)
        s = np.where(amax > 0, amax / 127.0, 1.0)
        out.append(s * np.round(t / s))
    return out[0], out[1]


def macs_prefill(cfg: Config, T: int) -> int:
    """Closed-form multiply-accumulate count for a prefill of length T.

    Per layer, reading straight off the implementation above:

        q,k,v projections      3 T d^2
        Q K^T                    T^2 d          (H heads x T x T x d_head)
        probs V                  T^2 d
        output projection        T d^2
        feed-forward           2 T d d_ff
        ------------------------------------
                        4 T d^2 + 2 T^2 d + 2 T d d_ff

    plus T d |V| once for the unembedding. Two terms matter and they scale
    differently: the projections are linear in T and quadratic in d, the
    attention is quadratic in T and linear in d. Which one dominates is
    therefore a question about T versus d, not a fact about transformers, and
    2.3 measures where the crossover falls.
    """
    d, dff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    per_layer = 4 * T * d * d + 2 * T * T * d + 2 * T * d * dff
    return L * per_layer + T * d * cfg.vocab


def macs_step(cfg: Config, t_ctx: int) -> int:
    """Cached decode of one token whose context already holds `t_ctx` tokens.

    Same expression with T = 1 for everything except the attention, which reads
    t_ctx + 1 cached positions:

        4 d^2 + 2 (t_ctx + 1) d + 2 d d_ff   per layer
    """
    d, dff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    per_layer = 4 * d * d + 2 * (t_ctx + 1) * d + 2 * d * dff
    return L * per_layer + d * cfg.vocab


def attention_share(cfg: Config, T: int) -> float:
    """Fraction of prefill MACs spent inside the attention product itself.

        share = 2 T^2 d / (2 T^2 d + 4 T d^2 + 2 T d d_ff)
              = T / (T + 2d + d_ff)

    So the crossover is linear in the model width, with a constant fixed by the
    feed-forward ratio. For this toy that is T = 384; for a d_model 4096 model
    with the usual 4x feed-forward it is around 24,000 tokens.
    """
    d, dff = cfg.d_model, cfg.d_ff
    return T / (T + 2 * d + dff)


def sliding_window(w: int):
    """Keep only the most recent `w` positions. Memory becomes O(w), not O(T).

    This is eviction, and eviction is forgetting: anything outside the window is
    unreachable, exactly as in S9.7 where truncating the conditioning set below
    the dependency length destroyed information that no decoder could recover.
    """
    def fn(k: np.ndarray, v: np.ndarray):
        return k[:, -w:, :], v[:, -w:, :]
    return fn


# --------------------------------------------------------------- memory formula

@dataclass
class ModelSpec:
    """Published architecture parameters for the memory arithmetic in 2.4."""

    name: str
    params: float
    layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> float:
        """2 * L * H_kv * d_head * bytes -- the 2 is one K and one V."""
        return 2 * self.layers * self.n_kv_heads * self.d_head * dtype_bytes

    def weight_bytes(self, dtype_bytes: int = 2) -> float:
        return self.params * dtype_bytes

    def crossover_tokens(self, batch: int = 1, dtype_bytes: int = 2) -> float:
        """Context length at which the KV cache equals the weights in size.

            T* = params * b_w / (2 * L * H_kv * d_head * b_kv * B)
        """
        return self.weight_bytes(dtype_bytes) / (
            self.kv_bytes_per_token(dtype_bytes) * batch)


SPECS = [
    # d_head 128 throughout, which is near-universal at this scale.
    ModelSpec("8B, multi-head (H_kv = H)", 8.03e9, 32, 32, 32, 128),
    ModelSpec("8B, grouped-query (H_kv = 8)", 8.03e9, 32, 32, 8, 128),
    ModelSpec("8B, multi-query (H_kv = 1)", 8.03e9, 32, 32, 1, 128),
    ModelSpec("70B, grouped-query (H_kv = 8)", 70.6e9, 80, 64, 8, 128),
]


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s11", "Context, attention and the KV cache",
                "one algebraic identity, and everything it pays for").open()

    cfg = Config()
    model = Transformer(cfg)
    g = rng(31)

    s.h("2.1  The object under study")
    s.kv("layers", cfg.n_layers)
    s.kv("d_model", cfg.d_model)
    s.kv("heads x d_head", f"{cfg.n_heads} x {cfg.d_head}")
    s.kv("vocabulary", cfg.vocab)
    s.kv("dtype", "float64")
    s.note("untrained random weights. Nothing below depends on the weights "
           "meaning anything: the claims are about the algebra of causal "
           "attention and the arithmetic of its memory, which a toy and a "
           "frontier model share exactly")

    # ------------------------------------------------------- 2.2 the identity
    s.h("2.2  Why the cache is exact, not approximate")
    s.eq("A_i = sum_{j<=i} softmax_j( q_i . k_j / sqrt(d_h) ) v_j",
         "position i attends only to positions j <= i, so K_j and V_j for j <= i "
         "cannot depend on anything appended after position i")
    s.say()
    s.say("  Test: score a 40-token prefix, then extend by 12 tokens two ways.")
    s.say("    (a) recompute the full 52-token sequence from scratch")
    s.say("    (b) run 12 cached single-token steps from the prefix cache")

    tokens = g.integers(0, cfg.vocab, size=52)
    n_pre = 40

    full_logits, _ = model.forward(tokens)
    _, cache = model.forward(tokens[:n_pre])
    inc = []
    for t in range(n_pre, len(tokens)):
        logit, cache = model.step(int(tokens[t]), t, cache)
        inc.append(logit)
    inc = np.array(inc)

    diff = float(np.max(np.abs(full_logits[n_pre:] - inc)))
    s.kv("max |logit difference| over 12 x 32 logits", f"{diff:.3e}")
    s.kv("float64 epsilon", f"{np.finfo(np.float64).eps:.3e}")
    s.check("cached incremental decoding equals full recomputation",
            diff < 1e-12,
            f"max abs difference {diff:.3e} across 384 logits, i.e. accumulated "
            "floating-point round-off and nothing else")

    p_full = softmax(full_logits[-1])
    p_inc = softmax(inc[-1])
    s.check("the resulting next-token distributions are identical to 1e-15",
            float(np.max(np.abs(p_full - p_inc))) < 1e-15,
            f"D_KL = {kl_bits(p_full, p_inc):.3e} bits")
    s.note("so a KV cache costs nothing in quality, ever. If a system's outputs "
           "change when caching is switched on, the cache is not the "
           "explanation -- a bug, a batching-dependent kernel or a changed "
           "prefix is. This is worth being firm about, because it is the one "
           "cache in the stack that is free, and 2.6 covers the variants that "
           "are not.")

    # -------------------------------------------------------- 2.3 the cost
    s.h("2.3  What the cache removes: exactly one factor of T")
    s.eq("prefill:      4 T d^2 + 2 T^2 d + 2 T d d_ff",
         "projections are linear in T and quadratic in d; attention is the "
         "reverse, so which term dominates depends on T against d")
    s.eq("cached step:  Theta(T d)   attention + Theta(d^2)   projections",
         "one query row against T cached keys")
    s.eq("uncached step: Theta(T^2 d) -- the whole prefix again, every token",
         "so generating T tokens without a cache is Theta(T^3 d) in total")

    # Validate the closed forms against the instrumented counter, then use them.
    # Instrumenting the uncached case out to T = 8192 would mean running 8192
    # full prefills of a growing sequence, which is hours of float64 NumPy for
    # numbers the formula gives exactly. Checking the formula against the
    # instrumentation at a size where both are cheap is the honest shortcut.
    c_pre = Counter()
    _, cache_probe = model.forward(np.asarray(tokens[:40]), c_pre)
    c_one = Counter()
    model.step(int(tokens[40]), 40, cache_probe, c_one)
    s.check("the closed-form MAC counts match the instrumented implementation",
            c_pre.macs == macs_prefill(cfg, 40) and c_one.macs == macs_step(cfg, 40),
            f"prefill {c_pre.macs} vs formula {macs_prefill(cfg, 40)}; "
            f"step {c_one.macs} vs formula {macs_step(cfg, 40)}")

    def gen_cached(p: int, n: int) -> int:
        return macs_prefill(cfg, p) + sum(macs_step(cfg, p + i) for i in range(n))

    def gen_uncached(p: int, n: int) -> int:
        return macs_prefill(cfg, p) + sum(macs_prefill(cfg, p + i + 1)
                                          for i in range(n))

    rows = []
    for T in (32, 128, 384, 1024, 4096, 8192):
        cch, unc = gen_cached(8, T), gen_uncached(8, T)
        rows.append([T, f"{cch:.3e}", f"{unc:.3e}", round(unc / cch, 1),
                     round(attention_share(cfg, T) * 100, 1)])
    s.table(["tokens generated", "MACs with cache", "MACs without", "speedup",
             "attention % of prefill"], rows, align="rrrrr")

    Ts = np.array([r[0] for r in rows], dtype=float)
    lo, hi = slice(0, 3), slice(3, 6)

    def slope(fn, sl):
        return float(np.polyfit(np.log(Ts[sl]),
                                np.log([fn(8, int(t)) for t in Ts[sl]]), 1)[0])

    s.table(["regime", "cached exponent", "uncached exponent", "gap"],
            [["T < d_model (projection-bound)", round(slope(gen_cached, lo), 3),
              round(slope(gen_uncached, lo), 3),
              round(slope(gen_uncached, lo) - slope(gen_cached, lo), 3)],
             ["T >> d_model (attention-bound)", round(slope(gen_cached, hi), 3),
              round(slope(gen_uncached, hi), 3),
              round(slope(gen_uncached, hi) - slope(gen_cached, hi), 3)]],
            align="lrrr")
    s.check("the cache removes exactly one factor of T, in either regime",
            abs((slope(gen_uncached, lo) - slope(gen_cached, lo)) - 1.0) < 0.2
            and abs((slope(gen_uncached, hi) - slope(gen_cached, hi)) - 1.0) < 0.2,
            f"gap {slope(gen_uncached, lo) - slope(gen_cached, lo):.3f} at small "
            f"T and {slope(gen_uncached, hi) - slope(gen_cached, hi):.3f} at "
            "large T")
    s.check("attention only dominates prefill past T ~ 2d + d_ff",
            attention_share(cfg, 2 * cfg.d_model + cfg.d_ff) - 0.5 < 1e-9
            and attention_share(cfg, 32) < 0.1,
            f"share is {attention_share(cfg, 32) * 100:.1f}% at T=32, "
            f"50% at T={2 * cfg.d_model + cfg.d_ff}, "
            f"{attention_share(cfg, 8192) * 100:.1f}% at T=8192")
    s.note("worth being precise about, because 'attention is quadratic' is "
           "usually quoted as though it were the dominant cost at every length. "
           "It is not. Prefill is 4Td^2 + 2T^2 d + 2Td*d_ff, so the quadratic "
           "term only takes over past T ~ 2d + d_ff -- 384 tokens for this toy, "
           "roughly 24k tokens for a d_model 4096 model with a 4x feed-forward. "
           "Below that, prompt cost is linear in length and set by model width. "
           "What is true at every length is the exponent gap: the cache removes "
           "one factor of T, which is why time-to-first-token and inter-token "
           "latency are quoted separately and respond to different fixes.")

    t0 = time.perf_counter()
    cch = [(k.copy(), v.copy()) for k, v in cache_probe]
    for i in range(64):
        _, cch = model.step(int(g.integers(cfg.vocab)), 40 + i, cch)
    t_cached = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(64):
        model.forward(np.asarray(list(tokens[:40]) + [0] * (i + 1)))
    t_naive = time.perf_counter() - t0
    s.kv("wall clock, 64 tokens cached", round(t_cached * 1e3, 1), "ms")
    s.kv("wall clock, 64 tokens uncached", round(t_naive * 1e3, 1), "ms")
    s.kv("wall-clock ratio", round(t_naive / t_cached, 2), "x")
    s.note("indicative only, and deliberately not a claim. Timing small NumPy "
           "calls on a memory-constrained machine measures the machine's mood; "
           "the MAC counts above are the reproducible statement of the same fact")

    # ------------------------------------------------------ 2.4 memory budget
    s.h("2.4  What the cache costs in memory, and when it beats the weights")
    s.eq("kv_bytes = 2 * L * H_kv * d_head * T * B * bytes_per_element",
         "the leading 2 is one K and one V; note it is linear in context and in "
         "batch, while the weights are constant in both")
    s.table(["configuration", "KV per token", "T* at B=1", "T* at B=32",
             "KV at T=32k, B=32"],
            [[sp.name,
              f"{sp.kv_bytes_per_token() / 1024:.0f} KiB",
              f"{sp.crossover_tokens(1):,.0f}",
              f"{sp.crossover_tokens(32):,.0f}",
              f"{sp.kv_bytes_per_token() * 32_768 * 32 / 2**30:,.0f} GiB"]
             for sp in SPECS])
    s.note("T* is the context length at which the cache is as large as the "
           "weights, in fp16 for both")

    mha, gqa, mqa = SPECS[0], SPECS[1], SPECS[2]
    s.check("grouped-query attention cuts KV memory by exactly H/H_kv",
            abs(mha.kv_bytes_per_token() / gqa.kv_bytes_per_token()
                - mha.n_heads / gqa.n_kv_heads) < 1e-9,
            f"{mha.kv_bytes_per_token() / 1024:.0f} KiB/token to "
            f"{gqa.kv_bytes_per_token() / 1024:.0f} KiB/token, a factor of "
            f"{mha.n_heads / gqa.n_kv_heads:.0f}")
    s.check("at serving batch sizes the KV cache overtakes the weights well "
            "inside advertised context lengths",
            gqa.crossover_tokens(32) < 32_768,
            f"8B GQA in fp16: weights {gqa.weight_bytes() / 2**30:.1f} GiB, "
            f"KV equals that at {gqa.crossover_tokens(32):,.0f} tokens when "
            "B=32")
    s.note("this single table explains most of the serving stack. GQA and MQA "
           "exist to shrink H_kv. Paged attention exists because that memory is "
           "allocated per sequence and fragments. Context-length pricing exists "
           "because context is the term that scales with both T and B while the "
           "weights do not. And for an agent the relevant number is not the "
           "window but the transcript: every step appends, so T only ever grows.")

    # ------------------------------------------------- 2.5 prefix reuse
    s.h("2.5  Prompt caching is prefix reuse, and one volatile token destroys it")
    s.eq("reusable(a, b) = max{ n : a_1..a_n == b_1..b_n }",
         "cached K,V for position i are valid for a new request only if every "
         "token up to i is unchanged, because attention at i reads all of them")

    def reusable(a: list[int], b: list[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    system = list(g.integers(0, cfg.vocab, size=44))     # static instructions
    tools = list(g.integers(0, cfg.vocab, size=20))      # static tool schemas
    stamp_a, stamp_b = [7, 7, 7], [9, 9, 9]              # the volatile token(s)
    user_a = list(g.integers(0, cfg.vocab, size=8))
    user_b = list(g.integers(0, cfg.vocab, size=8))

    s.say()
    s.say("  (a) Two requests sharing a system prompt and tool schemas, "
          "differing in the\n      user turn. The only variable is where the "
          "volatile token sits.")
    static = system + tools
    layouts = [
        ("stamp, static, user", stamp_a + static + user_a,
         stamp_b + static + user_b),
        ("static, stamp, user", static + stamp_a + user_a,
         static + stamp_b + user_b),
        ("static, user, stamp", static + user_a + stamp_a,
         static + user_b + stamp_b),
    ]
    rows = [[name, len(r1), reusable(r1, r2),
             f"{100 * reusable(r1, r2) / len(r1):.0f}%"]
            for name, r1, r2 in layouts]
    s.table(["prompt layout", "tokens", "reusable prefix", "% cacheable"], rows,
            align="lrrr")
    s.check("a volatile token at position 1 makes the entire prompt uncacheable",
            rows[0][2] == 0,
            f"reusable prefix {rows[0][2]} of {rows[0][1]} tokens, against "
            f"{rows[1][2]} once the stamp moves behind the static block")
    s.check("the reusable prefix stops at the first differing token, wherever "
            "that is",
            rows[1][2] == rows[2][2] == len(static),
            f"both layouts reuse exactly the {len(static)} static tokens: the "
            "differing user turn ends reuse, so moving the stamp after it "
            "changes nothing")
    s.note("only the first divergence matters, so 'volatile content last' is "
           "really 'volatile content after everything you want cached'. Putting "
           "a timestamp behind a user turn that already differs buys nothing")

    s.say()
    s.say("  (b) The case that matters for agents: successive steps of one task, "
          "where\n      each step appends the last tool result to the transcript.")
    transcript = static + user_a
    prev, rows = list(transcript), []
    for step in range(1, 6):
        # Append-only growth, which is what an agent loop does by default.
        nxt = prev + list(g.integers(0, cfg.vocab, size=14))
        rows.append([step, len(nxt), reusable(prev, nxt),
                     f"{100 * reusable(prev, nxt) / len(nxt):.0f}%", "append"])
        prev = nxt
    append_final = rows[-1][2]

    # Now the same five steps, except the orchestrator compacts history at
    # step 3 -- summarising, re-ordering, or dropping a stale tool result.
    prev2, rows2 = list(transcript), []
    for step in range(1, 6):
        if step == 3:
            nxt = static + list(g.integers(0, cfg.vocab, size=18))  # rewritten
        else:
            nxt = prev2 + list(g.integers(0, cfg.vocab, size=14))
        rows2.append([step, len(nxt), reusable(prev2, nxt),
                      f"{100 * reusable(prev2, nxt) / len(nxt):.0f}%",
                      "compact" if step == 3 else "append"])
        prev2 = nxt

    s.table(["step", "prompt tokens", "reusable prefix", "% cacheable", "action"],
            rows + [["", "", "", "", ""]] + rows2, align="rrrlr")
    s.check("an append-only transcript keeps almost the whole prompt cacheable "
            "at every step",
            all(r[2] / r[1] > 0.8 for r in rows),
            f"lowest reuse across five steps is "
            f"{min(r[2] / r[1] for r in rows) * 100:.0f}%")
    s.check("compacting history discards the cache from the first edited token on",
            rows2[2][2] == len(static) and rows2[2][2] < rows[2][2],
            f"reuse drops from {rows[2][2]} tokens to {rows2[2][2]} at the "
            "compaction step, and has to be rebuilt from there")
    s.note("this is the whole content of the advice to put static instructions "
           "first and volatile content last, and it is not a vendor quirk. The "
           "cache is keyed on an exact token prefix because that is what the "
           "causal mask makes it valid for. A timestamp, a request id or a "
           "reordered tool list at the top of a system prompt costs the full "
           "prefill on every call. For an agent the corollary is sharper: the "
           "transcript is append-only, so the prefix is stable by construction "
           "and reuse across steps of one task is nearly total -- unless the "
           "orchestrator rewrites or re-summarises history, which invalidates "
           "everything from the first edited token onward.")

    # ------------------------------------------------- 2.6 the lossy variants
    s.h("2.6  Quantising or evicting from the cache is a different function")
    _, base_cache = model.forward(tokens[:n_pre])
    ref, _ = model.step(int(tokens[n_pre]), n_pre,
                        [(k.copy(), v.copy()) for k, v in base_cache])
    p_ref = softmax(ref)

    rows = []
    variants = [("exact", None),
                ("int8 per-head quantised", int8_kv),
                ("sliding window, w=32", sliding_window(32)),
                ("sliding window, w=16", sliding_window(16)),
                ("sliding window, w=8", sliding_window(8))]
    for name, fn in variants:
        out, _ = model.step(int(tokens[n_pre]), n_pre,
                            [(k.copy(), v.copy()) for k, v in base_cache],
                            kv_transform=fn)
        p = softmax(out)
        mem = 1.0 if fn is None else (0.25 if fn is int8_kv
                                      else min(int(name.split("w=")[1]), n_pre + 1)
                                      / (n_pre + 1))
        rows.append([name, f"{np.max(np.abs(out - ref)):.2e}",
                     round(kl_bits(p, p_ref), 6),
                     "same" if np.argmax(p) == np.argmax(p_ref) else "CHANGED",
                     f"{mem:.2f}x"])
    s.table(["cache variant", "max |logit delta|", "D_KL vs exact (bits)",
             "argmax", "KV bytes at T=41"], rows, align="lrrlr")

    s.check("the exact cache is bit-identical while every compressed variant "
            "changes the output",
            rows[0][2] == 0.0 and all(r[2] > 0.0 for r in rows[1:]),
            f"exact D_KL = {rows[0][2]}, int8 = {rows[1][2]:.2e} bits, "
            f"w=8 = {rows[-1][2]:.2e} bits")
    s.check("divergence grows as the window shrinks",
            rows[2][2] < rows[3][2] < rows[4][2],
            f"w=32 {rows[2][2]:.2e} < w=16 {rows[3][2]:.2e} < w=8 "
            f"{rows[4][2]:.2e} bits")
    s.note("keep these in separate mental columns. The KV cache is an identity "
           "and costs nothing. int8 KV and windowed attention are lossy "
           "compressions that buy memory with accuracy, and the exchange rate "
           "is a function of the workload -- a window that is harmless for "
           "chat is not harmless for an agent whose decisive evidence is a tool "
           "result forty steps back. When a vendor says 'caching', find out "
           "which of these is meant.")

    s.h("2.7  Attention entropy: what the scaling factor is for")
    s.eq("q.k has variance ~ d_head for unit-variance inputs, hence the "
         "1/sqrt(d_head)",
         "without it the softmax saturates as d_head grows and attention "
         "collapses onto a single position")
    # 512 independent attention rows per width, each a query against 64 keys.
    # A single row is far too noisy to read a trend off: the entropy of one
    # softmax depends on the gap between its own largest two samples, which
    # varies more between draws than it does between these widths.
    rows = []
    for dh in (8, 32, 128, 512):
        gg = rng(77)
        q = gg.normal(size=(512, 1, dh))
        k = gg.normal(size=(512, 64, dh))
        raw = np.einsum("nid,njd->nj", q, k)            # (512, 64) score rows
        scaled = raw / np.sqrt(dh)
        ent_raw = float(np.mean([entropy_bits(softmax(r)) for r in raw]))
        ent_scaled = float(np.mean([entropy_bits(softmax(r)) for r in scaled]))
        rows.append([dh, round(float(raw.var()), 1), round(float(scaled.var()), 3),
                     round(ent_raw, 3), round(ent_scaled, 3)])
    s.table(["d_head", "var(q.k)", "var(q.k/sqrt(d))", "mean H unscaled bits",
             "mean H scaled bits"], rows, align="rrrrr")
    s.kv("uniform attention over 64 keys would be",
         round(float(np.log2(64)), 3), "bits")
    s.check("the raw dot-product variance tracks d_head and the scaled variance "
            "does not",
            all(abs(r[1] / r[0] - 1) < 0.15 for r in rows)
            and all(abs(r[2] - 1) < 0.15 for r in rows),
            "var(q.k)/d_head within 15% of 1 at every width; scaled variance "
            "within 15% of 1")
    unscaled = [r[3] for r in rows]
    scaled_h = [r[4] for r in rows]
    s.check("without scaling, attention entropy falls monotonically as d_head grows",
            all(unscaled[i] > unscaled[i + 1] for i in range(len(unscaled) - 1)),
            " > ".join(f"{h:.2f}" for h in unscaled) + " bits")
    s.check("with scaling, attention entropy is invariant to d_head",
            max(scaled_h) - min(scaled_h) < 0.1,
            f"spread of {max(scaled_h) - min(scaled_h):.3f} bits across a 64x "
            f"range of widths, all near {np.mean(scaled_h):.2f}")
    s.note("a saturated softmax is a hard argmax with no gradient, which is a "
           "training failure rather than an inference one -- but it is the same "
           "saturation S10.2 showed at low temperature, and worth recognising as "
           "one phenomenon appearing twice")

    s.close()
    return s


if __name__ == "__main__":
    run()
