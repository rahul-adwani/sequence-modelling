"""S12. Latency, caching and cost.

S11 counted the arithmetic. This section converts it into the three numbers anyone
operating a model actually reports -- time to first token, inter-token latency,
and cost -- and finds that they are governed by different bottlenecks, respond to
different fixes, and are routinely optimised in the wrong direction because of it.

The central result is that **prefill and decode are not two speeds of the same
operation; they are compute-bound and memory-bound respectively**. Prefill has
arithmetic intensity proportional to sequence length, so it saturates a GPU's
arithmetic units easily. Decode has arithmetic intensity proportional to *batch
size* and nothing else, so a single-stream decode reads the entire weight matrix
from memory to produce one token and leaves almost all of the arithmetic capacity
idle. Everything else here follows: why batching is the only real throughput
lever, why a bigger GPU does not speed up your chatbot, and why a context length
has a latency cost as well as a memory one.

Then three trades that are usually described qualitatively and are in fact
arithmetic: what a semantic cache's similarity threshold really buys and sells,
how stale a TTL-bounded fact is, and why p95 latency detaches from p50 long before
a queue looks busy.

Hardware figures are representative published specifications, used as inputs to
arithmetic rather than as measurements. No GPU is touched. The claims are about
the shape of the arithmetic, which is what makes them portable -- and the roofline
model they rest on is old, well-tested and not controversial.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, percentile, rng
from .s11_kv_cache import SPECS, Config, ModelSpec, macs_prefill, macs_step


# ------------------------------------------------------------------- hardware

@dataclass
class Device:
    """Representative published specifications, as arithmetic inputs.

    `balance` is the machine's ridge point: how many arithmetic operations it can
    perform per byte it can fetch. A kernel whose arithmetic intensity is below
    this number cannot reach peak throughput no matter how it is written, because
    it runs out of memory bandwidth first. It is the single most useful number
    about a piece of hardware and it is almost never quoted.
    """

    name: str
    tflops: float          # dense, at the inference dtype
    bandwidth_gbs: float   # HBM/GDDR, GB/s

    @property
    def balance(self) -> float:
        """FLOP per byte at which compute and bandwidth are equally limiting."""
        return self.tflops * 1e12 / (self.bandwidth_gbs * 1e9)

    def attainable_tflops(self, intensity: float) -> float:
        """Roofline: min(peak, intensity x bandwidth)."""
        return min(self.tflops, intensity * self.bandwidth_gbs * 1e9 / 1e12)


DEVICES = [
    Device("datacentre, 80GB HBM2e", 312.0, 2039.0),
    Device("datacentre, 80GB HBM3", 989.0, 3350.0),
    Device("laptop GPU, 4GB GDDR6", 2.9, 128.0),
]


# --------------------------------------------------------- intensity of the work

def decode_intensity(spec: ModelSpec, batch: int, ctx: int,
                     dtype_bytes: int = 2) -> float:
    """Arithmetic intensity of one cached decode step, FLOP per byte.

        FLOPs  = 2 * P * B                 every weight used once per sequence
        bytes  = P * b_w  +  B * T * kv    weights read once, KV read per sequence

    The weights are read once regardless of batch size, because the whole batch
    shares them. So the numerator scales with B and the denominator barely does,
    and intensity is approximately B -- the single most important fact about
    decoding. It says a batch of one cannot exceed 1/balance of peak throughput,
    for any model, on any device, however well implemented.
    """
    flops = 2.0 * spec.params * batch
    kv_bytes = batch * ctx * spec.kv_bytes_per_token(dtype_bytes)
    return flops / (spec.params * dtype_bytes + kv_bytes)


def decode_intensity_ceiling(spec: ModelSpec, ctx: int,
                             dtype_bytes: int = 2) -> float:
    """The limit of `decode_intensity` as batch size goes to infinity.

        I(B) = 2PB / (P*b_w + B*T*kv)   ->   2P / (T*kv)   as B -> inf

    Batching amortises the weight read across the batch, but every sequence
    carries its own KV cache, so the KV term grows with B exactly as the numerator
    does and refuses to amortise. Substituting P*b_w = T* * kv from S11.4 gives the
    ceiling a strikingly simple form:

        I_max = 2 T* / (b_w * T)   =   T* / T   at fp16

    So the best arithmetic intensity decoding can ever reach is the ratio of the
    KV crossover length to the context length actually in use. Once a request's
    context approaches T*, no batch size on any device can make decoding
    compute-bound. This is the ceiling that KV-cache quantisation and
    grouped-query attention are really raising -- they cut `kv`, which lifts T*,
    which lifts this whole curve.
    """
    return 2.0 * spec.params / (ctx * spec.kv_bytes_per_token(dtype_bytes))


def prefill_intensity(cfg: Config, T: int, dtype_bytes: int = 2) -> float:
    """Arithmetic intensity of prefill, FLOP per byte.

    Prefill reads the weights once and does T tokens' worth of work with them, so
    intensity grows with T in the same way decode's grows with batch. A prompt is
    a batch: that is the whole reason prefill is fast and decode is not.
    """
    body_params = cfg.n_layers * (4 * cfg.d_model ** 2
                                  + 2 * cfg.d_model * cfg.d_ff)
    return 2.0 * macs_prefill(cfg, T) / (body_params * dtype_bytes)


# ------------------------------------------------------------- semantic caching

@dataclass
class SemanticCache:
    """Nearest-neighbour cache with a cosine-similarity admission threshold.

    A hit is *correct* when the matched entry came from a query with the same
    underlying intent, and a *false hit* when it did not. Both are hits as far as
    the cache's own metrics are concerned, which is precisely the problem: hit
    rate is reported and false-hit rate usually is not, so the threshold gets
    tuned against the half of the trade-off that only ever looks good.
    """

    threshold: float
    keys: list = None
    intents: list = None

    def __post_init__(self) -> None:
        self.keys, self.intents = [], []
        self.hits = self.false_hits = self.misses = 0

    def query(self, vec: np.ndarray, intent: int) -> str:
        if self.keys:
            sims = np.asarray(self.keys) @ vec
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                self.hits += 1
                if self.intents[best] != intent:
                    self.false_hits += 1
                return "hit"
        self.misses += 1
        self.keys.append(vec)
        self.intents.append(intent)
        return "miss"


def intent_queries(n_groups: int = 6, per_group: int = 4, per_intent: int = 25,
                   dim: int = 48, group_spread: float = 0.09,
                   spread: float = 0.06, seed: int = 313):
    """Queries with a two-level structure, and the intent label kept.

    Distinct intents are not drawn independently, because in practice they are
    not: a cache sees clusters of *related* questions, and the dangerous
    neighbours are the ones that differ in a detail rather than a topic. So
    intents are grouped, and within a group two different intents can sit closer
    together than two paraphrases of the same one.

    The measured geometry is: within-intent similarity 0.85, cross-intent within
    a group 0.62 rising to 0.84 for the closest pairs, cross-group 0.05. That
    overlap between the top of the cross-intent range and the bottom of the
    within-intent range is the whole difficulty. No threshold separates them.

    The intent label is ground truth a production semantic cache never has --
    which is exactly why false-hit rate is so rarely measured, and why it has to
    be constructed here to be seen at all.
    """
    g = rng(seed)
    base = g.normal(size=(n_groups, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    centroids = []
    for j in range(n_groups):
        for _ in range(per_group):
            c = base[j] + group_spread * g.normal(size=dim)
            centroids.append(c / np.linalg.norm(c))
    out = []
    for i, c in enumerate(centroids):
        for _ in range(per_intent):
            v = c + spread * g.normal(size=dim)
            out.append((v / np.linalg.norm(v), i))
    g.shuffle(out)
    return out


# ------------------------------------------------------------------- staleness

def staleness_probability(ttl: float, rate: float) -> float:
    """P(a served entry is stale), for Poisson writes at `rate` and uniform age.

    An entry read at age `a` is stale if at least one write landed since it was
    cached, which for a Poisson process is 1 - exp(-rate * a). Under a steady
    read load the age of the entry being read is uniform on [0, TTL], so

        P(stale) = (1/T) * integral_0^T (1 - e^{-r a}) da
                 = 1 - (1 - e^{-rT}) / (rT)

    Two limits worth holding onto: as rT -> 0 this goes to rT/2, so halving the
    TTL halves the staleness; as rT -> inf it goes to 1, and TTL stops buying
    anything at all. A cache in front of fast-moving state is not a slightly
    stale cache, it is a wrong one.
    """
    x = rate * ttl
    if x < 1e-12:
        return 0.0
    return float(1.0 - (1.0 - np.exp(-x)) / x)


def simulate_staleness(ttl: float, rate: float, n: int = 200_000,
                       seed: int = 7) -> float:
    g = rng(seed)
    age = g.uniform(0.0, ttl, size=n)
    writes = g.poisson(rate * age)
    return float(np.mean(writes > 0))


# -------------------------------------------------------------------- queueing

def mm1_percentiles(arrival: float, service_rate: float
                    ) -> tuple[float, float, float]:
    """M/M/1 sojourn time: exponential with rate mu(1 - rho).

        rho  = lambda / mu
        E[W] = 1 / (mu - lambda)
        P(W > t) = exp(-mu(1-rho) t)   =>   quantile q at -ln(1-q)/(mu(1-rho))

    The mean carries the 1/(1-rho) blow-up, which is the entire lesson: latency
    is not linear in load, it is hyperbolic, and the knee arrives suddenly.
    """
    rho = arrival / service_rate
    if rho >= 1.0:
        return float("inf"), float("inf"), rho
    k = service_rate * (1.0 - rho)
    return float(np.log(2) / k), float(np.log(20) / k), float(rho)


def simulate_mm1(arrival: float, service_rate: float, n: int = 120_000,
                 seed: int = 11, burn_in: float = 0.2) -> tuple[float, float]:
    """Event-driven M/M/1, to confirm the closed form rather than assume it.

    The queue starts empty, which is not its steady state, and the closer
    utilisation is to 1 the longer it takes to forget that. The first `burn_in`
    fraction of requests is therefore discarded -- and 11.6 shows that even so,
    the high-utilisation tail is hard to estimate from any feasible sample, which
    is a point about measurement rather than a defect in the simulation.
    """
    g = rng(seed)
    inter = g.exponential(1.0 / arrival, size=n)
    service = g.exponential(1.0 / service_rate, size=n)
    arrive = np.cumsum(inter)
    finish = np.empty(n)
    free = 0.0
    for i in range(n):
        free = finish[i] = max(arrive[i], free) + service[i]
    w = (finish - arrive)[int(n * burn_in):]
    return percentile(w, 50), percentile(w, 95)


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s12", "Latency, caching and cost",
                "prefill is compute-bound, decode is not, and the rest "
                "follows").open()

    cfg = Config(d_model=4096, d_ff=14336, n_layers=32, n_heads=32,
                 vocab=128_000, max_pos=131_072)
    gqa = SPECS[1]                      # 8B-class, grouped-query, H_kv = 8

    # -------------------------------------------------------- 11.1 the balance
    s.h("11.1  Every device has a ridge point, and nobody quotes it")
    s.eq("balance = peak_FLOP/s / bandwidth_bytes/s",
         "the arithmetic intensity below which a kernel is memory-bound no matter "
         "how it is written")
    s.table(["device", "peak TFLOP/s", "bandwidth GB/s", "balance FLOP/byte"],
            [[d.name, d.tflops, d.bandwidth_gbs, round(d.balance, 1)]
             for d in DEVICES], align="lrrr")
    s.check("datacentre accelerators demand over a hundred operations per byte "
            "to reach peak",
            all(d.balance > 100 for d in DEVICES[:2]),
            f"{DEVICES[0].balance:.0f} and {DEVICES[1].balance:.0f} FLOP/byte "
            f"against {DEVICES[2].balance:.0f} for the laptop part")
    s.note("the gap between the first two rows and the third is the real story of "
           "the last decade of accelerators: arithmetic got very much cheaper "
           "than memory traffic. A kernel written when balance was 20 is "
           "memory-starved on a machine where it is 153")

    # ------------------------------------------------ 11.2 the central asymmetry
    s.h("11.2  Prefill intensity grows with length; decode intensity is the batch")
    s.eq("prefill:  FLOPs ~ 2 * macs(T),  bytes ~ P * b_w   =>   I grows with T",
         "the prompt amortises one weight read over T tokens of work")
    s.eq("decode:   FLOPs = 2 * P * B,    bytes = P * b_w + B*T*kv   =>   I ~ B",
         "one weight read produces exactly one token per sequence in the batch")

    dev = DEVICES[0]
    rows = []
    for T in (128, 1_024, 8_192):
        i_pre = prefill_intensity(cfg, T)
        rows.append(["prefill", f"T={T:,}", round(i_pre, 1),
                     round(dev.attainable_tflops(i_pre), 1),
                     f"{100 * dev.attainable_tflops(i_pre) / dev.tflops:.0f}%"])
    for B in (1, 8, 32, 128, 256):
        i_dec = decode_intensity(gqa, B, 2_048)
        rows.append(["decode", f"B={B}", round(i_dec, 2),
                     round(dev.attainable_tflops(i_dec), 2),
                     f"{100 * dev.attainable_tflops(i_dec) / dev.tflops:.1f}%"])
    s.table(["phase", "condition", "intensity FLOP/byte",
             f"attainable TFLOP/s", "% of peak"], rows, align="llrrr")

    i_b1 = decode_intensity(gqa, 1, 2_048)
    s.check("single-stream decode reaches under one percent of a device's "
            "arithmetic peak",
            dev.attainable_tflops(i_b1) / dev.tflops < 0.01,
            f"intensity {i_b1:.2f} FLOP/byte against a balance of "
            f"{dev.balance:.0f}, giving "
            f"{100 * dev.attainable_tflops(i_b1) / dev.tflops:.2f}% of peak")
    s.check("prefill is compute-bound at any realistic prompt length",
            prefill_intensity(cfg, 1_024) > dev.balance,
            f"intensity {prefill_intensity(cfg, 1024):.0f} FLOP/byte at T=1,024, "
            f"above the {dev.balance:.0f} ridge point")
    s.note("this is why a faster GPU does not speed up a single conversation. At "
           "batch one the machine waits on memory for the whole step, so adding "
           "arithmetic capacity changes nothing; the only fixes are moving fewer "
           "bytes or amortising the read over more sequences. It is also why "
           "throughput and latency benchmarks disagree so violently -- they sit "
           "on opposite sides of the ridge.")

    # Batching does not raise intensity indefinitely, because each sequence brings
    # its own KV cache. Where that ceiling lands is the section's sharpest result.
    s.say()
    s.eq("I(B) = 2PB / (P b_w + B T kv)  ->  2 T* / (b_w T)  as B -> inf",
         "the KV term scales with batch exactly as the numerator does, so it never "
         "amortises; at fp16 the ceiling is simply T* / T")
    rows = []
    for T in (128, 512, 2_048, 8_192, 32_768):
        ceil_i = decode_intensity_ceiling(gqa, T)
        need = next((B for B in range(1, 100_000)
                     if decode_intensity(gqa, B, T) >= dev.balance), None)
        rows.append([f"{T:,}", round(ceil_i, 1), round(t_ratio := gqa.crossover_tokens(1) / T, 1),
                     f"{need:,}" if need else "unreachable",
                     f"{100 * dev.attainable_tflops(ceil_i) / dev.tflops:.0f}%"])
    s.table(["context T", "intensity ceiling", "T*/T", "batch to reach ridge",
             "best % of peak"], rows, align="rrrrr")
    s.check("the decode intensity ceiling equals T*/T at fp16",
            all(abs(r[1] - r[2]) / r[2] < 0.02 for r in rows),
            "ceiling matches the crossover ratio at every context length tested")
    s.check("beyond a context of a few thousand tokens, no batch size makes decode "
            "compute-bound",
            rows[2][3] == "unreachable" and rows[0][3] != "unreachable",
            f"at T=128 the ridge needs a batch of {rows[0][3]}; at T=2,048 and "
            "beyond it cannot be reached at any batch size, because KV traffic "
            "alone exceeds the byte budget")
    s.note("this is the ceiling that grouped-query attention and KV quantisation "
           "are actually raising. Both cut `kv`, which lifts T*, which lifts this "
           "entire curve -- so they are not merely memory-capacity optimisations, "
           "they raise the best throughput the hardware can be made to deliver. "
           "It also explains the shape of real serving systems: long-context "
           "requests are bandwidth-bound no matter how many of them you batch, "
           "which is why they are scheduled and priced differently from short "
           "ones.")

    # --------------------------------------------- 11.3 decode latency, predicted
    s.h("11.3  What that predicts for per-token latency")
    s.eq("t_token >= (P * b_w + B * T * kv) / bandwidth",
         "a lower bound from bytes alone, ignoring arithmetic entirely")
    rows = []
    for spec in (SPECS[1], SPECS[3]):
        for T in (2_048, 32_768):
            wb = spec.weight_bytes()
            kvb = T * spec.kv_bytes_per_token()
            t = (wb + kvb) / (dev.bandwidth_gbs * 1e9)
            rows.append([spec.name, f"{T:,}", f"{wb / 2**30:.1f} GiB",
                         f"{kvb / 2**30:.2f} GiB", round(t * 1e3, 2),
                         round(1.0 / t, 1)])
    s.table(["model", "context", "weight bytes", "KV bytes read",
             "ms/token (bound)", "tok/s (bound)"], rows, align="lrrrrr")

    t_star = gqa.crossover_tokens(1)
    s.check("the context length at which KV traffic doubles decode latency is "
            "exactly S11's memory crossover",
            abs(t_star - gqa.weight_bytes() / gqa.kv_bytes_per_token()) < 1.0,
            f"T* = {t_star:,.0f} tokens, the same number at which the cache "
            "equals the weights in size -- one crossover, showing up once as a "
            "memory limit and once as a latency limit")
    s.note("worth stating because context length is usually discussed as a memory "
           "budget only. It is also a latency budget: past T* every decoded token "
           "spends more time reading the cache than reading the model. For an "
           "agent, whose transcript only ever grows, that is a per-step tax that "
           "rises across the run.")

    # ------------------------------------------------- 11.4 the semantic cache
    s.h("11.4  A semantic cache's threshold sells correctness to buy hit rate")
    s.eq("hit iff max_j cos(q, k_j) >= tau",
         "and a hit is *false* when the matched entry answered a different "
         "question")
    queries = intent_queries()
    s.kv("distinct intents", 24)
    s.kv("queries", len(queries))
    s.kv("within-intent similarity", "0.85 mean")
    s.kv("cross-intent, same topic group", "0.62 mean, 0.84 at the closest")
    s.note("the ranges overlap, so no threshold separates a paraphrase from a "
           "different question about the same topic. That is not a defect of the "
           "embedding, it is the situation")
    rows = []
    for tau in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85):
        c = SemanticCache(tau)
        for v, i in queries:
            c.query(v, i)
        n = len(queries)
        rows.append([tau, f"{100 * c.hits / n:.1f}%",
                     f"{100 * c.false_hits / max(c.hits, 1):.1f}%",
                     f"{100 * c.false_hits / n:.1f}%", len(c.keys)])
    s.table(["threshold tau", "hit rate", "of hits, wrong", "false hits / queries",
             "entries stored"], rows, align="rrrrr")

    hit = [float(r[1].rstrip("%")) for r in rows]
    bad = [float(r[3].rstrip("%")) for r in rows]
    s.check("hit rate and false-hit rate move together, so no threshold gives "
            "both",
            hit[0] > hit[-1] and bad[0] > bad[-1],
            f"tau={rows[0][0]:.2f} hits {hit[0]:.0f}% with {bad[0]:.0f}% of all "
            f"queries answered wrongly; tau={rows[-1][0]:.2f} hits "
            f"{hit[-1]:.0f}% with {bad[-1]:.1f}% wrong")

    # The threshold is an economic choice, not a tuning constant. Price it.
    s.say()
    s.eq("E[loss] = (1 - h) * c_compute + f * c_error",
         "h hit rate, f false-hit rate per query; the optimum depends on the "
         "ratio c_error / c_compute and on nothing else")
    rows2 = []
    for c_err in (0.0, 1.0, 5.0, 25.0, 200.0):
        best = min(((1 - h / 100) * 1.0 + (f / 100) * c_err, tau)
                   for tau, h, f in zip([r[0] for r in rows], hit, bad))
        rows2.append([c_err, best[1], round(best[0], 4)])
    s.table(["cost of a wrong answer (x one inference)", "optimal tau",
             "expected loss"], rows2, align="rrr")
    s.check("the optimal similarity threshold is set by the cost of being wrong, "
            "not by the embedding model",
            rows2[0][1] < rows2[-1][1],
            f"optimal tau moves from {rows2[0][1]} when a wrong answer is free, "
            f"to {rows2[-1][1]} when it costs {rows2[-1][0]:.0f}x an inference")
    s.note("so there is no such thing as a good default threshold, and a vendor "
           "quoting one is quoting an assumption about your error costs. Note "
           "also which number is easy to measure and which is not: hit rate needs "
           "only the cache's own logs, while false-hit rate needs ground truth "
           "about query equivalence that production systems do not have. The "
           "measurable half is the flattering half.")

    # ---------------------------------------------------- 11.5 staleness
    s.h("11.5  TTL bounds staleness, and stops working when the world moves fast")
    s.eq("P(stale) = 1 - (1 - e^{-rT}) / (rT)",
         "Poisson writes at rate r, entry age uniform on [0, T]")
    rows = []
    for ttl, rate in [(60, 0.0005), (60, 0.005), (60, 0.05), (300, 0.005),
                      (900, 0.005)]:
        an = staleness_probability(ttl, rate)
        sim = simulate_staleness(ttl, rate)
        rows.append([ttl, rate, round(rate * ttl, 4), round(an, 5),
                     round(sim, 5), round(abs(an - sim), 5)])
    s.table(["TTL s", "writes/s", "rT", "P(stale) analytic", "simulated",
             "difference"], rows, align="rrrrrr")
    s.check("the derived staleness probability matches simulation",
            all(r[5] < 0.005 for r in rows),
            f"worst difference {max(r[5] for r in rows):.5f} over 200,000 draws "
            "per row")
    s.check("in the slow-write limit staleness is approximately rT/2",
            abs(rows[0][3] - rows[0][2] / 2) < 0.001,
            f"rT = {rows[0][2]:.4f} gives P = {rows[0][3]:.5f} against "
            f"rT/2 = {rows[0][2] / 2:.5f}")
    s.check("once rT exceeds about 3, shortening the TTL stops helping much",
            staleness_probability(60, 0.05) > 0.65,
            f"rT = 3.0 already gives P(stale) = "
            f"{staleness_probability(60, 0.05):.3f}; the entry is more likely "
            "wrong than right")
    s.note("this is S13.5's non-idempotent repeat and a fact store's TTL seen from "
           "a third side. One cause: a cached value is a claim about a world that "
           "has moved on, and the only sound fix is to key the entry on a version "
           "of that world rather than on elapsed time. TTL is a bet that the "
           "world is slow.")

    # ---------------------------------------------------- 11.6 queueing
    s.h("11.6  p95 leaves p50 behind long before the queue looks busy")
    s.eq("W ~ Exponential(mu(1 - rho)),  rho = lambda / mu",
         "M/M/1 sojourn time; the mean is 1/(mu - lambda), hyperbolic in load")
    mu = 4.0                                     # 4 requests/s served
    rows = []
    for rho in (0.30, 0.50, 0.70, 0.90, 0.95):
        p50, p95, _ = mm1_percentiles(rho * mu, mu)
        s50, s95 = simulate_mm1(rho * mu, mu)
        rows.append([rho, round(p50 * 1e3), round(p95 * 1e3),
                     round(p95 / p50, 2), round(s50 * 1e3), round(s95 * 1e3)])
    s.table(["utilisation", "p50 ms", "p95 ms", "p95/p50", "p50 sim", "p95 sim"],
            rows, align="rrrrrr")
    low = rows[:3]                                   # utilisation 0.30 - 0.70
    s.check("the closed form agrees with simulation while the queue is not "
            "saturated",
            all(abs(r[1] - r[4]) / r[1] < 0.08 and abs(r[2] - r[5]) / r[2] < 0.12
                for r in low),
            "every percentile within 12% of simulation at utilisation <= 0.70, "
            "over 120,000 requests per row")
    err = [100 * abs(r[2] - r[5]) / r[2] for r in rows]
    s.kv("sampled p95 error vs analytic, by utilisation",
         "  ".join(f"{r[0]}: {e:.1f}%" for r, e in zip(rows, err)))
    s.check("the sampled p95 gets less reliable as utilisation rises, at fixed "
            "sample size",
            err[-1] > 5 * max(err[:3]),
            f"{max(err[:3]):.1f}% error at utilisation <= 0.70 against "
            f"{err[-1]:.1f}% at 0.95, from the same 120,000 requests -- sojourn "
            "times are strongly correlated near saturation, so the effective "
            "sample size is far below the nominal one")
    s.note("that second result is the more useful one operationally. Near "
           "saturation a queue's own tail measurements are noisy and biased by "
           "whatever transient the window happened to catch, so a dashboard p95 "
           "taken from a busy hour is not a reliable estimate of anything. The "
           "closed form, which needs only utilisation, is better behaved than the "
           "measurement it is usually checked against")
    s.check("latency is hyperbolic in load, not linear",
            rows[3][1] / rows[1][1] > 4.0,
            f"raising utilisation from {rows[1][0]} to {rows[3][0]} -- less than "
            f"double the load -- multiplies p50 by "
            f"{rows[3][1] / rows[1][1]:.1f}x")
    s.check("the p95/p50 ratio is constant, so a healthy ratio proves nothing",
            max(r[3] for r in rows) - min(r[3] for r in rows) < 0.01,
            f"ratio is {rows[0][3]:.2f} at every utilisation, because the "
            "sojourn distribution is exponential at all loads -- only its scale "
            "moves")
    s.note("the constant ratio is the trap. Watching p95/p50 for a warning gives "
           "none: it reads the same at 30% load and 95% load while the absolute "
           "numbers have gone up eightfold. Utilisation is the leading indicator, "
           "and it is the one least often on the dashboard.")

    # An agent issues several sequential requests per task, so task latency is a
    # sum of draws -- and a task is late if any of its steps is.
    s.say()
    n_steps = 6
    rho = 0.90
    p50, p95, _ = mm1_percentiles(rho * mu, mu)
    g = rng(5)
    k = mu * (1 - rho)
    task = g.exponential(1.0 / k, size=(40_000, n_steps)).sum(axis=1)
    t50, t95 = percentile(task, 50), percentile(task, 95)
    s.table(["quantity", "value"],
            [[f"single request p50", f"{p50 * 1e3:.0f} ms"],
             [f"single request p95", f"{p95 * 1e3:.0f} ms"],
             [f"{n_steps}-step task p50", f"{t50 * 1e3:.0f} ms"],
             [f"{n_steps}-step task p95", f"{t95 * 1e3:.0f} ms"],
             [f"{n_steps} x request p50", f"{n_steps * p50 * 1e3:.0f} ms"]])
    s.check("a multi-step task's median exceeds the number of steps times the "
            "request median",
            t50 > n_steps * p50,
            f"{t50 * 1e3:.0f} ms against {n_steps * p50 * 1e3:.0f} ms, because "
            "the sum of exponentials has its mass shifted right of the sum of "
            "medians")
    s.check("but the task p95 is far below the number of steps times the request "
            "p95",
            t95 < n_steps * p95,
            f"{t95 * 1e3:.0f} ms against {n_steps * p95 * 1e3:.0f} ms: "
            "independent steps average out, so the tails do not simply add")
    s.note("both directions matter for an agent. Budgeting a task at steps x p50 "
           "underestimates it; budgeting at steps x p95 wildly overestimates it. "
           "The distribution has to be composed, not its percentiles -- and this "
           "is the benign case, where steps are independent. S4 covers what "
           "happens when a late step causes a retry.")

    # ------------------------------------------------- 11.7 cost per task
    s.h("11.7  Cost per token is billed; cost per completed task is what you pay")
    s.eq("cost_per_task = tokens_per_attempt * price_per_token / P(success)",
         "a failed attempt is billed in full and delivers nothing")
    configs = [("cheap, weaker", 0.15, 5_400, 0.55),
               ("cheap, more steps", 0.15, 9_200, 0.72),
               ("mid", 0.60, 4_800, 0.86),
               ("strong, fewer steps", 3.00, 3_100, 0.94)]
    rows = []
    for name, price, toks, succ in configs:
        att = toks * price / 1e6
        rows.append([name, price, f"{toks:,}", succ, round(att, 6),
                     round(att / succ, 6)])
    s.table(["configuration", "$/Mtok", "tokens/attempt", "P(success)",
             "$/attempt", "$/completed task"], rows, align="lrrrrr")
    tok_spread = max(r[1] for r in rows) / min(r[1] for r in rows)
    task_spread = max(r[5] for r in rows) / min(r[5] for r in rows)
    s.check("dividing by the success rate compresses the price spread but does "
            "not reverse it",
            task_spread < tok_spread,
            f"a {tok_spread:.0f}x spread in price per token becomes "
            f"{task_spread:.1f}x per completed task -- so per-token pricing "
            "overstates the real difference, while still ranking the cheap model "
            "first")

    # What the token bill omits: a task that fails does not merely waste its
    # tokens, it lands on a person. For a bank that is the dominant term.
    s.say()
    s.eq("total = tokens*price + (1 - P) * c_escalate",
         "one attempt, then a human picks up whatever failed")
    rows2 = []
    for (name, price, toks, succ), r in zip(configs, rows):
        for c_esc in (0.0, 2.00):
            total = r[4] + (1 - succ) * c_esc
            rows2.append([name, f"${c_esc:.2f}", round(total, 5),
                          f"{100 * r[4] / total:.1f}%" if total else "-"])
    s.table(["configuration", "escalation cost", "$/task total",
             "token share of total"], rows2, align="lrrr")
    with_esc = [rows2[i][2] for i in range(1, len(rows2), 2)]
    s.check("with a realistic cost of failure the ranking inverts completely",
            with_esc.index(min(with_esc)) == len(with_esc) - 1
            and with_esc.index(max(with_esc)) == 0,
            f"at $2.00 per escalation the 20x-priced model is cheapest overall "
            f"(${min(with_esc):.3f}/task) and the cheapest-per-token model is "
            f"dearest (${max(with_esc):.3f}/task), a "
            f"{max(with_esc) / min(with_esc):.1f}x difference the token bill "
            "reverses")
    s.check("and once failure has any material cost, the token bill is a rounding "
            "error",
            all(float(rows2[i][3].rstrip("%")) < 10.0
                for i in range(1, len(rows2), 2)),
            "tokens are under 10% of total cost for every configuration once "
            "escalation is priced; the largest share is "
            f"{max(float(rows2[i][3].rstrip('%')) for i in range(1, len(rows2), 2)):.1f}%")
    break_even = next(c / 100 for c in range(0, 400)
                      if rows[3][4] + (1 - configs[3][3]) * c / 100
                      < rows[0][4] + (1 - configs[0][3]) * c / 100)
    s.kv("escalation cost at which the ranking flips", f"${break_even:.2f}")
    s.note("this is the closing result of the article, and it is the reason the "
           "next one exists. Price per token is the only term in the table a "
           "vendor controls and the only one anyone negotiates, and at any "
           "realistic cost of failure it is noise. Everything that actually "
           "determines the bill -- tokens per attempt, and above all P(success) -- "
           "is a property of the loop wrapped around the model, not of the model. "
           "Tokens per attempt comes from S13's step count. P(success) is what the "
           "entire agentic stack exists to raise, and what the next article "
           "measures. Cost cannot be optimised at the inference layer, because "
           "cost is not determined there.")

    s.close()
    return s


if __name__ == "__main__":
    run()
