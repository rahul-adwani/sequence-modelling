"""S10. Decoding: from a distribution to a token.

S9 established that a language model is a map `f: V^* -> Delta(V)`. Note what
that says and what it does not. The output is a probability distribution over
the whole vocabulary. It is never a token. Something else has to choose one, and
that something else is not part of the model, is not trained, and has its own
parameters, its own failure modes and its own arithmetic.

The choice splits into two questions that are usually run together.

**What do you do to the vector?** Temperature rescales it, top-k and top-p
delete part of it. Both are edits to a distribution, and the first result here is
that neither of them can change which token is most probable. Turning the
temperature down does not make a model pick a different action. It makes it pick
the same action more often.

**Which sequence are you trying to produce?** That question is a search, because
the most probable next token repeated T times is not the most probable sequence
of T tokens. The search space is `|V|^T`, exact decoding of a transformer is
hopeless for the reason spelled out in 3.6, and every production decoder is an
approximation to an objective nobody can compute.

Then the part that is genuinely surprising, in 3.7: the objective itself is
wrong. The most probable continuation of a finite-state process is eventually a
repeated cycle. That is not a pathology of neural networks and not a bug in beam
search. It is what maximising probability over a graph with finitely many states
does, and it is why the decoders that produce readable text are the ones that
deliberately do not maximise probability. Holtzman et al. (arXiv 1904.09751)
named the phenomenon; the object underneath it is a cycle in a finite graph.

The section closes on speculative decoding, which is the one entry here that
costs nothing at all: a cheap draft model proposes tokens and the real model
verifies them in a single pass, and the accept-or-resample rule is arranged so
that the tokens coming out are distributed exactly as the real model's own
samples would be. Not approximately. Exactly, and the run checks it to machine
precision. Leviathan et al. (arXiv 2211.17192) and Chen et al. (arXiv 2302.01318)
introduced it independently.

The model is the one S9 fitted, and for the same reason: the true distribution
of the process is available, so a decoder's output can be compared against
something other than taste.
"""
from __future__ import annotations

import itertools

import numpy as np

from .harness import Section, entropy_bits, kl_bits, rng, softmax
from .s09_llm_as_function import V, VOCAB, fitted, words

TINY = np.finfo(np.float64).tiny


# ------------------------------------------------------------------- truncation

def top_k(p: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest probabilities, renormalise, zero the rest."""
    q = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    q[idx] = p[idx]
    return q / q.sum()


def top_p(p: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """Nucleus sampling: the smallest prefix of the sorted distribution whose
    mass reaches `threshold`. Returns the renormalised distribution and the
    nucleus size, because that size is the interesting quantity -- it is how
    many options the model was actually considering at that position."""
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    n = int(np.searchsorted(csum, threshold) + 1)
    q = np.zeros_like(p)
    q[order[:n]] = p[order[:n]]
    return q / q.sum(), n


def temper(p: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature to a distribution rather than to logits.

    Identical to the usual `softmax(z/T)`, because the logits of this model are
    `log p` up to an additive constant and softmax removes additive constants.
    Written this way the operation is visibly what it is: raise every
    probability to the power 1/T and renormalise.
    """
    return softmax(np.log(np.maximum(p, TINY)), temperature=T, axis=-1)


# ---------------------------------------------------------------------- search

def brute_force_best(logP: np.ndarray, steps: int, u0: int,
                     v0: int) -> tuple[list[int], float]:
    """Exhaustive maximisation of log p over all V^steps continuations.

    Only used to validate the dynamic program below. At V=24 and 3 steps this
    enumerates 13,824 sequences; at 6 steps it would be 191 million, and at the
    vocabulary of a real model even 3 steps is about 10^15. That gap is the whole
    reason production decoding is approximate.
    """
    best, best_lp = None, -np.inf
    for cand in itertools.product(range(V), repeat=steps):
        u, v, lp = u0, v0, 0.0
        for w in cand:
            lp += float(logP[u, v, w])
            u, v = v, w
        if lp > best_lp:
            best, best_lp = list(cand), lp
    return best, best_lp


def exact_values(logP: np.ndarray, steps: int) -> np.ndarray:
    """Best achievable log2 p over `steps` tokens, from every start context.

    The trigram's state is the last two tokens, so the search is over a graph
    with V^2 nodes rather than a tree with V^steps leaves, and Bellman's equation
    applies:

        E_k(u,v) = max_w [ log p(w|u,v) + E_{k-1}(v,w) ],    E_0 = 0

    Cost is O(steps * V^3) instead of O(V^steps). This is exactly why exact
    decoding is tractable for an n-gram and hopeless for a transformer: the
    transformer's state is the entire prefix, so no two paths ever merge and
    there is no dynamic program to write.
    """
    E = np.zeros((V, V))
    for _ in range(steps):
        E = np.max(logP + E[None, :, :], axis=2)
    return E


def greedy_values(logP: np.ndarray, steps: int) -> np.ndarray:
    """log2 p of the greedy path from every start context.

    Same recursion with the max replaced by the model's own argmax, which is the
    definition of greedy: commit to the locally best token, then live with the
    state it puts you in.
    """
    wstar = np.argmax(logP, axis=2)                     # (V, V)
    step_lp = np.take_along_axis(logP, wstar[:, :, None], axis=2)[:, :, 0]
    v_of = np.tile(np.arange(V)[None, :], (V, 1))       # v index at [u, v]
    G = np.zeros((V, V))
    for _ in range(steps):
        G = step_lp + G[v_of, wstar]
    return G


def beam_value(logP: np.ndarray, steps: int, u0: int, v0: int,
               width: int) -> float:
    """log2 p of the best sequence beam search finds at the given width.

    Width 1 is greedy and width V^steps is exhaustive, so the beam is the dial
    between the two. Note what is being kept: the top `width` *prefixes* by
    cumulative log-probability, which is why beam search can recover a path
    whose first token was not the locally best one -- and why it still cannot
    guarantee the optimum, since a prefix that looks weak now may have been the
    only route to the best completion.
    """
    beam = [(0.0, u0, v0)]
    for _ in range(steps):
        cand = []
        for lp, u, v in beam:
            cand.extend((lp + float(logP[u, v, w]), v, w) for w in range(V))
        cand.sort(key=lambda t: -t[0])
        beam = cand[:width]
    return beam[0][0]


def stationary_policy(logP: np.ndarray, sweeps: int = 600, tau: float = 0.5
                      ) -> tuple[np.ndarray, int]:
    """The argmax policy of the infinite-horizon problem, by value iteration.

        E <- (1 - tau) E + tau * max_w [ log p(w|u,v) + E(v,w) ],  then E -= E(0,0)

    Two adjustments, both load-bearing. Subtracting a constant every sweep is
    relative value iteration: without it E grows without bound at the optimal
    rate per step, and the constant cancels inside the argmax so removing it
    changes nothing being computed. Mixing the old value in with weight
    `1 - tau` is the aperiodicity transformation, and it is needed precisely
    because of what 3.7 is about: the optimal policy here cycles, and undamped
    value iteration on a periodic chain oscillates forever instead of settling.
    The transformation leaves the optimal policy unchanged.

    Returns the policy and the sweep at which it last changed. The policy is a
    map from each of the V^2 states to one token, which makes it a function from
    a finite set into itself. That is the whole of 3.7: every trajectory of such
    a function must revisit a state, and from there it repeats forever.
    """
    E = np.zeros((V, V))
    policy, last_change = np.argmax(logP, axis=2), 0
    for k in range(1, sweeps + 1):
        Q = logP + E[None, :, :]
        new = np.argmax(Q, axis=2)
        E = (1.0 - tau) * E + tau * np.max(Q, axis=2)
        E = E - E[0, 0]
        if not np.array_equal(new, policy):
            last_change = k
        policy = new
    return policy, last_change


def policy_cycle(policy: np.ndarray, u0: int, v0: int
                 ) -> tuple[int, int, set]:
    """Where a deterministic policy starts repeating, and on what.

    Returns the step at which a state is first revisited, the period of the
    cycle entered, and the set of states on that cycle. A repeat is guaranteed
    within V^2 + 1 steps by the pigeonhole principle, so no step limit is
    needed as an argument.
    """
    seen: dict[tuple[int, int], int] = {}
    order: list[tuple[int, int]] = []
    u, v = u0, v0
    for t in range(V * V + 1):
        if (u, v) in seen:
            entry = seen[(u, v)]
            return entry, t - entry, set(order[entry:])
        seen[(u, v)] = t
        order.append((u, v))
        u, v = v, int(policy[u, v])
    raise AssertionError("a deterministic policy on V^2 states must repeat")


def policy_tokens(policy: np.ndarray, u0: int, v0: int,
                  steps: int) -> list[int]:
    """Emit exactly `steps` tokens by following a deterministic policy."""
    out, u, v = [], u0, v0
    for _ in range(steps):
        w = int(policy[u, v])
        out.append(w)
        u, v = v, w
    return out


def distinct_bigrams(seq: list[int]) -> int:
    return len({pair for pair in zip(seq, seq[1:])})


def _states(u0: int, v0: int, seq: list[int]):
    """Walk a sequence, yielding the (context, token) triples it passes through."""
    u, v = u0, v0
    for w in seq:
        yield u, v, w
        u, v = v, w


# -------------------------------------------------------------------- sampling

def sample_path(cum: np.ndarray, u0: int, v0: int, steps: int,
                draws: np.ndarray) -> list[int]:
    """Sample a continuation by inverse transform on a precomputed CDF.

    `np.random.Generator.choice` with a `p` argument rebuilds the cumulative
    distribution on every call, which is the dominant cost when the same 576
    distributions are sampled tens of thousands of times. The CDFs are built
    once and the uniforms are drawn in one block.
    """
    out, u, v = [], u0, v0
    for t in range(steps):
        w = int(np.searchsorted(cum[u, v], draws[t]))
        w = min(w, V - 1)
        out.append(w)
        u, v = v, w
    return out


def repeated_bigram_rate(seq: list[int]) -> float:
    """Fraction of adjacent token pairs in a sequence that occurred earlier.

    A blunt instrument, and the right one: it needs no reference text and it
    goes up for exactly the failure being measured.
    """
    if len(seq) < 3:
        return 0.0
    pairs = list(zip(seq, seq[1:]))
    seen, repeats = set(), 0
    for p in pairs:
        if p in seen:
            repeats += 1
        seen.add(p)
    return repeats / len(pairs)


def stationary_over_states(P: np.ndarray, iters: int = 400) -> np.ndarray:
    """Stationary distribution over the V^2 contexts of an order-2 chain.

    Power iteration on the state graph rather than a long simulated sample: the
    answer is the same object and it is exact to the iteration tolerance instead
    of to Monte Carlo error.
    """
    M = np.zeros((V * V, V * V))
    for u in range(V):
        for v in range(V):
            M[u * V + v, v * V:(v + 1) * V] = P[u, v]
    pi = np.full(V * V, 1.0 / (V * V))
    for _ in range(iters):
        pi = pi @ M
    return (pi / pi.sum()).reshape(V, V)


# --------------------------------------------------------- speculative decoding

def verified_distribution(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """What speculative decoding actually emits, computed rather than sampled.

    One drafted token x ~ q is accepted with probability min(1, p(x)/q(x)), and
    on rejection a token is drawn from the normalised positive part of p - q.
    The probability of emitting x is therefore

        q(x) * min(1, p(x)/q(x))  +  (1 - alpha) * max(p(x)-q(x), 0) / Z
            = min(p(x), q(x))     +  max(p(x) - q(x), 0)
            = p(x)

    because Z, the total residual mass, is itself exactly 1 - alpha. The
    identity is algebraic, so it holds for any draft whatsoever, including a bad
    one: a poor draft costs speed and cannot cost quality. That is the property
    that makes the technique usable, and it is the one most often described as
    "approximately the same output".
    """
    accept = np.minimum(P, Q)
    alpha = accept.sum(axis=-1, keepdims=True)
    residual = np.maximum(P - Q, 0.0)
    z = np.maximum(residual.sum(axis=-1, keepdims=True), TINY)
    return accept + (1.0 - alpha) * residual / z


def speculative_run(P: np.ndarray, Q: np.ndarray, cumP: np.ndarray,
                    cumQ: np.ndarray, draft_len: int, blocks: int,
                    seed: int) -> tuple[float, float, float]:
    """Simulate the loop: tokens per verification, accept rate, standard error.

    The standard error is returned rather than left implicit because the effect
    being measured at long draft lengths is smaller than the noise on a run of
    this size. Without it the measured column looks non-monotone and reads as a
    bug; with it, the exact per-context column is visibly the one to trust past
    the point where the two overlap.

    One block is one call to the target model. The draft proposes `draft_len`
    tokens on its own; the target scores all of them in a single pass, which is
    the whole point and is why this is a saving at all. S12 supplies the reason
    that pass costs about what scoring one token costs: single-stream decoding
    is bound by reading the weights out of memory, not by arithmetic, so the
    arithmetic for a handful of extra positions is close to free.
    """
    g = rng(seed)
    u, v = 0, 1
    produced = accepted = proposed = 0
    squares = 0
    for _ in range(blocks):
        state = (u, v)
        drafted, states = [], []
        du, dv = u, v
        for _ in range(draft_len):
            x = int(np.searchsorted(cumQ[du, dv], g.random()))
            x = min(x, V - 1)
            states.append((du, dv))
            drafted.append(x)
            du, dv = dv, x

        u, v = state
        emitted = 0
        for (su, sv), x in zip(states, drafted):
            proposed += 1
            p_x, q_x = P[su, sv, x], Q[su, sv, x]
            if g.random() < min(1.0, p_x / max(q_x, TINY)):
                accepted += 1
                emitted += 1
                u, v = sv, x
            else:
                residual = np.maximum(P[su, sv] - Q[su, sv], 0.0)
                total = max(float(residual.sum()), TINY)
                y = int(np.searchsorted(np.cumsum(residual / total), g.random()))
                y = min(y, V - 1)
                emitted += 1
                u, v = sv, y
                break
        else:
            # Every drafted token survived, so the target's own pass supplies
            # one more token for free. This is where the k+1 in the throughput
            # formula comes from.
            w = int(np.searchsorted(cumP[u, v], g.random()))
            w = min(w, V - 1)
            emitted += 1
            u, v = v, w
        produced += emitted
        squares += emitted * emitted
    mean = produced / blocks
    variance = max(squares / blocks - mean * mean, 0.0)
    return mean, accepted / max(proposed, 1), (variance / blocks) ** 0.5


def constant_rate_tokens(alpha: float, draft_len: int) -> float:
    """Expected tokens per verification if every position accepted at rate alpha.

        E = 1 + alpha + alpha^2 + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)

    The textbook figure. It is a geometric series, so its ceiling as the draft
    grows is 1/(1 - alpha) and nothing longer helps.
    """
    return float(sum(alpha ** j for j in range(draft_len + 1)))


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s10", "Decoding: from a distribution to a token",
                "two edits to one vector, a search over sequences, and what "
                "maximising probability actually returns").open()

    f = fitted()
    src, model = f["src"], f["model"]
    P = model.dist_tensor()
    logP = np.log2(np.maximum(P, TINY))
    n_ctx = V * V

    # ------------------------------------------------- 3.1 the vector, in bulk
    s.h("3.1  The model hands over a vector, not a token")
    s.eq("decode : Delta(V) -> V",
         "a separate function with its own parameters, applied to the output of "
         "f; nothing in the model constrains it and nothing in training saw it")

    ent = np.array([[entropy_bits(P[u, v]) for v in range(V)] for u in range(V)])
    top1 = P.max(axis=2)
    nuc99 = np.array([[top_p(P[u, v], 0.99)[1] for v in range(V)]
                      for u in range(V)])
    s.kv("contexts measured", n_ctx, "(every state of the model)")
    s.table(["quantity", "min", "median", "max"],
            [["entropy of p (bits)", round(float(ent.min()), 4),
              round(float(np.median(ent)), 4), round(float(ent.max()), 4)],
             ["probability of the top token", round(float(top1.min()), 4),
              round(float(np.median(top1)), 4), round(float(top1.max()), 4)],
             ["tokens holding 99% of the mass", int(nuc99.min()),
              int(np.median(nuc99)), int(nuc99.max())]], align="lrrr")
    s.check("how much the model is actually choosing between varies by more "
            "than a bit across its own contexts",
            float(ent.max() - ent.min()) > 1.0,
            f"entropy runs from {ent.min():.3f} to {ent.max():.3f} bits, a "
            f"spread of {ent.max() - ent.min():.3f} bits and so more than a "
            f"doubling of the effective number of options")
    s.note("this spread is the reason a decoder set once and applied everywhere "
           "is a compromise: the same setting meets a nearly certain "
           "continuation and a nearly open one, and it cannot be right for both")

    # -------------------------------------------------------- 3.2 temperature
    s.h("3.2  Temperature reweights, and never reorders")
    s.eq("p_i(T) = exp(z_i / T) / sum_j exp(z_j / T) = p_i^(1/T) / sum_j p_j^(1/T)",
         "T < 1 sharpens, T > 1 flattens; the second form makes the invariance "
         "obvious, since x -> x^(1/T) is increasing for every T > 0")

    u, v = VOCAB.index("the"), VOCAB.index("model")
    temps = (0.25, 0.5, 1.0, 2.0, 8.0)
    rows = []
    for T in temps:
        q = temper(P[u, v], T)
        rows.append([T, round(entropy_bits(q), 4), round(float(q.max()), 4),
                     int((q > 0.01).sum()), VOCAB[int(np.argmax(q))]])
    s.say(f"  context = ({VOCAB[u]!r}, {VOCAB[v]!r})")
    s.table(["T", "H bits", "max p", "tokens p>0.01", "argmax"], rows,
            align="rrrrl")

    base_order = np.argsort(-P, axis=2)
    order_kept, ent_rising, top_falling = True, True, True
    ent_prev, top_prev = None, None
    for T in temps:
        Q = temper(P, T)
        order_kept &= bool(np.array_equal(np.argsort(-Q, axis=2), base_order))
        e = np.array([[entropy_bits(Q[a, b]) for b in range(V)]
                      for a in range(V)])
        mx = Q.max(axis=2)
        if ent_prev is not None:
            ent_rising &= bool(np.all(e > ent_prev))
            top_falling &= bool(np.all(mx < top_prev))
        ent_prev, top_prev = e, mx

    s.check("temperature preserves the entire ranking of the vocabulary, not "
            "merely the top token",
            order_kept,
            f"the full argsort of all {V} tokens is identical at every "
            f"temperature in {temps}, at all {n_ctx} contexts")
    s.check("entropy rises and the top token's share falls monotonically with "
            "temperature, at every context",
            ent_rising and top_falling,
            f"checked pairwise across {len(temps)} temperatures at "
            f"{n_ctx} contexts")

    flat = temper(P, 64.0)
    uniform = np.full(V, 1.0 / V)
    far = max(kl_bits(flat[a, b], uniform) for a in range(V) for b in range(V))
    s.check("high temperature approaches the uniform distribution",
            far < 0.05,
            f"worst divergence from uniform at T=64 is {far:.4f} bits, over "
            f"all {n_ctx} contexts")
    s.note("this is the load-bearing negative result of the section. "
           "Temperature is a confidence dial, not a correctness dial. If the "
           "model ranks the wrong token first, no temperature reorders it, and "
           "the only thing lowering the temperature achieves is that the wrong "
           "token is chosen more consistently")

    # -------------------------------------------------------------- 3.3 top-k
    s.h("3.3  Top-k is a fixed count against a distribution that is not fixed")
    Psort = -np.sort(-P, axis=2)
    cum_sorted = np.cumsum(Psort, axis=2)
    rows, spreads = [], {}
    for k in (1, 2, 3, 5, 10):
        mass = cum_sorted[:, :, k - 1]
        spreads[k] = float(mass.max() - mass.min())
        rows.append([k, round(float(mass.min()), 4), round(float(mass.mean()), 4),
                     round(float(mass.max()), 4), round(spreads[k], 4)])
    s.table(["k", "least mass kept", "mean", "most mass kept", "spread"],
            rows, align="rrrrr")
    s.check("a fixed k keeps very different amounts of the distribution "
            "depending on the context",
            spreads[3] > 0.2,
            f"at k=3 the retained mass runs from {cum_sorted[:, :, 2].min():.3f} "
            f"to {cum_sorted[:, :, 2].max():.3f}, a spread of {spreads[3]:.3f}")

    # How much of what the process actually does gets deleted, measured against
    # the true conditionals rather than the model's own.
    lost = []
    for a in range(V):
        for b in range(V):
            keep = np.argsort(-P[a, b])[:3]
            true = src.true_dist(a, b)
            lost.append(1.0 - float(true[keep].sum()))
    lost = np.asarray(lost)
    total_miss = int((lost > 0.999).sum())
    s.kv("true mass deleted by k=3, mean", round(float(lost.mean()), 4))
    s.kv("true mass deleted by k=3, worst context", round(float(lost.max()), 4))
    s.kv("contexts where the top 3 contains no reachable token",
         f"{total_miss} of {n_ctx}")
    s.check("top-k deletes outcomes the process genuinely produces",
            float(lost.mean()) > 0.05,
            f"on average {100 * lost.mean():.1f}% of the true next-token mass "
            f"falls outside the model's top 3, rising to "
            f"{100 * lost.max():.1f}% at the worst context")
    s.note("the source here has five reachable tokens per context, so k=3 is "
           "guaranteed to cut two of them. The bound is a property of this "
           "demonstration, not of language, but the shape of the problem is "
           "general: k is chosen once and the number of plausible "
           "continuations is not constant")
    s.note("the worst context is worth reading carefully rather than as an "
           "error. Deleting all of the true mass means the model ranked three "
           "unreachable tokens above every reachable one, which happens at "
           "contexts it saw too few times to estimate and where it has fallen "
           "back on the lower orders. Truncation does not cause that failure, "
           "it makes it irreversible: sampling from the full distribution "
           "would still have reached a real continuation sometimes")

    # -------------------------------------------------------------- 3.4 top-p
    s.h("3.4  Top-p sizes the shortlist from the distribution itself")
    rows, sizes = [], []
    for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
        n_arr = np.zeros((V, V), dtype=np.int64)
        divs = []
        for a in range(V):
            for b in range(V):
                q, n = top_p(P[a, b], thr)
                n_arr[a, b] = n
                divs.append(kl_bits(q, P[a, b]))
        sizes.append(n_arr)
        rows.append([thr, int(n_arr.min()), round(float(n_arr.mean()), 3),
                     int(n_arr.max()), round(float(np.mean(divs)), 4)])
    s.table(["top-p", "smallest nucleus", "mean", "largest nucleus",
             "mean D_KL(kept || full) bits"], rows, align="rrrrr")
    s.check("the nucleus grows with the mass threshold at every context",
            all(bool(np.all(sizes[i] <= sizes[i + 1]))
                for i in range(len(sizes) - 1)),
            "checked at all "
            f"{n_ctx} contexts across five thresholds, not on the averages")

    nuc90 = sizes[2]
    buckets = [b for b in sorted(set(nuc90.ravel().tolist()))
               if int((nuc90 == b).sum()) >= 10]
    bucket_means = [float(ent[nuc90 == b].mean()) for b in buckets]
    s.table(["nucleus size at p=0.9", "contexts", "mean entropy (bits)"],
            [[b, int((nuc90 == b).sum()), round(m, 4)]
             for b, m in zip(buckets, bucket_means)], align="rrr")
    s.check("the nucleus size tracks the model's uncertainty rather than being "
            "set by hand",
            all(bucket_means[i] < bucket_means[i + 1]
                for i in range(len(bucket_means) - 1)),
            "mean entropy rises with every nucleus size holding at least ten "
            "contexts: " + " < ".join(f"{m:.3f}" for m in bucket_means))
    s.note("that is the entire argument for nucleus over top-k, and it is worth "
           "stating as what it is: top-p is an adaptive k, chosen per position "
           "from the distribution the model just produced")

    # ------------------------------------------------- 3.5 the dials interact
    s.h("3.5  The two dials interact, and their order is a third parameter")
    differ = same_argmax = 0
    for a in range(V):
        for b in range(V):
            p = P[a, b]
            first = np.nonzero(top_p(temper(p, 1.5), 0.9)[0])[0]
            second = np.nonzero(top_p(p, 0.9)[0])[0]
            differ += int(not np.array_equal(np.sort(first), np.sort(second)))
            same_argmax += int(int(np.argmax(temper(p, 1.5))) == int(np.argmax(p))
                               and int(np.argmax(top_k(p, 3))) == int(np.argmax(p))
                               and int(np.argmax(top_p(p, 0.5)[0]))
                               == int(np.argmax(p)))
    s.kv("contexts where the two orders admit different tokens",
         f"{differ} of {n_ctx}")
    s.check("temperature before truncation and truncation before temperature "
            "are different decoders",
            differ > 0.4 * n_ctx,
            f"at {100 * differ / n_ctx:.1f}% of contexts the surviving token "
            f"set differs, so 'temperature 1.5, top-p 0.9' does not specify a "
            f"decoder without saying which happens first")
    s.check("no setting of either dial can change what greedy decoding returns",
            same_argmax == n_ctx,
            f"the argmax is unchanged at all {n_ctx} contexts under "
            f"temperature 1.5, top-k 3 and top-p 0.5")
    s.note("both facts are consequences of the same thing. These operations are "
           "monotone in the probabilities, so they move mass without moving "
           "rank. They change what sampling does. They cannot change what "
           "argmax does")

    # ------------------------------------------------------------ 3.6 search
    s.h("3.6  Decoding is search, and greedy is not optimal")
    s.eq("argmax_{x_1..x_T} prod_t p(x_t|x_<t)  !=  the sequence of per-step argmaxes",
         "the product of locally best choices is not the best product")

    u0, v0 = 0, 1
    bf_seq, bf_lp = brute_force_best(logP, 3, u0, v0)
    dp_lp = float(exact_values(logP, 3)[u0, v0])
    s.check("the dynamic program agrees with exhaustive search over 13,824 "
            "sequences",
            abs(dp_lp - bf_lp) < 1e-12,
            f"DP {dp_lp:.10f} vs brute force {bf_lp:.10f} bits "
            f"({words(bf_seq)})")

    s.say()
    s.say("  Regret is how many bits the greedy path gives up against the exact "
          "optimum,\n  evaluated at every one of the 576 start contexts rather "
          "than a chosen one.")
    rows, frac_hist = [], []
    for T in (2, 3, 5, 8, 12):
        E, G = exact_values(logP, T), greedy_values(logP, T)
        regret = E - G
        frac = float(np.mean(regret > 1e-9))
        frac_hist.append(frac)
        rows.append([T, round(frac * 100, 1), round(float(regret.mean()), 4),
                     round(float(regret.max()), 4),
                     round(float(2 ** regret.max()), 2)])
    s.table(["horizon T", "% contexts greedy loses", "mean regret bits",
             "max regret bits", "max prob. ratio"], rows, align="rrrrr")
    s.check("greedy decoding is suboptimal at some start contexts",
            frac_hist[1] > 0,
            f"at T=3 greedy loses at {frac_hist[1] * 100:.1f}% of the 576 "
            f"contexts; at T=12, {frac_hist[-1] * 100:.1f}%")
    s.check("the greedy shortfall grows with the horizon",
            rows[0][2] < rows[-1][2],
            f"mean regret {rows[0][2]:.4f} bits at T=2 rising to "
            f"{rows[-1][2]:.4f} bits at T=12")
    s.note("greedy is often optimal and sometimes badly wrong, and which one you "
           "get depends on the context, not on the decoder. That is why a single "
           "well-chosen example proves nothing here and the sweep over all 576 "
           "start states does")

    T = 5
    regret = exact_values(logP, T) - greedy_values(logP, T)
    uw, vw = np.unravel_index(int(np.argmax(regret)), regret.shape)
    opt = float(exact_values(logP, T)[uw, vw])
    s.say()
    s.say(f"  worst context at T={T}: ({VOCAB[uw]!r}, {VOCAB[vw]!r}), "
          f"optimum {opt:.4f} bits")
    rows = []
    for width in (1, 2, 4, 8, 16, 32):
        bv = beam_value(logP, T, int(uw), int(vw), width)
        rows.append([width, round(bv, 4), round(max(opt - bv, 0.0), 4),
                     "yes" if opt - bv < 1e-9 else "no"])
    s.table(["beam width", "log2 p found", "regret bits", "optimal?"], rows,
            align="rrrl")
    gaps = [r[2] for r in rows]
    s.check("widening the beam never increases regret and eventually removes it",
            all(gaps[i] >= gaps[i + 1] - 1e-12 for i in range(len(gaps) - 1))
            and gaps[-1] < 1e-9,
            f"regret {gaps[0]:.4f} bits at width 1 falling to {gaps[-1]:.1e} "
            f"at width {rows[-1][0]}")
    s.note("the greedy path takes a locally strong token that leads into a worse "
           "continuation, and the beam recovers it by refusing to commit. "
           "Exhaustive search is V^T -- 13,824 here, around 10^15 for three "
           "tokens of a real vocabulary -- so production decoding is always an "
           "approximation to an objective nobody can compute")

    # -------------------------------------------- 3.7 what the optimum returns
    s.h("3.7  What maximising probability actually returns")
    policy, last_change = stationary_policy(logP)
    s.kv("value iteration sweeps run", 600)
    s.kv("last sweep at which the policy changed", last_change)
    s.check("value iteration settles on a single policy rather than oscillating",
            last_change < 600,
            f"the argmax policy over all {n_ctx} states was unchanged for the "
            f"final {600 - last_change} sweeps")
    s.note("the optimal policy is a map from each of the 576 states to one "
           "token. A map from a finite set into itself, followed forever, has "
           "to revisit a state; from that point it repeats. Degenerate "
           "repetition under likelihood-maximising decoding is not a quirk of "
           "neural networks, it is the pigeonhole principle")

    entries, periods, cyclic = [], [], set()
    argmax_bigrams, sampled_bigrams = [], []
    argmax_rep, sampled_rep = [], []
    cum = np.cumsum(P, axis=2)
    g = rng(909)
    horizon = 40
    for a in range(V):
        for b in range(V):
            entry, period, on_cycle = policy_cycle(policy, a, b)
            entries.append(entry)
            periods.append(period)
            cyclic |= on_cycle

            best = policy_tokens(policy, a, b, horizon)
            argmax_bigrams.append(distinct_bigrams(best))
            argmax_rep.append(repeated_bigram_rate(best))

            seq = sample_path(cum, a, b, horizon, g.random(horizon))
            sampled_bigrams.append(distinct_bigrams(seq))
            sampled_rep.append(repeated_bigram_rate(seq))

    s.table(["quantity", "min", "mean", "max"],
            [["steps before a state repeats", int(np.min(entries)),
              round(float(np.mean(entries)), 2), int(np.max(entries))],
             ["period of the cycle entered", int(np.min(periods)),
              round(float(np.mean(periods)), 2), int(np.max(periods))]],
            align="lrrr")
    s.kv("distinct contexts lying on any cycle", f"{len(cyclic)} of {n_ctx}")
    s.check("the most probable continuation collapses onto a small recurring "
            "set of states",
            len(cyclic) < 0.15 * n_ctx and int(np.max(periods)) < 24,
            f"{len(cyclic)} of {n_ctx} contexts lie on any cycle of the optimal "
            f"policy, and the longest cycle is {int(np.max(periods))} states")
    if int(np.max(periods)) == 1:
        s.note("in this demonstration the cycle turns out to have period one: "
               "the optimal policy walks into a single state and emits the same "
               "token from then on. The pigeonhole argument only guarantees "
               "some period, so period one is a fact about this model rather "
               "than a general one. It is also the sharpest possible version of "
               "the point, and the reason a real decoder's repetition looks "
               "like a repeated phrase rather than a repeated word is that its "
               "state is the whole prefix, so the cycle it finds is longer")

    s.kv("mean distinct bigrams in 40 tokens, argmax path",
         round(float(np.mean(argmax_bigrams)), 2))
    s.kv("mean distinct bigrams in 40 tokens, sampled",
         round(float(np.mean(sampled_bigrams)), 2))
    s.check("sampling covers several times more of the state space than the "
            "optimum does",
            float(np.mean(sampled_bigrams))
            > 2 * float(np.mean(argmax_bigrams)),
            f"{np.mean(sampled_bigrams):.1f} distinct bigrams against "
            f"{np.mean(argmax_bigrams):.1f}, over 40 tokens from each of the "
            f"{n_ctx} start contexts")
    s.check("the likelihood-maximising path repeats itself and the sampled one "
            "does not",
            float(np.mean(argmax_rep)) > 2 * float(np.mean(sampled_rep)),
            f"repeated-bigram rate {np.mean(argmax_rep):.3f} against "
            f"{np.mean(sampled_rep):.3f}")

    s.say()
    a0, b0 = VOCAB.index("the"), VOCAB.index("model")
    s.say(f"  argmax  : {words(policy_tokens(policy, a0, b0, 18))}")
    s.say(f"  sampled : {words(sample_path(cum, a0, b0, 18, rng(77).random(18)))}")
    s.note("this is the sentence the section exists for. Beam search is a better "
           "search than greedy, and searching harder makes the text worse, "
           "because the objective is not the one anybody wants. Likelihood "
           "ranks a repeated phrase above a varied one, and it is right to: a "
           "repeated phrase really is more probable")

    # --------------------------------------------- 3.8 the trade, in two units
    s.h("3.8  Sampling buys the shape of the process with likelihood")
    horizon = 24
    n_samples = 3
    greedy_policy = np.argmax(logP, axis=2)
    E = exact_values(logP, horizon) / horizon
    G = greedy_values(logP, horizon) / horizon
    lp_pure, lp_trunc = [], []
    tokens_pure, tokens_greedy = [], []
    g = rng(4242)
    Ptrunc = np.array([[top_p(P[a, b], 0.9)[0] for b in range(V)]
                       for a in range(V)])
    cum_trunc = np.cumsum(Ptrunc, axis=2)
    for a in range(V):
        for b in range(V):
            for _ in range(n_samples):
                seq = sample_path(cum, a, b, horizon, g.random(horizon))
                lp_pure.append(float(np.mean(
                    [logP[uu, vv, w] for uu, vv, w in _states(a, b, seq)])))
                tokens_pure.extend(seq)
                seq2 = sample_path(cum_trunc, a, b, horizon, g.random(horizon))
                lp_trunc.append(float(np.mean(
                    [logP[uu, vv, w] for uu, vv, w in _states(a, b, seq2)])))
            tokens_greedy.extend(policy_tokens(greedy_policy, a, b, horizon))

    by_decoder = [("exact optimum", float(E.mean())),
                  ("greedy", float(G.mean())),
                  ("top-p 0.9 sampling", float(np.mean(lp_trunc))),
                  ("pure sampling", float(np.mean(lp_pure)))]
    s.table(["decoder", "mean log2 p per token"],
            [[name, round(val, 4)] for name, val in by_decoder], align="lr")
    ordered = [val for _, val in by_decoder]
    s.check("the decoders order exactly by how much searching they do",
            all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)),
            " > ".join(f"{x:.4f}" for x in ordered))

    pi_true = stationary_over_states(src.P).sum(axis=0)
    emp_pure = np.bincount(tokens_pure, minlength=V) / len(tokens_pure)
    emp_greedy = np.bincount(tokens_greedy, minlength=V) / len(tokens_greedy)
    kl_pure = kl_bits(emp_pure, pi_true)
    kl_greedy = kl_bits(emp_greedy, pi_true)
    s.kv("D_KL(sampled tokens || true process)", round(kl_pure, 4), "bits")
    s.kv("D_KL(argmax tokens || true process)", round(kl_greedy, 4), "bits")
    s.check("the decoder with the higher likelihood produces the less faithful "
            "text",
            kl_greedy > 5 * kl_pure,
            f"{kl_greedy:.3f} bits against {kl_pure:.3f} bits of divergence "
            f"from the true unigram distribution of the process")
    s.note("two rankings, opposite directions, same four decoders. That is the "
           "whole trade and it has no resolution inside the decoder: "
           "probability of the output and resemblance to the source are "
           "different objectives, and any setting picks a point between them")

    # ------------------------------------------------------- 3.9 speculative
    s.h("3.9  Speculative decoding: fewer calls, identical distribution")
    draft = f["base"].with_lambdas((0.30, 0.70, 0.0))
    Q = draft.dist_tensor()
    s.note("the draft is the same counts read as a bigram, so it ignores the "
           "token two back. It is a genuinely worse model, which is the point: "
           "the argument below never assumes the draft is any good")

    s.eq("P(emit x) = min(p(x), q(x)) + max(p(x) - q(x), 0) = p(x)",
         "accept with probability min(1, p/q), otherwise draw from the "
         "normalised positive part of p - q")
    out = verified_distribution(P, Q)
    err = float(np.max(np.abs(out - P)))
    s.check("verified speculative decoding emits exactly the target "
            "distribution, at every context",
            err < 1e-14,
            f"largest absolute deviation {err:.3e} over all "
            f"{n_ctx * V} probabilities, computed algebraically rather than "
            f"sampled")

    alpha = np.minimum(P, Q).sum(axis=2)
    tv = 0.5 * np.abs(P - Q).sum(axis=2)
    s.check("the acceptance rate is one minus the total variation distance "
            "between draft and target",
            float(np.max(np.abs(alpha - (1.0 - tv)))) < 1e-14,
            f"agreement to {float(np.max(np.abs(alpha - (1.0 - tv)))):.2e} at "
            f"all {n_ctx} contexts")

    pi = stationary_over_states(P)
    m = float((pi * alpha).sum())
    s.kv("mean acceptance rate, weighted by how often each context occurs",
         round(m, 4))
    s.kv("ceiling on tokens per verification, 1/(1-a)", round(1 / (1 - m), 3))

    cumP, cumQ = np.cumsum(P, axis=2), np.cumsum(Q, axis=2)
    rows, per_context = [], {}
    for k in (1, 2, 4, 6, 8):
        got, acc, se = speculative_run(P, Q, cumP, cumQ, k, blocks=4_000,
                                       seed=1300 + k)
        # The same series evaluated at each context's own acceptance rate and
        # averaged over how often the chain visits that context. Not the exact
        # expected throughput, because the positions inside one block are not
        # independent draws from pi, but it is the right object to compare the
        # single-rate formula against.
        per_context[k] = float(
            (pi * np.array([[constant_rate_tokens(float(alpha[a, b]), k)
                             for b in range(V)] for a in range(V)])).sum())
        rows.append([k, f"{got:.4f} +/- {se:.4f}",
                     round(constant_rate_tokens(m, k), 4),
                     round(per_context[k], 4), round(acc, 4)])
    s.table(["draft length k", "tokens per verification, measured",
             "series at the mean rate", "series averaged per context",
             "accept rate over verified positions"], rows, align="rlrrr")
    s.note("the measured column carries the standard error of a 4,000-block "
           "run, and past k=4 that error is larger than the quantity still "
           "being gained, which is why it stops being monotone. The gap the "
           "next claim is about is between the two exact columns, not between "
           "the simulated one and anything")
    s.check("a decoder whose acceptance rate varies beats one with the same "
            "average rate everywhere",
            all(per_context[k] > constant_rate_tokens(m, k)
                for k in per_context if k > 1),
            "tokens per verification is a sum of powers of the acceptance rate "
            "and so convex in it, which is Jensen's inequality rather than a "
            "measurement artifact: "
            + ", ".join(f"k={k}: {per_context[k]:.4f} > "
                        f"{constant_rate_tokens(m, k):.4f}"
                        for k in (2, 4, 8)))
    # Taken from the exact column for the reason just given: the simulated
    # increments past k=4 are smaller than the noise on them, so a claim built
    # on them would be deciding a real question with a coin.
    early = per_context[2] - per_context[1]
    late = (per_context[8] - per_context[4]) / 4
    s.check("lengthening the draft has sharply diminishing returns",
            late < 0.25 * early,
            f"the second drafted token adds {early:.4f} tokens per "
            f"verification; each of the fifth through eighth adds {late:.4f}, "
            f"against a ceiling of {1 / (1 - m):.3f} that no draft length "
            f"passes")
    s.note("the technique buys latency and cannot buy or cost quality, which is "
           "unusual enough to be worth saying plainly. Its ceiling is set by a "
           "distance between two distributions, and the reason verifying k "
           "tokens costs about what verifying one costs is the memory-bound "
           "decode that S12 measures")

    s.say()
    s.note("where we are: a token has been chosen. What that costs, once the "
           "context is thousands of tokens long and the same prefix is read "
           "again on every step, is S11")

    s.close()
    return s


if __name__ == "__main__":
    run()
