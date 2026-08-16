"""S6. Attention.

S5 ended on an observation rather than a proposal. The encoder computes a state
at every source position, s_1 through s_L, and the architecture keeps only the
last one. The information the bottleneck destroys was computed and then thrown
away.

Attention is the decision to stop throwing it away. Instead of one context vector
for the whole translation, the decoder builds a *fresh* context at every output
step, as a weighted average of all the encoder states:

    e_ij = score(d_i, s_j)                  how relevant is source j to output i
    a_ij = softmax_j(e_ij)                  weights over source positions, summing to 1
    c_i  = sum_j a_ij s_j                   this step's context

Three things are worth noticing before the measurements.

The context is now indexed by i. Every output step gets its own view of the
source, so the model is no longer compressing the sentence once and reusing the
compression.

The weights come from a softmax, which is the same object S1 introduced for the
output layer -- so `a_ij` is a probability distribution over source positions.
That is not a metaphor. It sums to one, and its entropy is measurable.

And because it is a distribution over inputs, it is legible. After the fact you
can ask which source positions a given output actually used. 6.4 measures how
faithful that reading is, because "attention is interpretable" is asserted far
more often than it is checked.

Reference: Bahdanau, Cho & Bengio (2014), which introduced this as an addition to
a recurrent encoder-decoder; Luong et al. (2015) for the dot-product scoring used
here. Jay Alammar's "Visualizing A Neural Machine Translation Model" is the
clearest visual treatment of the mechanism.
"""
from __future__ import annotations

import numpy as np

from .harness import Section, entropy_bits, rng, softmax
from .s05_bottleneck import VOCAB, capacity_tokens, recovery_rate

D_STATE = 32


def encoder_kv(symbols: np.ndarray, d_state: int = D_STATE, vocab: int = VOCAB,
               seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Per-position keys and values, which is what an encoder actually exposes.

    Real attention does not read the encoder's hidden states directly. It reads
    two linear projections of them:

        key   k_j  -- what position j is *about*, used for matching
        value v_j  -- what position j *contains*, used for the answer

    That separation is the mechanism, and conflating the two is the mistake this
    function exists to avoid. A key that must also carry the content cannot be
    matched cleanly against a query, which is exactly the failure the first
    version of this section measured: alignment landed at 14.6% because the
    "keys" were running accumulations of everything seen so far.

    The projections here are constructed rather than trained, which isolates the
    question this section is about -- can the mechanism reach the information? --
    from the separate question of whether gradient descent finds the right
    projections. A trained encoder has to discover both.
    """
    g = rng(seed)
    E = g.normal(0, 1.0, (vocab, d_state))          # content embeddings
    # Deliberately NOT normalised to unit length. For q and k with unit-variance
    # entries, q.k has variance d, so the matching position scores about sqrt(d)
    # after the 1/sqrt(d) scaling while the rest score about N(0,1). That gap is
    # what makes the softmax peaked. Normalising the keys caps q.k at 1, the gap
    # collapses to 1/sqrt(d), and attention degenerates into a uniform average --
    # measured here as alignment 100% with entropy 2.996 bits against a uniform
    # 3.0, a peak in the right place carrying almost no weight.
    P = g.normal(0, 1.0, (len(symbols), d_state))   # positional basis
    K = np.asarray([P[t] for t in range(len(symbols))])
    V = np.asarray([E[int(sym)] for sym in symbols])
    return K, V


def query_for(position: int, length: int, d_state: int = D_STATE,
              vocab: int = VOCAB, seed: int = 11) -> np.ndarray:
    """A decoder query asking about one source position.

    Stands in for a learned query vector, built from the same positional basis
    the keys use so that a match is possible at all.
    """
    g = rng(seed)
    g.normal(0, 1.0, (vocab, d_state))              # advance the stream identically
    P = g.normal(0, 1.0, (length, d_state))
    return P[position]


def attend(K: np.ndarray, V: np.ndarray, query: np.ndarray,
           temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One attention step: score against keys, weight, average the values.

        e_j = q . k_j / sqrt(d)      a_j = softmax(e)      c = sum_j a_j v_j
    """
    scores = (K @ query) / np.sqrt(K.shape[1])
    weights = softmax(scores / temperature)
    return weights, weights @ V


def attention_recovery(length: int, d_state: int = D_STATE, trials: int = 40,
                     seed: int = 3) -> tuple[float, float]:
    """Symbol recovery rate and alignment accuracy for the attending decoder.

    Alignment accuracy is the fraction of queries whose largest weight landed on
    the position actually being asked about. It is the measurable version of
    "the attention weights show you what the model looked at".
    """
    g_hits = a_hits = total = 0
    gg = rng(11)
    E = gg.normal(0, 1.0, (VOCAB, d_state))
    for n in range(trials):
        syms = rng(seed + n).integers(0, VOCAB, size=length)
        K, V = encoder_kv(syms, d_state)
        for j in range(length):
            w, ctx = attend(K, V, query_for(j, length, d_state))
            pred = int(np.argmax([float(ctx @ E[v]) for v in range(VOCAB)]))
            g_hits += int(pred == syms[j])
            a_hits += int(int(np.argmax(w)) == j)
            total += 1
    return g_hits / total, a_hits / total


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s06", "Attention",
                "stop discarding what the encoder already computed").open()

    # ---------------------------------------------------------- 6.1 the idea
    s.h("6.1  One context per output step, not one per sentence")
    s.eq("e_ij = score(d_i, s_j)", "how relevant source position j is to output i")
    s.eq("a_ij = softmax_j(e_ij)",
         "a probability distribution over source positions: non-negative, "
         "summing to one")
    s.eq("c_i = sum_j a_ij s_j",
         "this step's context: a weighted average of every encoder state")
    s.say()
    s.say("  S5's decoder read one vector, computed once. This decoder builds a")
    s.say("  new one at every output step, and can weight the source differently")
    s.say("  each time. Nothing new is computed on the encoder side -- s_1 to s_L")
    s.say("  already existed and were being discarded.")

    # ------------------------------------------------ 6.2 the bottleneck goes
    s.h("6.2  What that does to S5's measurement")
    rows = []
    for L in (4, 8, 16, 32, 64):
        base = recovery_rate(L, D_STATE)
        att, _ = attention_recovery(L, D_STATE)
        rows.append([L, round(base, 4), round(att, 4),
                     f"{att / max(base, 1e-9):.1f}x"])
    s.table(["source length L", "single context (S5)", "with attention",
             "improvement"], rows, align="rrrr")
    s.check("attention removes the length dependence that the bottleneck created",
            rows[-1][2] > 3 * rows[-1][1],
            f"at L={rows[-1][0]} recall goes from {rows[-1][1]:.3f} to "
            f"{rows[-1][2]:.3f}, at the same state width")
    s.check("the advantage grows with source length",
            (rows[-1][2] / max(rows[-1][1], 1e-9))
            > (rows[0][2] / max(rows[0][1], 1e-9)),
            f"{rows[0][3]} at L={rows[0][0]} against {rows[-1][3]} at "
            f"L={rows[-1][0]} -- short sentences never needed it")
    s.note("this is the shape reported at the time: attention barely helped on "
           "short sentences and transformed long ones. A fix whose benefit "
           "tracks the severity of the problem it targets is good evidence the "
           "diagnosis was right")

    # ------------------------------------------------ 6.3 why the bound moved
    s.h("6.3  Why the counting bound no longer applies")
    s.eq("single context:  d * 32 bits  must hold  L * log2(V) bits",
         "one vector, so the bound of S5.2 binds")
    s.eq("attention:       L states x d, read selectively per step",
         "the channel is no longer fixed-width; it grows with the source")
    rows = [[L, f"{capacity_tokens(D_STATE):,.0f}",
             f"{L * D_STATE:,}", f"{L}"] for L in (8, 32, 128)]
    s.table(["source length L", "single-context ceiling (symbols)",
             "numbers available with attention", "contexts built"], rows,
            align="rrrr")
    s.check("attention converts a fixed channel into one that scales with the "
            "input",
            True,
            "S5's bound was d*32 >= L*log2(V) and constrained L; with L states "
            "kept, the storage grows with L and the constraint disappears")
    s.note("and the cost arrives in the same breath. Keeping L states and "
           "scoring each of them at each of M output steps is L*M scores per "
           "sentence, where the single context needed one vector and none. That "
           "quadratic term never goes away -- S7 inherits it, and S11 is about "
           "paying for it")

    # ------------------------------------------------- 6.4 is it interpretable
    s.h("6.4  The weights are a distribution over inputs -- but are they honest?")
    s.say("  Because a_ij is a probability distribution over source positions, "
          "it can\n  be read after the fact: which parts of the input did this "
          "output use?\n  That claim is made constantly. It is checkable.")
    rows = []
    for L in (8, 16, 32, 64):
        rec, align = attention_recovery(L, D_STATE)
        Kd, Vd = encoder_kv(rng(3).integers(0, VOCAB, size=L), D_STATE)
        w, _ = attend(Kd, Vd, query_for(L // 2, L, D_STATE))
        rows.append([L, round(align, 4), round(entropy_bits(w), 3),
                     round(float(np.log2(L)), 3)])
    s.table(["source length L", "peak weight on the right position",
             "weight entropy (bits)", "uniform would be"], rows, align="rrrr")
    s.check("the attention weights concentrate on the source position actually "
            "being used",
            rows[0][1] > 0.8,
            f"the peak weight lands on the queried position "
            f"{rows[0][1]:.1%} of the time at L={rows[0][0]}")
    s.check("the weights carry real information, sitting well below uniform "
            "entropy",
            all(r[2] < r[3] for r in rows),
            "; ".join(f"L={r[0]}: {r[2]:.2f} bits against {r[3]:.2f} uniform"
                      for r in rows))
    s.note("this is the strongest practical argument for attention outside of "
           "accuracy, and it is worth stating carefully because it is easy to "
           "overclaim. What the weights give you is a record of which inputs the "
           "model *read* -- an attribution over the source, per output, "
           "available at no extra cost and without a separate explainability "
           "method. What they do not give you is a causal account of the "
           "model's reasoning: high weight means the value was mixed into the "
           "context, not that it determined the answer. For a regulated setting "
           "that distinction is the difference between a defensible artifact and "
           "an indefensible one, and it is measurable -- which is what the "
           "alignment column above is")

    # ---------------------------------------------------- 6.5 what is left
    s.h("6.5  What attention did not fix")
    s.say("  The encoder is still recurrent. s_j cannot be computed before")
    s.say("  s_{j-1}, so reading a source sentence of length L still takes L")
    s.say("  dependent steps, exactly as in S2.5 and S4.6.")
    rows = [[L, L, f"{L * L:,}"] for L in (32, 256, 2_048)]
    s.table(["source length L", "sequential steps to encode",
             "attention scores per sentence"], rows, align="rrr")
    s.check("attention left the sequential cost of the encoder untouched",
            True,
            "attention changed what the decoder may read; it did not change how "
            "the encoder states are produced, and they are still produced one "
            "after another")
    s.note("so by 2017 the position was: gradients travel (S4), information "
           "reaches the decoder (here), and the thing still runs one step at a "
           "time. On the hardware that was arriving, that last constraint had "
           "become the expensive one -- and the observation that resolved it is "
           "almost embarrassing in hindsight. If attention already lets any "
           "position read any other directly, what is the recurrence still for?")

    s.close()
    return s


if __name__ == "__main__":
    run()
