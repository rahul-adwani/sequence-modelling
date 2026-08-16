"""S7. The transformer, and arriving at GPT.

S6 closed on a question. If attention already lets any position read any other
directly, what is the recurrence still for?

The answer turned out to be nothing, and the paper that said so was called
"Attention Is All You Need" (Vaswani et al., 2017). Remove the recurrence, apply
attention to the sequence and itself, and two costs that had constrained every
architecture in this article disappear at once:

    path length         O(T)  ->  O(1)     how far information travels between
                                           any two positions
    sequential steps    O(T)  ->  O(1)     how many operations must happen in
                                           order before the next can start

Both matter, and the second is the one that changed the field. Every architecture
so far read a sentence one word at a time because it had no choice. A transformer
reads all positions simultaneously, so training time stops scaling with sequence
length in wall-clock terms and starts scaling with available hardware. That is
not a quality argument. It is the reason models got large.

Two things have to be paid for. Attending over all pairs is quadratic in
sequence length, which S11 prices in detail. And removing the recurrence removes
the only thing that knew what order the words were in -- 7.3 demonstrates that
the resulting model is literally blind to word order, and 7.4 gives it back.

The section ends where the series' next article begins: a decoder-only
transformer with a causal mask, trained to predict the next token, which is GPT.

Reference: Vaswani et al. (2017). Jay Alammar's "The Illustrated Transformer" is
the explanation most practitioners actually learned this from, including me.
"""
from __future__ import annotations

import numpy as np

from .harness import Section, entropy_bits, rng, softmax

D_MODEL, N_HEADS, T_DEMO = 32, 4, 8


def self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray,
                   Wv: np.ndarray, mask: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Attention where the queries, keys and values all come from one sequence.

        Q = X Wq,  K = X Wk,  V = X Wv
        A = softmax( Q K^T / sqrt(d) + mask ) V

    S6's attention had the decoder querying the encoder: two different sequences.
    Here a sequence attends to itself, which is what "self" means and what makes
    the recurrence redundant -- position i can read position j directly, whatever
    the distance, in one operation.
    """
    d = X.shape[1]
    scores = (X @ Wq) @ (X @ Wk).T / np.sqrt(d)
    if mask is not None:
        scores = scores + mask
    W = softmax(scores, axis=-1)
    return W, W @ (X @ Wv)


def causal_mask(T: int) -> np.ndarray:
    """Zero on and below the diagonal, -inf above it.

    Position i may attend to j only for j <= i. Adding -inf before the softmax
    sends those weights to exactly zero, which is the same mechanism S6 used and
    the reason the KV cache of S11 is exact rather than approximate.
    """
    return np.triu(np.full((T, T), -np.inf), k=1)


def sinusoidal_positions(T: int, d: int) -> np.ndarray:
    """The original positional encoding: sines and cosines at geometric rates.

        PE[t, 2i]   = sin(t / 10000^(2i/d))
        PE[t, 2i+1] = cos(t / 10000^(2i/d))

    Added to the token representations before the first layer. The frequencies
    span many scales so that both fine and coarse positional differences are
    representable, and -- the property that made this choice attractive --
    the encoding for position t+k is a fixed linear function of the encoding for
    position t, so relative offsets are expressible.
    """
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / np.power(10_000.0, i / d)
    out = np.zeros((T, d))
    out[:, 0::2] = np.sin(angle)
    out[:, 1::2] = np.cos(angle[:, :out[:, 1::2].shape[1]])
    return out


def path_length(architecture: str, T: int) -> int:
    """Steps of computation between the first and last position's information."""
    return {"recurrent": T - 1, "self-attention": 1}[architecture]


def sequential_ops(architecture: str, T: int) -> int:
    """Operations that must run in order, with unlimited parallel hardware."""
    return {"recurrent": T, "self-attention": 1}[architecture]


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s07", "The transformer, and arriving at GPT",
                "if attention reaches everywhere, the recurrence is "
                "redundant").open()

    g = rng(17)
    X = g.normal(0, 1.0, (T_DEMO, D_MODEL))
    Wq, Wk, Wv = (g.normal(0, 1 / np.sqrt(D_MODEL), (D_MODEL, D_MODEL))
                  for _ in range(3))

    # ------------------------------------------------------- 7.1 the removal
    s.h("7.1  Attention applied to a sequence and itself")
    s.eq("Q = X W_q,  K = X W_k,  V = X W_v",
         "queries, keys and values are three different views of the *same* "
         "sequence")
    s.eq("A = softmax(Q K^T / sqrt(d)) V",
         "every position scores every position, then averages their values")
    s.say()
    s.say("  In S6 the decoder queried the encoder: two sequences, one attending")
    s.say("  to the other. Here a sequence attends to itself. Position 1 can read")
    s.say("  position 40 in a single operation, so nothing needs to be carried")
    s.say("  forward step by step -- which leaves the recurrence with no job.")
    W, out = self_attention(X, Wq, Wk, Wv)
    s.kv("sequence length", T_DEMO)
    s.kv("attention matrix shape", f"{W.shape[0]} x {W.shape[1]}")
    s.kv("rows summing to 1", int(np.sum(np.abs(W.sum(axis=1) - 1) < 1e-12)))
    s.check("every position produces a distribution over every position",
            np.allclose(W.sum(axis=1), 1.0) and np.all(W >= 0),
            f"all {W.shape[0]} rows are probability distributions over all "
            f"{W.shape[1]} positions")

    # --------------------------------------------------- 7.2 the two costs
    s.h("7.2  The two quantities that changed")
    s.table(["", "recurrent", "self-attention"],
            [["path length between distant positions",
              f"O(T) = {path_length('recurrent', 1000)} at T=1,000",
              f"O(1) = {path_length('self-attention', 1000)}"],
             ["sequential operations",
              f"O(T) = {sequential_ops('recurrent', 1000):,} at T=1,000",
              f"O(1) = {sequential_ops('self-attention', 1000)}"],
             ["operations per layer", "O(T d^2)", "O(T^2 d + T d^2)"]],
            align="lll")
    s.check("self-attention makes the distance between any two positions "
            "constant",
            path_length("self-attention", 10_000) == 1,
            "position 1 and position 10,000 are one operation apart, against "
            "9,999 for a recurrent network -- and S3 showed what happens to a "
            "gradient crossing 9,999 multiplications")
    s.check("and removes the sequential dependency entirely",
            sequential_ops("self-attention", 10_000) == 1,
            "all positions are computed at once, so wall-clock training time "
            "stops scaling with sequence length and starts scaling with "
            "available hardware")
    s.note("the first row also settles S3 quietly. A gradient travelling between "
           "two positions no longer passes through a product of T Jacobians, "
           "because there is no path of length T -- there is a path of length 1. "
           "Gating mitigated the geometric series; self-attention deletes it. "
           "The second row is why transformers won commercially: not better per "
           "parameter, but trainable at a scale the alternatives could not reach "
           "in affordable wall-clock time")

    # ------------------------------------------- 7.3 what got thrown away
    s.h("7.3  Removing the recurrence removed the word order")
    s.say("  The recurrence was doing a second job nobody had listed. Reading")
    s.say("  words one after another is what made the model know which came")
    s.say("  first. Take it away and the model sees a *set*, not a sequence.")
    perm = rng(23).permutation(T_DEMO)
    W_p, out_p = self_attention(X[perm], Wq, Wk, Wv)
    s.kv("max |attention output - permuted output, unpermuted back|",
         f"{float(np.max(np.abs(out[perm] - out_p))):.3e}")
    s.check("self-attention is permutation-equivariant: shuffling the input "
            "shuffles the output and changes nothing else",
            float(np.max(np.abs(out[perm] - out_p))) < 1e-12,
            f"agreement to {float(np.max(np.abs(out[perm] - out_p))):.1e} -- the "
            "mechanism has no way to tell 'dog bites man' from 'man bites dog'")
    s.note("this is not a subtlety, it is a hole. Every architecture up to here "
           "got word order for free from the fact that it read the words in "
           "order. A transformer has to be told")

    # --------------------------------------------- 7.4 giving position back
    s.h("7.4  Positional encoding puts it back")
    s.eq("PE[t, 2i] = sin(t / 10000^(2i/d)),   PE[t, 2i+1] = cos(...)",
         "a fixed pattern per position, added to the token representations "
         "before the first layer")
    PE = sinusoidal_positions(T_DEMO, D_MODEL)
    Xp = X + PE
    _, out_pe = self_attention(Xp, Wq, Wk, Wv)
    _, out_pe_perm = self_attention(X[perm] + PE, Wq, Wk, Wv)
    delta = float(np.max(np.abs(out_pe[perm] - out_pe_perm)))
    s.kv("same comparison, with positions added", f"{delta:.3e}")
    s.check("adding positional encodings breaks the permutation symmetry",
            delta > 1e-3,
            f"the same shuffle now changes the output by {delta:.2e}, against "
            f"{float(np.max(np.abs(out[perm] - out_p))):.1e} without positions")
    s.check("the encoding gives distinct positions distinct representations",
            float(np.min([np.linalg.norm(PE[i] - PE[j])
                          for i in range(T_DEMO) for j in range(T_DEMO)
                          if i != j])) > 1e-6,
            "no two positions share an encoding, so the model can distinguish "
            "them even though the mechanism reading them is order-blind")
    s.note("worth noticing what kind of fix this is. Order is not restored by "
           "changing the mechanism -- attention remains permutation-equivariant "
           "for ever. It is restored by putting the position *into the data*, so "
           "that an order-blind mechanism reading position-tagged inputs behaves "
           "as though it could see order. Everything later in this line, from "
           "learned to rotary encodings, is a different answer to the same "
           "question")

    # ------------------------------------------------------ 7.5 multi-head
    s.h("7.5  Several attentions at once")
    s.say("  One attention distribution per position means one weighted average.")
    s.say("  But a word may relate to its subject and its object and its tense")
    s.say("  simultaneously, and a single distribution has to choose.")
    s.eq("head_h = Attention(X W_q^h, X W_k^h, X W_v^h);  concat, then project",
         "d is split across heads, so multi-head attention costs the same as "
         "single-head at the same width")
    d_head = D_MODEL // N_HEADS
    ents = []
    for h in range(N_HEADS):
        gh = rng(50 + h)
        wq, wk, wv = (gh.normal(0, 1 / np.sqrt(d_head), (D_MODEL, d_head))
                      for _ in range(3))
        Wh, _ = self_attention(Xp, wq, wk, wv)
        ents.append(float(np.mean([entropy_bits(r) for r in Wh])))
    s.table(["head", "mean attention entropy (bits)"],
            [[h, round(e, 3)] for h, e in enumerate(ents)], align="rr")
    s.kv("uniform over 8 positions would be", round(float(np.log2(T_DEMO)), 3))
    s.kv("d_model split across heads", f"{D_MODEL} / {N_HEADS} = {d_head}")
    s.check("heads attend differently, so several relations can be represented "
            "at once",
            max(ents) - min(ents) > 0.05,
            f"mean entropy ranges {min(ents):.3f} to {max(ents):.3f} bits across "
            f"{N_HEADS} heads on identical input -- each is a different view")
    s.check("splitting the width keeps the parameter count unchanged",
            N_HEADS * d_head == D_MODEL,
            f"{N_HEADS} heads x {d_head} dimensions = {D_MODEL}, so multi-head "
            "is a reorganisation of the same budget rather than an addition")

    # ---------------------------------------------------- 7.6 arriving at GPT
    s.h("7.6  The causal mask, and where this article ends")
    s.say("  The 2017 architecture had an encoder and a decoder, because it was")
    s.say("  built for translation. For a model that only continues text, the")
    s.say("  encoder has nothing to encode -- there is one sequence, and each")
    s.say("  position should see everything before it and nothing after.")
    s.eq("mask_ij = 0 if j <= i, else -infinity",
         "added to the scores before the softmax, so forbidden weights become "
         "exactly zero")
    M = causal_mask(T_DEMO)
    W_c, _ = self_attention(Xp, Wq, Wk, Wv, mask=M)
    upper = np.triu(W_c, k=1)
    s.table(["position i", "positions attended", "weight on the future"],
            [[i, int(np.sum(W_c[i] > 0)), f"{float(np.sum(upper[i])):.1e}"]
             for i in (0, 3, 7)], align="rrr")
    s.check("with a causal mask each position attends to itself and its past, "
            "and nothing else",
            float(np.max(np.abs(upper))) == 0.0
            and all(int(np.sum(W_c[i] > 0)) == i + 1 for i in range(T_DEMO)),
            "position i attends to exactly i+1 positions and places exactly zero "
            "weight on every later one")
    s.check("the rows are still probability distributions after masking",
            np.allclose(W_c.sum(axis=1), 1.0),
            "the softmax renormalises over what remains, so masking redistributes "
            "attention rather than discarding it")
    s.say()
    s.note("that is GPT: a stack of masked self-attention layers with positional "
           "encodings, trained to predict the next token, with the "
           "softmax-and-cross-entropy output layer this article opened on in "
           "S1. Every component has now been derived, and every one of them "
           "exists because something before it failed in a way we measured -- "
           "the fixed input width of a feedforward network, the geometric decay "
           "of a recurrent gradient, the fixed width of a context vector, the "
           "sequential cost of recurrence.")
    s.say()
    s.note("the next article starts here and asks a different question. Not how "
           "this object was arrived at, but what it *is*: a function from a "
           "token sequence to a distribution over the next token, with no "
           "memory, no state, and no ability to act -- and what it costs to run "
           "one at scale.")

    s.close()
    return s


if __name__ == "__main__":
    run()
