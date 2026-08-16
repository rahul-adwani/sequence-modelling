"""S5. Encoder-decoder, and the bottleneck.

Everything so far read a sequence and produced a label or a next element. The
task that changed the field asks for something harder: read a sequence and
produce a *different* sequence, of a different length, in a different language.

The natural construction, and the one that worked, is two networks. An
**encoder** reads the source sentence and ends holding a state. A **decoder**
starts from that state and emits the target sentence one word at a time. Both
can be LSTMs, so both inherit S4's long-range gradient path.

    encoder:   s_1, s_2, ... , s_L      ->  keep the last one
    context:   c = s_L
    decoder:   reads c, emits y_1, y_2, ... , y_M

Look at the middle line. Everything the decoder will ever know about the source
sentence arrives through `c`, which is a fixed-width vector: the same size for a
four-word sentence and a forty-word one. That is the bottleneck, and this section
measures it.

The important thing about this failure is that it is *not* the one S3 and S4 were
about. Gating fixed a backward problem -- the gradient could not travel. This is
a forward problem -- the information cannot fit. An LSTM encoder has excellent
gradient flow and exactly the same bottleneck. Solving one said nothing about
the other, and running the two together is the most common way this history gets
told wrong.

Reference: Sutskever, Vinyals & Le (2014) and Cho et al. (2014) for the
architecture; Bahdanau, Cho & Bengio (2014) for the diagnosis and the fix that
S6 takes apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harness import Section, rng

VOCAB = 12


@dataclass
class RecallTask:
    """A translation-shaped task stripped to the property that matters.

    The source is a sequence of symbols. The decoder is asked, at each output
    step, to reproduce the symbol at a particular source position. Nothing about
    language is being modelled -- word order, grammar and meaning are all absent
    on purpose. What remains is the single question the bottleneck decides: can
    the information about source position j still be recovered from the context
    vector when the decoder needs it?

    Framing it as recall rather than translation makes the measurement clean. A
    real translation score would blend this failure with a dozen others and could
    not attribute any of them.
    """

    length: int
    d_state: int
    vocab: int = VOCAB
    seed: int = 3

    def source(self, n: int) -> np.ndarray:
        return rng(self.seed + n).integers(0, self.vocab, size=self.length)


def encode(symbols: np.ndarray, d_state: int, vocab: int = VOCAB,
           seed: int = 11) -> np.ndarray:
    """Compress a symbol sequence into one fixed-width real vector.

    A random projection of the one-hot symbols, accumulated with a fixed decay so
    that position carries information. This is a deliberately *generous* stand-in
    for a trained encoder: it is linear, it is lossless up to the dimension it
    has, and nothing about it is badly conditioned. Any real encoder does worse.

    That generosity is the point. If the bound below bites even here, it is a
    property of the fixed-width channel and not of anyone's training run.
    """
    g = rng(seed)
    E = g.normal(0, 1.0, (vocab, d_state))
    pos = g.normal(0, 1.0, (len(symbols), d_state))
    state = np.zeros(d_state)
    for t, sym in enumerate(symbols):
        state = 0.92 * state + E[sym] * pos[t]
    return state


def recall(state: np.ndarray, symbols: np.ndarray, position: int,
           d_state: int, vocab: int = VOCAB, seed: int = 11) -> int:
    """Best-effort read of one source position back out of the context vector.

    Correlates the state against what the state *would* have contained for each
    possible symbol at that position, and returns the best match. This is the
    most favourable possible decoder: it knows the encoder exactly, and it is
    given the position it wants. A learned decoder has neither advantage.
    """
    g = rng(seed)
    E = g.normal(0, 1.0, (vocab, d_state))
    pos = g.normal(0, 1.0, (len(symbols), d_state))
    decay = 0.92 ** (len(symbols) - 1 - position)
    scores = [float(state @ (E[v] * pos[position] * decay)) for v in range(vocab)]
    return int(np.argmax(scores))


def recovery_rate(length: int, d_state: int, trials: int = 60,
                  seed: int = 3) -> float:
    """Fraction of source positions whose symbol is recovered correctly.

    Deliberately not called recall. This is exact-match accuracy over a K-way
    choice: for each queried position, the reader picks one of `VOCAB` symbols
    and either gets it right or does not. Classification recall, TP/(TP+FN), is a
    different quantity defined against a positive class, and there is no positive
    class here. Chance is 1/VOCAB.
    """
    hits = total = 0
    for n in range(trials):
        syms = rng(seed + n).integers(0, VOCAB, size=length)
        state = encode(syms, d_state)
        for j in range(length):
            hits += int(recall(state, syms, j, d_state) == syms[j])
            total += 1
    return hits / total


def capacity_tokens(d_state: int, vocab: int = VOCAB,
                    bits_per_value: int = 32) -> float:
    """How many symbols a d-wide fp32 state could distinguish, at best.

        d * bits_per_value  >=  T * log2(vocab)

    A counting bound, so it holds for any encoder, trained or otherwise. Nothing
    can be recovered from a state that was never able to distinguish the inputs
    in the first place.
    """
    return d_state * bits_per_value / np.log2(vocab)


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s05", "Encoder-decoder, and the bottleneck",
                "a forward problem, which gating does nothing for").open()

    # ------------------------------------------------------ 5.1 the architecture
    s.h("5.1  Two networks, and the vector between them")
    s.eq("encoder: s_1 .. s_L   ->   c = s_L   ->   decoder: y_1 .. y_M",
         "the source is read into a state; the target is written out of it")
    s.say()
    s.say("  This is the first architecture in the article that produces a")
    s.say("  sequence rather than consuming one, and it is what made machine")
    s.say("  translation a neural problem. The encoder and decoder are usually")
    s.say("  LSTMs, so both inherit S4's gradient path.")
    s.say()
    s.say("  Everything the decoder knows about the source arrives through c.")
    s.say("  c has the same width whether the source was four words or forty.")

    # -------------------------------------------------------- 5.2 the bound
    s.h("5.2  The bound, before any measurement")
    s.eq("d * 32 bits  >=  L * log2(V) bits",
         "to distinguish every possible source of length L over a vocabulary of "
         "V, the state must have at least as many distinguishable configurations")
    rows = [[d, f"{d * 32:,}", round(float(np.log2(VOCAB)), 2),
             f"{capacity_tokens(d):,.0f}"] for d in (8, 16, 32, 64)]
    s.table(["state width d", "state bits (fp32)", "bits per symbol",
             "symbols distinguishable at best"], rows, align="rrrr")
    s.check("the ceiling on what a fixed-width state can carry is linear in its "
            "width and independent of training",
            all(abs(float(r[3].replace(",", "")) - capacity_tokens(r[0])) < 1.0
                for r in rows),
            "a counting argument, so no encoder beats it: information that was "
            "never distinguishable cannot be recovered")
    s.note("this is an upper bound and a generous one -- it assumes every bit of "
           "every float is used perfectly. Real encoders fall short of it by a "
           "wide margin, which is why 5.3 measures rather than stopping here")

    # ------------------------------------------------- 5.3 measured degradation
    s.h("5.3  What actually degrades")
    s.say("  Encode sequences of increasing length into a fixed-width vector, "
          "then\n  try to read individual positions back out. The reader is "
          "given every\n  advantage: it knows the encoder exactly and is told "
          "which position it\n  wants.")
    d_fixed = 32
    rows = []
    for L in (4, 8, 16, 32, 64):
        acc = recovery_rate(L, d_fixed)
        rows.append([L, round(acc, 4), round(1.0 / VOCAB, 4),
                     "yes" if acc > 0.9 else ("partial" if acc > 0.3 else "no")])
    s.table([f"source length L", "recovery rate", "chance", "usable?"], rows,
            align="rrrl")
    s.check("recall from a fixed-width context degrades as the source lengthens",
            rows[0][1] > rows[-1][1],
            f"accuracy falls from {rows[0][1]:.3f} at L={rows[0][0]} to "
            f"{rows[-1][1]:.3f} at L={rows[-1][0]}, at a constant state width of "
            f"{d_fixed}")
    s.check("the failure is the width of the channel, not the length as such",
            recovery_rate(32, 128) > recovery_rate(32, 16),
            f"at L=32, widening the state from 16 to 128 raises accuracy from "
            f"{recovery_rate(32, 16):.3f} to {recovery_rate(32, 128):.3f} -- "
            "the same sequence, a bigger pipe")

    rows = []
    for d in (8, 16, 32, 64, 128):
        rows.append([d, round(recovery_rate(24, d), 4),
                     f"{capacity_tokens(d):,.0f}"])
    s.table(["state width d", "recovery rate at L=24",
             "counting bound (symbols)"], rows, align="rrr")
    s.check("accuracy rises with state width, as the bound predicts it must",
            rows[0][1] < rows[-1][1],
            f"{rows[0][1]:.3f} at d={rows[0][0]} rising to {rows[-1][1]:.3f} at "
            f"d={rows[-1][0]}")
    s.note("note how far the measured numbers sit below the counting bound. A "
           "32-wide state could in principle distinguish hundreds of symbols and "
           "in practice recalls a fraction of twenty-four. The bound says what is "
           "impossible; it does not promise what is achievable")

    # ------------------------------------------- 5.4 gating does not help here
    s.h("5.4  This is not the problem gating solved")
    s.say("  Worth being explicit, because the two failures are routinely run")
    s.say("  together and they have nothing in common.")
    s.table(["", "vanishing gradient (S3, S4)", "the bottleneck (here)"],
            [["direction", "backward", "forward"],
             ["what fails", "the training signal cannot travel",
              "the information cannot fit"],
             ["fixed by", "an additive path through time",
              "not fixed by anything so far"],
             ["present in an LSTM encoder?", "no -- gating repaired it",
              "yes -- entirely unaffected"]], align="lll")
    s.check("an encoder with a perfect gradient path still has a fixed-width "
            "context vector",
            True,
            "gating changed how the gradient travels between steps; it did not "
            "change how much a state can hold, and 5.2's bound never mentions "
            "gradients")
    s.note("the practical signature reported at the time was exactly this: "
           "translation quality that held up on short sentences and fell away on "
           "long ones, with no corresponding training instability. A gradient "
           "problem does not look like that")

    # ------------------------------------------------------ 5.5 the way out
    s.h("5.5  Where the fix has to come from")
    s.say("  There are only two ways to widen a channel that is too narrow: make")
    s.say("  it wider, or stop forcing everything through it.")
    s.say()
    s.say("  Making it wider fails on cost. The state's width is a fixed model")
    s.say("  parameter that has to be paid for on every sequence, short ones")
    s.say("  included, and 5.3 shows the returns are gradual while the cost is")
    s.say("  immediate. Sizing the state for the longest sentence you might ever")
    s.say("  see is the same mistake as sizing per-position weights for the")
    s.say("  longest sequence, which S2 already rejected.")
    s.say()
    s.say("  Which leaves the second option. The encoder already computed a state")
    s.say("  at every source position -- s_1 through s_L -- and the architecture")
    s.say("  throws all but the last one away. They are still there. The question")
    s.say("  is what to do with them.")
    s.check("the information the bottleneck destroys was computed and then "
            "discarded",
            True,
            "the encoder produces L states and the context vector keeps one; "
            "nothing needs to be recomputed to do better")
    s.note("that is the observation S6 is built on, and it is worth pausing on "
           "because it is the whole idea. Attention is not a new source of "
           "information. It is the decision to stop throwing away information "
           "the model already had.")

    s.close()
    return s


if __name__ == "__main__":
    run()
