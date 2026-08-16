"""S8. The alphabet: what a token is.

Part 2 finished at a decoder-only transformer, a network that reads a sequence
of symbols and predicts the next symbol. It never said what a symbol is. The
output layer has one unit per symbol, so that choice fixes the width of the last
matrix, the length of every sequence, and the unit the bill is written in. It is
the first decision in the inference stack and the one most often skipped.

There are three candidates and the first two fail for reasons that can be
measured rather than argued.

**Words** give short sequences and an alphabet that never closes. Language
invents words faster than any vocabulary can hold them, so a held-out sample
always contains words the vocabulary does not, and every one of them arrives at
the model as the same symbol. In the vocabulary of Part 2 that is an
**information channel** failure: two different inputs become indistinguishable
before the network sees them, and no amount of training recovers a distinction
that was destroyed at the input.

**Characters** close the alphabet completely and pay for it in length. The same
text becomes several times longer, and Part 2 established that attention work
grows with the square of the sequence length. That is a **computation schedule**
cost, not a channel one.

**Byte-pair encoding** is what you get if you refuse both bills. Start from
characters, so nothing is unrepresentable, then buy back the length by merging
the pairs that actually occur, one merge at a time, until the vocabulary is as
large as you decided it should be. The result is neither words nor characters
and behaves like neither, which is why token counts do not match word counts,
why the same content costs different amounts in different scripts, and why a
model that writes flawless prose miscounts the letters in it.

The text here is generated rather than borrowed, for one reason: the section
compares two writing systems, and two real ones differ in vocabulary size, word
length, morphology and orthography all at once, so a measured difference cannot
be attributed to any of them. The second script here is the first script under a
one-to-one character substitution. The two are the same text in different
characters, identical in length and in every statistical respect, which leaves
exactly one thing that can explain a difference in token count.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .harness import Section, rng

# ------------------------------------------------------------------- the source

# Script A uses the first thirteen letters, script B the last thirteen. The map
# between them is rot13, chosen because it is visibly content-preserving and
# stays inside ASCII, so every log line renders on any console.
SCRIPT_A = "abcdefghijklm"
SCRIPT_B = "nopqrstuvwxyz"
ROT13 = str.maketrans(SCRIPT_A + SCRIPT_B + SCRIPT_A.upper() + SCRIPT_B.upper(),
                      SCRIPT_B + SCRIPT_A + SCRIPT_B.upper() + SCRIPT_A.upper())

VOWELS = "aei"
CONSONANTS = "bcdfghjklm"

WORD_MARK = "_"        # stands for "this token starts a word", i.e. one space


def syllable_word(g: np.random.Generator, n_syllables: int) -> str:
    return "".join(CONSONANTS[int(g.integers(len(CONSONANTS)))]
                   + VOWELS[int(g.integers(len(VOWELS)))]
                   for _ in range(n_syllables))


def lexicon(n_types: int, seed: int) -> list[str]:
    """A closed inventory of word types, built from consonant-vowel syllables.

    Syllable structure is not decoration. Byte-pair encoding finds whatever
    substructure the text has; a lexicon of random letter strings would have
    none, and the compression measured below would be an artifact of the
    generator rather than a property of the algorithm.
    """
    g = rng(seed)
    out, seen = [], set()
    while len(out) < n_types:
        k = int(g.choice((1, 2, 3, 4), p=(0.18, 0.42, 0.28, 0.12)))
        w = syllable_word(g, k)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def zipf_weights(n: int, s: float = 1.07) -> np.ndarray:
    """Frequencies falling as 1/rank^s, which is what word frequencies do."""
    w = 1.0 / np.arange(1, n + 1, dtype=np.float64) ** s
    return w / w.sum()


def generate_text(n_words: int, seed: int, lex: list[str], weights: np.ndarray,
                  p_novel: float = 0.03, p_number: float = 0.02) -> str:
    """A corpus with an open vocabulary.

    `p_novel` is the whole point of the generator. A closed lexicon would make
    a word-level vocabulary look adequate, because a held-out sample drawn from
    the same closed lexicon contains nothing new. Real text does not work that
    way: names, compounds, numbers and misspellings arrive forever, and any
    honest measurement of an out-of-vocabulary rate needs a source that keeps
    producing them.
    """
    g = rng(seed)
    known = set(lex)
    kind = g.random(n_words)
    picks = g.choice(len(lex), size=n_words, p=weights)
    numbers = (10.0 ** g.uniform(0, 5, size=n_words)).astype(np.int64)

    words: list[str] = []
    for i in range(n_words):
        if kind[i] < p_number:
            words.append(str(int(numbers[i])))
        elif kind[i] < p_number + p_novel:
            w = syllable_word(g, int(g.integers(3, 6)))
            while w in known:
                w = syllable_word(g, int(g.integers(3, 6)))
            words.append(w)
        else:
            words.append(lex[int(picks[i])])

    lengths = 6 + g.poisson(7, size=n_words)
    out, i, k = [], 0, 0
    while i < len(words):
        chunk = words[i:i + int(lengths[k])]
        if not chunk:
            break
        chunk[0] = chunk[0].capitalize()
        out.extend(chunk)
        out.append(".")
        i += len(chunk)
        k += 1
    return " ".join(out)


def take_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def sentences_of(words: list[str]) -> list[str]:
    """Split a word list at the sentence marks, dropping the marks themselves."""
    out, cur = [], []
    for w in words:
        if w == ".":
            if cur:
                out.append(" ".join(cur))
            cur = []
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def rot13(text: str) -> str:
    return text.translate(ROT13)


def script_of(token: str) -> str:
    """Which writing system a vocabulary entry belongs to, if either.

    The word mark, the sentence mark and the digits belong to neither, and a
    token made only of those is counted as neither rather than being assigned
    to whichever script happens to be listed first.
    """
    low = set(token.lower())
    in_a, in_b = bool(low & set(SCRIPT_A)), bool(low & set(SCRIPT_B))
    if in_a and not in_b:
        return "A"
    if in_b and not in_a:
        return "B"
    return "-"


# ------------------------------------------------------------------ the encoder

def pretokenize(text: str) -> list[str]:
    """Split on whitespace and mark word starts.

    Production tokenizers use a regular expression that also splits punctuation
    and digit runs, and they operate on bytes rather than characters so that no
    input is unrepresentable. Whitespace splitting is enough for everything
    measured here, and where it is not, the measurement says so.
    """
    return [WORD_MARK + w for w in text.split()]


def count_pairs(syms: list[list[str]], freq: list[int]) -> dict:
    """Frequency-weighted count of adjacent symbol pairs, without overlap.

    The correction at the end is the one place a naive implementation is wrong
    in a way nothing later catches. A run of L identical symbols has L-1
    adjacent positions holding the pair (x, x), but merging left to right
    consumes two symbols at a time and so performs only L//2 merges: the first
    merge destroys the second position. Counting the positions therefore
    overstates what the merge will do, and the overstatement shows up as a
    vocabulary whose token totals do not add up.
    """
    counts: dict = {}
    for s, f in zip(syms, freq):
        repeat = False
        for pair in zip(s, s[1:]):
            counts[pair] = counts.get(pair, 0) + f
            if pair[0] == pair[1]:
                repeat = True
        if repeat:
            j, n = 0, len(s)
            while j < n:
                k = j
                while k + 1 < n and s[k + 1] == s[j]:
                    k += 1
                run = k - j + 1
                if run >= 2:
                    counts[(s[j], s[j])] -= f * ((run - 1) - run // 2)
                j = k + 1
    return counts


def apply_merge(s: list[str], a: str, b: str, ab: str) -> list[str]:
    out, j, n = [], 0, len(s)
    while j < n:
        if j < n - 1 and s[j] == a and s[j + 1] == b:
            out.append(ab)
            j += 2
        else:
            out.append(s[j])
            j += 1
    return out


class BPE:
    """Byte-pair encoding, trained by counting and merging.

        1. Start with every word written as a list of single characters.
        2. Count every adjacent pair across the corpus, weighted by how often
           its word occurs.
        3. Merge the most frequent pair everywhere it appears. That merge is a
           new vocabulary entry.
        4. Repeat until the vocabulary is the size you asked for.

    Two properties follow immediately and both are checked in the run. The
    vocabulary size is exactly the number of starting characters plus the number
    of merges, so it is a dial rather than a property of the data. And each merge
    removes from the corpus exactly as many tokens as the pair had occurrences,
    so the compression is not an estimate.

    The merges are chosen greedily and in order, and no choice depends on a later
    one. Truncating a trained merge list to its first n entries therefore gives
    exactly the tokenizer that training with n merges would have produced, which
    is what `truncated` exploits to sweep vocabulary size without retraining.
    """

    def __init__(self, corpus: str, n_merges: int):
        self.word_freq = Counter(pretokenize(corpus))
        self.base = sorted({ch for w in self.word_freq for ch in w})
        self.n_merges_requested = n_merges
        self._train(n_merges)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = self.base + ["".join(p) for p in self.merges]
        self._cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ train

    def _train(self, n_merges: int) -> None:
        keys = list(self.word_freq)
        syms = [list(k) for k in keys]
        freq = [self.word_freq[k] for k in keys]

        self.merges: list[tuple[str, str]] = []
        self.merge_counts: list[int] = []
        self.predicted_tokens: list[int] = []   # running total, by arithmetic
        self.actual_tokens: list[int] = []      # running total, recounted

        total = sum(f * len(s) for f, s in zip(freq, syms))
        self.corpus_tokens_at_start = total

        for _ in range(n_merges):
            counts = count_pairs(syms, freq)
            if not counts:
                break
            # Ties broken on the pair itself, so the result does not depend on
            # dictionary iteration order.
            best = max(counts, key=lambda p: (counts[p], p))
            c = counts[best]
            if c < 2:
                break
            a, b = best
            ab = a + b
            for i, s in enumerate(syms):
                if len(s) > 1 and a in s:
                    syms[i] = apply_merge(s, a, b, ab)
            total -= c
            self.merges.append(best)
            self.merge_counts.append(c)
            self.predicted_tokens.append(total)
            self.actual_tokens.append(sum(f * len(s) for f, s in zip(freq, syms)))

    def truncated(self, n_merges: int) -> "BPE":
        clone = object.__new__(BPE)
        clone.__dict__.update(self.__dict__)
        clone.merges = self.merges[:n_merges]
        clone.merge_counts = self.merge_counts[:n_merges]
        clone.ranks = {p: i for i, p in enumerate(clone.merges)}
        clone.vocab = self.base + ["".join(p) for p in clone.merges]
        clone._cache = {}
        return clone

    # ----------------------------------------------------------------- encode

    def encode_word(self, word: str) -> list[str]:
        """Apply the lowest-ranked applicable merge, repeatedly.

        Rank is merge order, so this reproduces training exactly: every merge
        learned earlier is applied before any merge learned later, and within a
        rank the leftmost occurrence goes first.
        """
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        symbols = list(word)
        while len(symbols) > 1:
            best, at = None, -1
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                r = self.ranks.get(pair)
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at < 0:
                break
            symbols[at:at + 2] = [symbols[at] + symbols[at + 1]]
        self._cache[word] = symbols
        return symbols

    def encode(self, text: str) -> list[str]:
        out: list[str] = []
        for w in pretokenize(text):
            out.extend(self.encode_word(w))
        return out

    def decode(self, tokens: list[str]) -> str:
        """Concatenate and turn every word mark back into the space it stood for."""
        joined = "".join(tokens).replace(WORD_MARK, " ")
        return joined[1:] if joined.startswith(" ") else joined

    def n_tokens(self, text: str) -> int:
        return sum(len(self.encode_word(w)) for w in pretokenize(text))


def show(tokens: list[str], limit: int = 14) -> str:
    shown = " | ".join(tokens[:limit])
    return shown + (" | ..." if len(tokens) > limit else "")


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s08", "The alphabet: what a token is",
                "words, characters, and the encoding that refuses both bills").open()

    lex = lexicon(400, seed=31)
    weights = zipf_weights(len(lex))
    train = generate_text(18_000, seed=32, lex=lex, weights=weights)
    held = generate_text(4_000, seed=33, lex=lex, weights=weights)

    train_words = train.split()
    held_words = held.split()

    # ------------------------------------------------- 1.1 words do not close
    s.h("1.1  A word vocabulary never closes")
    s.kv("training words", len(train_words))
    s.kv("held-out words", len(held_words))
    s.say()
    s.say(f"  sample:  {' '.join(train_words[:16])}")
    s.note("the source has a fixed lexicon of 400 common words, plus a 3% rate "
           "of words it has never produced before and a 2% rate of numbers; "
           "that open tail is the property real text has and a closed generator "
           "does not")

    rows, seen, marks = [], set(), (2_000, 4_500, 9_000, 18_000)
    prev = 0
    for i, w in enumerate(train_words, start=1):
        seen.add(w)
        if i in marks:
            rows.append([i, len(seen), len(seen) - prev])
            prev = len(seen)
    s.table(["words read", "distinct types", "new types since last row"], rows,
            align="rrr")
    new_in_last_tenth = len({w for w in train_words[-len(train_words) // 10:]}
                            - set(train_words[:-len(train_words) // 10]))
    s.check("the type inventory is still growing when the corpus runs out",
            new_in_last_tenth > 0,
            f"{new_in_last_tenth} word types appear for the first time in the "
            f"final tenth of the corpus")

    freq = Counter(train_words)
    cap = 1_000
    kept = {w for w, _ in freq.most_common(cap)}
    oov = [w for w in held_words if w not in kept]
    oov_rate = len(oov) / len(held_words)
    s.say()
    s.kv("word vocabulary cap", cap, "types")
    s.kv("held-out words outside it", f"{len(oov)} of {len(held_words)}")
    s.kv("out-of-vocabulary rate", round(100 * oov_rate, 3), "%")
    s.kv("distinct types collapsed onto one symbol", len(set(oov)))
    s.check("a capped word vocabulary leaves part of held-out text unrepresentable",
            oov_rate > 0.01 and len(set(oov)) > 50,
            f"{100 * oov_rate:.2f}% of held-out words, spanning "
            f"{len(set(oov))} distinct types, all arriving as the same symbol")
    s.note("this is an information channel failure in the sense Part 2 defined: "
           "distinct inputs become identical before the network sees them. "
           "Training cannot undo it, because there is nothing left to learn from")

    # ---------------------------------------------- 1.2 characters, and length
    s.h("1.2  Characters close the alphabet and lengthen the sequence")
    train_chars = sorted(set(train.replace(" ", "")))
    held_chars = set(held.replace(" ", ""))
    s.kv("character vocabulary", len(train_chars), "symbols")
    s.say(f"  characters: {''.join(train_chars)}")
    s.check("a character vocabulary has no out-of-vocabulary rate at all",
            held_chars <= set(train_chars),
            f"{len(held_chars)} distinct held-out characters, all present in "
            f"the {len(train_chars)}-symbol training set")

    s.kv("letters per word, held out",
         round(len(held.replace(" ", "")) / len(held_words), 3))
    s.kv("characters per word including its space",
         round(len(held) / len(held_words), 3))
    s.note("a character-level model has to emit the spaces too, so the length "
           "to compare against is the whole string and not just its letters")
    s.note("closing the alphabet costs sequence length, and Part 2 showed "
           "attention work carries a term in the square of the sequence length; "
           "the multiplier on that term is measured once byte-pair encoding "
           "gives something to compare against, in 1.4")

    # ------------------------------------------------------------- 1.3 the BPE
    s.h("1.3  Byte-pair encoding, one merge at a time")
    s.eq("merge_k = argmax_(a,b) count(a,b);   V = base + {a+b : k = 1..m}",
         "count over the corpus written in the symbols surviving the previous "
         "k-1 merges, so every merge changes what the next one is counting")

    n_merges = 350
    bpe = BPE(train, n_merges=n_merges)
    s.kv("merges performed", len(bpe.merges))
    s.kv("base characters", len(bpe.base))
    s.kv("vocabulary size", len(bpe.vocab), "entries")
    s.say()
    s.table(["k", "pair merged", "new token", "occurrences", "corpus tokens after"],
            [[k + 1, f"{a!r} + {b!r}", repr(a + b), bpe.merge_counts[k],
              bpe.predicted_tokens[k]]
             for k, (a, b) in enumerate(bpe.merges[:12])], align="rllrr")
    s.note("the first merges are syllables, because syllables are what this "
           "text is made of; the later ones are whole frequent words, which is "
           "why a common word ends up as a single token and a rare one does not")

    s.check("no merge produces a string the vocabulary already has, so the "
            "vocabulary size is exactly the base plus the merge count",
            len(set(bpe.vocab)) == len(bpe.vocab)
            and len(bpe.vocab) == len(bpe.base) + len(bpe.merges),
            f"{len(bpe.base)} characters + {len(bpe.merges)} merges = "
            f"{len(bpe.vocab)} distinct entries")

    mismatch = max(abs(p - a) for p, a in
                   zip(bpe.predicted_tokens, bpe.actual_tokens))
    s.check("each merge removes exactly as many tokens as the pair had "
            "occurrences",
            mismatch == 0,
            f"predicted and recounted corpus totals agree at all "
            f"{len(bpe.merges)} merges, worst disagreement {mismatch} tokens")
    s.note("this is the claim that catches the overlap bug: counting the "
           "positions of a repeated pair rather than the merges it will "
           "actually support makes this number non-zero and nothing else in the "
           "section notices")

    round_trip = bpe.decode(bpe.encode(held))
    s.check("encoding and decoding held-out text is an exact round trip",
            round_trip == " ".join(held_words),
            f"{len(held_words)} words, {len(held)} characters, compared as "
            f"strings rather than approximately")

    sample = " ".join(held_words[:9])
    s.say()
    s.say(f"  text  :  {sample}")
    s.say(f"  tokens:  {show(bpe.encode(sample), limit=18)}")

    # ------------------------------------------------------ 1.4 what merges buy
    s.h("1.4  What the merges buy, and where they stop buying it")
    char_seq = len(held)          # what a character model would have to emit
    rows, tpw = [], []
    grid = (0, 25, 50, 100, 200, 350)
    for m in grid:
        t = bpe.truncated(m)
        n_tok = t.n_tokens(held)
        tpw.append(n_tok / len(held_words))
        rows.append([m, len(t.vocab), n_tok, round(n_tok / len(held_words), 4),
                     round(char_seq / n_tok, 4)])
    s.table(["merges", "vocabulary", "held-out tokens", "tokens per word",
             "characters per token"], rows, align="rrrrr")
    s.check("more merges never lengthen the encoding",
            all(tpw[i] >= tpw[i + 1] for i in range(len(tpw) - 1)),
            " >= ".join(f"{x:.3f}" for x in tpw))

    gains = [(tpw[i] - tpw[i + 1]) / (grid[i + 1] - grid[i])
             for i in range(len(grid) - 1)]
    s.check("the return on each additional merge falls as the vocabulary grows",
            all(gains[i] > gains[i + 1] for i in range(len(gains) - 1)),
            "tokens saved per word per merge: "
            + " > ".join(f"{x:.2e}" for x in gains))
    s.note("the first merges take the pairs that occur everywhere and the last "
           "ones take pairs that occur in a handful of words, so vocabulary "
           "size is a decision with a knee in it rather than a free parameter")

    n_tok_full = bpe.n_tokens(held)
    ratio = char_seq / n_tok_full
    s.kv("characters per token at full vocabulary", round(ratio, 4))
    s.kv("sequence-length multiplier, characters vs tokens", round(ratio, 3), "x")
    s.kv("multiplier on the quadratic attention term", round(ratio ** 2, 2), "x")
    s.check("moving to characters multiplies the quadratic term of attention by "
            "the square of the length ratio",
            ratio > 1.5,
            f"a character sequence is {ratio:.2f}x longer, so the T^2 term is "
            f"{ratio ** 2:.1f}x larger for identical content")

    # ------------------------------------------------ 1.5 tokens are not words
    s.h("1.5  A token is not a word, and the ratio is not a constant")
    sentences = sentences_of(held_words)
    per_sentence = [bpe.n_tokens(t) / len(t.split()) for t in sentences]
    arr = np.asarray(per_sentence)
    s.kv("sentences measured", len(sentences))
    s.kv("mean tokens per word", round(float(arr.mean()), 4))
    s.kv("minimum over sentences", round(float(arr.min()), 4))
    s.kv("maximum over sentences", round(float(arr.max()), 4))
    s.kv("ratio of maximum to minimum", round(float(arr.max() / arr.min()), 3))
    s.check("there is no fixed conversion between words and tokens",
            float(arr.max() / arr.min()) > 1.5,
            f"tokens per word ranges from {arr.min():.2f} to {arr.max():.2f} "
            f"across {len(sentences)} sentences of the same corpus")
    s.note("a prompt budget quoted in words is a budget with an unknown error "
           "bar; the same word count can differ by more than half again in what "
           "it costs, and the direction of the error depends on how ordinary "
           "the text is")

    # ------------------------------------------- 1.6 the same content, twice
    s.h("1.6  The same content in a second script costs more")
    base = generate_text(8_000, seed=34, lex=lex, weights=weights)
    base_words = base.split()
    n = len(base_words)
    skew_corpus = (take_words(base, int(0.9 * n)) + " "
                   + rot13(take_words(base, int(0.1 * n))))
    balanced_corpus = (take_words(base, n // 2) + " "
                       + rot13(take_words(base, n // 2)))

    eval_a = " ".join(generate_text(1_200, seed=35, lex=lex,
                                    weights=weights).split())
    eval_b = rot13(eval_a)
    s.check("the two texts are the same content in different characters",
            len(eval_a) == len(eval_b) and rot13(eval_b) == eval_a
            and eval_a != eval_b,
            f"{len(eval_a)} characters each, one the image of the other under a "
            f"one-to-one character map")

    rows, measured = [], {}
    for label, corpus in (("90% script A, 10% script B", skew_corpus),
                          ("50% script A, 50% script B", balanced_corpus)):
        tok = BPE(corpus, n_merges=220)
        scripts = Counter(script_of(a + b) for a, b in tok.merges)
        na, nb = tok.n_tokens(eval_a), tok.n_tokens(eval_b)
        measured[label] = (na, nb, nb / na)
        rows.append([label, scripts["A"], scripts["B"], na, nb,
                     round(nb / na, 4)])
    s.table(["tokenizer training mix", "merges in A", "merges in B",
             "tokens for the A text", "tokens for the B text", "B / A"],
            rows, align="lrrrrr")
    skew_na, skew_nb, skew_ratio = measured["90% script A, 10% script B"]
    _, _, bal_ratio = measured["50% script A, 50% script B"]

    s.check("identical content costs more tokens in the script the tokenizer "
            "saw less of",
            skew_ratio > 1.25,
            f"{skew_nb} tokens against {skew_na} for the same {len(eval_a)} "
            f"characters, a factor of {skew_ratio:.2f}")
    s.check("balancing the training mix removes the penalty",
            abs(bal_ratio - 1.0) < 0.05,
            f"B / A falls from {skew_ratio:.3f} to {bal_ratio:.3f} when the two "
            f"scripts get equal shares of the same total corpus")
    s.note("the two scripts are the same text under a character substitution, "
           "so nothing about either language explains the gap. The tokenizer's "
           "training mix explains all of it. Since inference is billed per "
           "token, that mix sets the price of the same work in different "
           "markets, and it is fixed long before anyone writes a prompt")

    # --------------------------------------------- 1.7 what the model cannot see
    s.h("1.7  What a token hides: the characters inside it")
    enc = [bpe.encode_word(w) for w in pretokenize(held)]
    single = sum(1 for e in enc if len(e) == 1)
    s.kv("held-out word occurrences", len(enc))
    s.kv("arriving as exactly one token", single)
    s.check("most held-out word occurrences reach the model as a single token",
            single / len(enc) > 0.5,
            f"{100 * single / len(enc):.1f}% of {len(enc)} occurrences, whose "
            f"characters are therefore not separately addressable")

    lens = [len(w) - 1 for w, e in zip(pretokenize(held), enc) if len(e) == 1]
    s.kv("characters in a single-token word", f"{min(lens)} to {max(lens)}")
    s.check("token count carries no information about character count",
            max(lens) > min(lens) + 2,
            f"single-token words in this vocabulary run from {min(lens)} to "
            f"{max(lens)} characters")
    s.note("spelling, reversing, counting a letter and rhyming are all "
           "character-level questions asked of a model whose input has no "
           "character level. The information is not absent from the world, it "
           "is absent from the representation, and recovering it means "
           "inverting the merge table from examples")

    counts = Counter(len(bpe.encode_word(WORD_MARK + str(v)))
                     for v in range(100, 1_000))
    s.say()
    s.table(["tokens", "three-digit numbers encoded that way"],
            [[k, counts[k]] for k in sorted(counts)], align="rr")
    s.check("consecutive numbers do not share a segmentation",
            len(counts) > 1,
            "the 900 three-digit numbers take "
            + ", ".join(f"{counts[k]} at {k} token{'s' if k > 1 else ''}"
                        for k in sorted(counts))
            + ", so digit position is not a fixed feature of the input")

    common = [w for w, _ in freq.most_common(60)
              if w.isalpha() and w.islower()][:20]
    disjoint = [w for w in common
                if not set(bpe.encode_word(WORD_MARK + w))
                & set(bpe.encode_word(WORD_MARK + w.capitalize()))]
    s.say()
    s.table(["word", "lower case", "capitalised"],
            [[w, show(bpe.encode_word(WORD_MARK + w), 6),
              show(bpe.encode_word(WORD_MARK + w.capitalize()), 6)]
             for w in common[:6]])
    s.check("a capitalised word and its lower-case form need not share a single "
            "token",
            len(disjoint) >= len(common) // 2,
            f"{len(disjoint)} of the {len(common)} most frequent words have no "
            f"token in common with their capitalised form")
    s.note("the model is not told that these are the same word. It has to infer "
           "it, from data, for every variant of every word, which is one of the "
           "quieter costs of the whole scheme")

    s.say()
    s.note("where we are: the alphabet is closed, nothing is unrepresentable, "
           "and the sequence is about as short as characters allow. What the "
           "model now reads is a sequence of vocabulary indices, and S9 asks "
           "what the function reading them actually is")

    s.close()
    return s


if __name__ == "__main__":
    run()
