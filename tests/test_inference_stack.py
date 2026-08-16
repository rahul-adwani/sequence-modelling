"""Invariants of S1, and of the Article 3 sections S8 through S12.

S2 through S7, which carry Article 2, are covered in
`test_sequence_modelling.py`.

Deliberately not a re-run of the sections. These exercise the underlying
functions directly, so the suite finishes in seconds and a failure points at a
function rather than at a paragraph of output.
"""
from __future__ import annotations

import numpy as np
import pytest

from foundations.harness import kl_bits, rng, softmax
from foundations.s01_output_and_loss import (BinaryNet, DelayedSource, categorical_ce,
                                           ce_grad_analytic, iris_like, num_grad,
                                           sigmoid, softmax_jacobian, window_model,
                                           window_predict)
from foundations.s08_tokenization import (BPE, WORD_MARK, apply_merge, count_pairs,
                                          generate_text, lexicon, pretokenize,
                                          rot13, script_of, zipf_weights)
from foundations.s09_llm_as_function import V, MarkovSource, TrigramLM
from foundations.s10_decoding import (brute_force_best, constant_rate_tokens,
                                      exact_values, greedy_values, policy_cycle,
                                      policy_tokens, stationary_over_states,
                                      stationary_policy, temper, top_k, top_p,
                                      verified_distribution)
from foundations.s11_kv_cache import (SPECS, Config, Counter, Transformer,
                                      attention_share, int8_kv, macs_prefill,
                                      macs_step, sliding_window)
from foundations.s12_cost import (DEVICES, decode_intensity,
                                  decode_intensity_ceiling, mm1_percentiles,
                                  simulate_staleness, staleness_probability)


# ------------------------------------------------------------------------- S1

def test_sigmoid_is_stable_at_extremes():
    z = np.array([-800.0, 0.0, 800.0])
    p = sigmoid(z)
    assert np.isfinite(p).all()
    assert p[0] == pytest.approx(0.0, abs=1e-300)
    assert p[1] == pytest.approx(0.5)
    assert p[2] == pytest.approx(1.0)


def test_binary_net_gradients_match_finite_differences():
    net, (X, Y) = BinaryNet(), iris_like(n=20)
    dW, _ = net.gradients(X[0], float(Y[0]))
    fd = net.finite_difference(X[0], float(Y[0]))
    a = np.concatenate([g.ravel() for g in dW])
    n = np.concatenate([g.ravel() for g in fd])
    rel = np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n))
    assert rel < 1e-8


def test_softmax_jacobian_rows_sum_to_zero():
    p = softmax(rng(21).normal(size=7))
    assert np.allclose(softmax_jacobian(p).sum(axis=1), 0.0, atol=1e-12)


def test_softmax_jacobian_is_symmetric():
    """J = diag(p) - p p^T is symmetric; an asymmetric result means a bug."""
    p = softmax(rng(22).normal(size=6))
    J = softmax_jacobian(p)
    assert np.allclose(J, J.T)


def test_ce_gradient_is_p_minus_y():
    z = rng(23).normal(size=5)
    y = np.eye(5)[3]
    assert np.allclose(ce_grad_analytic(z, y),
                       num_grad(lambda zz: categorical_ce(zz, y), z), atol=1e-8)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_k2_softmax_reduces_to_the_published_binary_case(label):
    """The load-bearing continuity claim of the whole series."""
    z = rng(24).normal(size=2)
    p = softmax(z)
    assert float(p[1]) == pytest.approx(float(sigmoid(z[1] - z[0])), abs=1e-15)

    y = np.array([1.0 - label, label])
    l_cat = categorical_ce(z, y)
    a = float(p[1])
    l_bin = -(label * np.log(a) + (1 - label) * np.log(1 - a))
    assert l_cat == pytest.approx(l_bin, abs=1e-12)
    assert float(ce_grad_analytic(z, y)[1]) == pytest.approx(a - label, abs=1e-12)


def test_window_model_rows_are_distributions():
    src = DelayedSource()
    toks = src.sample(2_000, seed=1)
    for w in (0, 1, 2):
        table = window_model(toks, w, src.vocab)
        assert np.allclose(table.sum(axis=1), 1.0)
        assert (table > 0).all()          # smoothing keeps the support full


def test_window_below_dependency_cannot_see_it():
    src = DelayedSource()
    train, test = src.sample(30_000, seed=2), src.sample(500, seed=3)
    # Build each count table once. Fitting it inside the comprehension refits it
    # per evaluated position, which is ~300x the work for an identical answer.
    short_table = window_model(train, 2, src.vocab)
    exact_table = window_model(train, src.lag, src.vocab)
    kl_short = np.mean([kl_bits(src.true_dist(test[:t]),
                                window_predict(short_table, test[:t], 2, src.vocab))
                        for t in range(8, 300)])
    kl_exact = np.mean([kl_bits(src.true_dist(test[:t]),
                                window_predict(exact_table, test[:t], src.lag,
                                               src.vocab))
                        for t in range(8, 300)])
    assert kl_short > 10 * kl_exact


# ------------------------------------------------------------------------- S8

def _corpus(n_words: int = 1_200, seed: int = 501) -> str:
    lex = lexicon(60, seed=seed)
    return generate_text(n_words, seed=seed + 1, lex=lex,
                         weights=zipf_weights(len(lex)))


def test_count_pairs_charges_a_repeated_pair_once_per_merge():
    """The overlap correction, on the smallest case that exposes it.

    A run of three identical symbols holds the pair at two adjacent positions
    but supports only one merge, because merging the first consumes the second.
    Getting this wrong makes the tokenizer's arithmetic silently inconsistent.
    """
    assert count_pairs([list("aaa")], [1])[("a", "a")] == 1
    assert count_pairs([list("aaaa")], [1])[("a", "a")] == 2
    assert count_pairs([list("aab")], [1])[("a", "a")] == 1
    assert count_pairs([list("abab")], [1])[("a", "b")] == 2


@pytest.mark.parametrize("word,expect", [("aaa", ["aa", "a"]),
                                         ("aaaa", ["aa", "aa"]),
                                         ("aab", ["aa", "b"]),
                                         ("bab", ["b", "a", "b"])])
def test_apply_merge_agrees_with_the_count(word, expect):
    out = apply_merge(list(word), "a", "a", "aa")
    assert out == expect
    assert len(word) - len(out) == count_pairs([list(word)], [1]).get(("a", "a"), 0)


def test_bpe_round_trips_exactly():
    text = _corpus()
    bpe = BPE(text, n_merges=80)
    assert bpe.decode(bpe.encode(text)) == text


def test_vocabulary_is_base_plus_merges_with_no_duplicates():
    bpe = BPE(_corpus(), n_merges=80)
    assert len(bpe.vocab) == len(bpe.base) + len(bpe.merges)
    assert len(set(bpe.vocab)) == len(bpe.vocab)


def test_each_merge_removes_exactly_its_pair_count():
    bpe = BPE(_corpus(), n_merges=80)
    assert bpe.predicted_tokens == bpe.actual_tokens


def test_truncating_a_merge_list_equals_training_with_fewer_merges():
    """The property the vocabulary-size sweep relies on: merges are greedy and
    ordered, so no early choice depends on how many come after it."""
    text = _corpus()
    long_run, short_run = BPE(text, n_merges=60), BPE(text, n_merges=20)
    assert long_run.truncated(20).merges == short_run.merges
    assert long_run.truncated(20).encode(text) == short_run.encode(text)


def test_more_merges_never_lengthen_the_encoding():
    text = _corpus()
    bpe = BPE(text, n_merges=80)
    lengths = [len(bpe.truncated(m).encode(text)) for m in (0, 10, 40, 80)]
    assert all(a >= b for a, b in zip(lengths, lengths[1:]))


def test_rot13_preserves_length_and_is_its_own_inverse():
    text = _corpus()
    assert rot13(rot13(text)) == text
    assert len(rot13(text)) == len(text)
    assert rot13(text) != text


def test_script_of_assigns_shared_symbols_to_neither():
    assert script_of("abc") == "A"
    assert script_of("nop") == "B"
    assert script_of(WORD_MARK + "ab") == "A"
    assert script_of(WORD_MARK + ".") == "-"
    assert script_of("an") == "-"


def test_pretokenize_marks_every_word_start():
    assert pretokenize("ab cd") == [WORD_MARK + "ab", WORD_MARK + "cd"]


# ------------------------------------------------------------------------- S9

def test_trigram_outputs_are_distributions():
    src = MarkovSource()
    lm = TrigramLM(src.sample(3_000, seed=4))
    for u, v in [(0, 0), (3, 11), (V - 1, V - 2)]:
        p = lm.dist(u, v)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p > 0).all()              # interpolation floors the support


def test_dist_tensor_matches_dist():
    src = MarkovSource()
    lm = TrigramLM(src.sample(3_000, seed=5))
    P = lm.dist_tensor()
    for u, v in [(0, 1), (7, 19), (V - 1, 0)]:
        assert np.allclose(P[u, v], lm.dist(u, v), atol=1e-12)


def test_model_is_a_pure_function():
    """S9.6's central claim, as an invariant."""
    lm = TrigramLM(MarkovSource().sample(2_000, seed=6))
    first = lm.dist(3, 4)
    for _ in range(25):
        lm.dist(int(rng(7).integers(V)), int(rng(8).integers(V)))
    assert np.array_equal(first, lm.dist(3, 4))


# ------------------------------------------------------------------------ S10

def _logP(seed: int = 9) -> np.ndarray:
    lm = TrigramLM(MarkovSource().sample(3_000, seed=seed))
    return np.log2(np.maximum(lm.dist_tensor(), np.finfo(np.float64).tiny))


def test_dynamic_program_matches_brute_force():
    logP = _logP(9)
    for u0, v0 in [(0, 1), (5, 9)]:
        _, bf = brute_force_best(logP, 3, u0, v0)
        assert float(exact_values(logP, 3)[u0, v0]) == pytest.approx(bf, abs=1e-12)


def test_greedy_never_beats_the_exact_optimum():
    logP = _logP(10)
    for T in (2, 4, 6):
        assert np.all(exact_values(logP, T) - greedy_values(logP, T) > -1e-9)


def test_truncation_renormalises():
    p = softmax(rng(12).normal(size=24))
    assert top_k(p, 5).sum() == pytest.approx(1.0)
    assert int((top_k(p, 5) > 0).sum()) == 5
    q, n = top_p(p, 0.9)
    assert q.sum() == pytest.approx(1.0)
    assert 1 <= n <= 24


def test_top_p_nucleus_is_monotone_in_threshold():
    p = softmax(rng(13).normal(size=32))
    sizes = [top_p(p, t)[1] for t in (0.5, 0.7, 0.9, 0.99)]
    assert all(a <= b for a, b in zip(sizes, sizes[1:]))


@pytest.mark.parametrize("T", [0.2, 0.75, 1.0, 3.0, 20.0])
def test_temperature_moves_mass_without_moving_rank(T):
    """S10.2, as an invariant: no temperature reorders anything."""
    p = softmax(rng(14).normal(size=24) * 3.0)
    assert np.array_equal(np.argsort(-temper(p, T)), np.argsort(-p))
    assert temper(p, T).sum() == pytest.approx(1.0)


def test_truncation_cannot_move_the_argmax():
    p = softmax(rng(15).normal(size=24) * 2.0)
    for q in (top_k(p, 1), top_k(p, 3), top_p(p, 0.5)[0], top_p(p, 0.95)[0]):
        assert int(np.argmax(q)) == int(np.argmax(p))


def test_speculative_verification_reproduces_the_target_exactly():
    """The identity holds for any draft, so the test uses a deliberately bad one."""
    g = rng(16)
    target = softmax(g.normal(size=(4, 4, 24)) * 2.0, axis=-1)
    draft = softmax(g.normal(size=(4, 4, 24)) * 0.3, axis=-1)
    assert np.max(np.abs(verified_distribution(target, draft) - target)) < 1e-14
    # and for the degenerate case where the draft is already the target
    assert np.max(np.abs(verified_distribution(target, target) - target)) < 1e-14


def test_acceptance_rate_is_one_minus_total_variation():
    g = rng(17)
    p = softmax(g.normal(size=(3, 3, 24)) * 2.0, axis=-1)
    q = softmax(g.normal(size=(3, 3, 24)), axis=-1)
    alpha = np.minimum(p, q).sum(axis=-1)
    tv = 0.5 * np.abs(p - q).sum(axis=-1)
    assert np.max(np.abs(alpha - (1.0 - tv))) < 1e-14


@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.9])
def test_constant_rate_tokens_is_the_geometric_series(alpha):
    for k in (1, 2, 5):
        assert constant_rate_tokens(alpha, k) == pytest.approx(
            (1 - alpha ** (k + 1)) / (1 - alpha) if alpha else 1.0)
    # the ceiling as the draft grows is 1 / (1 - alpha)
    if alpha:
        assert constant_rate_tokens(alpha, 200) == pytest.approx(
            1 / (1 - alpha), rel=1e-9)


def test_the_optimal_policy_really_does_cycle():
    """S10.7: a deterministic policy on finitely many states must repeat."""
    policy, last_change = stationary_policy(_logP(11), sweeps=300)
    assert last_change < 300               # it settled rather than oscillating
    for u0, v0 in [(0, 1), (7, 19), (V - 1, 3)]:
        entry, period, on_cycle = policy_cycle(policy, u0, v0)
        assert period >= 1 and len(on_cycle) == period
        # walk to the state where the repeat was detected, then `period` more
        u, v = u0, v0
        for _ in range(entry):
            u, v = v, int(policy[u, v])
        start = (u, v)
        assert start in on_cycle
        for _ in range(period):
            u, v = v, int(policy[u, v])
        assert (u, v) == start
        # and the emitted tokens repeat with that period
        emitted = policy_tokens(policy, u0, v0, entry + 3 * period)
        assert emitted[entry:entry + period] == \
            emitted[entry + period:entry + 2 * period]


def test_stationary_distribution_is_a_fixed_point():
    lm = TrigramLM(MarkovSource().sample(3_000, seed=12))
    P = lm.dist_tensor()
    pi = stationary_over_states(P)
    assert pi.sum() == pytest.approx(1.0)
    moved = np.zeros_like(pi)
    for u in range(V):
        for v in range(V):
            moved[v] += pi[u, v] * P[u, v]
    assert np.max(np.abs(moved - pi)) < 1e-9


# ------------------------------------------------------------------------- S11

def test_kv_cache_is_algebraically_exact():
    """The identity S11 rests on: appending cannot change earlier K,V."""
    cfg = Config()
    model = Transformer(cfg)
    tokens = rng(14).integers(0, cfg.vocab, size=30)
    full, _ = model.forward(tokens)
    _, cache = model.forward(tokens[:20])
    out = []
    for t in range(20, 30):
        logit, cache = model.step(int(tokens[t]), t, cache)
        out.append(logit)
    assert np.max(np.abs(full[20:] - np.array(out))) < 1e-12


def test_mac_formulas_match_instrumentation():
    cfg = Config()
    model = Transformer(cfg)
    tokens = rng(15).integers(0, cfg.vocab, size=25)
    c = Counter()
    _, cache = model.forward(tokens, c)
    assert c.macs == macs_prefill(cfg, 25)
    c2 = Counter()
    model.step(int(tokens[0]), 25, cache, c2)
    assert c2.macs == macs_step(cfg, 25)


def test_attention_share_crosses_half_at_2d_plus_dff():
    cfg = Config()
    assert attention_share(cfg, 2 * cfg.d_model + cfg.d_ff) == pytest.approx(0.5)
    assert attention_share(cfg, 1) < 0.5
    assert attention_share(cfg, 10 ** 7) > 0.99


def test_lossy_kv_variants_actually_change_the_output():
    cfg = Config()
    model = Transformer(cfg)
    tokens = rng(16).integers(0, cfg.vocab, size=40)
    _, cache = model.forward(tokens[:35])
    ref, _ = model.step(int(tokens[35]), 35, [(k.copy(), v.copy())
                                              for k, v in cache])
    for fn in (int8_kv, sliding_window(8)):
        out, _ = model.step(int(tokens[35]), 35,
                            [(k.copy(), v.copy()) for k, v in cache],
                            kv_transform=fn)
        assert np.max(np.abs(out - ref)) > 0.0


def test_gqa_reduces_kv_by_the_head_ratio():
    mha, gqa = SPECS[0], SPECS[1]
    assert (mha.kv_bytes_per_token() / gqa.kv_bytes_per_token()
            == pytest.approx(mha.n_heads / gqa.n_kv_heads))


# ------------------------------------------------------------------------ S12

def test_decode_intensity_approaches_its_ceiling():
    spec, ctx = SPECS[1], 2_048
    ceiling = decode_intensity_ceiling(spec, ctx)
    assert decode_intensity(spec, 10 ** 7, ctx) == pytest.approx(ceiling, rel=1e-3)
    assert decode_intensity(spec, 1, ctx) < ceiling


def test_decode_intensity_ceiling_is_tstar_over_context():
    spec = SPECS[1]
    for ctx in (256, 2_048, 32_768):
        assert (decode_intensity_ceiling(spec, ctx)
                == pytest.approx(spec.crossover_tokens(1) / ctx, rel=1e-9))


def test_device_balance_and_roofline():
    d = DEVICES[0]
    assert d.attainable_tflops(d.balance * 10) == pytest.approx(d.tflops)
    assert d.attainable_tflops(d.balance / 100) < d.tflops / 50


def test_staleness_limits_and_simulation():
    assert staleness_probability(60, 0.0) == pytest.approx(0.0)
    assert staleness_probability(1e6, 1.0) > 0.999
    # small rT  =>  approximately rT/2
    assert staleness_probability(10, 0.001) == pytest.approx(0.005, rel=0.02)
    an = staleness_probability(120, 0.004)
    assert simulate_staleness(120, 0.004, n=80_000) == pytest.approx(an, abs=0.01)


def test_mm1_diverges_at_saturation_with_constant_ratio():
    p50a, p95a, _ = mm1_percentiles(1.0, 4.0)
    p50b, p95b, _ = mm1_percentiles(3.8, 4.0)
    assert p50b > 5 * p50a
    assert p95a / p50a == pytest.approx(p95b / p50b)
    assert mm1_percentiles(4.0, 4.0)[0] == float("inf")
