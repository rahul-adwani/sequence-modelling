"""Invariants of the Article 2 sections: S02 through S07.

These test properties of the machinery, not the prose. They exercise the
underlying functions directly rather than re-running the sections, so the suite
finishes in seconds and a failure points at a function.

The distinction worth preserving: `run_all.py` checks that the article's *claims*
hold on this run. These check that the code computing them is not quietly broken
in ways a claim might survive. A gradient routine can agree with itself and be
wrong; agreeing with finite differences is a different question.
"""
from __future__ import annotations

import numpy as np
import pytest

from foundations.harness import entropy_bits, rng, softmax
from foundations.s02_recurrence import RNN, sequence
from foundations.s03_vanishing import (clip, decay_profile, fitted_ratio,
                                       horizon, log_linear_r2)
from foundations.s04_gating import GRU, LSTM
from foundations.s05_bottleneck import (VOCAB, capacity_tokens, encode,
                                        recovery_rate)
from foundations.s06_attention import attend, attention_recovery, encoder_kv, query_for
from foundations.s07_transformer import (causal_mask, path_length,
                                         self_attention, sequential_ops,
                                         sinusoidal_positions)


# ------------------------------------------------------------------------ S02

def test_parameters_do_not_depend_on_sequence_length():
    net = RNN()
    n = net.n_params
    for T in (3, 40, 300):
        states, ys = net.forward(sequence(T, net.d_in, seed=1))
        assert len(states) == T + 1
        assert ys[-1].shape == (net.d_out,)
    assert net.n_params == n


def test_bptt_matches_finite_differences():
    """The load-bearing check of section 2."""
    net = RNN()
    xs, target = sequence(12, net.d_in, seed=2), rng(3).normal(size=net.d_out)
    dW, _, _, _ = net.bptt(xs, target)
    fd = net.finite_difference_W(xs, target)
    rel = np.linalg.norm(dW - fd) / (np.linalg.norm(dW) + np.linalg.norm(fd))
    assert rel < 1e-8


def test_per_step_contributions_sum_to_the_gradient():
    net = RNN()
    xs, target = sequence(10, net.d_in, seed=4), rng(5).normal(size=net.d_out)
    dW, _, _, per_k = net.bptt(xs, target)
    assert np.allclose(sum(per_k), dW, atol=1e-14)


def test_first_step_contribution_is_structurally_zero():
    """s_0 is the zero vector, so w had nothing to act on at step 1."""
    net = RNN()
    xs, target = sequence(8, net.d_in, seed=6), rng(7).normal(size=net.d_out)
    _, _, _, per_k = net.bptt(xs, target)
    assert np.allclose(per_k[0], 0.0)


def test_sigma_max_rescaling_hits_the_requested_value():
    """The parameter sets the largest SINGULAR value, not the spectral radius."""
    for target in (0.5, 1.0, 3.0):
        net = RNN(d_hidden=16, sigma_max=target)
        assert float(np.linalg.norm(net.W, 2)) == pytest.approx(target, rel=1e-9)
        # and the spectral radius is a different, never-larger number
        rho = float(np.max(np.abs(np.linalg.eigvals(net.W))))
        assert rho <= target + 1e-9


@pytest.mark.parametrize("act,probe,expected", [("tanh", 0.5, 0.75),
                                                ("relu", 0.5, 1.0),
                                                ("relu", 0.0, 0.0)])
def test_activation_derivative(act, probe, expected):
    net = RNN(activation=act)
    got = float(net.fprime(np.array([probe]))[0])
    assert got == pytest.approx(expected)


# ------------------------------------------------------------------------ S03

def test_gradient_decay_is_monotone_in_distance():
    prof = decay_profile(0.9)
    assert prof[0] == pytest.approx(1.0)
    assert all(prof[i] >= prof[i + 1] for i in range(1, 40))


def test_smaller_sigma_max_decays_faster():
    assert fitted_ratio(decay_profile(0.5)) < fitted_ratio(decay_profile(2.0))


def test_decay_is_log_linear_in_the_contracting_regime():
    for sigma in (0.5, 1.0, 2.0):
        assert log_linear_r2(decay_profile(sigma)) > 0.99


def test_horizon_formula():
    assert horizon(0.5, 1e-6) == pytest.approx(np.log(1e-6) / np.log(0.5))
    assert horizon(1.0) == float("inf")
    assert horizon(1.5) == float("inf")


def test_clipping_is_a_ceiling_not_a_floor():
    big, small = np.array([50.0]), np.array([1e-24])
    assert float(np.linalg.norm(clip(big, 1.0))) == pytest.approx(1.0)
    assert float(clip(small, 1.0)[0]) == small[0]     # untouched


def test_relu_reaches_explosion_where_tanh_does_not():
    assert decay_profile(3.0, activation="relu")[40] > decay_profile(3.0)[40]


# ------------------------------------------------------------------------ S04

def test_gates_are_probabilities_and_candidate_is_bounded():
    net = LSTM()
    _, _, gs = net.forward(sequence(20, net.d_in, seed=8))
    for g in gs:
        for k in ("i", "f", "o"):
            assert np.all((g[k] > 0) & (g[k] < 1))
        assert np.all(np.abs(g["g"]) <= 1.0)


def test_cell_jacobian_is_exactly_diag_of_forget_gate():
    """Section 4's central claim, holding h_{t-1} fixed."""
    net = LSTM()
    xs = sequence(20, net.d_in, seed=9)
    hs, cs, _ = net.forward(xs)
    J = net.numeric_cell_jacobian(hs[10], cs[10], xs[10])
    f = net.gates(hs[10], xs[10])["f"]
    assert np.max(np.abs(J - np.diag(f))) < 1e-8
    off = J - np.diag(np.diag(J))
    assert np.max(np.abs(off)) < 1e-10          # units do not mix on this path


def test_forget_bias_raises_the_gate_and_slows_the_decay():
    xs = sequence(50, LSTM().d_in, seed=10)
    biased = LSTM(forget_bias=1.0).highway_decay(xs)
    unbiased = LSTM(forget_bias=0.0).highway_decay(xs)
    assert biased[40] > unbiased[40]


def test_gated_paths_beat_the_ungated_network():
    xs = sequence(50, 4, seed=11)
    vanilla = RNN(d_in=4, d_hidden=16, d_out=3, sigma_max=0.95, seed=4)
    v = np.asarray(vanilla.state_gradient_norms(xs, rng(12).normal(size=3)))
    v = v / v[0]
    assert LSTM().highway_decay(xs)[40] > 1e6 * v[40]
    assert GRU().carry_decay(xs)[40] > 1e6 * v[40]


# ------------------------------------------------------------------------ S05

def test_capacity_bound_is_linear_in_width():
    assert capacity_tokens(64) == pytest.approx(2 * capacity_tokens(32))


def test_encoding_is_deterministic_and_fixed_width():
    syms = rng(13).integers(0, VOCAB, size=30)
    a, b = encode(syms, 32), encode(syms, 32)
    assert np.array_equal(a, b)
    assert a.shape == (32,)
    assert encode(rng(14).integers(0, VOCAB, size=90), 32).shape == (32,)


def test_recall_falls_with_length_and_rises_with_width():
    assert recovery_rate(4, 32) > recovery_rate(64, 32)
    assert recovery_rate(32, 128) > recovery_rate(32, 16)


# ------------------------------------------------------------------------ S06

def test_attention_weights_are_a_distribution():
    syms = rng(15).integers(0, VOCAB, size=12)
    K, V = encoder_kv(syms)
    w, ctx = attend(K, V, query_for(5, 12))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert ctx.shape == (V.shape[1],)


def test_attention_keeps_recall_high_where_a_single_context_does_not():
    rec, align = attention_recovery(32, trials=8)
    assert rec > 0.9
    assert align > 0.9
    assert rec > recovery_rate(32, 32)


def test_attention_weights_are_concentrated():
    K, V = encoder_kv(rng(16).integers(0, VOCAB, size=16))
    w, _ = attend(K, V, query_for(7, 16))
    assert int(np.argmax(w)) == 7
    assert entropy_bits(w) < np.log2(16) / 2


# ------------------------------------------------------------------------ S07

def _blocks(d=32, seed=17):
    g = rng(seed)
    return (g.normal(0, 1.0, (8, d)),
            *(g.normal(0, 1 / np.sqrt(d), (d, d)) for _ in range(3)))


def test_self_attention_rows_are_distributions():
    X, Wq, Wk, Wv = _blocks()
    W, out = self_attention(X, Wq, Wk, Wv)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.all(W >= 0)
    assert out.shape == X.shape


def test_self_attention_is_permutation_equivariant():
    """Without positional information it sees a set, not a sequence."""
    X, Wq, Wk, Wv = _blocks()
    perm = rng(18).permutation(len(X))
    _, out = self_attention(X, Wq, Wk, Wv)
    _, out_p = self_attention(X[perm], Wq, Wk, Wv)
    assert np.max(np.abs(out[perm] - out_p)) < 1e-12


def test_positional_encoding_breaks_that_symmetry():
    X, Wq, Wk, Wv = _blocks()
    PE = sinusoidal_positions(len(X), X.shape[1])
    perm = rng(19).permutation(len(X))
    _, out = self_attention(X + PE, Wq, Wk, Wv)
    _, out_p = self_attention(X[perm] + PE, Wq, Wk, Wv)
    assert np.max(np.abs(out[perm] - out_p)) > 1e-3


def test_positional_encodings_are_distinct_and_bounded():
    PE = sinusoidal_positions(64, 32)
    assert np.all(np.abs(PE) <= 1.0)
    for i in range(0, 64, 9):
        for j in range(0, 64, 9):
            if i != j:
                assert np.linalg.norm(PE[i] - PE[j]) > 1e-6


def test_causal_mask_forbids_the_future_exactly():
    X, Wq, Wk, Wv = _blocks()
    W, _ = self_attention(X, Wq, Wk, Wv, mask=causal_mask(len(X)))
    assert np.max(np.abs(np.triu(W, k=1))) == 0.0
    for i in range(len(X)):
        assert int(np.sum(W[i] > 0)) == i + 1
    assert np.allclose(W.sum(axis=1), 1.0)      # renormalised over what remains


def test_path_length_and_sequential_ops():
    for T in (10, 1_000):
        assert path_length("self-attention", T) == 1
        assert sequential_ops("self-attention", T) == 1
        assert path_length("recurrent", T) == T - 1
        assert sequential_ops("recurrent", T) == T
