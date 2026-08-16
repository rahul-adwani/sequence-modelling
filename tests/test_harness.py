"""Invariants of the shared harness.

These test properties, not prose. A section's `check` calls assert things about
the agent stack; these assert that the tools used to measure it are sound. If
`softmax` or `kl_bits` were wrong, every section would agree with itself and be
wrong together, which is the failure mode a claim ledger cannot catch on its own.
"""
from __future__ import annotations

import numpy as np
import pytest

from foundations.harness import (Section, entropy_bits, kl_bits, percentile,
                                 perplexity, rng, softmax)


def test_softmax_normalises():
    for shape in [(5,), (3, 7), (2, 3, 4)]:
        z = rng(1).normal(size=shape)
        p = softmax(z)
        assert np.allclose(p.sum(axis=-1), 1.0)
        assert np.all(p > 0)


def test_softmax_shift_invariant():
    """Adding a constant to every logit must not change the distribution."""
    z = rng(2).normal(size=9)
    assert np.allclose(softmax(z), softmax(z + 37.5))


def test_softmax_survives_extreme_logits():
    """The max-subtraction is the reason this does not overflow to nan."""
    p = softmax(np.array([1000.0, 1001.0, -1000.0]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(), 1.0)
    assert p.argmax() == 1


def test_softmax_temperature_monotone_in_entropy():
    z = rng(3).normal(size=12)
    ents = [entropy_bits(softmax(z, temperature=t))
            for t in (0.25, 0.5, 1.0, 4.0)]
    assert all(a < b for a, b in zip(ents, ents[1:]))


def test_softmax_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        softmax(np.zeros(3), temperature=0.0)


def test_entropy_bounds():
    """0 <= H(p) <= log2 K, with both ends attained."""
    K = 8
    assert entropy_bits(np.eye(K)[0]) == pytest.approx(0.0)
    assert entropy_bits(np.full(K, 1 / K)) == pytest.approx(np.log2(K))
    p = softmax(rng(4).normal(size=K))
    assert 0.0 <= entropy_bits(p) <= np.log2(K)


def test_entropy_handles_zeros():
    """x log x -> 0 has to be imposed; the naive evaluation is nan."""
    p = np.array([0.5, 0.5, 0.0, 0.0])
    assert np.isfinite(entropy_bits(p))
    assert entropy_bits(p) == pytest.approx(1.0)


def test_kl_is_zero_iff_identical():
    p = softmax(rng(5).normal(size=10))
    assert kl_bits(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_non_negative():
    """Gibbs' inequality, over many random pairs."""
    g = rng(6)
    for _ in range(200):
        p = softmax(g.normal(size=6))
        q = softmax(g.normal(size=6))
        assert kl_bits(p, q) >= -1e-12


def test_kl_is_asymmetric():
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.3, 0.4, 0.3])
    assert kl_bits(p, q) != pytest.approx(kl_bits(q, p))


def test_kl_finite_when_q_has_zero_mass():
    """q is floored rather than the term being skipped: the answer should be
    enormous, not quietly small."""
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    d = kl_bits(p, q)
    assert np.isfinite(d) and d > 100


def test_perplexity_matches_uniform():
    for k in (2, 40, 50_257):
        assert perplexity(np.log2(k)) == pytest.approx(k)


def test_percentile_is_ordered():
    v = rng(7).normal(size=5_000)
    assert percentile(v, 50) < percentile(v, 95) < percentile(v, 99)


def test_rng_is_reproducible_and_seed_dependent():
    assert np.array_equal(rng(11).normal(size=50), rng(11).normal(size=50))
    assert not np.array_equal(rng(11).normal(size=50), rng(12).normal(size=50))


def test_section_records_claims_both_ways(tmp_path, monkeypatch):
    import foundations.harness as H
    monkeypatch.setattr(H, "LOG_DIR", tmp_path)
    s = H.Section("stest", "unit test")
    s.open()
    s.check("a true thing", True, "detail")
    s.check("a false thing", False)
    s.close()
    assert [c.ok for c in s.claims] == [True, False]
    assert (tmp_path / "stest.log").exists()
    assert "[FAIL]" in (tmp_path / "stest.log").read_text(encoding="utf-8")


def test_the_ledger_keeps_sections_that_did_not_run(tmp_path):
    """Running one article must not delete another article's evidence.

    This was a real bug: `run_all.py --article 2` overwrote the ledger with only
    Part 2's claims, so the file looked complete while holding a third of itself.
    """
    from foundations.harness import Claim, Section, write_claims

    ledger = tmp_path / "claims.json"

    a = Section("sA", "first")
    a.claims = [Claim("sA", "a holds", True, "measured"),
                Claim("sA", "a also holds", True, "")]
    first = write_claims([a], ledger)
    assert first["total"] == 2

    b = Section("sB", "second")
    b.claims = [Claim("sB", "b holds", True, "")]
    second = write_claims([b], ledger)

    assert second["total"] == 3                       # sA carried through
    assert second["sections"] == ["sA", "sB"]
    assert second["ran_this_time"] == ["sB"]
    assert set(second["produced"]) == {"sA", "sB"}


def test_rerunning_a_section_replaces_rather_than_appends(tmp_path):
    """A claim deleted from a module must leave the ledger on the next run."""
    from foundations.harness import Claim, Section, write_claims

    ledger = tmp_path / "claims.json"

    before = Section("sA", "first")
    before.claims = [Claim("sA", "kept", True, ""), Claim("sA", "removed", True, "")]
    write_claims([before], ledger)

    after = Section("sA", "first")
    after.claims = [Claim("sA", "kept", True, "")]
    payload = write_claims([after], ledger)

    assert payload["total"] == 1
    assert [c["claim"] for c in payload["claims"]] == ["kept"]


def test_a_failure_anywhere_in_the_ledger_is_listed(tmp_path):
    from foundations.harness import Claim, Section, write_claims

    ledger = tmp_path / "claims.json"
    bad = Section("sA", "first")
    bad.claims = [Claim("sA", "does not hold", False, "0.7 > 0.5")]
    write_claims([bad], ledger)

    good = Section("sB", "second")
    good.claims = [Claim("sB", "holds", True, "")]
    payload = write_claims([good], ledger)

    assert payload["passed"] == 1 and payload["total"] == 2
    assert [c["claim"] for c in payload["failed"]] == ["does not hold"]
