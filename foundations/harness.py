"""Shared harness for the foundations sections.

Every section is an experiment, not an illustration. That imposes three
requirements which this module exists to satisfy.

**A claim must be checked, not asserted.** `Section.check` takes a sentence in
English and a boolean computed from the run. It prints the verdict and records
it. `run_all.py` aggregates every claim in the repo into `logs/claims.json`, so
the statements in the writeup are not prose the reader has to trust: each one
carries the line of code that decided it. A claim that fails still gets printed
and recorded, and the exit code goes non-zero. Nothing is quietly dropped.

**A number must be reproducible.** Every source of randomness comes from
`rng(seed)`. No module reads the wall clock for anything that feeds a result,
and no module touches the network. Two runs on two machines produce the same
figures.

**A result must be legible without the code.** Sections write a plain-text log
to `logs/`. That log is the artifact quoted in the writeup, so it has to stand
on its own: units on every number, the condition next to the measurement, and
the arithmetic shown at the size where a reader can check it by hand.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"

RULE = "=" * 78
THIN = "-" * 78


def rng(seed: int) -> np.random.Generator:
    """The only entry point for randomness in this repo.

    `default_rng` is a counter-based PCG64 generator, so a given seed yields the
    same stream on any platform and NumPy version. `np.random.seed` and the
    legacy global state are deliberately never used: they are process-global,
    which makes one module's sampling depend on whether another module ran
    first, and that is exactly the kind of coupling that turns a reproducible
    result into an anecdote.
    """
    return np.random.default_rng(seed)


@dataclass
class Claim:
    section: str
    text: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"section": self.section, "claim": self.text,
                "ok": self.ok, "detail": self.detail}


@dataclass
class Section:
    """A single numbered section: logs to stdout and to `logs/<sid>.log`."""

    sid: str
    title: str
    subtitle: str = ""
    claims: list[Claim] = field(default_factory=list)
    _lines: list[str] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    # ------------------------------------------------------------------ output

    def _emit(self, line: str = "") -> None:
        self._lines.append(line)
        print(line, flush=True)

    def open(self) -> "Section":
        self._emit(RULE)
        self._emit(f"{self.sid}  {self.title}")
        if self.subtitle:
            self._emit(f"      {self.subtitle}")
        self._emit(RULE)
        return self

    def __enter__(self) -> "Section":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        # An exception inside a section is a failed experiment, and a failed
        # experiment is a result. It gets recorded as a failed claim so the
        # aggregate count stays honest, then re-raised so the run goes red.
        if exc_type is not None:
            self.claims.append(Claim(self.sid, f"section {self.sid} ran to completion",
                                     False, f"{exc_type.__name__}: {exc}"))
        self.close()
        return False

    def h(self, text: str) -> None:
        """Sub-heading within a section."""
        self._emit()
        self._emit(text)
        self._emit(THIN)

    def say(self, text: str = "") -> None:
        for line in (text.split("\n") if text else [""]):
            self._emit(line)

    def note(self, text: str) -> None:
        """Interpretation, marked so it is never confused with a measurement."""
        self._emit(f"  # {text}")

    def kv(self, key: str, value: Any, unit: str = "", width: int = 34) -> None:
        v = f"{value:.6g}" if isinstance(value, float) else str(value)
        self._emit(f"  {key:<{width}} {v}{(' ' + unit) if unit else ''}")

    def eq(self, latex: str, gloss: str = "") -> None:
        """Record the equation a result depends on, in the log next to it.

        The log is what gets quoted, and an equation that lives only in the
        writeup can drift from the code that implements it. Keeping it here
        means the two are edited together.
        """
        self._emit(f"  eq:  {latex}")
        if gloss:
            self._emit(f"       {gloss}")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[Any]],
              align: str = "") -> None:
        rows = [[f"{c:.6g}" if isinstance(c, float) else str(c) for c in r]
                for r in rows]
        widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows
                  else len(str(h)) for i, h in enumerate(headers)]
        align = (align + "l" * len(headers))[:len(headers)]

        def fmt(cells: Sequence[str]) -> str:
            out = []
            for i, c in enumerate(cells):
                out.append(c.rjust(widths[i]) if align[i] == "r"
                           else c.ljust(widths[i]))
            return "  " + "  ".join(out).rstrip()

        self._emit(fmt([str(h) for h in headers]))
        self._emit("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            self._emit(fmt(r))

    # ------------------------------------------------------------------ claims

    def check(self, text: str, ok: bool, detail: str = "") -> bool:
        """Record a claim and its verdict. This is the unit of evidence."""
        ok = bool(ok)
        self.claims.append(Claim(self.sid, text, ok, detail))
        mark = "PASS" if ok else "FAIL"
        self._emit(f"  [{mark}] {text}")
        if detail:
            self._emit(f"         {detail}")
        return ok

    # ------------------------------------------------------------------- close

    def close(self) -> None:
        elapsed = time.perf_counter() - self._t0
        n_ok = sum(c.ok for c in self.claims)
        self._emit()
        self._emit(f"{self.sid} claims {n_ok}/{len(self.claims)} passed "
                   f"in {elapsed:.2f}s")
        self._emit(RULE)
        self._emit()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{self.sid}.log").write_text(
            "\n".join(self._lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- helpers

def softmax(z: np.ndarray, temperature: float = 1.0,
            axis: int = -1) -> np.ndarray:
    """Numerically stable softmax with temperature.

        p_i = exp(z_i / T) / sum_j exp(z_j / T)

    The max is subtracted before exponentiating. This changes nothing
    mathematically, because a constant shift cancels between numerator and
    denominator, and it is the difference between a working function and
    `inf/inf` once any logit exceeds about 710 in float64.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use argmax for greedy")
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy_bits(p: np.ndarray) -> float:
    """H(p) = -sum p_i log2 p_i, in bits.

    Terms with p_i = 0 contribute 0 by the limit x log x -> 0, which has to be
    imposed explicitly because the floating-point evaluation is nan.
    """
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p || q) = sum p_i log2(p_i / q_i), in bits.

    Divergence is infinite wherever p has mass and q has none, so q is floored
    at the smallest normal double rather than silently skipping those terms.
    Skipping them would report a finite number for a model that assigns zero
    probability to something that happens, which is the single most misleading
    thing this function could do.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.maximum(np.asarray(q, dtype=np.float64), np.finfo(np.float64).tiny)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))


def perplexity(nll_bits_per_token: float) -> float:
    """Perplexity is the exponentiated cross-entropy: PP = 2^H for H in bits.

    Read it as an effective branching factor: a perplexity of 8 means the model
    is as uncertain as it would be choosing uniformly among 8 options.
    """
    return float(2.0 ** nll_bits_per_token)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, so p50 and p95 mean one fixed thing."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def environment() -> dict:
    return {"python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine()}


def write_claims(sections: Sequence[Section], path: Path | None = None) -> dict:
    """Merge this run's claims into the ledger, keeping sections that did not run.

    The merge is the whole point and it was a bug before. Writing only what this
    run produced means `run_all.py --article 2` deletes Article 3's evidence and
    vice versa, so the ledger silently holds whichever article was run last while
    still looking complete. Since the articles cite this file as the record that
    every quoted number was measured, a ledger that quietly drops two thirds of
    itself is worse than no ledger.

    A section that runs replaces its own entry entirely, so a claim that is
    deleted from a module disappears on the next run rather than lingering.
    Sections that did not run are carried through untouched, with the provenance
    block saying when each was last produced and on what. The timestamp is
    provenance and never feeds a result, which is why it is the one place in the
    repo that reads the clock.
    """
    path = path or (LOG_DIR / "claims.json")

    previous: dict = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}

    by_section: dict[str, list[dict]] = {}
    for claim in previous.get("claims", []):
        by_section.setdefault(claim["section"], []).append(claim)
    provenance: dict = dict(previous.get("produced", {}))

    env = environment()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for section in sections:
        by_section[section.sid] = [c.as_dict() for c in section.claims]
        provenance[section.sid] = {"at": stamp, "environment": env}

    claims = [c for sid in sorted(by_section) for c in by_section[sid]]
    payload = {"environment": env,
               "produced": provenance,
               "sections": sorted(by_section),
               "ran_this_time": sorted(s.sid for s in sections),
               "total": len(claims),
               "passed": sum(c["ok"] for c in claims),
               "failed": [c for c in claims if not c["ok"]],
               "claims": claims}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
