"""S13. The loop: what actually turns a model into an agent.

S9 left us with a stateless function `f: V* -> Delta(V)`. An agent is that
function placed inside a loop that can act on the world:

    a_t ~ pi(. | c_t)                 the policy proposes an action
    s_{t+1} = T(s_t, a_t)             the world changes
    o_t     = O(s_t, a_t)             an observation comes back
    c_{t+1} = c_t (+) (a_t, o_t)      the context grows, append-only
    tau     = min{ t : a_t = stop  or  t = N }

Read the list and notice how little of it the model supplies. The model supplies
`pi`. The action space A, the transition T, the observation function O, the
append rule, the ceiling N and the stopping rule are all code someone wrote.
"Agent" is a property of that assembly, not of the model inside it -- and the
degenerate case proves it: fix A = {stop} and N = 1 and the loop reduces exactly
to one LLM call, which 3.2 checks numerically.

The measurements here are exact rather than sampled. The task's state space is
small enough to enumerate every reachable trajectory with its probability, so
E[tau], P(success), the ceiling-hit rate and the entropy of the trajectory
distribution are computed in closed form, with no Monte Carlo error to argue
about. 3.7 samples as well, purely to confirm the enumeration.

A note on the policy, because it matters for how the results should be read.
`pi` here is a softmax over the action set with a tunable competence parameter,
not a language model. That is deliberate. Every claim in this section is about
the control structure -- when it terminates, how much it can wander, what a
repeat costs -- and those are properties of the loop as a function of policy
quality. One LLM at one fixed competence cannot show a trend across policy
quality; a parameterised policy can, and it is the same mathematical object as
S9's next-token distribution with the vocabulary replaced by an action set.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .harness import Section, entropy_bits, rng, softmax
from .s11_kv_cache import Config, macs_prefill

# ------------------------------------------------------------------ environment

# A deliberately abstract task with one property that matters: the answer is
# reachable only through indirection. "index" holds the name of the record that
# holds the value, so no single tool call can succeed and the agent must use the
# output of step one as the input to step two. That dependency is the minimal
# thing that makes a loop necessary rather than decorative.
STORE = {
    "index": "record_7",
    "record_3": "value_11",
    "record_7": "value_42",
    "record_9": "value_88",
}
ANSWER = "value_42"

ACTIONS = ["keys", "get:index", "get:record_3", "get:record_7", "get:record_9",
           "stop"]
A = len(ACTIONS)
STOP = ACTIONS.index("stop")


@dataclass(frozen=True)
class State:
    """Everything the world and the transcript know, as a hashable value.

    Frozen and hashable so that identical states from different action orders
    collapse to the same key, which is what makes the exact enumeration in
    `enumerate_trajectories` tractable rather than exponential in practice.
    """

    fetched: frozenset = frozenset()      # keys successfully read
    known: frozenset = frozenset()        # values now in context
    calls: int = 0
    wasted: int = 0                       # calls that returned nothing new

    def has_answer(self) -> bool:
        return ANSWER in self.known


def transition(st: State, action: int) -> tuple[State, bool]:
    """Apply an action. Returns the new state and whether anything was learned.

    Deterministic and idempotent: `get` on the same key always returns the same
    value, and a second call adds nothing. 3.5 depends on that being true, and
    3.6 depends on it being false for a different kind of tool.
    """
    name = ACTIONS[action]
    if name == "stop":
        return st, False
    if name == "keys":
        new = State(st.fetched, st.known | {"__keys__"}, st.calls + 1,
                    st.wasted + (1 if "__keys__" in st.known else 0))
        return new, "__keys__" not in st.known

    key = name.split(":", 1)[1]
    # The indirection: record_7 is only nameable once index has been read. A
    # policy that guesses it early is guessing, and the environment says so.
    if key.startswith("record") and "index" not in st.fetched \
            and key == STORE["index"]:
        reachable = False
    else:
        reachable = True

    if not reachable or key in st.fetched:
        return State(st.fetched, st.known, st.calls + 1, st.wasted + 1), False
    return (State(st.fetched | {key}, st.known | {STORE[key]}, st.calls + 1,
                  st.wasted), True)


# ---------------------------------------------------------------------- policy

@dataclass
class SoftmaxPolicy:
    """pi(a | state) as a softmax over action logits: S9's object, new vocabulary.

    `skill` is the logit advantage given to the action an oracle would take from
    this state. skill = 0 is a uniform random policy; large skill approaches a
    perfect one. `stop_bias` is an additive logit on the stop action, which is
    how eagerness to terminate is separated from competence -- they are different
    failure modes and conflating them is why "the agent rambles" and "the agent
    gives up early" get the same unhelpful diagnosis.
    """

    skill: float = 3.0
    stop_bias: float = 0.0
    temperature: float = 1.0
    allowed: tuple[int, ...] = tuple(range(A))

    def oracle_action(self, st: State) -> int:
        if "index" not in st.fetched:
            return ACTIONS.index("get:index")
        if not st.has_answer():
            return ACTIONS.index("get:record_7")
        return STOP

    def dist(self, st: State) -> np.ndarray:
        z = np.zeros(A)
        z[self.oracle_action(st)] += self.skill
        z[STOP] += self.stop_bias
        # Actions outside the permitted set get -inf, which softmax sends to
        # exactly zero probability. This is the same mechanism as S5's grammar
        # mask: the way to make an action impossible is to remove its mass
        # before sampling, not to ask the model nicely.
        mask = np.full(A, -np.inf)
        mask[list(self.allowed)] = 0.0
        return softmax(z + mask, temperature=self.temperature)


# ------------------------------------------------------- exact trajectory algebra

@dataclass
class LoopResult:
    p_success: float = 0.0
    p_ceiling: float = 0.0
    e_steps: float = 0.0
    e_calls: float = 0.0
    e_wasted: float = 0.0
    h_traj: float = 0.0
    n_traj: int = 0
    tau_pmf: dict = field(default_factory=dict)


def enumerate_trajectories(policy: SoftmaxPolicy, max_steps: int = 6,
                           loop_detect: bool = False,
                           prune: float = 1e-12) -> LoopResult:
    """Every reachable trajectory with its exact probability.

    Depth-first over the action tree, carrying the path probability. The result
    is the true trajectory distribution of this policy under this loop, so every
    statistic below is a closed-form expectation rather than an estimate. Paths
    whose probability falls below `prune` are dropped; the residual mass is
    reported so the reader can see it is negligible rather than take it on faith.

    This is only possible because the state is small. It is worth naming why a
    real agent cannot be analysed this way: its state is the whole transcript, so
    no two paths ever coincide, the tree never folds into a graph, and the number
    of trajectories is |A|^N with nothing to prune. Sampling is not laziness
    there, it is the only option -- which is exactly why S12 has to care about
    confidence intervals.
    """
    res = LoopResult()
    tau_pmf: dict[int, float] = {}
    total = 0.0

    # The policy depends only on which keys have been fetched and what is known,
    # never on the step counters, so its distribution is memoised on those two
    # fields. At depth 8 the walk visits nearly half a million nodes and there
    # are a handful of distinct policy states among them.
    dist_cache: dict[tuple[frozenset, frozenset], np.ndarray] = {}

    def dist_of(st: State) -> np.ndarray:
        key = (st.fetched, st.known)
        if key not in dist_cache:
            dist_cache[key] = policy.dist(st)
        return dist_cache[key]

    def walk(st: State, step: int, logp: float, done: frozenset,
             prob: float) -> None:
        nonlocal total
        if prob < prune:
            return
        if step > max_steps:
            # Ceiling reached without the policy choosing to stop.
            res.p_ceiling += prob
            tau_pmf[max_steps] = tau_pmf.get(max_steps, 0.0) + prob
            res.e_steps += prob * max_steps
            res.e_calls += prob * st.calls
            res.e_wasted += prob * st.wasted
            res.p_success += prob * st.has_answer()
            res.n_traj += 1
            total += prob
            res.h_traj += -prob * np.log2(prob)
            return

        p = dist_of(st)
        for a in range(A):
            if p[a] <= 0.0:
                continue
            branch = prob * float(p[a])
            if a == STOP:
                tau_pmf[step] = tau_pmf.get(step, 0.0) + branch
                res.e_steps += branch * step
                res.e_calls += branch * st.calls
                res.e_wasted += branch * st.wasted
                res.p_success += branch * st.has_answer()
                res.n_traj += 1
                total += branch
                res.h_traj += -branch * np.log2(branch) if branch > 0 else 0.0
                continue

            nxt, learned = transition(st, a)
            if loop_detect and not learned and a in done:
                # The control: an action that already ran and taught us nothing
                # is treated as a request to terminate. Left alone, a policy that
                # re-issues it will spend the entire budget doing so.
                tau_pmf[step] = tau_pmf.get(step, 0.0) + branch
                res.e_steps += branch * step
                res.e_calls += branch * nxt.calls
                res.e_wasted += branch * nxt.wasted
                res.p_success += branch * nxt.has_answer()
                res.n_traj += 1
                total += branch
                res.h_traj += -branch * np.log2(branch) if branch > 0 else 0.0
                continue

            walk(nxt, step + 1, logp, done | {a}, branch)

    walk(State(), 1, 0.0, frozenset(), 1.0)
    res.tau_pmf = {k: tau_pmf[k] for k in sorted(tau_pmf)}
    res.residual = 1.0 - total
    return res


def sample_trajectories(policy: SoftmaxPolicy, n: int, max_steps: int = 6,
                        loop_detect: bool = False, seed: int = 909
                        ) -> tuple[float, float, float]:
    """Monte Carlo, used only to confirm the enumeration in 3.7."""
    g = rng(seed)
    succ = steps = ceil = 0
    # The policy is a pure function of the state (S9.6 again), and the same few
    # states recur across tens of thousands of runs, so the distribution is
    # memoised. The key is (fetched, known) rather than the whole State: `calls`
    # and `wasted` are bookkeeping counters that the policy never reads, and
    # including them would make almost every lookup a miss -- which is exactly
    # the bug this comment exists to prevent someone reintroducing.
    dist_cache: dict[tuple[frozenset, frozenset], np.ndarray] = {}

    def dist(st: State) -> np.ndarray:
        key = (st.fetched, st.known)
        if key not in dist_cache:
            dist_cache[key] = policy.dist(st)
        return dist_cache[key]

    for _ in range(n):
        st, done = State(), set()
        for step in range(1, max_steps + 2):
            if step > max_steps:
                ceil += 1
                steps += max_steps
                succ += st.has_answer()
                break
            a = int(np.searchsorted(np.cumsum(dist(st)), g.random()))
            if a == STOP:
                steps += step
                succ += st.has_answer()
                break
            nxt, learned = transition(st, a)
            if loop_detect and not learned and a in done:
                steps += step
                succ += nxt.has_answer()
                break
            done.add(a)
            st = nxt
    return succ / n, steps / n, ceil / n


# -------------------------------------------------------------------------- run

def run() -> Section:
    s = Section("s13", "The loop: what turns a model into an agent",
                "the model supplies pi; everything else is code you wrote").open()

    s.h("3.1  The formalism, and where the model actually sits")
    s.eq("a_t ~ pi(.|c_t);  s_{t+1} = T(s_t,a_t);  o_t = O(s_t,a_t);  "
         "c_{t+1} = c_t (+) (a_t,o_t)",
         "the model contributes pi and nothing else")
    s.eq("tau = min{ t : a_t = stop  or  t = N }",
         "the stopping time; autonomy is the question of who decides it")
    s.say()
    s.say("  task: 'index' names the record holding the answer, so no single "
          "call can\n  succeed. The dependency is what makes the loop load-"
          "bearing rather than\n  decorative.")
    s.kv("action space |A|", A)
    s.kv("actions", ", ".join(ACTIONS))
    s.kv("shortest correct trajectory", "get:index -> get:record_7 -> stop")
    s.kv("target value", ANSWER)

    # ---------------------------------------------------- 3.2 the degenerate case
    s.h("3.2  An LLM is the degenerate agent")
    degenerate = SoftmaxPolicy(skill=3.0, allowed=(STOP,))
    d = enumerate_trajectories(degenerate, max_steps=1)
    s.table(["configuration", "trajectories", "E[tau]", "E[tool calls]",
             "H(trajectory) bits", "P(success)"],
            [["A = {stop}, N = 1", d.n_traj, round(d.e_steps, 4),
              round(d.e_calls, 4), round(d.h_traj, 4), round(d.p_success, 4)]],
            align="lrrrrr")
    s.check("with a single-element action space and N=1 the loop is exactly one "
            "model call",
            d.n_traj == 1 and d.e_calls == 0.0 and d.h_traj == 0.0
            and d.e_steps == 1.0,
            "one trajectory, zero tool calls, zero trajectory entropy")
    s.note("so 'LLM' and 'agent' are not two kinds of system. They are two "
           "settings of A and N. Everything the rest of this repo studies lives "
           "in the gap between this row and the next table, and none of it is "
           "inside the model.")

    # ------------------------------------------------------- 3.3 autonomy
    s.h("3.3  Autonomy is a measurable quantity")
    s.eq("H(trajectory) = - sum_z P(z) log2 P(z)  over complete trajectories z",
         "zero bits means the program decided everything; more bits means more "
         "of the decision was delegated to the policy")
    rows = []
    configs = [
        ("forced script (no choice at all)",
         SoftmaxPolicy(skill=60.0, allowed=(1, 3, STOP))),
        ("3 actions permitted, skill 3.0",
         SoftmaxPolicy(skill=3.0, allowed=(1, 3, STOP))),
        ("6 actions permitted, skill 3.0", SoftmaxPolicy(skill=3.0)),
        ("6 actions permitted, skill 1.0", SoftmaxPolicy(skill=1.0)),
        ("6 actions, skill 1.0, T = 2.0",
         SoftmaxPolicy(skill=1.0, temperature=2.0)),
    ]
    for name, pol in configs:
        r = enumerate_trajectories(pol, max_steps=6)
        rows.append([name, r.n_traj, round(r.h_traj, 3), round(r.e_steps, 3),
                     round(r.p_success, 4)])
    s.table(["configuration", "trajectories enumerated", "H(traj) bits",
             "E[tau]", "P(success)"], rows, align="lrrrr")
    s.check("constraining the action space strictly reduces trajectory entropy",
            rows[1][2] < rows[2][2],
            f"{rows[2][2]:.3f} bits at |A|=6 against {rows[1][2]:.3f} bits at "
            "|A|=3, same skill")
    s.check("a near-deterministic policy has almost no autonomy by this measure",
            rows[0][2] < 0.01,
            f"H = {rows[0][2]:.2e} bits for the forced script")
    s.check("raising temperature increases autonomy and reduces success together",
            rows[4][2] > rows[3][2] and rows[4][4] < rows[3][4],
            f"H {rows[3][2]:.3f} -> {rows[4][2]:.3f} bits while P(success) "
            f"{rows[3][4]:.4f} -> {rows[4][4]:.4f}")
    s.note("this is the trade every agent design makes, stated in bits. "
           "Autonomy is not a badge, it is the entropy you chose to permit, and "
           "you buy it with success rate. It is also the honest reply to 'is "
           "this really agentic': the question has a number, and the number is "
           "zero for a workflow with a fixed action sequence no matter what the "
           "marketing says.")

    # --------------------------------------------------- 3.4 termination
    s.h("3.4  Termination is a property of the policy, so the ceiling has to exist")
    s.eq("tau ~ Geometric(q)  =>  E[tau] = 1/q,  P(tau > n) = (1-q)^n",
         "if the policy stops with probability q at each step independently")
    rows = []
    for bias in (-3.0, -1.0, 0.0, 2.0):
        pol = SoftmaxPolicy(skill=2.0, stop_bias=bias)
        q = float(pol.dist(State())[STOP])
        r = enumerate_trajectories(pol, max_steps=8)
        rows.append([bias, round(q, 4), round(1 / q, 2), round(r.e_steps, 3),
                     round(r.p_ceiling, 4), round(r.p_success, 4)])
    s.table(["stop_bias", "q at step 1", "1/q", "E[tau] measured",
             "P(hit ceiling)", "P(success)"], rows, align="rrrrrr")
    s.check("a policy reluctant to stop spends the entire step budget",
            rows[0][4] > 0.5,
            f"stop_bias {rows[0][0]}: {rows[0][4] * 100:.0f}% of trajectories "
            f"reach the ceiling, E[tau] = {rows[0][3]:.2f} of 8")
    s.check("eagerness to stop and competence trade off against each other",
            rows[-1][5] < rows[1][5] and rows[-1][3] < rows[1][3],
            f"stop_bias {rows[-1][0]} terminates in {rows[-1][3]:.2f} steps but "
            f"succeeds {rows[-1][5]:.2%} against {rows[1][5]:.2%} at "
            f"stop_bias {rows[1][0]}")
    s.note("E[tau] measured sits below 1/q because the loop truncates at N and "
           "because q is not constant across states -- the oracle bonus moves "
           "onto stop once the answer is in hand. The geometric is the intuition, "
           "not the model. The operational point stands either way: nothing in "
           "the policy guarantees termination, so the ceiling N is not a "
           "defensive nicety but the only thing standing between you and an "
           "unbounded bill.")

    # ------------------------------------------------ 3.5 loop detection
    s.h("3.5  Loop detection, and what it costs when the tool is not idempotent")
    rows = []
    for name, pol in [("skill 3.0, reluctant to stop",
                       SoftmaxPolicy(skill=3.0, stop_bias=-2.0)),
                      ("skill 1.0, reluctant to stop",
                       SoftmaxPolicy(skill=1.0, stop_bias=-2.0))]:
        off = enumerate_trajectories(pol, max_steps=6, loop_detect=False)
        on = enumerate_trajectories(pol, max_steps=6, loop_detect=True)
        rows.append([name, "off", round(off.e_steps, 3), round(off.e_calls, 3),
                     round(off.e_wasted, 3), round(off.p_success, 4)])
        rows.append([name, "on", round(on.e_steps, 3), round(on.e_calls, 3),
                     round(on.e_wasted, 3), round(on.p_success, 4)])
    s.table(["policy", "loop detect", "E[tau]", "E[calls]", "E[wasted calls]",
             "P(success)"], rows, align="llrrrr")
    s.check("suppressing repeats of uninformative actions cuts steps and wasted "
            "calls",
            rows[1][2] < rows[0][2] and rows[1][4] < rows[0][4],
            f"E[tau] {rows[0][2]:.3f} -> {rows[1][2]:.3f}, wasted calls "
            f"{rows[0][4]:.3f} -> {rows[1][4]:.3f}")
    s.check("it costs some success, because terminating early forfeits recovery",
            rows[1][5] <= rows[0][5],
            f"P(success) {rows[0][5]:.4f} -> {rows[1][5]:.4f}")
    s.say()
    s.note("the control is sound only because `get` is idempotent here: a repeat "
           "is provably uninformative, so ending the run loses no information "
           "that another step could have supplied. Change the tool and the "
           "reasoning inverts. Poll a queue, read a clock, retry a write that "
           "timed out, or re-read a record another process is updating, and the "
           "second identical call is exactly the call that carries new "
           "information. Repeat-suppression then silently converts a correct "
           "retry into a premature stop. The signature to deduplicate on is "
           "therefore not (tool, arguments) but (tool, arguments, world "
           "version), which is the same staleness problem S6 and S12 hit from "
           "the memory and cache sides -- three symptoms, one cause.")

    # ------------------------------------- 3.6 the loop's compounding cost
    s.h("3.6  The loop's real cost: context grows, so prefill is paid again")
    s.eq("|c_t| = |c_0| + sum_{i<t} (|a_i| + |o_i|)",
         "append-only, so prompt length is monotone in step count")
    s.eq("cost(N) = sum_{t=1..N} prefill(|c_t|)  ~  Theta(N^2) even when each "
         "call is linear",
         "N calls over a linearly growing prompt")
    cfg = Config(d_model=4096, d_ff=16384, n_layers=32, n_heads=32, vocab=128_000,
                 max_pos=131_072)
    base, per_step = 1_200, 450
    rows, first = [], macs_prefill(cfg, base)
    for step in (1, 2, 4, 8, 16, 32):
        toks = base + per_step * (step - 1)
        cum = sum(macs_prefill(cfg, base + per_step * (i - 1))
                  for i in range(1, step + 1))
        rows.append([step, toks, f"{macs_prefill(cfg, toks):.3e}",
                     f"{cum:.3e}", round(cum / first, 1)])
    s.table(["steps taken", "prompt tokens", "MACs this call",
             "MACs cumulative", "x one call"], rows, align="rrrrr")

    n_steps = rows[-1][0]
    cum_ratio = rows[-1][4]                       # cumulative cost / one call
    tok_ratio = rows[-1][1] / base                # transcript growth factor
    call_ratio = macs_prefill(cfg, rows[-1][1]) / first
    s.check("total loop cost grows faster than the number of steps",
            cum_ratio / n_steps > 1.3,
            f"{n_steps} steps cost {cum_ratio}x a single call, i.e. "
            f"{cum_ratio / n_steps:.2f}x more than {n_steps} identical calls")
    s.check("and each individual call grows faster than the transcript does",
            call_ratio > tok_ratio,
            f"prompt grows {tok_ratio:.1f}x ({base} -> {rows[-1][1]} tokens) "
            f"while the call costs {call_ratio:.1f}x, because at 15k tokens the "
            "quadratic attention term is no longer negligible against 2d + "
            f"d_ff = {2 * cfg.d_model + cfg.d_ff}")
    s.note("the reason this is survivable in practice is S11.5: the transcript is "
           "append-only, so every step's prompt is a prefix extension of the last "
           "and the KV cache carries almost all of it. That is also why "
           "history-compaction is a genuinely hard call rather than an obvious "
           "win -- it shortens the prompt, and it throws away the cache from the "
           "first edited token, which S12 prices properly.")

    # ---------------------------------------------- 3.7 validate the enumeration
    s.h("3.7  Confirming the exact enumeration against sampling")
    pol = SoftmaxPolicy(skill=2.0, stop_bias=-1.0)
    ex = enumerate_trajectories(pol, max_steps=6)
    n = 40_000
    sm_succ, sm_steps, sm_ceil = sample_trajectories(pol, n, max_steps=6)
    se = float(np.sqrt(ex.p_success * (1 - ex.p_success) / n))
    s.table(["quantity", "exact", f"sampled (n={n:,})", "difference",
             "2 s.e."],
            [["P(success)", round(ex.p_success, 5), round(sm_succ, 5),
              round(abs(ex.p_success - sm_succ), 5), round(2 * se, 5)],
             ["E[tau]", round(ex.e_steps, 4), round(sm_steps, 4),
              round(abs(ex.e_steps - sm_steps), 4), "-"],
             ["P(ceiling)", round(ex.p_ceiling, 5), round(sm_ceil, 5),
              round(abs(ex.p_ceiling - sm_ceil), 5), "-"]], align="lrrrr")
    s.kv("trajectories enumerated", ex.n_traj)
    s.kv("probability mass unaccounted for", f"{ex.residual:.2e}")
    s.check("the sampled estimate agrees with the exact value inside sampling error",
            abs(ex.p_success - sm_succ) < 2 * se,
            f"|difference| {abs(ex.p_success - sm_succ):.5f} against 2 s.e. = "
            f"{2 * se:.5f}")
    s.check("the enumeration accounts for essentially all probability mass",
            abs(ex.residual) < 1e-9, f"residual {ex.residual:.2e}")
    s.note("this is the section's own control. The exact numbers above are only "
           "worth quoting if the enumeration is right, and agreement with an "
           "independent sampler at 40,000 draws is the cheapest way to show it "
           "is. Note also what the comparison costs: the sampler needed 40,000 "
           "runs to pin P(success) to about half a percent. That is the regime "
           "every real agent evaluation lives in, and S12 is about surviving it.")

    s.close()
    return s


if __name__ == "__main__":
    run()
