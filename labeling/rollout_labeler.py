"""Multi-sample truncated-rollout labelling.

Replaces `labeling.snapshot_labeler`. Two things changed, one statistical and
one computational, and they turn out to be the same change.

STATISTICAL: THE ROLLOUT IS NOW AN ESTIMATOR (Reviewer 2, 3)
------------------------------------------------------------
The submitted labeller called `env.fast_forward(t)` once per candidate rule and
rolled it forward. Because every stochastic quantity was pre-sampled at
construction, all four replays saw *the same* realised future — the one
belonging to the shift seed. So each label was the rule that happened to be best
on one path, not the rule with the lowest expected cost, and the rollout
variance was identically zero. Proposition 1's bias/variance trade-off had no
variance term anywhere in the code, and Section 6.4's explanation of the U-shape
at tau=4 described a mechanism the implementation did not contain.

Labels are now Monte Carlo means over `M` independent continuations of length
`tau` (not a walk to shift end):

    Jhat_h(s_t) = (1/M) sum_m  [ Phi_m(t + tau L) - Phi(t) ]

with the per-cell standard error recorded alongside, which is what the reviewer
asked to see reported.

COMMON RANDOM NUMBERS. `rollout_seed` depends on (shift, epoch, sample) and
deliberately NOT on the rule. Every candidate rule is therefore scored against
an identical future, the comparison is paired, and the variance of the pairwise
*difference* — which is all the tempered softmax actually reads — falls far
faster in M than the variance of either arm. This is why M=20 suffices.

COMPUTATIONAL: LABELLING GOT CHEAPER PER SAMPLE
-----------------------------------------------
The submitted labeller replayed each shift from t=0 for every (epoch, rule)
pair, costing sum_t |H|*(t + tau) = O(n_intervals^2 * |H|) interval-steps per
shift. This one walks the shift forward once and branches at each epoch:
O(n_intervals * |H| * M * tau). At the submitted setting (32 epochs, 4 rules,
tau=4) the old scheme cost ~2,500 steps per shift; the new scheme at M=20 costs
~10,300 with 4 rules. the corpus stays at 250 shifts and the candidate pool is nine rules (six
retained after screening), so total labelling work rises rather than falls: the saving is per
sample, and it is what makes M=20 affordable at all. Rule calibration, not
labelling, is the binding cost in this campaign — see
`experiments/calibrate_rules.py`, which runs at a deliberately smaller M.

`simulation.warehouse_env.simulated_steps()` instruments the exact figure, which
Reviewer 3 (1) asked for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from simulation.cost import CostWeights
from simulation.heuristics import resolve_pool
from simulation.state_extractor import FEATURE_NAMES
from simulation.warehouse_env import WarehouseEnv


def rollout_seed(base_seed: int, shift_seed: int, t: int, m: int) -> int:
    """Deterministic branch seed for (shift, epoch, sample).

    Independent of the candidate rule — see the common-random-numbers note in
    the module docstring. Derived through a `SeedSequence` rather than by
    arithmetic mixing so that nearby (t, m) do not produce correlated streams.
    """
    ss = np.random.SeedSequence([int(base_seed), int(shift_seed), int(t), int(m)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def behaviour_policy(
    kind: str, pool: Sequence[str], n_intervals: int, shift_seed: int
) -> list[str]:
    """The exploration policy that generates the state-visitation corpus.

    WHY THIS IS NO LONGER A ROUND ROBIN (Reviewer 1, 6.b)
    -----------------------------------------------------
    The submitted corpus used `rule = pool[interval_idx % |H|]`. That looks like
    uniform action coverage, and marginally it is — every rule is taken exactly
    32/|H| times per shift. But `interval_index_in_shift` is one of the observed
    state features, so the behaviour action was a DETERMINISTIC FUNCTION OF AN
    OBSERVED FEATURE. Conditional on the state, coverage was degenerate: at any
    state whose interval index is `i`, the logged action is always `i % |H|` and
    the other |H|-1 actions are never observed.

    That matters only for the offline-RL baseline, and it matters a lot. DAHS is
    unaffected because its labels are counterfactual by construction — every rule
    is rolled out at every state regardless of what the behaviour policy did. But
    fitted Q-iteration sees only the logged (s, a, r, s') tuples, so it was being
    asked to extrapolate Q(s, a) to actions never taken anywhere near s. The
    submitted Section 6.10 asserted the opposite — that round-robin coverage
    meant "the distribution-shift pathologies that motivate conservative offline
    RL do not arise" — and that assertion was false.

    `experiments/rl_sensitivity.py coverage` measures the effect directly.

    kind
      'random'                i.i.d. uniform over the pool, seeded per shift.
                              Breaks the state-action dependence; the default.
      'epsilon_round_robin'   round robin perturbed with probability 0.5, kept
                              as a middle case for the coverage comparison.
      'round_robin'           the submitted scheme, retained so the corpus can
                              be regenerated and the two compared head to head.
    """
    pool = list(pool)
    # Stream distinct from the rollout-branch streams, so changing the
    # behaviour policy cannot perturb the futures the labels are scored against.
    rng = np.random.default_rng(np.random.SeedSequence([int(shift_seed), 7351]))
    if kind == "round_robin":
        return [pool[i % len(pool)] for i in range(n_intervals)]
    if kind == "random":
        return [pool[int(rng.integers(len(pool)))] for _ in range(n_intervals)]
    if kind == "epsilon_round_robin":
        return [
            pool[i % len(pool)] if rng.random() > 0.5
            else pool[int(rng.integers(len(pool)))]
            for i in range(n_intervals)
        ]
    raise ValueError(
        f"Unknown labeling.observed_policy '{kind}'. "
        f"Valid: 'random', 'epsilon_round_robin', 'round_robin'."
    )


@dataclass
class SnapshotCosts:
    """Per-rule rollout cost at one decision epoch."""

    mean: dict[str, float] = field(default_factory=dict)
    stderr: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def best(self) -> str:
        return min(self.mean, key=lambda h: self.mean[h])

    def separation(self) -> float:
        """Gap between best and second-best, in units of the pooled standard error.

        A diagnostic for how much of the label is signal. Epochs where this is
        below ~1 are the ones the ambiguity filter is meant to catch; reporting
        the distribution answers the implicit question behind Reviewer 2 (3) —
        whether a single path could ever have identified the right rule.
        """
        if len(self.mean) < 2:
            return float("inf")
        ordered = sorted(self.mean.values())
        gap = ordered[1] - ordered[0]
        pooled = float(np.sqrt(sum(s**2 for s in self.stderr.values()) / len(self.stderr)))
        return float(gap / pooled) if pooled > 0 else float("inf")


def costs_at_epoch(
    env: WarehouseEnv,
    shift_seed: int,
    t: int,
    candidates: Sequence[str],
    tau: int,
    n_samples: int,
    base_seed: int,
    model_cfg=None,
) -> SnapshotCosts:
    """Monte Carlo rollout cost of every candidate rule at the current epoch.

    `env` is left untouched: `branch()` copies before it mutates. The start
    potential is shared by every branch and every rule, so it is computed once.

    `model_cfg` forecasts the future under parameters that may differ from the
    environment's own, which is how the misspecification experiment of Reviewer 2
    (5) gives a lookahead controller an imperfect model to plan with.
    """
    phi_start = env.potential()
    steps = min(int(tau), env.n_intervals - int(t))

    out = SnapshotCosts(n_samples=int(n_samples))
    if steps <= 0:
        # Terminal epoch: every rule is equivalent because nothing can follow.
        for h in candidates:
            out.mean[h] = 0.0
            out.stderr[h] = 0.0
        return out

    # Pre-derive the branch seeds so that rule `h` and rule `h'` provably see the
    # identical sequence of futures.
    seeds = [rollout_seed(base_seed, shift_seed, t, m) for m in range(int(n_samples))]

    for h in candidates:
        samples = np.empty(len(seeds), dtype=np.float64)
        for i, s in enumerate(seeds):
            branch = env.branch(s, model_cfg=model_cfg)
            branch.run_with_policy(h, n_steps=steps)
            samples[i] = branch.potential() - phi_start
        out.mean[h] = float(samples.mean())
        out.stderr[h] = (
            float(samples.std(ddof=1) / np.sqrt(samples.size)) if samples.size > 1 else 0.0
        )
    return out


def costs_at_epoch_successive_halving(
    env: WarehouseEnv,
    shift_seed: int,
    t: int,
    candidates: Sequence[str],
    tau: int,
    n_samples: int,
    base_seed: int,
    eta: int = 2,
    model_cfg=None,
) -> tuple[SnapshotCosts, int]:
    """Adaptive sample allocation across rules. Returns `(costs, samples_used)`.

    WHY (Reviewer 3, comment 2)
    ---------------------------
        "If the pool were expanded to 10 or 20 heuristics, the offline rollout
         cost would grow linearly. Please discuss ... whether sub-sampling or
         hierarchical selection could mitigate this cost."

    Uniform allocation spends `M` continuations on every rule, including rules
    that are obviously bad after two samples. Successive halving spends the same
    total budget adaptively: run a cheap round over all rules, discard the worst
    `1 - 1/eta` fraction, and reallocate the freed samples to the survivors.
    Precision ends up concentrated on the contenders, which is where the label
    actually needs it.

    This is compatible with the soft label rather than in tension with it. The
    tempered softmax maps a clearly-inferior rule to near-zero probability
    regardless of how precisely its cost was estimated, so a rule eliminated in
    round one loses nothing it would have contributed: its coarse, high mean
    already sends it to the same place. The rules whose probabilities are
    materially non-zero are exactly the ones successive halving keeps sampling.

    Samples accumulate across rounds — a survivor's mean uses every continuation
    drawn for it so far — and the seed for continuation `m` is the same one
    uniform allocation would have used, so common random numbers still hold
    across rules within a round.
    """
    phi_start = env.potential()
    steps = min(int(tau), env.n_intervals - int(t))
    out = SnapshotCosts(n_samples=int(n_samples))

    arms = list(candidates)
    if steps <= 0:
        for h in arms:
            out.mean[h] = 0.0
            out.stderr[h] = 0.0
        return out, 0

    budget = len(arms) * int(n_samples)
    n_rounds = max(1, int(np.ceil(np.log(len(arms)) / np.log(eta)))) if len(arms) > 1 else 1

    draws: dict[str, list[float]] = {h: [] for h in arms}
    used = 0
    alive = list(arms)

    for r in range(n_rounds):
        per_arm = max(1, int(budget // (len(alive) * n_rounds)))
        target = len(draws[alive[0]]) + per_arm
        for h in alive:
            for m in range(len(draws[h]), target):
                branch = env.branch(
                    rollout_seed(base_seed, shift_seed, t, m), model_cfg=model_cfg
                )
                branch.run_with_policy(h, n_steps=steps)
                draws[h].append(branch.potential() - phi_start)
                used += steps
        if r < n_rounds - 1 and len(alive) > 1:
            alive.sort(key=lambda h: float(np.mean(draws[h])))
            alive = alive[: max(1, len(alive) // eta)]

    for h in arms:
        s = np.asarray(draws[h], dtype=np.float64)
        out.mean[h] = float(s.mean())
        out.stderr[h] = (
            float(s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else float("nan")
        )
    return out, used


def label_one_shift(
    shift_id: int,
    shift_seed: int,
    cfg: DictConfig,
    tau: int | None = None,
    n_samples: int | None = None,
    candidates: Sequence[str] | None = None,
) -> list[dict]:
    """One row per decision epoch for a single shift.

    Walks the shift forward once under `cfg.labeling.observed_policy`, branching
    at each epoch. That policy determines only which states are VISITED — the
    labels themselves are counterfactual, since every candidate rule is rolled
    out at every epoch regardless of what the behaviour policy did. DAHS is
    therefore insensitive to the choice; the offline-RL baseline is not, which
    is the whole point of `behaviour_policy()` and of Reviewer 1 (6.b).

    `candidates` overrides the configured pool. Stage-1 screening and the ATC
    calibration pass the full candidate library here so that every rule is
    scored against the same states and the same futures.
    """
    pool = list(candidates) if candidates is not None else resolve_pool(cfg)
    tau = int(cfg.labeling.tau) if tau is None else int(tau)
    n_samples = (
        int(cfg.labeling.n_rollout_samples) if n_samples is None else int(n_samples)
    )
    base_seed = int(cfg.seeds.rollout)

    env = WarehouseEnv(int(shift_seed), cfg)
    observed = behaviour_policy(
        str(cfg.labeling.observed_policy), pool, env.n_intervals, shift_seed
    )

    rows: list[dict] = []
    for t in range(env.n_intervals):
        feat = env.observe()
        costs = costs_at_epoch(
            env, shift_seed, t, pool, tau, n_samples, base_seed
        )

        row: dict = {
            "shift_id": int(shift_id),
            "shift_seed": int(shift_seed),
            "interval_idx": int(t),
            "behaviour_rule": observed[t],
            "label_separation": float(costs.separation()),
        }
        for i, name in enumerate(FEATURE_NAMES):
            row[f"f_{name}"] = float(feat[i])
        for h in pool:
            row[f"cost_{h}"] = float(costs.mean[h])
            row[f"se_{h}"] = float(costs.stderr[h])
        rows.append(row)

        env.step(observed[t])

    return rows


def label_one_shift_counted(
    shift_id: int,
    shift_seed: int,
    cfg: DictConfig,
    tau: int | None = None,
    n_samples: int | None = None,
    candidates: Sequence[str] | None = None,
) -> tuple[list[dict], int]:
    """`label_one_shift` plus the interval-steps it actually simulated.

    WHY THIS EXISTS. `simulation.warehouse_env._STEP_COUNTER` is a module-level
    global, so it is PROCESS-LOCAL. Under joblib's default loky backend every
    shift is labelled in a worker process and the parent's counter never sees
    that work — reading `simulated_steps()` in the driver after a parallel map
    returns the parent's own count, which is zero. The offline simulation cost
    Reviewer 3 (1) asks for would then be reported as ~0 regardless of the
    campaign size.

    Resetting inside the task and returning the count makes the figure additive
    and correct under any backend, including `n_jobs=1` where the workers and
    the parent are the same process.
    """
    from simulation.warehouse_env import reset_step_counter, simulated_steps

    reset_step_counter()
    rows = label_one_shift(
        shift_id, shift_seed, cfg, tau=tau, n_samples=n_samples, candidates=candidates
    )
    return rows, simulated_steps()


def rollout_step_budget(
    n_shifts: int, n_intervals: int, n_rules: int, tau: int, n_samples: int
) -> int:
    """Interval-steps required to label a corpus. Answers Reviewer 3 (1) a priori.

    The realised figure is measured by `simulation.warehouse_env.simulated_steps()`;
    this is the closed form used to size a run before launching it.
    """
    walk = n_shifts * n_intervals
    rollouts = n_shifts * n_intervals * n_rules * n_samples * tau
    return int(walk + rollouts)


# Backwards-compatible alias so existing callers keep working during migration.
compute_costs_at_snapshot = costs_at_epoch


__all__ = [
    "SnapshotCosts",
    "behaviour_policy",
    "costs_at_epoch",
    "costs_at_epoch_successive_halving",
    "label_one_shift",
    "label_one_shift_counted",
    "rollout_seed",
    "rollout_step_budget",
    "CostWeights",
]
