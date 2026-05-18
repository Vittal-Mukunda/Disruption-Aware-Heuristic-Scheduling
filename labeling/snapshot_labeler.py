"""Per-snapshot rollout cost labeler — promoted from `experiments/pilot.py`.

Given a shift seed and a decision-time index `t`, this module:
  1. Re-instantiates `WarehouseEnv(seed, cfg)`,
  2. Replays the observed policy for `t` intervals (`fast_forward`),
  3. For each candidate heuristic h, runs h for the next `min(tau, n_intervals - t)`
     intervals and records the composite rollout cost J_h(s_t).

The cost function is identical to the one used in the Phase 2 pilot — same
weights, same residual-queue penalty, same convention of counting only orders
that finished during the rollout window. Phase 3 must reuse it verbatim so the
soft labels are consistent with the diversity gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from simulation.warehouse_env import WarehouseEnv

W_BREACH: float = 3.0
W_TARDY: float = 0.2
W_UNFINISHED: float = 0.005


def composite_cost(env: WarehouseEnv, n_completed_before_rollout: int) -> float:
    """J_h(s_t) over the rollout window only.

    Counts orders that finished during the rollout (so the shared pre-rollout
    history cancels across heuristics) plus the residual queue length at the
    end of the rollout.
    """
    new_completed = env.completed[n_completed_before_rollout:]
    new_breaches = sum(1 for o in new_completed if o.finish_time > o.sla_due)
    new_tardy_sum = sum(max(o.finish_time - o.sla_due, 0.0) for o in new_completed)
    queue_at_end = len(env.queue)
    return (
        W_BREACH * new_breaches
        + W_TARDY * new_tardy_sum
        + W_UNFINISHED * queue_at_end
    )


def compute_costs_at_snapshot(
    seed: int,
    cfg: DictConfig,
    observed_policy: Sequence[str],
    t: int,
    candidate_heuristics: Sequence[str],
    tau: int,
    env: WarehouseEnv | None = None,
) -> dict[str, float]:
    """Compute the rollout cost J_h(s_t) for each candidate heuristic h.

    Parameters
    ----------
    seed
        Shift seed; combined with `cfg`, fully determines the arrival stream.
    cfg
        Frozen simulation config (loaded once at the driver entry point).
    observed_policy
        Per-interval heuristic names that define the trajectory s_0 -> s_t.
        Must satisfy `len(observed_policy) >= t`.
    t
        Decision-time interval index in `[0, n_intervals)`.
    candidate_heuristics
        Iterable of heuristic names to evaluate. Each gets one rollout.
    tau
        Maximum rollout length in intervals (capped at the remaining shift).
    env
        Optional pre-built `WarehouseEnv`; saves the order-generation cost.
        Will be `fast_forward`'d in-place. If None, a fresh one is built.

    Returns
    -------
    dict mapping heuristic name -> composite cost (float).
    """
    if env is None:
        env = WarehouseEnv(seed, cfg)
    n_intervals = env.n_intervals
    rollout_steps = min(int(tau), n_intervals - int(t))

    costs: dict[str, float] = {}
    for h in candidate_heuristics:
        env.fast_forward(t_intervals=int(t), policy_history=list(observed_policy[:t]))
        n_completed_before = len(env.completed)
        env.run_with_policy(h, n_steps=rollout_steps)
        costs[h] = float(composite_cost(env, n_completed_before))
    return costs
