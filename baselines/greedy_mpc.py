"""Phase 5 — greedy 1-step MPC baseline (rollout oracle proxy).

At every decision interval, the policy deep-copies the live `WarehouseEnv`,
simulates one step (`tau=1`) under each candidate heuristic, scores the result
with the Phase 3 `labeling.snapshot_labeler.composite_cost`, and returns the
argmin heuristic. This is the same scoring rule we used to label snapshots,
so greedy_mpc is the closest analytic upper bound on what a snapshot ranker
could ever pick — useful as the "oracle proxy" reference in E2.

The harness must use `evaluate_policy_env_aware` (not `evaluate_policy`), since
this policy needs the live env to deep-copy. See `experiments.evaluate`.

Wall-clock note: per shift this is ~k_heuristics × k_intervals deep copies +
single-step simulations. With 4 heuristics × 32 intervals × 50 shifts the
overhead is ~6400 deep-copies; expect ~minute-level wall time per 50-shift
sweep on a laptop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from labeling.snapshot_labeler import composite_cost
from simulation.heuristics import HEURISTIC_NAMES
from simulation.warehouse_env import WarehouseEnv


@dataclass
class GreedyMPCPolicy:
    """Env-aware 1-step lookahead policy.

    `__call__(env)` is invoked once per interval by `run_shift_env_aware`.
    """

    candidates: list[str]

    def __call__(self, env: WarehouseEnv) -> str:
        best_h: str | None = None
        best_cost: float = float("inf")
        for h in self.candidates:
            env_copy = copy.deepcopy(env)
            n_before = len(env_copy.completed)
            env_copy.step(h)
            c = composite_cost(env_copy, n_before)
            if c < best_cost:
                best_cost = c
                best_h = h
        # `candidates` is non-empty by construction, but mypy-friendly assert:
        assert best_h is not None
        return best_h


def make_greedy_mpc_policy() -> GreedyMPCPolicy:
    """Build the canonical greedy_mpc policy over the active heuristic pool."""
    return GreedyMPCPolicy(candidates=list(HEURISTIC_NAMES))
