"""Phase 5 — Contextual LinUCB baseline (Li, Chu, Langford, Schapire 2010).

Per-arm ridge regression on context features with UCB exploration:

    A_k       = I_d                            (d x d, regularized Gram)
    b_k       = 0                              (d x 1, response)
    theta_k   = A_k^{-1} b_k
    p_k(x)    = theta_k^T x + alpha * sqrt(x^T A_k^{-1} x)
    pull      = argmax_k p_k(x)

Reward on each pull is `-composite_cost(env, n_completed_before_step)` over the
single interval that just ran — i.e. the same J_h penalty used at labeling time,
sign-flipped because LinUCB maximizes. We refit `A_k` / `b_k` after every pull.

This is the "minimal-learning" baseline. No pretraining; no neural net; just a
ridge bandit told to optimize cumulative reward across the 50 test shifts.

`make_linucb_policy()` returns an env-aware callable used with
`evaluate_policy_env_aware` (so we can read `env.completed` before/after each
step to compute the reward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from labeling.snapshot_labeler import composite_cost
from seed import SEED_MODEL
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class LinUCBPolicy:
    """Env-aware contextual LinUCB policy with per-arm ridge regression.

    State is persistent across shifts by design — this is *online* learning;
    no `reset()` between shifts. The 50 test shifts together form one bandit
    trajectory of ~1600 interactions.
    """

    alpha: float
    feature_dim: int
    n_arms: int
    arms: list[str]
    A: list[np.ndarray] = field(default_factory=list)
    b: list[np.ndarray] = field(default_factory=list)
    _pending: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.A:
            self.A = [np.eye(self.feature_dim) for _ in range(self.n_arms)]
        if not self.b:
            self.b = [np.zeros((self.feature_dim,)) for _ in range(self.n_arms)]

    def _score(self, x: np.ndarray, k: int) -> float:
        Ainv = np.linalg.inv(self.A[k])
        theta = Ainv @ self.b[k]
        mean = float(theta @ x)
        bonus = float(self.alpha * np.sqrt(x @ Ainv @ x))
        return mean + bonus

    def __call__(self, env: WarehouseEnv) -> str:
        # 1) Apply pending reward from the *previous* pull (if any).
        if "arm" in self._pending:
            prev_k = self._pending["arm"]
            x_prev = self._pending["x"]
            n_before = self._pending["n_completed_before"]
            cost = composite_cost(env, n_before)
            reward = -float(cost)
            self.A[prev_k] += np.outer(x_prev, x_prev)
            self.b[prev_k] += reward * x_prev

        # 2) Read the current 25-D state and pick the next arm.
        x = env.current_state().astype(np.float64, copy=True)
        # numerical guard: tame any unusually-large lag features so the UCB
        # bonus stays finite. State is already finite by construction.
        x = np.clip(x, -1e6, 1e6)
        scores = [self._score(x, k) for k in range(self.n_arms)]
        chosen_k = int(np.argmax(scores))
        chosen = self.arms[chosen_k]

        # 3) Stash the (x, arm) pair so the next call can compute the reward.
        self._pending = {
            "arm": chosen_k,
            "x": x,
            "n_completed_before": len(env.completed),
        }
        return chosen

    def reset(self) -> None:
        """No-op: LinUCB intentionally retains learned weights across shifts."""
        # Drop only the in-flight pending step so the first call of a new shift
        # doesn't try to score a reward against an unrelated env.
        self._pending = {}


def make_linucb_policy(
    alpha: float | None = None,
    feature_dim: int | None = None,
    seed: int = SEED_MODEL,  # noqa: ARG001 - reserved for future stochastic ties
) -> LinUCBPolicy:
    """Build a LinUCB policy with hyperparameters from `cfg.baselines.linucb`."""
    cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    a = float(cfg.baselines.linucb.alpha) if alpha is None else float(alpha)
    d = int(cfg.baselines.linucb.feature_dim) if feature_dim is None else int(feature_dim)
    if d != N_FEATURES:
        raise ValueError(
            f"linucb.feature_dim={d} but simulation state is {N_FEATURES}-D; align them."
        )
    return LinUCBPolicy(
        alpha=a, feature_dim=d, n_arms=len(HEURISTIC_NAMES),
        arms=list(HEURISTIC_NAMES),
    )
