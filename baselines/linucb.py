"""Contextual LinUCB baseline (Li, Chu, Langford & Schapire 2010).

Per-arm ridge regression on context features with UCB exploration:

    A_k     = I_d                              (d x d, regularised Gram)
    b_k     = 0                                (d,)
    theta_k = A_k^{-1} b_k
    p_k(x)  = theta_k^T x + alpha * sqrt(x^T A_k^{-1} x)
    pull    = argmax_k p_k(x)

This is the minimal-learning baseline: no pretraining, no network, a ridge
bandit told to minimise cost across the test shifts. State persists across
shifts by design — the 50 test shifts form one bandit trajectory of ~1600
interactions.

TWO REVISION CHANGES
--------------------
1. THE REWARD IS THE DEPLOYED OBJECTIVE (Reviewer 1, 6.c).
   The submitted policy scored each pull with
   `labeling.snapshot_labeler.composite_cost`, which counted only orders
   *completed* inside the interval, weighted every order equally, and charged an
   unserved order 0.005 against a breach's 3.0. That module is gone. The reward
   is now the negated increment of the shared cost potential,

       r_t = -( Phi(t+1) - Phi(t) ),

   the same functional `simulation.cost` gives the labeller and the evaluation
   harness. A baseline optimising a different objective than the one it is
   scored on is not a fair comparison, and under the old cost LinUCB was being
   rewarded for leaving hard orders in the queue.

2. FEATURES ARE STANDARDISED (Reviewer 1, 6.b).
   `A = I` is a unit-variance prior on every coordinate, but the raw state spans
   queue length in [0, 200] and utilisation in [0, 1]. The ridge penalty is then
   effectively ~200x weaker on the large-scale columns, the UCB bonus
   `sqrt(x^T A^{-1} x)` is dominated by whichever column happens to be biggest,
   and the arm choice is driven by feature scaling rather than by reward. The
   reviewer flags exactly this for the RL baselines; it applies verbatim here.

   Standardisation is running (Welford), not fitted offline, because the bandit
   is online and has no training split to fit on. The first few dozen decisions
   therefore see drifting statistics; at ~1600 interactions the effect is
   confined to the burn-in and is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simulation.heuristics import resolve_pool
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class RunningStandardiser:
    """Welford running mean/variance, applied as a z-score.

    `update` then `transform` on every observation. Variance is floored so a
    constant column maps to zero rather than to a division by zero.
    """

    dim: int
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.mean.size == 0:
            self.mean = np.zeros(self.dim, dtype=np.float64)
        if self.m2.size == 0:
            self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.m2 / (self.count - 1))
        std = np.where(std < 1e-8, 1.0, std)
        return (x - self.mean) / std


@dataclass
class LinUCBPolicy:
    """Env-aware contextual LinUCB policy with per-arm ridge regression."""

    alpha: float
    feature_dim: int
    arms: list[str]
    standardize: bool = True
    A: list[np.ndarray] = field(default_factory=list)
    b: list[np.ndarray] = field(default_factory=list)
    scaler: RunningStandardiser | None = None
    _pending: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_arms = len(self.arms)
        if not self.A:
            self.A = [np.eye(self.feature_dim) for _ in range(n_arms)]
        if not self.b:
            self.b = [np.zeros((self.feature_dim,)) for _ in range(n_arms)]
        if self.scaler is None:
            self.scaler = RunningStandardiser(dim=self.feature_dim)

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def _score(self, x: np.ndarray, k: int) -> float:
        Ainv = np.linalg.inv(self.A[k])
        theta = Ainv @ self.b[k]
        mean = float(theta @ x)
        bonus = float(self.alpha * np.sqrt(max(x @ Ainv @ x, 0.0)))
        return mean + bonus

    def __call__(self, env: WarehouseEnv) -> str:
        # 1) Settle the previous pull. The reward is the negated increment of the
        #    cost potential over the interval that just ran — the same functional
        #    the labeller and the evaluation harness use.
        if "arm" in self._pending:
            prev_k = self._pending["arm"]
            x_prev = self._pending["x"]
            reward = -float(env.potential() - self._pending["phi_before"])
            self.A[prev_k] += np.outer(x_prev, x_prev)
            self.b[prev_k] += reward * x_prev

        # 2) Read the current observation and pick the next arm.
        raw = np.asarray(env.current_state(), dtype=np.float64)
        raw = np.clip(raw, -1e6, 1e6)
        if self.standardize:
            self.scaler.update(raw)
            x = self.scaler.transform(raw)
        else:
            x = raw

        scores = [self._score(x, k) for k in range(self.n_arms)]
        chosen_k = int(np.argmax(scores))

        # 3) Stash what the next call needs to compute the reward.
        self._pending = {
            "arm": chosen_k,
            "x": x,
            "phi_before": float(env.potential()),
        }
        return self.arms[chosen_k]

    def reset(self) -> None:
        """No-op by design: LinUCB retains learned weights across shifts.

        Only the in-flight pull is dropped, so the first decision of a new shift
        does not settle a reward against an unrelated environment.
        """
        self._pending = {}


def make_linucb_policy(
    cfg: DictConfig | None = None,
    alpha: float | None = None,
) -> LinUCBPolicy:
    """Build a LinUCB policy from `cfg.baselines.linucb`.

    The context dimension is `simulation.state_extractor.N_FEATURES`, derived
    rather than configured: the submitted `cfg.baselines.linucb.feature_dim` was
    a second declaration of the same fact that could drift from the state
    extractor, and did when the feature set changed in revision.
    """
    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    lin = cfg.baselines.linucb
    return LinUCBPolicy(
        alpha=float(lin.alpha) if alpha is None else float(alpha),
        feature_dim=N_FEATURES,
        arms=resolve_pool(cfg),
        standardize=bool(lin.get("standardize_features", True)),
    )
