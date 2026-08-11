"""Rolling-horizon rollout MPC — the teacher DAHS distils.

WHY THIS EXISTS (Reviewer 2, 6)
-------------------------------
DAHS is trained to reproduce a tau-step rollout, and the paper's central claim is
that it *amortises* that lookahead into a single forward pass. The submitted
version never evaluated the lookahead. The only MPC baseline, `greedy_mpc`, was
tau=1, while the deployed model distilled tau=4 — so the comparison that would
tell you whether distillation approaches, matches or loses to its teacher was
simply absent, and the amortisation claim was unquantified in both directions.

This baseline is that teacher, run online:

    at each epoch, evaluate every rule over `tau` intervals, averaged over
    `n_samples` independent continuations; commit the arg-min rule for ONE
    interval; discard the rest of the plan and replan at the next epoch.

It is the standard receding-horizon construction, and it is deliberately the
same estimator the labeller uses (`labeling.rollout_labeler.costs_at_epoch`) so
that any gap between this policy and DAHS is attributable to the function
approximation and the deployment guardrails, not to a different scoring rule.

WHAT IT COSTS
-------------
`|H| * n_samples * tau` interval-steps *per decision*, every decision, forever.
DAHS pays a comparable cost once, offline, and then answers in one forward pass.
`per_decision_seconds()` reports the measured latency so the amortisation
argument appears in the paper as a number rather than an assertion — which is
also what Reviewer 3 (1) and (5) ask for.

`greedy_mpc` is the tau=1 special case and is kept as a named alias so the
submitted results table remains comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from omegaconf import DictConfig, OmegaConf

from labeling.rollout_labeler import costs_at_epoch
from simulation.heuristics import resolve_pool
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# Deployment-time branches are drawn from a different stream than the labelling
# branches. The corpora are disjoint so there is no leakage either way, but
# keeping the streams separate means a future change to the labelling seed
# cannot silently perturb a baseline.
DEPLOY_SEED_OFFSET: int = 7919


@dataclass
class RollingHorizonMPC:
    """Env-aware receding-horizon policy. Call once per decision epoch."""

    candidates: list[str]
    tau: int
    n_samples: int
    base_seed: int
    # The controller's model of the world. `None` means it plans with the true
    # dynamics, which is the nominal setting. The misspecification experiment
    # (Reviewer 2, 5) passes the nominal config here while the environment runs
    # perturbed parameters, so the lookahead is not handed the truth for free —
    # the same handicap DAHS carries, whose labels were generated under nominal.
    model_cfg: DictConfig | None = None

    _decision_times: list[float] = field(default_factory=list)

    def reset(self) -> None:
        """No policy state to clear; the timing trace persists across shifts."""

    def __call__(self, env: WarehouseEnv) -> str:
        t0 = perf_counter()
        costs = costs_at_epoch(
            env,
            shift_seed=env.seed,
            t=env.interval_idx,
            candidates=self.candidates,
            tau=self.tau,
            n_samples=self.n_samples,
            base_seed=self.base_seed,
            model_cfg=self.model_cfg,
        )
        chosen = costs.best()
        self._decision_times.append(perf_counter() - t0)
        return chosen

    # -- instrumentation ------------------------------------------------

    def per_decision_seconds(self) -> dict[str, float]:
        """Latency summary over every decision taken so far."""
        if not self._decision_times:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0.0}
        a = np.asarray(self._decision_times, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "n": float(a.size),
        }

    def steps_per_decision(self) -> int:
        """Interval-steps simulated per decision — the analytic cost."""
        return len(self.candidates) * self.n_samples * self.tau


def make_rolling_horizon_mpc(
    cfg: DictConfig | None = None,
    tau: int | None = None,
    n_samples: int | None = None,
) -> RollingHorizonMPC:
    """Build the receding-horizon policy from config, with optional overrides."""
    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    mpc = cfg.baselines.rolling_horizon_mpc
    return RollingHorizonMPC(
        candidates=resolve_pool(cfg),
        tau=int(mpc.tau) if tau is None else int(tau),
        n_samples=int(mpc.n_samples) if n_samples is None else int(n_samples),
        base_seed=int(cfg.seeds.rollout) + DEPLOY_SEED_OFFSET,
    )


def make_greedy_mpc_policy(cfg: DictConfig | None = None) -> RollingHorizonMPC:
    """The tau=1 special case, under its submitted name.

    Note that unlike the submitted `greedy_mpc` this averages over `n_samples`
    branches rather than evaluating the single pre-sampled future. At tau=1 the
    two rarely disagree, but the estimator is now consistent with everything
    else in the campaign.
    """
    return make_rolling_horizon_mpc(cfg, tau=1)


__all__ = [
    "RollingHorizonMPC",
    "make_rolling_horizon_mpc",
    "make_greedy_mpc_policy",
]
