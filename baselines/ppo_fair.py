"""PPO baseline — now configurable and normalisable, so its failure can be tested.

WHY THIS WAS REWRITTEN (Reviewer 1, 6.b)
-----------------------------------------
The submitted implementation trained stable-baselines3 PPO with stock
hyperparameters, no observation normalisation and no reward normalisation, on a
feature vector whose columns span queue lengths in the hundreds and utilisations
in [0, 1]. It then varied the training budget alone — 8k against 500k steps —
and concluded that PPO's deficit was "structural, not budgetary".

That conclusion did not follow from that experiment. Unnormalised observations
are the most common single cause of a policy-gradient method failing to learn,
and the submitted run could not distinguish "policy gradients are the wrong
instrument for rule selection" from "this run was misconfigured". DAHS, being a
tree ensemble, is scale-invariant and never paid the same cost.

Three changes make the claim testable:

  * `train_ppo_fair(..., hyperparams=...)` accepts the full configuration, so
    `experiments/rl_sensitivity.py` can sweep discount, GAE lambda, rollout
    horizon and entropy coefficient — the knobs the reviewer names.
  * `normalize_obs` / `normalize_reward` wrap the env in `VecNormalize`, and the
    running statistics are saved and reloaded with the policy. Evaluating a
    normalised policy without its statistics silently produces a different
    policy, which is a subtle and common way to under-report a baseline.
  * The reward is the per-interval potential difference of `simulation.cost`,
    identical to the labelling objective, so PPO and DAHS optimise the same
    thing (Reviewer 1, 6.c asked for this to be made explicit).

The submitted defaults are preserved in `experiments.rl_sensitivity.PPO_BASELINE`
so every swept configuration reads as a delta from what the paper reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from seed import SEED_PPO, seed_all, shift_corpora
from simulation.heuristics import resolve_pool
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "ppo_fair"

# Keys consumed by the SB3 constructor; everything else in a hyperparams dict is
# handled here (the two normalisation flags) or ignored.
_SB3_KEYS = (
    "gamma", "gae_lambda", "n_steps", "ent_coef",
    "learning_rate", "n_epochs", "batch_size",
)


class WarehouseGymEnv(gym.Env):
    """Single-shift Gym env. One step = one 15-minute decision epoch.

    Resets cycle through the training shift seeds so the learner sees a
    different instance each episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: DictConfig, train_shift_seeds: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.pool = resolve_pool(cfg)
        self.train_shift_seeds = list(train_shift_seeds)
        self._episode_idx = 0
        self.action_space = gym.spaces.Discrete(len(self.pool))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float64
        )
        self.env: WarehouseEnv | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        shift_seed = self.train_shift_seeds[
            self._episode_idx % len(self.train_shift_seeds)
        ]
        self._episode_idx += 1
        self.env = WarehouseEnv(int(shift_seed), self.cfg)
        return self.env.observe(), {}

    def step(self, action: int):
        assert self.env is not None
        # Reward is the negated potential difference — exactly the quantity the
        # rollout labeller integrates and the quantity `composite_cost` reports.
        phi_before = self.env.potential()
        self.env.step(self.pool[int(action)])
        reward = -float(self.env.potential() - phi_before)
        terminated = bool(self.env.interval_idx >= self.env.n_intervals)
        obs = (
            self.env.observe()
            if not terminated
            else np.zeros(N_FEATURES, dtype=np.float64)
        )
        return obs, reward, terminated, False, {}


def train_ppo_fair(
    run_dir: Path = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
    total_timesteps: int | None = None,
    hyperparams: dict | None = None,
) -> Path:
    """Train PPO and persist the policy plus any normalisation statistics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    hp = dict(hyperparams or {})
    if total_timesteps is None:
        total_timesteps = int(cfg.baselines.ppo_fair.total_timesteps)

    train_seeds = shift_corpora(cfg)["train"]
    seed_all(SEED_PPO)

    venv = DummyVecEnv([lambda: WarehouseGymEnv(cfg, train_seeds)])
    norm_obs = bool(hp.pop("normalize_obs", False))
    norm_rew = bool(hp.pop("normalize_reward", False))
    if norm_obs or norm_rew:
        venv = VecNormalize(venv, norm_obs=norm_obs, norm_reward=norm_rew, clip_obs=10.0)

    sb3_kwargs = {k: v for k, v in hp.items() if k in _SB3_KEYS}
    model = PPO(policy="MlpPolicy", env=venv, seed=SEED_PPO, verbose=0, **sb3_kwargs)
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "ppo_fair.zip"))
    if isinstance(venv, VecNormalize):
        # Without these the evaluated policy is not the trained policy.
        venv.save(str(run_dir / "vecnormalize.pkl"))
    return run_dir / "ppo_fair.zip"


@dataclass
class PPOPolicy:
    """Deterministic evaluation wrapper, applying saved observation statistics."""

    model: object
    pool: list[str]
    obs_rms: object | None = None
    clip_obs: float = 10.0
    epsilon: float = 1e-8

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return state
        z = (state - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs)

    def __call__(self, state: np.ndarray) -> str:
        obs = self._normalize(np.asarray(state, dtype=np.float64))
        action, _ = self.model.predict(obs, deterministic=True)
        return self.pool[int(action)]

    def reset(self) -> None:
        """Markov policy — nothing to clear between shifts."""


def load_ppo_fair(
    run_dir: Path | str = DEFAULT_RUN_DIR, cfg: DictConfig | None = None
) -> PPOPolicy:
    """Reconstruct the evaluation policy, including normalisation statistics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    run_dir = Path(run_dir)
    model_path = run_dir / "ppo_fair.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found at {model_path}.")
    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)

    obs_rms = None
    vec_path = run_dir / "vecnormalize.pkl"
    if vec_path.exists():
        vn = VecNormalize.load(str(vec_path), venv=None)
        obs_rms = vn.obs_rms if vn.norm_obs else None

    return PPOPolicy(
        model=PPO.load(str(model_path)), pool=resolve_pool(cfg), obs_rms=obs_rms
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--total-timesteps", type=int, default=None)
    a = p.parse_args()
    train_ppo_fair(a.run_dir, total_timesteps=a.total_timesteps)
