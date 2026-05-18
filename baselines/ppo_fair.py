"""Phase 5 — PPO baseline (matched-budget; "fair" version).

A small Gym wrapper exposes `WarehouseEnv` to `stable_baselines3.PPO`:
  - `observation_space = Box(low=-inf, high=inf, shape=(25,))`
  - `action_space      = Discrete(len(HEURISTIC_NAMES))`
  - One step = one 15-minute decision interval. Episode ends when the shift's
    32 intervals are exhausted.
  - Reward = -composite_cost(env, n_completed_before_step) per interval
    (same sign convention LinUCB uses; PPO maximizes).

Training budget = `cfg.baselines.ppo_fair.total_timesteps` (8000 = 250 shifts ×
32 steps — matched to our labeling budget per HANDOFF §1.4 in the master plan
fairness argument).

`train_ppo_fair(run_dir, ...)` trains, writes `ppo_fair.zip`, dumps the per-
episode reward learning curve as `learning_curve.png`, and returns the model.
`load_ppo_fair(run_dir)` reconstructs an evaluation-time callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")  # headless — no display on Windows servers
import matplotlib.pyplot as plt  # noqa: E402

from labeling.snapshot_labeler import composite_cost  # noqa: E402
from seed import SEED_PPO, seed_all, spawn_shift_seeds  # noqa: E402
from simulation.heuristics import HEURISTIC_NAMES  # noqa: E402
from simulation.state_extractor import N_FEATURES  # noqa: E402
from simulation.warehouse_env import WarehouseEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "ppo_fair"


class WarehouseGymEnv(gym.Env):
    """Single-shift Gym env. Episode terminates after `n_intervals` steps.

    Resets cycle through `train_shift_seeds` round-robin so the PPO learner
    sees a different shift per episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: DictConfig, train_shift_seeds: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_shift_seeds = list(train_shift_seeds)
        self._episode_idx = 0
        self.action_space = gym.spaces.Discrete(len(HEURISTIC_NAMES))
        # State is unbounded in principle (queue length, lagged breaches, etc.).
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
        return self.env.current_state(), {}

    def step(self, action: int):
        assert self.env is not None
        h = HEURISTIC_NAMES[int(action)]
        n_before = len(self.env.completed)
        self.env.step(h)
        cost = composite_cost(self.env, n_before)
        reward = -float(cost)
        terminated = bool(self.env.interval_idx >= self.env.n_intervals)
        obs = (
            self.env.current_state()
            if not terminated
            else np.zeros(N_FEATURES, dtype=np.float64)
        )
        return obs, reward, terminated, False, {}


from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402


class _EpisodeRewardCallback(BaseCallback):
    """SB3 callback that records per-episode total reward + timestep."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.timesteps: list[int] = []
        self.episode_rewards: list[float] = []
        self._current: float = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for i in range(len(rewards)):
            self._current += float(rewards[i])
            if bool(dones[i]):
                self.timesteps.append(int(self.num_timesteps))
                self.episode_rewards.append(self._current)
                self._current = 0.0
        return True


def _save_learning_curve(callback: _EpisodeRewardCallback, out_path: Path) -> None:
    if not callback.episode_rewards:
        print("[ppo_fair] no episode rewards recorded; skipping learning curve plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(callback.timesteps, callback.episode_rewards, color="steelblue", lw=1.0)
    if len(callback.episode_rewards) >= 10:
        window = max(5, len(callback.episode_rewards) // 20)
        smoothed = np.convolve(
            callback.episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            callback.timesteps[window - 1:],
            smoothed,
            color="firebrick",
            lw=2.0,
            label=f"rolling mean (w={window})",
        )
        ax.legend(loc="lower right")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (sum of -composite_cost)")
    ax.set_title("PPO-fair learning curve")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ppo_fair] wrote {out_path}")


def train_ppo_fair(
    run_dir: Path = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
    total_timesteps: int | None = None,
) -> Path:
    """Train PPO on the warehouse env. Returns the path to the saved zip."""
    from stable_baselines3 import PPO

    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    if total_timesteps is None:
        total_timesteps = int(cfg.baselines.ppo_fair.total_timesteps)

    n_train = int(cfg.labeling.n_train_shifts)
    seeds = spawn_shift_seeds(
        n=n_train + int(cfg.labeling.n_test_shifts),
        root=int(cfg.labeling.seed_seq_root),
    )
    train_shift_seeds = seeds[:n_train]

    seed_all(SEED_PPO)
    gym_env = WarehouseGymEnv(cfg, train_shift_seeds=train_shift_seeds)
    model = PPO(
        policy="MlpPolicy",
        env=gym_env,
        seed=SEED_PPO,
        verbose=0,
        n_steps=64,           # 2 episodes per rollout (each shift = 32 steps)
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        n_epochs=4,
    )
    print(f"[ppo_fair] training PPO for {total_timesteps} timesteps "
          f"(= {total_timesteps // 32} shifts of 32 steps).")
    callback = _EpisodeRewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "ppo_fair.zip"
    model.save(str(model_path))
    print(f"[ppo_fair] saved {model_path}")
    _save_learning_curve(callback, run_dir / "learning_curve.png")

    # Persist the raw reward trace for later inspection / replotting.
    np.savez(
        run_dir / "learning_curve.npz",
        timesteps=np.asarray(callback.timesteps, dtype=np.int64),
        episode_rewards=np.asarray(callback.episode_rewards, dtype=np.float64),
    )
    return model_path


@dataclass
class PPOPolicy:
    """Stateless evaluation wrapper. The harness calls `policy(state)`."""
    model: object  # stable_baselines3.PPO

    def __call__(self, state: np.ndarray) -> str:
        action, _ = self.model.predict(state.astype(np.float64), deterministic=True)
        return HEURISTIC_NAMES[int(action)]

    def reset(self) -> None:
        pass


def load_ppo_fair(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,  # noqa: ARG001 - kept for API symmetry
) -> PPOPolicy:
    """Reconstruct a deterministic evaluation policy from `ppo_fair.zip`."""
    from stable_baselines3 import PPO

    run_dir = Path(run_dir)
    model_path = run_dir / "ppo_fair.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO-fair model not found at {model_path}; run `train_ppo_fair` first."
        )
    model = PPO.load(str(model_path))
    return PPOPolicy(model=model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()
    train_ppo_fair(args.run_dir, total_timesteps=args.total_timesteps)
