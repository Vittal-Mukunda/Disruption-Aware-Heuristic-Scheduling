"""Offline value-learning baseline — fitted Q-iteration with action masking.

A faithful offline-RL competitor for the DAHS comparison. Where DAHS distils a
counterfactual rollout-cost vector (every rule's cost at every state), this
baseline sees only the standard offline-RL signal: the single (s, a, r, s')
transition the round-robin behaviour policy actually produced. It learns a
state-action value Q(s, a) by fitted Q-iteration (Ernst, Geurts & Wehenkel
2005) and deploys the masked greedy policy argmax_a Q(s, a).

Choices that keep the DAHS-vs-this comparison fair and sharp:
  * Same gradient-boosted-tree model class as the DAHS ranker, so the
    comparison isolates the *training signal*, not the function approximator.
  * Same 250 training shifts, same round-robin behaviour policy, and the same
    per-interval reward (-composite_cost) as the PPO baseline.
  * FEFO action masking in both the Bellman backup and at deployment -- the
    "maskable action-value learning" of the Offline-LD family.

The round-robin behaviour policy gives uniform action coverage, so the
distribution-shift pathologies that motivate conservative offline RL (CQL/IQL)
do not arise: a standard fitted-Q baseline is the appropriate representative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBRegressor

from labeling.snapshot_labeler import composite_cost
from seed import spawn_shift_seeds
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "offline_fqi"

N_ACTIONS: int = len(HEURISTIC_NAMES)
FEFO_IDX: int = HEURISTIC_NAMES.index("FEFO")
PCT_PERISHABLE_IDX: int = 4  # state feature index; see simulation/state_extractor.py
_ONEHOT = np.eye(N_ACTIONS, dtype=np.float64)


def log_transitions(shift_seeds: list[int], cfg: DictConfig) -> dict[str, np.ndarray]:
    """Replay each shift under the round-robin behaviour policy; log transitions.

    Returns a dict of stacked arrays: states (M,25), actions (M,), rewards (M,),
    next_states (M,25), dones (M,), shift_id (M,) with M = n_shifts * n_intervals.
    The behaviour policy (interval i -> rule i % 4) is byte-identical to the one
    that produced the DAHS rollout corpus, so the two methods train on the same
    underlying simulated experience and differ only in the supervision signal.
    """
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    dones: list[bool] = []
    shift_id: list[int] = []
    for sid, seed in enumerate(shift_seeds):
        env = WarehouseEnv(int(seed), cfg)
        while env.interval_idx < env.n_intervals:
            s = np.asarray(env.current_state(), dtype=np.float64)
            a = env.interval_idx % N_ACTIONS
            n_before = len(env.completed)
            env.step(HEURISTIC_NAMES[a])
            r = -float(composite_cost(env, n_before))
            done = bool(env.interval_idx >= env.n_intervals)
            s_next = (
                np.zeros(N_FEATURES, dtype=np.float64)
                if done
                else np.asarray(env.current_state(), dtype=np.float64)
            )
            states.append(s)
            actions.append(int(a))
            rewards.append(r)
            next_states.append(s_next)
            dones.append(done)
            shift_id.append(int(sid))
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "next_states": np.asarray(next_states, dtype=np.float64),
        "dones": np.asarray(dones, dtype=bool),
        "shift_id": np.asarray(shift_id, dtype=np.int64),
    }


def subset_transitions(
    transitions: dict[str, np.ndarray], shift_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep only transitions whose `shift_id` is in `shift_ids`."""
    keep = np.isin(transitions["shift_id"], np.asarray(shift_ids))
    return {k: v[keep] for k, v in transitions.items()}


def _q_values(model: XGBRegressor, states: np.ndarray) -> np.ndarray:
    """Q(s, a) for every action -> (n, N_ACTIONS)."""
    n = states.shape[0]
    rep = np.repeat(states, N_ACTIONS, axis=0)
    onehot = np.tile(_ONEHOT, (n, 1))
    q = model.predict(np.hstack([rep, onehot]))
    return np.asarray(q, dtype=np.float64).reshape(n, N_ACTIONS)


def _mask_fefo(q: np.ndarray, states: np.ndarray, threshold: float) -> np.ndarray:
    """Set Q[:, FEFO] to -inf where pct_perishable < threshold (FEFO invalid)."""
    q = q.copy()
    invalid = states[:, PCT_PERISHABLE_IDX] < threshold
    q[invalid, FEFO_IDX] = -np.inf
    return q


def train_fqi(
    transitions: dict[str, np.ndarray],
    *,
    gamma: float,
    n_iterations: int,
    xgb_params: dict,
    fefo_threshold: float,
    model_seed: int,
    verbose: bool = False,
) -> XGBRegressor:
    """Fitted Q-iteration with a gradient-boosted-tree Q-approximator.

    Q_0 == 0; each sweep regresses Q_{k+1} to the masked Bellman target
    r + gamma * max_{a'} Q_k(s', a'). Terminal transitions use the reward only.
    """
    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]
    dones = transitions["dones"]

    X = np.hstack([states, _ONEHOT[actions]])
    model: XGBRegressor | None = None
    for it in range(int(n_iterations)):
        if model is None:
            targets = rewards.copy()
        else:
            q_next = _mask_fefo(
                _q_values(model, next_states), next_states, fefo_threshold
            )
            max_q_next = q_next.max(axis=1)
            targets = rewards + float(gamma) * np.where(dones, 0.0, max_q_next)
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=int(model_seed),
            n_jobs=-1,
            verbosity=0,
            **xgb_params,
        )
        model.fit(X, targets)
        if verbose:
            print(
                f"  [FQI] iter {it + 1:>2d}/{n_iterations}  "
                f"mean|target|={float(np.mean(np.abs(targets))):.4f}"
            )
    assert model is not None
    return model


@dataclass
class OfflineFQIPolicy:
    """Evaluation wrapper: the masked greedy policy argmax_a Q(s, a)."""

    model: XGBRegressor
    fefo_threshold: float

    def reset(self) -> None:
        """Markov policy — no per-shift state to clear."""

    def __call__(self, state: np.ndarray) -> str:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (N_FEATURES,):
            raise ValueError(
                f"expected 1-D state of length {N_FEATURES}, got {state.shape}"
            )
        row = state.reshape(1, -1)
        q = _mask_fefo(_q_values(self.model, row), row, self.fefo_threshold)[0]
        return HEURISTIC_NAMES[int(np.argmax(q))]


def save_offline_fqi(run_dir: Path | str, model: XGBRegressor, meta: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "qmodel.joblib")
    joblib.dump(dict(meta), run_dir / "meta.joblib")


def load_offline_fqi(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
) -> OfflineFQIPolicy:
    """Reconstruct the deterministic evaluation policy from a trained run dir."""
    run_dir = Path(run_dir)
    model_path = run_dir / "qmodel.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"offline-FQI model not found at {model_path}; "
            f"run `python -m experiments.e9_offline_fqi eval` first."
        )
    model = joblib.load(model_path)
    meta_path = run_dir / "meta.joblib"
    meta = joblib.load(meta_path) if meta_path.exists() else {}
    if cfg is None:
        cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    threshold = float(meta.get("fefo_threshold", cfg.heuristics.fefo_mask_threshold))
    return OfflineFQIPolicy(model=model, fefo_threshold=threshold)


def default_hp(cfg: DictConfig) -> dict:
    """The deployed hyperparameters (config defaults; set by the e9 HP search)."""
    fqi = cfg.baselines.offline_fqi
    return {
        "gamma": float(fqi.gamma),
        "n_iterations": int(fqi.n_iterations),
        "xgb_params": {
            "max_depth": int(fqi.xgb.max_depth),
            "n_estimators": int(fqi.xgb.n_estimators),
            "learning_rate": float(fqi.xgb.learning_rate),
        },
    }


def train_offline_fqi(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
    n_train_shifts: int | None = None,
    transitions: dict[str, np.ndarray] | None = None,
    hp: dict | None = None,
    verbose: bool = True,
) -> Path:
    """Log transitions (if not supplied), run FQI, persist the model. End-to-end."""
    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    hp = hp if hp is not None else default_hp(cfg)
    if transitions is None:
        n_train = int(n_train_shifts or cfg.labeling.n_train_shifts)
        all_seeds = spawn_shift_seeds(
            n=int(cfg.labeling.n_train_shifts) + int(cfg.labeling.n_test_shifts),
            root=int(cfg.labeling.seed_seq_root),
        )
        train_seeds = all_seeds[:n_train]
        if verbose:
            print(f"[offline_fqi] logging transitions for {len(train_seeds)} shifts...")
        transitions = log_transitions(train_seeds, cfg)
    model = train_fqi(
        transitions,
        gamma=hp["gamma"],
        n_iterations=hp["n_iterations"],
        xgb_params=hp["xgb_params"],
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        model_seed=int(cfg.seeds.model),
        verbose=verbose,
    )
    meta = {
        "gamma": hp["gamma"],
        "n_iterations": hp["n_iterations"],
        "xgb_params": hp["xgb_params"],
        "fefo_threshold": float(cfg.heuristics.fefo_mask_threshold),
        "n_transitions": int(transitions["states"].shape[0]),
    }
    save_offline_fqi(run_dir, model, meta)
    if verbose:
        print(f"[offline_fqi] saved model to {run_dir}")
    return Path(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the offline-FQI baseline.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--n-train-shifts", type=int, default=None)
    args = parser.parse_args()
    train_offline_fqi(run_dir=args.run_dir, n_train_shifts=args.n_train_shifts)
