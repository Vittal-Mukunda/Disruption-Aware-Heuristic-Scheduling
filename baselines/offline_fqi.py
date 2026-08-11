"""Offline value-learning baseline — fitted Q-iteration with action masking.

The offline-RL competitor in the DAHS comparison. Where DAHS distils a
counterfactual rollout-cost vector — every rule's cost at every state — this
baseline sees only the standard offline-RL signal: the single (s, a, r, s')
transition the behaviour policy actually produced. It learns Q(s, a) by fitted
Q-iteration (Ernst, Geurts & Wehenkel 2005) and deploys masked `argmax_a Q`.

WHAT KEEPS THE COMPARISON FAIR, AND WHAT DID NOT
------------------------------------------------
Same model class (gradient-boosted trees) as the DAHS ranker, same training
shifts, same behaviour policy. Three things were claimed and are now true:

1. THE SAME OBJECTIVE (Reviewer 1, 6.c). The reward was
   `-labeling.snapshot_labeler.composite_cost`, an unweighted cost that counted
   only orders completed inside the interval and charged an unserved order 600x
   less than a breach. FQI was therefore trained to optimise something the
   evaluation no longer measures. The reward is now the negated increment of the
   shared potential, `-(Phi(t+1) - Phi(t))` from `simulation.cost`.

2. THE SAME STATE (Reviewer 1, 6.b). The submitted FQI saw the bare feature
   vector while DAHS saw it plus the regime posteriors, so a comparison billed
   as isolating the *training signal* also varied the *feature set*.
   `cfg.baselines.offline_fqi.use_regime_features` was declared for this and
   read by nothing; it is now honoured, and the fitted GMM travels with the
   model so deployment builds the identical vector.

3. A BEHAVIOUR POLICY WITH REAL COVERAGE (Reviewer 1, 6.b). The submitted logger
   used `a = interval_idx % |H|`. That is uniform *marginally*, but
   `interval_index_in_shift` is an observed feature, so the action was a
   deterministic function of the state: at any state with interval index i the
   logged action is always `i % |H|` and the other |H|-1 are never seen there.
   FQI was being asked to extrapolate Q(s, a) to actions never taken near s,
   which is precisely the distribution shift the submitted Section 6.10 claimed
   did not arise. The logger now uses
   `labeling.rollout_labeler.behaviour_policy`, and
   `experiments/rl_sensitivity.py coverage` measures the difference.

The state is also read through `env.observe()` rather than `env.current_state()`
so admission has happened before the observation, matching every other policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBRegressor

from labeling.rollout_labeler import behaviour_policy
from seed import shift_corpora
from simulation.heuristics import resolve_pool
from simulation.state_extractor import FEATURE_NAMES, N_FEATURES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "offline_fqi"
DEFAULT_REGIME_RUN = REPO_ROOT / "runs" / "phase4"

# Located by name. The submitted module hardcoded `4` while every other call
# site used `FEATURE_NAMES.index(...)`; the revision inserted three expiry
# features and the constant would have silently pointed at a different column.
PCT_PERISHABLE_IDX: int = FEATURE_NAMES.index("pct_perishable")

MASKED_RULE: str = "FEFO"


def _onehot(n_actions: int) -> np.ndarray:
    return np.eye(n_actions, dtype=np.float64)


def _augment(state: np.ndarray, regime_gmm) -> np.ndarray:
    """Append regime posteriors when the ranker's GMM is supplied."""
    if regime_gmm is None:
        return state
    post = regime_gmm.predict_proba(state.reshape(1, -1))[0]
    return np.concatenate([state, post])


def log_transitions(
    shift_seeds: list[int],
    cfg: DictConfig,
    pool: list[str] | None = None,
    regime_gmm=None,
) -> dict[str, np.ndarray]:
    """Replay each shift under the behaviour policy; log (s, a, r, s').

    The reward is the negated increment of the cost potential over the interval,
    which is the same quantity `labeling.rollout_labeler` accumulates over a
    rollout window and `experiments.evaluate` reports at shift end.
    """
    pool = pool if pool is not None else resolve_pool(cfg)
    n_actions = len(pool)
    kind = str(cfg.labeling.observed_policy)

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[np.ndarray] = []
    dones: list[bool] = []
    shift_id: list[int] = []

    for sid, seed in enumerate(shift_seeds):
        env = WarehouseEnv(int(seed), cfg)
        schedule = behaviour_policy(kind, pool, env.n_intervals, int(seed))
        while env.interval_idx < env.n_intervals:
            s = _augment(np.asarray(env.observe(), dtype=np.float64), regime_gmm)
            rule = schedule[env.interval_idx]
            a = pool.index(rule)
            phi_before = env.potential()
            env.step(rule)
            r = -float(env.potential() - phi_before)
            done = bool(env.interval_idx >= env.n_intervals)
            if done:
                s_next = np.zeros_like(s)
            else:
                s_next = _augment(
                    np.asarray(env.observe(), dtype=np.float64), regime_gmm
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
        "n_actions": np.asarray([n_actions], dtype=np.int64),
    }


def subset_transitions(
    transitions: dict[str, np.ndarray], shift_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep only transitions whose `shift_id` is in `shift_ids`."""
    keep = np.isin(transitions["shift_id"], np.asarray(shift_ids))
    return {
        k: (v if k == "n_actions" else v[keep]) for k, v in transitions.items()
    }


def _q_values(model: XGBRegressor, states: np.ndarray, n_actions: int) -> np.ndarray:
    """Q(s, a) for every action -> (n, n_actions)."""
    n = states.shape[0]
    rep = np.repeat(states, n_actions, axis=0)
    onehot = np.tile(_onehot(n_actions), (n, 1))
    q = model.predict(np.hstack([rep, onehot]))
    return np.asarray(q, dtype=np.float64).reshape(n, n_actions)


def _mask_fefo(
    q: np.ndarray, states: np.ndarray, threshold: float, fefo_idx: int | None
) -> np.ndarray:
    """Set Q[:, FEFO] to -inf where pct_perishable < threshold.

    A no-op when the Stage-1 screen dropped FEFO from the pool.
    """
    if fefo_idx is None:
        return q
    q = q.copy()
    invalid = states[:, PCT_PERISHABLE_IDX] < threshold
    q[invalid, fefo_idx] = -np.inf
    return q


def train_fqi(
    transitions: dict[str, np.ndarray],
    *,
    gamma: float,
    n_iterations: int,
    xgb_params: dict,
    fefo_threshold: float,
    fefo_idx: int | None,
    model_seed: int,
    verbose: bool = False,
) -> XGBRegressor:
    """Fitted Q-iteration with a gradient-boosted-tree Q-approximator.

    Q_0 == 0; each sweep regresses Q_{k+1} onto the masked Bellman target
    `r + gamma * max_a' Q_k(s', a')`. Terminal transitions use the reward alone.
    """
    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]
    dones = transitions["dones"]
    n_actions = int(transitions["n_actions"][0])

    X = np.hstack([states, _onehot(n_actions)[actions]])
    model: XGBRegressor | None = None
    for it in range(int(n_iterations)):
        if model is None:
            targets = rewards.copy()
        else:
            q_next = _mask_fefo(
                _q_values(model, next_states, n_actions),
                next_states, fefo_threshold, fefo_idx,
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
    """Evaluation wrapper: the masked greedy policy `argmax_a Q(s, a)`."""

    model: XGBRegressor
    fefo_threshold: float
    arms: list[str]
    regime_gmm: object | None = None
    _fefo_idx: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._fefo_idx = (
            self.arms.index(MASKED_RULE) if MASKED_RULE in self.arms else None
        )

    def reset(self) -> None:
        """Markov policy — no per-shift state to clear."""

    def __call__(self, state: np.ndarray) -> str:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (N_FEATURES,):
            raise ValueError(
                f"expected 1-D state of length {N_FEATURES}, got {state.shape}"
            )
        row = _augment(state, self.regime_gmm).reshape(1, -1)
        q = _mask_fefo(
            _q_values(self.model, row, len(self.arms)),
            row, self.fefo_threshold, self._fefo_idx,
        )[0]
        return self.arms[int(np.argmax(q))]


def save_offline_fqi(run_dir: Path | str, model: XGBRegressor, meta: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "qmodel.joblib")
    joblib.dump(dict(meta), run_dir / "meta.joblib")


def _load_regime_gmm(cfg: DictConfig, regime_run: Path | str = DEFAULT_REGIME_RUN):
    """The ranker's fitted GMM, so FQI sees the same features DAHS does."""
    if not bool(cfg.baselines.offline_fqi.get("use_regime_features", False)):
        return None
    path = Path(regime_run) / "regime.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"cfg.baselines.offline_fqi.use_regime_features is true but "
            f"{path} does not exist. Run `python -m experiments.train_ranker` "
            f"first, or set the flag to false and say so in the paper — the "
            f"claim that the comparison isolates the training signal depends on "
            f"both methods seeing the same state."
        )
    return joblib.load(path)


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

    arms = list(meta.get("arms") or resolve_pool(cfg))
    regime_gmm = None
    if meta.get("use_regime_features"):
        regime_gmm = _load_regime_gmm(cfg, meta.get("regime_run", DEFAULT_REGIME_RUN))
    return OfflineFQIPolicy(
        model=model,
        fefo_threshold=float(
            meta.get("fefo_threshold", cfg.heuristics.fefo_mask_threshold)
        ),
        arms=arms,
        regime_gmm=regime_gmm,
    )


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
    regime_run: Path | str = DEFAULT_REGIME_RUN,
    verbose: bool = True,
) -> Path:
    """Log transitions (if not supplied), run FQI, persist the model."""
    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)
    hp = hp if hp is not None else default_hp(cfg)
    pool = resolve_pool(cfg)
    use_regime = bool(cfg.baselines.offline_fqi.get("use_regime_features", False))
    regime_gmm = _load_regime_gmm(cfg, regime_run)

    if transitions is None:
        train_seeds = shift_corpora(cfg)["train"]
        if n_train_shifts is not None:
            train_seeds = train_seeds[: int(n_train_shifts)]
        if verbose:
            print(f"[offline_fqi] logging transitions for {len(train_seeds)} shifts "
                  f"under behaviour policy '{cfg.labeling.observed_policy}'...")
        transitions = log_transitions(train_seeds, cfg, pool, regime_gmm)

    fefo_idx = pool.index(MASKED_RULE) if MASKED_RULE in pool else None
    model = train_fqi(
        transitions,
        gamma=hp["gamma"],
        n_iterations=hp["n_iterations"],
        xgb_params=hp["xgb_params"],
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        fefo_idx=fefo_idx,
        model_seed=int(cfg.seeds.model),
        verbose=verbose,
    )
    meta = {
        "gamma": hp["gamma"],
        "n_iterations": hp["n_iterations"],
        "xgb_params": hp["xgb_params"],
        "fefo_threshold": float(cfg.heuristics.fefo_mask_threshold),
        "n_transitions": int(transitions["states"].shape[0]),
        "arms": list(pool),
        "behaviour_policy": str(cfg.labeling.observed_policy),
        "use_regime_features": use_regime,
        "regime_run": str(regime_run),
        "state_dim": int(transitions["states"].shape[1]),
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
