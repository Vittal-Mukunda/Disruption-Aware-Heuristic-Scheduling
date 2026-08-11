"""Phase 5 — OUR controller (Phase 4 calibrated ranker + switching controller).

This is the headline method in the E2 comparison. We reuse the Phase 4 training
code so that "inference-side glue" stays in one place: the calibrator, the GMM,
and the switching controller's dwell/entropy logic are all imported, not
re-implemented.

Wire-up:

  state (N_FEATURES-D)    from `WarehouseEnv.observe()`
       |
       v
  regime_post (K-D)       `GaussianMixture.predict_proba` on the state
       |
       v
  full feature vector     concat in the order persisted as
                          `ranker_meta["feature_cols"]`
       |
       v
  CalibratedRanker        loaded from `calibrator.joblib` (wraps the prefit
                          XGBClassifier + isotonic per-class calibrator)
       |
       v
  SwitchingController     FEFO mask, dwell counter T_min, entropy gate

`load_ours(run_dir)` returns an `OursPolicy` whose `__call__` takes a state
vector and returns a rule name, plus a `reset()` the harness calls between
shifts to clear dwell state.

The rule pool comes from `ranker_meta["classes"]` — the class order the model
was fitted with — not from `cfg.heuristics.pool`, which Stage 1 rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.mixture import GaussianMixture

from models.switching_controller import PCT_PERISHABLE_IDX, SwitchingController
from simulation.state_extractor import N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class OursPolicy:
    """Stateful callable wrapping the Phase 4 inference stack.

    Use `OursPolicy.reset()` between shifts. The harness in `experiments.evaluate`
    does this automatically when a `reset` attribute is present.
    """

    controller: SwitchingController
    regime_gmm: GaussianMixture | None
    feature_cols: list[str]
    # Index map from the full state into the ranker's base columns. Non-trivial
    # only for the parsimony ablation (Reviewer 3, 4), which fits on a subset.
    base_idx: list[int] | None = None

    def reset(self) -> None:
        self.controller.reset()

    def __call__(self, state: np.ndarray) -> str:
        if state.shape != (N_FEATURES,):
            raise ValueError(
                f"expected 1-D state of length {N_FEATURES}, got shape {state.shape}"
            )
        base = state if self.base_idx is None else state[self.base_idx]
        if self.regime_gmm is None:
            full = np.asarray(base, dtype=np.float64)
        else:
            regime_post = self.regime_gmm.predict_proba(state.reshape(1, -1))[0]
            full = np.concatenate([base, regime_post]).astype(np.float64)
        if full.shape[0] != len(self.feature_cols):
            raise RuntimeError(
                f"feature length mismatch: built {full.shape[0]} from state+regime "
                f"but ranker expects {len(self.feature_cols)} "
                f"(feature_cols={self.feature_cols[:3]}...)"
            )
        # Read the mask input from the FULL state, not from the ranker's vector:
        # under the parsimony ablation the latter is a subset and the positional
        # index would point at a different feature.
        return self.controller.select(
            full, pct_perishable=float(state[PCT_PERISHABLE_IDX])
        )


def load_ours(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
) -> OursPolicy:
    """Load the three Phase 4 artifacts and assemble the switching controller.

    Parameters
    ----------
    run_dir
        Directory holding `model.json`, `calibrator.joblib`, `regime.joblib`,
        and `ranker_meta.joblib`. Defaults to `runs/phase4/`.
    cfg
        Pre-loaded OmegaConf. Defaults to loading `config.yaml`.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Stage 3 run dir not found: {run_dir}")
    for required in ("calibrator.joblib", "ranker_meta.joblib"):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"missing artifact: {run_dir / required}")

    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)

    calibrator = joblib.load(run_dir / "calibrator.joblib")
    meta = joblib.load(run_dir / "ranker_meta.joblib")
    feature_cols = list(meta["feature_cols"])

    # The `no_regime` ablation trains without the layer, so there is no GMM to
    # load. Everything else is identical, which is what makes it an ablation.
    no_regime = bool(meta.get("no_regime", False))
    regime_gmm = None
    if not no_regime:
        if not (run_dir / "regime.joblib").exists():
            raise FileNotFoundError(f"missing artifact: {run_dir / 'regime.joblib'}")
        regime_gmm = joblib.load(run_dir / "regime.joblib")

    # Base columns may be a SUBSET of the full state (parsimony ablation), so the
    # inference-time vector is rebuilt by name rather than assumed to be the
    # whole feature map followed by the posteriors.
    from simulation.state_extractor import FEATURE_NAMES

    n_regime = 0 if regime_gmm is None else int(regime_gmm.n_components)
    base_cols = [c for c in feature_cols if not c.startswith("regime_post_")]
    base_idx: list[int] | None = None
    if base_cols != [f"f_{n}" for n in FEATURE_NAMES]:
        try:
            base_idx = [FEATURE_NAMES.index(c[len("f_"):]) for c in base_cols]
        except ValueError as exc:
            raise RuntimeError(
                f"ranker_meta feature_cols contains a column that is neither a "
                f"regime posterior nor a known state feature: {exc}"
            ) from None
    expected_total = len(base_cols) + n_regime
    if len(feature_cols) != expected_total:
        raise RuntimeError(
            f"ranker_meta feature_cols has {len(feature_cols)} entries, "
            f"expected {expected_total} ({len(base_cols)} base + {n_regime} "
            f"regime). Did Stage 3 finish cleanly?"
        )

    # The class order the model was TRAINED with, not the current config pool.
    # Stage 1 rescreens the pool, so reading `cfg.heuristics.pool` here would
    # remap class indices onto different rule names whenever the config moved
    # after training — the controller would deploy the wrong rule under the
    # right name, silently and without erroring.
    classes = list(meta.get("classes") or [])
    if not classes:
        raise RuntimeError(
            f"{run_dir / 'ranker_meta.joblib'} has no 'classes' entry. It was "
            f"written by a pre-revision Stage 3, when the pool was a fixed "
            f"module constant. Retrain with the current "
            f"`experiments/train_ranker.py`."
        )

    controller = SwitchingController(
        ranker=calibrator,
        cfg_switching=cfg.ranker.switching,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        feature_cols=feature_cols,
        heuristic_names=classes,
    )
    return OursPolicy(
        controller=controller,
        regime_gmm=regime_gmm,
        feature_cols=feature_cols,
        base_idx=base_idx,
    )
