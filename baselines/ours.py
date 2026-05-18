"""Phase 5 — OUR controller (Phase 4 calibrated ranker + switching controller).

This is the headline method in the E2 comparison. We reuse the Phase 4 training
code so that "inference-side glue" stays in one place: the calibrator, the GMM,
and the switching controller's dwell/entropy logic are all imported, not
re-implemented.

Wire-up:

  state (25-D)            from `WarehouseEnv.current_state()`
       |
       v
  regime_post (K-D)       `GaussianMixture.predict_proba` on the 25-D state
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

`load_ours(run_dir)` returns an `OursPolicy` instance with a `__call__` that
takes a 25-D state and returns one of HEURISTIC_NAMES, and a `reset()` that
the harness calls between shifts to clear dwell state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.mixture import GaussianMixture

from models.switching_controller import SwitchingController
from simulation.heuristics import HEURISTIC_NAMES
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
    regime_gmm: GaussianMixture
    feature_cols: list[str]

    def reset(self) -> None:
        self.controller.reset()

    def __call__(self, state: np.ndarray) -> str:
        if state.shape != (N_FEATURES,):
            raise ValueError(
                f"expected 1-D state of length {N_FEATURES}, got shape {state.shape}"
            )
        regime_post = self.regime_gmm.predict_proba(state.reshape(1, -1))[0]
        full = np.concatenate([state, regime_post]).astype(np.float64)
        if full.shape[0] != len(self.feature_cols):
            raise RuntimeError(
                f"feature length mismatch: built {full.shape[0]} from state+regime "
                f"but ranker expects {len(self.feature_cols)} "
                f"(feature_cols={self.feature_cols[:3]}...)"
            )
        return self.controller.select(full)


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
        raise FileNotFoundError(f"Phase 4 run dir not found: {run_dir}")
    for required in ("calibrator.joblib", "regime.joblib", "ranker_meta.joblib"):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"missing artifact: {run_dir / required}")

    cfg = cfg if cfg is not None else OmegaConf.load(DEFAULT_CONFIG_PATH)

    calibrator = joblib.load(run_dir / "calibrator.joblib")
    regime_gmm = joblib.load(run_dir / "regime.joblib")
    meta = joblib.load(run_dir / "ranker_meta.joblib")
    feature_cols = list(meta["feature_cols"])

    expected_total = N_FEATURES + int(regime_gmm.n_components)
    if len(feature_cols) != expected_total:
        raise RuntimeError(
            f"ranker_meta feature_cols has {len(feature_cols)} entries, "
            f"expected {expected_total} ({N_FEATURES} base + "
            f"{regime_gmm.n_components} regime). Did Phase 4 finish cleanly?"
        )

    controller = SwitchingController(
        ranker=calibrator,
        cfg_switching=cfg.ranker.switching,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        feature_cols=feature_cols,
        heuristic_names=HEURISTIC_NAMES,
    )
    return OursPolicy(
        controller=controller,
        regime_gmm=regime_gmm,
        feature_cols=feature_cols,
    )
