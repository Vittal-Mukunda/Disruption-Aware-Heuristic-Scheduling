"""Switching controller — inference-time wrapper around the calibrated ranker.

Three responsibilities, applied in this order to every snapshot:

1. **FEFO mask.** If `state[pct_perishable] < cfg.heuristics.fefo_mask_threshold`,
   force `p_FEFO = 0` and renormalize the remaining classes. Same mask we use
   at label time (`labeling.soft_label_converter.fefo_mask`) so the calibrator
   never has to model "FEFO on a non-perishable queue".

2. **Minimum-dwell `T_min`.** Once a heuristic is selected, keep it for at
   least `cfg.ranker.switching.t_min_intervals` consecutive intervals. While
   the dwell counter is active, the controller only switches if the next
   override condition fires.

3. **Entropy gate.** Override the dwell when the post-mask distribution is
   *confident*, i.e. `H(p) < cfg.ranker.switching.entropy_gate_ratio * log(K)`.
   Confident = a clear winner; the controller is allowed to switch even mid-dwell.

The controller is stateful across snapshots of a single shift. Call
`reset()` between shifts (or instantiate one controller per shift). The
public `select(features)` returns the chosen heuristic name. The batch
helper `select_sequence(features_stream)` resets internally and steps through
a 2-D feature stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from labeling.soft_label_converter import fefo_mask
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import FEATURE_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig

PCT_PERISHABLE_IDX: int = FEATURE_NAMES.index("pct_perishable")


class _RankerLike(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def _row_entropy_single(p: np.ndarray) -> float:
    safe = np.where(p > 0.0, p, 1.0)
    return float(-np.sum(np.where(p > 0.0, p * np.log(safe), 0.0)))


class SwitchingController:
    """Inference wrapper around a (calibrated) probabilistic ranker.

    Parameters
    ----------
    ranker
        Any object with `predict_proba(X) -> (n, K) ndarray`. Typically the
        `CalibratedRanker` from `models/calibration.py`.
    cfg_switching
        `cfg.ranker.switching` sub-tree. Reads `t_min_intervals` (int) and
        `entropy_gate_ratio` (float in [0, 1]).
    fefo_threshold
        `cfg.heuristics.fefo_mask_threshold`. State rows below this drop FEFO.
    feature_cols
        Column order used by `ranker.predict_proba`. Required only if you call
        the `select_from_dataframe` helper.
    heuristic_names
        Defaults to `simulation.heuristics.HEURISTIC_NAMES`. Override only in
        tests / ablations.
    """

    def __init__(
        self,
        ranker: _RankerLike,
        cfg_switching: "DictConfig",
        fefo_threshold: float,
        feature_cols: list[str] | None = None,
        heuristic_names: list[str] | None = None,
    ) -> None:
        self.ranker = ranker
        self.t_min = int(cfg_switching.t_min_intervals)
        self.entropy_gate_ratio = float(cfg_switching.entropy_gate_ratio)
        self.fefo_threshold = float(fefo_threshold)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.heuristic_names = list(heuristic_names or HEURISTIC_NAMES)
        self.K = len(self.heuristic_names)
        self.entropy_max = float(np.log(self.K))
        self._current: str | None = None
        self._dwell_remaining: int = 0
        self._last_probs: np.ndarray | None = None

    def reset(self) -> None:
        """Forget current heuristic and dwell state. Call between shifts."""
        self._current = None
        self._dwell_remaining = 0
        self._last_probs = None

    @property
    def current_heuristic(self) -> str | None:
        return self._current

    @property
    def dwell_remaining(self) -> int:
        return self._dwell_remaining

    @property
    def last_probs(self) -> np.ndarray | None:
        """Post-mask probability vector from the last `select` call (or None)."""
        return None if self._last_probs is None else self._last_probs.copy()

    def _apply_fefo_mask(self, probs: np.ndarray, pct_perishable: float) -> np.ndarray:
        masked = fefo_mask(
            probs.reshape(1, -1),
            np.array([pct_perishable], dtype=np.float64),
            threshold=self.fefo_threshold,
        )
        return masked[0]

    def select(self, features: np.ndarray) -> str:
        """Choose a heuristic for the given 1-D feature vector."""
        if features.ndim != 1:
            raise ValueError(f"features must be 1-D, got shape {features.shape}")
        raw_probs = self.ranker.predict_proba(features.reshape(1, -1))[0]
        masked = self._apply_fefo_mask(raw_probs, float(features[PCT_PERISHABLE_IDX]))
        self._last_probs = masked

        candidate = self.heuristic_names[int(np.argmax(masked))]

        if self._current is None or self._dwell_remaining <= 0:
            # No dwell constraint -- accept the greedy candidate.
            chosen = candidate
        else:
            entropy = _row_entropy_single(masked)
            confident = entropy < self.entropy_gate_ratio * self.entropy_max
            chosen = candidate if confident else self._current

        if chosen != self._current:
            self._current = chosen
            # T_min intervals INCLUDING the current one; decrement at the end.
            self._dwell_remaining = max(self.t_min - 1, 0)
        else:
            self._dwell_remaining = max(self._dwell_remaining - 1, 0)

        return chosen

    def select_sequence(self, feature_stream: np.ndarray) -> list[str]:
        """Reset, then `select` over each row of a (T, D) feature matrix."""
        if feature_stream.ndim != 2:
            raise ValueError(
                f"feature_stream must be 2-D, got shape {feature_stream.shape}"
            )
        self.reset()
        return [self.select(row) for row in feature_stream]


def select_for_dataframe(
    controller: SwitchingController,
    df,
    feature_cols: list[str],
    group_col: str = "shift_id",
) -> list[str]:
    """Run the controller per-shift over a flat DataFrame.

    Resets between groups so the dwell counter doesn't bleed across shifts.
    Rows are processed in their existing order; ensure the caller has sorted by
    `[group_col, interval_idx]` first.
    """
    import pandas as pd  # local to avoid module-level pandas import here

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    decisions: list[str] = [""] * len(df)
    for _, group in df.groupby(group_col, sort=False):
        controller.reset()
        feats = group[feature_cols].to_numpy(dtype=np.float64)
        chosen = [controller.select(row) for row in feats]
        for local_i, global_i in enumerate(group.index):
            decisions[df.index.get_loc(global_i)] = chosen[local_i]
    return decisions
