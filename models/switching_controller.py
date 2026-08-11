"""Switching controller — the deployment-time wrapper around the calibrated ranker.

Three responsibilities, applied in order at every decision epoch:

1. **FEFO mask.** Where the queue carries essentially no perishables, the
   expiry-aware rule cannot act on anything, so its probability mass is zeroed and
   the distribution renormalised. The same mask is applied at labelling time, so
   the ranker is never asked to model "FEFO on a queue with no product deadlines".

2. **Minimum dwell.** Once a rule is selected, hold it for at least
   `t_min_intervals` epochs. Rule thrashing is operationally disruptive — pickers
   are re-sequenced mid-shift — so this is a deployability constraint, not a
   performance device.

3. **Entropy gate.** Override the dwell when the post-mask distribution is
   confident, `H(p) < ratio * log|H|`. A clear winner is allowed to interrupt a
   dwell; a marginal one is not.

INSTRUMENTATION (Reviewer 3, comment 3)
---------------------------------------
    "Does DAHS's rule-selection distribution shift significantly under
     saturation? Could the minimum dwell constraint of the switching controller
     prevent timely adaptation in this regime?"

The submitted controller kept no record of its own decisions, so neither question
could be answered from a run — the dwell either helped or hurt in aggregate and
there was no way to see which epochs it bound at. Every `select()` call now
appends a row to `self.trace` recording what the ranker wanted, what the
controller did, and why they differed. `blocked_switch_rate()` is the direct
answer to the second question: the share of epochs at which the arg-max rule
differed from the deployed rule *because the dwell was still active*. If that rate
rises under saturation, the dwell is preventing timely adaptation and the
scenario-level KPI gap has a named cause rather than an attribution.

`experiments/saturation_analysis.py` consumes the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from labeling.soft_label_converter import fefo_mask
from simulation.state_extractor import FEATURE_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig

PCT_PERISHABLE_IDX: int = FEATURE_NAMES.index("pct_perishable")


class _RankerLike(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log(nz))) if nz.size else 0.0


@dataclass
class SwitchDecision:
    """One epoch of the controller's reasoning, for post-hoc analysis."""

    interval_idx: int
    wanted: str          # arg-max of the calibrated, masked distribution
    chosen: str          # what was actually deployed
    switched: bool
    dwell_remaining: int
    entropy: float
    gate_open: bool      # the entropy gate would permit a mid-dwell switch
    blocked: bool        # wanted != chosen, and the dwell is why


class SwitchingController:
    """Inference wrapper around a (calibrated) probabilistic ranker.

    Parameters
    ----------
    ranker
        Any object exposing `predict_proba(X) -> (n, K)`.
    cfg_switching
        `cfg.ranker.switching`: `t_min_intervals`, `entropy_gate_ratio`.
    fefo_threshold
        Mask FEFO below this queue perishable fraction.
    heuristic_names
        The active pool, in the ranker's class order. Required — the submitted
        version defaulted to a module-level constant fixed at four rules, which
        silently mismatched any run with a rescreened pool.
    """

    def __init__(
        self,
        ranker: _RankerLike,
        cfg_switching: "DictConfig",
        fefo_threshold: float,
        heuristic_names: list[str],
        feature_cols: list[str] | None = None,
    ) -> None:
        if not heuristic_names:
            raise ValueError("heuristic_names must be the active rule pool")
        self.ranker = ranker
        self.t_min = int(cfg_switching.t_min_intervals)
        self.entropy_gate_ratio = float(cfg_switching.entropy_gate_ratio)
        self.fefo_threshold = float(fefo_threshold)
        self.heuristic_names = list(heuristic_names)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.K = len(self.heuristic_names)
        self.entropy_max = float(np.log(self.K))

        self._current: str | None = None
        self._dwell_remaining: int = 0
        self._last_probs: np.ndarray | None = None
        self._epoch: int = 0
        self.trace: list[SwitchDecision] = []

    # ------------------------------------------------------------------

    def reset(self, keep_trace: bool = True) -> None:
        """Clear dwell state between shifts. The trace accumulates by default."""
        self._current = None
        self._dwell_remaining = 0
        self._last_probs = None
        self._epoch = 0
        if not keep_trace:
            self.trace = []

    @property
    def current_heuristic(self) -> str | None:
        return self._current

    @property
    def dwell_remaining(self) -> int:
        return self._dwell_remaining

    @property
    def last_probs(self) -> np.ndarray | None:
        return None if self._last_probs is None else self._last_probs.copy()

    # ------------------------------------------------------------------

    def select(self, features: np.ndarray) -> str:
        if features.ndim != 1:
            raise ValueError(f"features must be 1-D, got shape {features.shape}")

        raw = self.ranker.predict_proba(features.reshape(1, -1))[0]
        masked = fefo_mask(
            raw.reshape(1, -1),
            np.array([float(features[PCT_PERISHABLE_IDX])], dtype=np.float64),
            threshold=self.fefo_threshold,
            heuristic_names=self.heuristic_names,
        )[0]
        self._last_probs = masked

        wanted = self.heuristic_names[int(np.argmax(masked))]
        ent = _entropy(masked)
        gate_open = ent < self.entropy_gate_ratio * self.entropy_max
        dwell_before = self._dwell_remaining

        if self._current is None or dwell_before <= 0:
            chosen = wanted
        else:
            chosen = wanted if gate_open else self._current

        blocked = (wanted != chosen)
        switched = chosen != self._current and self._current is not None

        if chosen != self._current:
            self._current = chosen
            # T_min epochs *including* this one, so decrement by one here.
            self._dwell_remaining = max(self.t_min - 1, 0)
        else:
            self._dwell_remaining = max(self._dwell_remaining - 1, 0)

        self.trace.append(
            SwitchDecision(
                interval_idx=self._epoch,
                wanted=wanted,
                chosen=chosen,
                switched=switched,
                dwell_remaining=dwell_before,
                entropy=ent,
                gate_open=gate_open,
                blocked=blocked,
            )
        )
        self._epoch += 1
        return chosen

    def select_sequence(self, feature_stream: np.ndarray) -> list[str]:
        if feature_stream.ndim != 2:
            raise ValueError(
                f"feature_stream must be 2-D, got shape {feature_stream.shape}"
            )
        self.reset()
        return [self.select(row) for row in feature_stream]

    # ------------------------------------------------------------------
    # Diagnostics (Reviewer 3, comment 3)
    # ------------------------------------------------------------------

    def selection_distribution(self) -> dict[str, float]:
        """Share of epochs at which each rule was deployed.

        The quantity the reviewer asks about directly: whether the selector's
        behaviour shifts between operating regimes, or collapses onto one rule.
        """
        if not self.trace:
            return {h: 0.0 for h in self.heuristic_names}
        counts = {h: 0 for h in self.heuristic_names}
        for d in self.trace:
            counts[d.chosen] += 1
        n = len(self.trace)
        return {h: c / n for h, c in counts.items()}

    def selection_entropy(self) -> float:
        """Exponentiated entropy of the deployed-rule distribution.

        Equals |H| when the selector spreads uniformly and 1.0 when it has
        collapsed onto a single rule. Reported per scenario so that "the policy
        adapts" is a measurement rather than a description.
        """
        p = np.array(list(self.selection_distribution().values()), dtype=np.float64)
        nz = p[p > 0]
        return float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0

    def switch_rate(self) -> float:
        return float(np.mean([d.switched for d in self.trace])) if self.trace else 0.0

    def blocked_switch_rate(self) -> float:
        """Share of epochs where the dwell overrode the ranker's preference.

        This is the answer to "could the minimum dwell prevent timely adaptation":
        a high value under saturation means the controller is holding a rule the
        ranker has already abandoned.
        """
        return float(np.mean([d.blocked for d in self.trace])) if self.trace else 0.0

    def gate_open_rate(self) -> float:
        """How often the ranker was confident enough to interrupt a dwell."""
        return float(np.mean([d.gate_open for d in self.trace])) if self.trace else 0.0

    def trace_frame(self):
        """The decision trace as a DataFrame, for per-epoch analysis."""
        import pandas as pd

        return pd.DataFrame([d.__dict__ for d in self.trace])


def select_for_dataframe(
    controller: SwitchingController,
    df,
    feature_cols: list[str],
    group_col: str = "shift_id",
) -> list[str]:
    """Run the controller per shift over a flat frame, resetting between groups."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    decisions: list[str] = [""] * len(df)
    positions = {idx: i for i, idx in enumerate(df.index)}
    for _, group in df.groupby(group_col, sort=False):
        controller.reset()
        feats = group[feature_cols].to_numpy(dtype=np.float64)
        for local_i, global_i in enumerate(group.index):
            decisions[positions[global_i]] = controller.select(feats[local_i])
    return decisions
