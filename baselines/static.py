"""Static (fixed) dispatching-rule baselines.

One policy per rule: it ignores the state and always returns its bound name.
These are the floor of the comparison — any adaptive method must beat the best
of them on the composite objective to be worth shipping.

Validity is checked against the whole rule LIBRARY (`simulation.heuristics.
HEURISTICS`), not against the deployed pool. A rule the Stage-1 screen dropped
from the selector's action set is still a legitimate standalone benchmark, and
reporting it is what makes the screen's verdict auditable — the submitted module
rejected any name outside the four-rule pool, so a screened-out rule could not
be benchmarked at all.

Usage:

    from baselines.static import make_static_policy
    from experiments.evaluate import evaluate_policy, canonical_test_seeds

    cfg = OmegaConf.load("config.yaml")
    seeds = canonical_test_seeds(cfg)
    df = evaluate_policy("FIFO", make_static_policy("FIFO"), seeds, cfg)

`run_static_baselines(seeds, cfg)` evaluates the deployed pool in one call.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from simulation.heuristics import HEURISTICS, resolve_pool


def make_static_policy(name: str) -> Callable[[np.ndarray], str]:
    """Return `policy(state) -> name` for the given rule."""
    if name not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {sorted(HEURISTICS)}"
        )
    bound = str(name)

    def policy(state: np.ndarray) -> str:  # noqa: ARG001 - state ignored by design
        return bound

    policy.__name__ = f"static_{bound.lower()}"
    return policy


def run_static_baselines(
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None = None,
    verbose: bool = False,
    rules: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate each rule in `rules` (default: the deployed pool); one parquet each."""
    from experiments.evaluate import evaluate_policy

    out: dict[str, pd.DataFrame] = {}
    for h in (rules if rules is not None else resolve_pool(cfg)):
        if verbose:
            print(f"[static] {h}")
        out[h] = evaluate_policy(
            h, make_static_policy(h), test_seeds, cfg,
            results_dir=results_dir, verbose=verbose,
        )
    return out
