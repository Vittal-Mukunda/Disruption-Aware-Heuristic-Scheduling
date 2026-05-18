"""Phase 5 — static (fixed) heuristic baselines.

One policy per name in `simulation.heuristics.HEURISTIC_NAMES`. The policy
ignores the state vector and always returns its bound heuristic name. These
are the floor of the Phase 5 comparison: any adaptive method must beat them
on SLA breach rate to be worth shipping.

Usage:

    from baselines.static import make_static_policy
    from experiments.evaluate import evaluate_policy, canonical_test_seeds

    cfg = OmegaConf.load("config.yaml")
    seeds = canonical_test_seeds(cfg)
    df = evaluate_policy("FIFO", make_static_policy("FIFO"), seeds, cfg)

Or use `run_static_baselines(seeds, cfg)` to evaluate all four in one call.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from simulation.heuristics import HEURISTIC_NAMES


def make_static_policy(name: str) -> Callable[[np.ndarray], str]:
    """Return `policy(state) -> name` for the given heuristic.

    Raises ValueError if `name` is not in the active pool. CR is intentionally
    excluded (see HANDOFF gotcha #16).
    """
    if name not in HEURISTIC_NAMES:
        raise ValueError(
            f"Heuristic '{name}' not in active pool {HEURISTIC_NAMES}. "
            f"CR was dropped in Phase 2 — see gotcha #16."
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
) -> dict[str, pd.DataFrame]:
    """Evaluate every heuristic in `HEURISTIC_NAMES` and persist one parquet each."""
    from experiments.evaluate import evaluate_policy

    out: dict[str, pd.DataFrame] = {}
    for h in HEURISTIC_NAMES:
        if verbose:
            print(f"[static] {h}")
        out[h] = evaluate_policy(
            h, make_static_policy(h), test_seeds, cfg,
            results_dir=results_dir, verbose=verbose,
        )
    return out
