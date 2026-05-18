"""Phase 3 — unit tests for the snapshot labeler.

Pinning down two contracts:
  1. compute_costs_at_snapshot returns finite floats for every heuristic.
  2. Two calls with the same args produce identical costs (replay determinism).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from labeling.snapshot_labeler import (
    W_BREACH,
    W_TARDY,
    W_UNFINISHED,
    compute_costs_at_snapshot,
)
from simulation.heuristics import HEURISTIC_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


def _round_robin(n: int) -> list[str]:
    pool = HEURISTIC_NAMES
    return [pool[i % len(pool)] for i in range(n)]


def test_cost_weights_match_pilot():
    # Frozen by Phase 2 GATE: PASS — do not retune.
    assert W_BREACH == 3.0
    assert W_TARDY == 0.2
    assert W_UNFINISHED == 0.005


def test_costs_finite_for_all_heuristics(cfg):
    observed = _round_robin(32)
    costs = compute_costs_at_snapshot(
        seed=42,
        cfg=cfg,
        observed_policy=observed,
        t=10,
        candidate_heuristics=HEURISTIC_NAMES,
        tau=int(cfg.labeling.tau),
    )
    assert set(costs.keys()) == set(HEURISTIC_NAMES)
    for h, j in costs.items():
        assert math.isfinite(j), f"{h} cost not finite: {j}"


def test_costs_reproducible_across_calls(cfg):
    observed = _round_robin(32)
    args = dict(
        seed=42,
        cfg=cfg,
        observed_policy=observed,
        t=8,
        candidate_heuristics=HEURISTIC_NAMES,
        tau=int(cfg.labeling.tau),
    )
    a = compute_costs_at_snapshot(**args)
    b = compute_costs_at_snapshot(**args)
    assert a == b, f"non-deterministic costs: {a} vs {b}"


def test_costs_at_t0_and_at_last_interval(cfg):
    observed = _round_robin(32)
    # t=0 -> rollout starts from fresh env; should still produce finite costs.
    costs_t0 = compute_costs_at_snapshot(
        seed=42, cfg=cfg, observed_policy=observed, t=0,
        candidate_heuristics=HEURISTIC_NAMES, tau=int(cfg.labeling.tau),
    )
    # t = n_intervals - 1 -> rollout is clipped to a single step.
    costs_last = compute_costs_at_snapshot(
        seed=42, cfg=cfg, observed_policy=observed, t=31,
        candidate_heuristics=HEURISTIC_NAMES, tau=int(cfg.labeling.tau),
    )
    for h in HEURISTIC_NAMES:
        assert math.isfinite(costs_t0[h])
        assert math.isfinite(costs_last[h])
