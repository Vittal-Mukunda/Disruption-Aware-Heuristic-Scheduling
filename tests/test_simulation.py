"""Phase 1 acceptance gate.

All three core tests (plus the fast_forward replay test) must pass before
generating pilot data (Phase 2).

    pytest tests/test_simulation.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from simulation.heuristics import HEURISTIC_NAMES
from simulation.kpis import compute_kpis
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


def test_one_shift_under_5s(cfg):
    env = WarehouseEnv(seed=42, cfg=cfg)
    start = time.perf_counter()
    kpis = env.run_with_policy("FIFO")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"shift took {elapsed:.2f}s (limit 5s)"
    assert kpis["throughput"] > 0, "expected at least one order completed under FIFO"


def test_all_25_features_finite(cfg):
    env = WarehouseEnv(seed=42, cfg=cfg)
    for _ in range(env.n_intervals):
        state = env.step("FIFO")
        assert state.shape == (N_FEATURES,), f"shape mismatch: {state.shape}"
        assert np.all(np.isfinite(state)), (
            f"non-finite features at interval {env.interval_idx}: {state}"
        )


def test_determinism(cfg):
    env1 = WarehouseEnv(seed=42, cfg=cfg)
    env2 = WarehouseEnv(seed=42, cfg=cfg)
    k1 = env1.run_with_policy("FIFO")
    k2 = env2.run_with_policy("FIFO")
    assert k1 == k2, f"non-deterministic KPIs:\n  env1={k1}\n  env2={k2}"


def test_fast_forward_replay_matches_native(cfg):
    """fast_forward(T, history) must produce the same end state as stepping natively."""
    reps = (32 // len(HEURISTIC_NAMES)) + 1
    history = (HEURISTIC_NAMES * reps)[:32]
    assert len(history) == 32

    env_native = WarehouseEnv(seed=42, cfg=cfg)
    for h in history:
        env_native.step(h)
    k_native = compute_kpis(
        env_native.completed, env_native.queue, env_native.n_pickers, env_native.shift_minutes
    )

    env_ff = WarehouseEnv(seed=42, cfg=cfg)
    env_ff.fast_forward(t_intervals=32, policy_history=history)
    k_ff = compute_kpis(
        env_ff.completed, env_ff.queue, env_ff.n_pickers, env_ff.shift_minutes
    )

    assert k_native == k_ff
    assert env_native.policy_history == env_ff.policy_history
