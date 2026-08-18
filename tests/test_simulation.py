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

from simulation.heuristics import HEURISTIC_NAMES, with_default_scales
from simulation.kpis import compute_kpis
from simulation.state_extractor import N_FEATURES
from simulation.warehouse_env import WarehouseEnv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return with_default_scales(OmegaConf.load(CONFIG_PATH))


def test_one_shift_under_5s(cfg):
    env = WarehouseEnv(seed=42, cfg=cfg)
    start = time.perf_counter()
    kpis = env.run_with_policy("FIFO")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"shift took {elapsed:.2f}s (limit 5s)"
    assert kpis["throughput"] > 0, "expected at least one order completed under FIFO"


def test_all_features_finite(cfg):
    """`step` returns None; the observation comes from `observe`.

    The submitted test read `state = env.step(...)`, which happened to work when
    `step` returned the next state. It does not: admission and observation are
    now one call (`observe`) precisely so that what a policy sees cannot depend
    on the call site.
    """
    env = WarehouseEnv(seed=42, cfg=cfg)
    for _ in range(env.n_intervals):
        state = env.observe()
        assert state.shape == (N_FEATURES,), f"shape mismatch: {state.shape}"
        assert np.all(np.isfinite(state)), (
            f"non-finite features at interval {env.interval_idx}: {state}"
        )
        env.step("FIFO")


def test_determinism(cfg):
    env1 = WarehouseEnv(seed=42, cfg=cfg)
    env2 = WarehouseEnv(seed=42, cfg=cfg)
    k1 = env1.run_with_policy("FIFO")
    k2 = env2.run_with_policy("FIFO")
    assert k1 == k2, f"non-deterministic KPIs:\n  env1={k1}\n  env2={k2}"


def _kpis(env) -> dict:
    """compute_kpis with the full argument list.

    `dropped` and `n_intervals` are required now: refused arrivals are counted
    as arrived demand rather than discarded, which is what closes the accounting
    loophole in Reviewer 2 (1).
    """
    return compute_kpis(
        env.completed, env.queue, env.dropped, env.n_pickers,
        env.shift_minutes, env.n_intervals, env.weights,
    )


def test_fast_forward_replay_matches_native(cfg):
    """fast_forward(T, history) must produce the same end state as stepping natively."""
    reps = (32 // len(HEURISTIC_NAMES)) + 1
    history = (HEURISTIC_NAMES * reps)[:32]
    assert len(history) == 32

    env_native = WarehouseEnv(seed=42, cfg=cfg)
    for h in history:
        env_native.observe()
        env_native.step(h)
    k_native = _kpis(env_native)

    env_ff = WarehouseEnv(seed=42, cfg=cfg)
    env_ff.fast_forward(t_intervals=32, policy_history=history)
    k_ff = _kpis(env_ff)

    assert k_native == k_ff
    assert env_native.policy_history == env_ff.policy_history


def test_arrived_partition_is_exhaustive(cfg):
    """served + unserved + dropped == arrived, at every shift end.

    The submitted KPI set reported breaches over *completed* orders only, so
    orders abandoned in the queue escaped the metric entirely. The partition is
    the structural guarantee that they cannot (Reviewer 2, 1).
    """
    env = WarehouseEnv(seed=7, cfg=cfg)
    k = env.run_with_policy("FIFO")
    assert k["arrived"] == k["throughput"] + k["unserved"] + k["dropped"]
    assert k["arrived"] == len(env.arrived_orders())
    assert 0.0 <= k["service_failure_rate"] <= 1.0


def test_unserved_not_yet_due_is_not_a_service_failure(cfg):
    """KPI accounting must not add p_o onto unserved orders.

    An order still waiting whose sla_due is after shift end is not overdue.
    The objective may still charge t_ref + p_o so inaction is not cheaper than
    dispatch; that charge must not leak into service_failure_rate.
    """
    from simulation.orders import Order

    o = Order(
        order_id=0, arrival_time=470.0, processing_time=12.0,
        sla_due=485.0, is_perishable=False, priority_class="low",
    )
    t_end = 480.0
    assert o.sla_due > t_end
    assert not o.is_overdue_at(t_end)
    assert not o.is_service_failure(t_end)
    # The objective still treats dispatch-now as the lower bound: t_end + p_o
    # (492) is past sla_due (485), so inaction is charged even though the
    # clock has not yet elapsed.
    assert o.is_late(t_end)
