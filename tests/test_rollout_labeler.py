"""Unit tests for the multi-sample rollout labeller.

Replaces `tests/test_snapshot_labeler.py`. That file asserted the submitted
labeller's two defining properties — that a re-run returned bit-identical costs,
and that the cost weights matched the pilot's — both of which were true because
every "stochastic rollout" replayed the single pre-sampled future belonging to
the shift seed. The estimator now has genuine sampling variation, so the
properties worth pinning are different ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from labeling.rollout_labeler import (
    behaviour_policy,
    costs_at_epoch,
    label_one_shift,
    rollout_seed,
    rollout_step_budget,
)
from simulation.cost import CostWeights, potential
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.state_extractor import FEATURE_NAMES
from simulation.warehouse_env import WarehouseEnv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return with_default_scales(OmegaConf.load(CONFIG_PATH))


@pytest.fixture(scope="module")
def pool(cfg):
    return resolve_pool(cfg)


# ---------------------------------------------------------------------------
# Branching — the mechanism that makes the label an estimator (Reviewer 2, 3)
# ---------------------------------------------------------------------------


def test_branch_preserves_history_and_redraws_the_future(cfg):
    env = WarehouseEnv(11, cfg)
    env.run_with_policy("FIFO", n_steps=8)
    served_before = list(env.completed)

    a = env.branch(rollout_seed(1, 11, 8, 0))
    b = env.branch(rollout_seed(1, 11, 8, 1))

    # Realised history is copied verbatim.
    assert [o.order_id for o in a.completed] == [o.order_id for o in served_before]
    assert a.t == env.t and a.interval_idx == env.interval_idx

    # The unrealised tail is redrawn independently, so two branches differ.
    tail_a = [o.arrival_time for o in a._all_orders[a._next_order_idx:]]
    tail_b = [o.arrival_time for o in b._all_orders[b._next_order_idx:]]
    assert tail_a != tail_b, "two branches drew the identical future"

    # Branching must not mutate the parent.
    assert env.interval_idx == 8
    assert len(env.completed) == len(served_before)


def test_branch_seeds_are_shared_across_rules_not_across_samples():
    """Common random numbers: the seed depends on (shift, epoch, sample), not the rule."""
    s1 = rollout_seed(20260804, 42, 3, 0)
    s2 = rollout_seed(20260804, 42, 3, 0)
    assert s1 == s2, "same (shift, epoch, sample) must give the same future"
    assert rollout_seed(20260804, 42, 3, 1) != s1
    assert rollout_seed(20260804, 42, 4, 0) != s1
    assert rollout_seed(20260804, 43, 3, 0) != s1


def test_costs_at_epoch_is_paired_and_leaves_env_untouched(cfg, pool):
    env = WarehouseEnv(7, cfg)
    env.run_with_policy("FIFO", n_steps=5)
    before = (env.interval_idx, env.t, len(env.completed), len(env.queue))

    costs = costs_at_epoch(env, 7, 5, pool, tau=2, n_samples=3, base_seed=1234)

    assert set(costs.mean) == set(pool)
    assert all(np.isfinite(v) for v in costs.mean.values())
    assert all(v >= 0.0 for v in costs.stderr.values())
    assert costs.best() in pool
    assert (env.interval_idx, env.t, len(env.completed), len(env.queue)) == before


def test_multi_sample_reduces_the_standard_error(cfg, pool):
    """The variance term Reviewer 2 (3) asks to see must actually be there."""
    env = WarehouseEnv(3, cfg)
    env.run_with_policy("FIFO", n_steps=4)

    one = costs_at_epoch(env, 3, 4, pool, tau=4, n_samples=1, base_seed=99)
    many = costs_at_epoch(env, 3, 4, pool, tau=4, n_samples=16, base_seed=99)

    # M=1 cannot estimate a standard error at all; M=16 must, and it must be
    # finite and non-degenerate for at least one rule (the futures differ).
    assert all(v == 0.0 for v in one.stderr.values())
    assert all(np.isfinite(v) for v in many.stderr.values())
    assert any(v > 0.0 for v in many.stderr.values()), (
        "no rule showed rollout variation across 16 independent futures — "
        "branch() is not redrawing the future"
    )


# ---------------------------------------------------------------------------
# The objective the label integrates
# ---------------------------------------------------------------------------


def test_dispatching_is_never_worse_than_abandoning(cfg):
    """The invariant that closes Reviewer 2 (1)'s loophole.

    An undispatched order is assessed at `t_ref + p_o` — the earliest it could
    possibly finish. So dispatching onto a free picker at `t_ref` costs exactly
    what leaving it costs, and dispatching earlier costs strictly less. Without
    the `+ p_o` the objective would reward inaction, which is the same pathology
    the submitted `W_u = 0.005` produced in subtler form.
    """
    from simulation.orders import Order

    w = CostWeights.from_cfg(cfg)
    t_ref = 100.0
    kwargs = dict(
        order_id=0, arrival_time=50.0, processing_time=6.0, sla_due=101.0,
        is_perishable=False, priority_class="high",
    )
    waiting = Order(**kwargs)
    dispatched_now = Order(**kwargs)
    dispatched_now.start_time = t_ref
    dispatched_now.finish_time = t_ref + kwargs["processing_time"]
    dispatched_early = Order(**kwargs)
    dispatched_early.start_time = t_ref - 20.0
    dispatched_early.finish_time = t_ref - 20.0 + kwargs["processing_time"]

    c_wait = potential([waiting], 1, t_ref, w)
    c_now = potential([dispatched_now], 0, t_ref, w)
    c_early = potential([dispatched_early], 0, t_ref, w)

    assert c_now <= c_wait + 1e-9, "abandoning an order was cheaper than serving it"
    assert c_early < c_wait, "serving earlier was not strictly better"


# ---------------------------------------------------------------------------
# Behaviour policy (Reviewer 1, 6.b)
# ---------------------------------------------------------------------------


def test_random_behaviour_policy_is_not_a_function_of_the_interval_index(pool):
    a = behaviour_policy("random", pool, 32, 1)
    b = behaviour_policy("random", pool, 32, 2)
    rr = behaviour_policy("round_robin", pool, 32, 1)
    assert len(a) == len(b) == len(rr) == 32
    assert set(a).issubset(set(pool))
    assert a != b, "the random behaviour policy ignored the shift seed"
    assert rr == [pool[i % len(pool)] for i in range(32)]


def test_unknown_behaviour_policy_raises(pool):
    with pytest.raises(ValueError):
        behaviour_policy("no_such_policy", pool, 32, 1)


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------


def test_label_one_shift_row_schema(cfg, pool):
    rows = label_one_shift(0, 5, cfg, tau=2, n_samples=2, candidates=pool)
    n_intervals = WarehouseEnv(5, cfg).n_intervals
    assert len(rows) == n_intervals

    r = rows[0]
    for name in FEATURE_NAMES:
        assert f"f_{name}" in r
    for h in pool:
        assert f"cost_{h}" in r and f"se_{h}" in r
    assert r["behaviour_rule"] in pool
    assert {"shift_id", "shift_seed", "interval_idx", "label_separation"} <= set(r)

    # The LAST LABELLED epoch is t = n_intervals - 1, where one step still
    # remains, so the rules do NOT tie there. The tie happens only at
    # t = n_intervals, which the labelling loop never reaches — assert it
    # against `costs_at_epoch` directly rather than against the last row.
    env_end = WarehouseEnv(5, cfg)
    env_end.run_with_policy("FIFO")
    terminal = costs_at_epoch(
        env_end, 5, env_end.n_intervals, pool, tau=2, n_samples=2, base_seed=1,
    )
    assert set(terminal.mean.values()) == {0.0}
    assert set(terminal.stderr.values()) == {0.0}


def test_rollout_step_budget_matches_the_closed_form():
    assert rollout_step_budget(
        n_shifts=2, n_intervals=32, n_rules=8, tau=4, n_samples=20
    ) == 2 * 32 + 2 * 32 * 8 * 20 * 4
