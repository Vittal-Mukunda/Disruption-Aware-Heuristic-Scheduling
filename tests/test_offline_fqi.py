"""Unit tests for the offline-FQI baseline (baselines/offline_fqi.py).

Acceptance: the transition logger produces correctly shaped transitions under
the configured behaviour policy; fitted Q-iteration trains and yields a usable
model; the policy returns valid rules and obeys the FEFO action mask; save/load
round-trips; and the trained baseline plugs into the evaluation harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.offline_fqi import (
    MASKED_RULE,
    PCT_PERISHABLE_IDX,
    OfflineFQIPolicy,
    _mask_fefo,
    _q_values,
    load_offline_fqi,
    log_transitions,
    save_offline_fqi,
    subset_transitions,
    train_fqi,
)
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from seed import shift_corpora
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.state_extractor import N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
N_SMOKE_SHIFTS = 3
_TINY_XGB = {"max_depth": 3, "n_estimators": 30, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def cfg():
    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    # The regime GMM belongs to a trained Stage-3 run that need not exist during
    # unit tests; the feature-parity path is exercised by the e9 driver.
    cfg.baselines.offline_fqi.use_regime_features = False
    return cfg


@pytest.fixture(scope="module")
def pool(cfg):
    return resolve_pool(cfg)


@pytest.fixture(scope="module")
def transitions(cfg, pool):
    seeds = shift_corpora(cfg)["train"][:N_SMOKE_SHIFTS]
    return log_transitions(seeds, cfg, pool)


@pytest.fixture(scope="module")
def fqi_model(cfg, transitions, pool):
    return train_fqi(
        transitions, gamma=0.95, n_iterations=3, xgb_params=_TINY_XGB,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        fefo_idx=pool.index(MASKED_RULE) if MASKED_RULE in pool else None,
        model_seed=int(cfg.seeds.model),
    )


def _policy(cfg, model, pool) -> OfflineFQIPolicy:
    return OfflineFQIPolicy(
        model=model,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        arms=pool,
    )


def test_log_transitions_shapes(transitions, pool):
    n = N_SMOKE_SHIFTS * 32
    assert transitions["states"].shape == (n, N_FEATURES)
    assert transitions["next_states"].shape == (n, N_FEATURES)
    assert transitions["actions"].shape == (n,)
    assert int(transitions["n_actions"][0]) == len(pool)
    # exactly one terminal transition per shift
    assert int(transitions["dones"].sum()) == N_SMOKE_SHIFTS
    assert np.isfinite(transitions["rewards"]).all()
    # Reward is the negated increment of a non-decreasing cost potential, so it
    # is non-positive. This is the property that makes "do nothing" unrewarding.
    assert (transitions["rewards"] <= 1e-9).all()
    assert set(np.unique(transitions["actions"])).issubset(set(range(len(pool))))


def test_behaviour_policy_is_not_a_function_of_the_interval_index(transitions):
    """Reviewer 1 (6.b): coverage must not be degenerate given the state.

    Under the submitted `a = interval_idx % |H|` scheme, every shift logs the
    identical action sequence, so conditional on the interval index — an
    observed feature — only one action is ever seen. Distinct sequences across
    shifts is the observable signature that this has been fixed.
    """
    per_shift = [
        transitions["actions"][transitions["shift_id"] == s]
        for s in np.unique(transitions["shift_id"])
    ]
    assert any(
        not np.array_equal(per_shift[0], other) for other in per_shift[1:]
    ), "every shift logged the same action sequence — coverage is still degenerate"


def test_subset_transitions(transitions):
    sub = subset_transitions(transitions, np.array([0, 2]))
    assert sub["states"].shape[0] == 2 * 32
    assert set(np.unique(sub["shift_id"])) == {0, 2}


def test_train_fqi_returns_usable_model(transitions, fqi_model, pool):
    q = _q_values(fqi_model, transitions["states"][:5], len(pool))
    assert q.shape == (5, len(pool))
    assert np.isfinite(q).all()


def test_policy_returns_valid_heuristic(cfg, transitions, fqi_model, pool):
    policy = _policy(cfg, fqi_model, pool)
    for state in transitions["states"][:20]:
        assert policy(state) in pool
    policy.reset()  # no-op, must not raise


def test_policy_rejects_wrong_shape(cfg, fqi_model, pool):
    policy = _policy(cfg, fqi_model, pool)
    with pytest.raises(ValueError):
        policy(np.zeros(N_FEATURES + 1, dtype=np.float64))


@pytest.mark.skipif(
    MASKED_RULE not in resolve_pool(with_default_scales(OmegaConf.load(CONFIG_PATH))),
    reason="FEFO is not in the deployed pool; the mask is a no-op by design.",
)
def test_fefo_mask_excludes_fefo_below_threshold(pool):
    fefo_idx = pool.index(MASKED_RULE)
    q = np.full((1, len(pool)), -10.0)
    q[0, fefo_idx] = 100.0
    non_perishable = np.zeros((1, N_FEATURES))  # pct_perishable = 0 < threshold
    masked = _mask_fefo(q, non_perishable, threshold=0.05, fefo_idx=fefo_idx)
    assert masked[0, fefo_idx] == -np.inf
    assert int(np.argmax(masked[0])) != fefo_idx
    # With enough perishables in the queue, FEFO stays available.
    perishable = np.zeros((1, N_FEATURES))
    perishable[0, PCT_PERISHABLE_IDX] = 0.5
    kept = _mask_fefo(q, perishable, threshold=0.05, fefo_idx=fefo_idx)
    assert kept[0, fefo_idx] == 100.0


def test_mask_is_a_noop_without_fefo(pool):
    q = np.zeros((1, len(pool)))
    assert np.array_equal(
        _mask_fefo(q, np.zeros((1, N_FEATURES)), threshold=0.05, fefo_idx=None), q
    )


def test_pct_perishable_index_is_resolved_by_name():
    """The submitted module hardcoded 4; revision inserted three expiry features."""
    from simulation.state_extractor import FEATURE_NAMES

    assert FEATURE_NAMES[PCT_PERISHABLE_IDX] == "pct_perishable"


def test_save_load_roundtrip(cfg, fqi_model, transitions, pool, tmp_path):
    threshold = float(cfg.heuristics.fefo_mask_threshold)
    save_offline_fqi(
        tmp_path, fqi_model,
        {"fefo_threshold": threshold, "arms": list(pool),
         "use_regime_features": False},
    )
    assert (tmp_path / "qmodel.joblib").exists()
    loaded = load_offline_fqi(tmp_path, cfg=cfg)
    assert loaded.fefo_threshold == threshold
    assert loaded.arms == pool
    ref = _policy(cfg, fqi_model, pool)
    for state in transitions["states"][:10]:
        assert loaded(state) == ref(state)


@pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "offline_fqi" / "qmodel.joblib").exists(),
    reason="offline_fqi model not present at runs/offline_fqi/",
)
def test_offline_fqi_via_evaluate_harness(cfg, tmp_path):
    """The trained baseline plugs into the evaluation harness for one shift."""
    policy = load_offline_fqi(REPO_ROOT / "runs" / "offline_fqi", cfg=cfg)
    seeds = canonical_test_seeds(cfg)[:1]
    df = evaluate_policy(
        "offline_fqi", policy, seeds, cfg, results_dir=tmp_path, save=True
    )
    assert len(df) == 1
    assert 0.0 <= df.iloc[0]["service_failure_rate"] <= 1.0
    assert (tmp_path / "offline_fqi.parquet").exists()
