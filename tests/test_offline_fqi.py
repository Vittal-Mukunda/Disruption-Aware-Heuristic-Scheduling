"""Unit tests for the offline-FQI baseline (baselines/offline_fqi.py).

Acceptance: the transition logger produces correctly shaped, round-robin
transitions; fitted Q-iteration trains and yields a usable model; the policy
returns valid heuristics and obeys the FEFO action mask; save/load round-trips
and the trained baseline plugs into the evaluation harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.offline_fqi import (
    FEFO_IDX,
    N_ACTIONS,
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
from seed import spawn_shift_seeds
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
N_SMOKE_SHIFTS = 3
_TINY_XGB = {"max_depth": 3, "n_estimators": 30, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def transitions(cfg):
    all_seeds = spawn_shift_seeds(
        n=int(cfg.labeling.n_train_shifts) + int(cfg.labeling.n_test_shifts),
        root=int(cfg.labeling.seed_seq_root),
    )
    return log_transitions(all_seeds[:N_SMOKE_SHIFTS], cfg)


@pytest.fixture(scope="module")
def fqi_model(cfg, transitions):
    return train_fqi(
        transitions, gamma=0.95, n_iterations=3, xgb_params=_TINY_XGB,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        model_seed=int(cfg.seeds.model),
    )


def test_log_transitions_shapes_and_roundrobin(transitions):
    n = N_SMOKE_SHIFTS * 32
    assert transitions["states"].shape == (n, N_FEATURES)
    assert transitions["next_states"].shape == (n, N_FEATURES)
    assert transitions["actions"].shape == (n,)
    # round-robin behaviour policy: interval i -> rule i % 4
    assert list(transitions["actions"][:8]) == [0, 1, 2, 3, 0, 1, 2, 3]
    # exactly one terminal transition per shift
    assert int(transitions["dones"].sum()) == N_SMOKE_SHIFTS
    assert np.isfinite(transitions["rewards"]).all()
    # reward = -composite_cost, so rewards are non-positive
    assert (transitions["rewards"] <= 0.0).all()


def test_subset_transitions(transitions):
    sub = subset_transitions(transitions, np.array([0, 2]))
    assert sub["states"].shape[0] == 2 * 32
    assert set(np.unique(sub["shift_id"])) == {0, 2}


def test_train_fqi_returns_usable_model(transitions, fqi_model):
    q = _q_values(fqi_model, transitions["states"][:5])
    assert q.shape == (5, N_ACTIONS)
    assert np.isfinite(q).all()


def test_policy_returns_valid_heuristic(cfg, transitions, fqi_model):
    policy = OfflineFQIPolicy(
        model=fqi_model, fefo_threshold=float(cfg.heuristics.fefo_mask_threshold)
    )
    for state in transitions["states"][:20]:
        assert policy(state) in HEURISTIC_NAMES
    policy.reset()  # no-op, must not raise


def test_policy_rejects_wrong_shape(cfg, fqi_model):
    policy = OfflineFQIPolicy(
        model=fqi_model, fefo_threshold=float(cfg.heuristics.fefo_mask_threshold)
    )
    with pytest.raises(ValueError):
        policy(np.zeros(N_FEATURES + 1, dtype=np.float64))


def test_fefo_mask_excludes_fefo_below_threshold():
    # FEFO is the greedy action by raw Q-value.
    q = np.full((1, N_ACTIONS), -10.0)
    q[0, FEFO_IDX] = 100.0
    non_perishable = np.zeros((1, N_FEATURES))  # pct_perishable = 0 < threshold
    masked = _mask_fefo(q, non_perishable, threshold=0.05)
    assert masked[0, FEFO_IDX] == -np.inf
    assert int(np.argmax(masked[0])) != FEFO_IDX
    # With enough perishables in the queue, FEFO stays available.
    perishable = np.zeros((1, N_FEATURES))
    perishable[0, PCT_PERISHABLE_IDX] = 0.5
    kept = _mask_fefo(q, perishable, threshold=0.05)
    assert kept[0, FEFO_IDX] == 100.0


def test_save_load_roundtrip(cfg, fqi_model, transitions, tmp_path):
    threshold = float(cfg.heuristics.fefo_mask_threshold)
    save_offline_fqi(tmp_path, fqi_model, {"fefo_threshold": threshold})
    assert (tmp_path / "qmodel.joblib").exists()
    loaded = load_offline_fqi(tmp_path, cfg=cfg)
    assert loaded.fefo_threshold == threshold
    ref = OfflineFQIPolicy(model=fqi_model, fefo_threshold=threshold)
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
    assert 0.0 <= df.iloc[0]["sla_breach_rate"] <= 1.0
    assert (tmp_path / "offline_fqi.parquet").exists()
