"""Phase 5 — smoke tests for the evaluation harness and the OURS policy.

Acceptance for this turn:
  - `evaluate_policy` on 1 test shift with the static FIFO policy returns a
    one-row DataFrame with the canonical Phase 5 schema and finite KPIs.
  - `evaluate_policy` on the same test shift with the OURS policy loaded from
    `runs/phase4/` also returns a finite, schema-conformant row.

We don't compare numbers — the harness is what we're verifying. Method-vs-method
comparisons are Phase 6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.static import make_static_policy
from experiments.evaluate import (
    KPI_COLUMNS,
    canonical_test_seeds,
    evaluate_policy,
    evaluate_policy_env_aware,
)
from simulation.heuristics import HEURISTIC_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
PHASE4_DIR = REPO_ROOT / "runs" / "phase4"
PHASE4_TAU1_DIR = REPO_ROOT / "runs" / "phase4_tau1"
PPO_DIR = REPO_ROOT / "runs" / "ppo_fair"


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def one_test_seed(cfg):
    return canonical_test_seeds(cfg)[:1]


def _assert_kpi_row(row: dict) -> None:
    """The row must have every required column and every numeric value finite."""
    for col in KPI_COLUMNS:
        assert col in row, f"missing column {col!r}"
    assert row["throughput"] > 0, "expected some orders completed"
    assert 0.0 <= row["sla_breach_rate"] <= 1.0
    assert 0.0 <= row["spoilage_rate"] <= 1.0
    assert 0.0 <= row["picker_utilization"] <= 1.0
    assert row["mean_tardiness"] >= 0.0
    assert row["mean_cost"] >= 0.0
    assert row["wall_clock_s"] > 0.0
    for col in KPI_COLUMNS:
        assert np.isfinite(row[col]), f"{col} = {row[col]!r} not finite"


def test_static_fifo_one_shift(cfg, one_test_seed, tmp_path):
    policy = make_static_policy("FIFO")
    df = evaluate_policy(
        "FIFO", policy, one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "FIFO.parquet").exists()


def test_static_all_heuristics_smoke(cfg, one_test_seed, tmp_path):
    for h in HEURISTIC_NAMES:
        df = evaluate_policy(
            h, make_static_policy(h), one_test_seed, cfg,
            results_dir=tmp_path, save=False,
        )
        assert len(df) == 1, f"{h}: expected 1 row"
        _assert_kpi_row(df.iloc[0].to_dict())


def test_make_static_policy_rejects_cr():
    with pytest.raises(ValueError, match="CR"):
        make_static_policy("CR")


@pytest.mark.skipif(
    not (PHASE4_DIR / "calibrator.joblib").exists(),
    reason="Phase 4 artifacts not present at runs/phase4/",
)
def test_ours_one_shift(cfg, one_test_seed, tmp_path):
    from baselines.ours import load_ours

    policy = load_ours(PHASE4_DIR, cfg=cfg)
    df = evaluate_policy(
        "ours", policy, one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "ours.parquet").exists()


@pytest.mark.skipif(
    not (PHASE4_DIR / "calibrator.joblib").exists(),
    reason="Phase 4 artifacts not present at runs/phase4/",
)
def test_ours_reset_clears_dwell(cfg):
    from baselines.ours import load_ours

    policy = load_ours(PHASE4_DIR, cfg=cfg)
    assert policy.controller.current_heuristic is None
    dummy_state = np.zeros(25, dtype=np.float64)
    _ = policy(dummy_state)
    assert policy.controller.current_heuristic is not None
    policy.reset()
    assert policy.controller.current_heuristic is None


def test_greedy_mpc_one_shift(cfg, one_test_seed, tmp_path):
    """Env-aware greedy 1-step MPC: smoke + parquet round-trip."""
    from baselines.greedy_mpc import make_greedy_mpc_policy

    df = evaluate_policy_env_aware(
        "greedy_mpc", make_greedy_mpc_policy(), one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "greedy_mpc.parquet").exists()


def test_linucb_one_shift(cfg, one_test_seed, tmp_path):
    """LinUCB env-aware policy: smoke + parquet round-trip."""
    from baselines.linucb import make_linucb_policy

    df = evaluate_policy_env_aware(
        "linucb", make_linucb_policy(), one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "linucb.parquet").exists()


@pytest.mark.skipif(
    not (PHASE4_TAU1_DIR / "calibrator.joblib").exists(),
    reason="snapshot_xgb (phase4_tau1) artifacts not present at runs/phase4_tau1/",
)
def test_snapshot_xgb_one_shift(cfg, one_test_seed, tmp_path):
    """snapshot_xgb (tau=1 ablation of OURS): smoke + parquet round-trip."""
    from baselines.snapshot_xgb import load_snapshot_xgb

    policy = load_snapshot_xgb(PHASE4_TAU1_DIR, cfg=cfg)
    df = evaluate_policy(
        "snapshot_xgb", policy, one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "snapshot_xgb.parquet").exists()


@pytest.mark.skipif(
    not (PPO_DIR / "ppo_fair.zip").exists(),
    reason="PPO-fair model not present at runs/ppo_fair/",
)
def test_ppo_fair_one_shift(cfg, one_test_seed, tmp_path):
    """PPO-fair deterministic eval: smoke + parquet round-trip."""
    from baselines.ppo_fair import load_ppo_fair

    policy = load_ppo_fair(PPO_DIR, cfg=cfg)
    df = evaluate_policy(
        "ppo_fair", policy, one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "ppo_fair.parquet").exists()
