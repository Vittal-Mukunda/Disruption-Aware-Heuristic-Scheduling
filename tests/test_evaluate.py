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
from simulation.heuristics import HEURISTIC_NAMES, with_default_scales
from simulation.state_extractor import N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
PHASE4_DIR = REPO_ROOT / "runs" / "phase4"
PHASE4_TAU1_DIR = REPO_ROOT / "runs" / "phase4_tau1"
PPO_DIR = REPO_ROOT / "runs" / "ppo_fair"


def _usable_ranker(run_dir: Path) -> bool:
    """True only for a run trained by the CURRENT Stage 3.

    A pre-revision run directory is present but unusable: its `ranker_meta`
    carries no `classes`, so `load_ours` refuses it rather than guessing the
    class order. That refusal is the guard working, not a test failure — skip
    instead, and let `make clean-stale` be the fix.
    """
    import joblib

    if not (run_dir / "calibrator.joblib").exists():
        return False
    meta_path = run_dir / "ranker_meta.joblib"
    if not meta_path.exists():
        return False
    try:
        return bool(joblib.load(meta_path).get("classes"))
    except Exception:
        return False


def _usable_ppo(run_dir: Path) -> bool:
    """Pre-revision PPO runs have no `ppo_meta.json` recording the action set."""
    return (run_dir / "ppo_fair.zip").exists() and (run_dir / "ppo_meta.json").exists()


@pytest.fixture(scope="module")
def cfg():
    """Config with provisional ATC/COVERT scales filled in.

    Committed `config.yaml` carries fitted ATC/COVERT scales (`3.0` / `4.0`).
    `with_default_scales` is still used so tests remain valid if a scale is unset.
    """
    return with_default_scales(OmegaConf.load(CONFIG_PATH))


@pytest.fixture(scope="module")
def one_test_seed(cfg):
    return canonical_test_seeds(cfg)[:1]


def _assert_kpi_row(row: dict) -> None:
    """The row must have every required column and every numeric value finite."""
    for col in KPI_COLUMNS:
        assert col in row, f"missing column {col!r}"
    assert row["throughput"] > 0, "expected some orders completed"
    assert 0.0 <= row["service_failure_rate"] <= 1.0
    assert 0.0 <= row["sla_breach_rate_arrived"] <= 1.0
    assert 0.0 <= row["sla_breach_rate_served"] <= 1.0
    assert 0.0 <= row["spoilage_rate"] <= 1.0
    assert 0.0 <= row["picker_utilization"] <= 1.0
    assert row["mean_tardiness"] >= 0.0
    assert row["composite_cost"] >= 0.0
    assert row["wall_clock_s"] > 0.0
    # The outcome partition must account for every arrived order (Reviewer 2, 1).
    assert row["arrived"] == pytest.approx(
        row["throughput"] + row["unserved"] + row["dropped"]
    )
    # An unserved order can only ever hurt, so counting arrivals cannot report a
    # lower failure rate than counting completions alone.
    assert row["sla_breach_rate_arrived"] >= 0.0
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


def test_make_static_policy_accepts_any_library_rule():
    """A screened-out rule is still a legitimate standalone benchmark.

    The submitted module rejected any name outside the deployed four-rule pool,
    so CR — dropped by the original pilot — could not be benchmarked at all, and
    the screen's verdict was therefore unauditable (Reviewer 1, 4.b/4.d).
    """
    assert make_static_policy("CR")(np.zeros(N_FEATURES)) == "CR"
    assert make_static_policy("SPT")(np.zeros(N_FEATURES)) == "SPT"


def test_make_static_policy_rejects_unknown_rule():
    with pytest.raises(ValueError, match="Unknown heuristic"):
        make_static_policy("NOT_A_RULE")


@pytest.mark.skipif(
    not _usable_ranker(PHASE4_DIR),
    reason="no current-revision ranker at runs/phase4/ (run make clean-stale + the pipeline)",
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
    not _usable_ranker(PHASE4_DIR),
    reason="no current-revision ranker at runs/phase4/ (run make clean-stale + the pipeline)",
)
def test_ours_reset_clears_dwell(cfg):
    from baselines.ours import load_ours

    policy = load_ours(PHASE4_DIR, cfg=cfg)
    assert policy.controller.current_heuristic is None
    dummy_state = np.zeros(N_FEATURES, dtype=np.float64)
    _ = policy(dummy_state)
    assert policy.controller.current_heuristic is not None
    policy.reset()
    assert policy.controller.current_heuristic is None


def test_greedy_mpc_one_shift(cfg, one_test_seed, tmp_path):
    """Env-aware greedy 1-step MPC: smoke + parquet round-trip.

    `baselines/greedy_mpc.py` is gone: it scored with the deleted
    `snapshot_labeler.composite_cost` and evaluated the single pre-sampled
    future. The tau=1 case is now a named alias of the rolling-horizon teacher,
    so both MPC baselines share one estimator (Reviewer 2, 6).
    """
    from baselines.rolling_horizon_mpc import make_greedy_mpc_policy

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
    not _usable_ranker(PHASE4_TAU1_DIR),
    reason="no current-revision ranker at runs/phase4_tau1/ (run make tau1)",
)
def test_snapshot_xgb_one_shift(cfg, one_test_seed, tmp_path):
    """snapshot_xgb (tau=1 ablation of OURS): smoke + parquet round-trip."""
    from baselines.snapshot_xgb import load_snapshot_xgb

    policy = load_snapshot_xgb(PHASE4_TAU1_DIR, cfg=cfg)
    assert policy.parallel_safe is True
    df = evaluate_policy(
        "snapshot_xgb", policy, one_test_seed, cfg,
        results_dir=tmp_path, save=True,
    )
    assert len(df) == 1
    _assert_kpi_row(df.iloc[0].to_dict())
    assert (tmp_path / "snapshot_xgb.parquet").exists()


@pytest.mark.skipif(
    not _usable_ppo(PPO_DIR),
    reason="no current-revision PPO at runs/ppo_fair/ (missing ppo_meta.json)",
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
