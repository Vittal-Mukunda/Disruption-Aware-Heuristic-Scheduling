"""Phase 6 smoke tests — stats helpers + each experiment's CLI shape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from experiments.stats import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    compare_methods,
    paired_wilcoxon,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def test_bootstrap_mean_ci_covers_truth():
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=1.0, scale=0.2, size=80)
    ci = bootstrap_mean_ci(vals, n_resamples=2000, seed=7)
    assert ci.lo < 1.0 < ci.hi
    assert ci.n == 80
    assert ci.n_resamples == 2000


def test_bootstrap_paired_diff_zero_under_equal_distributions():
    rng = np.random.default_rng(99)
    a = rng.normal(0.0, 1.0, size=60)
    b = a.copy()  # identical
    ci = bootstrap_paired_diff_ci(a, b, n_resamples=1500, seed=11)
    assert ci.point == pytest.approx(0.0, abs=1e-9)
    assert ci.lo == pytest.approx(0.0, abs=1e-9)
    assert ci.hi == pytest.approx(0.0, abs=1e-9)


def test_paired_wilcoxon_detects_shift():
    rng = np.random.default_rng(123)
    a = rng.normal(0.0, 1.0, size=50)
    b = a + 0.5
    _, p = paired_wilcoxon(a, b)
    assert p < 0.01


def test_paired_wilcoxon_degenerate_returns_one():
    a = np.ones(20)
    b = np.ones(20)
    stat, p = paired_wilcoxon(a, b)
    assert stat == 0.0
    assert p == 1.0


def test_benjamini_hochberg_step_up():
    p = np.array([0.001, 0.04, 0.03, 0.20, 0.50])
    reject, p_adj = benjamini_hochberg(p, q=0.05)
    # smallest two should reject; others not
    assert reject[0]
    assert not reject[-1]
    # monotonicity after order restoration
    order = np.argsort(p)
    assert np.all(np.diff(p_adj[order]) >= -1e-12)


def test_compare_methods_runs_on_phase5_parquets():
    df_long_parts = []
    for m in ("fifo", "ours"):
        path = RESULTS_DIR / f"{m}.parquet"
        if not path.exists():
            pytest.skip(f"missing {path}")
        df_long_parts.append(pd.read_parquet(path).assign(method=m))
    df_long = pd.concat(df_long_parts, ignore_index=True)
    out = compare_methods(
        df_long, metric="composite_cost", baseline="ours",
        n_resamples=1000,
    )
    assert set(out["method"]) == {"fifo", "ours"}
    fifo_row = out[out["method"] == "fifo"].iloc[0]
    assert fifo_row["p_raw"] < 0.05  # the gap is huge; should be tiny p
    assert fifo_row["reject_bh"]


def test_diversity_grid_runs_on_stage1_costs():
    """Replaces the E1 per-shift heatmap (Reviewer 1, 4.e).

    The submitted Figure 1 varied the INSTANCE (shift, interval) and so said
    nothing about complementarity across the STATE SPACE. Its driver and its
    `data/pilot_costs.parquet` input are both gone; the replacement bins win
    rate over queue length x deadline pressure and is produced by Stage 1.
    """
    costs = (REPO_ROOT / "results" / "S1_calibration"
             / "calibration_epoch_costs.parquet")
    if not costs.exists():
        pytest.skip(f"missing {costs}; run `make stage1-screen` first")
    cmd = [sys.executable, "-m", "experiments.calibrate_rules", "diversity"]
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert (REPO_ROOT / "results" / "S1_calibration"
            / "diversity_state_grid.parquet").exists()


def test_e2_stats_runs_on_existing_parquets():
    if not (RESULTS_DIR / "ours.parquet").exists():
        pytest.skip("results/ours.parquet missing")
    cmd = [
        sys.executable, "-m", "experiments.e2_main", "stats",
        "--scenario", "default",
        "--methods", "fifo", "ours",
        "--metrics", "composite_cost",
    ]
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert (RESULTS_DIR / "E2" / "default_stats.parquet").exists()


def test_e2_apply_scenario_overlays_arrival_rate():
    from experiments.e2_main import apply_scenario
    cfg = OmegaConf.load(REPO_ROOT / "config.yaml")
    base_rate = float(cfg.sim.arrivals.base_rate_per_minute)
    new_cfg = apply_scenario(cfg, "low_load")
    assert float(new_cfg.sim.arrivals.base_rate_per_minute) == 1.0
    # Default unchanged.
    same = apply_scenario(cfg, "default")
    assert float(same.sim.arrivals.base_rate_per_minute) == base_rate


def test_e3_no_calibration_policy_returns_valid_heuristic():
    from baselines.ours import REPO_ROOT as _ROOT
    from experiments.e3_ablations import load_no_calibration
    from simulation.heuristics import HEURISTIC_NAMES, with_default_scales
    from simulation.warehouse_env import WarehouseEnv
    run_dir = _ROOT / "runs" / "phase4"
    if not run_dir.exists():
        pytest.skip("runs/phase4 not present")
    policy = load_no_calibration(run_dir)
    cfg = with_default_scales(OmegaConf.load(REPO_ROOT / "config.yaml"))
    env = WarehouseEnv(42, cfg)
    h = policy(env.current_state())
    assert h in HEURISTIC_NAMES


def test_e3_no_switching_policy_argmax_equivalent():
    """T_min=0 means dwell never fires; controller always returns the argmax."""
    from experiments.e3_ablations import load_no_switching
    from simulation.heuristics import HEURISTIC_NAMES, with_default_scales
    from simulation.warehouse_env import WarehouseEnv
    run_dir = REPO_ROOT / "runs" / "phase4"
    if not run_dir.exists():
        pytest.skip("runs/phase4 not present")
    policy = load_no_switching(run_dir)
    cfg = with_default_scales(OmegaConf.load(REPO_ROOT / "config.yaml"))
    env = WarehouseEnv(42, cfg)
    h = policy(env.current_state())
    assert h in HEURISTIC_NAMES


def test_e4_t_min_one_value(tmp_path):
    if not (REPO_ROOT / "runs" / "phase4").exists():
        pytest.skip("runs/phase4 not present")
    cmd = [
        sys.executable, "-m", "experiments.e4_sensitivity",
        "t_min", "--values", "0", "--n-test", "2",
    ]
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_e5_reliability_table_present():
    """If E5 has been run, the calibration table should exist with 3 metrics."""
    path = REPO_ROOT / "results" / "E5" / "calibration_table.parquet"
    if not path.exists():
        pytest.skip("E5 not run yet")
    df = pd.read_parquet(path)
    assert set(df["metric"]) == {"ece", "brier", "soft_xent"}
    assert (df["post"] >= 0).all()


def test_e5_shap_importance_present():
    path = REPO_ROOT / "results" / "E5" / "shap_global_importance.parquet"
    if not path.exists():
        pytest.skip("E5 SHAP not run yet")
    df = pd.read_parquet(path)
    assert (df["mean_abs_shap"] >= 0).all()
    assert len(df) >= 25  # 25 state features (regime cols may or may not be present)
