"""Phase 3 — generate the full labeled dataset.

For each of `cfg.labeling.n_train_shifts + cfg.labeling.n_test_shifts` shifts:
  * Build `WarehouseEnv(seed, cfg)`.
  * Use a round-robin observed policy over `simulation.heuristics.HEURISTIC_NAMES`.
  * At every interval `t in [0, n_intervals)`:
      - `fast_forward(t, observed[:t])` and capture the 25-feature state.
      - For each heuristic h in the pool, replay then run `tau` more steps,
        record `J_h(s_t)` via `composite_cost`.

The aggregated long rows are turned into:
  * `data/train.parquet` — 250 * 32 = 8000 rows: ids + 25 features + 4 costs + 4 probs.
  * `data/test.parquet`  — 50 * 32 rows, ambiguity-filtered (`theta_confidence=0.55`).

Soft probabilities come from `costs_to_probs` (tempered softmax with beta-search
over the training set; the chosen beta is reused for the test set).

Usage:
    python -m experiments.generate_labels                          # full 250 + 50
    python -m experiments.generate_labels --n-train 5 --n-test 2   # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from labeling.ambiguity_filter import filter_ambiguous
from labeling.snapshot_labeler import compute_costs_at_snapshot
from labeling.soft_label_converter import costs_to_probs, fefo_mask
from seed import spawn_shift_seeds
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import FEATURE_NAMES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def label_one_shift(
    shift_id: int, shift_seed: int, cfg: DictConfig, tau: int | None = None,
) -> list[dict]:
    """Produce one row per interval for a single shift.

    `tau` defaults to `cfg.labeling.tau`; callers can override (e.g. snapshot_xgb
    ablation rebuilds the dataset at τ=1 without mutating the loaded cfg).
    """
    env = WarehouseEnv(int(shift_seed), cfg)
    n_intervals = env.n_intervals
    pool_len = len(HEURISTIC_NAMES)
    observed = [HEURISTIC_NAMES[i % pool_len] for i in range(n_intervals)]
    tau = int(cfg.labeling.tau) if tau is None else int(tau)

    rows: list[dict] = []
    for t in range(n_intervals):
        env.fast_forward(t_intervals=t, policy_history=observed[:t])
        feat = env.current_state()

        costs = compute_costs_at_snapshot(
            seed=int(shift_seed),
            cfg=cfg,
            observed_policy=observed,
            t=t,
            candidate_heuristics=HEURISTIC_NAMES,
            tau=tau,
            env=env,
        )

        row: dict = {
            "shift_id": int(shift_id),
            "shift_seed": int(shift_seed),
            "interval_idx": int(t),
        }
        for i, name in enumerate(FEATURE_NAMES):
            row[f"f_{name}"] = float(feat[i])
        for h in HEURISTIC_NAMES:
            row[f"cost_{h}"] = float(costs[h])
        rows.append(row)

    return rows


def _flatten(shift_rows: list[list[dict]]) -> pd.DataFrame:
    return pd.DataFrame([r for shift in shift_rows for r in shift])


def _attach_probs(
    df: pd.DataFrame, probs: np.ndarray
) -> pd.DataFrame:
    for i, h in enumerate(HEURISTIC_NAMES):
        df[f"p_{h}"] = probs[:, i]
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=None,
                        help="Override cfg.labeling.n_train_shifts (smoke testing).")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Override cfg.labeling.n_test_shifts (smoke testing).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="joblib n_jobs (default -1 = all cores).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override cfg.run_id; outputs go to runs/<run_id>/.")
    parser.add_argument("--tau", type=int, default=None,
                        help="Override cfg.labeling.tau (rollout horizon in intervals).")
    parser.add_argument("--train-out", type=Path, default=None,
                        help="Override train output path (default: data/train.parquet).")
    parser.add_argument("--test-out", type=Path, default=None,
                        help="Override test output path (default: data/test.parquet).")
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    n_train = int(args.n_train) if args.n_train is not None else int(cfg.labeling.n_train_shifts)
    n_test = int(args.n_test) if args.n_test is not None else int(cfg.labeling.n_test_shifts)
    tau_override = int(args.tau) if args.tau is not None else int(cfg.labeling.tau)
    n_total = n_train + n_test
    run_id = args.run_id or str(cfg.run_id)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 label generation: {n_train} train + {n_test} test shifts")
    print(f"  tau={tau_override}  pool={list(HEURISTIC_NAMES)}  "
          f"observed_policy=round_robin")
    print(f"  beta_grid={list(cfg.labeling.beta_grid)} "
          f"target_median_entropy={list(cfg.labeling.target_median_entropy)}")
    print(f"  fefo_mask_threshold={cfg.heuristics.fefo_mask_threshold}  "
          f"theta_confidence={cfg.labeling.ambiguity_filter.theta_confidence}")
    print(f"  n_jobs={args.n_jobs}  cpu_count={os.cpu_count()}")

    seeds = spawn_shift_seeds(n=n_total, root=int(cfg.labeling.seed_seq_root))
    train_seeds, test_seeds = seeds[:n_train], seeds[n_train:]

    t0 = time.perf_counter()
    print(f"\nLabeling {n_train} TRAIN shifts...")
    train_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(label_one_shift)(i, s, cfg, tau_override)
        for i, s in enumerate(train_seeds)
    )
    t_train = time.perf_counter() - t0
    print(f"  TRAIN labeling: {t_train:.1f}s  ({t_train/max(n_train,1):.2f}s/shift)")

    t0 = time.perf_counter()
    print(f"\nLabeling {n_test} TEST shifts...")
    test_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(label_one_shift)(n_train + i, s, cfg, tau_override)
        for i, s in enumerate(test_seeds)
    )
    t_test = time.perf_counter() - t0
    print(f"  TEST labeling: {t_test:.1f}s  ({t_test/max(n_test,1):.2f}s/shift)")

    train_df = _flatten(train_rows)
    test_df = _flatten(test_rows)
    print(f"\nAggregated: train={len(train_df)} rows, test={len(test_df)} rows")

    cost_cols = [f"cost_{h}" for h in HEURISTIC_NAMES]
    train_costs = train_df[cost_cols].to_numpy(dtype=np.float64)
    test_costs = test_df[cost_cols].to_numpy(dtype=np.float64)

    train_probs, beta = costs_to_probs(train_costs, cfg.labeling)
    test_probs, _ = costs_to_probs(test_costs, cfg.labeling, beta=beta)

    fefo_threshold = float(cfg.heuristics.fefo_mask_threshold)
    train_probs = fefo_mask(
        train_probs, train_df["f_pct_perishable"].to_numpy(dtype=np.float64),
        threshold=fefo_threshold,
    )
    test_probs = fefo_mask(
        test_probs, test_df["f_pct_perishable"].to_numpy(dtype=np.float64),
        threshold=fefo_threshold,
    )

    train_df = _attach_probs(train_df, train_probs)
    test_df = _attach_probs(test_df, test_probs)

    # Acceptance-criterion logging on TRAIN.
    median_entropy = float(np.median(_row_entropies(train_probs)))
    target_lo, target_hi = (float(x) for x in cfg.labeling.target_median_entropy)
    print(f"\nbeta chosen = {beta:.6f}")
    print(f"  median train row entropy = {median_entropy:.4f} "
          f"(target [{target_lo}, {target_hi}])")
    in_band = (target_lo <= median_entropy <= target_hi)
    print(f"  entropy band: {'OK' if in_band else 'OUT OF BAND'}")

    beta_path = run_dir / "phase3_beta.txt"
    beta_path.write_text(
        f"beta={beta}\n"
        f"median_train_entropy={median_entropy}\n"
        f"target_band=[{target_lo}, {target_hi}]\n"
        f"in_band={in_band}\n",
        encoding="utf-8",
    )
    print(f"  wrote {beta_path.relative_to(REPO_ROOT)}")

    # Filter test split for confident snapshots only.
    theta = float(cfg.labeling.ambiguity_filter.theta_confidence)
    keep_mask = filter_ambiguous(test_probs, theta=theta)
    n_kept = int(keep_mask.sum())
    n_dropped = len(test_df) - n_kept
    print(f"\nTEST ambiguity filter (theta={theta}): "
          f"kept {n_kept}/{len(test_df)} rows (dropped {n_dropped}).")
    test_df = test_df[keep_mask].reset_index(drop=True)

    train_path = Path(args.train_out) if args.train_out else DATA_DIR / "train.parquet"
    test_path = Path(args.test_out) if args.test_out else DATA_DIR / "test.parquet"
    train_path = train_path if train_path.is_absolute() else (REPO_ROOT / train_path).resolve()
    test_path = test_path if test_path.is_absolute() else (REPO_ROOT / test_path).resolve()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"\nSaved:")
    try:
        print(f"  {train_path.relative_to(REPO_ROOT)}  ({len(train_df)} rows)")
        print(f"  {test_path.relative_to(REPO_ROOT)}  ({len(test_df)} rows)")
    except ValueError:
        print(f"  {train_path}  ({len(train_df)} rows)")
        print(f"  {test_path}  ({len(test_df)} rows)")
    print(f"\nTotal wall-clock: {t_train + t_test:.1f}s")
    return 0


def _row_entropies(probs: np.ndarray) -> np.ndarray:
    safe = np.where(probs > 0.0, probs, 1.0)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=1)


if __name__ == "__main__":
    sys.exit(main())
