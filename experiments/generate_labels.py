"""Stage 2 — build the labelled corpus.

Orchestration only. The estimator lives in `labeling.rollout_labeler`, the
objective in `simulation.cost`, and the pool in `simulation.heuristics`; this
module walks the shift corpora, converts cost vectors to soft labels, and
persists the two parquets every downstream stage reads.

    python -m experiments.generate_labels                     # full corpus
    python -m experiments.generate_labels --n-train 3 --n-test 2   # smoke test

WHAT CHANGED IN REVISION
------------------------
1. THE LABEL IS AN ESTIMATOR (Reviewer 2, 3). The submitted driver called
   `snapshot_labeler.compute_costs_at_snapshot`, which replayed the *one*
   pre-sampled future belonging to the shift seed once per rule. Rollout
   variance was identically zero. Labels are now Monte Carlo means over `M`
   independent continuations drawn by `WarehouseEnv.branch`, and the per-cell
   standard error is persisted alongside the mean, which is what the reviewer
   asked to see reported.

2. THE POOL IS VARIABLE (Reviewer 1, 4.d). The submitted driver hard-wired the
   four-rule `HEURISTIC_NAMES`. The pool is now whatever Stage 1 retained, read
   through `resolve_pool(cfg)`, and it is written into `label_meta.json` so the
   ranker cannot silently train against a different class order than the one the
   labels were built with.

3. THE CORPORA ARE THREE-WAY (Reviewer 1, 4.c). `seed.shift_corpora` splits one
   SeedSequence into disjoint train / calibration / test blocks. Stage 1 fits
   rule hyperparameters on the calibration block, so nothing here touches it.

4. SIMULATION COST IS MEASURED, NOT ESTIMATED (Reviewer 3, 1). The realised
   interval-step count is recorded next to the a priori budget.

SMOKE TESTING. `--n-train` / `--n-test` take a *prefix* of the corresponding
corpus block rather than editing `cfg.shifts`. Block boundaries are contiguous
slices of one spawn, so changing `cfg.shifts.n_train` re-draws every downstream
block and a smoke run would silently label different shifts than the full run.
Slicing keeps the smoke corpus a subset of the real one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from labeling.ambiguity_filter import filter_ambiguous, resolve_theta
from labeling.provenance import write_label_meta
from labeling.rollout_labeler import label_one_shift_counted, rollout_step_budget
from labeling.soft_label_converter import (
    costs_to_probs,
    entropy_band,
    fefo_mask,
    row_entropy,
)
from seed import shift_corpora
from simulation.heuristics import resolve_pool, with_default_scales

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def _flatten(shift_rows: list[list[dict]]) -> pd.DataFrame:
    return pd.DataFrame([r for shift in shift_rows for r in shift])


def _label_block(
    name: str,
    seeds: list[int],
    cfg: DictConfig,
    pool: list[str],
    tau: int,
    n_samples: int,
    shift_id_offset: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, float, int]:
    """Label one corpus block. Returns `(rows, wall_seconds, interval_steps)`.

    Steps are summed from the per-shift counts the workers return, not read from
    the parent's counter: that counter is process-local and would report zero
    under joblib's process backend. See `label_one_shift_counted`.
    """
    print(f"\n[stage2] labelling {len(seeds)} {name.upper()} shifts "
          f"(tau={tau}, M={n_samples}, |H|={len(pool)})...")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(label_one_shift_counted)(
            shift_id_offset + i, s, cfg, tau=tau, n_samples=n_samples, candidates=pool
        )
        for i, s in enumerate(seeds)
    )
    wall = time.perf_counter() - t0
    steps = int(sum(s for _, s in out))
    print(f"  {name}: {wall:.1f}s  ({wall / max(len(seeds), 1):.2f}s/shift)  "
          f"{steps:,} interval-steps")
    return _flatten([rows for rows, _ in out]), wall, steps


def _attach_probs(df: pd.DataFrame, probs: np.ndarray, pool: list[str]) -> pd.DataFrame:
    for i, h in enumerate(pool):
        df[f"p_{h}"] = probs[:, i]
    return df


def _se_summary(df: pd.DataFrame, pool: list[str]) -> dict[str, float]:
    """Rollout-precision diagnostics — the variance term Reviewer 2 (3) asks for."""
    se = df[[f"se_{h}" for h in pool]].to_numpy(np.float64)
    sep = df["label_separation"].to_numpy(np.float64)
    finite = sep[np.isfinite(sep)]
    return {
        "mean_standard_error": float(se.mean()),
        "median_standard_error": float(np.median(se)),
        "max_standard_error": float(se.max()),
        # Fraction of epochs where the best/second-best gap is under one pooled
        # standard error — the states where a single continuation could not have
        # identified the right rule, and the reason M > 1 is not optional.
        "frac_separation_below_1se": float((finite < 1.0).mean()) if finite.size else 0.0,
        "frac_separation_below_2se": float((finite < 2.0).mean()) if finite.size else 0.0,
        "median_separation": float(np.median(finite)) if finite.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=None,
                        help="Label only the first N train shifts (smoke test).")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Label only the first N test shifts (smoke test).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="joblib n_jobs (default -1 = all cores).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override cfg.run_id; metadata goes to runs/<run_id>/.")
    parser.add_argument("--tau", type=int, default=None,
                        help="Override cfg.labeling.tau (rollout horizon).")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override cfg.labeling.n_rollout_samples (M).")
    parser.add_argument("--theta", type=float, default=None,
                        help="Override the test ambiguity threshold "
                             "(absolute top-1 probability; normally derived "
                             "from theta_confidence_uniform_multiple). "
                             "Used by the E4 theta sweep.")
    parser.add_argument("--train-out", type=Path, default=None)
    parser.add_argument("--test-out", type=Path, default=None)
    parser.add_argument("--allow-provisional-scales", action="store_true",
                        help="Substitute the calibration-grid midpoint for any "
                             "unfitted ATC/COVERT scale. SMOKE TESTING ONLY — "
                             "labels produced this way are not deployable and "
                             "are stamped as provisional.")
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    if args.allow_provisional_scales:
        cfg = with_default_scales(cfg)
    pool = resolve_pool(cfg)
    tau = int(args.tau) if args.tau is not None else int(cfg.labeling.tau)
    n_samples = (
        int(args.n_samples) if args.n_samples is not None
        else int(cfg.labeling.n_rollout_samples)
    )

    corpora = shift_corpora(cfg)
    train_seeds = corpora["train"]
    test_seeds = corpora["test"]
    if args.n_train is not None:
        train_seeds = train_seeds[: int(args.n_train)]
    if args.n_test is not None:
        test_seeds = test_seeds[: int(args.n_test)]

    run_id = args.run_id or str(cfg.run_id)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # An uncalibrated ATC/COVERT must fail here, loudly, rather than label the
    # whole corpus against an arbitrary look-ahead scale. Stage 1 writes these.
    unset = [
        key for rule, key in (("ATC", "atc_lookahead_k"), ("COVERT", "covert_lookahead_k"))
        if rule in pool and cfg.heuristics.get(key) is None
    ]
    if unset:
        raise SystemExit(
            f"Rule scale parameter(s) {unset} are unset but the pool contains the "
            f"rule(s) that need them. Run `python -m experiments.calibrate_rules "
            f"calibrate` and write the fitted values into config.yaml first "
            f"(Reviewer 1, 4.c). Labelling against an uncalibrated k would "
            f"reproduce the submitted result that WSPT beats ATC, which is an "
            f"artefact of the scale, not a property of the rules."
        )

    n_intervals = int(round(cfg.sim.shift_hours * 60 / cfg.sim.interval_minutes))
    budget = rollout_step_budget(
        len(train_seeds) + len(test_seeds), n_intervals, len(pool), tau, n_samples
    )
    print(f"[stage2] pool = {pool}")
    print(f"[stage2] tau={tau}  M={n_samples}  "
          f"observed_policy={cfg.labeling.observed_policy}")
    print(f"[stage2] corpus: {len(train_seeds)} train + {len(test_seeds)} test shifts")
    print(f"[stage2] a priori budget: {budget:,} interval-steps")
    print(f"[stage2] beta_grid={list(cfg.labeling.beta_grid)}  "
          f"target_median_entropy={[round(x, 4) for x in entropy_band(cfg.labeling, len(pool))]}"
          f" nats (|H|={len(pool)})")
    print(f"[stage2] n_jobs={args.n_jobs}  cpu_count={os.cpu_count()}")

    train_df, t_train, steps_train = _label_block(
        "train", train_seeds, cfg, pool, tau, n_samples, 0, args.n_jobs
    )
    test_df, t_test, steps_test = _label_block(
        "test", test_seeds, cfg, pool, tau, n_samples, len(train_seeds), args.n_jobs
    )
    steps_total = steps_train + steps_test
    print(f"\n[stage2] aggregated: train={len(train_df)} rows, test={len(test_df)} rows")

    cost_cols = [f"cost_{h}" for h in pool]
    train_costs = train_df[cost_cols].to_numpy(dtype=np.float64)
    test_costs = test_df[cost_cols].to_numpy(dtype=np.float64)

    # One temperature, fitted on train and reused on test — otherwise the two
    # label sets are not on the same scale and the test cross-entropy is
    # measuring the temperature rather than the ranker.
    train_probs, beta = costs_to_probs(train_costs, cfg.labeling)
    test_probs, _ = costs_to_probs(test_costs, cfg.labeling, beta=beta)

    fefo_threshold = float(cfg.heuristics.fefo_mask_threshold)
    train_probs = fefo_mask(
        train_probs, train_df["f_pct_perishable"].to_numpy(np.float64),
        threshold=fefo_threshold, heuristic_names=pool,
    )
    test_probs = fefo_mask(
        test_probs, test_df["f_pct_perishable"].to_numpy(np.float64),
        threshold=fefo_threshold, heuristic_names=pool,
    )

    train_df = _attach_probs(train_df, train_probs, pool)
    test_df = _attach_probs(test_df, test_probs, pool)

    median_entropy = float(np.median(row_entropy(train_probs)))
    target_lo, target_hi = entropy_band(cfg.labeling, len(pool))
    in_band = bool(target_lo <= median_entropy <= target_hi)
    print(f"\n[stage2] beta = {beta:.6f}")
    print(f"  median train row entropy = {median_entropy:.4f} "
          f"(target [{target_lo}, {target_hi}]) -> "
          f"{'OK' if in_band else 'OUT OF BAND'}")

    theta = (
        float(args.theta) if args.theta is not None
        else resolve_theta(cfg.labeling.ambiguity_filter, len(pool))
    )
    keep_mask = filter_ambiguous(test_probs, theta=theta)
    n_kept = int(keep_mask.sum())
    print(f"  test ambiguity filter (theta={theta}): kept {n_kept}/{len(test_df)} "
          f"(dropped {len(test_df) - n_kept})")

    train_se = _se_summary(train_df, pool)
    print(f"  rollout SE: mean={train_se['mean_standard_error']:.4f}  "
          f"median={train_se['median_standard_error']:.4f}")
    print(f"  epochs with best/second gap < 1 SE: "
          f"{train_se['frac_separation_below_1se']:.1%}")

    test_unfiltered = test_df.copy()
    test_df = test_df[keep_mask].reset_index(drop=True)

    train_path = Path(args.train_out) if args.train_out else DATA_DIR / "train.parquet"
    test_path = Path(args.test_out) if args.test_out else DATA_DIR / "test.parquet"
    train_path = train_path if train_path.is_absolute() else (REPO_ROOT / train_path).resolve()
    test_path = test_path if test_path.is_absolute() else (REPO_ROOT / test_path).resolve()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # The PRE-FILTER test set, kept so the `random_ambiguity_filter` control can
    # drop a matched NUMBER of rows at random instead of by confidence. Without
    # it that ablation cannot be run honestly — the submitted implementation
    # substituted a hardcoded drop fraction measured in an earlier campaign.
    unfiltered_path = test_path.with_name(
        test_path.name.replace(".parquet", "_unfiltered.parquet")
    )
    test_unfiltered.to_parquet(unfiltered_path, index=False)

    # The pool and its ORDER are part of the dataset contract: `p_<rule>` column
    # order defines the ranker's class indices. Downstream stages read this
    # rather than re-deriving the pool, so a config edit between stages cannot
    # silently permute the classes.
    meta = {
        "pool": pool,
        "prob_columns": [f"p_{h}" for h in pool],
        "tau": tau,
        "n_rollout_samples": n_samples,
        "observed_policy": str(cfg.labeling.observed_policy),
        "beta": float(beta),
        "median_train_entropy": median_entropy,
        "target_median_entropy": [target_lo, target_hi],
        "entropy_in_band": in_band,
        "theta_confidence": theta,
        "n_train_shifts": len(train_seeds),
        "n_test_shifts": len(test_seeds),
        "n_train_rows": int(len(train_df)),
        "n_test_rows_prefilter": int(len(keep_mask)),
        "n_test_rows": int(len(test_df)),
        "fefo_mask_threshold": fefo_threshold,
        "atc_lookahead_k": cfg.heuristics.get("atc_lookahead_k"),
        "covert_lookahead_k": cfg.heuristics.get("covert_lookahead_k"),
        # Marks a smoke-test corpus so a provisional run can never be mistaken
        # for a deployable one downstream.
        "provisional_scales": bool(args.allow_provisional_scales),
        "rollout_precision_train": train_se,
        # Reviewer 3 (1): the offline simulation cost, measured.
        "simulated_interval_steps": int(steps_total),
        "simulated_interval_steps_train": int(steps_train),
        "simulated_interval_steps_test": int(steps_test),
        "budget_interval_steps": int(budget),
        "wall_clock_s": {"train": t_train, "test": t_test},
        "objective_weights": OmegaConf.to_container(cfg.objective, resolve=True),
    }
    meta_path = run_dir / "label_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    # A second copy NEXT TO THE PARQUETS, which is what makes it a provenance
    # stamp rather than a run log — see `labeling/provenance.py` for why a label
    # parquet cannot be trusted without one.
    sidecar = write_label_meta(train_path, meta)
    if test_path.parent != train_path.parent:
        write_label_meta(test_path, meta)

    print(f"\n[stage2] saved:")
    for p, n in ((train_path, len(train_df)), (test_path, len(test_df)),
                 (unfiltered_path, len(test_unfiltered))):
        try:
            print(f"  {p.relative_to(REPO_ROOT)}  ({n} rows)")
        except ValueError:
            print(f"  {p}  ({n} rows)")
    print(f"  {meta_path.relative_to(REPO_ROOT)}")
    print(f"  {sidecar.relative_to(REPO_ROOT)}")
    print(f"[stage2] simulated interval-steps: {steps_total:,} "
          f"(a priori budget {budget:,})")
    print(f"[stage2] total wall-clock: {t_train + t_test:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
