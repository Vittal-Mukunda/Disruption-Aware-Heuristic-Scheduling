"""Stage 3 — fit the regime layer, the ranker, and the calibrator.

Inputs:
  data/train.parquet  one row per decision epoch: features + per-rule costs,
                      standard errors, and soft labels
  data/test.parquet   the same, ambiguity-filtered at cfg theta
  config.yaml         hyperparameters (NOT the class set — see below)

The RULE POOL is read from the label columns, never from config. Stage 1 may
screen rules out after config.yaml was last edited, and the submitted code
declared the class count in a third place (`cfg.ranker.num_class`) that could
disagree with both. See `models.heuristic_ranker.prob_columns`.

Outputs (under `runs/<run_id>/`):
  regime.joblib            -- fitted GaussianMixture (k_star, full covariance)
  model.json               -- XGBoost ranker (multi:softprob, K = |pool|)
  calibrator.joblib        -- CalibratedRanker wrapper (isotonic, prefit)
  phase4_regime.json       -- BIC sweep, K_star, mean ARI, stability flag
  phase4_metrics.json      -- ranker CV scores + calibration report (test set)
  phase4_cv_calibration.json -- 5-fold-CV mean ECE pre/post (acceptance gate)

Run modes:
  python -m experiments.train_ranker                  # full HP grid (~hours)
  python -m experiments.train_ranker --smoke          # 1 HP combo (~minutes)
  python -m experiments.train_ranker --skip-cv-cal    # skip the 5-fold acceptance gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from models.calibration import (
    CalibratedRanker,
    cv_calibration_report,
    evaluate_calibration,
    split_calibration_shifts,
)
from models.heuristic_ranker import (
    FEATURE_COLUMNS,
    RankerCVResult,
    build_ldl_training_arrays,
    cross_validate_ranker,
    pool_from_frame,
    prob_columns,
    soft_xent,
)
from regime.regime_discovery import (
    attach_regime_posteriors,
    discover_regimes,
    regime_posterior_columns,
)
from seed import SEED_MODEL

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

SMOKE_HYPERPARAMS = {
    "max_depth": [6],
    "n_estimators": [200],
    "learning_rate": [0.1],
}


def _maybe_override_smoke(cfg: DictConfig, smoke: bool) -> DictConfig:
    if not smoke:
        return cfg
    with open_dict(cfg):
        cfg.ranker.hyperparams = OmegaConf.create(SMOKE_HYPERPARAMS)
    return cfg


def _single_combo_cfg(cfg_ranker: DictConfig, params: dict) -> DictConfig:
    """A `cfg.ranker` clone whose HP grid is the single combination `params`.

    `num_class` is deliberately absent: the class count is the width of the
    label matrix, read by `cross_validate_ranker` from the frame itself.
    """
    return OmegaConf.create({
        "objective": str(cfg_ranker.objective),
        "sample_weight": str(cfg_ranker.sample_weight),
        "cv": {
            "n_splits": int(cfg_ranker.cv.n_splits),
            "group_col": str(cfg_ranker.cv.group_col),
        },
        "hyperparams": {k: [v] for k, v in params.items()},
        "calibration": OmegaConf.to_container(cfg_ranker.calibration, resolve=True),
        "switching": OmegaConf.to_container(cfg_ranker.switching, resolve=True),
    })


def _fit_ranker_fixed_params_factory(best_params: dict, base_cols=None):
    """Return a fit-fn for `cv_calibration_report` that skips the inner HP search."""
    def fit_fn(df_tr, cfg_ranker, seed, extra_cols):
        # cross_validate_ranker will GroupKFold internally and refit the best
        # (here: only) combination. `base_cols` must be threaded through, or the
        # calibration report would be computed on the full feature set while the
        # deployed model was fitted on a subset.
        return cross_validate_ranker(
            df_tr, _single_combo_cfg(cfg_ranker, best_params), seed,
            extra_feature_cols=extra_cols, base_feature_cols=base_cols,
        )
    return fit_fn


def _argmax_distribution(probs: np.ndarray, pool: list[str]) -> dict[str, int]:
    """Per-class count of argmax labels, keyed by rule name."""
    if probs.shape[0] == 0:
        return {h: 0 for h in pool}
    counts = np.bincount(probs.argmax(axis=1), minlength=len(pool))
    return {h: int(counts[i]) for i, h in enumerate(pool)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Use a 1-combo HP grid for a fast end-to-end check.")
    parser.add_argument("--skip-cv-cal", action="store_true",
                        help="Skip the 5-fold-CV calibration report.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override cfg.run_id; outputs go to runs/<run_id>/.")
    parser.add_argument("--seed", type=int, default=SEED_MODEL,
                        help="Base random state (default SEED_MODEL=1337).")
    parser.add_argument("--train-path", type=Path, default=None,
                        help="Override the train parquet path (default: data/train.parquet).")
    parser.add_argument("--test-path", type=Path, default=None,
                        help="Override the test parquet path (default: data/test.parquet).")
    parser.add_argument("--no-regime", action="store_true",
                        help="Skip the regime layer entirely (E3 `no_regime` "
                             "ablation). Dropping the posterior COLUMNS from the "
                             "input parquet does not ablate anything, because "
                             "this driver re-fits the GMM and re-attaches them.")
    parser.add_argument("--feature-cols", nargs="*", default=None,
                        help="Restrict the base state to these f_* columns "
                             "(E3 `top5_features`, Reviewer 3 comment 4).")
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = _maybe_override_smoke(cfg, args.smoke)
    run_id = args.run_id or f"phase4{'_smoke' if args.smoke else ''}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 4] run_id={run_id}  smoke={args.smoke}  seed={args.seed}")

    train_path = Path(args.train_path) if args.train_path else DATA_DIR / "train.parquet"
    test_path = Path(args.test_path) if args.test_path else DATA_DIR / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Missing {train_path} or {test_path}; run "
            f"`python -m experiments.generate_labels` first."
        )
    print(f"[Phase 4] data: {train_path} / {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    print(f"[Phase 4] train: {len(df_train)} rows; test: {len(df_test)} rows")

    # PROVENANCE GATE. Pre-revision label parquets are structurally valid — the
    # `f_*` and `p_*` columns are all present and well formed — so every schema
    # check below passes on them and the run would quietly fit a model to labels
    # generated under the deleted objective and the four-rule pool. The only
    # thing that distinguishes them is the stamp Stage 2 writes beside the file.
    meta_path = train_path.parent / "label_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"No {meta_path} beside {train_path.name}.\n"
            f"These labels were not produced by the current Stage 2, so they "
            f"predate the corrected objective, the two-clock order model and the "
            f"screened pool. Training on them would succeed and produce a wrong "
            f"model.\n"
            f"Run `python -m experiments.generate_labels` first (and delete the "
            f"stale data/*.parquet)."
        )
    label_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"[Phase 4] labels: tau={label_meta.get('tau')} "
          f"M={label_meta.get('n_rollout_samples')} "
          f"beta={label_meta.get('beta'):.4f} "
          f"policy={label_meta.get('observed_policy')}")

    # The pool is a property of the labels, not of the config: Stage 1 may have
    # screened rules out since config.yaml was last edited. Train and test must
    # agree, or the class indices mean different things on the two splits.
    pool = pool_from_frame(df_train)
    if list(label_meta.get("pool", pool)) != pool:
        raise SystemExit(
            f"label_meta.json records pool {label_meta['pool']} but "
            f"{train_path.name} carries {pool}. The parquet and its stamp are "
            f"from different runs; regenerate both."
        )
    test_pool = pool_from_frame(df_test)
    if pool != test_pool:
        raise SystemExit(
            f"Label pools differ between splits: train={pool}, test={test_pool}. "
            f"Regenerate both with a single `generate_labels` run."
        )
    print(f"[Phase 4] pool (from labels, K={len(pool)}) = {pool}")

    base_cols = list(args.feature_cols) if args.feature_cols else None
    if base_cols:
        missing = [c for c in base_cols if c not in df_train.columns]
        if missing:
            raise SystemExit(f"--feature-cols not in the data: {missing}")
        print(f"[Phase 4] restricted state ({len(base_cols)} features): {base_cols}")

    # --- 1) Regime discovery -------------------------------------------------
    regime_result = None
    extra_cols: list[str] = []
    if args.no_regime:
        # The ablation has to happen HERE. Stripping `regime_post_*` from the
        # input parquet achieves nothing, because this driver would re-fit the
        # GMM and re-attach them — which is why the submitted `no_regime`
        # ablation, had it been run, would have reproduced the full model.
        print("[Phase 4] --no-regime: skipping the regime layer entirely.")
    else:
        t0 = time.perf_counter()
        print("[Phase 4] fitting regime layer (GMM, K-sweep + ARI stability)...")
        regime_result = discover_regimes(df_train, cfg.regime, seed=args.seed)
        t_regime = time.perf_counter() - t0
        print(f"  K* = {regime_result.k_star}  BIC = {regime_result.bic:.1f}  "
              f"mean ARI = {regime_result.mean_ari:.4f}  "
              f"stable={regime_result.stable}  ({t_regime:.1f}s)")
        print(f"  BIC sweep: {regime_result.bic_per_k}")

        df_train = attach_regime_posteriors(df_train, regime_result.gmm)
        df_test = attach_regime_posteriors(df_test, regime_result.gmm)
        extra_cols = regime_posterior_columns(regime_result.k_star)
        print(f"  attached regime posteriors: {extra_cols}")

        joblib.dump(regime_result.gmm, run_dir / "regime.joblib")

    if regime_result is not None:
        (run_dir / "phase4_regime.json").write_text(
            json.dumps({
                "k_star": int(regime_result.k_star),
                "bic": float(regime_result.bic),
                "mean_ari": float(regime_result.mean_ari),
                "stable": bool(regime_result.stable),
                "ari_threshold": float(cfg.regime.ari_threshold),
                "bic_per_k": {
                    int(k): float(v) for k, v in regime_result.bic_per_k.items()
                },
                "extra_feature_cols": list(extra_cols),
            }, indent=2),
            encoding="utf-8",
        )

    # --- 2) Ranker HP search on training data --------------------------------
    t0 = time.perf_counter()
    print("[Phase 4] HP search on training data (cross_validate_ranker)...")
    cv_result: RankerCVResult = cross_validate_ranker(
        df_train, cfg.ranker, seed=args.seed, extra_feature_cols=extra_cols,
        base_feature_cols=base_cols,
    )
    t_hp = time.perf_counter() - t0
    print(f"  best params = {cv_result.best_params}")
    print(f"  mean CV soft-xent = {cv_result.best_mean_xent:.5f}  "
          f"per-fold = {[f'{x:.4f}' for x in cv_result.per_fold_xent]}")
    print(f"  HP-search wall: {t_hp:.1f}s  (combos: {len(cv_result.grid_results)})")

    # --- 3) 5-fold CV calibration acceptance gate ----------------------------
    cv_cal_summary: dict | None = None
    if args.skip_cv_cal:
        print("[Phase 4] skip-cv-cal flag set; not running 5-fold calibration report.")
    else:
        t0 = time.perf_counter()
        print("[Phase 4] 5-fold-CV calibration report (acceptance: ECE_post < 0.05)...")
        fit_fn = _fit_ranker_fixed_params_factory(cv_result.best_params, base_cols)
        cv_cal_summary = cv_calibration_report(
            df_train, fit_fn, cfg.ranker, seed=args.seed,
            extra_feature_cols=extra_cols,
        )
        t_cv_cal = time.perf_counter() - t0
        print(f"  mean ECE pre = {cv_cal_summary['mean_ece_pre']:.4f}  "
              f"post = {cv_cal_summary['mean_ece_post']:.4f}")
        print(f"  mean Brier pre = {cv_cal_summary['mean_brier_pre']:.4f}  "
              f"post = {cv_cal_summary['mean_brier_post']:.4f}")
        print(f"  mean soft-xent post = {cv_cal_summary['mean_soft_xent_post']:.5f}")
        print(f"  CV-calibration wall: {t_cv_cal:.1f}s")
        (run_dir / "phase4_cv_calibration.json").write_text(
            json.dumps(cv_cal_summary, indent=2), encoding="utf-8"
        )
        if cv_cal_summary["mean_ece_post"] >= 0.05:
            print(f"  WARNING: mean post-cal ECE {cv_cal_summary['mean_ece_post']:.4f}"
                  f" >= 0.05 acceptance threshold.")

    # --- 4) Fit final calibrator on a 20% held-out shift split ---------------
    cal_frac = float(cfg.ranker.calibration.calibration_split)
    inner_tr, inner_cal = split_calibration_shifts(
        df_train, cal_frac, seed=args.seed,
        group_col=str(cfg.ranker.cv.group_col),
    )
    df_inner_tr = df_train.iloc[inner_tr].reset_index(drop=True)
    df_inner_cal = df_train.iloc[inner_cal].reset_index(drop=True)
    n_tr_shifts = df_inner_tr["shift_id"].nunique()
    n_cal_shifts = df_inner_cal["shift_id"].nunique()
    print(f"[Phase 4] final calibration split: "
          f"{len(df_inner_tr)} train rows ({n_tr_shifts} shifts) / "
          f"{len(df_inner_cal)} calibration rows ({n_cal_shifts} shifts)")

    t0 = time.perf_counter()
    print("[Phase 4] refit ranker on inner-train with best HPs...")
    refit_cfg = _single_combo_cfg(cfg.ranker, cv_result.best_params)
    final_cv = cross_validate_ranker(
        df_inner_tr, refit_cfg, seed=args.seed, extra_feature_cols=extra_cols,
        base_feature_cols=base_cols,
    )
    t_refit = time.perf_counter() - t0
    print(f"  refit wall: {t_refit:.1f}s")

    t0 = time.perf_counter()
    print("[Phase 4] fitting isotonic calibrator on the 20% shift split...")
    calibrator = CalibratedRanker(final_cv.model, final_cv.feature_cols).fit(df_inner_cal)
    t_cal = time.perf_counter() - t0
    print(f"  calibration wall: {t_cal:.1f}s")

    # --- 5) Evaluate calibration on the test parquet -------------------------
    print("[Phase 4] evaluating calibration on test.parquet...")
    test_report = evaluate_calibration(final_cv, df_test, calibrator)
    print(f"  ECE  pre={test_report.ece_pre:.4f}   post={test_report.ece_post:.4f}")
    print(f"  Brier pre={test_report.brier_pre:.4f}  post={test_report.brier_post:.4f}")
    print(f"  soft-xent  pre={test_report.soft_xent_pre:.5f}  "
          f"post={test_report.soft_xent_post:.5f}")
    print(f"  test rows: {test_report.n_rows}")

    # --- 6) Persist artifacts ------------------------------------------------
    model_path = run_dir / "model.json"
    try:
        final_cv.model.save_model(str(model_path))
    except TypeError:
        final_cv.model.get_booster().save_model(str(model_path))
    joblib.dump(calibrator, run_dir / "calibrator.joblib")
    joblib.dump(
        {"feature_cols": final_cv.feature_cols, "best_params": cv_result.best_params,
         "regime_extra_cols": list(extra_cols),
         "k_star": int(regime_result.k_star) if regime_result is not None else 0,
         "no_regime": bool(args.no_regime),
         # Class-index -> rule name. The deployment wrapper maps argmax through
         # this; without it a re-screened pool would silently reindex the actions.
         "classes": list(final_cv.classes)},
        run_dir / "ranker_meta.joblib",
    )

    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "pool": list(pool),
        "n_classes": len(pool),
        "n_train_rows": int(len(df_train)),
        "n_test_rows": int(len(df_test)),
        "n_train_shifts_inner": int(n_tr_shifts),
        "n_cal_shifts_inner": int(n_cal_shifts),
        "regime": (
            None if regime_result is None else {
                "k_star": int(regime_result.k_star),
                "mean_ari": float(regime_result.mean_ari),
                "stable": bool(regime_result.stable),
            }
        ),
        "ranker": {
            "best_params": cv_result.best_params,
            "best_mean_cv_soft_xent": float(cv_result.best_mean_xent),
            "per_fold_cv_soft_xent": [float(x) for x in cv_result.per_fold_xent],
            "grid_size": int(len(cv_result.grid_results)),
            "feature_cols": list(final_cv.feature_cols),
        },
        "test_calibration": asdict(test_report),
        "cv_calibration": cv_cal_summary,
        "argmax_distribution_test_pre":
            _argmax_distribution(final_cv.model.predict_proba(
                df_test[final_cv.feature_cols].to_numpy(np.float64)), pool),
        "argmax_distribution_test_post":
            _argmax_distribution(calibrator.predict_proba(
                df_test[final_cv.feature_cols].to_numpy(np.float64)), pool),
        "argmax_distribution_test_truth":
            _argmax_distribution(
                df_test[prob_columns(df_test)].to_numpy(np.float64), pool),
    }
    (run_dir / "phase4_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nSaved artifacts under {run_dir.relative_to(REPO_ROOT)}/")
    for p in sorted(run_dir.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
