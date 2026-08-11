"""E3 — ablation studies. One entry per `cfg.experiments.e3_ablations`.

Three tiers, by what has to be recomputed:

INFERENCE-ONLY — swap the deployment stack, reuse the trained model.
  - no_calibration:           raw XGBoost softprob, no isotonic wrapper.
  - no_switching_controller:  calibrated probs + FEFO mask, no dwell, no gate.

RETRAIN — `python -m experiments.e3_ablations retrain <name>`.
  - no_regime:                fit without the regime layer at all.
  - hard_labels:              one-hot argmax(p) instead of the soft KL target.
  - random_ambiguity_filter:  drop the same NUMBER of test rows at random as the
                              theta filter drops by confidence.
  - top5_features:            refit on the five most important state features
                              (Reviewer 3, comment 4 — parsimony).

RELABEL — `python -m experiments.e3_ablations relabel <name>` prints the recipe;
these change the ESTIMATOR, so no transform of an existing parquet produces them.
  - tau1_snapshot_only:       tau=1 labels (this is `snapshot_xgb`).
  - single_sample_rollout:    M=1, the submitted labelling scheme.

TWO OF THESE WERE PREVIOUSLY UNRUNNABLE, not merely unrun. `no_regime` was
implemented by deleting `regime_post_*` columns from the input parquet, but
`experiments/train_ranker.py` re-fits the GMM and re-attaches them, so the
"ablated" model was the full model; it is now a driver flag. And
`random_ambiguity_filter` compared against a hardcoded drop fraction from an
earlier campaign because the pre-filter test set was not on disk; the labeller
now writes it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from experiments.evaluate import (
    canonical_test_seeds,
    evaluate_policy,
)
from labeling.provenance import stamp_derived
from models.heuristic_ranker import prob_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4"
RESULTS_DIR = REPO_ROOT / "results" / "E3"

INFERENCE_ABLATIONS = ["no_calibration", "no_switching_controller"]
RETRAIN_ABLATIONS = [
    "no_regime", "hard_labels", "random_ambiguity_filter", "top5_features",
]
#: Ablations that require a fresh LABELLING pass, not just a retrain. Both change
#: the rollout estimator itself, so `experiments/generate_labels.py` has to run
#: first; `cmd_relabel` prints the exact two-command recipe.
RELABEL_ABLATIONS = {
    "tau1_snapshot_only": ["--tau", "1"],
    "single_sample_rollout": ["--n-samples", "1"],
}


# ---------------------------------------------------------------------------
# Inference-only ablations
# ---------------------------------------------------------------------------


class _NoCalibrationRanker:
    """Wraps the raw XGB model to expose `predict_proba` without isotonic."""

    def __init__(self, base_model, feature_cols: list[str]) -> None:
        self.base_model = base_model
        self.feature_cols = list(feature_cols)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)


def load_no_calibration(run_dir: Path = DEFAULT_RUN_DIR, cfg: DictConfig | None = None):
    """OUR controller with the calibrator stripped — raw softprob, then mask + switching."""
    import joblib

    from baselines.ours import OursPolicy
    from models.switching_controller import SwitchingController

    cfg = cfg if cfg is not None else OmegaConf.load(CONFIG_PATH)
    calibrator = joblib.load(run_dir / "calibrator.joblib")
    regime_gmm = joblib.load(run_dir / "regime.joblib")
    meta = joblib.load(run_dir / "ranker_meta.joblib")
    feature_cols = list(meta["feature_cols"])
    # Class order as TRAINED, not as currently configured — Stage 1 rewrites
    # `cfg.heuristics.pool`, and an ablation that remapped the class indices
    # would be measuring the remapping rather than the missing component.
    classes = list(meta.get("classes") or [])
    if not classes:
        raise RuntimeError(
            f"{run_dir / 'ranker_meta.joblib'} has no 'classes' entry; retrain "
            f"with the current experiments/train_ranker.py."
        )

    raw = _NoCalibrationRanker(calibrator.base_model, feature_cols)
    controller = SwitchingController(
        ranker=raw,
        cfg_switching=cfg.ranker.switching,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        feature_cols=feature_cols,
        heuristic_names=classes,
    )
    return OursPolicy(controller=controller, regime_gmm=regime_gmm,
                      feature_cols=feature_cols)


def load_no_switching(run_dir: Path = DEFAULT_RUN_DIR, cfg: DictConfig | None = None):
    """OUR controller with T_min=0 and entropy_gate_ratio=∞ — always argmax."""
    cfg = cfg if cfg is not None else OmegaConf.load(CONFIG_PATH)
    new_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    # T_min=0 disables dwell; setting entropy_gate_ratio above log(K) means the
    # gate never fires anyway, but T_min=0 alone is sufficient.
    new_cfg.ranker.switching.t_min_intervals = 0
    new_cfg.ranker.switching.entropy_gate_ratio = 999.0

    from baselines.ours import load_ours
    return load_ours(run_dir, cfg=new_cfg)


def cmd_inference(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    scenario = args.scenario
    if scenario != "default":
        from experiments.e2_main import apply_scenario
        cfg = apply_scenario(cfg, scenario)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    results_dir = (
        RESULTS_DIR if scenario == "default"
        else RESULTS_DIR / f"scenario_{scenario}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    ablations = args.ablations if args.ablations else INFERENCE_ABLATIONS

    loaders = {
        "no_calibration": load_no_calibration,
        "no_switching_controller": load_no_switching,
    }
    for ab in ablations:
        if ab not in loaders:
            raise SystemExit(f"unknown inference ablation: {ab}. "
                             f"Options: {list(loaders.keys())}")
        print(f"\n[E3 inference] running {ab} on {len(seeds)} test shifts "
              f"(scenario={scenario})")
        policy = loaders[ab](args.run_dir or DEFAULT_RUN_DIR, cfg=cfg)
        df = evaluate_policy(
            ab, policy, seeds, cfg,
            results_dir=results_dir, save=True, verbose=args.verbose,
        )
        summary = df[["service_failure_rate", "mean_tardiness",
                      "composite_cost", "throughput"]].mean()
        print(f"  sla_breach={summary['service_failure_rate']:.4f}  "
              f"mean_tard={summary['mean_tardiness']:.4f}  "
              f"composite_cost={summary['composite_cost']:.4f}  "
              f"throughput={summary['throughput']:.2f}")
    return 0


# ---------------------------------------------------------------------------
# Retrain ablations — wire only; the user kicks these off explicitly.
# ---------------------------------------------------------------------------


def _copy_through(
    train_path: Path, test_path: Path, out_dir: Path
) -> tuple[Path, Path]:
    """No data transform: the ablation is a train_ranker FLAG, not a data edit.

    `no_regime` and `top5_features` were previously attempted by editing the
    parquet — dropping `regime_post_*` columns in the first case. That achieves
    nothing: `experiments/train_ranker.py` re-fits the GMM and re-attaches the
    posteriors, so the "ablated" model was the full model. Both are now real
    flags on the driver (`--no-regime`, `--feature-cols`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return train_path, test_path


def _top_k_feature_cols(run_dir: Path, k: int = 5) -> list[str]:
    """The k most important state features of the deployed model (Reviewer 3, 4).

    Prefers the SHAP table if E5 has been run, since that is what the paper
    reports; falls back to the booster's own gain importance. Regime posteriors
    are excluded — this ablation is about the hand-crafted state.
    """
    shap_path = REPO_ROOT / "results" / "E5" / "shap_global_importance.parquet"
    if shap_path.exists():
        imp = pd.read_parquet(shap_path).sort_values(
            "mean_abs_shap", ascending=False
        )
        ranked = [str(f) for f in imp["feature"]]
    else:
        import joblib as _joblib
        import xgboost as _xgb

        meta = _joblib.load(run_dir / "ranker_meta.joblib")
        booster = _xgb.XGBClassifier()
        booster.load_model(str(run_dir / "model.json"))
        cols = list(meta["feature_cols"])
        order = np.argsort(-np.asarray(booster.feature_importances_))
        ranked = [cols[i] for i in order]
        print(f"[E3 top5] {shap_path} not found; ranking by XGBoost gain instead.")

    base = [c for c in ranked if c.startswith("f_")][:k]
    if len(base) < k:
        raise SystemExit(
            f"only {len(base)} state features available to rank; expected >= {k}"
        )
    return base


def _prepare_hard_labels_data(
    train_path: Path, test_path: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Replace soft `p_*` columns with one-hot of `argmax(p_*)`. Same row count.

    Columns come from the FRAME, not from `HEURISTIC_NAMES`: the deployed pool is
    whatever the Stage-1 screen retained, so a hardcoded eight-rule list raises a
    KeyError the moment a rule is screened out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [(train_path, out_dir / "train.parquet"),
                     (test_path, out_dir / "test.parquet")]:
        df = pd.read_parquet(src)
        prob_cols = prob_columns(df)
        P = df[prob_cols].to_numpy(dtype=np.float64)
        argmax = P.argmax(axis=1)
        one_hot = np.zeros_like(P)
        one_hot[np.arange(len(argmax)), argmax] = 1.0
        for i, c in enumerate(prob_cols):
            df[c] = one_hot[:, i]
        df.to_parquet(dst, index=False)
    stamp_derived(train_path, out_dir / "train.parquet",
                  "hard labels: one-hot argmax(p) replacing the soft target")
    return out_dir / "train.parquet", out_dir / "test.parquet"


def _prepare_random_filter_data(
    train_path: Path, test_path: Path, out_dir: Path, seed: int
) -> tuple[Path, Path]:
    """Replace the theta-filter with a random drop of the SAME SIZE.

    The question this answers is whether the reported test numbers come from the
    confidence filter selecting genuinely unambiguous states, or merely from
    evaluating on fewer rows. The honest control keeps the row count identical
    and randomises which rows survive.

    That requires the UNFILTERED test set. The submitted implementation did not
    have it on disk and substituted a hardcoded `0.4593` drop fraction — the
    ratio 735/1600 measured in one earlier run — applied to the TRAIN split,
    which is a different quantity on a different corpus and would silently go
    stale the moment either changed. `experiments/generate_labels.py` now writes
    `test_unfiltered.parquet` beside the filtered one, so the control is exact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    unfiltered_path = test_path.with_name(
        test_path.name.replace(".parquet", "_unfiltered.parquet")
    )
    if not unfiltered_path.exists():
        raise SystemExit(
            f"missing {unfiltered_path}. The random-filter control needs the "
            f"pre-filter test set to drop a matched NUMBER of rows at random. "
            f"Re-run `python -m experiments.generate_labels`, which writes it."
        )
    df_full = pd.read_parquet(unfiltered_path)
    n_keep = len(pd.read_parquet(test_path))
    if n_keep > len(df_full):
        raise SystemExit(
            f"filtered test set ({n_keep} rows) is larger than the unfiltered "
            f"one ({len(df_full)}); the two are from different runs."
        )
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(len(df_full), size=n_keep, replace=False)
    df_random = df_full.iloc[np.sort(keep_idx)].reset_index(drop=True)
    print(f"[E3 random_filter] kept {n_keep}/{len(df_full)} test rows at random "
          f"(theta-filter kept the same count by confidence)")

    tr_out = out_dir / "train.parquet"
    te_out = out_dir / "test.parquet"
    pd.read_parquet(train_path).to_parquet(tr_out, index=False)
    df_random.to_parquet(te_out, index=False)
    stamp_derived(
        train_path, tr_out,
        f"random test filter: {n_keep}/{len(df_full)} rows kept at random "
        f"instead of by confidence",
    )
    return tr_out, te_out


def cmd_retrain(args: argparse.Namespace) -> int:
    """Spawn `experiments.train_ranker` with a transformed train/test parquet."""
    ablation = args.ablation
    if ablation not in RETRAIN_ABLATIONS:
        raise SystemExit(f"unknown retrain ablation: {ablation}. "
                         f"Options: {RETRAIN_ABLATIONS}")
    out_dir = REPO_ROOT / "data" / f"e3_{ablation}"
    runs_dir = REPO_ROOT / "runs" / f"e3_{ablation}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_path = REPO_ROOT / "data" / "train.parquet"
    test_path = REPO_ROOT / "data" / "test.parquet"

    extra_flags: list[str] = []
    if ablation == "no_regime":
        tr, te = _copy_through(train_path, test_path, out_dir)
        extra_flags = ["--no-regime"]
    elif ablation == "top5_features":
        tr, te = _copy_through(train_path, test_path, out_dir)
        top5 = _top_k_feature_cols(DEFAULT_RUN_DIR, k=5)
        print(f"[E3 top5_features] retaining {top5}")
        extra_flags = ["--feature-cols", *top5]
    elif ablation == "hard_labels":
        tr, te = _prepare_hard_labels_data(train_path, test_path, out_dir)
    else:  # random_ambiguity_filter
        tr, te = _prepare_random_filter_data(
            train_path, test_path, out_dir, seed=args.seed
        )

    print(f"[E3 retrain] {ablation}: train={tr}  test={te}")
    if args.dry_run:
        print(f"  --dry-run: not invoking train_ranker (flags: {extra_flags}).")
        return 0

    cmd = [
        sys.executable, "-m", "experiments.train_ranker",
        "--run-id", f"e3_{ablation}",
        "--train-path", str(tr),
        "--test-path", str(te),
        "--skip-cv-cal",
        *extra_flags,
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    from baselines.ours import load_ours
    cfg = OmegaConf.load(CONFIG_PATH)
    policy = load_ours(runs_dir)
    seeds = canonical_test_seeds(cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = evaluate_policy(
        ablation, policy, seeds, cfg,
        results_dir=RESULTS_DIR, save=True, verbose=False,
    )
    print(f"\n[E3 retrain] {ablation}: "
          f"sla_breach={df['service_failure_rate'].mean():.4f}  "
          f"composite_cost={df['composite_cost'].mean():.4f}")
    return 0


def cmd_relabel(args: argparse.Namespace) -> int:
    """Print the recipe for the two ablations that need a fresh labelling pass.

    `tau1_snapshot_only` and `single_sample_rollout` change the ESTIMATOR, not
    the model, so no data transform on an existing parquet can produce them.

    `single_sample_rollout` (M=1) is the submitted labelling scheme, and it is
    the ablation that justifies M=20: at M=1 the rollout variance is identically
    zero and each label is hindsight-optimal for one realised future rather than
    an estimate of expected cost. If its KPIs match the deployed model's, the
    multi-sample machinery Reviewer 2 (3) asked for bought nothing and the paper
    must say so.
    """
    name = args.ablation
    flags = " ".join(RELABEL_ABLATIONS[name])
    data = f"data/e3_{name}"
    print(f"[E3 relabel] {name} — changes the estimator, so it needs a full "
          f"labelling pass.\n")
    print(f"  1) python -m experiments.generate_labels {flags} \\")
    print(f"       --train-out {data}/train.parquet \\")
    print(f"       --test-out {data}/test.parquet")
    print(f"  2) python -m experiments.train_ranker --run-id e3_{name} \\")
    print(f"       --train-path {data}/train.parquet \\")
    print(f"       --test-path {data}/test.parquet --skip-cv-cal")
    print(f"  3) python -m experiments.evaluate --method ours \\")
    print(f"       --run-dir runs/e3_{name} --results-dir results/E3")
    print("\n  Step 1 is the expensive one; run it from a detached session.")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Aggregate all E3 result parquets into one comparison table vs OURS."""
    cfg = OmegaConf.load(CONFIG_PATH)
    from experiments.stats import compare_methods

    scenario = args.scenario
    if scenario == "default":
        ours_path = REPO_ROOT / "results" / "ours.parquet"
        e3_dir = RESULTS_DIR
    else:
        ours_path = REPO_ROOT / "results" / f"scenario_{scenario}" / "ours.parquet"
        e3_dir = RESULTS_DIR / f"scenario_{scenario}"
    if not ours_path.exists():
        raise SystemExit(f"missing {ours_path}")
    parts: list[pd.DataFrame] = [pd.read_parquet(ours_path).assign(method="ours")]

    if not e3_dir.exists():
        raise SystemExit(f"no E3 results yet under {e3_dir}")
    for p in sorted(e3_dir.glob("*.parquet")):
        if p.stem in ("ours", "e3_summary"):
            continue
        parts.append(pd.read_parquet(p).assign(method=p.stem))
    df_long = pd.concat(parts, ignore_index=True)

    metrics = ["service_failure_rate", "mean_tardiness", "composite_cost"]
    out_rows: list[pd.DataFrame] = []
    for m in metrics:
        d = compare_methods(
            df_long, metric=m, baseline="ours",
            n_resamples=int(cfg.experiments.stats.bootstrap_resamples),
            q=float(cfg.experiments.stats.fdr_q),
        )
        d["metric"] = m
        out_rows.append(d)
        print(f"\n[E3 summary] {m}:")
        cols = ["method", "point", "ci_lo", "ci_hi",
                "diff_point", "p_raw", "p_adj_bh", "reject_bh"]
        cols = [c for c in cols if c in d.columns]
        print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = e3_dir / "e3_summary.parquet"
    pd.concat(out_rows, ignore_index=True).to_parquet(out_path, index=False)
    print(f"\n[E3 summary] wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_inf = sub.add_parser("inference", help="Run inference-only ablations.")
    p_inf.add_argument("--ablations", type=str, nargs="*", default=None,
                       help=f"Subset of {INFERENCE_ABLATIONS}; default = all.")
    p_inf.add_argument("--run-dir", type=Path, default=None)
    p_inf.add_argument("--n-test", type=int, default=None)
    p_inf.add_argument("--verbose", action="store_true")
    p_inf.add_argument("--scenario", type=str, default="default")
    p_inf.set_defaults(func=cmd_inference)

    p_re = sub.add_parser("retrain", help="Run a retrain ablation.")
    p_re.add_argument("ablation", type=str,
                      choices=RETRAIN_ABLATIONS)
    p_re.add_argument("--seed", type=int, default=1337)
    p_re.add_argument("--dry-run", action="store_true",
                      help="Prepare data but don't kick off training.")
    p_re.set_defaults(func=cmd_retrain)

    p_rl = sub.add_parser("relabel",
                          help="Recipe for ablations needing a labelling pass.")
    p_rl.add_argument("ablation", type=str, choices=sorted(RELABEL_ABLATIONS))
    p_rl.set_defaults(func=cmd_relabel)

    p_sum = sub.add_parser("summary", help="Aggregate E3 results + stats.")
    p_sum.add_argument("--scenario", type=str, default="default")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
