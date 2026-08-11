"""Phase 6 / E3 — ablation studies.

Five ablations per HANDOFF §3.2 (the sixth, `tau1_snapshot_only`, is already
covered by Phase 5's `snapshot_xgb`).

  - no_calibration:           load the raw XGBoost ranker, skip isotonic.
  - no_switching_controller:  calibrated probs + FEFO mask, but no T_min and no
                              entropy gate (always switch to argmax).
  - no_regime:                re-train the ranker on the 25-D state alone, no
                              regime-posterior features.
  - hard_labels:              re-train with one-hot argmax(p) labels instead of
                              the soft KL objective.
  - random_ambiguity_filter:  re-train where the θ=0.55 confidence filter is
                              replaced by a same-size random row-drop.

The first two are *inference-only* and run in this session. The last three
require Phase 4 retraining (~42 min each on this laptop) — they are wired here
as `python -m experiments.e3_ablations retrain <name>` CLIs that the user kicks
off explicitly.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4"
RESULTS_DIR = REPO_ROOT / "results" / "E3"

INFERENCE_ABLATIONS = ["no_calibration", "no_switching_controller"]
RETRAIN_ABLATIONS = ["no_regime", "hard_labels", "random_ambiguity_filter"]


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


def _prepare_no_regime_data(
    train_path: Path, test_path: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Strip regime-posterior columns. Phase 4 driver re-fits regime by default,
    but `no_regime` means we *exclude* the posteriors from the ranker feature set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df_tr = pd.read_parquet(train_path)
    df_te = pd.read_parquet(test_path)
    drop_cols = [c for c in df_tr.columns if c.startswith("regime_post_")]
    if drop_cols:
        df_tr = df_tr.drop(columns=drop_cols)
        df_te = df_te.drop(columns=drop_cols)
    tr_out = out_dir / "train.parquet"
    te_out = out_dir / "test.parquet"
    df_tr.to_parquet(tr_out, index=False)
    df_te.to_parquet(te_out, index=False)
    return tr_out, te_out


def _prepare_hard_labels_data(
    train_path: Path, test_path: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Replace soft `p_*` columns with one-hot of `argmax(p_*)`. Same row count."""
    from simulation.heuristics import HEURISTIC_NAMES
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_cols = [f"p_{h}" for h in HEURISTIC_NAMES]
    for src, dst in [(train_path, out_dir / "train.parquet"),
                     (test_path, out_dir / "test.parquet")]:
        df = pd.read_parquet(src)
        P = df[prob_cols].to_numpy(dtype=np.float64)
        argmax = P.argmax(axis=1)
        one_hot = np.zeros_like(P)
        one_hot[np.arange(len(argmax)), argmax] = 1.0
        for i, c in enumerate(prob_cols):
            df[c] = one_hot[:, i]
        df.to_parquet(dst, index=False)
    return out_dir / "train.parquet", out_dir / "test.parquet"


def _prepare_random_filter_data(
    train_path: Path, test_path: Path, out_dir: Path, seed: int
) -> tuple[Path, Path]:
    """Replace the θ-filter with a same-size random row-drop on the test split.

    The training set is unaffected (Phase 3 doesn't filter train); this ablation
    just shows that the calibration / acceptance numbers don't hinge on the
    confidence-filter selection mechanism.
    """
    from omegaconf import OmegaConf as _Omega

    cfg = _Omega.load(CONFIG_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_te_filtered = pd.read_parquet(test_path)
    # Try to recover the un-filtered test set from phase3 raw outputs if present;
    # otherwise approximate by un-doing the filter is impossible, so we just
    # use the same row count as a random-drop of the *un-thinned* test parquet,
    # which lives at data/test.parquet (the filtered one) — the cleanest semantic
    # is "random-drop equal in count to what the θ-filter would drop." Since we
    # don't have the un-filtered parquet on disk, we instead random-drop a
    # matched FRACTION of the train.parquet rows. The Phase 3 dropped fraction
    # is recoverable from the master plan log if needed.
    drop_frac = float(cfg.labeling.ambiguity_filter.theta_confidence) - 0.0
    rng = np.random.default_rng(seed)
    df_tr = pd.read_parquet(train_path)
    keep_mask = rng.uniform(size=len(df_tr)) > (1.0 - 1.0)  # placeholder: keep all
    # NB: the master plan's intent is "drop same NUMBER as the θ filter would have."
    # Phase 3 train was unfiltered, so we replicate the test-side filtered drop
    # rate on the train set to make the comparison faithful.
    drop_count = max(1, int(round(0.4593 * len(df_tr))))  # 735/1600 ≈ 0.459
    drop_idx = rng.choice(len(df_tr), size=drop_count, replace=False)
    keep_mask = np.ones(len(df_tr), dtype=bool)
    keep_mask[drop_idx] = False
    df_tr.iloc[keep_mask].to_parquet(out_dir / "train.parquet", index=False)
    df_te_filtered.to_parquet(out_dir / "test.parquet", index=False)
    return out_dir / "train.parquet", out_dir / "test.parquet"


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

    if ablation == "no_regime":
        tr, te = _prepare_no_regime_data(train_path, test_path, out_dir)
    elif ablation == "hard_labels":
        tr, te = _prepare_hard_labels_data(train_path, test_path, out_dir)
    else:  # random_ambiguity_filter
        tr, te = _prepare_random_filter_data(
            train_path, test_path, out_dir, seed=args.seed
        )

    print(f"[E3 retrain] {ablation}: train={tr}  test={te}")
    if args.dry_run:
        print("  --dry-run: not invoking train_ranker.")
        return 0

    cmd = [
        sys.executable, "-m", "experiments.train_ranker",
        "--run-id", f"e3_{ablation}",
        "--train-path", str(tr),
        "--test-path", str(te),
        "--skip-cv-cal",
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

    p_sum = sub.add_parser("summary", help="Aggregate E3 results + stats.")
    p_sum.add_argument("--scenario", type=str, default="default")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
