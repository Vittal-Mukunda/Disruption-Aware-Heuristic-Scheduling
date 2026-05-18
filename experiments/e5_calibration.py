"""Phase 6 / E5 — calibration quality + SHAP explanations.

Operates on `data/test.parquet` (the snapshot soft labels) and the Phase 4
artifacts under `runs/phase4/`. Three deliverables per HANDOFF §3.2:

  1. Reliability diagrams (top-1 confidence vs accuracy) pre- and post-isotonic,
     plus per-class reliability diagrams.
  2. ECE / Brier / soft-xent tables (pre vs post calibration).
  3. SHAP global summary + per-class summary on a stratified test subsample.

Outputs:
  figures/E5/reliability_pre_post.{pdf,png}
  figures/E5/per_class_reliability.{pdf,png}
  figures/E5/shap_summary.{pdf,png}
  figures/E5/shap_per_class.{pdf,png}
  results/E5/calibration_table.parquet
  results/E5/shap_global_importance.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from models.calibration import brier_one_hot, soft_cross_entropy, top1_ece  # noqa: E402
from models.heuristic_ranker import PROB_COLUMNS  # noqa: E402
from simulation.heuristics import HEURISTIC_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "test.parquet"
FIG_DIR = REPO_ROOT / "figures" / "E5"
RESULTS_DIR = REPO_ROOT / "results" / "E5"


def _reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True),
                      0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        m = bin_ids == b
        n_b = int(m.sum())
        if n_b == 0:
            rows.append({"bin": b, "lo": float(bin_edges[b]),
                         "hi": float(bin_edges[b + 1]),
                         "n": 0, "conf": np.nan, "acc": np.nan})
        else:
            rows.append({
                "bin": b,
                "lo": float(bin_edges[b]),
                "hi": float(bin_edges[b + 1]),
                "n": n_b,
                "conf": float(confidences[m].mean()),
                "acc": float(correct[m].mean()),
            })
    return pd.DataFrame(rows)


def _plot_reliability_pre_post(
    pre: pd.DataFrame, post: pd.DataFrame, ece_pre: float, ece_post: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, df, title, ece in [
        (axes[0], pre, "Pre-calibration", ece_pre),
        (axes[1], post, "Post-calibration (isotonic)", ece_post),
    ]:
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        ax.bar(df["lo"], df["acc"], width=df["hi"] - df["lo"],
               align="edge", color="firebrick", alpha=0.55,
               edgecolor="black", label="Accuracy")
        ax.plot(df["conf"], df["acc"], "o-", color="navy", lw=1.5,
                label="Mean conf vs acc")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Empirical accuracy")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _per_class_reliability(
    probs_post: np.ndarray, y_true: np.ndarray, out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for k, h in enumerate(HEURISTIC_NAMES):
        ax = axes.flat[k]
        conf_k = probs_post[:, k]
        correct_k = (y_true == k).astype(np.float64)
        df = _reliability_bins(conf_k, correct_k, n_bins=10)
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        ax.plot(df["conf"], df["acc"], "o-", color="firebrick", lw=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted p(class)")
        ax.set_ylabel("Empirical P(true=k)")
        ax.set_title(f"Class {h}")
    fig.suptitle("E5 — Per-class reliability (post-calibration)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def cmd_reliability(args: argparse.Namespace) -> int:
    run_dir = args.run_dir or DEFAULT_RUN_DIR
    test_path = args.test_path or DEFAULT_TEST_PATH
    print(f"[E5] run_dir={run_dir}  test_path={test_path}")
    if not test_path.exists():
        raise SystemExit(f"missing {test_path}")

    df = pd.read_parquet(test_path)
    print(f"[E5] {len(df)} test rows.")
    meta = joblib.load(run_dir / "ranker_meta.joblib")
    feature_cols = list(meta["feature_cols"])
    calibrator = joblib.load(run_dir / "calibrator.joblib")
    base_model = calibrator.base_model

    if not all(c in df.columns for c in feature_cols):
        # If the test parquet doesn't have regime cols yet, attach them.
        regime_gmm = joblib.load(run_dir / "regime.joblib")
        from regime.regime_discovery import attach_regime_posteriors
        df = attach_regime_posteriors(df, regime_gmm)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    P_true = df[PROB_COLUMNS].to_numpy(dtype=np.float64)
    y_hard = P_true.argmax(axis=1).astype(np.int64)
    probs_pre = base_model.predict_proba(X)
    probs_post = calibrator.predict_proba(X)

    ece_pre = top1_ece(probs_pre, y_hard)
    ece_post = top1_ece(probs_post, y_hard)
    brier_pre = brier_one_hot(probs_pre, y_hard)
    brier_post = brier_one_hot(probs_post, y_hard)
    soft_pre = soft_cross_entropy(probs_pre, P_true)
    soft_post = soft_cross_entropy(probs_post, P_true)
    print(f"  ECE  pre={ece_pre:.4f}   post={ece_post:.4f}")
    print(f"  Brier pre={brier_pre:.4f}  post={brier_post:.4f}")
    print(f"  soft-xent pre={soft_pre:.4f}  post={soft_post:.4f}")

    bins_pre = _reliability_bins(probs_pre.max(axis=1),
                                 (probs_pre.argmax(axis=1) == y_hard).astype(float))
    bins_post = _reliability_bins(probs_post.max(axis=1),
                                  (probs_post.argmax(axis=1) == y_hard).astype(float))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_reliability_pre_post(
        bins_pre, bins_post, ece_pre, ece_post,
        FIG_DIR / "reliability_pre_post.png",
    )
    _per_class_reliability(probs_post, y_hard,
                           FIG_DIR / "per_class_reliability.png")

    table = pd.DataFrame([
        {"metric": "ece", "pre": ece_pre, "post": ece_post},
        {"metric": "brier", "pre": brier_pre, "post": brier_post},
        {"metric": "soft_xent", "pre": soft_pre, "post": soft_post},
    ])
    table.to_parquet(RESULTS_DIR / "calibration_table.parquet", index=False)
    print(f"  wrote {(RESULTS_DIR / 'calibration_table.parquet').relative_to(REPO_ROOT)}")
    return 0


def cmd_shap(args: argparse.Namespace) -> int:
    import shap

    cfg = OmegaConf.load(CONFIG_PATH)  # noqa: F841 - reserved for future config use
    run_dir = args.run_dir or DEFAULT_RUN_DIR
    test_path = args.test_path or DEFAULT_TEST_PATH

    meta = joblib.load(run_dir / "ranker_meta.joblib")
    feature_cols = list(meta["feature_cols"])
    booster = xgb.XGBClassifier()
    booster.load_model(str(run_dir / "model.json"))

    df = pd.read_parquet(test_path)
    if not all(c in df.columns for c in feature_cols):
        regime_gmm = joblib.load(run_dir / "regime.joblib")
        from regime.regime_discovery import attach_regime_posteriors
        df = attach_regime_posteriors(df, regime_gmm)

    rng = np.random.default_rng(args.seed)
    n_sample = min(int(args.sample), len(df))
    idx = rng.choice(len(df), size=n_sample, replace=False)
    X_sample = df.iloc[idx][feature_cols].to_numpy(dtype=np.float64)
    print(f"[E5 SHAP] sampling {n_sample} rows from {len(df)} test rows.")

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv_array = np.stack(shap_values, axis=0)  # (K, n, d)
    else:
        sv_array = np.transpose(shap_values, (2, 0, 1))  # (K, n, d)

    K, n, d = sv_array.shape
    assert d == len(feature_cols), f"SHAP dim mismatch {d} vs {len(feature_cols)}"
    mean_abs = np.abs(sv_array).mean(axis=1)  # (K, d)
    overall = mean_abs.mean(axis=0)  # (d,)

    df_imp = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": overall})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print("\n[E5 SHAP] Top 10 features by mean |SHAP|:")
    print(df_imp.head(10).to_string(index=False,
                                    float_format=lambda x: f"{x:.4f}"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_imp.to_parquet(RESULTS_DIR / "shap_global_importance.parquet", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top_k = min(20, len(df_imp))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * top_k)))
    ax.barh(df_imp["feature"][:top_k][::-1], df_imp["mean_abs_shap"][:top_k][::-1],
            color="firebrick")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"E5 — Global feature importance (top {top_k})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    fig.savefig(FIG_DIR / "shap_summary.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.3 * top_k)))
    top_features_idx = np.argsort(-overall)[:top_k]
    y = np.arange(top_k)
    width = 0.18
    for k, h in enumerate(HEURISTIC_NAMES):
        vals = mean_abs[k, top_features_idx]
        ax.barh(y + (k - 1.5) * width, vals[::-1], height=width, label=h)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_cols[i] for i in top_features_idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("E5 — Per-class feature importance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_per_class.png", dpi=150)
    fig.savefig(FIG_DIR / "shap_per_class.pdf")
    plt.close(fig)

    print(f"\n[E5 SHAP] figures -> {FIG_DIR.relative_to(REPO_ROOT)}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p_rel = sub.add_parser("reliability",
                           help="Reliability diagrams + calibration table.")
    p_rel.add_argument("--run-dir", type=Path, default=None)
    p_rel.add_argument("--test-path", type=Path, default=None)
    p_rel.set_defaults(func=cmd_reliability)

    p_shap = sub.add_parser("shap",
                            help="Global + per-class SHAP summary.")
    p_shap.add_argument("--run-dir", type=Path, default=None)
    p_shap.add_argument("--test-path", type=Path, default=None)
    p_shap.add_argument("--sample", type=int, default=500,
                        help="Subsample size for the SHAP computation.")
    p_shap.add_argument("--seed", type=int, default=1337)
    p_shap.set_defaults(func=cmd_shap)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
