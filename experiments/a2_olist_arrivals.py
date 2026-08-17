"""Phase 8 / Deliverable A-stretch — real-arrival-shape robustness.

Deliverable A (a_realdata_validation.py) shows the simulator's *Poisson* arrival
process differs in shape from the Olist trace (CV 1 vs 2.68, skew 2 vs 11). That
is a passive finding ("the sim differs from reality"). This experiment turns it
active: it re-runs the four headline methods under an arrival stream that
bootstraps the empirical Olist inter-arrival distribution — injecting the real
burstiness while holding the mean rate fixed at base_rate_per_minute — and asks
whether the method ranking from the main results survives.

No re-tuning, no re-training: the frozen runs/phase4 ranker (§3.3) and the
Phase 2 tuned config (arrival_rate=1.65, sla=default) are used verbatim. The
ONLY thing that changes between the two cells is sim.arrivals.arrival_mode.

Cells:
  - poisson : config.yaml baseline (exponential gaps, CV=1)
  - olist   : empirical Olist burstiness (CV=2.68, skew=11), same mean rate

Methods: ours, greedy_mpc, snapshot_xgb, fefo — on the canonical 50 test shifts.

Output:
  - results/A2/<mode>/<method>.parquet      (per-shift KPI, 4 methods x 2 modes)
  - results/A2/olist_arrivals_summary.parquet (mean + 95% CI per method x mode)
  - figures/A2/olist_arrivals_compare.{png,pdf} (poisson vs olist, per metric)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.evaluate import (  # noqa: E402
    canonical_test_seeds,
    evaluate_policy,
    evaluate_policy_env_aware,
    _build_policy,
)
from experiments.stats import bootstrap_mean_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "A2"
FIG_DIR = REPO_ROOT / "figures" / "A2"

ARRIVAL_MODES = ["poisson", "olist"]
# See the note in e8_robustness_grid: EEDD is the screened pool's strongest
# static rule, FEFO was dropped for zero marginal contribution.
METHODS = ["ours", "greedy_mpc", "snapshot_xgb", "eedd"]
METRICS = ["service_failure_rate", "composite_cost", "mean_tardiness"]
SUMMARY_PATH = RESULTS_DIR / "olist_arrivals_summary.parquet"


def _build_arrival_mode_cfg(base_cfg: DictConfig, arrival_mode: str) -> DictConfig:
    """Overlay sim.arrivals.arrival_mode on the (tuned) base config."""
    new_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(new_cfg):
        new_cfg.sim.arrivals.arrival_mode = str(arrival_mode)
    return new_cfg


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate all methods under each arrival mode."""
    base_cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(base_cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    modes = [args.mode_only] if args.mode_only else ARRIVAL_MODES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    if SUMMARY_PATH.exists():
        # Preserve rows for modes not re-run in this invocation.
        existing = pd.read_parquet(SUMMARY_PATH)
        summary_rows.extend(
            existing[~existing["arrival_mode"].isin(modes)].to_dict("records")
        )

    for arrival_mode in modes:
        cfg = _build_arrival_mode_cfg(base_cfg, arrival_mode)
        mode_dir = RESULTS_DIR / arrival_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[A2] arrival_mode = {arrival_mode}  (n_test={len(seeds)})")

        for method in METHODS:
            print(f"  {method}...", end="", flush=True)
            policy, env_aware = _build_policy(method, args.run_dir)
            eval_fn = evaluate_policy_env_aware if env_aware else evaluate_policy
            df = eval_fn(
                method, policy, seeds, cfg,
                results_dir=mode_dir, save=True, verbose=False,
            )
            for metric in METRICS:
                ci = bootstrap_mean_ci(
                    df[metric].to_numpy(dtype=np.float64),
                    n_resamples=10000, seed=1337,
                )
                summary_rows.append({
                    "arrival_mode": arrival_mode,
                    "method": method,
                    "metric": metric,
                    **ci.as_row(),
                })
            print(" OK", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(SUMMARY_PATH, index=False)
    print(f"\n[A2] summary written to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


def _rank_by(subset: pd.DataFrame) -> dict[str, int]:
    """Rank methods 1..N by point estimate (lower is better)."""
    ordered = subset.sort_values("point")["method"].tolist()
    return {m: i + 1 for i, m in enumerate(ordered)}


def cmd_summary(args: argparse.Namespace) -> int:
    """Print the poisson-vs-olist comparison and render the figure."""
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"A2 summary not found: {SUMMARY_PATH}. Run `eval` first.")

    df = pd.read_parquet(SUMMARY_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[A2] Olist-arrival robustness -- poisson vs olist")
    print("=" * 86)

    for metric in METRICS:
        print(f"\n{metric.upper()}  (lower is better)")
        print("-" * 86)
        print(f"  {'method':14s} {'poisson':>22s} {'olist':>22s} {'delta':>12s}")
        poisson_sub = df[(df["metric"] == metric) & (df["arrival_mode"] == "poisson")]
        olist_sub = df[(df["metric"] == metric) & (df["arrival_mode"] == "olist")]
        p_rank = _rank_by(poisson_sub)
        o_rank = _rank_by(olist_sub)
        for method in METHODS:
            p_row = poisson_sub[poisson_sub["method"] == method]
            o_row = olist_sub[olist_sub["method"] == method]
            if p_row.empty or o_row.empty:
                print(f"  {method:14s}  — incomplete (run eval)")
                continue
            p = float(p_row["point"].iloc[0])
            o = float(o_row["point"].iloc[0])
            p_lo, p_hi = float(p_row["ci_lo"].iloc[0]), float(p_row["ci_hi"].iloc[0])
            o_lo, o_hi = float(o_row["ci_lo"].iloc[0]), float(o_row["ci_hi"].iloc[0])
            print(
                f"  {method:14s} "
                f"{p:7.4f} [{p_lo:6.4f},{p_hi:6.4f}] #{p_rank[method]} "
                f"{o:7.4f} [{o_lo:6.4f},{o_hi:6.4f}] #{o_rank[method]} "
                f"{o - p:+11.4f}"
            )
        winner_p = min(p_rank, key=p_rank.get) if p_rank else "?"
        winner_o = min(o_rank, key=o_rank.get) if o_rank else "?"
        verdict = "OURS HOLDS" if winner_o == "ours" else f"RANK SHIFT -> {winner_o}"
        print(f"  best: poisson={winner_p}  olist={winner_o}   [{verdict}]")

    _make_figure(df)
    print(f"\n[A2] figure -> {FIG_DIR.relative_to(REPO_ROOT)}/olist_arrivals_compare.png")
    return 0


def _make_figure(df: pd.DataFrame) -> None:
    """Grouped bars: poisson vs olist per method, one panel per metric."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.5))
    x = np.arange(len(METHODS))
    width = 0.38
    colors = {"poisson": "#2c7fb8", "olist": "#de2d26"}

    for ax, metric in zip(axes, METRICS):
        sub = df[df["metric"] == metric]
        for i, mode in enumerate(ARRIVAL_MODES):
            ms = sub[sub["arrival_mode"] == mode].set_index("method")
            pts, los, his = [], [], []
            for m in METHODS:
                if m in ms.index:
                    pts.append(float(ms.loc[m, "point"]))
                    los.append(float(ms.loc[m, "point"] - ms.loc[m, "ci_lo"]))
                    his.append(float(ms.loc[m, "ci_hi"] - ms.loc[m, "point"]))
                else:
                    pts.append(np.nan)
                    los.append(0.0)
                    his.append(0.0)
            ax.bar(
                x + (i - 0.5) * width, pts, width,
                yerr=[los, his], capsize=3, label=mode, color=colors[mode],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=20, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.legend(title="arrival_mode", fontsize=8)

    fig.suptitle(
        "Deliverable A-stretch — method KPIs under Poisson vs empirical Olist arrivals "
        "(95% bootstrap CI)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "olist_arrivals_compare.png", dpi=150)
    fig.savefig(FIG_DIR / "olist_arrivals_compare.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 / Deliverable A-stretch — Olist-arrival robustness."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Run eval under each arrival mode.")
    p_eval.add_argument("--n-test", type=int, default=None,
                        help="Cap on test shifts (for smoke testing).")
    p_eval.add_argument("--run-dir", type=Path, default=None,
                        help="Override artifact dir for OURS.")
    p_eval.add_argument("--mode-only", type=str, default=None,
                        choices=ARRIVAL_MODES,
                        help="Run a single arrival mode instead of both.")
    p_eval.set_defaults(func=cmd_eval)

    p_summ = sub.add_parser("summary", help="Print comparison table and figure.")
    p_summ.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
