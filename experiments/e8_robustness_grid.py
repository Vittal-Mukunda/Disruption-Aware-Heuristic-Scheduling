"""Phase 8 / E8 — robustness grid across un-tuned configs.

Evaluates OURS + greedy_mpc + snapshot_xgb + best_static (fefo) across a 2-D grid
of arrival_rate × sla_tightness configs that were NOT tuned in the pilot (Phase 2).
The pilot was tuned on arrival_rate=1.65, sla_tightness=default only.

Claim: the relative ranking of methods is stable (or degrades gracefully) across
the untouched cells, defusing the "tuned sandbox" reviewer objection.

Grid definition:
  - arrival_rate ∈ {1.4, 1.5, 1.65*, 1.8}  (* = tuned in Phase 2 pilot)
  - sla_tightness ∈ {tight, default, loose}

Default = config.yaml base; tight = narrower SLA window; loose = wider SLA window.

THE SHELF LIFE IS HELD FIXED, DELIBERATELY. Only `sla_due_triangular` moves;
`shelf_life_triangular` stays at its configured value. Under the two-clock order
model of `simulation/orders.py` that means the grid also varies *which clock
binds*: in the `tight` cells the customer deadline dominates and expiry rarely
binds first, in `loose` the reverse. That is the intended reading — the sweep
asks whether the ranking survives a shift in the balance between the two failure
modes, not merely a uniform rescaling of urgency. Say so when reporting, because
a reviewer who has read Section 3 will notice the interaction.

Output:
  - results/E8/robustness_grid_<arrival>_<sla>.parquet (per-shift KPI, 4 methods each)
  - results/E8/robustness_grid_summary.parquet (mean + CI per method × config cell)
  - figures/E8/robustness_grid_heatmap_{metric}.{png,pdf} (ranking matrix)
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
import seaborn as sns  # noqa: E402

from experiments.evaluate import (  # noqa: E402
    canonical_test_seeds,
    evaluate_policy,
    evaluate_policy_env_aware,
    _build_policy,
)
from experiments.stats import bootstrap_mean_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E8"
FIG_DIR = REPO_ROOT / "figures" / "E8"

# Grid definition — these cells were NOT tuned in the Phase 2 pilot.
# The pilot tuned on arrival_rate=1.65, sla_tightness=default only.
ARRIVAL_RATES = [1.4, 1.5, 1.65, 1.8]  # 1.65 = tuned
SLA_TIGHTNESS_LEVELS = {
    "tight": [10.0, 30.0, 60.0],      # tighter window (minutes)
    "default": [15.0, 45.0, 90.0],    # config.yaml baseline (tuned)
    "loose": [20.0, 60.0, 120.0],     # wider window
}
METHODS = ["ours", "greedy_mpc", "snapshot_xgb", "fefo"]


def _config_cell_id(arrival_rate: float, sla_tight_key: str) -> str:
    """Human-readable cell identifier."""
    return f"arr{arrival_rate:.2f}_{sla_tight_key}"


def _build_robustness_cfg(
    base_cfg: DictConfig,
    arrival_rate: float,
    sla_tightness_triangular: list[float],
) -> DictConfig:
    """Create a cfg by overlaying arrival_rate and sla_due_triangular.

    Does NOT use apply_scenario (which is reserved for explicit named scenarios).
    Instead, directly patches sim.arrivals and sim.order_attrs.
    """
    new_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(new_cfg):
        new_cfg.sim.arrivals.base_rate_per_minute = float(arrival_rate)
        new_cfg.sim.order_attrs.sla_due_triangular = list(
            float(v) for v in sla_tightness_triangular
        )
    return new_cfg


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate all methods on each grid cell."""
    base_cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(base_cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    total_cells = len(ARRIVAL_RATES) * len(SLA_TIGHTNESS_LEVELS)
    cell_idx = 0

    for arrival_rate in ARRIVAL_RATES:
        for sla_key, sla_tri in SLA_TIGHTNESS_LEVELS.items():
            cell_idx += 1
            cell_id = _config_cell_id(arrival_rate, sla_key)
            cfg = _build_robustness_cfg(base_cfg, arrival_rate, sla_tri)

            print(f"\n[E8 grid] cell {cell_idx}/{total_cells}: {cell_id}")
            cell_results_dir = RESULTS_DIR / cell_id
            cell_results_dir.mkdir(parents=True, exist_ok=True)

            for method in METHODS:
                print(f"  {method}...", end="", flush=True)
                policy, env_aware = _build_policy(method, args.run_dir)
                eval_fn = (
                    evaluate_policy_env_aware if env_aware
                    else evaluate_policy
                )
                df = eval_fn(
                    method, policy, seeds, cfg,
                    results_dir=cell_results_dir, save=True, verbose=False,
                )

                # Summarize this cell × method with bootstrap CI
                for metric in ("service_failure_rate", "composite_cost", "mean_tardiness"):
                    ci = bootstrap_mean_ci(
                        df[metric].to_numpy(dtype=np.float64),
                        n_resamples=10000, seed=1337,
                    )
                    summary_rows.append({
                        "arrival_rate": float(arrival_rate),
                        "sla_tightness": sla_key,
                        "method": method,
                        "metric": metric,
                        **ci.as_row(),
                    })
                print(" OK", end="", flush=True)

    print("\n")
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "robustness_grid_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    print(f"[E8 grid] summary written to {summary_path.relative_to(REPO_ROOT)}")

    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Generate summary table + heatmaps from existing grid results."""
    summary_path = RESULTS_DIR / "robustness_grid_summary.parquet"
    if not summary_path.exists():
        raise SystemExit(f"Grid summary not found: {summary_path}")

    df_summary = pd.read_parquet(summary_path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Print summary table: mean (CI) per method × arrival_rate × sla_tightness × metric
    print("\n[E8 grid] Robustness Grid Summary")
    print("=" * 100)

    for metric in ("service_failure_rate", "composite_cost", "mean_tardiness"):
        print(f"\n{metric.upper()}")
        print("-" * 100)

        for sla_key in SLA_TIGHTNESS_LEVELS.keys():
            print(f"\n  SLA Tightness = {sla_key}")
            subset = df_summary[
                (df_summary["metric"] == metric)
                & (df_summary["sla_tightness"] == sla_key)
            ]
            for arrival in sorted(ARRIVAL_RATES):
                cell = subset[subset["arrival_rate"] == arrival]
                if cell.empty:
                    continue
                tuned_marker = " [TUNED]" if arrival == 1.65 and sla_key == "default" else ""
                print(f"    arrival_rate={arrival}{tuned_marker}")
                for method in METHODS:
                    row = cell[cell["method"] == method]
                    if row.empty:
                        print(f"      {method:15s} — N/A")
                    else:
                        pt = float(row["point"].iloc[0])
                        lo = float(row["ci_lo"].iloc[0])
                        hi = float(row["ci_hi"].iloc[0])
                        print(f"      {method:15s} {pt:8.4f} [{lo:8.4f}, {hi:8.4f}]")

    # Heatmaps: one per metric, rows = method, cols = (arrival × sla), values = point estimate
    for metric in ("service_failure_rate", "composite_cost"):
        df_pivot = df_summary[df_summary["metric"] == metric].copy()
        df_pivot["cell"] = (
            df_pivot["arrival_rate"].astype(str) + "_" + df_pivot["sla_tightness"]
        )

        # Create pivot: rows = method, cols = cell
        pivot = df_pivot.pivot_table(
            index="method", columns="cell", values="point", aggfunc="first"
        )

        # Reorder columns by (arrival_rate, sla_tightness)
        col_order = []
        for arr in ARRIVAL_RATES:
            for sla_key in SLA_TIGHTNESS_LEVELS.keys():
                col_id = f"{arr}_" + sla_key
                if col_id in pivot.columns:
                    col_order.append(col_id)
        pivot = pivot[col_order]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap(
            pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"E8 — Robustness Grid: {metric}")
        ax.set_xlabel("(arrival_rate, sla_tightness)")
        ax.set_ylabel("method")
        # Highlight the tuned cell
        tuned_col_idx = col_order.index("1.65_default")
        ax.add_patch(
            plt.Rectangle((tuned_col_idx, 0), 1, len(METHODS),
                         fill=False, edgecolor="blue", lw=2.5)
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"robustness_grid_heatmap_{metric}.png", dpi=150)
        fig.savefig(FIG_DIR / f"robustness_grid_heatmap_{metric}.pdf")
        plt.close(fig)

    print(f"\n[E8 grid] heatmaps written to {FIG_DIR.relative_to(REPO_ROOT)}/")
    print(f"[E8 grid] summary parquet -> {summary_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 / E8 — robustness grid across un-tuned configs."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("eval", help="Run eval on all grid cells.")
    p_eval.add_argument("--n-test", type=int, default=None,
                        help="Cap on test shifts (for smoke testing).")
    p_eval.add_argument("--run-dir", type=Path, default=None,
                        help="Override artifact dir for OURS (snapshot_xgb, etc).")
    p_eval.set_defaults(func=cmd_eval)

    p_summ = sub.add_parser("summary", help="Generate summary tables and heatmaps.")
    p_summ.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
