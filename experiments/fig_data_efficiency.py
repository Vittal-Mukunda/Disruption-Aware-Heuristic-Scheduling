"""Phase 8 / manuscript — data-efficiency hero figure.

Plots the sample-efficiency curve: DAHS KPI vs the number of simulated training
shifts, against the snapshot (tau=1) ranker and the analytic greedy-MPC baseline
shown as horizontal reference lines. The point: DAHS trained on as few as 25
shifts already sits well below both baselines.

This script ONLY reads existing result files and renders a figure. It runs no
simulation and trains no model (HANDOFF 3.3 -- no re-running frozen phases).

Inputs:
  results/data_efficiency/data_efficiency_summary.json  (5 budgets x 5 reps)
  results/snapshot_xgb.parquet, results/greedy_mpc.parquet  (reference lines)

Output:
  figures/data_efficiency/data_efficiency_curve.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DE_JSON = ROOT / "results" / "data_efficiency" / "data_efficiency_summary.json"
FIG_DIR = ROOT / "figures" / "data_efficiency"


REQUIRED = ("service_failure_rate_mean", "composite_cost_mean")


def main() -> int:
    if not DE_JSON.exists():
        raise SystemExit(
            f"{DE_JSON} not found. This figure is built from the DAHS "
            f"data-efficiency sweep; run "
            f"`python -m experiments.e2_main data_efficiency` first."
        )
    de = pd.DataFrame(json.loads(DE_JSON.read_text()))
    missing = [c for c in REQUIRED if c not in de.columns]
    if missing:
        # A pre-revision summary has the same filename and the OLD metric keys,
        # so it loads cleanly and fails inside groupby.agg with a bare KeyError
        # naming neither the file nor the cause. This is the figure Section 6.3
        # calls the central one, so it is worth failing legibly.
        raise SystemExit(
            f"{DE_JSON} is missing {missing} — it holds {sorted(de.columns)}, "
            f"which is the pre-revision schema. Re-run "
            f"`python -m experiments.e2_main data_efficiency` to rebuild it "
            f"against the corrected objective."
        )
    grp = de.groupby("budget").agg(
        sla_mean=("service_failure_rate_mean", "mean"),
        sla_std=("service_failure_rate_mean", "std"),
        cost_mean=("composite_cost_mean", "mean"),
        cost_std=("composite_cost_mean", "std"),
    ).reset_index()

    snap_path = ROOT / "results" / "snapshot_xgb.parquet"
    mpc_path = ROOT / "results" / "greedy_mpc.parquet"
    if not snap_path.exists() or not mpc_path.exists():
        raise SystemExit(
            f"need {snap_path.name} and {mpc_path.name} as reference lines. "
            "Run `python -m experiments.evaluate --method snapshot_xgb --n-jobs=-1` "
            "and ensure greedy_mpc.parquet exists."
        )
    snap = pd.read_parquet(snap_path)
    mpc = pd.read_parquet(mpc_path)
    ref = {
        "snapshot_xgb": {
            "sla": float(snap["service_failure_rate"].mean()),
            "cost": float(snap["composite_cost"].mean()),
        },
        "greedy_mpc": {
            "sla": float(mpc["service_failure_rate"].mean()),
            "cost": float(mpc["composite_cost"].mean()),
        },
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left panel plots `service_failure_rate`, not the submitted `sla_breach_rate`.
    # They are different quantities — the first counts every arrived order, the
    # second only completed ones — so the axis has to say which (Reviewer 2, 1).
    panels = [
        ("sla_mean", "sla_std", "sla", "Service-failure rate (per arrived order)"),
        ("cost_mean", "cost_std", "cost", "Composite cost"),
    ]
    for ax, (mcol, scol, refkey, ylabel) in zip(axes, panels):
        ax.errorbar(
            grp["budget"], grp[mcol], yerr=grp[scol],
            marker="o", capsize=4, lw=2, color="#1a9850",
            label="DAHS (ours) -- mean +/- std, 5 reps",
        )
        ax.axhline(
            ref["snapshot_xgb"][refkey], ls="--", color="#d73027",
            label="Snapshot ranker (tau=1, 250 shifts)",
        )
        ax.axhline(
            ref["greedy_mpc"][refkey], ls=":", color="#4575b4",
            label="Greedy-MPC (analytic 1-step lookahead)",
        )
        ax.set_xlabel("Training-shift budget")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(grp["budget"]))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Neutral title. The submitted one asserted the conclusion ("trained on 25
    # shifts already outperforms..."), which the rebuild may not reproduce — the
    # sample-efficiency result is being re-derived, not carried over.
    fig.suptitle(
        "Sample efficiency: DAHS vs the snapshot ranker and analytic MPC, "
        "by training-shift budget",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "data_efficiency_curve.png", dpi=150)
    fig.savefig(FIG_DIR / "data_efficiency_curve.pdf")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR.relative_to(ROOT)}/data_efficiency_curve.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
