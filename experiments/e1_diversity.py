"""Phase 6 / E1 — heuristic diversity heatmap.

Reads `data/pilot_costs.parquet` (50 shifts × 32 intervals × 4 heuristics, one
row per (shift, interval, heuristic) cost). For each (shift, interval)
snapshot the winner is `argmin_h cost`; aggregating across all snapshots gives
the per-heuristic win rate, and aggregating per `interval_idx` (or per shift)
yields the two-axis heatmap the paper uses to motivate "the right rule depends
on context".

Outputs:
  figures/E1/diversity_heatmap_interval.{pdf,png}  -- wins per (interval, heuristic)
  figures/E1/diversity_heatmap_shift.{pdf,png}     -- wins per (shift, heuristic)
  results/E1/win_rate_table.parquet                -- overall, per-interval, per-shift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulation.heuristics import HEURISTIC_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_PARQUET = REPO_ROOT / "data" / "pilot_costs.parquet"
DEFAULT_FIG_DIR = REPO_ROOT / "figures" / "E1"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "E1"

TIE_JITTER = 1e-9


def compute_winners(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """For each (shift_id, interval_idx) return argmin heuristic. Tie-break with rng."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["cost_j"] = work["cost"].to_numpy() + TIE_JITTER * rng.uniform(size=len(work))
    pivot = work.pivot_table(
        index=["shift_id", "interval_idx"], columns="heuristic", values="cost_j"
    )
    pivot = pivot.reindex(columns=HEURISTIC_NAMES)
    winners = pivot.idxmin(axis=1).reset_index().rename(columns={0: "winner"})
    winners.columns = ["shift_id", "interval_idx", "winner"]
    return winners


def heatmap_by_axis(
    winners: pd.DataFrame, axis: str
) -> pd.DataFrame:
    """Counts of wins per (axis_value, heuristic), then row-normalized."""
    if axis not in ("shift_id", "interval_idx"):
        raise ValueError(f"axis must be shift_id or interval_idx, got {axis}")
    counts = (
        winners.groupby([axis, "winner"]).size()
        .unstack("winner", fill_value=0)
        .reindex(columns=HEURISTIC_NAMES, fill_value=0)
    )
    row_sums = counts.sum(axis=1).replace(0, 1)
    return counts.div(row_sums, axis=0)


def _render_heatmap(mat: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.18 * len(mat))))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(v) for v in mat.index])
    ax.set_xlabel("Heuristic")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Win rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-parquet", type=Path, default=DEFAULT_PILOT_PARQUET)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.pilot_parquet.exists():
        raise SystemExit(f"missing {args.pilot_parquet}; run `make pilot` first.")

    df = pd.read_parquet(args.pilot_parquet)
    print(f"[E1] loaded {len(df)} cost records over "
          f"{df.shift_id.nunique()} shifts × {df.interval_idx.nunique()} intervals.")

    winners = compute_winners(df, seed=args.seed)
    overall = (
        winners["winner"].value_counts(normalize=True)
        .reindex(HEURISTIC_NAMES, fill_value=0.0)
    )
    print("\n[E1] overall win rate:")
    for h in HEURISTIC_NAMES:
        print(f"  {h:5s}  {overall[h]*100:6.2f}%")

    interval_mat = heatmap_by_axis(winners, "interval_idx")
    shift_mat = heatmap_by_axis(winners, "shift_id")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    _render_heatmap(
        interval_mat,
        title="E1 — Heuristic win rate by decision interval",
        ylabel="Interval index (0–31)",
        out_path=args.fig_dir / "diversity_heatmap_interval.png",
    )
    _render_heatmap(
        shift_mat,
        title="E1 — Heuristic win rate by shift",
        ylabel="Shift index",
        out_path=args.fig_dir / "diversity_heatmap_shift.png",
    )

    # Save the long-form table for §5.1 of the paper.
    overall_df = pd.DataFrame({"heuristic": HEURISTIC_NAMES,
                               "win_rate": overall.values})
    overall_df.to_parquet(args.results_dir / "overall_win_rates.parquet", index=False)
    interval_mat.reset_index().to_parquet(
        args.results_dir / "win_rate_by_interval.parquet", index=False
    )
    shift_mat.reset_index().to_parquet(
        args.results_dir / "win_rate_by_shift.parquet", index=False
    )
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT)) + "/"
        except ValueError:
            return str(p) + "/"
    print(f"\n[E1] figures in {_rel(args.fig_dir)}")
    print(f"[E1] tables  in {_rel(args.results_dir)}")

    # Stress check: master plan's pilot gate is per-heuristic 8-60%.
    min_w, max_w = float(overall.min()), float(overall.max())
    print(f"\n[E1] heuristic-diversity range: [{min_w:.3f}, {max_w:.3f}]  "
          f"(pilot gate: [0.08, 0.60])")
    if not (min_w >= 0.08 and max_w <= 0.60):
        print("  NOTE: diversity range outside pilot gate (informational, not a failure).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
