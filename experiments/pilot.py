"""Phase 2 — Pilot study and heuristic-diversity gate.

For each of 50 pilot shifts and every 15-minute interval boundary, take a
state snapshot under a round-robin observed policy, then for each candidate
heuristic h replay the warehouse for tau intervals using h and record the
composite cost J_h(s_t). The snapshot 'winner' is argmin_h J_h(s_t).

Gate (cfg.pilot.acceptance):
    - min(win_rate_per_heuristic) >= 0.08
    - max(win_rate_per_heuristic) <= 0.60

Usage:
    python -m experiments.pilot
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from seed import spawn_shift_seeds
from simulation.heuristics import HEURISTIC_NAMES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
OUTPUT_PARQUET = REPO_ROOT / "data" / "pilot_costs.parquet"

W_BREACH = 3.0
W_TARDY = 0.2
W_UNFINISHED = 0.005
TIE_JITTER = 1e-9


def composite_cost(env: WarehouseEnv, n_completed_before_rollout: int) -> float:
    """J_h(s_t) over the rollout window only.

    Counts orders that finished during the rollout (so the shared pre-rollout
    history cancels across heuristics) plus the residual queue length at the
    end of the rollout (which is shaped entirely by the heuristic from the
    shared snapshot).
    """
    new_completed = env.completed[n_completed_before_rollout:]
    new_breaches = sum(1 for o in new_completed if o.finish_time > o.sla_due)
    new_tardy_sum = sum(max(o.finish_time - o.sla_due, 0.0) for o in new_completed)
    queue_at_end = len(env.queue)
    return (
        W_BREACH * new_breaches
        + W_TARDY * new_tardy_sum
        + W_UNFINISHED * queue_at_end
    )


def run_pilot(cfg: DictConfig) -> pd.DataFrame:
    n_shifts = int(cfg.pilot.n_shifts)
    tau = int(cfg.labeling.tau)
    shift_seeds = spawn_shift_seeds(n=n_shifts, root=int(cfg.seeds.sim))

    records: list[dict] = []

    for shift_id, shift_seed in enumerate(shift_seeds):
        env = WarehouseEnv(shift_seed, cfg)
        n_intervals = env.n_intervals
        observed_policy = [
            HEURISTIC_NAMES[i % len(HEURISTIC_NAMES)] for i in range(n_intervals)
        ]

        for t in range(n_intervals):
            rollout_steps = min(tau, n_intervals - t)
            for h in HEURISTIC_NAMES:
                env.fast_forward(t_intervals=t, policy_history=observed_policy[:t])
                n_completed_before = len(env.completed)
                env.run_with_policy(h, n_steps=rollout_steps)
                records.append(
                    {
                        "seed": int(shift_seed),
                        "shift_id": int(shift_id),
                        "interval_idx": int(t),
                        "heuristic": h,
                        "cost": float(composite_cost(env, n_completed_before)),
                    }
                )

        print(
            f"  shift {shift_id + 1:>2}/{n_shifts} (seed={shift_seed}) "
            f"-> {len(records)} cost records so far",
            flush=True,
        )

    return pd.DataFrame.from_records(records)


def compute_win_rates(
    df: pd.DataFrame, tiebreak_seed: int
) -> tuple[pd.Series, pd.Series]:
    """Per-snapshot winners and per-heuristic win rates.

    Ties are broken by adding a deterministic micro-jitter so we don't bias
    toward the first column in pivot order.
    """
    rng = np.random.default_rng(tiebreak_seed)
    df_j = df.copy()
    df_j["cost_jittered"] = df_j["cost"].to_numpy() + TIE_JITTER * rng.uniform(
        size=len(df_j)
    )
    pivot = df_j.pivot_table(
        index=["shift_id", "interval_idx"],
        columns="heuristic",
        values="cost_jittered",
    )
    winners = pivot.idxmin(axis=1)
    win_rates = (
        winners.value_counts(normalize=True)
        .reindex(HEURISTIC_NAMES, fill_value=0.0)
        .astype(float)
    )
    return winners, win_rates


def main() -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    print(f"Phase 2 pilot: {cfg.pilot.n_shifts} shifts, tau={cfg.labeling.tau}")
    print(
        f"Cost weights: breach={W_BREACH}, tardiness={W_TARDY}, unfinished={W_UNFINISHED}"
    )
    print(f"Sim config: arrivals.base_rate_per_minute={cfg.sim.arrivals.base_rate_per_minute}, "
          f"sla_due_triangular={list(cfg.sim.order_attrs.sla_due_triangular)}, "
          f"n_pickers={cfg.sim.n_pickers}")

    df = run_pilot(cfg)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved {len(df)} cost records -> {OUTPUT_PARQUET.relative_to(REPO_ROOT)}")

    winners, win_rates = compute_win_rates(df, tiebreak_seed=int(cfg.seeds.sim))
    n_snapshots = len(winners)

    print("\nWin-rate matrix:")
    for h in HEURISTIC_NAMES:
        rate = float(win_rates[h])
        wins = int(round(rate * n_snapshots))
        print(f"  {h:5s}  {rate * 100:6.2f}%  ({wins:>5}/{n_snapshots})")

    print("\nMean cost per heuristic:")
    for h in HEURISTIC_NAMES:
        sub = df[df["heuristic"] == h]["cost"]
        print(f"  {h:5s}  mean={sub.mean():8.4f}  std={sub.std():8.4f}  "
              f"min={sub.min():8.4f}  max={sub.max():8.4f}")

    min_rate = float(win_rates.min())
    max_rate = float(win_rates.max())
    min_required = float(cfg.pilot.acceptance.min_win_rate)
    max_allowed = float(cfg.pilot.acceptance.max_top_win_rate)
    print(
        f"\nmin win rate = {min_rate:.4f}  (require >= {min_required})\n"
        f"max win rate = {max_rate:.4f}  (require <= {max_allowed})"
    )

    passed = (min_rate >= min_required) and (max_rate <= max_allowed)
    verdict = "PASS" if passed else "FAIL"
    print(f"\nGATE: {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
