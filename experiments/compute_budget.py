"""Offline and online computational cost, and how both scale with the rule pool.

REVIEWER 3, COMMENTS 1 AND 2
---------------------------
    1. "While the paper correctly argues that online rollouts are too slow and
        DRL is too sample-inefficient, it does not quantify the offline
        computational cost. For the reported setup (250 shifts x 32 intervals x
        4 rules x tau=4), what is the total number of simulated steps? How long
        does labeling take on standard hardware? This is crucial for assessing
        scalability, especially if the heuristic pool were to grow."

    2. "If the pool were expanded to 10 or 20 heuristics, the offline rollout
        cost would grow linearly. Please discuss: the impact of pool size on
        offline training time and sample efficiency; whether sub-sampling or
        hierarchical selection could mitigate this cost."

The paper's operational argument is that an expensive lookahead is paid once and
amortised. That is a claim about two numbers and the submitted version reported
neither. This module produces both, plus the scaling analysis.

    python -m experiments.compute_budget analytic     # closed-form budgets
    python -m experiments.compute_budget measure      # steps/sec on this machine
    python -m experiments.compute_budget scaling      # cost and mitigation vs |H|

WHY THE SUBMITTED SCHEME WAS QUADRATIC
--------------------------------------
The submitted labeller reconstructed each decision state by replaying the shift
from t=0, once per rule:

    sum_{t<N} |H| * (t + tau)  =  |H| * ( N(N-1)/2 + N*tau )   steps per shift

For the reviewer's stated setup (N=32, |H|=4, tau=4) that is 2,496 steps per shift
and 624,000 over 250 shifts. The replay term dominates and buys nothing: it
re-derives a state the walk already passed through.

Walking each shift forward once and branching at each epoch costs

    N  +  N * |H| * M * tau                                     steps per shift

which is linear in N. The N^2 term the new scheme does not pay is what funds the
M independent continuations that Reviewer 2 (3) requires. Both formulas are
implemented below so the comparison is computed, not asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from labeling.rollout_labeler import (
    costs_at_epoch,
    costs_at_epoch_successive_halving,
    rollout_step_budget,
)
from seed import shift_corpora
from simulation.heuristics import SCREENING_POOL, resolve_pool, with_default_scales
from simulation.warehouse_env import (
    WarehouseEnv,
    reset_step_counter,
    simulated_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E12_compute"


def legacy_budget(n_shifts: int, n_intervals: int, n_rules: int, tau: int) -> int:
    """Steps for the submitted replay-from-zero labeller. Quadratic in N."""
    per_shift = n_rules * (n_intervals * (n_intervals - 1) // 2 + n_intervals * tau)
    return int(n_shifts * per_shift)


def analytic_table(cfg) -> pd.DataFrame:
    """Budgets for the submitted setup and for this revision's setup."""
    N = int(round(cfg.sim.shift_hours * 60 / cfg.sim.interval_minutes))
    rows = [
        {
            "scheme": "submitted (replay-from-zero, M=1)",
            "n_shifts": 250, "n_rules": 4, "tau": 4, "M": 1,
            "steps": legacy_budget(250, N, 4, 4),
        },
        {
            "scheme": "revision (forward walk + branch)",
            "n_shifts": int(cfg.shifts.n_train),
            "n_rules": len(resolve_pool(cfg)),
            "tau": int(cfg.labeling.tau),
            "M": int(cfg.labeling.n_rollout_samples),
            "steps": rollout_step_budget(
                int(cfg.shifts.n_train), N, len(resolve_pool(cfg)),
                int(cfg.labeling.tau), int(cfg.labeling.n_rollout_samples),
            ),
        },
        {
            "scheme": "revision, Stage-1 screening",
            "n_shifts": int(cfg.shifts.n_calib),
            "n_rules": len(SCREENING_POOL),
            "tau": int(cfg.labeling.tau), "M": 10,
            "steps": rollout_step_budget(
                int(cfg.shifts.n_calib), N, len(SCREENING_POOL),
                int(cfg.labeling.tau), 10,
            ),
        },
    ]
    df = pd.DataFrame(rows)
    df["samples_per_cell"] = df["M"]
    df["steps_per_sample"] = (df["steps"] / (df["M"] * df["n_rules"])).round(0)
    return df


def cmd_analytic(args: argparse.Namespace) -> int:
    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = analytic_table(cfg)
    df.to_parquet(RESULTS_DIR / "analytic_budget.parquet", index=False)
    print(df.to_string(index=False))
    base, rev = int(df.loc[0, "steps"]), int(df.loc[1, "steps"])
    print(f"\n  submitted total : {base:,} interval-steps at M=1")
    print(f"  revision total  : {rev:,} interval-steps at "
          f"M={int(df.loc[1, 'M'])} ({rev / base:.2f}x) ")
    print("  The revision buys "
          f"{int(df.loc[1, 'M'])}x the samples per cell for "
          f"{rev / base:.2f}x the compute, because the O(N^2) replay term is gone.")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Measure interval-steps per second on this machine, then extrapolate."""
    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    pool = resolve_pool(cfg)
    seeds = shift_corpora(cfg)["calib"][: args.n_shifts]
    tau = int(cfg.labeling.tau)
    M = int(cfg.labeling.n_rollout_samples)
    base_seed = int(cfg.seeds.rollout)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reset_step_counter()
    t0 = time.perf_counter()
    for sid, seed in enumerate(seeds):
        env = WarehouseEnv(int(seed), cfg)
        for t in range(env.n_intervals):
            env.observe()
            costs_at_epoch(env, int(seed), t, pool, tau, M, base_seed)
            env.step(pool[t % len(pool)])
    wall = time.perf_counter() - t0
    steps = simulated_steps()
    rate = steps / max(wall, 1e-9)

    N = int(round(cfg.sim.shift_hours * 60 / cfg.sim.interval_minutes))
    full = rollout_step_budget(int(cfg.shifts.n_train), N, len(pool), tau, M)

    report = {
        "machine_note": args.machine or "unspecified — record CPU model and core count",
        "n_shifts_measured": len(seeds),
        "measured_steps": int(steps),
        "wall_clock_s": float(wall),
        "interval_steps_per_second_single_core": float(rate),
        "seconds_per_shift": float(wall / max(len(seeds), 1)),
        "full_corpus_steps": int(full),
        "full_corpus_hours_single_core": float(full / rate / 3600.0),
        "note": (
            "Labelling is embarrassingly parallel over shifts; divide the "
            "single-core estimate by the usable core count. The measurement "
            "itself is single-threaded so the per-core rate is not confounded "
            "by scheduler effects."
        ),
    }
    (RESULTS_DIR / "measured_throughput.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[compute] measured {steps:,} steps in {wall:.1f}s "
          f"({rate:,.0f} steps/s, single core)")
    print(f"[compute] full corpus {full:,} steps "
          f"-> {report['full_corpus_hours_single_core']:.2f} core-hours")
    return 0


def cmd_scaling(args: argparse.Namespace) -> int:
    """Cost and label quality against pool size, with the two mitigations."""
    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    N = int(round(cfg.sim.shift_hours * 60 / cfg.sim.interval_minutes))
    tau = int(cfg.labeling.tau)
    M = int(cfg.labeling.n_rollout_samples)
    base_seed = int(cfg.seeds.rollout)
    seeds = shift_corpora(cfg)["calib"][: args.n_shifts]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- analytic scaling ---
    rows = []
    for h in (2, 4, 8, 10, 16, 20):
        rows.append({
            "n_rules": h,
            "steps_uniform": rollout_step_budget(int(cfg.shifts.n_train), N, h, tau, M),
            # Successive halving spends the same total budget but concentrates it;
            # the saving shows up as fewer samples needed for the same label
            # quality, measured empirically below.
            "classes_to_learn": h,
            "min_states_rule_of_thumb": 50 * h,
        })
    scaling = pd.DataFrame(rows)
    scaling["steps_relative_to_pool_4"] = (
        scaling["steps_uniform"] / scaling.loc[scaling["n_rules"] == 4, "steps_uniform"].iloc[0]
    ).round(2)
    scaling.to_parquet(RESULTS_DIR / "pool_scaling_analytic.parquet", index=False)
    print("Analytic scaling in pool size (labelling steps are linear in |H|):")
    print(scaling.to_string(index=False))

    # --- empirical: does successive halving preserve the label? ---
    pool = list(SCREENING_POOL)
    agree, sh_steps, uni_steps, kl = [], 0, 0, []
    for seed in seeds:
        env = WarehouseEnv(int(seed), cfg)
        for t in range(env.n_intervals):
            env.observe()
            uni = costs_at_epoch(env, int(seed), t, pool, tau, M, base_seed)
            sh, used = costs_at_epoch_successive_halving(
                env, int(seed), t, pool, tau, M, base_seed, eta=args.eta
            )
            uni_steps += len(pool) * M * min(tau, env.n_intervals - t)
            sh_steps += used
            agree.append(uni.best() == sh.best())
            # Label-level agreement matters more than arg-max agreement, since
            # the ranker is fitted to the distribution.
            a = np.array([uni.mean[h] for h in pool])
            b = np.array([sh.mean[h] for h in pool])
            pa = np.exp(-(a - a.min())); pa /= pa.sum()
            pb = np.exp(-(b - b.min())); pb /= pb.sum()
            kl.append(float(np.sum(pa * np.log((pa + 1e-12) / (pb + 1e-12)))))
            env.step(pool[t % len(pool)])

    result = {
        "eta": int(args.eta),
        "pool_size": len(pool),
        "argmax_agreement_rate": float(np.mean(agree)),
        "mean_label_kl": float(np.mean(kl)),
        "steps_uniform": int(uni_steps),
        "steps_successive_halving": int(sh_steps),
        "step_saving": float(1.0 - sh_steps / max(uni_steps, 1)),
        "verdict": (
            "Successive halving preserves the label at materially lower cost; "
            "it is the recommended allocation for pools beyond ~8 rules."
            if float(np.mean(agree)) > 0.95 and sh_steps < uni_steps
            else "Successive halving degrades the label at this budget; report "
            "uniform allocation and note the mitigation as unsuccessful here."
        ),
    }
    (RESULTS_DIR / "successive_halving.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nSuccessive halving (eta={args.eta}) over {len(pool)} rules, "
          f"{len(seeds)} shifts:")
    print(f"  arg-max agreement with uniform : {result['argmax_agreement_rate']:.1%}")
    print(f"  mean KL between labels         : {result['mean_label_kl']:.4f}")
    print(f"  step saving                    : {result['step_saving']:.1%}")
    print(f"  {result['verdict']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pa = sub.add_parser("analytic", help="Closed-form step budgets.")
    pa.set_defaults(func=cmd_analytic)

    pm = sub.add_parser("measure", help="Measure throughput and extrapolate.")
    pm.add_argument("--n-shifts", type=int, default=3)
    pm.add_argument("--machine", type=str, default=None,
                    help="CPU model and core count, recorded in the report.")
    pm.set_defaults(func=cmd_measure)

    ps = sub.add_parser("scaling", help="Cost and label quality vs pool size.")
    ps.add_argument("--n-shifts", type=int, default=5)
    ps.add_argument("--eta", type=int, default=2)
    ps.set_defaults(func=cmd_scaling)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
