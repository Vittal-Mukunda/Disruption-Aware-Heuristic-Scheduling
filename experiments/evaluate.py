"""Phase 5 — per-shift policy evaluation harness.

Every Phase 5 baseline plugs into this single function:

    df = evaluate_policy(name, policy_fn, test_seeds, cfg, results_dir=...)

`policy_fn` is a callable `(env_state: np.ndarray) -> heuristic_name: str` invoked
once per decision interval. If `policy_fn` has a `reset()` method it is called
before every shift (the switching controller needs this to clear dwell state).

The harness produces one row per `test_seed` with the canonical Phase 5 schema
required by HANDOFF.md §3.3:

    shift_seed, sla_breach_rate, mean_tardiness, spoilage_rate, throughput,
    picker_utilization, unfinished, mean_cost, wall_clock_s

`mean_cost` is the per-interval composite cost: the same W_BREACH * breaches +
W_TARDY * sum_tardy + W_UNFINISHED * leftover_queue weighted sum used by the
Phase 3 rollout labeler (`labeling.snapshot_labeler`), evaluated against the
whole shift's completions and divided by `env.n_intervals` so it is on the same
scale as the labeling-time rollout cost.

CLI usage (smoke):

    python -m experiments.evaluate --method fifo --n-test 1
    python -m experiments.evaluate --method ours --run-dir runs/phase4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from labeling.snapshot_labeler import W_BREACH, W_TARDY, W_UNFINISHED
from seed import spawn_shift_seeds
from simulation.kpis import compute_kpis
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results"

KPI_COLUMNS: list[str] = [
    "shift_seed",
    "sla_breach_rate",
    "mean_tardiness",
    "spoilage_rate",
    "throughput",
    "picker_utilization",
    "unfinished",
    "mean_cost",
    "wall_clock_s",
]


class PolicyCallable(Protocol):
    def __call__(self, state: np.ndarray) -> str: ...


class EnvPolicyCallable(Protocol):
    """Policies that need the live `WarehouseEnv` (e.g. greedy 1-step lookahead)."""

    def __call__(self, env: WarehouseEnv) -> str: ...


def _shift_mean_cost(env: WarehouseEnv) -> float:
    """Whole-shift composite cost / n_intervals.

    Same weighted sum used by `labeling.snapshot_labeler.composite_cost`, but
    applied to the entire shift's completions (and the residual queue). Divided
    by n_intervals so it sits on the same per-interval scale as the labeling
    rollout cost.
    """
    breaches = sum(1 for o in env.completed if o.finish_time > o.sla_due)
    tardy_sum = sum(max(o.finish_time - o.sla_due, 0.0) for o in env.completed)
    total = W_BREACH * breaches + W_TARDY * tardy_sum + W_UNFINISHED * len(env.queue)
    return float(total / max(env.n_intervals, 1))


def run_shift(seed: int, cfg: DictConfig, policy_fn: PolicyCallable) -> dict[str, float]:
    """Drive one `WarehouseEnv(seed, cfg)` to completion under `policy_fn`."""
    if hasattr(policy_fn, "reset") and callable(getattr(policy_fn, "reset")):
        policy_fn.reset()  # type: ignore[attr-defined]
    env = WarehouseEnv(seed, cfg)
    wall_start = time.perf_counter()
    while env.interval_idx < env.n_intervals:
        state = env.current_state()
        h = policy_fn(state)
        env.step(h)
    wall = time.perf_counter() - wall_start
    kpis = compute_kpis(env.completed, env.queue, env.n_pickers, env.shift_minutes)
    row: dict[str, float] = {"shift_seed": int(seed)}
    row.update({k: float(v) for k, v in kpis.items()})
    row["mean_cost"] = _shift_mean_cost(env)
    row["wall_clock_s"] = float(wall)
    return row


def run_shift_env_aware(
    seed: int, cfg: DictConfig, policy_fn_env: EnvPolicyCallable
) -> dict[str, float]:
    """Like `run_shift`, but `policy_fn_env(env)` sees the live env each interval.

    Used by lookahead policies (e.g. `baselines.greedy_mpc`) that need to
    `copy.deepcopy(env)` and simulate candidate heuristics before deciding.
    """
    if hasattr(policy_fn_env, "reset") and callable(getattr(policy_fn_env, "reset")):
        policy_fn_env.reset()  # type: ignore[attr-defined]
    env = WarehouseEnv(seed, cfg)
    wall_start = time.perf_counter()
    while env.interval_idx < env.n_intervals:
        h = policy_fn_env(env)
        env.step(h)
    wall = time.perf_counter() - wall_start
    kpis = compute_kpis(env.completed, env.queue, env.n_pickers, env.shift_minutes)
    row: dict[str, float] = {"shift_seed": int(seed)}
    row.update({k: float(v) for k, v in kpis.items()})
    row["mean_cost"] = _shift_mean_cost(env)
    row["wall_clock_s"] = float(wall)
    return row


def evaluate_policy(
    name: str,
    policy_fn: PolicyCallable,
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None = None,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run `policy_fn` on each test shift and return the per-shift KPI table.

    Parameters
    ----------
    name
        Method name; used to derive the parquet filename when `save=True`.
    policy_fn
        Callable `state -> heuristic_name`. May expose `reset()` for stateful
        adaptive policies (e.g., SwitchingController dwell counter).
    test_seeds
        Sequence of shift seeds. Should be `spawn_shift_seeds(n=300, root=42)[250:]`
        for the canonical 50 Phase 5 test shifts.
    cfg
        Loaded OmegaConf config (the same one Phase 3/4 used).
    results_dir
        Directory under which `{name}.parquet` is written. Defaults to
        `<repo>/results/`. Ignored when `save=False`.
    save
        If True, also persist the DataFrame to `<results_dir>/<name>.parquet`.
    verbose
        Print a one-line progress update per shift.
    """
    rows: list[dict[str, float]] = []
    for i, seed in enumerate(test_seeds):
        row = run_shift(int(seed), cfg, policy_fn)
        rows.append(row)
        if verbose:
            print(
                f"  [{i + 1:>3d}/{len(test_seeds)}] seed={int(seed):<10d} "
                f"sla_breach={row['sla_breach_rate']:.3f} "
                f"mean_cost={row['mean_cost']:.3f} "
                f"wall={row['wall_clock_s']:.2f}s"
            )

    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in KPI_COLUMNS]
    df = df[KPI_COLUMNS + extra]

    if save:
        out_dir = Path(results_dir) if results_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        if verbose:
            print(f"  wrote {out_path.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    return df


def evaluate_policy_env_aware(
    name: str,
    policy_fn_env: EnvPolicyCallable,
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None = None,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Env-aware sibling of `evaluate_policy` (passes env, not state)."""
    rows: list[dict[str, float]] = []
    for i, seed in enumerate(test_seeds):
        row = run_shift_env_aware(int(seed), cfg, policy_fn_env)
        rows.append(row)
        if verbose:
            print(
                f"  [{i + 1:>3d}/{len(test_seeds)}] seed={int(seed):<10d} "
                f"sla_breach={row['sla_breach_rate']:.3f} "
                f"mean_cost={row['mean_cost']:.3f} "
                f"wall={row['wall_clock_s']:.2f}s"
            )

    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in KPI_COLUMNS]
    df = df[KPI_COLUMNS + extra]

    if save:
        out_dir = Path(results_dir) if results_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        if verbose:
            print(f"  wrote {out_path.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    return df


def canonical_test_seeds(cfg: DictConfig) -> list[int]:
    """The 50 test seeds Phase 3 used. Phase 5 must reuse them verbatim."""
    n_train = int(cfg.labeling.n_train_shifts)
    n_test = int(cfg.labeling.n_test_shifts)
    seeds = spawn_shift_seeds(n=n_train + n_test, root=int(cfg.labeling.seed_seq_root))
    return seeds[n_train:]


def _build_policy(
    method: str, run_dir: Path | None
) -> tuple[Callable, bool]:
    """Resolve a method name to (policy_callable, env_aware_flag).

    Knows the methods that ship a turn-key CLI entry. Other Phase 5 baselines
    wire themselves in by importing this module and calling `evaluate_policy`
    (or `evaluate_policy_env_aware`) directly.
    """
    method_l = method.lower()
    static_aliases = {"fifo": "FIFO", "fefo": "FEFO", "wspt": "WSPT", "atc": "ATC"}
    if method_l in static_aliases:
        from baselines.static import make_static_policy

        return make_static_policy(static_aliases[method_l]), False
    if method_l == "ours":
        from baselines.ours import load_ours

        if run_dir is None:
            run_dir = REPO_ROOT / "runs" / "phase4"
        return load_ours(run_dir), False
    if method_l in ("greedy_mpc", "greedy-mpc", "mpc"):
        from baselines.greedy_mpc import make_greedy_mpc_policy

        return make_greedy_mpc_policy(), True
    if method_l in ("snapshot_xgb", "snapshot-xgb"):
        from baselines.snapshot_xgb import load_snapshot_xgb

        if run_dir is None:
            run_dir = REPO_ROOT / "runs" / "phase4_tau1"
        return load_snapshot_xgb(run_dir), False
    if method_l == "linucb":
        from baselines.linucb import make_linucb_policy

        return make_linucb_policy(), True
    if method_l in ("ppo_fair", "ppo-fair"):
        from baselines.ppo_fair import load_ppo_fair

        if run_dir is None:
            run_dir = REPO_ROOT / "runs" / "ppo_fair"
        return load_ppo_fair(run_dir), False
    if method_l in ("ppo_full", "ppo-full"):
        from baselines.ppo_fair import load_ppo_fair

        if run_dir is None:
            run_dir = REPO_ROOT / "runs" / "ppo_full"
        return load_ppo_fair(run_dir), False
    if method_l in ("offline_fqi", "offline-fqi"):
        from baselines.offline_fqi import load_offline_fqi

        if run_dir is None:
            run_dir = REPO_ROOT / "runs" / "offline_fqi"
        return load_offline_fqi(run_dir), False
    raise SystemExit(
        f"Unknown method '{method}'. Supported: fifo/fefo/wspt/atc/ours/"
        f"greedy_mpc/snapshot_xgb/linucb/ppo_fair/ppo_full/offline_fqi."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        help="One of fifo, fefo, wspt, atc, ours.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Override Phase 4 artifact dir for --method ours.")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Optional cap on the number of test shifts.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override the default <repo>/results/ output dir.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip the parquet write (smoke testing).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print one progress line per shift.")
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    print(f"[Phase 5] method={args.method}  n_test={len(seeds)}")
    policy, env_aware = _build_policy(args.method, args.run_dir)
    out_name = args.method.lower()
    eval_fn = evaluate_policy_env_aware if env_aware else evaluate_policy
    df = eval_fn(
        out_name, policy, seeds, cfg,
        results_dir=args.results_dir, save=(not args.no_save),
        verbose=args.verbose,
    )
    summary = df[[c for c in KPI_COLUMNS if c != "shift_seed"]].mean(numeric_only=True)
    print(f"\n[Phase 5] {args.method} summary over {len(df)} shifts:")
    for k, v in summary.items():
        print(f"  mean {k:<22s} = {float(v):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
