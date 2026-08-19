"""Per-shift policy evaluation harness. Every method plugs in here.

    df = evaluate_policy(name, policy_fn, test_seeds, cfg)

`policy_fn` is `(observation) -> rule_name`, invoked once per decision epoch.
Env-aware policies (the rolling-horizon MPC, LinUCB) receive the live env
instead and go through `evaluate_policy_env_aware`. A `reset()` method, if
present, is called before every shift.

WHAT CHANGED IN REVISION
------------------------
1. `_shift_mean_cost` is gone. The objective now lives in one place,
   `simulation.cost`, and is reported by `simulation.kpis.compute_kpis`. The
   submitted code defined the cost here *and* in the labeller; they agreed by
   inspection rather than by construction.

2. The reported schema carries the full outcome partition — arrived, served,
   unserved, dropped — and `service_failure_rate` is primary. The submitted
   `sla_breach_rate` counted only completed orders, so orders abandoned in the
   queue escaped the metric entirely (Reviewer 2, 1). Both quantities are now
   reported, under names that say which is which.

3. Policies observe through `env.observe()`, which admits due arrivals and then
   returns the feature vector. In the submitted harness the loop called
   `env.current_state()` while admission happened inside `env.step()`, so what a
   policy could see depended on the call site.

4. Per-decision inference latency is recorded (Reviewer 3, 5). It is the other
   half of the amortisation argument: DAHS's claim is that it buys the
   lookahead's decision quality at a forward pass's latency, and that is now
   measured on both sides rather than asserted.
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
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from seed import shift_corpora
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results"

KPI_COLUMNS: list[str] = [
    "shift_seed",
    # --- primary ---
    "composite_cost",
    "service_failure_rate",
    # --- outcome partition ---
    "arrived",
    "throughput",
    "unserved",
    "dropped",
    # --- decomposition ---
    "sla_breach_rate_arrived",
    "sla_breach_rate_served",
    "spoilage_rate",
    "mean_tardiness",
    "mean_tardiness_served",
    "picker_utilization",
    "mean_interval_cost",
    # --- cost of running the controller ---
    "wall_clock_s",
    "decision_latency_ms_mean",
    "decision_latency_ms_p95",
]


class PolicyCallable(Protocol):
    def __call__(self, state: np.ndarray) -> str: ...


class EnvPolicyCallable(Protocol):
    """Policies needing the live env — e.g. the rolling-horizon MPC."""

    def __call__(self, env: WarehouseEnv) -> str: ...


def _finish_row(
    seed: int, env: WarehouseEnv, wall: float, latencies: list[float]
) -> dict[str, float]:
    row: dict[str, float] = {"shift_seed": int(seed)}
    row.update({k: float(v) for k, v in env.kpis().items()})
    row["wall_clock_s"] = float(wall)
    lat = np.asarray(latencies, dtype=np.float64) * 1e3
    row["decision_latency_ms_mean"] = float(lat.mean()) if lat.size else 0.0
    row["decision_latency_ms_p95"] = (
        float(np.percentile(lat, 95)) if lat.size else 0.0
    )
    return row


def run_shift(seed: int, cfg: DictConfig, policy_fn: PolicyCallable) -> dict[str, float]:
    """Drive one `WarehouseEnv(seed, cfg)` to completion under `policy_fn`."""
    if callable(getattr(policy_fn, "reset", None)):
        policy_fn.reset()  # type: ignore[attr-defined]
    env = WarehouseEnv(seed, cfg)
    latencies: list[float] = []
    wall_start = time.perf_counter()
    while env.interval_idx < env.n_intervals:
        obs = env.observe()
        t0 = time.perf_counter()
        h = policy_fn(obs)
        latencies.append(time.perf_counter() - t0)
        env.step(h)
    if bool(cfg.sim.get("terminal_admit", False)):
        env.admit_if_shift_complete()
    return _finish_row(seed, env, time.perf_counter() - wall_start, latencies)


def run_shift_env_aware(
    seed: int, cfg: DictConfig, policy_fn_env: EnvPolicyCallable
) -> dict[str, float]:
    """Like `run_shift`, but the policy sees the live env each epoch."""
    if callable(getattr(policy_fn_env, "reset", None)):
        policy_fn_env.reset()  # type: ignore[attr-defined]
    env = WarehouseEnv(seed, cfg)
    latencies: list[float] = []
    wall_start = time.perf_counter()
    while env.interval_idx < env.n_intervals:
        env.observe()
        t0 = time.perf_counter()
        h = policy_fn_env(env)
        latencies.append(time.perf_counter() - t0)
        env.step(h)
    if bool(cfg.sim.get("terminal_admit", False)):
        env.admit_if_shift_complete()
    return _finish_row(seed, env, time.perf_counter() - wall_start, latencies)


def _evaluate(
    name: str,
    runner,
    policy,
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None,
    save: bool,
    verbose: bool,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Evaluate `policy` on every test shift.

    SERIAL BY DEFAULT, AND THAT IS LOAD-BEARING. Most policies are Markov and
    shifts are independent, but LinUCB is an *online learner* whose weights
    persist across shifts by design — its `reset()` is a documented no-op and
    the 50 test shifts form one bandit trajectory of ~1600 interactions.
    Evaluating it in parallel would silently turn it into 50 independent
    cold-start bandits and understate the baseline, which is exactly the kind of
    under-powered comparison Reviewer 1 (6.b) objects to.

    So parallelism is opt-in per call AND per policy: a policy that carries
    state across shifts sets `parallel_safe = False` and is always run serially,
    whatever the caller asks for (LinUCB). DAHS and snapshot_xgb reset per
    shift and are parallel-safe; saturation traces must still be collected
    with n_jobs=1 so the in-process controller sees every epoch. This matters
    because the online lookahead baselines dominate the campaign — on measured
    throughput the misspecification sweep alone is ~14 hours serial and under
    an hour across 16 cores.
    """
    parallel_ok = bool(getattr(policy, "parallel_safe", True))
    if n_jobs != 1 and not parallel_ok:
        print(f"  [eval] {name}: policy retains state across shifts; "
              f"ignoring n_jobs={n_jobs} and running serially.")
        n_jobs = 1

    if n_jobs == 1:
        rows: list[dict[str, float]] = []
        for i, seed in enumerate(test_seeds):
            row = runner(int(seed), cfg, policy)
            rows.append(row)
            if verbose:
                print(
                    f"  [{i + 1:>3d}/{len(test_seeds)}] seed={int(seed):<10d} "
                    f"cost={row['composite_cost']:8.2f} "
                    f"fail={row['service_failure_rate']:.4f} "
                    f"served={row['throughput']:.0f} unserved={row['unserved']:.0f} "
                    f"wall={row['wall_clock_s']:.2f}s"
                )
    else:
        rows = list(
            Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
                delayed(runner)(int(seed), cfg, policy) for seed in test_seeds
            )
        )

    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in KPI_COLUMNS]
    df = df[[c for c in KPI_COLUMNS if c in df.columns] + extra]

    if save:
        out_dir = Path(results_dir) if results_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        if verbose:
            print(f"  wrote {out_path} ({len(df)} rows)")
    return df


def evaluate_policy(
    name: str,
    policy_fn: PolicyCallable,
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None = None,
    save: bool = True,
    verbose: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run `policy_fn` on each test shift; return the per-shift KPI table."""
    return _evaluate(
        name, run_shift, policy_fn, test_seeds, cfg, results_dir, save, verbose,
        n_jobs=n_jobs,
    )


def evaluate_policy_env_aware(
    name: str,
    policy_fn_env: EnvPolicyCallable,
    test_seeds: list[int],
    cfg: DictConfig,
    results_dir: Path | None = None,
    save: bool = True,
    verbose: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Env-aware sibling of `evaluate_policy`."""
    return _evaluate(
        name, run_shift_env_aware, policy_fn_env, test_seeds, cfg,
        results_dir, save, verbose, n_jobs=n_jobs,
    )


def canonical_test_seeds(cfg: DictConfig) -> list[int]:
    """The held-out test block. Disjoint from train and from calibration."""
    return shift_corpora(cfg)["test"]


def _build_policy(method: str, run_dir: Path | None) -> tuple[Callable, bool]:
    """Resolve a method name to `(policy, env_aware)`."""
    m = method.lower()

    from simulation.heuristics import HEURISTICS

    static = {h.lower(): h for h in HEURISTICS}
    if m in static:
        from baselines.static import make_static_policy

        return make_static_policy(static[m]), False

    if m == "ours":
        from baselines.ours import load_ours

        return load_ours(run_dir or REPO_ROOT / "runs" / "phase4"), False

    if m in ("rolling_mpc", "rolling-mpc", "mpc_tau", "teacher"):
        # The distillation teacher (Reviewer 2, 6).
        from baselines.rolling_horizon_mpc import make_rolling_horizon_mpc

        return make_rolling_horizon_mpc(), True

    if m in ("greedy_mpc", "greedy-mpc", "mpc"):
        from baselines.rolling_horizon_mpc import make_greedy_mpc_policy

        return make_greedy_mpc_policy(), True

    if m in ("snapshot_xgb", "snapshot-xgb"):
        from baselines.snapshot_xgb import load_snapshot_xgb

        return load_snapshot_xgb(run_dir or REPO_ROOT / "runs" / "phase4_tau1"), False

    if m == "linucb":
        from baselines.linucb import make_linucb_policy

        return make_linucb_policy(), True

    if m in ("ppo_fair", "ppo-fair", "ppo_full", "ppo-full"):
        from baselines.ppo_fair import load_ppo_fair

        default = "ppo_full" if "full" in m else "ppo_fair"
        return load_ppo_fair(run_dir or REPO_ROOT / "runs" / default), False

    if m in ("offline_fqi", "offline-fqi"):
        from baselines.offline_fqi import load_offline_fqi

        return load_offline_fqi(run_dir or REPO_ROOT / "runs" / "offline_fqi"), False

    raise SystemExit(
        f"Unknown method '{method}'. Static rules: {sorted(static)}. "
        f"Learned/analytic: ours, rolling_mpc, greedy_mpc, snapshot_xgb, "
        f"linucb, ppo_fair, ppo_full, offline_fqi."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # The plumbing for this already existed in `_evaluate`; the CLI never exposed
    # it, so `--method rolling_mpc` ran single-threaded. That is the single most
    # expensive evaluation in the campaign — the tau-step teacher costs
    # |H| * M * tau = 480 simulated interval-steps per decision, measured at
    # ~32 s per shift, so 50 shifts is ~26 minutes serial on a 16-thread machine.
    # LinUCB keeps weights across the 50-shift trajectory (`parallel_safe=False`)
    # and is forced serial. DAHS / snapshot_xgb reset per shift and honour this.
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Shift-level parallelism. -1 uses every core. "
                             "Ignored for policies that are not parallel-safe.")
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    print(f"[eval] method={args.method}  n_test={len(seeds)}")
    policy, env_aware = _build_policy(args.method, args.run_dir)
    runner = evaluate_policy_env_aware if env_aware else evaluate_policy
    df = runner(
        args.method.lower(), policy, seeds, cfg,
        results_dir=args.results_dir, save=(not args.no_save), verbose=args.verbose,
        n_jobs=args.n_jobs,
    )

    print(f"\n[eval] {args.method} over {len(df)} shifts:")
    for k in KPI_COLUMNS:
        if k == "shift_seed" or k not in df.columns:
            continue
        print(f"  mean {k:<26s} = {float(df[k].mean()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
