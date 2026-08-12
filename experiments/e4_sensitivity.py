"""Phase 6 / E4 — one-at-a-time sensitivity sweeps.

Four knobs per HANDOFF §3.2:

  - t_min:         dwell length in {0, 1, 2, 3, 4}. **Eval-only**, swaps the
                   switching controller hyperparameter.
  - arrival_noise: arrival-rate multiplier in {0.5, 1.0, 1.5, 2.0}. **Eval-only**;
                   scales `cfg.sim.arrivals.base_rate_per_minute`.
  - tau:           rollout horizon in {1, 2, 3, 4}. τ=1 lives at `runs/phase4_tau1/`
                   and τ=4 at `runs/phase4/`; τ=2 and τ=3 need **re-labeling +
                   retraining** (~hours each). Wired as a CLI; runs on request.
  - theta:         ambiguity-filter threshold in {0.40, 0.50, 0.55, 0.60, 0.70}.
                   Needs **re-labeling** (Phase 3) + retraining. Wired only.

Each eval-only sweep writes per-value parquets under
`results/E4/<knob>/<value>.parquet`, plus a roll-up `results/E4/<knob>_summary.parquet`
with mean and bootstrap CI per value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.evaluate import (  # noqa: E402
    canonical_test_seeds,
    evaluate_policy,
)
from experiments.stats import bootstrap_mean_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4"
RESULTS_DIR = REPO_ROOT / "results" / "E4"
FIG_DIR = REPO_ROOT / "figures" / "E4"


def _load_ours_with_overrides(
    run_dir: Path,
    cfg: DictConfig,
    t_min: int | None = None,
    entropy_gate_ratio: float | None = None,
):
    new_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if t_min is not None:
        new_cfg.ranker.switching.t_min_intervals = int(t_min)
    if entropy_gate_ratio is not None:
        new_cfg.ranker.switching.entropy_gate_ratio = float(entropy_gate_ratio)
    from baselines.ours import load_ours
    return load_ours(run_dir, cfg=new_cfg)


def _apply_arrival_noise(cfg: DictConfig, multiplier: float) -> DictConfig:
    new_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    base = float(cfg.sim.arrivals.base_rate_per_minute)
    new_cfg.sim.arrivals.base_rate_per_minute = base * float(multiplier)
    return new_cfg


def _bootstrap_summary(df: pd.DataFrame, metric: str, value, label: str) -> dict:
    ci = bootstrap_mean_ci(
        df[metric].to_numpy(dtype=np.float64), n_resamples=10000, seed=1337,
    )
    return {
        "knob": label,
        "value": value,
        "metric": metric,
        **ci.as_row(),
    }


def _plot_curve(df_summary: pd.DataFrame, knob: str, metric: str) -> None:
    sub = df_summary[df_summary["metric"] == metric].sort_values("value")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(
        sub["value"], sub["point"],
        yerr=[sub["point"] - sub["ci_lo"], sub["ci_hi"] - sub["point"]],
        fmt="o-", color="firebrick", lw=2, capsize=4,
    )
    ax.set_xlabel(knob)
    ax.set_ylabel(metric)
    ax.set_title(f"E4 — {metric} vs {knob}")
    ax.grid(True, alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{knob}_{metric}.png", dpi=150)
    fig.savefig(FIG_DIR / f"{knob}_{metric}.pdf")
    plt.close(fig)


def cmd_t_min(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    values = args.values if args.values else list(cfg.experiments.e4_sensitivity.t_min)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]
    out_dir = RESULTS_DIR / "t_min"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for v in values:
        policy = _load_ours_with_overrides(
            args.run_dir or DEFAULT_RUN_DIR, cfg, t_min=int(v)
        )
        print(f"[E4 t_min] T_min={v}  on {len(seeds)} shifts")
        df = evaluate_policy(
            f"t_min_{v}", policy, seeds, cfg,
            results_dir=out_dir, save=True, verbose=False,
        )
        for metric in ("service_failure_rate", "mean_tardiness", "composite_cost",
                       "throughput", "picker_utilization"):
            summary_rows.append(_bootstrap_summary(df, metric, int(v), "t_min"))
        sla = df["service_failure_rate"].mean()
        cost = df["composite_cost"].mean()
        print(f"  sla_breach={sla:.4f}  composite_cost={cost:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(RESULTS_DIR / "t_min_summary.parquet", index=False)
    for metric in ("service_failure_rate", "composite_cost"):
        _plot_curve(summary, "t_min", metric)
    print(f"\n[E4 t_min] summary -> "
          f"{(RESULTS_DIR / 't_min_summary.parquet').relative_to(REPO_ROOT)}")
    return 0


def cmd_arrival_noise(args: argparse.Namespace) -> int:
    base_cfg = OmegaConf.load(CONFIG_PATH)
    values = (args.values if args.values
              else list(base_cfg.experiments.e4_sensitivity.arrival_noise))
    seeds = canonical_test_seeds(base_cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]
    out_dir = RESULTS_DIR / "arrival_noise"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for mult in values:
        cfg = _apply_arrival_noise(base_cfg, float(mult))
        from baselines.ours import load_ours
        policy = load_ours(args.run_dir or DEFAULT_RUN_DIR, cfg=cfg)
        print(f"[E4 arrival_noise] x={mult}  "
              f"(rate={cfg.sim.arrivals.base_rate_per_minute:.3f})")
        df = evaluate_policy(
            f"arrival_x{mult}", policy, seeds, cfg,
            results_dir=out_dir, save=True, verbose=False,
        )
        for metric in ("service_failure_rate", "mean_tardiness", "composite_cost",
                       "throughput", "picker_utilization"):
            summary_rows.append(_bootstrap_summary(df, metric, float(mult),
                                                   "arrival_noise"))
        sla = df["service_failure_rate"].mean()
        cost = df["composite_cost"].mean()
        print(f"  sla_breach={sla:.4f}  composite_cost={cost:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(RESULTS_DIR / "arrival_noise_summary.parquet", index=False)
    for metric in ("service_failure_rate", "composite_cost"):
        _plot_curve(summary, "arrival_noise", metric)
    print(f"\n[E4 arrival_noise] summary -> "
          f"{(RESULTS_DIR / 'arrival_noise_summary.parquet').relative_to(REPO_ROOT)}")
    return 0


def cmd_tau(args: argparse.Namespace) -> int:
    """τ ∈ {1,2,3,4}. τ=1 and τ=4 already trained; τ=2/τ=3 need re-labeling + retrain.

    With `--dry-run`, just reports which runs exist and which need work.
    """
    cfg = OmegaConf.load(CONFIG_PATH)
    values = args.values if args.values else list(cfg.experiments.e4_sensitivity.tau)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]
    out_dir = RESULTS_DIR / "tau"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_built = {
        1: REPO_ROOT / "runs" / "phase4_tau1",
        2: REPO_ROOT / "runs" / "phase4_tau2",
        3: REPO_ROOT / "runs" / "phase4_tau3",
        4: REPO_ROOT / "runs" / "phase4",
    }
    summary_rows: list[dict] = []
    for tau in values:
        run_dir = pre_built.get(int(tau))
        if run_dir is None or not run_dir.exists():
            print(f"[E4 tau] tau={tau}: no pretrained ranker at "
                  f"{REPO_ROOT / 'runs' / f'phase4_tau{tau}'} — needs re-label + retrain.")
            print("  Skipping (use experiments.generate_labels --tau <v> then "
                  "experiments.train_ranker --run-id phase4_tau<v>).")
            continue
        from baselines.ours import load_ours
        policy = load_ours(run_dir, cfg=cfg)
        print(f"[E4 tau] tau={tau}  run_dir={run_dir.relative_to(REPO_ROOT)}")
        df = evaluate_policy(
            f"tau_{tau}", policy, seeds, cfg,
            results_dir=out_dir, save=True, verbose=False,
        )
        for metric in ("service_failure_rate", "mean_tardiness", "composite_cost"):
            summary_rows.append(_bootstrap_summary(df, metric, int(tau), "tau"))
        sla = df["service_failure_rate"].mean()
        cost = df["composite_cost"].mean()
        print(f"  sla_breach={sla:.4f}  composite_cost={cost:.4f}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_parquet(RESULTS_DIR / "tau_summary.parquet", index=False)
        for metric in ("service_failure_rate", "composite_cost"):
            _plot_curve(summary, "tau", metric)
        print(f"\n[E4 tau] summary -> "
              f"{(RESULTS_DIR / 'tau_summary.parquet').relative_to(REPO_ROOT)}")
    return 0


def cmd_theta(args: argparse.Namespace) -> int:
    """θ sweep — re-labeling required. Wired only; prints the plan."""
    cfg = OmegaConf.load(CONFIG_PATH)
    values = args.values if args.values else list(cfg.experiments.e4_sensitivity.theta)
    print(f"[E4 theta] values={values}  (re-labeling pipeline)")
    print("  For each θ:")
    print("    1) python -m experiments.generate_labels "
          "--theta <v> --train-out data/e4_theta_<v>/train.parquet "
          "--test-out data/e4_theta_<v>/test.parquet")
    print("    2) python -m experiments.train_ranker "
          "--run-id e4_theta_<v> "
          "--train-path data/e4_theta_<v>/train.parquet "
          "--test-path data/e4_theta_<v>/test.parquet")
    print("    3) python -m experiments.evaluate --method ours "
          "--run-dir runs/e4_theta_<v> "
          "--results-dir results/E4/theta/theta_<v>")
    print("  This is multi-hour compute; run from a screen/nohup session.")
    return 0


WEIGHT_AXES: tuple[str, ...] = ("w_breach", "w_spoil", "w_tardy", "w_holding")


def cmd_weights(args: argparse.Namespace) -> int:
    """Objective-weight sensitivity — does the METHOD RANKING survive the weights?

    WHY (Reviewer 1, 6.c). Promoting the composite cost to primary metric makes
    its weights the load-bearing assumption behind every comparison in the paper,
    and `w_spoil = 5.0` in particular is a judgement introduced in this revision:
    it encodes "destroyed stock costs more than a late shipment". A reviewer who
    accepts that the objective is primary will immediately ask what happens when
    it changes. Declaring the weights "fixed before learning" answers the
    tuning-bias question, not this one.

    WHAT THIS IS AND IS NOT. Policies are NOT re-optimised per weight vector —
    that would mean re-labelling and retraining once per cell, which is the whole
    campaign several times over. Each cell re-runs evaluation with a different
    cost functional, so the quantity measured is the robustness of the RANKING to
    the decision-maker's weights, holding the controllers fixed at the ones
    trained under nominal weights. That is the conservative direction: DAHS is
    the method most specialised to the nominal weights, so any cell where it
    still wins is evidence that the ranking is not an artefact of the weighting,
    while a cell where it loses is a genuine caveat and must be reported as one.
    The paper must state at which ratio the conclusion flips, if it does.

    ONE ASYMMETRY, DELIBERATELY LEFT IN. The rolling-horizon MPC scores its
    rollouts through `env.potential()`, so it re-optimises against each cell's
    weights for free, while DAHS carries labels distilled under the nominal ones.
    The lookahead is therefore advantaged here — which is again the conservative
    direction, since DAHS is being asked to beat a teacher that has adapted to
    the new objective when it has not. Report it rather than leaving it implicit.
    """
    base_cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(base_cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]
    axes = args.axes if args.axes else list(WEIGHT_AXES)
    methods = list(args.methods)
    out_dir = RESULTS_DIR / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    from experiments.evaluate import _build_policy, evaluate_policy_env_aware

    rows: list[dict] = []
    for axis in axes:
        grid = list(base_cfg.experiments.e4_sensitivity[axis])
        nominal = float(base_cfg.objective[axis])
        print(f"\n[E4 weights] {axis}: {grid}  (nominal {nominal})")
        for v in grid:
            cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
            cfg.objective[axis] = float(v)
            for m in methods:
                policy, env_aware = _build_policy(m, args.run_dir)
                runner = evaluate_policy_env_aware if env_aware else evaluate_policy
                df = runner(
                    f"{axis}_{v}_{m}", policy, seeds, cfg,
                    results_dir=out_dir, save=True, verbose=False,
                    n_jobs=args.n_jobs,
                )
                rows.append({
                    "axis": axis,
                    "value": float(v),
                    "is_nominal": bool(float(v) == nominal),
                    "method": m,
                    "composite_cost": float(df["composite_cost"].mean()),
                    "service_failure_rate": float(df["service_failure_rate"].mean()),
                    "spoilage_rate": float(df["spoilage_rate"].mean()),
                })
                print(f"  {axis}={v:<8} {m:<14} "
                      f"cost={rows[-1]['composite_cost']:9.3f}  "
                      f"fail={rows[-1]['service_failure_rate']:.4f}")

    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS_DIR / "weights_summary.parquet", index=False)

    # The headline: is the arg-min method the same in every cell?
    print("\n[E4 weights] winner by cell (lowest composite cost):")
    winners: list[str] = []
    for (axis, value), grp in table.groupby(["axis", "value"], sort=True):
        w = grp.loc[grp["composite_cost"].idxmin(), "method"]
        winners.append(str(w))
        print(f"  {axis}={value:<8} -> {w}")
    unique = sorted(set(winners))
    if len(unique) == 1:
        print(f"\n[E4 weights] RANKING INVARIANT: {unique[0]} wins all "
              f"{len(winners)} cells. The conclusion does not depend on the "
              f"objective weights over the swept ranges.")
    else:
        print(f"\n[E4 weights] RANKING IS NOT INVARIANT — winners: {unique}. "
              f"Report the cells where the conclusion flips, and at which ratio. "
              f"Do NOT report only the favourable cells.")
    print(f"[E4 weights] summary -> "
          f"{(RESULTS_DIR / 'weights_summary.parquet').relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="knob", required=True)

    p_w = sub.add_parser("weights", help="Objective-weight sensitivity (R1, 6.c).")
    p_w.add_argument("--axes", nargs="*", default=None,
                     help=f"Subset of {list(WEIGHT_AXES)}.")
    p_w.add_argument("--methods", nargs="*",
                     default=["ours", "rolling_mpc", "edd", "fifo"])
    p_w.add_argument("--run-dir", type=Path, default=None)
    p_w.add_argument("--n-test", type=int, default=None)
    p_w.add_argument("--n-jobs", type=int, default=-1,
                     help="Shifts evaluated in parallel; the lookahead baseline "
                          "dominates this sweep (~8.5 h serial).")
    p_w.set_defaults(func=cmd_weights)

    p_t = sub.add_parser("t_min", help="Dwell length sweep.")
    p_t.add_argument("--values", type=int, nargs="*", default=None)
    p_t.add_argument("--run-dir", type=Path, default=None)
    p_t.add_argument("--n-test", type=int, default=None)
    p_t.set_defaults(func=cmd_t_min)

    p_a = sub.add_parser("arrival_noise", help="Arrival-rate multiplier sweep.")
    p_a.add_argument("--values", type=float, nargs="*", default=None)
    p_a.add_argument("--run-dir", type=Path, default=None)
    p_a.add_argument("--n-test", type=int, default=None)
    p_a.set_defaults(func=cmd_arrival_noise)

    p_tau = sub.add_parser("tau", help="Rollout horizon sweep.")
    p_tau.add_argument("--values", type=int, nargs="*", default=None)
    p_tau.add_argument("--n-test", type=int, default=None)
    p_tau.set_defaults(func=cmd_tau)

    p_th = sub.add_parser("theta", help="Confidence-filter threshold sweep.")
    p_th.add_argument("--values", type=float, nargs="*", default=None)
    p_th.set_defaults(func=cmd_theta)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
