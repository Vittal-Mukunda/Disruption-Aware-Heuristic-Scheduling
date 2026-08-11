"""Phase 6 / E2 — main comparison across methods, scenarios, and budgets.

Three sub-commands map 1:1 to the three pieces of E2 in HANDOFF §3.2:

  1. `eval`           — run any method against a named scenario, write a per-
                        method parquet under `results/<scenario>/`. Scenarios
                        are config sub-trees in `cfg.experiments.scenarios`.
  2. `stats`          — load per-method parquets for a scenario (default: the
                        existing `results/*.parquet` snapshot), compute bootstrap
                        95% CI per method + paired Wilcoxon vs OURS + BH-FDR.
                        Writes `results/E2/<scenario>_stats.parquet` and the
                        forest plot `figures/E2/<scenario>_forest_<metric>.{pdf,png}`.
  3. `data_efficiency`— re-train OURS at each budget in
                        `cfg.experiments.e2.data_efficiency_budgets` (subsets
                        the train.parquet to N shifts), then evaluate on the
                        canonical 50 test shifts. Heavy: each retrain is ~42 min
                        of XGB CV.

The `stats` sub-command is the only one that runs end-to-end in this session.
`eval` and `data_efficiency` exist as CLIs but are multi-hour compute and are
the user's call to kick off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.evaluate import (  # noqa: E402
    KPI_COLUMNS,
    _build_policy,
    canonical_test_seeds,
    evaluate_policy,
    evaluate_policy_env_aware,
)
from experiments.stats import compare_methods, load_phase5_results  # noqa: E402
from labeling.provenance import stamp_derived  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_ROOT = REPO_ROOT / "results"
FIG_ROOT = REPO_ROOT / "figures" / "E2"
# EDD is in this list for a specific reason: the rule the submitted paper called
# "FEFO" sorted on `sla_due` and was in fact EDD (Reviewer 1, 1.f). Without an
# EDD row a reader cannot map the old Table 1 onto the new one, and the genuinely
# expiry-aware FEFO would silently inherit the old rule's reputation.
DEFAULT_METHODS: list[str] = [
    "fifo", "edd", "fefo", "wspt", "atc", "linucb",
    "rolling_mpc", "greedy_mpc", "snapshot_xgb", "ppo_fair", "offline_fqi", "ours",
]
DEFAULT_METRICS: list[str] = ["service_failure_rate", "mean_tardiness",
                              "composite_cost", "throughput", "picker_utilization"]


def apply_scenario(cfg: DictConfig, scenario: str) -> DictConfig:
    """Overlay `cfg.experiments.scenarios.<scenario>` onto `cfg.sim`. Returns a new cfg.

    Pass-through when scenario == 'default' (keeps cfg.sim as-is).
    """
    if scenario == "default":
        return cfg
    scenarios = cfg.experiments.scenarios
    if scenario not in scenarios:
        raise SystemExit(
            f"unknown scenario '{scenario}'. Options: "
            f"{['default', *list(scenarios.keys())]}"
        )
    overlay = scenarios[scenario]
    new_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(new_cfg):
        if "arrivals" in overlay:
            new_cfg.sim.arrivals = OmegaConf.merge(
                new_cfg.sim.arrivals, overlay.arrivals
            )
        if "order_attrs" in overlay:
            new_cfg.sim.order_attrs = OmegaConf.merge(
                new_cfg.sim.order_attrs, overlay.order_attrs
            )
    return new_cfg


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = apply_scenario(cfg, args.scenario)
    seeds = canonical_test_seeds(cfg)
    if args.n_test is not None:
        seeds = seeds[: int(args.n_test)]

    out_dir = (args.results_dir or RESULTS_ROOT) / f"scenario_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods if args.methods else DEFAULT_METHODS
    for method in methods:
        print(f"[E2 eval] scenario={args.scenario}  method={method}  "
              f"n_seeds={len(seeds)}")
        policy, env_aware = _build_policy(method, args.run_dir)
        runner = evaluate_policy_env_aware if env_aware else evaluate_policy
        runner(method, policy, seeds, cfg, results_dir=out_dir,
               save=True, verbose=args.verbose)
    print(f"\n[E2 eval] wrote {out_dir.relative_to(REPO_ROOT)}/")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    methods = args.methods if args.methods else DEFAULT_METHODS
    baseline = args.baseline

    if args.scenario == "default":
        results_dir = args.results_dir or RESULTS_ROOT
    else:
        results_dir = (args.results_dir or RESULTS_ROOT) / f"scenario_{args.scenario}"
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir}")

    df_long = load_phase5_results(results_dir, methods=methods)
    print(f"[E2 stats] {len(df_long)} rows across "
          f"{df_long['method'].nunique()} methods, "
          f"{df_long['shift_seed'].nunique()} seeds.")

    n_resamples = int(cfg.experiments.stats.bootstrap_resamples)
    q = float(cfg.experiments.stats.fdr_q)

    stats_out_dir = RESULTS_ROOT / "E2"
    stats_out_dir.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[pd.DataFrame] = []
    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    for metric in metrics:
        stats_df = compare_methods(
            df_long, metric=metric, baseline=baseline,
            n_resamples=n_resamples, q=q,
        )
        stats_df["scenario"] = args.scenario
        all_metric_rows.append(stats_df)

        # Forest plot ordered by point estimate.
        plot_df = stats_df.sort_values("point").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(plot_df) + 1.2))
        y = np.arange(len(plot_df))
        ax.hlines(
            y, plot_df["ci_lo"], plot_df["ci_hi"],
            color="steelblue", lw=2.5, alpha=0.8,
        )
        ax.plot(plot_df["point"], y, "o", color="firebrick", ms=6)
        if baseline is not None and baseline in plot_df["method"].values:
            base_point = plot_df.loc[plot_df["method"] == baseline, "point"].iloc[0]
            ax.axvline(base_point, color="grey", ls=":", lw=1.0,
                       label=f"{baseline}")
            ax.legend(loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["method"])
        ax.set_xlabel(metric)
        ax.set_title(
            f"E2 — {metric}  (scenario={args.scenario})\n"
            f"50 test shifts · 250 training shifts · 95% bootstrap CI"
        )
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_ROOT / f"{args.scenario}_forest_{metric}.png", dpi=150)
        fig.savefig(FIG_ROOT / f"{args.scenario}_forest_{metric}.pdf")
        plt.close(fig)

        print(f"\n[E2 stats] {metric} (scenario={args.scenario}):")
        cols = ["method", "point", "ci_lo", "ci_hi"]
        if baseline is not None:
            cols += ["diff_point", "p_raw", "p_adj_bh", "reject_bh"]
        print(stats_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_table = pd.concat(all_metric_rows, ignore_index=True)
    out_path = stats_out_dir / f"{args.scenario}_stats.parquet"
    out_table.to_parquet(out_path, index=False)
    print(f"\n[E2 stats] wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


def cmd_data_efficiency(args: argparse.Namespace) -> int:
    """Re-train OURS at each budget in `cfg.experiments.e2.data_efficiency_budgets`.

    Heavy: each retrain runs the same Phase 4 pipeline as `train_ranker.py`,
    just on a subset of shifts. Outputs are written to
    `runs/data_efficiency/ours_n<budget>/` and evaluated to
    `results/data_efficiency/ours_n<budget>.parquet`.
    """
    cfg = OmegaConf.load(CONFIG_PATH)
    budgets = (
        args.budgets if args.budgets
        else list(cfg.experiments.e2.data_efficiency_budgets)
    )
    print(f"[E2 data_efficiency] budgets = {budgets}  "
          f"reps = {int(cfg.experiments.e2.data_efficiency_reps)}")

    df_train_full = pd.read_parquet(REPO_ROOT / "data" / "train.parquet")
    all_shift_ids = sorted(df_train_full["shift_id"].unique())
    rng = np.random.default_rng(args.seed)

    run_root = REPO_ROOT / "runs" / "data_efficiency"
    out_root = RESULTS_ROOT / "data_efficiency"
    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    rep_log: list[dict] = []
    n_reps = int(cfg.experiments.e2.data_efficiency_reps)
    for budget in budgets:
        if budget > len(all_shift_ids):
            print(f"  [skip] budget={budget} > {len(all_shift_ids)} train shifts")
            continue
        for rep in range(n_reps):
            chosen = rng.choice(all_shift_ids, size=int(budget), replace=False)
            sub = df_train_full[df_train_full["shift_id"].isin(chosen)]
            tag = f"ours_n{budget}_rep{rep}"
            run_dir = run_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            train_subset_path = run_dir / "train_subset.parquet"
            sub.to_parquet(train_subset_path, index=False)
            # Carry the labelling stamp onto the subset, or Stage 3 will refuse
            # it as unprovenanced (labeling/provenance.py).
            stamp_derived(
                REPO_ROOT / "data" / "train.parquet", train_subset_path,
                f"shift subset: budget={budget} rep={rep}",
                n_train_shifts=int(budget),
            )

            print(f"[E2 data_efficiency] budget={budget} rep={rep} "
                  f"shifts={len(chosen)} -> {run_dir.relative_to(REPO_ROOT)}")
            if args.dry_run:
                continue

            import subprocess
            cmd = [
                sys.executable, "-m", "experiments.train_ranker",
                "--run-id", f"data_efficiency/{tag}",
                "--train-path", str(train_subset_path),
                "--test-path", str(REPO_ROOT / "data" / "test.parquet"),
                "--skip-cv-cal",
            ]
            subprocess.check_call(cmd, cwd=str(REPO_ROOT))

            from baselines.ours import load_ours
            policy = load_ours(REPO_ROOT / "runs" / "data_efficiency" / tag)
            seeds = canonical_test_seeds(cfg)
            df_eval = evaluate_policy(
                tag, policy, seeds, cfg,
                results_dir=out_root, save=True, verbose=False,
            )
            rep_log.append({
                "budget": int(budget),
                "rep": int(rep),
                "service_failure_rate_mean": float(df_eval["service_failure_rate"].mean()),
                "composite_cost_mean": float(df_eval["composite_cost"].mean()),
            })

    if not args.dry_run and rep_log:
        log_path = out_root / "data_efficiency_summary.json"
        log_path.write_text(json.dumps(rep_log, indent=2), encoding="utf-8")
        print(f"\n[E2 data_efficiency] summary -> "
              f"{log_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate methods on a scenario.")
    p_eval.add_argument("--scenario", type=str, default="default")
    p_eval.add_argument("--methods", type=str, nargs="*", default=None)
    p_eval.add_argument("--n-test", type=int, default=None)
    p_eval.add_argument("--run-dir", type=Path, default=None)
    p_eval.add_argument("--results-dir", type=Path, default=None)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_stats = sub.add_parser("stats", help="Compute bootstrap CI + Wilcoxon + BH-FDR.")
    p_stats.add_argument("--scenario", type=str, default="default")
    p_stats.add_argument("--methods", type=str, nargs="*", default=None)
    p_stats.add_argument("--baseline", type=str, default="ours")
    p_stats.add_argument("--metrics", type=str, nargs="*", default=None)
    p_stats.add_argument("--results-dir", type=Path, default=None)
    p_stats.set_defaults(func=cmd_stats)

    p_de = sub.add_parser("data_efficiency",
                          help="Re-train OURS at multiple budgets.")
    p_de.add_argument("--budgets", type=int, nargs="*", default=None)
    p_de.add_argument("--seed", type=int, default=1337)
    p_de.add_argument("--dry-run", action="store_true",
                      help="Print the plan without retraining.")
    p_de.set_defaults(func=cmd_data_efficiency)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
