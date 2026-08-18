"""What the selector does under saturation, and whether the dwell holds it back.

REVIEWER 3, COMMENT 3
---------------------
    "In the high-load-perishable scenario, DAHS's SLA-breach rate (19.43%) is
     slightly worse than greedy_mpc's (18.84%). The authors attribute this to
     saturation effects, but a deeper analysis would be helpful: Does DAHS's
     rule-selection distribution shift significantly under saturation? Could the
     minimum dwell constraint of the switching controller prevent timely
     adaptation in this regime? Understanding these boundary conditions is
     important for practitioners."

The submitted paper explained the one cell it loses as "a saturation effect" and
left it there. That is an attribution, not an analysis, and the reviewer is right
that a practitioner needs to know *which* boundary condition bites. The
controller now records every decision it takes (`models.switching_controller`),
so both questions can be answered from a run rather than argued about.

QUESTION 1 — does the selection distribution shift under load?
    Reported per scenario: the share of epochs each rule is deployed, and the
    exponentiated entropy of that distribution, which equals |H| when the
    selector spreads across the pool and 1.0 when it has collapsed onto a single
    rule. A selector that adapts should look different at low load and at
    saturation; one that has collapsed is not selecting at all, and its scenario
    KPI is really the KPI of whichever rule it settled on.

QUESTION 2 — does the minimum dwell prevent timely adaptation?
    `blocked_switch_rate` is the share of epochs at which the ranker's arg-max
    differed from the deployed rule *because the dwell was still active*. If that
    rate climbs with load, the dwell is holding rules the ranker has already
    abandoned, and the guardrail is costing adaptivity exactly where adaptivity
    matters most. The `dwell` sub-command settles it causally by sweeping
    `t_min_intervals` inside the saturated scenario alone.

    python -m experiments.saturation_analysis trace
    python -m experiments.saturation_analysis dwell --scenario high_load_perish
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.e2_main import apply_scenario
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from simulation.heuristics import resolve_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E13_saturation"
FIG_DIR = REPO_ROOT / "figures" / "E13_saturation"

SCENARIOS = ["low_load", "balanced", "default", "high_load_perish"]


def _load_ours(cfg: DictConfig, run_dir: Path | None = None):
    from baselines.ours import load_ours

    return load_ours(run_dir or REPO_ROOT / "runs" / "phase4", cfg=cfg)


def _controller_summary(controller, scenario: str, extra: dict) -> dict:
    dist = controller.selection_distribution()
    return {
        "scenario": scenario,
        "selection_entropy_effective_rules": controller.selection_entropy(),
        "switch_rate": controller.switch_rate(),
        "blocked_switch_rate": controller.blocked_switch_rate(),
        "gate_open_rate": controller.gate_open_rate(),
        **{f"share_{h}": v for h, v in dist.items()},
        **extra,
    }


def cmd_trace(args: argparse.Namespace) -> int:
    """Run DAHS across scenarios and summarise its own decision trace."""
    base = OmegaConf.load(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    traces: list[pd.DataFrame] = []

    for scenario in (args.scenarios or SCENARIOS):
        cfg = apply_scenario(base, scenario)
        seeds = canonical_test_seeds(cfg)
        if args.n_test:
            seeds = seeds[: args.n_test]

        policy = _load_ours(cfg)
        controller = policy.controller
        controller.reset(keep_trace=False)

        df = evaluate_policy(
            f"ours_{scenario}", policy, seeds, cfg,
            results_dir=RESULTS_DIR / "kpis", save=True,
            n_jobs=1,
        )
        rows.append(_controller_summary(controller, scenario, {
            "composite_cost": float(df["composite_cost"].mean()),
            "service_failure_rate": float(df["service_failure_rate"].mean()),
            "picker_utilization": float(df["picker_utilization"].mean()),
            "n_epochs": len(controller.trace),
        }))
        t = controller.trace_frame()
        t["scenario"] = scenario
        traces.append(t)
        print(f"[saturation] {scenario:<18s} "
              f"eff.rules={rows[-1]['selection_entropy_effective_rules']:.2f}  "
              f"switch={rows[-1]['switch_rate']:.3f}  "
              f"blocked={rows[-1]['blocked_switch_rate']:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_parquet(RESULTS_DIR / "scenario_behaviour.parquet", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(
        RESULTS_DIR / "decision_traces.parquet", index=False
    )

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(summary) > 1:
        lo = summary.iloc[0]
        hi = summary.iloc[-1]
        print(f"\n  selection entropy {lo['scenario']} -> {hi['scenario']}: "
              f"{lo['selection_entropy_effective_rules']:.2f} -> "
              f"{hi['selection_entropy_effective_rules']:.2f}")
        print(f"  blocked-switch rate {lo['scenario']} -> {hi['scenario']}: "
              f"{lo['blocked_switch_rate']:.3f} -> {hi['blocked_switch_rate']:.3f}")
        print("  A rising blocked-switch rate under load is direct evidence that "
              "the dwell is preventing timely adaptation; a falling selection "
              "entropy means the selector is collapsing onto one rule.")
    return 0


def cmd_dwell(args: argparse.Namespace) -> int:
    """Sweep the minimum dwell inside one scenario — the causal test."""
    base = OmegaConf.load(CONFIG_PATH)
    cfg = apply_scenario(base, args.scenario)
    seeds = canonical_test_seeds(cfg)
    if args.n_test:
        seeds = seeds[: args.n_test]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for t_min in (args.t_min or list(base.experiments.e4_sensitivity.t_min)):
        cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(cfg_t):
            cfg_t.ranker.switching.t_min_intervals = int(t_min)

        policy = _load_ours(cfg_t)
        controller = policy.controller
        controller.reset(keep_trace=False)
        df = evaluate_policy(
            f"ours_tmin{t_min}", policy, seeds, cfg_t,
            results_dir=RESULTS_DIR / f"dwell_{args.scenario}", save=True,
        )
        rows.append(_controller_summary(controller, args.scenario, {
            "t_min": int(t_min),
            "composite_cost": float(df["composite_cost"].mean()),
            "service_failure_rate": float(df["service_failure_rate"].mean()),
        }))
        print(f"  t_min={t_min}  cost={rows[-1]['composite_cost']:8.3f}  "
              f"fail={rows[-1]['service_failure_rate']:.4f}  "
              f"switch={rows[-1]['switch_rate']:.3f}  "
              f"blocked={rows[-1]['blocked_switch_rate']:.3f}")

    table = pd.DataFrame(rows).sort_values("t_min")
    table.to_parquet(
        RESULTS_DIR / f"dwell_sweep_{args.scenario}.parquet", index=False
    )

    best = table.loc[table["composite_cost"].idxmin()]
    deployed = table[table["t_min"] == int(base.ranker.switching.t_min_intervals)]
    verdict = {
        "scenario": args.scenario,
        "best_t_min": int(best["t_min"]),
        "best_cost": float(best["composite_cost"]),
        "deployed_t_min": int(base.ranker.switching.t_min_intervals),
        "deployed_cost": (
            float(deployed["composite_cost"].iloc[0]) if len(deployed) else None
        ),
        "note": (
            "If the cost-minimising dwell in this scenario is shorter than the "
            "deployed one, the guardrail is the boundary condition the reviewer "
            "asked about, and the paper should say so and quantify the trade "
            "against switching frequency rather than defending the default."
        ),
    }
    (RESULTS_DIR / f"dwell_verdict_{args.scenario}.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(f"\n[saturation] best t_min in {args.scenario}: {verdict['best_t_min']} "
          f"(deployed {verdict['deployed_t_min']})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("trace", help="Selection behaviour across scenarios.")
    pt.add_argument("--scenarios", nargs="*", default=None)
    pt.add_argument("--n-test", type=int, default=None)
    pt.set_defaults(func=cmd_trace)

    pdw = sub.add_parser("dwell", help="Sweep t_min inside one scenario.")
    pdw.add_argument("--scenario", type=str, default="high_load_perish")
    pdw.add_argument("--t-min", nargs="*", type=int, default=None)
    pdw.add_argument("--n-test", type=int, default=None)
    pdw.set_defaults(func=cmd_dwell)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
