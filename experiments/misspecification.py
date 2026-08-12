"""Model misspecification: what happens when the simulator is wrong.

REVIEWER 2, COMMENT 5
---------------------
    "The offline rollout procedure implicitly assumes that the simulator
     perfectly mirrors real-world order arrivals, processing times, and picker
     dynamics. In practice, this is almost never the case. Any model
     misspecification will inevitably corrupt the rollout labels, and these
     errors will compound over the 4-interval rollout horizon, potentially
     flipping the preferred rule. While Proposition 1 addresses truncation error,
     it completely ignores model error. The paper should acknowledge this
     limitation. Ideally, the authors should test the robustness of DAHS in a
     scenario where the simulator used for evaluation has slightly different
     parameters than the one used to generate the training labels."

The reviewer is right that Proposition 1 bounds the wrong thing on its own, and
right that the compounding is the concern. Section 4.4 now carries a second
bound, Proposition 2, showing that model error accumulates as O(eps * tau^2)
while truncation bias decays as O(H - tau) — so the two act in opposite
directions in tau and the optimal horizon is interior for a reason that has
nothing to do with estimator variance. This module is the empirical test of that
prediction.

DESIGN
------
Labels are generated once under the NOMINAL configuration. Evaluation then runs
under PERTURBED configurations along the four axes a warehouse model is most
likely to get wrong: arrival rate, processing-time scale, due-window scale, and
picker headcount. Nothing is retrained. The frozen selector meets a world its
training labels never described.

FAIRNESS OF THE LOOKAHEAD COMPARATOR
------------------------------------
A rolling-horizon controller would trivially win this experiment if its internal
rollouts were allowed to use the perturbed dynamics, because it would then be
planning with a model nobody has. It is therefore run with `model_cfg` pinned to
the NOMINAL configuration, exactly the model DAHS's labels were built from. Both
controllers are wrong about the world in the same way and to the same degree, and
the experiment measures which degrades more gracefully. This is the comparison
that carries information: an amortised lookahead cannot re-plan against evidence,
whereas an online lookahead re-plans every epoch — but re-plans with a wrong
model, so it is not obvious in advance which effect dominates.

The static rules are included as the misspecification-free reference: they carry
no model at all, so their degradation is the environment's, not the method's.

    python -m experiments.misspecification run --axis arrival_rate_scale
    python -m experiments.misspecification summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.evaluate import (
    canonical_test_seeds,
    evaluate_policy,
    evaluate_policy_env_aware,
    _build_policy,
)
from experiments.stats import bootstrap_mean_ci
from simulation.heuristics import resolve_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E10_misspecification"
FIG_DIR = REPO_ROOT / "figures" / "E10_misspecification"

# Methods spanning the three relevant categories: model-based and amortised
# (ours), model-based and online (rolling_mpc), and model-free (the static rules
# and the best learned model-free baseline).
METHODS = ["ours", "rolling_mpc", "offline_fqi"]


def perturb(cfg: DictConfig, axis: str, value: float) -> DictConfig:
    """Apply one misspecification axis to a copy of `cfg`.

    Each axis is a multiplicative or additive perturbation of a parameter an
    operator would have to estimate from data, and would estimate imperfectly.
    """
    new = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(new):
        if axis == "arrival_rate_scale":
            new.sim.arrivals.base_rate_per_minute = float(
                cfg.sim.arrivals.base_rate_per_minute * value
            )
        elif axis == "processing_time_scale":
            new.sim.order_attrs.processing_time_triangular = [
                float(v * value) for v in cfg.sim.order_attrs.processing_time_triangular
            ]
        elif axis == "sla_scale":
            new.sim.order_attrs.sla_due_triangular = [
                float(v * value) for v in cfg.sim.order_attrs.sla_due_triangular
            ]
        elif axis == "shelf_life_scale":
            # The PRODUCT clock. Added because this revision introduced it: shelf
            # life is now a modelled parameter that an operator estimates from
            # data as imperfectly as any other, and perturbing only the customer
            # clock would leave the newly-added dimension untested. It is also
            # the axis that moves WHICH clock binds, so it is the sharpest test
            # of whether the expiry-aware machinery earns its place.
            new.sim.order_attrs.shelf_life_triangular = [
                float(v * value) for v in cfg.sim.order_attrs.shelf_life_triangular
            ]
        elif axis == "n_pickers_delta":
            new.sim.n_pickers = max(1, int(cfg.sim.n_pickers + int(value)))
        else:
            raise ValueError(f"unknown misspecification axis '{axis}'")
    return new


def _is_nominal(axis: str, value: float) -> bool:
    return value == (0.0 if axis == "n_pickers_delta" else 1.0)


def cmd_run(args: argparse.Namespace) -> int:
    nominal = OmegaConf.load(CONFIG_PATH)
    axes = (
        {args.axis: list(nominal.experiments.e10_misspecification[args.axis])}
        if args.axis
        else {
            a: list(v) for a, v in nominal.experiments.e10_misspecification.items()
        }
    )
    methods = list(args.methods) if args.methods else METHODS + resolve_pool(nominal)[:2]
    seeds = canonical_test_seeds(nominal)
    if args.n_test:
        seeds = seeds[: args.n_test]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for axis, values in axes.items():
        for value in values:
            eval_cfg = perturb(nominal, axis, float(value))
            cell = f"{axis}={value}"
            print(f"\n[misspec] {cell}   ({len(methods)} methods x {len(seeds)} shifts)")

            for method in methods:
                policy, env_aware = _build_policy(method, None)

                # Pin any model-based online controller to the NOMINAL model.
                # Without this the lookahead plans with dynamics DAHS never saw
                # and the comparison stops being about misspecification.
                if hasattr(policy, "model_cfg"):
                    policy.model_cfg = nominal

                runner = evaluate_policy_env_aware if env_aware else evaluate_policy
                df = runner(
                    f"{method}", policy, seeds, eval_cfg,
                    results_dir=RESULTS_DIR / cell.replace("=", "_"),
                    save=True, n_jobs=args.n_jobs,
                )
                ci = bootstrap_mean_ci(df["composite_cost"].to_numpy(np.float64))
                rows.append({
                    "axis": axis,
                    "value": float(value),
                    "nominal": _is_nominal(axis, float(value)),
                    "method": method,
                    "composite_cost": float(df["composite_cost"].mean()),
                    "ci_lo": ci.lo,
                    "ci_hi": ci.hi,
                    "service_failure_rate": float(df["service_failure_rate"].mean()),
                    "throughput": float(df["throughput"].mean()),
                })
                print(f"    {method:<14s} cost={rows[-1]['composite_cost']:8.3f}  "
                      f"fail={rows[-1]['service_failure_rate']:.4f}")

    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS_DIR / "misspecification.parquet", index=False)
    print(f"\n[misspec] wrote {RESULTS_DIR.relative_to(REPO_ROOT)}/misspecification.parquet")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Degradation slope per method: how fast does cost rise with model error?"""
    path = RESULTS_DIR / "misspecification.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `misspecification run` first.")
    df = pd.read_parquet(path)

    out: list[dict] = []
    for (axis, method), g in df.groupby(["axis", "method"]):
        base = g[g["nominal"]]
        if base.empty:
            continue
        c0 = float(base["composite_cost"].iloc[0])
        g = g.sort_values("value")
        # Perturbation magnitude, normalised so axes are comparable: relative
        # deviation from nominal for the multiplicative axes, absolute headcount
        # change scaled by the nominal count for the additive one.
        if axis == "n_pickers_delta":
            mag = (g["value"].abs() / 10.0).to_numpy(np.float64)
        else:
            mag = (g["value"] - 1.0).abs().to_numpy(np.float64)
        rel = (g["composite_cost"].to_numpy(np.float64) - c0) / max(abs(c0), 1e-9)
        slope = float(np.polyfit(mag, rel, 1)[0]) if len(mag) > 1 else 0.0
        out.append({
            "axis": axis,
            "method": method,
            "nominal_cost": c0,
            "worst_cost": float(g["composite_cost"].max()),
            "relative_degradation_slope": slope,
        })

    summary = pd.DataFrame(out).sort_values(["axis", "relative_degradation_slope"])
    summary.to_parquet(RESULTS_DIR / "degradation_slopes.parquet", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ranked = (
        summary.groupby("method")["relative_degradation_slope"].mean().sort_values()
    )
    (RESULTS_DIR / "degradation_summary.json").write_text(
        json.dumps({
            "mean_slope_by_method": {k: float(v) for k, v in ranked.items()},
            "most_robust": str(ranked.index[0]),
            "least_robust": str(ranked.index[-1]),
            "note": (
                "Slope is the relative rise in composite cost per unit of "
                "relative model error, averaged over axes. Static rules carry no "
                "model, so their slope is the environment's own difficulty "
                "gradient and is the reference against which the model-based "
                "methods should be read."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n[misspec] most robust: {ranked.index[0]}   "
          f"least robust: {ranked.index[-1]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("run", help="Evaluate frozen methods under perturbed dynamics.")
    pr.add_argument("--axis", type=str, default=None)
    pr.add_argument("--n-jobs", type=int, default=-1,
                    help="Shifts evaluated in parallel. This sweep is dominated "
                         "by the online lookahead baseline — ~14 h serial, under "
                         "an hour across 16 cores.")
    pr.add_argument("--methods", nargs="*", default=None)
    pr.add_argument("--n-test", type=int, default=None)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("summary", help="Degradation slopes per method.")
    ps.set_defaults(func=cmd_summary)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
