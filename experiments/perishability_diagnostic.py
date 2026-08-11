"""Does perishability actually bind at a 15-minute decision horizon?

REVIEWER 1, COMMENT 1.d
-----------------------
    "'Perishability-constrained' is supported only if product expiration affects
     feasibility, quality, waste, inventory allocation, or the objective within
     the modelled horizon. Does perishability matter over such a short horizon?
     In what percentage of decisions does delaying an order by one interval
     (15 minutes) alter its feasibility, quality, or economic value?"

This is a fair challenge and the submitted paper could not answer it, because
perishability never entered the objective and there was no product clock to
delay against: "spoilage" was defined as a perishable order missing its *due
date*, so the question was unanswerable by construction.

With the two-clock order model of `simulation.orders` the question is now
well posed and this module answers it directly, in the reviewer's own units.

WHAT IS MEASURED
----------------
At every decision epoch `t`, for every order `o` waiting in the queue, compare
the two options actually available to the dispatcher:

    serve now   -> completes at  t + p_o
    serve next  -> completes at  t + L + p_o        (L = interval length)

An order is PIVOTAL at that epoch if the two options straddle a deadline:

    expiry-pivotal   t + p_o <= x_o  <  t + L + p_o     (in-date -> spoiled)
    due-pivotal      t + p_o <= d_o  <  t + L + p_o     (on-time -> late)

`expiry_pivotal_rate` is the number the reviewer asked for: the fraction of
(epoch, queued order) pairs where one interval of delay is the difference
between saleable goods and waste. `epoch_any_expiry_pivotal` is the same
quantity aggregated to decisions — the fraction of decision epochs at which at
least one order in the queue is expiry-pivotal, i.e. the fraction of decisions
where perishability is live rather than latent.

Also reported, because "affects the objective" is the reviewer's criterion:

  * `expiry_binds_rate`     — share of perishables whose product clock is
                              strictly tighter than their customer clock. If
                              this is near zero, expiry never binds and FEFO is
                              a relabelled EDD regardless of the delay analysis.
  * `pivotal_value_share`   — share of total economic weight `w_o` sitting on
                              expiry-pivotal orders. Feasibility flips are only
                              interesting in proportion to what they cost.
  * `spoilage_cost_share`   — share of realised composite cost attributable to
                              the `w_spoil` term under each static rule.

HONEST OUTCOME
--------------
If `epoch_any_expiry_pivotal` is negligible, the correct response is to drop
"perishability-constrained" from the title and framing, remove FEFO from the
pool, and report this diagnostic as the reason. The module is written to make
that outcome as easy to act on as the favourable one; `verdict()` states which
way the evidence falls against a pre-registered threshold.

    python -m experiments.perishability_diagnostic --n-shifts 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from labeling.rollout_labeler import rollout_seed
from seed import shift_corpora
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "S1_perishability"

# Pre-registered thresholds, fixed before the diagnostic runs so the decision
# rule cannot be fitted to the answer. All three must hold for the
# "perishability-constrained" framing to stand.
#
#   MATERIALITY      share of decisions with at least one expiry-pivotal order.
#                    A necessary condition, and a weak one: with a large queue it
#                    saturates near 1 whether or not perishability matters.
#   EXPIRY_BINDS     share of perishables whose product clock is tighter than
#                    their customer clock. If this is ~0, FEFO is EDD by another
#                    name and the second clock is decorative.
#   DISCRIMINATION   share of decisions at which the CHOICE OF RULE changes how
#                    much spoils. This is the binding condition — the others can
#                    hold while this fails, and then perishability constrains the
#                    world but not the decision.
MATERIALITY_THRESHOLD: float = 0.05
EXPIRY_BINDS_THRESHOLD: float = 0.10
DISCRIMINATION_THRESHOLD: float = 0.10


def _scan_one_shift(seed: int, cfg: DictConfig, rule: str) -> list[dict]:
    """Walk one shift under `rule`, recording pivotality at every epoch.

    The behaviour rule matters only through the states it visits; the pivotality
    test itself is a property of the queue, not of the action taken. Scanning
    under each static rule in turn and pooling gives coverage of the queue
    configurations a controller could actually face.
    """
    env = WarehouseEnv(int(seed), cfg)
    L = env.interval_minutes
    rows: list[dict] = []

    while env.interval_idx < env.n_intervals:
        env.observe()
        t = env.t
        q = env.queue
        if q:
            now = np.array([t + o.processing_time for o in q])
            later = now + L
            due = np.array([o.sla_due for o in q])
            w = np.array([o.weight for o in q])
            perish = np.array([o.expiry_time is not None for o in q])
            exp = np.array(
                [o.expiry_time if o.expiry_time is not None else np.inf for o in q]
            )

            due_pivotal = (now <= due) & (due < later)
            exp_pivotal = perish & (now <= exp) & (exp < later)
            # Already unrecoverable on the product clock: delay changes nothing
            # because the goods are lost either way.
            exp_lost = perish & (exp < now)

            rows.append({
                "shift_seed": int(seed),
                "behaviour_rule": rule,
                "interval_idx": int(env.interval_idx),
                "queue_len": int(len(q)),
                "n_perishable": int(perish.sum()),
                "n_due_pivotal": int(due_pivotal.sum()),
                "n_expiry_pivotal": int(exp_pivotal.sum()),
                "n_expiry_already_lost": int(exp_lost.sum()),
                "weight_total": float(w.sum()),
                "weight_expiry_pivotal": float(w[exp_pivotal].sum()),
                "weight_due_pivotal": float(w[due_pivotal].sum()),
            })
        env.step(rule)

    # Structural check, independent of any decision: how often is the product
    # clock the binding one at all?
    arrived = env.arrived_orders()
    perishables = [o for o in arrived if o.expiry_time is not None]
    n_binds = sum(1 for o in perishables if o.expiry_binds())
    kpi = env.kpis()
    for r in rows:
        r["shift_n_perishable_arrived"] = len(perishables)
        r["shift_n_expiry_binds"] = n_binds
        r["shift_spoilage_rate"] = kpi["spoilage_rate"]
        r["shift_composite_cost"] = kpi["composite_cost"]
    return rows


def scan(cfg: DictConfig, seeds: list[int], rules: list[str], n_jobs: int) -> pd.DataFrame:
    out = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_scan_one_shift)(s, cfg, r) for s in seeds for r in rules
    )
    return pd.DataFrame([row for shift in out for row in shift])


def summarise(df: pd.DataFrame) -> dict:
    """Reduce the epoch-level scan to the numbers Reviewer 1 (1.d) asked for."""
    n_epochs = len(df)
    n_order_epochs = float(df["queue_len"].sum())
    n_perish_order_epochs = float(df["n_perishable"].sum())

    return {
        # --- the direct answer, per (epoch, queued order) ---
        "expiry_pivotal_rate_all_orders": float(
            df["n_expiry_pivotal"].sum() / max(n_order_epochs, 1)
        ),
        "expiry_pivotal_rate_perishables": float(
            df["n_expiry_pivotal"].sum() / max(n_perish_order_epochs, 1)
        ),
        "due_pivotal_rate_all_orders": float(
            df["n_due_pivotal"].sum() / max(n_order_epochs, 1)
        ),
        # --- the same thing aggregated to decisions ---
        "epoch_any_expiry_pivotal": float((df["n_expiry_pivotal"] > 0).mean()),
        "epoch_any_due_pivotal": float((df["n_due_pivotal"] > 0).mean()),
        # --- is the product clock ever the binding one? ---
        "expiry_binds_rate": float(
            df.groupby("shift_seed")["shift_n_expiry_binds"].first().sum()
            / max(df.groupby("shift_seed")["shift_n_perishable_arrived"].first().sum(), 1)
        ),
        # --- economic exposure, not just count ---
        "pivotal_value_share": float(
            df["weight_expiry_pivotal"].sum() / max(df["weight_total"].sum(), 1e-9)
        ),
        # --- context ---
        "mean_already_lost_per_epoch": float(df["n_expiry_already_lost"].mean()),
        "mean_queue_len": float(df["queue_len"].mean()),
        "n_decision_epochs": int(n_epochs),
    }


def _spoiled(env: WarehouseEnv, t_ref: float) -> tuple[int, float]:
    """(count, economic weight) of arrived orders spoiled as of `t_ref`."""
    sp = [o for o in env.arrived_orders() if o.is_spoiled(t_ref)]
    return len(sp), float(sum(o.weight for o in sp))


def _discriminate_one_shift(
    seed: int, cfg: DictConfig, pool: list[str], tau: int, n_samples: int,
    base_seed: int, behaviour: str,
) -> list[dict]:
    """Per epoch, does the CHOICE OF RULE change how much spoils?

    This is the sharp form of the reviewer's question and the pivotality scan
    above is only its precondition. An order can be expiry-pivotal at an epoch —
    one interval of delay would spoil it — while every rule in the pool happens
    to treat it identically, in which case perishability constrains the *world*
    but not the *decision*, and an expiry-aware rule earns nothing.

    For each rule we roll forward `tau` intervals over `n_samples` shared
    continuations and record the spoilage accrued inside the window. The spread
    across rules is the quantity that matters: if it is zero the rule choice is
    irrelevant to spoilage, however pivotal the individual orders were.
    """
    env = WarehouseEnv(int(seed), cfg)
    rows: list[dict] = []

    for t in range(env.n_intervals):
        env.observe()
        steps = min(tau, env.n_intervals - t)
        if steps <= 0:
            break
        sp0_n, sp0_w = _spoiled(env, env.t)
        phi0 = env.potential()

        per_rule_spoil: dict[str, float] = {}
        per_rule_cost: dict[str, float] = {}
        for h in pool:
            sn, sw, cc = [], [], []
            for m in range(n_samples):
                b = env.branch(rollout_seed(base_seed, seed, t, m))
                b.run_with_policy(h, n_steps=steps)
                n1, w1 = _spoiled(b, b.t)
                sn.append(n1 - sp0_n)
                sw.append(w1 - sp0_w)
                cc.append(b.potential() - phi0)
            per_rule_spoil[h] = float(np.mean(sw))
            per_rule_cost[h] = float(np.mean(cc))

        spoil_vals = np.array(list(per_rule_spoil.values()))
        cost_vals = np.array(list(per_rule_cost.values()))
        rows.append({
            "shift_seed": int(seed),
            "interval_idx": int(t),
            "spoil_spread": float(spoil_vals.max() - spoil_vals.min()),
            "cost_spread": float(cost_vals.max() - cost_vals.min()),
            "best_for_spoilage": pool[int(spoil_vals.argmin())],
            "best_for_cost": pool[int(cost_vals.argmin())],
            **{f"spoil_{h}": v for h, v in per_rule_spoil.items()},
        })
        env.step(behaviour)
    return rows


def rule_discrimination(
    cfg: DictConfig, seeds: list[int], pool: list[str], n_jobs: int
) -> tuple[pd.DataFrame, dict]:
    """Does the rule choice move spoilage at all, and does FEFO win when it does?"""
    tau = int(cfg.labeling.tau)
    m = int(cfg.heuristics.calibration.get("n_rollout_samples", 5))
    base_seed = int(cfg.seeds.rollout)
    behaviour = pool[0]

    out = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_discriminate_one_shift)(s, cfg, pool, tau, m, base_seed, behaviour)
        for s in seeds
    )
    df = pd.DataFrame([r for shift in out for r in shift])
    if df.empty:
        return df, {}

    discriminating = df["spoil_spread"] > 1e-9
    fefo_share = (
        float((df.loc[discriminating, "best_for_spoilage"] == "FEFO").mean())
        if discriminating.any() else 0.0
    )
    return df, {
        # THE decision-relevance number.
        "epochs_where_rule_changes_spoilage": float(discriminating.mean()),
        "mean_spoilage_spread_weighted": float(df["spoil_spread"].mean()),
        "mean_spoilage_spread_when_discriminating": (
            float(df.loc[discriminating, "spoil_spread"].mean())
            if discriminating.any() else 0.0
        ),
        # Does the expiry-aware rule actually win the spoilage criterion? If FEFO
        # rarely minimises spoilage even where spoilage is contested, it does not
        # earn its slot and the Stage-1 screen should drop it.
        "fefo_wins_spoilage_share": fefo_share,
        "mean_cost_spread": float(df["cost_spread"].mean()),
    }


def verdict(summary: dict, disc: dict | None = None) -> dict:
    """State plainly which way the evidence falls, against fixed thresholds.

    TWO conditions, both pre-registered, and both required. The pivotality rate
    alone is a weak test: with a queue of a hundred-odd orders, roughly a fifth
    of them perishable, *some* order is nearly always within one interval of its
    expiry, so `epoch_any_expiry_pivotal` approaches 1 whether or not
    perishability matters to the controller. The binding condition is the second
    one — that the choice of rule actually moves spoilage.
    """
    pivotal = summary["epoch_any_expiry_pivotal"]
    binds = summary["expiry_binds_rate"]
    discriminates = (disc or {}).get("epochs_where_rule_changes_spoilage", 0.0)
    fefo_wins = (disc or {}).get("fefo_wins_spoilage_share", 0.0)

    cond_pivotal = bool(pivotal >= MATERIALITY_THRESHOLD)
    cond_binds = bool(binds >= EXPIRY_BINDS_THRESHOLD)
    cond_discriminates = bool(discriminates >= DISCRIMINATION_THRESHOLD)
    supported = cond_pivotal and cond_binds and cond_discriminates

    if supported:
        rec = (
            "Retain the 'perishability-constrained' framing and keep FEFO as a "
            "screening candidate. Report all three statistics in Section 3.5."
        )
    elif cond_pivotal and cond_binds and not cond_discriminates:
        rec = (
            "Perishability is present in the instances but NOT decision-relevant: "
            "the rule choice does not move spoilage. Keep the two-clock order "
            "model, since it is the honest formulation, but withdraw the claim "
            "that the setting is perishability-CONSTRAINED, drop FEFO, and report "
            "this diagnostic as the reason. This is a publishable negative result "
            "and directly answers Reviewer 1 (1.d)."
        )
    else:
        rec = (
            "Product expiry rarely binds before the customer deadline under the "
            "current shelf-life distribution. Either re-parameterise shelf life "
            "so the two clocks genuinely compete and re-run, or drop "
            "perishability from the paper entirely. Do NOT re-parameterise and "
            "then report only the favourable configuration."
        )

    return {
        "thresholds": {
            "epoch_any_expiry_pivotal": MATERIALITY_THRESHOLD,
            "expiry_binds_rate": EXPIRY_BINDS_THRESHOLD,
            "epochs_where_rule_changes_spoilage": DISCRIMINATION_THRESHOLD,
        },
        "measured": {
            "epoch_any_expiry_pivotal": pivotal,
            "expiry_binds_rate": binds,
            "epochs_where_rule_changes_spoilage": discriminates,
            "fefo_wins_spoilage_share": fefo_wins,
        },
        "conditions_met": {
            "orders_are_pivotal": cond_pivotal,
            "expiry_binds_before_due": cond_binds,
            "rule_choice_moves_spoilage": cond_discriminates,
        },
        "perishability_claim_supported": supported,
        "recommendation": rec,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-shifts", type=int, default=30,
                   help="Calibration shifts to scan (default: the whole block).")
    p.add_argument("--rules", nargs="*", default=None,
                   help="Behaviour rules to scan under. Default: the whole pool.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--skip-discrimination", action="store_true",
                   help="Pivotality scan only; skip the rollout-based "
                        "rule-discrimination pass.")
    args = p.parse_args()

    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    seeds = shift_corpora(cfg)["calib"][: args.n_shifts]
    rules = list(args.rules) if args.rules else resolve_pool(cfg)

    print(f"[perishability] scanning {len(seeds)} shifts x {len(rules)} behaviour rules")
    print(f"[perishability] shelf life ~ Tri{list(cfg.sim.order_attrs.shelf_life_triangular)}, "
          f"SLA ~ Tri{list(cfg.sim.order_attrs.sla_due_triangular)}, "
          f"interval = {cfg.sim.interval_minutes} min")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = scan(cfg, seeds, rules, args.n_jobs)
    summary = summarise(df)
    df.to_parquet(RESULTS_DIR / "pivotality_scan.parquet", index=False)

    disc_df, disc = (pd.DataFrame(), {})
    if not args.skip_discrimination:
        print("\n[perishability] rule-discrimination pass "
              "(does the CHOICE of rule move spoilage?)")
        disc_df, disc = rule_discrimination(cfg, seeds, rules, args.n_jobs)
        if not disc_df.empty:
            disc_df.to_parquet(RESULTS_DIR / "rule_discrimination.parquet", index=False)

    v = verdict(summary, disc)
    (RESULTS_DIR / "pivotality_summary.json").write_text(
        json.dumps(
            {"summary": summary, "discrimination": disc, "verdict": v},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )

    print("\n--- Reviewer 1, comment 1.d ---")
    print("  necessary conditions")
    print(f"    decisions with >=1 expiry-pivotal order          : "
          f"{summary['epoch_any_expiry_pivotal']:.1%}  "
          f"(>= {MATERIALITY_THRESHOLD:.0%})")
    print(f"    perishables whose expiry binds before their due  : "
          f"{summary['expiry_binds_rate']:.1%}  "
          f"(>= {EXPIRY_BINDS_THRESHOLD:.0%})")
    print("  binding condition")
    print(f"    decisions where the RULE CHOICE moves spoilage   : "
          f"{disc.get('epochs_where_rule_changes_spoilage', float('nan')):.1%}  "
          f"(>= {DISCRIMINATION_THRESHOLD:.0%})")
    print(f"    FEFO minimises spoilage, where contested         : "
          f"{disc.get('fefo_wins_spoilage_share', float('nan')):.1%}")
    print("  context")
    print(f"    queued perishables that are expiry-pivotal       : "
          f"{summary['expiry_pivotal_rate_perishables']:.1%}")
    print(f"    economic weight on expiry-pivotal orders         : "
          f"{summary['pivotal_value_share']:.1%}")
    print(f"    (reference) decisions with >=1 due-pivotal order : "
          f"{summary['epoch_any_due_pivotal']:.1%}")
    print(f"\n  claim supported: {v['perishability_claim_supported']}")
    print(f"  {v['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
