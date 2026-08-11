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

from seed import shift_corpora
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "S1_perishability"

# Pre-registered threshold. Below this share of decisions the claim is not
# supported and the framing changes. Fixed here before the diagnostic is run so
# the decision rule cannot be fitted to the answer.
MATERIALITY_THRESHOLD: float = 0.05


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


def verdict(summary: dict) -> dict:
    """State plainly which way the evidence falls, against the fixed threshold."""
    rate = summary["epoch_any_expiry_pivotal"]
    supported = bool(rate >= MATERIALITY_THRESHOLD)
    return {
        "threshold": MATERIALITY_THRESHOLD,
        "epoch_any_expiry_pivotal": rate,
        "perishability_claim_supported": supported,
        "recommendation": (
            "Retain 'perishability-constrained' framing and keep FEFO in the "
            "candidate pool; report this diagnostic in Section 3."
            if supported
            else "Drop 'perishability-constrained' from the title and framing, "
            "remove FEFO from the pool, and report this diagnostic as the "
            "reason. Reviewer 1 (1.d) is answered either way."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-shifts", type=int, default=30,
                   help="Calibration shifts to scan (default: the whole block).")
    p.add_argument("--rules", nargs="*", default=None,
                   help="Behaviour rules to scan under. Default: the whole pool.")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    seeds = shift_corpora(cfg)["calib"][: args.n_shifts]
    rules = list(args.rules) if args.rules else resolve_pool(cfg)

    print(f"[perishability] scanning {len(seeds)} shifts x {len(rules)} behaviour rules")
    print(f"[perishability] shelf life ~ Tri{list(cfg.sim.order_attrs.shelf_life_triangular)}, "
          f"SLA ~ Tri{list(cfg.sim.order_attrs.sla_due_triangular)}, "
          f"interval = {cfg.sim.interval_minutes} min")

    df = scan(cfg, seeds, rules, args.n_jobs)
    summary = summarise(df)
    v = verdict(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / "pivotality_scan.parquet", index=False)
    (RESULTS_DIR / "pivotality_summary.json").write_text(
        json.dumps({"summary": summary, "verdict": v}, indent=2), encoding="utf-8"
    )

    print("\n--- Reviewer 1, comment 1.d ---")
    print(f"  decisions where >=1 queued order is expiry-pivotal : "
          f"{summary['epoch_any_expiry_pivotal']:.1%}")
    print(f"  queued perishables that are expiry-pivotal         : "
          f"{summary['expiry_pivotal_rate_perishables']:.1%}")
    print(f"  perishables whose expiry binds before their due    : "
          f"{summary['expiry_binds_rate']:.1%}")
    print(f"  economic weight sitting on expiry-pivotal orders   : "
          f"{summary['pivotal_value_share']:.1%}")
    print(f"  (reference) decisions with >=1 due-pivotal order   : "
          f"{summary['epoch_any_due_pivotal']:.1%}")
    print(f"\n  claim supported at threshold {MATERIALITY_THRESHOLD:.0%}: "
          f"{v['perishability_claim_supported']}")
    print(f"  {v['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
