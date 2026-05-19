"""demo/build_run_log.py -- faithful side-by-side event log for the browser demo.

Runs the REAL DAHS controller AND a chosen baseline on the SAME seed (identical
pre-generated order stream) and emits a deterministic JSON log that the
demo/dahs-app dashboard replays as a side-by-side comparison. The browser does
not simulate anything -- it only replays this log.

Faithful
    - Both policies drive simulation.warehouse_env.WarehouseEnv (the exact
      Phase-1 simulator). DAHS = baselines.ours; the baseline is built by
      experiments.evaluate._build_policy, the same factory the paper's E2
      evaluation uses.
    - The per-interval loop matches experiments.evaluate.run_shift /
      run_shift_env_aware: the decision state is env.current_state() read
      BEFORE env.step(). Final KPIs are computed by simulation.kpis.compute_kpis
      and the order partition is asserted, so each floor IS a real eval shift.
    - Picker assignment is reconstructed by replaying the env's own greedy
      argmin over the dispatch-ordered completions (exact; no frozen code is
      modified).

Reproducible
    A fixed --seed gives a byte-identical log (orders are pre-generated from one
    seeded np.random.default_rng; model artifacts are frozen).

Run:  python demo/build_run_log.py --seed 42 --baseline fefo
      python demo/build_run_log.py --seed 7  --baseline greedy_mpc --stdout

By default the log is written to demo/dahs-app/runs/run_<seed>_<baseline>.json,
where the static dashboard fetches it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baselines.ours import load_ours  # noqa: E402
from experiments.evaluate import _build_policy  # noqa: E402
from simulation.heuristics import HEURISTIC_NAMES  # noqa: E402
from simulation.kpis import compute_kpis  # noqa: E402
from simulation.warehouse_env import WarehouseEnv  # noqa: E402

RUNS_DIR = REPO / "demo" / "dahs-app" / "runs"

# Selectable baselines, with the source each rule / method comes from. Keys map
# to experiments.evaluate._build_policy. Artifact-backed methods are flagged.
BASELINES: dict[str, dict] = {
    "fifo": {"label": "FIFO",
             "source": "First-In-First-Out - classical dispatching rule",
             "blurb": "Serves orders strictly in arrival order; deadline-blind."},
    "fefo": {"label": "FEFO",
             "source": "First-Expire-First-Out - classical deadline-aware rule",
             "blurb": "Serves the earliest due-date first; the strongest static rule here."},
    "wspt": {"label": "WSPT",
             "source": "Weighted Shortest Processing Time - Smith (1956)",
             "blurb": "Ranks by priority weight over processing time."},
    "atc": {"label": "ATC",
            "source": "Apparent Tardiness Cost - Vepsalainen & Morton (1987)",
            "blurb": "Slack-and-processing composite urgency index."},
    "greedy_mpc": {"label": "Greedy MPC",
                   "source": "One-step-lookahead oracle proxy - this work (cf. Bertsekas, 2020)",
                   "blurb": "Each interval, simulates every rule one step and picks the cheapest."},
    "snapshot_xgb": {"label": "Snapshot-XGB (tau=1)",
                     "source": "DAHS rollout-horizon ablation - this work",
                     "blurb": "DAHS pipeline with the rollout horizon collapsed to one step.",
                     "needs": "runs/phase4_tau1"},
    "linucb": {"label": "LinUCB",
               "source": "Contextual bandit - Li, Chu, Langford & Schapire (2010)",
               "blurb": "Per-arm ridge regression with UCB exploration."},
    "ppo_fair": {"label": "PPO",
                 "source": "Proximal Policy Optimization - Schulman et al. (2017)",
                 "blurb": "Deep RL policy trained under a budget matched to DAHS.",
                 "needs": "runs/ppo_fair"},
}

ENT_MAX = math.log(len(HEURISTIC_NAMES))


def _classify(env: WarehouseEnv, order_picker: dict[int, int]) -> tuple[list, dict]:
    """Resolve every pre-generated order to its final outcome + picker."""
    completed_ids = {o.order_id for o in env.completed}
    queued_ids = {o.order_id for o in env.queue}
    orders = []
    for o in env._all_orders:
        if o.order_id in completed_ids:
            breached = bool(o.finish_time > o.sla_due)
            outcome = "breached" if breached else "shipped"
            spoiled = breached and o.is_perishable
        elif o.order_id in queued_ids:
            outcome, breached, spoiled = "unfinished", False, False
        else:
            outcome, breached, spoiled = "dropped", False, False
        orders.append({
            "id": o.order_id,
            "arrival": round(float(o.arrival_time), 4),
            "proc": round(float(o.processing_time), 4),
            "due": round(float(o.sla_due), 4),
            "perishable": bool(o.is_perishable),
            "priority": o.priority_class,
            "start": round(float(o.start_time), 4) if o.start_time is not None else None,
            "finish": round(float(o.finish_time), 4) if o.finish_time is not None else None,
            "picker": order_picker.get(o.order_id),
            "outcome": outcome,
            "breached": breached,
            "spoiled": bool(spoiled),
        })
    counts = {
        "shipped": sum(1 for o in orders if o["outcome"] == "shipped"),
        "breached": sum(1 for o in orders if o["outcome"] == "breached"),
        "unfinished": sum(1 for o in orders if o["outcome"] == "unfinished"),
        "dropped": sum(1 for o in orders if o["outcome"] == "dropped"),
        "spoiled": sum(1 for o in orders if o["spoiled"]),
    }
    assert counts["shipped"] + counts["breached"] == len(env.completed)
    assert counts["unfinished"] == len(env.queue)
    assert sum(counts[k] for k in ("shipped", "breached", "unfinished", "dropped")) == len(orders)
    return orders, counts


def run_one(seed: int, cfg, policy, env_aware: bool) -> dict:
    """Drive one WarehouseEnv(seed) to completion under `policy`; log everything.

    Mirrors experiments.evaluate.run_shift (state-based) and run_shift_env_aware
    (env-based). `policy.controller` (present for the DAHS stack) is read for the
    rule-probability vector and the switch-decision trace.
    """
    if hasattr(policy, "reset"):
        policy.reset()
    env = WarehouseEnv(seed, cfg)
    ctrl = getattr(policy, "controller", None)        # SwitchingController or None
    regime_gmm = getattr(policy, "regime_gmm", None)

    picker_free = [0.0] * env.n_pickers
    order_picker: dict[int, int] = {}
    intervals, switch_log = [], []
    prev_rule: str | None = None
    prev_completed = 0

    for i in range(env.n_intervals):
        t_start = env.t
        dwell_before = int(ctrl.dwell_remaining) if ctrl is not None else None

        if env_aware:
            rule = policy(env)                        # greedy_mpc / linucb
            probs, regime = None, None
        else:
            state = env.current_state()               # PRE-pull state == run_shift
            rule = policy(state)
            probs = (ctrl.last_probs.tolist()
                     if (ctrl is not None and ctrl.last_probs is not None) else None)
            regime = (int(np.argmax(regime_gmm.predict_proba(state.reshape(1, -1))[0]))
                      if regime_gmm is not None else None)

        env.step(rule)

        newly = env.completed[prev_completed:]
        for o in newly:
            pidx = int(np.argmin(picker_free))
            picker_free[pidx] = o.finish_time
            order_picker[o.order_id] = pidx

        switched = prev_rule is not None and rule != prev_rule
        candidate = entropy = reason = None
        if probs is not None:
            candidate = HEURISTIC_NAMES[int(np.argmax(probs))]
            entropy = float(-sum(p * math.log(p) for p in probs if p > 0))
            if prev_rule is None:
                reason = "initial selection"
            elif switched:
                reason = ("minimum-dwell elapsed: moved to the arg-max rule"
                          if dwell_before is not None and dwell_before <= 0
                          else "entropy gate fired: distribution confident, dwell overridden")
            elif candidate != rule and dwell_before and dwell_before > 0:
                reason = f"held by minimum-dwell (arg-max preferred {candidate})"
            else:
                reason = "arg-max rule unchanged"

        rec = {
            "idx": i,
            "tStart": round(float(t_start), 4),
            "tEnd": round(float(env.t), 4),
            "rule": rule,
            "probs": [round(float(p), 6) for p in probs] if probs is not None else None,
            "regime": regime,
            "dwell": dwell_before,
            "switched": switched,
            "candidate": candidate,
            "entropy": round(entropy, 6) if entropy is not None else None,
            "reason": reason,
            "fromRule": prev_rule,
            "nArrived": int(env._n_arrivals_this_interval),
            "nDispatched": len(newly),
            "queueLenEnd": int(env._history_queue_length[i]),
            "breachRateEnd": round(float(env._history_breach_rate[i]), 6),
        }
        intervals.append(rec)
        switch_log.append({k: rec[k] for k in
                           ("idx", "tStart", "rule", "fromRule", "switched",
                            "candidate", "reason", "probs", "dwell", "entropy",
                            "regime", "queueLenEnd")})
        prev_rule = rule
        prev_completed = len(env.completed)

    orders, counts = _classify(env, order_picker)
    kpis = compute_kpis(env.completed, env.queue, env.n_pickers, env.shift_minutes)
    return {
        "orders": orders,
        "intervals": intervals,
        "switchLog": switch_log,
        "counts": counts,
        "kpis": {k: round(float(v), 6) for k, v in kpis.items()},
    }


def build(seed: int, baseline_key: str) -> dict:
    if baseline_key not in BASELINES:
        raise SystemExit(f"unknown baseline '{baseline_key}'. "
                         f"choices: {', '.join(BASELINES)}")
    cfg = OmegaConf.load(REPO / "config.yaml")

    dahs_policy, dahs_env_aware = _build_policy("ours", None)
    base_policy, base_env_aware = _build_policy(baseline_key, None)

    dahs = run_one(seed, cfg, dahs_policy, dahs_env_aware)
    baseline = run_one(seed, cfg, base_policy, base_env_aware)

    return {
        "meta": {
            "seed": int(seed),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "shiftMinutes": float(cfg.sim.shift_hours * 60),
            "intervalMinutes": float(cfg.sim.interval_minutes),
            "nIntervals": int(round(cfg.sim.shift_hours * 60 / cfg.sim.interval_minutes)),
            "nPickers": int(cfg.sim.n_pickers),
            "queueCapacity": int(cfg.sim.queue_capacity),
            "arrivalMode": str(cfg.sim.arrivals.get("arrival_mode", "poisson")),
            "arrivalRate": float(cfg.sim.arrivals.base_rate_per_minute),
            "heuristics": list(HEURISTIC_NAMES),
            "nOrders": len(dahs["orders"]),
            "baseline": {
                "key": baseline_key,
                "label": BASELINES[baseline_key]["label"],
                "source": BASELINES[baseline_key]["source"],
                "blurb": BASELINES[baseline_key]["blurb"],
            },
        },
        "dahs": dahs,
        "baseline": baseline,
    }


def main() -> int:
    cfg = OmegaConf.load(REPO / "config.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=int(cfg.seeds.sim),
                        help="WarehouseEnv seed (default: cfg.seeds.sim = 42).")
    parser.add_argument("--baseline", type=str, default="fefo",
                        choices=list(BASELINES),
                        help="Baseline policy to race DAHS against.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path "
                             "(default: demo/dahs-app/runs/run_<seed>_<baseline>.json).")
    parser.add_argument("--stdout", action="store_true",
                        help="Print the JSON to stdout instead of writing a file.")
    args = parser.parse_args()

    log = build(args.seed, args.baseline)
    payload = json.dumps(log, separators=(",", ":"))

    out = args.out or (RUNS_DIR / f"run_{args.seed}_{args.baseline}.json")
    if args.stdout:
        sys.stdout.write(payload)
        sys.stdout.flush()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")

    # human summary -> stderr (keeps stdout clean for --stdout consumers)
    m = log["meta"]
    d, b = log["dahs"], log["baseline"]
    log_line = (
        f"[run-log] seed={m['seed']}  DAHS vs {m['baseline']['label']}  "
        f"({m['nOrders']} orders, {m['nIntervals']} intervals)\n"
        f"  DAHS      sla_breach={d['kpis']['sla_breach_rate']:.4f}  "
        f"shipped={d['counts']['shipped']}  breached={d['counts']['breached']}  "
        f"unfinished={d['counts']['unfinished']}\n"
        f"  {m['baseline']['label']:<9s} sla_breach={b['kpis']['sla_breach_rate']:.4f}  "
        f"shipped={b['counts']['shipped']}  breached={b['counts']['breached']}  "
        f"unfinished={b['counts']['unfinished']}"
    )
    sys.stderr.write(log_line + "\n")
    if not args.stdout:
        sys.stderr.write(f"  wrote {out.relative_to(REPO)} ({len(payload) / 1024:.0f} KB)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
