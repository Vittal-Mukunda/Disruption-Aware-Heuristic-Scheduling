"""How much does the feature map lose relative to the true state?

REVIEWER 2, COMMENT 4
---------------------
    "The authors treat the 25-dimensional feature vector as the system state.
     However, looking at the features, most are aggregate statistics. They fail
     to capture the exact processing time, due date, arrival time, and
     perishability status of individual queued orders. Consequently, two entirely
     different queue configurations could map to the exact same 25-D vector while
     exhibiting completely different transition dynamics and costs under the same
     dispatching rule. Strictly speaking, this 25-D vector is an observation, not
     a Markov state. The paper needs to distinguish the true full state S_t from
     its feature representation x_t = phi(S_t). The authors must either
     rigorously justify why x_t acts as a sufficient statistic or reframe the
     problem as a Partially Observable Markov Decision Process."

The reviewer is correct, the submitted paper was wrong to call `phi(S_t)` the
state, and we do not attempt the sufficiency justification — it is false. `phi`
records marginal summaries (mean and standard deviation of slack, mean processing
time, counts) and discards the JOINT distribution over per-order attributes, which
is exactly what determines what a ranking rule does next. Section 3 reframes the
problem as a POMDP and DAHS as a policy-function approximation over `phi`.

What is left is a quantitative question the reviewer's phrasing implies but does
not ask outright: *how much* does the aliasing cost? A feature map can be
formally insufficient and practically adequate, or formally insufficient and
badly lossy, and those are different papers. This module measures it two ways.

1. CONSTRUCTIVE DEMONSTRATION — `aliasing_witness()`
   Searches for a pair of queues with (numerically) identical `phi` whose
   tau-step costs under the SAME rule differ. This is the reviewer's hypothetical
   made concrete, and it is self-verifying: the routine asserts the feature
   vectors match to tolerance before reporting the cost gap, so the witness
   cannot be an artefact of a sloppy comparison. The mechanism it exploits is
   that `phi` fixes the mean and standard deviation of slack and the mean
   processing time, but not their pairing — whether the tight deadline sits on
   the long order or the short one is invisible to the controller and decisive
   for the outcome.

2. EMPIRICAL ALIASING RATE — `empirical_aliasing()`
   Over the training corpus, finds mutual near-neighbours in standardised `phi`
   space and asks how often two states that look nearly identical to the
   controller have different cost-minimising rules, and what following the wrong
   one costs. This is the operationally relevant number: it upper-bounds how much
   of the residual regret is attributable to partial observability rather than to
   the learner, and it belongs in the limitations section as a measured quantity
   rather than a caveat.

    python -m experiments.observability_analysis --train data/train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from simulation.cost import CostWeights
from simulation.heuristics import resolve_pool, with_default_scales
from simulation.orders import Order
from simulation.state_extractor import FEATURE_NAMES
from simulation.warehouse_env import WarehouseEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "S1_observability"


# ---------------------------------------------------------------------------
# 1. Constructive witness
# ---------------------------------------------------------------------------


def _env_with_queue(cfg: DictConfig, queue: list[Order]) -> WarehouseEnv:
    """An env holding exactly `queue`, with no further arrivals.

    Future arrivals are suppressed so the two members of a witness pair differ
    only in queue composition. Any cost gap is then attributable to the queue,
    not to a difference in the exogenous stream.
    """
    env = WarehouseEnv(0, cfg)
    env._all_orders = list(queue)
    env._next_order_idx = len(queue)
    env.queue = list(queue)
    env.completed = []
    env.dropped = []
    env.picker_busy_until = [0.0] * env.n_pickers
    env.t = 0.0
    env.interval_idx = 0
    env._n_arrivals_last_interval = len(queue)
    env._history_queue_length = []
    env._history_failure_rate = []
    return env


def _mk(oid: int, p: float, slack: float, shelf: float | None = None) -> Order:
    """An order waiting at t=0 with processing time `p` and slack `d - t - p`."""
    return Order(
        order_id=oid,
        arrival_time=0.0,
        processing_time=p,
        sla_due=slack + p,
        is_perishable=shelf is not None,
        priority_class="low",
        expiry_time=shelf,
    )


def aliasing_witness(
    cfg: DictConfig, rule: str, tau: int, tol: float = 1e-9
) -> dict:
    """Two queues with identical `phi`, different cost under the same rule.

    The construction swaps which order carries the tight deadline while holding
    every summary statistic `phi` records fixed:

        queue A:  (p = p_short, slack = s_loose), (p = p_long, slack = s_tight)
        queue B:  (p = p_short, slack = s_tight), (p = p_long, slack = s_loose)

    Both queues have the same length, the same arrival times and hence the same
    ages, the same mean and standard deviation of slack, and the same mean
    processing time. `phi` cannot tell them apart. The dynamics can: in A the
    binding deadline sits on the order that occupies a picker longest.

    Parameter values are searched over a small grid rather than hard-coded, so
    the witness is found rather than asserted, and the routine verifies that the
    feature vectors really do coincide before reporting anything.
    """
    weights = CostWeights.from_cfg(cfg)
    best: dict | None = None

    for p_short, p_long in ((2.0, 12.0), (3.0, 14.0), (2.0, 20.0), (4.0, 18.0)):
        for s_tight, s_loose in ((0.0, 40.0), (2.0, 60.0), (-5.0, 30.0)):
            qa = [_mk(0, p_short, s_loose), _mk(1, p_long, s_tight)]
            qb = [_mk(0, p_short, s_tight), _mk(1, p_long, s_loose)]

            ea, eb = _env_with_queue(cfg, qa), _env_with_queue(cfg, qb)
            fa, fb = ea.observe(), eb.observe()
            gap = float(np.abs(fa - fb).max())
            if gap > tol:
                continue  # phi differs; not a witness

            phi0_a, phi0_b = ea.potential(), eb.potential()
            ea.run_with_policy(rule, n_steps=tau)
            eb.run_with_policy(rule, n_steps=tau)
            cost_a = ea.potential() - phi0_a
            cost_b = eb.potential() - phi0_b

            if best is None or abs(cost_a - cost_b) > abs(best["cost_gap"]):
                best = {
                    "rule": rule,
                    "tau": int(tau),
                    "p_short": p_short, "p_long": p_long,
                    "slack_tight": s_tight, "slack_loose": s_loose,
                    "phi_max_abs_diff": gap,
                    "cost_A": float(cost_a),
                    "cost_B": float(cost_b),
                    "cost_gap": float(cost_a - cost_b),
                }

    if best is None:
        return {"found": False}

    # Self-verification: the claim is only meaningful if phi really did coincide.
    assert best["phi_max_abs_diff"] <= tol, "witness search returned unequal phi"
    best["found"] = True
    best["interpretation"] = (
        "Two queues indistinguishable to the controller incur different cost "
        "under the same rule. phi is therefore not a sufficient statistic, and "
        "no amount of training data can recover the difference from phi alone."
    )
    return best


# ---------------------------------------------------------------------------
# 2. Empirical aliasing over the corpus
# ---------------------------------------------------------------------------


def empirical_aliasing(
    df: pd.DataFrame, pool: list[str], k: int, radius_pct: float
) -> dict:
    """How often do near-identical observations disagree about the best rule?

    Neighbours are taken in standardised `phi` space, and "near-identical" is a
    percentile of the observed nearest-neighbour distance distribution rather than
    an absolute threshold, so the definition does not depend on feature scaling
    choices. For each state we compare its cost-minimising rule against those of
    its k nearest neighbours, and record the regret from adopting the neighbour's
    choice — the price the controller pays for being unable to tell them apart.
    """
    feat_cols = [f"f_{n}" for n in FEATURE_NAMES if f"f_{n}" in df.columns]
    cost_cols = [f"cost_{h}" for h in pool]
    X = df[feat_cols].to_numpy(np.float64)
    C = df[cost_cols].to_numpy(np.float64)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scipy is required for the neighbour search") from exc

    tree = cKDTree(Z)
    # k+1 because the first neighbour of a point is itself.
    dist, idx = tree.query(Z, k=min(k + 1, len(Z)))
    dist, idx = dist[:, 1:], idx[:, 1:]

    thresh = float(np.percentile(dist[:, 0], radius_pct))
    close = dist <= thresh

    best = C.argmin(axis=1)
    own_cost = C[np.arange(len(C)), best]

    disagree, regret = [], []
    for i in range(len(Z)):
        for j_pos in np.where(close[i])[0]:
            j = idx[i, j_pos]
            disagree.append(bool(best[i] != best[j]))
            # Cost of acting on the neighbour's preferred rule at state i.
            regret.append(float(C[i, best[j]] - own_cost[i]))

    disagree_arr = np.asarray(disagree, dtype=bool)
    regret_arr = np.asarray(regret, dtype=np.float64)
    total_spread = float((C.max(axis=1) - C.min(axis=1)).mean())

    return {
        "n_states": int(len(Z)),
        "k_neighbours": int(k),
        "radius_percentile": float(radius_pct),
        "radius_standardised": thresh,
        "n_pairs": int(disagree_arr.size),
        "aliasing_rate": float(disagree_arr.mean()) if disagree_arr.size else 0.0,
        "mean_regret": float(regret_arr.mean()) if regret_arr.size else 0.0,
        "mean_regret_given_disagreement": (
            float(regret_arr[disagree_arr].mean()) if disagree_arr.any() else 0.0
        ),
        "mean_cost_spread_across_rules": total_spread,
        # The share of the achievable rule-choice benefit that partial
        # observability puts out of reach. This is the number for Section 8.
        "aliasing_share_of_achievable_benefit": (
            float(regret_arr[disagree_arr].mean() / total_spread)
            if disagree_arr.any() and total_spread > 0 else 0.0
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=REPO_ROOT / "data" / "train.parquet")
    p.add_argument("--rule", type=str, default=None,
                   help="Rule for the constructive witness. Default: first in pool.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--radius-pct", type=float, default=10.0,
                   help="Percentile of nearest-neighbour distance defining "
                        "'near-identical'.")
    args = p.parse_args()

    cfg = with_default_scales(OmegaConf.load(CONFIG_PATH))
    pool = resolve_pool(cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # --- 1. constructive witness ---
    rule = args.rule or pool[0]
    witness = aliasing_witness(cfg, rule, int(cfg.labeling.tau))
    report["witness"] = witness
    if witness.get("found"):
        print(f"[observability] constructive witness under {rule}, tau={witness['tau']}")
        print(f"  phi max |A - B|   = {witness['phi_max_abs_diff']:.2e}  (identical)")
        print(f"  cost(queue A)     = {witness['cost_A']:.4f}")
        print(f"  cost(queue B)     = {witness['cost_B']:.4f}")
        print(f"  gap               = {witness['cost_gap']:+.4f}")
    else:
        print("[observability] no witness found on the search grid; widen it "
              "before concluding phi is sufficient.")

    # --- 2. empirical aliasing ---
    if args.train.exists():
        df = pd.read_parquet(args.train)
        emp = empirical_aliasing(df, pool, args.k, args.radius_pct)
        report["empirical"] = emp
        print(f"\n[observability] empirical aliasing over {emp['n_states']:,} states")
        print(f"  near-identical pairs                 : {emp['n_pairs']:,}")
        print(f"  disagree on the best rule            : {emp['aliasing_rate']:.1%}")
        print(f"  mean regret | disagreement           : "
              f"{emp['mean_regret_given_disagreement']:.4f}")
        print(f"  as a share of achievable benefit     : "
              f"{emp['aliasing_share_of_achievable_benefit']:.1%}")
    else:
        print(f"\n[observability] {args.train} not found — witness only. "
              f"Run Stage 2 labelling for the empirical aliasing rate.")

    (RESULTS_DIR / "observability.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
