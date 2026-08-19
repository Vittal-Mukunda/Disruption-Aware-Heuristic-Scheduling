"""The dispatching-rule library — the action set of the selector.

Each `rank` function takes a queue snapshot and a context dict and returns a new
list ordered from highest priority (index 0) to lowest. Functions are pure (they
do not mutate the input) so the queue can be safely re-ranked by rollout
labellers and by the online-MPC baseline.

Context keys consumed:
  t          current decision time (minutes from shift start)
  atc_k      ATC look-ahead scale — CALIBRATED, see `experiments/calibrate_rules.py`
  covert_k   COVERT look-ahead scale — calibrated the same way

TIE-BREAKING. `sorted` is stable and the queue is maintained in arrival order,
so ties resolve first-come-first-served. This is deterministic and is the
documented convention; no rule needs an explicit secondary key.

WHY THESE RULES (Reviewer 1, 4.a)
---------------------------------
The pool spans the four information sources a dispatcher can key on, so that
"no single rule dominates" is a structural statement about the design space
rather than an empirical accident:

  arrival only ............ FIFO                        (no-information reference)
  customer deadline ....... EDD, EEDD, MS, MDD          (Jackson 1955; Conway,
                                                         Maxwell & Miller 1967;
                                                         Baker & Bertrand 1982)
  both clocks ............. EEDD = min(d, x)            (once two deadlines exist)
  product deadline ........ FEFO                        (the only expiry-only rule)
  processing + weight ..... WSPT                        (Smith 1955)
  deadline x processing ... ATC, COVERT                 (Vepsalainen & Morton 1987;
                                                         Carroll 1965)

FIFO is retained deliberately as the zero-information control (Reviewer 1, 4.b
asks what it adds in a due-date setting). If the Stage-1 screen shows it never
wins a decision under the corrected model, it is dropped and that is reported —
in the submitted version FIFO was the arg-min in 0 of 865 filtered test states,
which is itself a finding.

WSPT IS THE k -> inf LIMIT OF ATC (Reviewer 1, 4.c)
---------------------------------------------------
In `atc`, `exp(-slack / (k * p_bar)) -> 1` as `k -> inf`, leaving exactly the
WSPT key `w/p`. A correctly calibrated ATC therefore cannot be beaten by WSPT.
The submitted paper used an uncalibrated `atc_lookahead_k = 2.0` and reported
WSPT winning 32% of decisions against ATC's 10%, which is the signature of a
mis-set k rather than a property of the rules. `experiments/calibrate_rules.py`
now calibrates k twice, exactly as the reviewer requests: once for standalone
use (ATC is itself a benchmark) and once for portfolio contribution (ATC inside
the selector's pool).

FEFO masking is handled OUTSIDE this module — at label time and at inference
time. A rule always ranks; it never refuses to be invoked.
"""

from __future__ import annotations

import math

import numpy as np
from collections.abc import Callable

from simulation.orders import Order

# Re-exported for callers that historically imported it from here.
from simulation.orders import PRIORITY_WEIGHTS  # noqa: F401

# Sentinel used to sort non-perishable orders behind every perishable one under
# FEFO. Finite, so arithmetic on the key never produces a NaN.
_NO_EXPIRY: float = float(1e12)


def _p_bar(queue: list[Order]) -> float:
    """Mean processing time of the queue — the scale unit for ATC and COVERT."""
    return max(sum(o.processing_time for o in queue) / len(queue), 1e-9)


# ---------------------------------------------------------------------------
# Arrival-driven
# ---------------------------------------------------------------------------


def fifo(queue: list[Order], ctx: dict) -> list[Order]:
    """First-in-first-out by arrival. The zero-information control."""
    return sorted(queue, key=lambda o: o.arrival_time)


# ---------------------------------------------------------------------------
# Customer-deadline-driven
# ---------------------------------------------------------------------------


def edd(queue: list[Order], ctx: dict) -> list[Order]:
    """Earliest due date (Jackson 1955). Ascending `sla_due`.

    This is the rule the submitted paper called "FEFO". It is a *due-date* rule
    and has nothing to do with perishability (Reviewer 1, 1.f). It keeps its
    results and its role in the pool under the correct name.
    """
    return sorted(queue, key=lambda o: o.sla_due)


def ms(queue: list[Order], ctx: dict) -> list[Order]:
    """Minimum slack (Conway, Maxwell & Miller 1967). Ascending `d - t - p`."""
    t = float(ctx["t"])
    return sorted(queue, key=lambda o: o.sla_due - t - o.processing_time)


def mdd(queue: list[Order], ctx: dict) -> list[Order]:
    """Modified due date (Baker & Bertrand 1982). Ascending `max(d, t + p)`.

    Behaves as EDD while the queue is comfortable and degrades gracefully to SPT
    once orders are already past due — the modification exists precisely for the
    saturated regime where EDD chases unrecoverable orders.
    """
    t = float(ctx["t"])
    return sorted(queue, key=lambda o: max(o.sla_due, t + o.processing_time))


# ---------------------------------------------------------------------------
# Product-deadline-driven
# ---------------------------------------------------------------------------


def eedd(queue: list[Order], ctx: dict) -> list[Order]:
    """Earliest EFFECTIVE due date: ascending `min(sla_due, expiry_time)`.

    The only rule in the library that reads BOTH clocks. It exists because
    without it the pool cannot answer Reviewer 1 (1.d) fairly.

    `Order.effective_deadline()` was added in this revision and, until now, no
    rule called it. The two expiry-relevant rules each ignored half the model:
    EDD sorts on the customer deadline and never looks at expiry, while FEFO
    sorts perishables by expiry and dumps every non-perishable behind them
    regardless of how late they are. On a corpus that is ~80% non-perishable
    FEFO is close to a strawman, so concluding "no expiry-aware rule earns a
    slot in the pool" from FEFO's screening failure would not be a fair test of
    whether the product clock matters.

    EEDD is the rule a scheduler would actually write once told orders have two
    deadlines: treat each order by whichever of its own clocks binds first. If
    this rule also fails to earn a slot, the conclusion that expiry-awareness
    buys nothing at this operating point is a real result rather than an
    artefact of a badly-chosen candidate.
    """
    return sorted(queue, key=lambda o: o.effective_deadline())


def fefo(queue: list[Order], ctx: dict) -> list[Order]:
    """First-expire-first-out — genuinely expiry-aware.

    Perishables are ordered by `expiry_time`; non-perishables carry no product
    clock and sort behind all of them. This is the only rule in the pool that
    reads `expiry_time`, which is what earns it a slot: under the two-clock
    order model of `simulation.orders` it is not a relabelling of EDD.
    """
    return sorted(
        queue,
        key=lambda o: o.expiry_time if o.expiry_time is not None else _NO_EXPIRY,
    )


# ---------------------------------------------------------------------------
# Processing-time and composite
# ---------------------------------------------------------------------------


def wspt(queue: list[Order], ctx: dict) -> list[Order]:
    """Weighted shortest processing time (Smith 1955). Descending `w / p`."""
    return sorted(queue, key=lambda o: -o.weight / max(o.processing_time, 1e-9))


def atc(queue: list[Order], ctx: dict) -> list[Order]:
    """Apparent Tardiness Cost (Vepsalainen & Morton 1987).

        I(o, t) = (w_o / p_o) * exp(-max(d_o - p_o - t, 0) / (k * p_bar))

    Descending `I`. `k` controls how far ahead urgency is discounted: small `k`
    collapses ATC onto a pure urgency rule, large `k` collapses it onto WSPT.
    It is calibrated, not assumed.
    """
    if not queue:
        return []
    t = float(ctx["t"])
    if ctx.get("atc_k") is None:
        raise ValueError(
            "ATC requires a calibrated look-ahead scale, but "
            "cfg.heuristics.atc_lookahead_k is unset. Run "
            "`python -m experiments.calibrate_rules calibrate` and write the "
            "fitted value into config.yaml before using ATC. The submitted "
            "paper's uncalibrated k=2.0 is what produced WSPT beating ATC, "
            "which cannot happen for a fitted k (Section 3.6)."
        )
    k = float(ctx["atc_k"])
    pbar = _p_bar(queue)
    scale = max(k * pbar, 1e-9)

    def neg_index(o: Order) -> float:
        slack = max(o.sla_due - o.processing_time - t, 0.0)
        urgency = o.weight / max(o.processing_time, 1e-9)
        return -urgency * math.exp(-slack / scale)

    return sorted(queue, key=neg_index)


def covert(queue: list[Order], ctx: dict) -> list[Order]:
    """Cost OVER Time (Carroll 1965).

        C(o, t) = (w_o / p_o) * max(0, 1 - max(d_o - p_o - t, 0) / (k_c * p_bar))

    Descending `C`. The linear sibling of ATC: it truncates to zero once slack
    exceeds `k_c * p_bar`, so orders that cannot yet become tardy are ignored
    entirely rather than exponentially discounted.
    """
    if not queue:
        return []
    t = float(ctx["t"])
    if ctx.get("covert_k") is None:
        raise ValueError(
            "COVERT requires a calibrated look-ahead scale, but "
            "cfg.heuristics.covert_lookahead_k is unset. Run "
            "`python -m experiments.calibrate_rules calibrate` first."
        )
    k = float(ctx["covert_k"])
    pbar = _p_bar(queue)
    scale = max(k * pbar, 1e-9)

    def neg_index(o: Order) -> float:
        slack = max(o.sla_due - o.processing_time - t, 0.0)
        urgency = o.weight / max(o.processing_time, 1e-9)
        return -urgency * max(0.0, 1.0 - slack / scale)

    return sorted(queue, key=neg_index)


def cr(queue: list[Order], ctx: dict) -> list[Order]:
    """Critical ratio: `(d - t - p) / p`, ascending. Tie-break FIFO.

    Screened out of the deployed pool in the original pilot as structurally
    dominated. Retained in the library so the Stage-1 screen can re-test that
    decision under the corrected model rather than inherit it.
    """
    t = float(ctx["t"])

    def key(o: Order) -> tuple[float, float]:
        slack = o.sla_due - t - o.processing_time
        return (slack / max(o.processing_time, 1e-9), o.arrival_time)

    return sorted(queue, key=key)


def spt(queue: list[Order], ctx: dict) -> list[Order]:
    """Unweighted shortest processing time. Throughput reference for screening."""
    return sorted(queue, key=lambda o: o.processing_time)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HEURISTICS: dict[str, Callable[[list[Order], dict], list[Order]]] = {
    "FIFO": fifo,
    "EDD": edd,
    "EEDD": eedd,
    "FEFO": fefo,
    "WSPT": wspt,
    "ATC": atc,
    "MS": ms,
    "MDD": mdd,
    "COVERT": covert,
    "CR": cr,
    "SPT": spt,
}

# The nine rules that enter Stage-1 screening (Reviewer 1, 4.d). CR and SPT are
# in the library but out of the screen by default: CR was already screened out
# once, SPT exists as an unweighted throughput reference for the screening table.
SCREENING_POOL: list[str] = [
    "FIFO", "EDD", "EEDD", "FEFO", "WSPT", "ATC", "MS", "MDD", "COVERT",
]

# The deployed pool. Stage 1 overwrites `cfg.heuristics.pool` with the survivors
# of the screen; every entry point should call `resolve_pool(cfg)` rather than
# reading this module-level default, which exists only so that imports without a
# config in hand still work.
HEURISTIC_NAMES: list[str] = list(SCREENING_POOL)


def resolve_pool(cfg=None) -> list[str]:
    """The active rule pool: `cfg.heuristics.pool` if given, else the default.

    Validates every name against the library so a typo in the config fails at
    load time rather than producing a silently smaller action space.
    """
    if cfg is None:
        return list(HEURISTIC_NAMES)
    pool = [str(h) for h in cfg.heuristics.pool]
    unknown = [h for h in pool if h not in HEURISTICS]
    if unknown:
        raise ValueError(
            f"cfg.heuristics.pool contains unknown rules {unknown}. "
            f"Available: {sorted(HEURISTICS)}"
        )
    if not pool:
        raise ValueError("cfg.heuristics.pool is empty")
    return pool


def with_default_scales(cfg):
    """Fill any unset rule scale parameter with its grid median.

    Committed `config.yaml` carries the fitted scales (`3.0` / `4.0`). If a scale
    is missing, ATC/COVERT fail at step time rather than silently reusing 2.0.
    Stage-1 diagnostics call this to substitute the calibration-grid midpoint
    before those fitted values exist. Results produced that way are diagnostics,
    never reported rule benchmarks.
    """
    from omegaconf import OmegaConf, open_dict

    new = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    grid = [float(x) for x in cfg.heuristics.calibration.k_grid]
    mid = float(np.median(grid)) if grid else 2.0
    filled: list[str] = []
    with open_dict(new):
        for key in ("atc_lookahead_k", "covert_lookahead_k"):
            if new.heuristics.get(key) is None:
                new.heuristics[key] = mid
                filled.append(key)
    if filled:
        new_meta = {"provisional_scales": filled, "value": mid}
        with open_dict(new):
            new.heuristics["_provisional"] = new_meta
    return new


def rank(name: str, queue: list[Order], ctx: dict) -> list[Order]:
    """Dispatch to a rule by name, with a clear error on an unknown name."""
    try:
        fn = HEURISTICS[name]
    except KeyError:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {sorted(HEURISTICS)}"
        ) from None
    return fn(queue, ctx)
