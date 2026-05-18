"""Five canonical dispatching heuristics — the action space of the ranker.

Each `rank` function takes a queue snapshot and a context dict, and returns a
new list ordered from highest priority (index 0) to lowest. Functions are pure
(do not mutate input) so the queue can be safely re-ranked by rollout labelers.

Context dict keys consumed:
  - t       : current decision-time (minutes from shift start) — used by CR, ATC
  - atc_k   : ATC lookahead parameter (default 2.0; see cfg.heuristics.atc_lookahead_k)

FEFO masking is handled OUTSIDE this module — at label time (rollout labeler)
and at inference time (switching controller). The heuristic itself always
ranks; it does not refuse to be invoked on non-perishable queues.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from simulation.orders import Order

PRIORITY_WEIGHTS: dict[str, float] = {"low": 1.0, "medium": 2.0, "high": 4.0}


def fifo(queue: list[Order], ctx: dict) -> list[Order]:
    return sorted(queue, key=lambda o: o.arrival_time)


def fefo(queue: list[Order], ctx: dict) -> list[Order]:
    return sorted(queue, key=lambda o: o.sla_due)


def wspt(queue: list[Order], ctx: dict) -> list[Order]:
    return sorted(
        queue,
        key=lambda o: -PRIORITY_WEIGHTS[o.priority_class] / max(o.processing_time, 1e-9),
    )


def cr(queue: list[Order], ctx: dict) -> list[Order]:
    """Critical Ratio = (sla_due - t - p) / p, ascending. Tie-break FIFO."""
    t = float(ctx["t"])

    def key(o: Order) -> tuple[float, float]:
        slack = o.sla_due - t - o.processing_time
        ratio = slack / max(o.processing_time, 1e-9)
        return (ratio, o.arrival_time)

    return sorted(queue, key=key)


def atc(queue: list[Order], ctx: dict) -> list[Order]:
    """Apparent Tardiness Cost (Vepsalainen & Morton 1987).

        I(j, t) = (w_j / p_j) * exp(-max(d_j - p_j - t, 0) / (k * p_bar))

    Higher I -> higher priority. We sort ascending on -I so index 0 is best.
    """
    if not queue:
        return []
    t = float(ctx["t"])
    k = float(ctx["atc_k"])
    p_bar = max(sum(o.processing_time for o in queue) / len(queue), 1e-9)

    def neg_score(o: Order) -> float:
        slack = max(o.sla_due - o.processing_time - t, 0.0)
        urgency = PRIORITY_WEIGHTS[o.priority_class] / max(o.processing_time, 1e-9)
        return -urgency * math.exp(-slack / (k * p_bar))

    return sorted(queue, key=neg_score)


HEURISTICS: dict[str, Callable[[list[Order], dict], list[Order]]] = {
    "FIFO": fifo,
    "FEFO": fefo,
    "WSPT": wspt,
    "CR": cr,
    "ATC": atc,
}

# Active heuristic pool. CR was dropped in Phase 2 — see HANDOFF §2.C and gotcha #16.
# `cr` itself stays defined so anyone needing the original 5-rule set can opt back in.
HEURISTIC_NAMES: list[str] = ["FIFO", "FEFO", "WSPT", "ATC"]
