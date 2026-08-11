"""Shift-level KPIs. Pure function of the shift's order population.

WHAT CHANGED, AND WHY (Reviewer 2, 1)
-------------------------------------
The submitted metric was

    sla_breach_rate = breaches / len(completed)

Orders still queued at shift end appeared in neither the numerator nor the
denominator; they were reported only as a bare `unfinished` count that no table
in the paper carried. Combined with `W_u = 0.005` against `W_b = 3.0`, a
controller could lower its reported breach rate by declining to touch hard
orders. In the submitted Table 1, DAHS completed 721.6 orders against FIFO's
750.6 — the shortfall was invisible to the headline metric.

Measured on the repository's own committed demo run logs (10 seeds, frozen
model, default configuration), counting every arrived order rather than every
completed one moves DAHS from 3.10% to 15.00%, FIFO from 11.75% to 17.97%, and
PPO from 9.40% to 16.44%. The ranking survives; the margin does not.

This module therefore reports the full outcome partition of every order that
arrived, and makes `service_failure_rate` the primary metric:

    arrived = served + unserved + dropped

    served    dispatched before the shift ended
    unserved  still in the queue at shift end
    dropped   refused at the door because the queue was at capacity

`sla_breach_rate_served` is retained under an explicit name so the submitted
numbers remain comparable, and because the reviewer asked for the formula to be
stated rather than for the old quantity to disappear.
"""

from __future__ import annotations

import numpy as np

from simulation.cost import CostWeights, potential
from simulation.orders import Order


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_kpis(
    completed: list[Order],
    queue: list[Order],
    dropped: list[Order],
    n_pickers: int,
    shift_minutes: float,
    n_intervals: int,
    weights: CostWeights | None = None,
) -> dict[str, float]:
    """Canonical KPI dict. All values float so equality comparisons are stable.

    Parameters
    ----------
    completed
        Orders dispatched during the shift (may finish after `shift_minutes`).
    queue
        Orders still waiting at shift end.
    dropped
        Orders refused at arrival because the queue was at capacity. These are
        genuine demand and are counted as arrived; omitting them would recreate
        the loophole this module exists to close.
    weights
        Objective weights. Defaults to `CostWeights()`.

    Keys
    ----
    service_failure_rate   PRIMARY. (late or spoiled) / arrived, over every
                           arrived order, served or not.
    sla_breach_rate_arrived  late / arrived — the direct answer to Reviewer 2 (1).
    sla_breach_rate_served   late / served — the submitted metric, kept for
                           comparability and explicitly named.
    spoilage_rate          spoiled perishables / arrived perishables.
    mean_tardiness         mean censored tardiness over arrived orders.
    mean_tardiness_served  mean tardiness over served orders only (submitted).
    throughput             count of served orders.
    unserved / dropped / arrived   the outcome partition.
    picker_utilization     picker-busy time within the shift / total capacity.
    composite_cost         objective value (see `simulation.cost`).
    mean_interval_cost     composite_cost / n_intervals.
    """
    w = weights if weights is not None else CostWeights()
    t_ref = float(shift_minutes)

    arrived: list[Order] = list(completed) + list(queue) + list(dropped)
    n_arrived = len(arrived)
    n_served = len(completed)

    if n_arrived == 0:
        return {
            "service_failure_rate": 0.0,
            "sla_breach_rate_arrived": 0.0,
            "sla_breach_rate_served": 0.0,
            "spoilage_rate": 0.0,
            "mean_tardiness": 0.0,
            "mean_tardiness_served": 0.0,
            "throughput": 0.0,
            "unserved": float(len(queue)),
            "dropped": float(len(dropped)),
            "arrived": 0.0,
            "picker_utilization": 0.0,
            "composite_cost": 0.0,
            "mean_interval_cost": 0.0,
        }

    n_failed = sum(1 for o in arrived if o.is_service_failure(t_ref))
    n_late = sum(1 for o in arrived if o.is_late(t_ref))
    n_late_served = sum(1 for o in completed if o.is_late(t_ref))

    perishables = [o for o in arrived if o.is_perishable]
    n_spoiled = sum(1 for o in perishables if o.is_spoiled(t_ref))

    tardiness_all = [o.tardiness(t_ref) for o in arrived]
    tardiness_served = [o.tardiness(t_ref) for o in completed]

    # Picker-busy time clipped to the shift window. An order dispatched near the
    # end keeps its picker busy past `shift_minutes`; only the in-shift portion
    # counts toward utilisation.
    busy = sum(
        min(o.finish_time, shift_minutes) - max(o.start_time, 0.0)
        for o in completed
        if o.start_time is not None and o.start_time < shift_minutes
    )
    capacity = n_pickers * shift_minutes

    cost = potential(arrived, queue_size=len(queue), t_ref=t_ref, w=w)

    return {
        "service_failure_rate": _safe_div(n_failed, n_arrived),
        "sla_breach_rate_arrived": _safe_div(n_late, n_arrived),
        "sla_breach_rate_served": _safe_div(n_late_served, n_served),
        "spoilage_rate": _safe_div(n_spoiled, len(perishables)),
        "mean_tardiness": float(np.mean(tardiness_all)),
        "mean_tardiness_served": float(np.mean(tardiness_served)) if completed else 0.0,
        "throughput": float(n_served),
        "unserved": float(len(queue)),
        "dropped": float(len(dropped)),
        "arrived": float(n_arrived),
        "picker_utilization": _safe_div(busy, capacity),
        "composite_cost": float(cost),
        "mean_interval_cost": _safe_div(cost, max(n_intervals, 1)),
    }
