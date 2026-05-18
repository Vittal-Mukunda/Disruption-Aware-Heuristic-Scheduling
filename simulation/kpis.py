"""Shift-level KPIs. Pure function of (completed, queue, n_pickers, shift_minutes)."""

from __future__ import annotations

import numpy as np

from simulation.orders import Order


def compute_kpis(
    completed: list[Order],
    queue: list[Order],
    n_pickers: int,
    shift_minutes: float,
) -> dict[str, float]:
    """Canonical KPI dict. All values float so equality comparisons are stable.

    Keys:
        sla_breach_rate     fraction of completed orders with finish_time > sla_due
        mean_tardiness      mean of max(finish - due, 0) over completed
        spoilage_rate       fraction of completed perishables that breached SLA
        throughput          count of completed (dispatched-by-shift-end) orders
        picker_utilization  total picker-busy time within shift / (n_pickers * shift_minutes)
        unfinished          count of orders still in queue at shift end
    """
    n = len(completed)
    if n == 0:
        return {
            "sla_breach_rate": 0.0,
            "mean_tardiness": 0.0,
            "spoilage_rate": 0.0,
            "throughput": 0.0,
            "picker_utilization": 0.0,
            "unfinished": float(len(queue)),
        }

    breaches = sum(1 for o in completed if o.finish_time > o.sla_due)
    tardiness = [max(o.finish_time - o.sla_due, 0.0) for o in completed]
    perish = [o for o in completed if o.is_perishable]
    perish_breaches = sum(1 for o in perish if o.finish_time > o.sla_due)

    busy = sum(
        min(o.finish_time, shift_minutes) - max(o.start_time, 0.0)
        for o in completed
        if o.start_time is not None and o.start_time < shift_minutes
    )
    total_capacity = n_pickers * shift_minutes

    return {
        "sla_breach_rate": breaches / n,
        "mean_tardiness": float(np.mean(tardiness)),
        "spoilage_rate": (perish_breaches / len(perish)) if perish else 0.0,
        "throughput": float(n),
        "picker_utilization": busy / total_capacity if total_capacity > 0 else 0.0,
        "unfinished": float(len(queue)),
    }
