"""25-feature state vector — extracted at every 15-min decision point.

CANONICAL FEATURE ORDER (must match HANDOFF.md §3.2 exactly):
   0: queue_length
   1: mean_queue_age
   2: max_queue_age
   3: pct_critical                 fraction of queue with (sla_due - t) < 30 min
   4: pct_perishable
   5: labor_utilization            fraction of pickers currently busy
   6: n_pickers_busy
   7: mean_pickup_time_recent      mean processing_time of orders finished in last 60 min
   8: n_orders_breached_so_far     completed orders with finish_time > sla_due (and finish_time<=t)
   9: n_orders_at_risk_30min       queue orders with sla_due-t<30 AND slack<0
  10: mean_slack_minutes           mean (sla_due - t - p) over queue
  11: arrival_rate_recent_60min    arrivals in (t-60, t] / 60
  12: time_to_next_expected_carrier  1 / configured arrival rate
  13: queue_length_lag_1
  14: queue_length_lag_2
  15: queue_length_lag_3
  16: breach_rate_lag_1
  17: breach_rate_lag_2
  18: breach_rate_lag_3
  19: interval_index_in_shift
  20: intervals_remaining
  21: pct_high_priority
  22: mean_processing_time_remaining
  23: std_slack_minutes
  24: n_arrivals_this_interval

All features must be finite (np.isfinite). Empty queues / empty history yield 0.
"""

from __future__ import annotations

import numpy as np

from simulation.orders import Order

FEATURE_NAMES: list[str] = [
    "queue_length",
    "mean_queue_age",
    "max_queue_age",
    "pct_critical",
    "pct_perishable",
    "labor_utilization",
    "n_pickers_busy",
    "mean_pickup_time_recent",
    "n_orders_breached_so_far",
    "n_orders_at_risk_30min",
    "mean_slack_minutes",
    "arrival_rate_recent_60min",
    "time_to_next_expected_carrier",
    "queue_length_lag_1",
    "queue_length_lag_2",
    "queue_length_lag_3",
    "breach_rate_lag_1",
    "breach_rate_lag_2",
    "breach_rate_lag_3",
    "interval_index_in_shift",
    "intervals_remaining",
    "pct_high_priority",
    "mean_processing_time_remaining",
    "std_slack_minutes",
    "n_arrivals_this_interval",
]

N_FEATURES: int = 25
assert len(FEATURE_NAMES) == N_FEATURES


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _max(xs: list[float]) -> float:
    return float(np.max(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    return float(np.std(xs)) if len(xs) >= 2 else 0.0


def _lag(history: list[float], k: int) -> float:
    return float(history[-k]) if len(history) >= k else 0.0


def extract_features(
    *,
    t: float,
    queue: list[Order],
    completed: list[Order],
    picker_busy_until: list[float],
    interval_idx: int,
    n_intervals: int,
    history_queue_length: list[int],
    history_breach_rate: list[float],
    arrival_times_recent_60min: list[float],
    current_arrival_rate: float,
    n_arrivals_this_interval: int,
) -> np.ndarray:
    n_pickers = len(picker_busy_until)

    queue_ages = [t - o.arrival_time for o in queue]
    slacks = [o.sla_due - t - o.processing_time for o in queue]
    sla_remaining = [o.sla_due - t for o in queue]

    qlen = len(queue)
    pct_critical = sum(1 for s in sla_remaining if s < 30.0) / qlen if qlen else 0.0
    pct_perishable = sum(1 for o in queue if o.is_perishable) / qlen if qlen else 0.0
    pct_high = sum(1 for o in queue if o.priority_class == "high") / qlen if qlen else 0.0
    n_at_risk_30 = sum(
        1
        for o in queue
        if (o.sla_due - t) < 30.0 and (o.sla_due - t - o.processing_time) < 0.0
    )

    n_busy = sum(1 for b in picker_busy_until if b > t)
    labor_util = n_busy / n_pickers if n_pickers else 0.0

    # Completed-derived (only orders that have actually finished by `t`).
    finished_by_t = [
        o for o in completed if o.finish_time is not None and o.finish_time <= t
    ]
    n_breached_so_far = sum(1 for o in finished_by_t if o.finish_time > o.sla_due)
    recent_finished = [o for o in finished_by_t if o.finish_time >= t - 60.0]
    mean_pickup_recent = _mean([o.processing_time for o in recent_finished])

    arrival_rate_60 = len(arrival_times_recent_60min) / 60.0
    time_to_next = 1.0 / max(float(current_arrival_rate), 1e-9)

    feat = np.array(
        [
            qlen,
            _mean(queue_ages),
            _max(queue_ages),
            pct_critical,
            pct_perishable,
            labor_util,
            n_busy,
            mean_pickup_recent,
            n_breached_so_far,
            n_at_risk_30,
            _mean(slacks),
            arrival_rate_60,
            time_to_next,
            _lag(history_queue_length, 1),
            _lag(history_queue_length, 2),
            _lag(history_queue_length, 3),
            _lag(history_breach_rate, 1),
            _lag(history_breach_rate, 2),
            _lag(history_breach_rate, 3),
            interval_idx,
            max(n_intervals - interval_idx, 0),
            pct_high,
            _mean([o.processing_time for o in queue]),
            _std(slacks),
            n_arrivals_this_interval,
        ],
        dtype=np.float64,
    )
    return feat
