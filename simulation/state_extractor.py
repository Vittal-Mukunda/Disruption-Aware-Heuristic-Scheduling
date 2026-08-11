"""The feature map `phi(S_t)` — what the selector observes at a decision epoch.

THIS IS AN OBSERVATION, NOT A STATE (Reviewer 2, 4)
---------------------------------------------------
The submitted paper called this vector "the state". It is not. The true state
`S_t` is the full queue with every order's arrival time, processing time, due
time, expiry and priority, plus the picker-availability vector and the clock.
This module computes `x_t = phi(S_t)`, a fixed-length summary of it. Two queues
with different contents can map to the same `x_t` and then evolve differently
under the same rule, so `phi` is not a sufficient statistic and the control
problem is a POMDP. DAHS is a policy-function approximation over `phi`.

The manuscript now says so in Section 3 rather than eliding it, gives the
explicit counterexample, and reports the empirical cost of the approximation
(label disagreement among states with near-identical `phi`).

FEATURES REMOVED IN REVISION (Reviewer 1, 3.a)
----------------------------------------------
Two of the submitted 25 features were degenerate:

  `time_to_next_expected_carrier` = 1 / arrival_rate — CONSTANT within a
      configuration. Zero variance, zero information.
  `intervals_remaining` = n_intervals - interval_index — an EXACT affine
      function of `interval_index_in_shift`; the two sum to 32 by construction.

Together they made the feature matrix exactly singular, which is why the
regime GMM (`covariance_type="full"`, `reg_covar=1e-6`) produced a BIC that fell
monotonically to the edge of the K grid: each extra component bought likelihood
by collapsing further onto the degenerate directions. K=6 was selected at the
grid boundary, and the reported ARI of 0.998 recorded EM reliably finding the
same degenerate basin. Both features are dropped.

FEATURES ADDED IN REVISION (Reviewer 1, 1.c/1.d; Reviewer 2, 2)
---------------------------------------------------------------
Perishability now carries a product clock independent of the customer clock and
enters the objective. The selector must be able to see that pressure, otherwise
FEFO can never be chosen for a reason. Three expiry features are added.

`n_arrivals_this_interval` is renamed `n_arrivals_last_interval`, which is what
it always measured: admission happens at the epoch boundary, so at decision time
the count refers to the interval that just closed.
"""

from __future__ import annotations

import numpy as np

from simulation.orders import Order

FEATURE_NAMES: list[str] = [
    # --- Queue ---
    "queue_length",
    "mean_queue_age",
    "max_queue_age",
    "pct_critical",
    "pct_perishable",
    "n_arrivals_last_interval",
    # --- Resources ---
    "labor_utilization",
    "n_pickers_busy",
    "mean_pickup_time_recent",
    # --- Customer-deadline pressure ---
    "n_orders_late_so_far",
    "n_orders_at_risk_30min",
    "mean_slack_minutes",
    "std_slack_minutes",
    "mean_processing_time_remaining",
    "pct_high_priority",
    # --- Product-deadline (expiry) pressure ---
    "pct_expiring_30min",
    "mean_expiry_slack",
    "n_spoiled_so_far",
    # --- Arrivals ---
    "arrival_rate_recent_60min",
    # --- History ---
    "queue_length_lag_1",
    "queue_length_lag_2",
    "queue_length_lag_3",
    "failure_rate_lag_1",
    "failure_rate_lag_2",
    "failure_rate_lag_3",
    # --- Temporal ---
    "interval_index_in_shift",
]

N_FEATURES: int = len(FEATURE_NAMES)

# Provenance table for Appendix A (Reviewer 1, 3.a asked how the features were
# identified and selected). `source` is the design rationale or the literature
# the feature is drawn from; `group` matches the manuscript's grouping.
FEATURE_PROVENANCE: dict[str, tuple[str, str]] = {
    "queue_length": ("queue", "Standard congestion state; de Koster et al. (2007)."),
    "mean_queue_age": ("queue", "Waiting-time proxy; distinguishes fresh from stale backlog."),
    "max_queue_age": ("queue", "Tail of the waiting-time distribution — starvation detector."),
    "pct_critical": ("queue", "Share of queue within 30 min of its due time."),
    "pct_perishable": ("queue", "Gates the FEFO mask; required for the expiry-aware rule."),
    "n_arrivals_last_interval": ("queue", "Short-run demand shock; wave arrivals, Boysen et al. (2019)."),
    "labor_utilization": ("resources", "Classical queueing load indicator rho."),
    "n_pickers_busy": ("resources", "Absolute capacity remaining this epoch."),
    "mean_pickup_time_recent": ("resources", "Realised service-rate estimate; drifts with order mix."),
    "n_orders_late_so_far": ("deadline", "Realised failures to date; regime indicator."),
    "n_orders_at_risk_30min": ("deadline", "Count with negative slack inside 30 min — the actionable set."),
    "mean_slack_minutes": ("deadline", "Mean d - t - p; the ATC/MS/COVERT decision variable."),
    "std_slack_minutes": ("deadline", "Slack dispersion; separates uniform from bimodal pressure."),
    "mean_processing_time_remaining": ("deadline", "Expected work content of the queue."),
    "pct_high_priority": ("deadline", "Share of the economically weighted tail."),
    "pct_expiring_30min": ("expiry", "Product-clock analogue of pct_critical (revision)."),
    "mean_expiry_slack": ("expiry", "Mean x - t - p over perishables; FEFO's decision variable (revision)."),
    "n_spoiled_so_far": ("expiry", "Realised spoilage to date (revision)."),
    "arrival_rate_recent_60min": ("arrivals", "Non-stationarity detector; empirical lambda-hat."),
    "queue_length_lag_1": ("history", "Congestion trend; standard lag structure."),
    "queue_length_lag_2": ("history", "Congestion trend."),
    "queue_length_lag_3": ("history", "Congestion trend."),
    "failure_rate_lag_1": ("history", "Failure trend; replaces breach_rate lag under the new metric."),
    "failure_rate_lag_2": ("history", "Failure trend."),
    "failure_rate_lag_3": ("history", "Failure trend."),
    "interval_index_in_shift": ("temporal", "Position in the finite horizon; end-of-shift effects."),
}


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
    history_queue_length: list[int],
    history_failure_rate: list[float],
    arrival_times_recent_60min: list[float],
    n_arrivals_last_interval: int,
) -> np.ndarray:
    """Compute `phi(S_t)`. Every entry is finite; empty queues yield 0."""
    n_pickers = len(picker_busy_until)
    qlen = len(queue)

    queue_ages = [t - o.arrival_time for o in queue]
    slacks = [o.sla_due - t - o.processing_time for o in queue]
    sla_remaining = [o.sla_due - t for o in queue]

    pct_critical = _safe_frac(sum(1 for s in sla_remaining if s < 30.0), qlen)
    pct_perishable = _safe_frac(sum(1 for o in queue if o.is_perishable), qlen)
    pct_high = _safe_frac(sum(1 for o in queue if o.priority_class == "high"), qlen)
    n_at_risk_30 = sum(
        1
        for o in queue
        if (o.sla_due - t) < 30.0 and (o.sla_due - t - o.processing_time) < 0.0
    )

    # --- Expiry pressure: the product clock, new in this revision -------------
    perishable_queue = [o for o in queue if o.expiry_time is not None]
    pct_expiring_30 = _safe_frac(
        sum(1 for o in perishable_queue if (o.expiry_time - t) < 30.0), qlen
    )
    mean_expiry_slack = _mean(
        [o.expiry_time - t - o.processing_time for o in perishable_queue]
    )

    n_busy = sum(1 for b in picker_busy_until if b > t)
    labor_util = _safe_frac(n_busy, n_pickers)

    # Only orders that have physically finished by `t` inform realised history.
    finished_by_t = [
        o for o in completed if o.finish_time is not None and o.finish_time <= t
    ]
    n_late_so_far = sum(1 for o in finished_by_t if o.is_late(t))
    n_spoiled_so_far = sum(1 for o in finished_by_t if o.is_spoiled(t))
    recent_finished = [o for o in finished_by_t if o.finish_time >= t - 60.0]
    mean_pickup_recent = _mean([o.processing_time for o in recent_finished])

    feat = np.array(
        [
            qlen,
            _mean(queue_ages),
            _max(queue_ages),
            pct_critical,
            pct_perishable,
            n_arrivals_last_interval,
            labor_util,
            n_busy,
            mean_pickup_recent,
            n_late_so_far,
            n_at_risk_30,
            _mean(slacks),
            _std(slacks),
            _mean([o.processing_time for o in queue]),
            pct_high,
            pct_expiring_30,
            mean_expiry_slack,
            n_spoiled_so_far,
            len(arrival_times_recent_60min) / 60.0,
            _lag(history_queue_length, 1),
            _lag(history_queue_length, 2),
            _lag(history_queue_length, 3),
            _lag(history_failure_rate, 1),
            _lag(history_failure_rate, 2),
            _lag(history_failure_rate, 3),
            interval_idx,
        ],
        dtype=np.float64,
    )
    assert feat.shape == (N_FEATURES,), (
        f"feature vector has {feat.shape[0]} entries but FEATURE_NAMES has "
        f"{N_FEATURES}; the two lists have drifted apart."
    )
    return feat


def _safe_frac(num: float, den: float) -> float:
    return float(num / den) if den else 0.0
