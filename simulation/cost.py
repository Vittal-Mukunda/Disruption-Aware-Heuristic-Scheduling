"""The composite objective — one definition, shared by labelling and evaluation.

The submitted version defined the cost twice: `labeling.snapshot_labeler.
composite_cost` for rollout labels and `experiments.evaluate._shift_mean_cost`
for reported KPIs. They agreed by inspection, not by construction. They are now
a single functional, applied with different reference horizons.

THE OBJECTIVE
-------------
Per order `o` that has arrived by the reference time `T`:

    c_o(T) = w_o * [ W_b * 1(late)  +  W_t * tardiness  +  W_s * 1(spoiled) ]

and the system potential is

    Phi(T) = sum_o c_o(T)  +  W_hold * |queue at T|

`w_o` is the order's economic weight (`Order.weight`). Three deliberate
departures from the submitted objective, each answering a specific review point:

1. PRIORITY WEIGHTS ENTER THE OBJECTIVE (Reviewer 1, 1.c and 6.a).
   WSPT and ATC have always ranked by `w_o / p_o`, but the submitted cost counted
   every order equally, so those rules optimised a criterion the evaluation never
   measured. That mismatch is a large part of why WSPT appeared pathological in
   Table 1. Rule and objective now agree.

2. UNSERVED ORDERS ARE CHARGED (Reviewer 2, 1).
   `Order.is_late` / `is_spoiled` / `tardiness` are defined for undispatched
   orders, so an order abandoned in the queue past its due time is charged the
   full breach weight. In the submitted objective it escaped with `W_u = 0.005`
   against `W_b = 3.0` — a 600x discount for never touching an order, and a
   controller could lower its reported breach rate simply by leaving hard orders
   alone. `W_hold` survives, but strictly as a work-in-progress holding cost, not
   as the penalty for failure.

3. SPOILAGE IS A DISTINCT FAILURE MODE (Reviewer 1, 1.c/1.d; Reviewer 2, 2).
   Perishables carry an `expiry_time` independent of `sla_due`, so spoiling and
   shipping late are different events with different weights. Under the submitted
   model spoilage was definitionally identical to an SLA breach and perishability
   never entered the cost at all.

WINDOW COSTS AS POTENTIAL DIFFERENCES
-------------------------------------
A truncated rollout must score only what happened inside its window, otherwise
the shared pre-window history dominates every rule's cost and the tempered
softmax collapses to uniform. The submitted code achieved this by counting only
orders *completed* during the window — which is exactly the accounting that lets
an unserved order disappear.

The correct construction is a potential difference:

    J_window = Phi(T_end) - Phi(T_start)

Common history cancels because it appears in both terms. An order that becomes
overdue inside the window is charged whether or not it was served. An order that
was already overdue at `T_start` keeps accruing tardiness. And because an
undispatched order is assessed at `T + p_o` (see `Order._resolution_time`) —
the earliest it could possibly finish — dispatching an order is always weakly
cheaper than leaving it, so the objective can never reward inaction.

Evaluation is the same functional with `T_start = 0` and `T_end = shift end`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from simulation.orders import Order


@dataclass(frozen=True)
class CostWeights:
    """Objective weights. Fixed before any learning; never tuned per method.

    `breach`  charge for missing the customer deadline
    `tardy`   charge per minute of lateness
    `spoil`   charge for a perishable order passing its expiry
    `holding` work-in-progress holding cost per queued order
    `use_priority_weights` multiply every per-order charge by `Order.weight`
    """

    breach: float = 3.0
    tardy: float = 0.2
    spoil: float = 5.0
    holding: float = 0.005
    use_priority_weights: bool = True

    @classmethod
    def from_cfg(cls, cfg) -> "CostWeights":
        c = cfg.objective
        return cls(
            breach=float(c.w_breach),
            tardy=float(c.w_tardy),
            spoil=float(c.w_spoil),
            holding=float(c.w_holding),
            use_priority_weights=bool(c.use_priority_weights),
        )


def order_charge(o: Order, t_ref: float, w: CostWeights) -> float:
    """Charge accrued by a single order, assessed at `t_ref`.

    Well defined for dispatched and undispatched orders alike.
    """
    weight = o.weight if w.use_priority_weights else 1.0
    charge = w.tardy * o.tardiness(t_ref)
    if o.is_late(t_ref):
        charge += w.breach
    if o.is_spoiled(t_ref):
        charge += w.spoil
    return weight * charge


def potential(
    arrived: Iterable[Order], queue_size: int, t_ref: float, w: CostWeights
) -> float:
    """System potential `Phi(t_ref)` over every order that has arrived."""
    total = sum(order_charge(o, t_ref, w) for o in arrived)
    return float(total + w.holding * queue_size)


def window_cost(phi_start: float, phi_end: float) -> float:
    """Cost attributable to a rollout window: `Phi(T_end) - Phi(T_start)`.

    Trivial by itself; named so that call sites read as the potential-difference
    argument documented above rather than as an unexplained subtraction.
    """
    return float(phi_end - phi_start)
