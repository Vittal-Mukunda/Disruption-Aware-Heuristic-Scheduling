"""Order dataclass — the unit of work in the warehouse simulation.

An Order is sampled from the seeded exogenous-information stream and lives
through three states:

  1. waiting    — `start_time is None`; sitting in `env.queue`
  2. in flight  — `start_time is not None`; scheduled to complete at `finish_time`
  3. complete   — same as in flight; dispatch commits both at once

TWO INDEPENDENT CLOCKS (revision, R1.1c/d/f, R2.2)
--------------------------------------------------
Before this revision an order carried a single deadline, `sla_due`, and the
rule named "FEFO" sorted on it — so the implemented rule was in fact EDD, and
"spoilage" was definitionally identical to an SLA breach. Reviewer 1 (1.f) and
Reviewer 2 (2) both caught this. An order now carries:

  `sla_due`      absolute timestamp — the *customer* deadline. Missing it is a
                 service failure: the order ships late.
  `expiry_time`  absolute timestamp — the *product* deadline, and only for
                 perishables (`None` otherwise). Missing it destroys the goods.

The two are sampled independently, so for a perishable order either can bind.
`effective_deadline()` reports whichever binds first; the fraction of the corpus
where expiry binds strictly before `sla_due` is what makes "perishability-
constrained" a substantive claim rather than a label, and is measured by
`experiments/perishability_diagnostic.py` (answers R1.1d).

CENSORED OUTCOMES
-----------------
Every outcome predicate takes a reference time `t_ref` and is well defined for
orders that were never dispatched. That is what closes the accounting loophole
Reviewer 2 (1) identified: an order left in the queue past its due time is late,
and is charged as such, rather than escaping the metric because it has no
`finish_time`. For an undispatched order, tardiness is *censored* — the order is
at least `t_ref - sla_due` late, and would be later still if the horizon ran on.
"""

from __future__ import annotations

from dataclasses import dataclass

# Economic weight of an order, by priority class.
#
# Moved here from `simulation.heuristics` during the revision. It used to live
# beside the rules, which meant WSPT and ATC ranked by `w_o / p_o` while the
# objective counted every order equally — the rules optimised a weighted
# criterion the evaluation never measured. That mismatch is a large part of why
# WSPT looked pathological in the submitted Table 1 (Reviewer 1, 6.a). The
# weights now live with the order and are consumed by *both* the rules and
# `simulation.cost`, so rule and objective finally agree.
PRIORITY_WEIGHTS: dict[str, float] = {"low": 1.0, "medium": 2.0, "high": 4.0}


@dataclass
class Order:
    order_id: int
    arrival_time: float
    processing_time: float
    sla_due: float
    is_perishable: bool
    priority_class: str

    # Absolute expiry timestamp; None for non-perishable orders.
    expiry_time: float | None = None

    start_time: float | None = None
    finish_time: float | None = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_dispatched(self) -> bool:
        return self.start_time is not None

    @property
    def weight(self) -> float:
        """Economic weight `w_o`. Used by the rules *and* by the objective."""
        return PRIORITY_WEIGHTS[self.priority_class]

    def effective_deadline(self) -> float:
        """The first of the two clocks to bind.

        For a non-perishable order this is `sla_due`. For a perishable order it
        is `min(sla_due, expiry_time)` — whichever failure mode arrives first.
        """
        if self.expiry_time is None:
            return self.sla_due
        return min(self.sla_due, self.expiry_time)

    def expiry_binds(self) -> bool:
        """True when the product clock is strictly tighter than the customer clock."""
        return self.expiry_time is not None and self.expiry_time < self.sla_due

    # ------------------------------------------------------------------
    # Outcome predicates — defined for dispatched *and* undispatched orders
    # ------------------------------------------------------------------

    def _resolution_time(self, t_ref: float) -> float:
        """When this order's fate is assessed.

        Dispatched orders resolve at `finish_time`. An undispatched order has not
        resolved, and is assessed at `t_ref + processing_time` — the earliest it
        could possibly finish, since it still needs a full pick.

        That `+ p_o` is load-bearing, not cosmetic. Assessing an undispatched
        order at `t_ref` alone would make abandoning it strictly cheaper than
        dispatching it late, and the objective would reward inaction — the same
        pathology, in subtler form, that Reviewer 2 (1) identified in the
        `W_u = 0.005` holding term. With the `+ p_o` lower bound, dispatching an
        order at `t_ref` onto a free picker costs exactly what leaving it costs,
        and dispatching it any earlier costs strictly less. Doing the work is
        therefore always weakly optimal, as it must be.
        """
        if self.finish_time is not None:
            return self.finish_time
        return t_ref + self.processing_time

    def is_late(self, t_ref: float) -> bool:
        """Missed the customer deadline, whether or not it was ever dispatched."""
        return self._resolution_time(t_ref) > self.sla_due

    def is_spoiled(self, t_ref: float) -> bool:
        """Perishable and past its product deadline.

        An undispatched perishable order whose expiry has passed is spoiled: the
        goods are destroyed regardless of what happens next.
        """
        if self.expiry_time is None:
            return False
        return self._resolution_time(t_ref) > self.expiry_time

    def tardiness(self, t_ref: float) -> float:
        """Lateness against the customer deadline, floored at zero.

        Censored for undispatched orders: the true value is at least this large.
        """
        return max(self._resolution_time(t_ref) - self.sla_due, 0.0)

    def is_service_failure(self, t_ref: float) -> bool:
        """The headline per-order failure predicate.

        An order fails if it ships late, or if it spoils. This is the quantity
        Reviewer 2 (1) asked to see reported: it does not care whether the order
        was completed late or abandoned in the queue.
        """
        return self.is_late(t_ref) or self.is_spoiled(t_ref)
