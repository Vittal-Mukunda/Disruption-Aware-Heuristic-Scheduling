"""Order dataclass — the unit of work in the warehouse simulation.

An Order is created at construction time of WarehouseEnv (deterministically from
the seeded RNG) and lives through three states:

  1. waiting    — `start_time is None`; sitting in env.queue
  2. in flight  — `start_time is not None and finish_time is not None`, scheduled
                  for completion at `finish_time`
  3. complete   — same as in flight; we don't distinguish "scheduled" vs
                  "actually finished" because dispatch commits both at once.

`sla_due` is an absolute timestamp (minutes from shift start), not an offset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    order_id: int
    arrival_time: float
    processing_time: float
    sla_due: float
    is_perishable: bool
    priority_class: str

    start_time: float | None = None
    finish_time: float | None = None

    @property
    def is_dispatched(self) -> bool:
        return self.start_time is not None
