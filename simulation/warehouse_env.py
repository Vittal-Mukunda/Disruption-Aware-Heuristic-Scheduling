"""Warehouse environment for one 8-hour shift.

NOTE: the master plan calls this a "SimPy" simulator. We implement it as a
pure-Python discrete-interval model rather than using `simpy.Environment` /
coroutines for two reasons:

  1. Determinism. All stochastic events are pre-sampled at construction time
     from a single seeded `np.random.default_rng`. Re-instantiating with the
     same seed reproduces the same arrival stream byte-for-byte, which is what
     Phase 3's replay-from-start rollout labeler requires.
  2. Simplicity. Decisions happen at 32 fixed interval boundaries; nothing
     about that needs SimPy's event scheduler. Pickers are tracked as a list
     of `busy_until` floats — equivalent semantics to `simpy.Resource`, but
     trivially serializable and replay-safe.

Public API (matches HANDOFF.md §3.1):

    env = WarehouseEnv(seed, cfg)
    state = env.current_state()                       # 25-vector
    state = env.step("FIFO")                          # advance 1 interval
    kpis  = env.run_with_policy("FIFO", n_steps=None) # to end-of-shift
    env.fast_forward(t_intervals, policy_history)     # reset + replay

Dispatch model within an interval [t, t+15):
    1. Pull in all orders with arrival_time <= t+15 (queue grows up to cap).
    2. Rank queue with the chosen heuristic at decision-time t.
    3. Greedily assign each ranked order to the earliest-free picker;
       start = max(picker.free_at, order.arrival_time, t). Stop when no
       picker can start before t+15. Orders whose finish extends past t+15
       keep the picker busy into the next interval.
    4. Record end-of-interval queue_length and breach_rate to history.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from simulation.heuristics import HEURISTICS, HEURISTIC_NAMES
from simulation.kpis import compute_kpis
from simulation.orders import Order
from simulation.state_extractor import N_FEATURES, extract_features


class WarehouseEnv:
    def __init__(self, seed: int, cfg: DictConfig):
        self.seed: int = int(seed)
        self.cfg: DictConfig = cfg
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        self.shift_minutes: float = float(cfg.sim.shift_hours * 60)
        self.interval_minutes: float = float(cfg.sim.interval_minutes)
        self.n_intervals: int = int(round(self.shift_minutes / self.interval_minutes))
        self.n_pickers: int = int(cfg.sim.n_pickers)
        self.queue_cap: int = int(cfg.sim.queue_capacity)

        self._all_orders: list[Order] = self._generate_orders()
        self._next_order_idx: int = 0

        self.t: float = 0.0
        self.interval_idx: int = 0
        self.queue: list[Order] = []
        self.picker_busy_until: list[float] = [0.0] * self.n_pickers
        self.completed: list[Order] = []
        self.policy_history: list[str] = []

        self._n_arrivals_this_interval: int = 0
        self._history_queue_length: list[int] = []
        self._history_breach_rate: list[float] = []

    # ------------------------------------------------------------------
    # Deterministic order pre-generation
    # ------------------------------------------------------------------

    def _generate_orders(self) -> list[Order]:
        sim_cfg = self.cfg.sim
        rate = float(sim_cfg.arrivals.base_rate_per_minute)
        arrival_mode = str(sim_cfg.arrivals.get("arrival_mode", "poisson")).lower()
        pt_lo, pt_mode, pt_hi = sim_cfg.order_attrs.processing_time_triangular
        sla_lo, sla_mode, sla_hi = sim_cfg.order_attrs.sla_due_triangular
        p_perish = float(sim_cfg.order_attrs.perishable_prob)
        prio_classes = list(sim_cfg.order_attrs.priority_classes)
        prio_weights = np.asarray(sim_cfg.order_attrs.priority_weights, dtype=np.float64)
        prio_weights = prio_weights / prio_weights.sum()

        # Inter-arrival gap source. "poisson" -> exponential gaps (CV=1);
        # "olist" -> bootstrap of the empirical mean-normalized Olist trace
        # (CV=2.68, skew=11). Both rescale to hold the mean rate fixed at
        # `rate`, so the only thing that changes is arrival-stream shape.
        # The poisson branch issues the identical rng.exponential() draw as
        # the original implementation, so frozen-seed runs stay byte-identical.
        gap_scale = 1.0 / max(rate, 1e-9)
        if arrival_mode == "poisson":
            def _next_gap() -> float:
                return float(self.rng.exponential(gap_scale))
        elif arrival_mode == "olist":
            from simulation.olist_arrivals import load_interarrival_pool

            pool = load_interarrival_pool()

            def _next_gap() -> float:
                return float(pool[self.rng.integers(pool.size)]) * gap_scale
        else:
            raise ValueError(
                f"Unknown sim.arrivals.arrival_mode '{arrival_mode}'. "
                f"Valid: 'poisson', 'olist'."
            )

        orders: list[Order] = []
        t = 0.0
        oid = 0
        while True:
            t += _next_gap()
            if t >= self.shift_minutes:
                break
            pt = float(self.rng.triangular(pt_lo, pt_mode, pt_hi))
            sla_offset = float(self.rng.triangular(sla_lo, sla_mode, sla_hi))
            is_perish = bool(self.rng.random() < p_perish)
            prio_idx = int(self.rng.choice(len(prio_classes), p=prio_weights))
            orders.append(
                Order(
                    order_id=oid,
                    arrival_time=t,
                    processing_time=pt,
                    sla_due=t + sla_offset,
                    is_perishable=is_perish,
                    priority_class=prio_classes[prio_idx],
                )
            )
            oid += 1
        return orders

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def current_state(self) -> np.ndarray:
        rate = float(self.cfg.sim.arrivals.base_rate_per_minute)
        arrivals_recent = [
            o.arrival_time
            for o in self._all_orders[: self._next_order_idx]
            if self.t - 60.0 < o.arrival_time <= self.t
        ]
        feat = extract_features(
            t=self.t,
            queue=self.queue,
            completed=self.completed,
            picker_busy_until=self.picker_busy_until,
            interval_idx=self.interval_idx,
            n_intervals=self.n_intervals,
            history_queue_length=self._history_queue_length,
            history_breach_rate=self._history_breach_rate,
            arrival_times_recent_60min=arrivals_recent,
            current_arrival_rate=rate,
            n_arrivals_this_interval=self._n_arrivals_this_interval,
        )
        assert feat.shape == (N_FEATURES,)
        return feat

    # ------------------------------------------------------------------
    # Single-interval step
    # ------------------------------------------------------------------

    def step(self, heuristic_name: str) -> np.ndarray:
        if heuristic_name not in HEURISTICS:
            raise ValueError(
                f"Unknown heuristic '{heuristic_name}'. Valid: {HEURISTIC_NAMES}"
            )
        if self.interval_idx >= self.n_intervals:
            raise RuntimeError("Shift already complete; cannot step further.")

        interval_start = self.t
        interval_end = interval_start + self.interval_minutes

        n_new = 0
        while self._next_order_idx < len(self._all_orders):
            o = self._all_orders[self._next_order_idx]
            if o.arrival_time > interval_end:
                break
            if len(self.queue) < self.queue_cap:
                self.queue.append(o)
            self._next_order_idx += 1
            n_new += 1
        self._n_arrivals_this_interval = n_new

        decision_state = self.current_state()

        ctx = {"t": interval_start, "atc_k": float(self.cfg.heuristics.atc_lookahead_k)}
        ranker = HEURISTICS[heuristic_name]
        ranked = ranker(self.queue.copy(), ctx)

        dispatched: list[Order] = []
        for o in ranked:
            picker_idx = int(np.argmin(self.picker_busy_until))
            free_at = self.picker_busy_until[picker_idx]
            start = max(free_at, o.arrival_time, interval_start)
            if start >= interval_end:
                break
            o.start_time = start
            o.finish_time = start + o.processing_time
            self.picker_busy_until[picker_idx] = o.finish_time
            dispatched.append(o)

        for o in dispatched:
            self.queue.remove(o)
            self.completed.append(o)

        self._history_queue_length.append(len(self.queue))
        n_completed = len(self.completed)
        breaches = sum(1 for x in self.completed if x.finish_time > x.sla_due)
        self._history_breach_rate.append(
            breaches / n_completed if n_completed else 0.0
        )

        self.t = interval_end
        self.interval_idx += 1
        self.policy_history.append(heuristic_name)

        return decision_state

    # ------------------------------------------------------------------
    # Multi-interval driver
    # ------------------------------------------------------------------

    def run_with_policy(
        self, heuristic_name: str, n_steps: int | None = None
    ) -> dict[str, float]:
        remaining = self.n_intervals - self.interval_idx
        n = remaining if n_steps is None else min(int(n_steps), remaining)
        for _ in range(n):
            self.step(heuristic_name)
        return compute_kpis(self.completed, self.queue, self.n_pickers, self.shift_minutes)

    # ------------------------------------------------------------------
    # Replay-from-start (Phase 3 rollout labeler depends on this)
    # ------------------------------------------------------------------

    def fast_forward(self, t_intervals: int, policy_history: list[str]) -> None:
        """Reset to t=0 and replay `t_intervals` steps using `policy_history`.

        Bit-identical to running step() from a fresh `WarehouseEnv(seed, cfg)`
        for `t_intervals` calls with the same heuristic names.
        """
        if t_intervals > len(policy_history):
            raise ValueError(
                f"policy_history length {len(policy_history)} < t_intervals {t_intervals}"
            )
        if t_intervals > self.n_intervals:
            raise ValueError(
                f"t_intervals {t_intervals} > n_intervals {self.n_intervals}"
            )
        self.__init__(self.seed, self.cfg)
        for i in range(t_intervals):
            self.step(policy_history[i])
