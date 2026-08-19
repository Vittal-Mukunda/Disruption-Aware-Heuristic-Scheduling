"""Warehouse environment for one 8-hour shift — the transition function `S^M`.

Implemented as a pure-Python periodic-review model rather than with SimPy
coroutines, for two reasons that have only become more important in revision:

  1. Determinism. The exogenous stream is sampled from a seeded
     `np.random.default_rng`, so a shift is reproducible from its seed.
  2. Branchability. Multi-sample rollout labelling (Reviewer 2, 3) needs to
     freeze a state and then draw *independent* futures from it. That is
     `branch()`, and it is cheap here because the whole state is plain data.

PERIODIC REVIEW AND THE ADMISSION RULE (Reviewer 1, 6.a)
--------------------------------------------------------
The submitted version admitted every order with `arrival_time <= interval_end`
— fifteen minutes of look-ahead — and then computed
`start = max(picker_free, arrival_time, interval_start)`. Ranking an order that
had not yet arrived therefore RESERVED A PICKER AND LEFT IT IDLE until that
order showed up. Rules sorted by arrival (FIFO) never paid this; arrival-agnostic
rules (WSPT, ATC) paid it constantly. That is the mechanism behind the
"counterintuitive" Table 1 the reviewer flagged: FIFO reached 0.983 picker
utilisation while WSPT sat at 0.686 with a queue pinned near the 200-order cap.
A picker cannot idle 31% of a shift with 180 orders waiting unless the dispatch
model idles it.

The model is now properly periodic-review: at decision epoch `t` the controller
sees, and the dispatcher may assign, exactly the orders that have arrived by
`t`. Orders arriving inside `[t, t+15)` wait for the next epoch. This also
removes an undisclosed clairvoyance — the submitted feature vector summarised
up to fifteen minutes of future arrivals.

There is no terminal admit at shift end `T` unless `sim.terminal_admit` is true.
When false (the committed default, and every live table), the last review is at
`t = T - L`, so arrivals in `(T - L, T]` never enter the queue. Mean `|A| = 767`
against a Poisson mean of `1.65 * 480 = 792` is that convention. When true, a
gated `_admit()` runs only after `interval_idx == n_intervals`, so truncated
rollouts are unchanged. Do not call `_admit()` unconditionally at the end of
`run_with_policy`: that would poison labelling.

With that rule, `start = max(picker_free, t)` and the early exit on line ~"no
picker can start before the interval ends" is provably safe: `picker_free` is
the minimum over pickers, so if the earliest-free picker cannot start, none can.

Dispatch within `[t, t+15)`:
    1. Admit every order with `arrival_time <= t` (queue capped; overflow is
       recorded as `dropped`, not silently discarded).
    2. Rank the queue with the chosen rule, evaluated at `t`.
    3. Assign down the ranking to the earliest-free picker,
       `start = max(picker_free, t)`; stop once no picker can start before
       `t+15`. Orders finishing past `t+15` hold their picker into the next
       interval.
    4. Record end-of-interval queue length and realised service-failure rate.
"""

from __future__ import annotations

import copy

import numpy as np
from omegaconf import DictConfig

from simulation.cost import CostWeights, potential
from simulation.heuristics import HEURISTICS, rank
from simulation.kpis import compute_kpis
from simulation.orders import Order
from simulation.state_extractor import N_FEATURES, extract_features

# Global counter of simulated interval-steps. Reviewer 3 (1) asks for the total
# offline simulation cost; instrumenting it here means the number is measured
# rather than estimated, across labelling, screening and evaluation alike.
_STEP_COUNTER: int = 0


def simulated_steps() -> int:
    """Total interval-steps simulated in this process since the last reset."""
    return _STEP_COUNTER


def reset_step_counter() -> None:
    global _STEP_COUNTER
    _STEP_COUNTER = 0


def sample_order_stream(
    rng: np.random.Generator,
    cfg: DictConfig,
    *,
    start_time: float,
    start_oid: int,
    until: float,
) -> list[Order]:
    """Draw the exogenous arrival stream on `(start_time, until)`.

    Factored out of `__init__` so that `branch()` can resample the unrealised
    tail of a shift with an independent generator while keeping the realised
    history byte-identical.

    A fixed number of variates is drawn per order — processing time, SLA offset,
    shelf life, perishability, priority — regardless of whether the shelf life is
    used. Keeping the draw count constant makes the stream position a function of
    the order index alone, which keeps branches reproducible.
    """
    sim = cfg.sim
    rate = float(sim.arrivals.base_rate_per_minute)
    mode = str(sim.arrivals.get("arrival_mode", "poisson")).lower()
    pt_lo, pt_mode, pt_hi = sim.order_attrs.processing_time_triangular
    sla_lo, sla_mode, sla_hi = sim.order_attrs.sla_due_triangular
    shelf_lo, shelf_mode, shelf_hi = sim.order_attrs.shelf_life_triangular
    p_perish = float(sim.order_attrs.perishable_prob)
    classes = list(sim.order_attrs.priority_classes)
    class_w = np.asarray(sim.order_attrs.priority_weights, dtype=np.float64)
    class_w = class_w / class_w.sum()

    gap_scale = 1.0 / max(rate, 1e-9)
    if mode == "poisson":
        def next_gap() -> float:
            return float(rng.exponential(gap_scale))
    elif mode == "olist":
        from simulation.olist_arrivals import load_interarrival_pool

        pool = load_interarrival_pool()

        def next_gap() -> float:
            return float(pool[rng.integers(pool.size)]) * gap_scale
    else:
        raise ValueError(
            f"Unknown sim.arrivals.arrival_mode '{mode}'. Valid: 'poisson', 'olist'."
        )

    orders: list[Order] = []
    t = float(start_time)
    oid = int(start_oid)
    while True:
        t += next_gap()
        if t >= until:
            break
        pt = float(rng.triangular(pt_lo, pt_mode, pt_hi))
        sla_offset = float(rng.triangular(sla_lo, sla_mode, sla_hi))
        shelf = float(rng.triangular(shelf_lo, shelf_mode, shelf_hi))
        is_perish = bool(rng.random() < p_perish)
        prio = classes[int(rng.choice(len(classes), p=class_w))]
        orders.append(
            Order(
                order_id=oid,
                arrival_time=t,
                processing_time=pt,
                sla_due=t + sla_offset,
                is_perishable=is_perish,
                priority_class=prio,
                # Two independent clocks: the customer deadline above and the
                # product deadline here. Either can bind first.
                expiry_time=(t + shelf) if is_perish else None,
            )
        )
        oid += 1
    return orders


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
        self.weights: CostWeights = CostWeights.from_cfg(cfg)

        self._all_orders: list[Order] = sample_order_stream(
            self.rng, cfg, start_time=0.0, start_oid=0, until=self.shift_minutes
        )
        self._next_order_idx: int = 0

        self.t: float = 0.0
        self.interval_idx: int = 0
        self.queue: list[Order] = []
        self.completed: list[Order] = []
        self.dropped: list[Order] = []
        self.picker_busy_until: list[float] = [0.0] * self.n_pickers
        self.policy_history: list[str] = []

        self._n_arrivals_last_interval: int = 0
        self._history_queue_length: list[int] = []
        self._history_failure_rate: list[float] = []

    # ------------------------------------------------------------------
    # Order population
    # ------------------------------------------------------------------

    def arrived_orders(self) -> list[Order]:
        """Every order that has entered the system: served, waiting, or refused.

        This is the denominator of `service_failure_rate` and the domain of the
        cost potential. Including `dropped` is deliberate — a refused order is
        real demand that was not met (Reviewer 2, 1).
        """
        return self.completed + self.queue + self.dropped

    def _admit(self) -> int:
        """Admit every order that has arrived by the current epoch. Idempotent.

        Returns the number of arrivals processed. Orders arriving when the queue
        is at capacity are recorded in `self.dropped` rather than vanishing, as
        they did in the submitted version.
        """
        n_new = 0
        while self._next_order_idx < len(self._all_orders):
            o = self._all_orders[self._next_order_idx]
            if o.arrival_time > self.t:
                break
            if len(self.queue) < self.queue_cap:
                self.queue.append(o)
            else:
                self.dropped.append(o)
            self._next_order_idx += 1
            n_new += 1
        return n_new

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Admit due arrivals, then return `phi(S_t)`.

        Every controller must go through this, so the observation and the queue
        the dispatcher acts on are the same object. In the submitted code the
        harness read `current_state()` before `step()` performed admission, so
        which arrivals a policy could see depended on the call site.
        """
        self._n_arrivals_last_interval = self._admit()
        return self.current_state()

    def current_state(self) -> np.ndarray:
        """Pure read of the feature map. No side effects."""
        arrivals_recent = [
            o.arrival_time
            for o in self._all_orders[: self._next_order_idx]
            if self.t - 60.0 < o.arrival_time <= self.t
        ]
        return extract_features(
            t=self.t,
            queue=self.queue,
            completed=self.completed,
            picker_busy_until=self.picker_busy_until,
            interval_idx=self.interval_idx,
            history_queue_length=self._history_queue_length,
            history_failure_rate=self._history_failure_rate,
            arrival_times_recent_60min=arrivals_recent,
            n_arrivals_last_interval=self._n_arrivals_last_interval,
        )

    def potential(self) -> float:
        """Cost potential `Phi(t)` of the current system. See `simulation.cost`."""
        return potential(
            self.arrived_orders(), len(self.queue), self.t, self.weights
        )

    # ------------------------------------------------------------------
    # Single-interval transition
    # ------------------------------------------------------------------

    def step(self, heuristic_name: str) -> None:
        global _STEP_COUNTER
        if heuristic_name not in HEURISTICS:
            raise ValueError(
                f"Unknown heuristic '{heuristic_name}'. "
                f"Available: {sorted(HEURISTICS)}"
            )
        if self.interval_idx >= self.n_intervals:
            raise RuntimeError("Shift already complete; cannot step further.")

        interval_start = self.t
        interval_end = interval_start + self.interval_minutes

        # Idempotent: a no-op when the caller already went through observe().
        n_new = self._admit()
        if n_new:
            self._n_arrivals_last_interval = n_new

        # Scale parameters are resolved lazily. Committed config carries fitted
        # k=3.0/4.0; if a scale is missing, only the rule that needs it fails,
        # rather than `float(None)` on every step for every rule.
        ctx = {"t": interval_start}
        for key, cfg_key in (("atc_k", "atc_lookahead_k"),
                             ("covert_k", "covert_lookahead_k")):
            raw = self.cfg.heuristics.get(cfg_key)
            if raw is not None:
                ctx[key] = float(raw)
        ranked = rank(heuristic_name, self.queue.copy(), ctx)

        dispatched: list[Order] = []
        for o in ranked:
            picker_idx = int(np.argmin(self.picker_busy_until))
            start = max(self.picker_busy_until[picker_idx], interval_start)
            # Safe early exit: `picker_busy_until[picker_idx]` is the minimum
            # over pickers, so if this one cannot start before the interval ends,
            # no picker can, and no later-ranked order could be dispatched either.
            if start >= interval_end:
                break
            o.start_time = start
            o.finish_time = start + o.processing_time
            self.picker_busy_until[picker_idx] = o.finish_time
            dispatched.append(o)

        for o in dispatched:
            self.queue.remove(o)
            self.completed.append(o)

        self.t = interval_end
        self.interval_idx += 1
        self.policy_history.append(heuristic_name)

        arrived = self.arrived_orders()
        self._history_queue_length.append(len(self.queue))
        self._history_failure_rate.append(
            sum(1 for o in arrived if o.is_service_failure(self.t)) / len(arrived)
            if arrived
            else 0.0
        )
        _STEP_COUNTER += 1

    # ------------------------------------------------------------------
    # Multi-interval driver
    # ------------------------------------------------------------------

    def run_with_policy(
        self, heuristic_name: str, n_steps: int | None = None
    ) -> dict[str, float]:
        remaining = self.n_intervals - self.interval_idx
        n = remaining if n_steps is None else min(int(n_steps), remaining)
        for _ in range(n):
            self.observe()
            self.step(heuristic_name)
        if bool(self.cfg.sim.get("terminal_admit", False)):
            self.admit_if_shift_complete()
        return self.kpis()

    def admit_if_shift_complete(self) -> int:
        """Admit arrivals with arrival_time <= T after the last review only.

        No-op while a truncated rollout is still inside the shift. That is what
        keeps labelling at tau < N independent of the completeness-run flag.
        """
        if self.interval_idx < self.n_intervals:
            return 0
        return self._admit()

    def kpis(self) -> dict[str, float]:
        return compute_kpis(
            self.completed,
            self.queue,
            self.dropped,
            self.n_pickers,
            self.shift_minutes,
            self.n_intervals,
            self.weights,
        )

    # ------------------------------------------------------------------
    # Branching — the mechanism behind multi-sample rollouts
    # ------------------------------------------------------------------

    def branch(
        self, rollout_seed: int, model_cfg: DictConfig | None = None
    ) -> "WarehouseEnv":
        """Freeze this state and draw an independent future from it.

        `model_cfg` overrides the parameters used to resample the future, leaving
        the realised history untouched. This separates the *world* from the
        *model of the world*: passing a config that differs from the environment's
        own makes the branch a mis-specified forecast, which is what Reviewer 2
        (5) asks us to test. It also keeps the misspecification experiment fair —
        an online lookahead controller must plan with the same imperfect model
        that generated the training labels, rather than being handed the true
        dynamics for free.

        The realised history — orders already arrived, dispatched, queued or
        dropped — is copied verbatim. The unrealised tail of the arrival stream
        is discarded and redrawn from `np.random.default_rng(rollout_seed)`.

        This is what makes the rollout estimator an estimator (Reviewer 2, 3).
        In the submitted version every "stochastic rollout" replayed the one
        pre-sampled future belonging to the shift seed, so each label was
        hindsight-optimal for a single realisation and the rollout variance was
        identically zero — which also means Proposition 1's bias/variance
        trade-off had no variance term in the code.

        COMMON RANDOM NUMBERS. `rollout_seed` must depend on the shift, the
        epoch and the sample index but NOT on the rule under test, so that every
        candidate rule is scored against the identical future. The comparison is
        then paired and the variance of the *difference* — which is what the
        label actually encodes — is far smaller than the variance of either arm.
        `labeling.rollout_labeler.rollout_seed()` enforces this.
        """
        # `cfg` is read-only throughout and is shared rather than deep-copied:
        # branching happens |H| * M times per epoch, and copying an OmegaConf
        # tree that many times dominates the rollout itself.
        cfg = self.cfg
        self.cfg = None  # type: ignore[assignment]
        try:
            new = copy.deepcopy(self)
        finally:
            self.cfg = cfg
        new.cfg = cfg

        # The future is drawn from the *model*; the dynamics that will execute it
        # remain the environment's own.
        forecast_cfg = model_cfg if model_cfg is not None else cfg
        new.rng = np.random.default_rng(int(rollout_seed))
        head = new._all_orders[: new._next_order_idx]
        new._all_orders = head + sample_order_stream(
            new.rng,
            forecast_cfg,
            start_time=new.t,
            start_oid=len(head),
            until=new.shift_minutes,
        )
        return new

    # ------------------------------------------------------------------
    # Replay-from-start (retained; the labeller no longer needs it)
    # ------------------------------------------------------------------

    def fast_forward(self, t_intervals: int, policy_history: list[str]) -> None:
        """Reset to t=0 and replay `t_intervals` steps under `policy_history`.

        Retained for tests and for any caller that wants an explicit replay. The
        rollout labeller no longer uses it: it walks a single shift forward once
        and branches at each epoch, which turns labelling from
        O(n_intervals^2 * |H|) into O(n_intervals * |H| * M).
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
            self.observe()
            self.step(policy_history[i])


__all__ = [
    "WarehouseEnv",
    "sample_order_stream",
    "simulated_steps",
    "reset_step_counter",
    "N_FEATURES",
]
