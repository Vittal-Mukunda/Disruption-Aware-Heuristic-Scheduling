"""Regression tests for two reproducibility defects found in the audit.

Neither was raised by a reviewer, but both would have been fair game, and one of
them produced a visible contradiction in the submitted manuscript.

DEFECT 1 — THE SAME CONFIGURATION READ DIFFERENTLY DOWN TWO CODE PATHS
----------------------------------------------------------------------
`experiments/e8_robustness_grid.py` evaluates a 12-cell grid of arrival rates
crossed with deadline-tightness levels. One of those cells — arrival rate 1.65 at
the default tightness — is byte-identical in configuration to the default scenario
of Table 1 and draws the same test seeds, so it must reproduce Table 1 exactly.
In the submitted results it did not: Figure 6 read 0.0956 for the best static rule
where Table 1 read 0.1181. A static rule carries no learned artefact and is
deterministic given a seed, so the discrepancy could not be a model difference; it
had to be the environment or the seed stream.

`test_grid_default_cell_matches_default_scenario` pins that. It runs the static
rules through both paths on a handful of seeds and requires exact agreement. If
the two paths ever diverge again — because one grows a config overlay the other
lacks, or resolves seeds differently — this fails instead of surfacing months
later as two irreconcilable numbers in two figures.

DEFECT 2 — TWO RUNS THAT SHOULD BE IDENTICAL WERE NOT
------------------------------------------------------
Section 6.6 of the submitted paper reported calibration metrics from
`runs/data_efficiency/ours_n250_rep0` while Table 1 reported KPIs from
`runs/phase4`. Those two runs share their data, their seed and their
hyperparameters and should be the same model; their cross-validated soft
cross-entropy differed in the sixth decimal place (0.6768496 against 0.6762263).
A difference that size is not a bug in itself, but it means the pipeline is not
bit-reproducible, and a paper cannot cite two runs interchangeably unless it is.

`test_rollout_labelling_is_bit_reproducible` and
`test_static_rule_kpis_are_bit_reproducible` pin determinism at the two places it
has to hold: the labelling estimator and the evaluation harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Deliberately small. These tests guard determinism and cross-path agreement,
# neither of which needs a full corpus to detect, and they run on every commit.
N_SEEDS = 3
STATIC_RULES = ["FIFO", "EDD", "WSPT"]

# Wall-clock columns are measurements of the machine, not results of the model.
# They vary run to run by construction and are excluded from every equality
# check below; everything else must match exactly.
TIMING_SUBSTRINGS = ("latency", "wall_clock", "_ms", "seconds", "_s_")


def _is_timing(col: str) -> bool:
    return any(t in col for t in TIMING_SUBSTRINGS)


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


def _kpi_row(method: str, seeds: list[int], cfg) -> dict:
    """Evaluate one static rule and return its KPIs as plain floats."""
    from baselines.static import make_static_policy
    from experiments.evaluate import evaluate_policy

    df = evaluate_policy(
        method, make_static_policy(method), seeds, cfg,
        results_dir=None, save=False, verbose=False,
    )
    numeric = df.select_dtypes(include=[np.number])
    return {
        c: float(numeric[c].sum())
        for c in sorted(numeric.columns)
        if not _is_timing(c)
    }


def _default_grid_cell_cfg(cfg):
    """Rebuild the grid cell that is supposed to equal the default scenario."""
    from experiments.e8_robustness_grid import _build_robustness_cfg

    return _build_robustness_cfg(
        cfg,
        arrival_rate=float(cfg.sim.arrivals.base_rate_per_minute),
        sla_tightness_triangular=list(cfg.sim.order_attrs.sla_due_triangular),
    )


def test_grid_default_cell_config_matches_base(cfg):
    """The overlay must be a no-op when it re-applies the base values.

    Cheap structural half of the check: if the two configs already differ, no
    amount of seed alignment will make the KPIs agree.
    """
    cell = _default_grid_cell_cfg(cfg)
    assert OmegaConf.to_container(cell, resolve=True) == OmegaConf.to_container(
        cfg, resolve=True
    ), "the 1.65/default grid cell is no longer identical to the base configuration"


def test_grid_default_cell_matches_default_scenario(cfg):
    """Same config, same seeds, same static rule -> byte-identical KPIs.

    This is the direct regression test for the Figure 6 / Table 1 contradiction.
    """
    from experiments.evaluate import canonical_test_seeds

    seeds = canonical_test_seeds(cfg)[:N_SEEDS]
    cell_cfg = _default_grid_cell_cfg(cfg)

    for rule in STATIC_RULES:
        base = _kpi_row(rule, seeds, cfg)
        cell = _kpi_row(rule, seeds, cell_cfg)
        assert base.keys() == cell.keys(), f"{rule}: KPI schema differs across paths"
        for k in base:
            assert base[k] == pytest.approx(cell[k], rel=0, abs=0), (
                f"{rule}: KPI {k!r} differs between the default scenario "
                f"({base[k]!r}) and the identically-configured grid cell "
                f"({cell[k]!r}). These two paths must not diverge."
            )


def test_static_rule_kpis_are_bit_reproducible(cfg):
    """Re-running the same evaluation must give bit-identical numbers."""
    from experiments.evaluate import canonical_test_seeds

    seeds = canonical_test_seeds(cfg)[:N_SEEDS]
    for rule in STATIC_RULES:
        first = _kpi_row(rule, seeds, cfg)
        second = _kpi_row(rule, seeds, cfg)
        for k in first:
            assert first[k] == second[k], (
                f"{rule}: KPI {k!r} is not reproducible across runs "
                f"({first[k]!r} then {second[k]!r})"
            )


def test_static_rule_kpis_are_seed_sensitive(cfg):
    """Guard against the above passing because the harness returns constants."""
    from experiments.evaluate import canonical_test_seeds

    seeds = canonical_test_seeds(cfg)
    a = _kpi_row("FIFO", seeds[:N_SEEDS], cfg)
    b = _kpi_row("FIFO", seeds[N_SEEDS : 2 * N_SEEDS], cfg)
    assert a != b, "KPIs are identical on disjoint seed blocks — seeding is inert"


def test_rollout_labelling_is_bit_reproducible(cfg):
    """The labelling estimator must be deterministic given its seeds.

    `branch()` resamples the unrealised future from a stream keyed on
    (shift, epoch, sample), so Monte Carlo averaging must not cost
    reproducibility. If this fails, no two campaign runs are comparable.
    """
    from labeling.rollout_labeler import label_one_shift_counted
    from simulation.heuristics import resolve_pool, with_default_scales

    pool = resolve_pool(cfg)
    local = with_default_scales(cfg)

    def run():
        rows, _ = label_one_shift_counted(
            shift_id=0,
            shift_seed=int(cfg.shifts.seed_seq_root),
            cfg=local,
            tau=2,
            n_samples=3,
            candidates=pool,
        )
        return np.array([[r[f"cost_{h}"] for h in pool] for r in rows], dtype=float)

    first, second = run(), run()
    assert first.shape == second.shape
    assert np.array_equal(first, second), (
        "rollout labelling is not bit-reproducible: identical seeds produced "
        "different cost vectors"
    )


def test_rollout_labelling_varies_with_sample_count(cfg):
    """Guard: the reproducibility test must not pass on a degenerate estimator.

    If `n_samples` did not actually change the estimate, the labels would be
    reproducible for the wrong reason — which is exactly the state the submitted
    single-path labeller was in, where rollout variance was identically zero.
    """
    from labeling.rollout_labeler import label_one_shift_counted
    from simulation.heuristics import resolve_pool, with_default_scales

    pool = resolve_pool(cfg)
    local = with_default_scales(cfg)
    cols = [f"cost_{h}" for h in pool]

    def run(m: int):
        rows, _ = label_one_shift_counted(
            shift_id=0,
            shift_seed=int(cfg.shifts.seed_seq_root),
            cfg=local,
            tau=2,
            n_samples=m,
            candidates=pool,
        )
        return np.array([[r[c] for c in cols] for r in rows], dtype=float)

    assert not np.array_equal(run(1), run(8)), (
        "averaging over 8 continuations gives the same costs as 1 — the labels "
        "are not Monte Carlo estimates"
    )


# --- The R2.4 aliasing witness (revision) -----------------------------------
#
# Section 3.2 answers Reviewer 2's partial-observability point with a
# constructive witness: two queues with identical phi that incur different cost
# under the same rule. The first implementation put a two-order queue in front of
# the deployed ten pickers, so both orders started at t=0 whatever the ranking
# said, every rule produced the identical trajectory, and the reported gap was
# 0.0000 — a witness that proved nothing, backing a claim it could not support.
#
# These pin the two halves of the claim: phi really coincides, and the cost
# really differs.


def test_aliasing_witness_exists_and_has_a_nonzero_gap(cfg):
    """At least one rule must admit a genuine witness."""
    from experiments.observability_analysis import aliasing_witness
    from simulation.heuristics import resolve_pool, with_default_scales

    local = with_default_scales(cfg)
    tau = int(local.labeling.tau)
    found = {
        r: w
        for r in resolve_pool(local)
        if (w := aliasing_witness(local, r, tau)).get("found")
    }
    assert found, (
        "no rule admits an aliasing witness — Section 3.2's POMDP claim is "
        "undemonstrated and must not be reported as measured"
    )
    for rule, w in found.items():
        assert w["phi_max_abs_diff"] <= 1e-9, f"{rule}: phi does not actually coincide"
        assert abs(w["cost_gap"]) > 1e-9, f"{rule}: reported found with a zero gap"


def test_aliasing_witness_requires_contention(cfg):
    """The witness must be built under contention, not with idle pickers.

    Characterisation test: it documents WHY the construction pins the picker
    count. With a picker free for every order the ranking cannot bind, so no
    rule can separate the two queues.
    """
    from experiments.observability_analysis import _env_with_queue, _mk
    from simulation.heuristics import resolve_pool, with_default_scales

    local = with_default_scales(cfg)
    tau = int(local.labeling.tau)
    qa = [_mk(0, 4.0, 30.0), _mk(1, 18.0, -5.0)]
    qb = [_mk(0, 4.0, -5.0), _mk(1, 18.0, 30.0)]

    for rule in resolve_pool(local):
        ea = _env_with_queue(local, qa, n_pickers=len(qa))
        eb = _env_with_queue(local, qb, n_pickers=len(qb))
        a0, b0 = ea.potential(), eb.potential()
        ea.run_with_policy(rule, n_steps=tau)
        eb.run_with_policy(rule, n_steps=tau)
        gap = (ea.potential() - a0) - (eb.potential() - b0)
        assert abs(gap) <= 1e-9, (
            f"{rule}: a gap appeared with one picker per order — the premise of "
            f"pinning the picker count no longer holds, so revisit the witness"
        )
