"""Phase 3 — unit tests for soft-label conversion and FEFO masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from labeling.soft_label_converter import (
    costs_to_probs,
    entropy_band,
    fefo_mask,
    row_entropy,
)
from simulation.heuristics import HEURISTIC_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEFO_IDX = HEURISTIC_NAMES.index("FEFO")
K = len(HEURISTIC_NAMES)


@pytest.fixture(scope="module")
def cfg_labeling():
    cfg = OmegaConf.load(CONFIG_PATH)
    return cfg.labeling


def test_probs_sum_to_one(cfg_labeling):
    rng = np.random.default_rng(0)
    costs = rng.uniform(0, 10, size=(64, K))
    probs, _ = costs_to_probs(costs, cfg_labeling)
    sums = probs.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9), f"row sums off: {sums.min()}..{sums.max()}"


def test_lower_cost_gets_higher_prob(cfg_labeling):
    rng = np.random.default_rng(1)
    costs = rng.uniform(0, 10, size=(64, K))
    probs, _ = costs_to_probs(costs, cfg_labeling)
    # Argmin of cost should equal argmax of prob, row by row.
    assert np.array_equal(costs.argmin(axis=1), probs.argmax(axis=1))


def test_beta_search_hits_target_band_on_realistic_costs(cfg_labeling):
    # Simulate costs that span an O(1)..O(10) range, similar to pilot scale.
    rng = np.random.default_rng(2)
    costs = rng.uniform(0, 6, size=(2000, K))
    probs, beta = costs_to_probs(costs, cfg_labeling)
    median = float(np.median(row_entropy(probs)))
    # The band scales with log|H|; reading it in absolute nats made this test
    # pass at |H|=4 and fail at |H|=8 for no reason but the pool size.
    lo, hi = entropy_band(cfg_labeling, K)
    # The search returns either an in-band beta or the closest-to-midpoint one;
    # on realistic costs there should always be an in-band candidate in beta_grid.
    assert lo <= median <= hi, f"median entropy {median} not in [{lo}, {hi}]; beta={beta}"


def test_entropy_band_scales_with_pool_size(cfg_labeling):
    """Doubling the pool must not silently sharpen the labels (Reviewer 1, 4.d)."""
    lo4, hi4 = entropy_band(cfg_labeling, 4)
    lo8, hi8 = entropy_band(cfg_labeling, 8)
    # Same fraction of maximum entropy at either width.
    assert lo4 / np.log(4) == pytest.approx(lo8 / np.log(8))
    assert hi4 / np.log(4) == pytest.approx(hi8 / np.log(8))
    assert hi8 > hi4


def test_locked_beta_path(cfg_labeling):
    rng = np.random.default_rng(3)
    costs = rng.uniform(0, 10, size=(32, K))
    probs_a, beta_a = costs_to_probs(costs, cfg_labeling)
    probs_b, beta_b = costs_to_probs(costs, cfg_labeling, beta=beta_a)
    assert beta_a == beta_b
    assert np.allclose(probs_a, probs_b, atol=1e-12)


def test_degenerate_zero_variance_returns_uniform(cfg_labeling):
    costs = np.full((10, K), 1.234)
    probs, _ = costs_to_probs(costs, cfg_labeling)
    expected = np.full((10, K), 1.0 / K)
    assert np.allclose(probs, expected)


def test_fefo_mask_zeroes_below_threshold():
    uniform = np.full(K, 1.0 / K)
    skewed = np.full(K, 0.5 / (K - 1))
    skewed[0] = 0.5
    probs = np.vstack([uniform, skewed])
    pct_perish = np.array([0.0, 0.5])   # row 0 below threshold, row 1 above
    out = fefo_mask(probs, pct_perish, threshold=0.05, heuristic_names=HEURISTIC_NAMES)

    # Row 0: FEFO -> 0, others renormalised over the remaining mass.
    assert out[0, FEFO_IDX] == 0.0
    assert np.isclose(out[0].sum(), 1.0)
    # Row 1: unchanged.
    assert np.allclose(out[1], probs[1])


def test_fefo_mask_handles_degenerate_row():
    """A row that is ~all FEFO must fall back to uniform, not to NaN."""
    probs = np.zeros((1, K))
    probs[0, FEFO_IDX] = 1.0
    out = fefo_mask(probs, np.array([0.0]), threshold=0.05,
                    heuristic_names=HEURISTIC_NAMES)
    assert np.isclose(out[0, FEFO_IDX], 0.0)
    assert np.isclose(out[0].sum(), 1.0)
    assert np.isfinite(out).all()


def test_fefo_mask_threshold_boundary():
    # pct_perishable exactly at threshold should NOT be masked (strict <).
    probs = np.full((1, K), 1.0 / K)
    out = fefo_mask(probs, np.array([0.05]), threshold=0.05,
                    heuristic_names=HEURISTIC_NAMES)
    assert np.allclose(out, probs)


def test_fefo_mask_is_a_noop_when_the_rule_was_screened_out():
    """The Stage-1 screen may legitimately drop FEFO (Reviewer 1, 4.d)."""
    pool = [h for h in HEURISTIC_NAMES if h != "FEFO"]
    probs = np.full((2, len(pool)), 1.0 / len(pool))
    out = fefo_mask(probs, np.array([0.0, 0.0]), threshold=0.05,
                    heuristic_names=pool)
    assert np.array_equal(out, probs)


def test_fefo_mask_rejects_a_pool_of_the_wrong_width():
    """A label matrix and a pool that disagree is an error, not a silent shift."""
    probs = np.full((1, K), 1.0 / K)
    with pytest.raises(ValueError):
        fefo_mask(probs, np.array([0.0]), threshold=0.05,
                  heuristic_names=HEURISTIC_NAMES[:-1])


# --- Per-row tempering (revision) -------------------------------------------
#
# The corrected objective charges unserved-and-overdue orders, so a state's cost
# scale grows with its outstanding work and the per-row spread of the cost vector
# varies by two orders of magnitude across a shift. A single global temperature
# is set by the loud rows and leaves the quiet ones near-uniform, which is what
# drove the median label entropy out of its target band. These tests pin the
# per-row construction that fixes it, and pin `global` as still reproducible.


def _heteroscedastic_costs(n_rows: int = 512, seed: int = 7) -> np.ndarray:
    """Cost vectors whose per-row spread ramps over three orders of magnitude.

    This is the shape the corrected objective produces: early epochs cost little
    and the rules differ by little; late epochs cost a lot and differ by a lot.
    The *relative* ordering strength is held fixed, so a correct temperature
    should give every row a comparable label entropy.
    """
    rng = np.random.default_rng(seed)
    shape = rng.standard_normal((n_rows, K))
    scale = np.geomspace(0.1, 100.0, n_rows).reshape(-1, 1)
    return shape * scale


def test_per_row_tempering_equalises_entropy_across_cost_scales(cfg_labeling):
    """Rows differing only in cost SCALE must get comparable label entropy."""
    costs = _heteroscedastic_costs()
    per_row = OmegaConf.merge(cfg_labeling, {"beta_mode": "per_row"})
    probs, _ = costs_to_probs(costs, per_row)

    ent = row_entropy(probs)
    quiet, loud = ent[:64], ent[-64:]
    # Under per-row scaling the two decades of rows are on the same footing.
    assert abs(float(quiet.mean()) - float(loud.mean())) < 0.15, (
        f"per-row tempering left a scale gradient: quiet={quiet.mean():.3f} "
        f"loud={loud.mean():.3f}"
    )


def test_global_tempering_leaves_a_scale_gradient(cfg_labeling):
    """The submitted construction: the same data comes out scale-dependent.

    This is a characterisation test, not an aspiration — it documents WHY the
    default changed. If it ever fails, the premise of the change is wrong.
    """
    costs = _heteroscedastic_costs()
    glob = OmegaConf.merge(cfg_labeling, {"beta_mode": "global"})
    probs, _ = costs_to_probs(costs, glob)

    ent = row_entropy(probs)
    quiet, loud = ent[:64], ent[-64:]
    assert float(quiet.mean()) - float(loud.mean()) > 0.5, (
        "global tempering no longer shows the scale gradient it was changed for"
    )


def test_per_row_lands_median_entropy_in_band(cfg_labeling):
    """The band is reachable on heteroscedastic costs — the smoke-run failure."""
    costs = _heteroscedastic_costs()
    per_row = OmegaConf.merge(cfg_labeling, {"beta_mode": "per_row"})
    probs, _ = costs_to_probs(costs, per_row)
    lo, hi = entropy_band(per_row, K)
    median = float(np.median(row_entropy(probs)))
    assert lo <= median <= hi, f"median entropy {median:.4f} outside [{lo}, {hi}]"


def test_beta_transfers_between_corpora_under_per_row(cfg_labeling):
    """Test labels must reuse the training multiplier, not refit their own."""
    costs_a = _heteroscedastic_costs(seed=11)
    costs_b = _heteroscedastic_costs(seed=12)
    per_row = OmegaConf.merge(cfg_labeling, {"beta_mode": "per_row"})
    _, beta = costs_to_probs(costs_a, per_row)
    probs_b, beta_b = costs_to_probs(costs_b, per_row, beta=beta)
    assert beta_b == beta
    assert np.allclose(probs_b.sum(axis=1), 1.0)
    assert np.array_equal(costs_b.argmin(axis=1), probs_b.argmax(axis=1))


def test_tied_rows_return_uniform_not_nan(cfg_labeling):
    """A row where every rule ties has sigma = 0 and must not divide by zero."""
    costs = _heteroscedastic_costs(n_rows=128, seed=3)
    costs[::8, :] = 4.2  # every rule identical on these rows
    per_row = OmegaConf.merge(cfg_labeling, {"beta_mode": "per_row"})
    probs, _ = costs_to_probs(costs, per_row)
    assert np.isfinite(probs).all(), "per-row temperature produced non-finite labels"
    assert np.allclose(probs[::8], 1.0 / K), "tied rows should be uniform"


def test_unknown_beta_mode_is_rejected(cfg_labeling):
    bad = OmegaConf.merge(cfg_labeling, {"beta_mode": "sqrt"})
    with pytest.raises(ValueError, match="beta_mode"):
        costs_to_probs(_heteroscedastic_costs(n_rows=32), bad)
