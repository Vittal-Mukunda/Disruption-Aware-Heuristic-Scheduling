"""Phase 3 — unit tests for soft-label conversion and FEFO masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from labeling.soft_label_converter import (
    _row_entropy,
    costs_to_probs,
    fefo_mask,
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
    median = float(np.median(_row_entropy(probs)))
    lo, hi = (float(x) for x in cfg_labeling.target_median_entropy)
    # The search returns either an in-band beta or the closest-to-midpoint one;
    # on realistic costs there should always be an in-band candidate in beta_grid.
    assert lo <= median <= hi, f"median entropy {median} not in [{lo}, {hi}]; beta={beta}"


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
    probs = np.array([
        [0.25, 0.25, 0.25, 0.25],   # below threshold -> FEFO zeroed
        [0.10, 0.40, 0.30, 0.20],   # above threshold -> untouched
    ])
    pct_perish = np.array([0.0, 0.5])
    out = fefo_mask(probs, pct_perish, threshold=0.05)

    # Row 0: FEFO -> 0, others renormalized over the remaining mass.
    assert out[0, FEFO_IDX] == 0.0
    assert np.isclose(out[0].sum(), 1.0)
    # Row 1: unchanged.
    assert np.allclose(out[1], probs[1])


def test_fefo_mask_handles_degenerate_row():
    """If softmax produces a row that's ~all FEFO, masking should fall back gracefully."""
    probs = np.zeros((1, K))
    probs[0, FEFO_IDX] = 1.0
    out = fefo_mask(probs, np.array([0.0]), threshold=0.05)
    assert np.isclose(out[0, FEFO_IDX], 0.0)
    assert np.isclose(out[0].sum(), 1.0)


def test_fefo_mask_threshold_boundary():
    # pct_perishable exactly at threshold should NOT be masked (strict <).
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    out = fefo_mask(probs, np.array([0.05]), threshold=0.05)
    assert np.allclose(out, probs)
