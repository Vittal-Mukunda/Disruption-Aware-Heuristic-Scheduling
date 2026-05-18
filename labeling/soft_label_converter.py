"""Tempered softmax cost-to-probability conversion + FEFO mask.

The training labels are *soft*: each snapshot gets a probability distribution
over the heuristic pool, not a single argmin. This module converts the cost
matrix `J_h(s_t)` to probabilities via

    p_h(s_t) = exp(-J_h(s_t) / beta) / sum_h' exp(-J_h'(s_t) / beta)

with a temperature `beta` chosen so the median row-entropy of the resulting
training distribution lies in `cfg.labeling.target_median_entropy` (default
[0.3, 0.7]). Beta is searched over `cfg.labeling.beta_grid * sigma_J`, where
sigma_J is the per-row standard deviation of costs averaged across rows.

A second post-processing step is the FEFO mask: rows where the queue is
essentially non-perishable (`pct_perishable < cfg.heuristics.fefo_mask_threshold`)
have `p_FEFO` zeroed out and the remainder renormalized. Same mask must be
applied at inference time — see gotcha #4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from simulation.heuristics import HEURISTIC_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _softmax_neg_costs(costs: np.ndarray, beta: float) -> np.ndarray:
    """Row-wise softmax of `-costs / beta`, numerically stable."""
    scaled = -costs / max(float(beta), 1e-12)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, in nats. Zeros are treated as 0*log(0)=0."""
    safe = np.where(probs > 0.0, probs, 1.0)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=1)


def costs_to_probs(
    cost_matrix: np.ndarray,
    cfg_labeling: "DictConfig",
    beta: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert a cost matrix to soft labels via tempered softmax.

    Parameters
    ----------
    cost_matrix
        Float array of shape (n_rows, n_heuristics). Column order must match
        `simulation.heuristics.HEURISTIC_NAMES`.
    cfg_labeling
        The `labeling` sub-tree of `config.yaml`. Reads `beta_grid` (multipliers
        of sigma_J to try) and `target_median_entropy` ([lo, hi] interval).
    beta
        If provided, skip the search and use this value directly. Used for the
        test split (we lock in the train-derived beta for label consistency).

    Returns
    -------
    (probs, beta_chosen)
        `probs` has the same shape as `cost_matrix`, each row sums to 1.
        `beta_chosen` is the temperature used.
    """
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2-D, got shape {cost_matrix.shape}")
    if cost_matrix.shape[1] != len(HEURISTIC_NAMES):
        raise ValueError(
            f"cost_matrix has {cost_matrix.shape[1]} columns; "
            f"expected {len(HEURISTIC_NAMES)} (HEURISTIC_NAMES order)."
        )

    # Average per-row std of costs; gives a problem-scale temperature unit.
    sigma_J = float(np.mean(cost_matrix.std(axis=1, ddof=0)))
    if sigma_J < 1e-12:
        # All rows are degenerate ties; any beta produces the uniform.
        probs = np.full(cost_matrix.shape, 1.0 / cost_matrix.shape[1])
        return probs, 1.0

    if beta is not None:
        probs = _softmax_neg_costs(cost_matrix, float(beta))
        return probs, float(beta)

    target_lo, target_hi = (float(x) for x in cfg_labeling.target_median_entropy)
    target_mid = 0.5 * (target_lo + target_hi)

    best_beta: float | None = None
    best_distance: float = float("inf")
    for mult in cfg_labeling.beta_grid:
        beta_candidate = float(mult) * sigma_J
        probs_candidate = _softmax_neg_costs(cost_matrix, beta_candidate)
        median_entropy = float(np.median(_row_entropy(probs_candidate)))
        if target_lo <= median_entropy <= target_hi:
            return probs_candidate, beta_candidate
        distance = abs(median_entropy - target_mid)
        if distance < best_distance:
            best_distance = distance
            best_beta = beta_candidate

    assert best_beta is not None  # beta_grid is non-empty
    probs = _softmax_neg_costs(cost_matrix, best_beta)
    return probs, best_beta


def fefo_mask(
    probs: np.ndarray,
    pct_perishable: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Zero `p_FEFO` for low-perishable rows and renormalize the remainder.

    Rows where every other heuristic also gets probability ~0 (shouldn't happen
    with softmax, but guard anyway) fall back to uniform-over-non-FEFO.
    """
    if probs.shape[1] != len(HEURISTIC_NAMES):
        raise ValueError(
            f"probs has {probs.shape[1]} cols; expected {len(HEURISTIC_NAMES)}"
        )
    if "FEFO" not in HEURISTIC_NAMES:
        return probs
    fefo_idx = HEURISTIC_NAMES.index("FEFO")
    mask_rows = pct_perishable < float(threshold)
    if not np.any(mask_rows):
        return probs

    out = probs.copy()
    out[mask_rows, fefo_idx] = 0.0
    row_sums = out[mask_rows].sum(axis=1, keepdims=True)

    # Guard against degenerate rows where everything would be zero.
    safe = row_sums.copy()
    safe[safe < 1e-12] = 1.0
    out[mask_rows] = out[mask_rows] / safe

    degenerate = (row_sums.squeeze(-1) < 1e-12)
    if np.any(degenerate):
        n_other = len(HEURISTIC_NAMES) - 1
        idx_mask = np.array(
            [i != fefo_idx for i in range(len(HEURISTIC_NAMES))]
        )
        # Find the absolute row indices in `out` that are degenerate.
        mask_row_ids = np.where(mask_rows)[0][degenerate]
        out[mask_row_ids] = 0.0
        out[np.ix_(mask_row_ids, idx_mask)] = 1.0 / n_other

    return out
