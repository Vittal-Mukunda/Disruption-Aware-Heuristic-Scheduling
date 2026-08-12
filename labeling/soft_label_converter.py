"""Cost vector -> training label: tempered softmax, and the FEFO mask.

Converts the per-rule rollout cost vector of `labeling.rollout_labeler` into a
distribution over rules,

    p_h(s) = exp(-J_h(s) / beta) / sum_h' exp(-J_h'(s) / beta),

with the temperature `beta` chosen once so the median row entropy of the training
labels lands in `cfg.labeling.target_median_entropy`. Beta is searched over
`cfg.labeling.beta_grid` in units of sigma_J, the mean per-row standard deviation
of costs, so the grid is scale-free with respect to the objective's units.

Two changes from the submitted version.

VARIABLE POOL SIZE. Everything here was hard-wired to a four-rule pool through a
module-level constant. The pool is now whatever the Stage-1 screen retains
(`simulation.heuristics.resolve_pool`), so the number of classes is a parameter
and the FEFO column is located by name rather than by a fixed index. A mismatch
between the label width and the deployed pool is now an error rather than a silent
mis-alignment of probability columns.

THE MASK IS ABOUT THE PRODUCT CLOCK. FEFO ranks on `expiry_time` (Section 3.6), so
on a queue with essentially no perishables it has nothing to rank by and its mass
is moved to the rules that do. Under the submitted model, where "FEFO" sorted on
the due date, the mask was masking a due-date rule for the absence of perishables,
which made no mechanical sense — it happened to help only because the rule it
suppressed was redundant with EDD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

MASKED_RULE: str = "FEFO"


def _softmax_neg_costs(costs: np.ndarray, beta: float) -> np.ndarray:
    """Row-wise softmax of `-costs / beta`, numerically stable."""
    scaled = -costs / max(float(beta), 1e-12)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def row_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, in nats, treating 0*log(0) as 0."""
    safe = np.where(probs > 0.0, probs, 1.0)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=1)


def entropy_band(cfg_labeling: "DictConfig", n_rules: int) -> tuple[float, float]:
    """The target median-entropy band, in nats, for a pool of `n_rules`.

    THE BAND HAS TO SCALE WITH log|H|. The submitted config carried an absolute
    band of [0.3, 0.7] nats chosen for a four-rule pool, where max entropy is
    log 4 = 1.386 — so it targeted 22-51% of maximum. Reviewer 1 (4.d) doubled
    the pool to eight, where log 8 = 2.079, and the same absolute band silently
    becomes 14-34% of maximum. The temperature search would then drive the
    labels markedly sharper than intended, weakening exactly the soft-label
    signal the method rests on, purely as a side effect of adding rules.

    So the band is stored as a FRACTION of log|H| and converted here. The
    default reproduces the submitted band's meaning at whatever K is deployed.
    """
    if "target_median_entropy_ratio" in cfg_labeling:
        lo, hi = (float(x) for x in cfg_labeling.target_median_entropy_ratio)
    else:
        # Pre-revision config: absolute nats calibrated for |H| = 4.
        abs_lo, abs_hi = (float(x) for x in cfg_labeling.target_median_entropy)
        lo, hi = abs_lo / np.log(4.0), abs_hi / np.log(4.0)
    max_entropy = float(np.log(max(int(n_rules), 2)))
    return lo * max_entropy, hi * max_entropy


def costs_to_probs(
    cost_matrix: np.ndarray,
    cfg_labeling: "DictConfig",
    beta: float | None = None,
) -> tuple[np.ndarray, float]:
    """Tempered-softmax conversion. Returns `(probs, beta_used)`.

    Pass `beta` to reuse a temperature fitted elsewhere — the test and
    calibration corpora must use the training corpus's beta, or their labels are
    not on the same scale as the ones the ranker was fitted to.
    """
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2-D, got shape {cost_matrix.shape}")

    sigma_J = float(np.mean(cost_matrix.std(axis=1, ddof=0)))
    if sigma_J < 1e-12:
        # Every rule ties at every state: no information, return the uniform.
        return np.full(cost_matrix.shape, 1.0 / cost_matrix.shape[1]), 1.0

    if beta is not None:
        return _softmax_neg_costs(cost_matrix, float(beta)), float(beta)

    lo, hi = entropy_band(cfg_labeling, cost_matrix.shape[1])
    mid = 0.5 * (lo + hi)

    best_beta: float | None = None
    best_distance = float("inf")
    for mult in cfg_labeling.beta_grid:
        candidate = float(mult) * sigma_J
        probs = _softmax_neg_costs(cost_matrix, candidate)
        median_entropy = float(np.median(row_entropy(probs)))
        if lo <= median_entropy <= hi:
            return probs, candidate
        distance = abs(median_entropy - mid)
        if distance < best_distance:
            best_distance, best_beta = distance, candidate

    assert best_beta is not None
    return _softmax_neg_costs(cost_matrix, best_beta), best_beta


def fefo_mask(
    probs: np.ndarray,
    pct_perishable: np.ndarray,
    threshold: float,
    heuristic_names: list[str],
) -> np.ndarray:
    """Zero the expiry-aware rule on effectively non-perishable queues.

    `heuristic_names` gives the column order and must match `probs.shape[1]`. If
    the pool does not contain the masked rule, `probs` is returned unchanged —
    the Stage-1 screen may legitimately drop it.
    """
    if probs.shape[1] != len(heuristic_names):
        raise ValueError(
            f"probs has {probs.shape[1]} columns but the pool has "
            f"{len(heuristic_names)} rules {heuristic_names}; the label matrix "
            f"and the deployed pool have drifted apart."
        )
    if MASKED_RULE not in heuristic_names:
        return probs

    idx = heuristic_names.index(MASKED_RULE)
    rows = pct_perishable < float(threshold)
    if not np.any(rows):
        return probs

    out = probs.copy()
    out[rows, idx] = 0.0
    sums = out[rows].sum(axis=1, keepdims=True)

    safe = sums.copy()
    safe[safe < 1e-12] = 1.0
    out[rows] = out[rows] / safe

    # Guard: a row whose entire remaining mass was zero falls back to uniform
    # over the non-masked rules rather than to NaN.
    degenerate = sums.squeeze(-1) < 1e-12
    if np.any(degenerate):
        others = np.array([i != idx for i in range(len(heuristic_names))])
        bad = np.where(rows)[0][degenerate]
        out[bad] = 0.0
        out[np.ix_(bad, others)] = 1.0 / (len(heuristic_names) - 1)
    return out
