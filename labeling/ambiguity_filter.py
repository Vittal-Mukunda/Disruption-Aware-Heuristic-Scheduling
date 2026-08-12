"""Ambiguity filter for the test split.

Drops snapshots whose top-1 soft-label probability is too low to call a winner.
The training split keeps everything — the inverse-entropy sample weights already
down-weight ambiguous rows — but the test split is the evaluation surface, and
scoring a ranker against states with no defensible correct answer measures noise.

THE THRESHOLD IS A MULTIPLE OF UNIFORM, NOT AN ABSOLUTE PROBABILITY
------------------------------------------------------------------
The submitted config carried `theta_confidence: 0.55` in absolute probability,
chosen for a four-rule pool where uniform is 0.25 — so it meant "the top rule is
2.2x more likely than chance", and it kept 865 of 1600 test states (54%).

Reviewer 1 (4.d) doubled the pool to eight. Uniform becomes 0.125, and the same
absolute 0.55 silently means 4.4x uniform. Measured on a smoke corpus it kept
4 of 64 states — 6%. The test set would have been destroyed by the pool
expansion alone, and the paper would have reported a filtered corpus an order of
magnitude smaller without any deliberate change to the filter.

So the threshold is stored as a multiple of uniform and converted per pool size.
`theta_confidence_uniform_multiple: 2.2` reproduces the submitted filter's
meaning at any K.
"""

from __future__ import annotations

import numpy as np


def resolve_theta(cfg_ambiguity, n_rules: int) -> float:
    """The absolute top-1 probability threshold for a pool of `n_rules`.

    Accepts either the scale-free multiple (preferred) or a pre-revision
    absolute value, which is interpreted as having been set for |H| = 4.
    """
    if "theta_confidence_uniform_multiple" in cfg_ambiguity:
        mult = float(cfg_ambiguity.theta_confidence_uniform_multiple)
    else:
        mult = float(cfg_ambiguity.theta_confidence) * 4.0
    k = max(int(n_rules), 2)
    return float(np.clip(mult / k, 0.0, 1.0))


def filter_ambiguous(probs: np.ndarray, theta: float) -> np.ndarray:
    """Return a 1-D boolean mask of rows to *keep*.

    A row is kept iff `max(probs[row]) >= theta`. `theta` is absolute here;
    callers get it from `resolve_theta` so the pool size is accounted for.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
    return probs.max(axis=1) >= float(theta)
