"""Ambiguity filter for the test split.

Drops snapshots where the top-1 soft-label probability is below `theta`
(default 0.55 — see `cfg.labeling.ambiguity_filter.theta_confidence`). The
training split keeps everything; the inverse-entropy sample weights used in
Phase 4 downweight ambiguous rows already. The test set, however, is the
ground-truth evaluation surface and we want to score the ranker against
snapshots that have a defensibly "correct" answer.
"""

from __future__ import annotations

import numpy as np


def filter_ambiguous(probs: np.ndarray, theta: float) -> np.ndarray:
    """Return a 1-D boolean mask of rows to *keep*.

    A row is kept iff `max(probs[row]) >= theta`.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
    return probs.max(axis=1) >= float(theta)
