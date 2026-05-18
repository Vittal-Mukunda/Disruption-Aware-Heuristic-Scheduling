"""Phase 3 — unit tests for the ambiguity filter."""

from __future__ import annotations

import numpy as np

from labeling.ambiguity_filter import filter_ambiguous


def test_drops_rows_below_theta():
    probs = np.array([
        [0.60, 0.20, 0.15, 0.05],  # max=0.60 -> keep at theta=0.55
        [0.30, 0.30, 0.25, 0.15],  # max=0.30 -> drop at theta=0.55
        [0.55, 0.20, 0.15, 0.10],  # max=0.55 -> keep at theta=0.55 (>= boundary)
    ])
    keep = filter_ambiguous(probs, theta=0.55)
    assert keep.tolist() == [True, False, True]


def test_empty_array_returns_empty_mask():
    probs = np.empty((0, 4))
    keep = filter_ambiguous(probs, theta=0.55)
    assert keep.shape == (0,)
    assert keep.dtype == bool
