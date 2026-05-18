"""Phase 4 — unit tests for the LDL-trained heuristic ranker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from models.heuristic_ranker import (
    FEATURE_COLUMNS,
    PROB_COLUMNS,
    build_ldl_training_arrays,
    cross_validate_ranker,
    inverse_entropy_weights,
    soft_xent,
)
from simulation.heuristics import HEURISTIC_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
K = len(HEURISTIC_NAMES)


@pytest.fixture(scope="module")
def cfg_ranker_small():
    cfg = OmegaConf.load(CONFIG_PATH)
    # Shrink the HP grid to keep tests fast.
    cfg.ranker.hyperparams = OmegaConf.create({
        "max_depth": [3],
        "n_estimators": [40],
        "learning_rate": [0.2],
    })
    cfg.ranker.cv.n_splits = 2  # 2-fold for speed
    return cfg.ranker


def _make_easy_dataset(n_shifts: int = 6, n_per: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic data where one feature axis perfectly predicts argmax class."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for sid in range(n_shifts):
        for t in range(n_per):
            # Pick a target class deterministically by t mod K.
            cls = t % K
            feats = rng.normal(0.0, 0.5, size=len(FEATURE_COLUMNS))
            # Make feature 0 a strong proxy for the class.
            feats[0] = float(cls) + rng.normal(0.0, 0.05)
            probs = np.full(K, 0.05)
            probs[cls] = 1.0 - 0.05 * (K - 1)
            row: dict = {"shift_id": sid, "interval_idx": t}
            for i, c in enumerate(FEATURE_COLUMNS):
                row[c] = float(feats[i])
            for j, h in enumerate(HEURISTIC_NAMES):
                row[f"p_{h}"] = float(probs[j])
            rows.append(row)
    return pd.DataFrame(rows)


def test_inverse_entropy_weights_normalized():
    probs = np.array([
        [0.7, 0.1, 0.1, 0.1],   # low entropy -> higher raw weight
        [0.25, 0.25, 0.25, 0.25],  # max entropy -> lowest raw weight
    ])
    w = inverse_entropy_weights(probs)
    assert w.shape == (2,)
    assert np.isclose(float(w.mean()), 1.0)
    assert w[0] > w[1]


def test_build_ldl_training_arrays_shapes():
    n, d = 5, len(FEATURE_COLUMNS)
    X = np.zeros((n, d))
    P = np.full((n, K), 1.0 / K)
    X_aug, y_aug, w_aug = build_ldl_training_arrays(X, P)
    assert X_aug.shape == (n * K, d)
    assert y_aug.shape == (n * K,)
    assert w_aug.shape == (n * K,)
    # Each "original row" contributes exactly K replicas of each class label.
    assert set(np.unique(y_aug)) == set(range(K))


def test_build_ldl_training_arrays_weights_track_prob():
    """For a row with high-prob class k, the replica with y=k must have the largest weight."""
    n, d = 3, len(FEATURE_COLUMNS)
    X = np.zeros((n, d))
    P = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ])
    _, y_aug, w_aug = build_ldl_training_arrays(X, P)
    for row in range(n):
        block = w_aug[row * K : (row + 1) * K]
        assert int(np.argmax(block)) == int(np.argmax(P[row]))


def test_cross_validate_ranker_predicts_valid_distribution(cfg_ranker_small):
    df = _make_easy_dataset()
    result = cross_validate_ranker(df, cfg_ranker_small, seed=42)
    X = df[result.feature_cols].to_numpy(np.float64)
    P_pred = result.model.predict_proba(X)
    assert P_pred.shape == (len(df), K)
    assert np.allclose(P_pred.sum(axis=1), 1.0, atol=1e-6)


def test_cross_validate_ranker_learns_clear_signal(cfg_ranker_small):
    """On the synthetic dataset, soft-xent must be much lower than uniform CE."""
    df = _make_easy_dataset()
    result = cross_validate_ranker(df, cfg_ranker_small, seed=42)
    X = df[result.feature_cols].to_numpy(np.float64)
    P_pred = result.model.predict_proba(X)
    P_true = df[PROB_COLUMNS].to_numpy(np.float64)
    xent = soft_xent(P_true, P_pred)
    # Uniform-prediction baseline: H(uniform) = log K. Model must beat it.
    assert xent < float(np.log(K)) - 0.2, f"model xent {xent} too close to uniform"


def test_num_class_mismatch_raises(cfg_ranker_small):
    bad_cfg = OmegaConf.merge(cfg_ranker_small, OmegaConf.create({"num_class": K + 1}))
    df = _make_easy_dataset()
    with pytest.raises(ValueError):
        cross_validate_ranker(df, bad_cfg, seed=0)


def test_num_class_matches_heuristic_names():
    cfg = OmegaConf.load(CONFIG_PATH)
    assert int(cfg.ranker.num_class) == len(HEURISTIC_NAMES)
