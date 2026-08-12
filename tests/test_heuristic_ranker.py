"""Phase 4 — unit tests for the LDL-trained heuristic ranker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from models.heuristic_ranker import (
    FEATURE_COLUMNS,
    build_ldl_training_arrays,
    cross_validate_ranker,
    inverse_entropy_weights,
    pool_from_frame,
    prob_columns,
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
    # Stride by P's OWN width. Slicing by the module-level K silently misaligned
    # the blocks once the pool grew past the four columns this fixture builds.
    k = P.shape[1]
    for row in range(n):
        block = w_aug[row * k : (row + 1) * k]
        assert int(np.argmax(block)) == int(np.argmax(P[row]))
        assert list(y_aug[row * k : (row + 1) * k]) == list(range(k))


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
    P_true = df[prob_columns(df)].to_numpy(np.float64)
    xent = soft_xent(P_true, P_pred)
    # Uniform-prediction baseline: H(uniform) = log K. Model must beat it.
    assert xent < float(np.log(K)) - 0.2, f"model xent {xent} too close to uniform"


def test_class_set_comes_from_the_labels_not_the_config(cfg_ranker_small):
    """The pool is a property of the dataset (Reviewer 1, 4.d).

    The submitted code declared it in three places — `HEURISTIC_NAMES`,
    `cfg.ranker.num_class`, and the label columns — and guarded only the first
    two against each other. Dropping a rule from the labels must now simply
    produce a (K-1)-class model, with no config edit anywhere.
    """
    df = _make_easy_dataset()
    dropped = HEURISTIC_NAMES[-1]
    df_small = df.drop(columns=[f"p_{dropped}"])

    # Renormalise so the remaining soft labels still sum to one per row.
    cols = prob_columns(df_small)
    P = df_small[cols].to_numpy(np.float64)
    df_small[cols] = P / P.sum(axis=1, keepdims=True)

    assert pool_from_frame(df_small) == HEURISTIC_NAMES[:-1]
    result = cross_validate_ranker(df_small, cfg_ranker_small, seed=0)
    assert result.classes == HEURISTIC_NAMES[:-1]
    P_pred = result.model.predict_proba(
        df_small[result.feature_cols].to_numpy(np.float64)
    )
    assert P_pred.shape == (len(df_small), K - 1)


def test_missing_label_columns_raise(cfg_ranker_small):
    df = _make_easy_dataset().drop(columns=[f"p_{h}" for h in HEURISTIC_NAMES])
    with pytest.raises(ValueError):
        cross_validate_ranker(df, cfg_ranker_small, seed=0)
