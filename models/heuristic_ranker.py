"""XGBoost `multi:softprob` ranker trained on rollout-induced soft labels.

The training set is `data/train.parquet`, with one row per snapshot and a
4-vector `[p_FIFO, p_FEFO, p_WSPT, p_ATC]` soft label. XGBoost's softprob
objective expects integer class labels, so we convert via the standard
*Label Distribution Learning row-replication* trick:

  Each row (x_i, p_i) becomes K replica rows (x_i, k, w_ik) for k = 0..K-1
  with replica weight  w_ik = p_ik * (1 / (1 + H(p_i)))  (normalized to mean 1).

This gives an effective objective of weighted cross-entropy
  -sum_i sum_k w_inv_i * p_ik * log q_ik,
which is KL(p || q) (up to a constant) plus inverse-entropy down-weighting of
ambiguous snapshots. See Geng (IEEE TKDE 2016, §4.2) for the standard
LDL -> softprob proxy; the inverse-entropy weighting is from the master plan
Phase 4 spec.

Cross-validation: 5-fold `GroupKFold` over `shift_id`. Hyperparameter grid is
`cfg.ranker.hyperparams` (default 3*3*2 = 18 combos). Scoring is the
weighted cross-entropy on the validation split, computed against the soft labels
(no replication of validation rows).

Optional regime-posterior features can be concatenated to the 25-D state
vector via `extra_feature_cols`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import FEATURE_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig

FEATURE_COLUMNS: list[str] = [f"f_{name}" for name in FEATURE_NAMES]
PROB_COLUMNS: list[str] = [f"p_{h}" for h in HEURISTIC_NAMES]


@dataclass
class RankerCVResult:
    """Outcome of `cross_validate_ranker`.

    Attributes
    ----------
    model
        Refit XGBClassifier on the full training data with `best_params`.
    best_params
        The hyperparameter dict that produced the lowest mean CV cross-entropy.
    best_mean_xent
        Mean weighted cross-entropy (in nats) across folds for `best_params`.
    per_fold_xent
        List of per-fold cross-entropy values for `best_params`.
    grid_results
        List of (params_dict, mean_xent) for the full HP search.
    feature_cols
        Exact column order used at fit time -- must be reused at inference.
    """

    model: xgb.XGBClassifier
    best_params: dict
    best_mean_xent: float
    per_fold_xent: list[float]
    grid_results: list[tuple[dict, float]] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.where(probs > 0.0, probs, 1.0)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=1)


def inverse_entropy_weights(probs: np.ndarray) -> np.ndarray:
    """Per-row weight `1 / (1 + H(p_row))`, normalized so mean weight = 1.

    Inputs sums-to-1 per row; entropy is in nats.
    """
    w = 1.0 / (1.0 + _row_entropy(probs))
    mean_w = float(w.mean())
    if mean_w <= 0.0:
        return np.ones_like(w)
    return w / mean_w


def build_ldl_training_arrays(
    X: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (X, soft P) into the K-replica (X_aug, y_aug, w_aug) for softprob.

    Each input row contributes K rows -- one per class -- with weight
    `p_ik * inv_entropy_i`. The final weights are normalized to mean 1 to
    keep the effective learning rate stable across hyperparameter combos.
    """
    n, k = P.shape
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows, P has {n} rows")

    w_inv = inverse_entropy_weights(P)
    X_aug = np.repeat(X, k, axis=0)
    y_aug = np.tile(np.arange(k, dtype=np.int64), n)
    w_per_replica = (P.reshape(-1)) * np.repeat(w_inv, k)

    mean_w = float(w_per_replica.mean())
    if mean_w > 0.0:
        w_per_replica = w_per_replica / mean_w

    return X_aug, y_aug, w_per_replica


def soft_xent(probs_true: np.ndarray, probs_pred: np.ndarray) -> float:
    """Mean cross-entropy `-sum_k p_true_k log p_pred_k` in nats."""
    probs_pred = np.clip(probs_pred, 1e-12, 1.0)
    per_row = -np.sum(probs_true * np.log(probs_pred), axis=1)
    return float(per_row.mean())


def _param_grid(hp_cfg: "DictConfig") -> list[dict]:
    keys = list(hp_cfg.keys())
    value_lists = [list(hp_cfg[k]) for k in keys]
    combos: list[dict] = []
    for combo in product(*value_lists):
        combos.append({k: v for k, v in zip(keys, combo, strict=True)})
    return combos


def _make_classifier(
    params: dict, num_class: int, seed: int
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=int(num_class),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=int(seed),
        verbosity=0,
        **params,
    )


def cross_validate_ranker(
    df: pd.DataFrame,
    cfg_ranker: "DictConfig",
    seed: int,
    extra_feature_cols: list[str] | None = None,
    n_jobs_fit: int = 1,
) -> RankerCVResult:
    """5-fold GroupKFold HP search; return the refit best model.

    Parameters
    ----------
    df
        Must contain the 25 `f_<name>` columns, the 4 `p_<H>` columns, and
        `shift_id` for the group split.
    cfg_ranker
        `cfg.ranker` sub-tree. Reads `num_class`, `cv.n_splits`, `cv.group_col`,
        and the `hyperparams` grid.
    seed
        Base random_state for XGBoost (use `SEED_MODEL`).
    extra_feature_cols
        Optional list of additional feature columns to concatenate to the
        canonical 25-D state (e.g., `regime_post_*`).
    n_jobs_fit
        n_jobs passed to each XGBClassifier. Default 1 keeps the call serial
        because the outer HP loop is already CPU-bound.
    """
    feature_cols = list(FEATURE_COLUMNS) + list(extra_feature_cols or [])
    missing = [c for c in feature_cols + PROB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing[:5]}...")
    if cfg_ranker.cv.group_col not in df.columns:
        raise ValueError(f"DataFrame missing group column '{cfg_ranker.cv.group_col}'")
    num_class = int(cfg_ranker.num_class)
    if num_class != len(HEURISTIC_NAMES):
        raise ValueError(
            f"cfg.ranker.num_class ({num_class}) != len(HEURISTIC_NAMES) "
            f"({len(HEURISTIC_NAMES)}). Phase 2 dropped CR; update config."
        )

    X = df[feature_cols].to_numpy(dtype=np.float64)
    P = df[PROB_COLUMNS].to_numpy(dtype=np.float64)
    groups = df[str(cfg_ranker.cv.group_col)].to_numpy()
    n_splits = int(cfg_ranker.cv.n_splits)

    grid = _param_grid(cfg_ranker.hyperparams)
    splitter = GroupKFold(n_splits=n_splits)
    hard_argmax = np.argmax(P, axis=1)
    folds = list(splitter.split(X, hard_argmax, groups=groups))

    grid_results: list[tuple[dict, float]] = []
    best_score: float = float("inf")
    best_params: dict | None = None
    best_fold_scores: list[float] = []

    for params in grid:
        fold_scores: list[float] = []
        params_for_xgb = {**params, "n_jobs": n_jobs_fit}
        for tr_idx, va_idx in folds:
            X_tr, P_tr = X[tr_idx], P[tr_idx]
            X_va, P_va = X[va_idx], P[va_idx]
            X_aug, y_aug, w_aug = build_ldl_training_arrays(X_tr, P_tr)
            clf = _make_classifier(params_for_xgb, num_class, seed)
            clf.fit(X_aug, y_aug, sample_weight=w_aug)
            P_pred = clf.predict_proba(X_va)
            fold_scores.append(soft_xent(P_va, P_pred))
        mean_score = float(np.mean(fold_scores))
        grid_results.append((dict(params), mean_score))
        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)
            best_fold_scores = list(fold_scores)

    assert best_params is not None
    X_aug, y_aug, w_aug = build_ldl_training_arrays(X, P)
    final_clf = _make_classifier(
        {**best_params, "n_jobs": n_jobs_fit}, num_class, seed
    )
    final_clf.fit(X_aug, y_aug, sample_weight=w_aug)

    return RankerCVResult(
        model=final_clf,
        best_params=best_params,
        best_mean_xent=best_score,
        per_fold_xent=best_fold_scores,
        grid_results=grid_results,
        feature_cols=feature_cols,
    )


def predict_proba(
    model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    """Run `model.predict_proba` on `df[feature_cols]`. Returns (n, K) float64."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return model.predict_proba(X)
