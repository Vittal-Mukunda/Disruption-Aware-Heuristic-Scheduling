"""XGBoost `multi:softprob` ranker trained on rollout-induced soft labels.

The training set is `data/train.parquet`, one row per decision epoch with a
`|H|`-vector soft label `[p_<rule> ...]`. XGBoost's softprob objective expects
integer class labels, so we convert via the standard *Label Distribution
Learning row-replication* trick:

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

Optional regime-posterior features can be concatenated to the state vector via
`extra_feature_cols`.

THE CLASS SET COMES FROM THE DATA (revision, Reviewer 1, 4.d)
------------------------------------------------------------
The submitted module hard-wired a four-rule pool: `PROB_COLUMNS` was built from
the module-level `HEURISTIC_NAMES` and the class count was read from a separate
`cfg.ranker.num_class` key, with a guard that raised when the two disagreed.
Two independent declarations of the same fact, and neither was the labels.

The pool is now whatever the Stage-1 screen retained, so it is a property of the
dataset rather than of the config. `prob_columns()` reads it off the frame in
the order `experiments/generate_labels.py` wrote it, which is the order that
defines the classifier's class indices. A config edited between labelling and
training can no longer silently permute the classes, because the config is no
longer consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from simulation.state_extractor import FEATURE_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig

FEATURE_COLUMNS: list[str] = [f"f_{name}" for name in FEATURE_NAMES]

PROB_PREFIX: str = "p_"


def prob_columns(df: pd.DataFrame) -> list[str]:
    """The soft-label columns, in the order the labeller wrote them.

    That order IS the classifier's class indexing, so it must be read from the
    frame rather than re-derived from a config that may have been edited since.
    """
    cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    if not cols:
        raise ValueError(
            f"No '{PROB_PREFIX}<rule>' soft-label columns in the frame "
            f"(columns: {list(df.columns)[:12]}...). Run "
            f"`python -m experiments.generate_labels` first."
        )
    return cols


def pool_from_frame(df: pd.DataFrame) -> list[str]:
    """The deployed rule pool, recovered from the label columns."""
    return [c[len(PROB_PREFIX):] for c in prob_columns(df)]


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
    classes
        Rule names in class-index order, recovered from the label columns. The
        deployment wrapper maps `argmax` back to a rule name through this, so it
        must travel with the model rather than being re-derived from config.
    """

    model: xgb.XGBClassifier
    best_params: dict
    best_mean_xent: float
    per_fold_xent: list[float]
    grid_results: list[tuple[dict, float]] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)


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
    base_feature_cols: list[str] | None = None,
) -> RankerCVResult:
    """5-fold GroupKFold HP search; return the refit best model.

    Parameters
    ----------
    df
        Must contain the `f_<name>` feature columns, the `p_<rule>` soft-label
        columns, and `shift_id` for the group split.
    cfg_ranker
        `cfg.ranker` sub-tree. Reads `cv.n_splits`, `cv.group_col`, and the
        `hyperparams` grid. The class count is NOT read from config — it is the
        number of soft-label columns in `df`.
    seed
        Base random_state for XGBoost (use `SEED_MODEL`).
    extra_feature_cols
        Optional list of additional feature columns to concatenate to the
        canonical state (e.g., `regime_post_*`).
    n_jobs_fit
        n_jobs passed to each XGBClassifier. Default 1 keeps the call serial
        because the outer HP loop is already CPU-bound.
    base_feature_cols
        Override the canonical state columns. Used by the parsimony ablation
        Reviewer 3 (4) asks for, which refits on a top-k subset; leaving this
        `None` uses the full feature map.
    """
    feature_cols = (
        list(base_feature_cols) if base_feature_cols is not None
        else list(FEATURE_COLUMNS)
    ) + list(extra_feature_cols or [])
    prob_cols = prob_columns(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing feature columns: {missing[:5]}...")
    if cfg_ranker.cv.group_col not in df.columns:
        raise ValueError(f"DataFrame missing group column '{cfg_ranker.cv.group_col}'")
    num_class = len(prob_cols)
    if num_class < 2:
        raise ValueError(
            f"Only {num_class} rule(s) in the label set {prob_cols}; the "
            f"selection problem is degenerate. Check the Stage-1 screen."
        )

    X = df[feature_cols].to_numpy(dtype=np.float64)
    P = df[prob_cols].to_numpy(dtype=np.float64)
    groups = df[str(cfg_ranker.cv.group_col)].to_numpy()
    n_groups = int(len(np.unique(groups)))
    n_splits = int(cfg_ranker.cv.n_splits)

    # CLAMP TO THE NUMBER OF SHIFTS. Folds are grouped on `shift_id` so a corpus
    # of n shifts admits at most n folds, and `GroupKFold` raises rather than
    # degrading when asked for more. `--smoke` shrinks the hyperparameter grid
    # but not `cv.n_splits`, so the documented smoke gate — the thing the
    # campaign runs BEFORE committing to the expensive stages — died here on a
    # 3-shift corpus with `n_splits=5 > n_groups=3`. The data-efficiency sweep
    # retrains at budgets as small as 10 shifts and would hit the same edge if
    # the smallest budget were ever lowered.
    #
    # Clamping is right rather than merely convenient: k-fold with k = n_groups
    # is leave-one-shift-out, which is the strictest grouped validation the
    # corpus supports. The count is recorded so a run cannot silently claim
    # 5-fold when it did something else.
    if n_groups < 2:
        raise ValueError(
            f"grouped cross-validation needs at least 2 distinct "
            f"'{cfg_ranker.cv.group_col}' values, found {n_groups}. A single "
            f"shift cannot be split into training and validation folds."
        )
    if n_splits > n_groups:
        print(
            f"[ranker] cv.n_splits={n_splits} exceeds the {n_groups} shifts in "
            f"this corpus; using {n_groups}-fold (leave-one-shift-out)."
        )
        n_splits = n_groups

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
        classes=[c[len(PROB_PREFIX):] for c in prob_cols],
    )


def predict_proba(
    model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    """Run `model.predict_proba` on `df[feature_cols]`. Returns (n, K) float64."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return model.predict_proba(X)
