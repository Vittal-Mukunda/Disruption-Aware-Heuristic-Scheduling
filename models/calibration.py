"""Per-class isotonic calibration of the multi:softprob ranker.

Phase 4 step 3. We fit the ranker (`models/heuristic_ranker.py`) on a
shift-disjoint *training* split and use a held-out *calibration* split
(`cfg.ranker.calibration.calibration_split`, default 20% of shifts) to
fit one isotonic regressor per class via `CalibratedClassifierCV(cv='prefit',
method='isotonic')`. Held-out shifts are required so calibration sees outputs
the model has never trained on.

We report Expected Calibration Error (ECE, top-1) and Brier score on:
  - the calibration split, pre- vs post-calibration (sanity check); and
  - the test parquet (`data/test.parquet`), which is the headline number.

ECE is the standard top-1 reliability gap binned on `max(probs)` vs.
`predicted_class == true_class`. Brier score is the mean squared error of the
predicted distribution against either the hard label (one-hot, default) or
the soft label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupShuffleSplit

from models.heuristic_ranker import FEATURE_COLUMNS, PROB_COLUMNS
from simulation.heuristics import HEURISTIC_NAMES


@dataclass
class CalibrationReport:
    """Pre/post calibration metrics on one split.

    `ece_pre` and `ece_post` are top-1 ECE; `brier_pre`/`brier_post` are
    Brier-MSE against the one-hot of `argmax(soft_label)`. `soft_xent_pre/post`
    is the cross-entropy against the soft labels themselves (used to confirm
    calibration didn't degrade the ranker's KL fit).
    """

    ece_pre: float
    ece_post: float
    brier_pre: float
    brier_post: float
    soft_xent_pre: float
    soft_xent_post: float
    n_rows: int


def split_calibration_shifts(
    df: pd.DataFrame,
    cal_frac: float,
    seed: int,
    group_col: str = "shift_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Index split of `df` into (train, cal) such that no `shift_id` appears in both."""
    if not (0.0 < cal_frac < 1.0):
        raise ValueError(f"cal_frac must be in (0,1), got {cal_frac}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cal_frac, random_state=seed)
    tr_idx, cal_idx = next(splitter.split(df, groups=df[group_col].to_numpy()))
    return tr_idx, cal_idx


def top1_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Top-1 Expected Calibration Error (binned reliability)."""
    if probs.shape[0] == 0:
        return 0.0
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(max_probs, bin_edges[1:-1], right=True), 0, n_bins - 1)
    total_n = probs.shape[0]
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        avg_conf = float(max_probs[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (n_b / total_n) * abs(avg_conf - avg_acc)
    return float(ece)


def brier_one_hot(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier = mean_i sum_k (p_pred_ik - y_onehot_ik)^2."""
    n, k = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def soft_cross_entropy(probs_pred: np.ndarray, probs_true: np.ndarray) -> float:
    probs_pred = np.clip(probs_pred, 1e-12, 1.0)
    return float(np.mean(-np.sum(probs_true * np.log(probs_pred), axis=1)))


class CalibratedRanker:
    """Wrap a prefit XGBClassifier with per-class isotonic calibration.

    Use `fit(df_calibration)` to fit the calibrator (the underlying ranker is
    NOT refit). Then `predict_proba(df)` returns the calibrated distribution.
    """

    def __init__(self, base_model, feature_cols: list[str]):
        self.base_model = base_model
        self.feature_cols = list(feature_cols)
        self._calibrator: CalibratedClassifierCV | None = None

    def fit(self, df_cal: pd.DataFrame) -> "CalibratedRanker":
        X_cal = df_cal[self.feature_cols].to_numpy(dtype=np.float64)
        P_cal = df_cal[PROB_COLUMNS].to_numpy(dtype=np.float64)
        y_cal = np.argmax(P_cal, axis=1).astype(np.int64)

        # sklearn >=1.6 deprecated `cv='prefit'`; wrap the fit model in
        # FrozenEstimator so CalibratedClassifierCV treats it as already trained.
        self._calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(self.base_model), method="isotonic"
        )
        self._calibrator.fit(X_cal, y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            return self.base_model.predict_proba(X)
        return self._calibrator.predict_proba(X)


def evaluate_calibration(
    ranker_cv,
    df_cal: pd.DataFrame,
    calibrator: CalibratedRanker | None,
) -> CalibrationReport:
    """Compute pre vs post calibration metrics on a held-out DataFrame.

    `ranker_cv` is the un-calibrated XGBoost model; `calibrator` is the
    fit `CalibratedRanker` (or None if you only want the pre-cal numbers).
    """
    X = df_cal[ranker_cv.feature_cols].to_numpy(dtype=np.float64)
    P_true = df_cal[PROB_COLUMNS].to_numpy(dtype=np.float64)
    y_hard = np.argmax(P_true, axis=1).astype(np.int64)

    probs_pre = ranker_cv.model.predict_proba(X)
    if calibrator is not None:
        probs_post = calibrator.predict_proba(X)
    else:
        probs_post = probs_pre

    return CalibrationReport(
        ece_pre=top1_ece(probs_pre, y_hard),
        ece_post=top1_ece(probs_post, y_hard),
        brier_pre=brier_one_hot(probs_pre, y_hard),
        brier_post=brier_one_hot(probs_post, y_hard),
        soft_xent_pre=soft_cross_entropy(probs_pre, P_true),
        soft_xent_post=soft_cross_entropy(probs_post, P_true),
        n_rows=len(df_cal),
    )


def cv_calibration_report(
    df_train: pd.DataFrame,
    fit_ranker_fn,
    cfg_ranker,
    seed: int,
    extra_feature_cols: list[str] | None = None,
) -> dict:
    """5-fold GroupKFold over training shifts; report mean ECE pre/post.

    Used to verify Phase 4's acceptance criterion (ECE < 0.05 in 5-fold CV).

    Parameters
    ----------
    df_train
        Training parquet with `shift_id`, 25 features, 4 soft-label cols.
    fit_ranker_fn
        Callable `(df_fold_train, cfg_ranker, seed, extra_feature_cols) ->
        RankerCVResult`. Typically `models.heuristic_ranker.cross_validate_ranker`
        with the inner CV reduced to a single split, OR a single XGB fit on the
        fold-train data using fixed hyperparameters.
    cfg_ranker
        `cfg.ranker` sub-tree.
    seed
        Base random state.
    extra_feature_cols
        Optional regime-posterior columns.
    """
    from sklearn.model_selection import GroupKFold

    feature_cols = list(FEATURE_COLUMNS) + list(extra_feature_cols or [])
    if cfg_ranker.cv.group_col not in df_train.columns:
        raise ValueError(f"DataFrame missing group column '{cfg_ranker.cv.group_col}'")
    groups = df_train[str(cfg_ranker.cv.group_col)].to_numpy()
    splitter = GroupKFold(n_splits=int(cfg_ranker.cv.n_splits))

    cal_frac = float(cfg_ranker.calibration.calibration_split)
    per_fold: list[dict] = []
    for fold_i, (tr_idx, va_idx) in enumerate(
        splitter.split(df_train, np.argmax(
            df_train[PROB_COLUMNS].to_numpy(np.float64), axis=1), groups=groups)
    ):
        df_fold_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_fold_va = df_train.iloc[va_idx].reset_index(drop=True)
        inner_tr, inner_cal = split_calibration_shifts(
            df_fold_tr, cal_frac, seed + fold_i,
            group_col=str(cfg_ranker.cv.group_col),
        )
        df_inner_tr = df_fold_tr.iloc[inner_tr].reset_index(drop=True)
        df_inner_cal = df_fold_tr.iloc[inner_cal].reset_index(drop=True)

        ranker_cv = fit_ranker_fn(
            df_inner_tr, cfg_ranker, seed + fold_i, extra_feature_cols
        )
        calibrator = CalibratedRanker(ranker_cv.model, ranker_cv.feature_cols).fit(
            df_inner_cal
        )
        report = evaluate_calibration(ranker_cv, df_fold_va, calibrator)
        per_fold.append({
            "fold": fold_i,
            "n_val": report.n_rows,
            "ece_pre": report.ece_pre,
            "ece_post": report.ece_post,
            "brier_pre": report.brier_pre,
            "brier_post": report.brier_post,
            "soft_xent_pre": report.soft_xent_pre,
            "soft_xent_post": report.soft_xent_post,
        })

    return {
        "per_fold": per_fold,
        "mean_ece_pre": float(np.mean([r["ece_pre"] for r in per_fold])),
        "mean_ece_post": float(np.mean([r["ece_post"] for r in per_fold])),
        "mean_brier_pre": float(np.mean([r["brier_pre"] for r in per_fold])),
        "mean_brier_post": float(np.mean([r["brier_post"] for r in per_fold])),
        "mean_soft_xent_post": float(np.mean([r["soft_xent_post"] for r in per_fold])),
        "K": len(HEURISTIC_NAMES),
    }
