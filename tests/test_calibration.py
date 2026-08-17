"""Phase 4 — unit tests for isotonic calibration and reliability metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from models.calibration import (
    CalibratedRanker,
    brier_one_hot,
    evaluate_calibration,
    split_calibration_shifts,
    top1_ece,
)
from models.heuristic_ranker import (
    FEATURE_COLUMNS,
    cross_validate_ranker,
)
from simulation.heuristics import HEURISTIC_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
K = len(HEURISTIC_NAMES)


@pytest.fixture(scope="module")
def cfg_ranker_small():
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.ranker.hyperparams = OmegaConf.create({
        "max_depth": [3],
        "n_estimators": [40],
        "learning_rate": [0.2],
    })
    cfg.ranker.cv.n_splits = 2
    return cfg.ranker


def _make_dataset(n_shifts: int = 12, n_per: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for sid in range(n_shifts):
        for t in range(n_per):
            cls = (sid + t) % K
            feats = rng.normal(0.0, 0.5, size=len(FEATURE_COLUMNS))
            feats[0] = float(cls) + rng.normal(0.0, 0.1)
            probs = np.full(K, 0.05); probs[cls] = 1.0 - 0.05 * (K - 1)
            row = {"shift_id": sid, "interval_idx": t}
            for i, c in enumerate(FEATURE_COLUMNS):
                row[c] = float(feats[i])
            for j, h in enumerate(HEURISTIC_NAMES):
                row[f"p_{h}"] = float(probs[j])
            rows.append(row)
    return pd.DataFrame(rows)


def test_top1_ece_on_well_calibrated_predictions():
    """Draw true labels from each row's predicted distribution -> ECE should be small."""
    n = 5000
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=np.ones(K) * 0.5, size=n)
    # Sample labels from each row's predicted distribution -> empirical accuracy
    # in any confidence bin will match the average confidence in that bin.
    y = np.array([rng.choice(K, p=probs[i]) for i in range(n)])
    ece = top1_ece(probs, y, n_bins=10)
    assert ece < 0.05, f"ECE {ece} too high for well-calibrated draws"


def test_top1_ece_on_overconfident_predictions():
    """All probs are (0.99, 0.003, 0.003, 0.004) but true labels are random uniform."""
    n = 400
    rng = np.random.default_rng(7)
    probs = np.tile(np.array([0.99, 0.003, 0.003, 0.004]), (n, 1))
    y = rng.integers(0, K, size=n)
    ece = top1_ece(probs, y, n_bins=10)
    # Accuracy ~ 0.25, confidence 0.99 -> gap ~ 0.74
    assert ece > 0.5


def test_brier_zero_on_perfect_predictions():
    n = 50
    rng = np.random.default_rng(3)
    y = rng.integers(0, K, size=n)
    probs = np.zeros((n, K))
    probs[np.arange(n), y] = 1.0
    assert brier_one_hot(probs, y) == 0.0


def test_split_calibration_shifts_no_leakage():
    df = _make_dataset(n_shifts=20, n_per=5, seed=11)
    tr_idx, cal_idx = split_calibration_shifts(df, cal_frac=0.25, seed=0)
    tr_shifts = set(df.iloc[tr_idx]["shift_id"].unique())
    cal_shifts = set(df.iloc[cal_idx]["shift_id"].unique())
    assert tr_shifts.isdisjoint(cal_shifts)
    # Approximate 75/25 split at the shift level.
    n_total = len(tr_shifts) + len(cal_shifts)
    assert 0.15 <= len(cal_shifts) / n_total <= 0.35


def test_calibrated_ranker_predicts_valid_distribution(cfg_ranker_small):
    df = _make_dataset()
    tr_idx, cal_idx = split_calibration_shifts(df, cal_frac=0.25, seed=0)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_cal = df.iloc[cal_idx].reset_index(drop=True)
    cv = cross_validate_ranker(df_tr, cfg_ranker_small, seed=0)
    calibrator = CalibratedRanker(cv.model, cv.feature_cols).fit(df_cal)
    probs = calibrator.predict_proba(df_cal[cv.feature_cols].to_numpy(np.float64))
    assert probs.shape == (len(df_cal), K)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_calibration_returns_report(cfg_ranker_small):
    df = _make_dataset()
    tr_idx, cal_idx = split_calibration_shifts(df, cal_frac=0.25, seed=0)
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_cal = df.iloc[cal_idx].reset_index(drop=True)
    cv = cross_validate_ranker(df_tr, cfg_ranker_small, seed=0)
    calibrator = CalibratedRanker(cv.model, cv.feature_cols).fit(df_cal)
    report = evaluate_calibration(cv, df_cal, calibrator)
    assert report.n_rows == len(df_cal)
    assert 0.0 <= report.ece_post <= 1.0
    assert report.brier_post >= 0.0
    assert report.soft_xent_post >= 0.0


# --- Rules absent from the calibration split (revision) ----------------------
#
# Stage-1 screening retained rules winning from 65% of decisions down to 1.1%,
# so on a calibration split of a few hundred epochs the tail rules appear a
# handful of times and sometimes not at all. The previous implementation used
# CalibratedClassifierCV, which infers its class set from argmax labels: with a
# rule absent it saw fewer classes than the ranker emits and indexed off the end
# of predict_proba. It crashed on the smoke corpus where EDD won nothing, and
# would have crashed at full scale AFTER the labelling stage had already run.


class _FixedRanker:
    """A stand-in ranker emitting a fixed K-column distribution."""

    def __init__(self, probs):
        self._p = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        out = np.resize(self._p, (n, self._p.shape[-1]))
        return out / out.sum(axis=1, keepdims=True)


def _cal_frame(n_rows: int, k: int, winners, feature_cols):
    """A calibration frame whose argmax labels cover only `winners`."""
    rng = np.random.default_rng(0)
    P = rng.uniform(0.01, 0.05, size=(n_rows, k))
    for i in range(n_rows):
        P[i, winners[i % len(winners)]] = 0.9
    P /= P.sum(axis=1, keepdims=True)
    df = pd.DataFrame(rng.standard_normal((n_rows, len(feature_cols))),
                      columns=feature_cols)
    for j in range(k):
        df[f"p_rule{j}"] = P[:, j]
    return df


def test_calibrator_survives_a_rule_that_never_wins():
    """The exact failure that killed Stage 3 on the smoke corpus."""
    from models.calibration import CalibratedRanker

    k, feats = 6, ["f0", "f1", "f2"]
    # Only classes 0..3 ever win; 4 and 5 never do.
    df = _cal_frame(120, k, winners=[0, 1, 2, 3], feature_cols=feats)
    ranker = _FixedRanker(np.full(k, 1.0 / k))

    cal = CalibratedRanker(ranker, feats).fit(df)
    out = cal.predict_proba(df[feats].to_numpy())

    assert out.shape == (len(df), k), "calibrated output lost a class column"
    assert np.isfinite(out).all(), "calibration produced non-finite probabilities"
    assert np.allclose(out.sum(axis=1), 1.0), "calibrated rows are not a distribution"


def test_calibrator_passes_absent_rules_through_uncalibrated():
    """An absent rule has no evidence to correct it, so it must be untouched."""
    from models.calibration import CalibratedRanker

    k, feats = 4, ["f0", "f1"]
    df = _cal_frame(80, k, winners=[0, 1], feature_cols=feats)
    base = np.array([0.4, 0.3, 0.2, 0.1])
    cal = CalibratedRanker(_FixedRanker(base), feats).fit(df)

    assert set(cal._uncalibrated) == {2, 3}, (
        f"expected classes 2 and 3 to be uncalibrated, got {cal._uncalibrated}"
    )
    # Their RELATIVE proportion is preserved (renormalisation scales both alike).
    out = cal.predict_proba(df[feats].to_numpy())
    assert np.allclose(out[:, 2] / out[:, 3], base[2] / base[3]), (
        "absent classes were altered relative to one another"
    )


def test_calibrator_still_calibrates_when_every_rule_wins():
    """Guard: the robustness path must not disable calibration outright."""
    from models.calibration import CalibratedRanker

    k, feats = 4, ["f0", "f1"]
    df = _cal_frame(200, k, winners=[0, 1, 2, 3], feature_cols=feats)
    cal = CalibratedRanker(_FixedRanker(np.full(k, 1.0 / k)), feats).fit(df)
    assert cal._uncalibrated == [], "a class was skipped despite winning"
    assert sum(i is not None for i in cal._iso) == k, "not every class was fitted"
