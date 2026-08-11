"""Regime discovery via Gaussian Mixture Model on the 25-D feature matrix.

Phase 4 — per-state soft regime assignment. The pipeline:
  1. Sweep K in `cfg.regime.k_grid = [3, 4, 5, 6]`, fit a `GaussianMixture` with
     each, score with BIC, pick K_star = argmin BIC.
  2. Re-fit K_star `cfg.regime.n_ari_refits` (default 10) times with different
     random seeds. Compute Adjusted Rand Index between all pairs of hard
     assignments. Require mean pairwise ARI >= `cfg.regime.ari_threshold` (0.85)
     for the chosen K to be considered stable.
  3. Return the best-BIC GMM among the stability re-fits (so the persisted model
     is reproducible from the seed registry).

Outputs:
  - `RegimeDiscoveryResult` containing the fitted GMM, K_star, BIC, mean ARI,
    and the predict_proba matrix.
  - `attach_regime_posteriors(df, gmm)` appends `regime_post_0..K-1` columns
    to a parquet-style DataFrame.

The regime layer feeds the ranker as auxiliary features (concat to the 25-D
state) and is also persisted so we can replay it on the test set without
re-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from simulation.state_extractor import FEATURE_NAMES

if TYPE_CHECKING:
    from omegaconf import DictConfig

FEATURE_COLUMNS: list[str] = [f"f_{name}" for name in FEATURE_NAMES]


@dataclass
class RegimeDiscoveryResult:
    """Outcome of `discover_regimes`.

    Attributes
    ----------
    gmm
        The fitted `GaussianMixture` to use for inference.
    k_star
        Number of components selected by BIC over the k_grid sweep.
    bic
        BIC of `gmm` on the fit data.
    mean_ari
        Mean pairwise Adjusted Rand Index across the `n_ari_refits` re-fits at
        K_star. Stability gate: must be >= `cfg.regime.ari_threshold`.
    stable
        True iff `mean_ari >= cfg.regime.ari_threshold`.
    posteriors
        `predict_proba(X)` of the persisted `gmm`. Shape (n_rows, k_star).
    bic_per_k
        Dict of K -> BIC for the initial sweep (for the paper's appendix table).
    """

    gmm: GaussianMixture
    k_star: int
    bic: float
    mean_ari: float
    stable: bool
    posteriors: np.ndarray
    bic_per_k: dict[int, float]


def _extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing feature columns: {missing[:3]}...")
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)


def _fit_gmm(X: np.ndarray, k: int, seed: int, n_init: int = 1) -> GaussianMixture:
    """Fit one GMM. `n_init` restarts, best-likelihood kept (Reviewer 1, 3.a).

    A single EM start makes the BIC at each K a function of initialisation luck
    as much as of K, so the sweep compares "K=6 from a good start" against "K=4
    from a bad one". The submitted sweep ran `n_init=1` and fell monotonically
    to the edge of the grid; `cfg.regime.n_init` now controls this and defaults
    to 5, so each K is represented by its best fit rather than its first.
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=int(seed),
        n_init=int(n_init),
        reg_covar=1e-6,
        max_iter=200,
    )
    gmm.fit(X)
    return gmm


def _mean_pairwise_ari(labels_list: list[np.ndarray]) -> float:
    n = len(labels_list)
    if n < 2:
        return 1.0
    scores: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return float(np.mean(scores))


def discover_regimes(
    df: pd.DataFrame,
    cfg_regime: "DictConfig",
    seed: int,
) -> RegimeDiscoveryResult:
    """Run the K-sweep + ARI-stability protocol on the training feature matrix.

    Parameters
    ----------
    df
        DataFrame with the `f_<name>` feature columns (training rows only).
    cfg_regime
        `cfg.regime` sub-tree. Reads `k_grid`, `n_init`, `n_ari_refits`,
        `ari_threshold`.
    seed
        Base random_state. The K sweep uses `seed`; ARI re-fits use
        `seed + 1, ..., seed + n_ari_refits`.
    """
    X = _extract_feature_matrix(df)
    k_grid = [int(k) for k in cfg_regime.k_grid]
    if not k_grid:
        raise ValueError("cfg.regime.k_grid is empty")
    n_init = int(cfg_regime.get("n_init", 1))

    bic_per_k: dict[int, float] = {}
    for k in k_grid:
        gmm_k = _fit_gmm(X, k, seed=seed, n_init=n_init)
        bic_per_k[k] = float(gmm_k.bic(X))

    k_star = min(bic_per_k, key=lambda k: bic_per_k[k])
    if k_star in (k_grid[0], k_grid[-1]) and len(k_grid) > 1:
        # BIC selecting at a grid endpoint means the grid, not the data, chose K.
        # The submitted sweep did exactly this (K=6 on a {3,4,5,6} grid) because
        # two degenerate features made the covariance singular; both are now
        # removed. If it recurs, the grid must be widened before K is reported.
        print(
            f"[regime] WARNING: BIC-optimal K={k_star} sits at the EDGE of "
            f"k_grid={k_grid}. K is being chosen by the grid boundary, not by "
            f"the data. Widen cfg.regime.k_grid and re-run before reporting K*."
        )

    n_refits = int(cfg_regime.n_ari_refits)
    refit_gmms: list[GaussianMixture] = []
    labels_list: list[np.ndarray] = []
    for i in range(n_refits):
        gmm_i = _fit_gmm(X, k_star, seed=seed + 1 + i, n_init=n_init)
        refit_gmms.append(gmm_i)
        labels_list.append(gmm_i.predict(X))

    mean_ari = _mean_pairwise_ari(labels_list) if labels_list else 1.0
    ari_threshold = float(cfg_regime.ari_threshold)
    stable = mean_ari >= ari_threshold

    # Persist the best-BIC GMM among the stability re-fits — reproducible from seed
    # registry and slightly more robust than a single fit.
    best_idx = int(np.argmin([g.bic(X) for g in refit_gmms]))
    persisted_gmm = refit_gmms[best_idx]
    persisted_bic = float(persisted_gmm.bic(X))
    posteriors = persisted_gmm.predict_proba(X)

    return RegimeDiscoveryResult(
        gmm=persisted_gmm,
        k_star=k_star,
        bic=persisted_bic,
        mean_ari=mean_ari,
        stable=stable,
        posteriors=posteriors,
        bic_per_k=bic_per_k,
    )


def attach_regime_posteriors(
    df: pd.DataFrame, gmm: GaussianMixture
) -> pd.DataFrame:
    """Compute `gmm.predict_proba` on `df`'s features and append as columns.

    Mutates a copy; returns the new DataFrame. Column names are
    `regime_post_0`, ..., `regime_post_{K-1}` where K = `gmm.n_components`.
    """
    X = _extract_feature_matrix(df)
    posteriors = gmm.predict_proba(X)
    out = df.copy()
    for k in range(gmm.n_components):
        out[f"regime_post_{k}"] = posteriors[:, k]
    return out


def regime_posterior_columns(k: int) -> list[str]:
    return [f"regime_post_{i}" for i in range(k)]
