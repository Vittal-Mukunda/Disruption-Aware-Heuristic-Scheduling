"""Phase 4 — unit tests for GMM-based regime discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from regime.regime_discovery import (
    FEATURE_COLUMNS,
    attach_regime_posteriors,
    discover_regimes,
    regime_posterior_columns,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg_regime():
    cfg = OmegaConf.load(CONFIG_PATH)
    return cfg.regime


def _make_three_cluster_features(n_per: int = 200, seed: int = 0) -> pd.DataFrame:
    """3 well-separated isotropic Gaussians in 25-D feature space."""
    rng = np.random.default_rng(seed)
    d = len(FEATURE_COLUMNS)
    centers = np.zeros((3, d))
    # Large inter-cluster gap so BIC's parameter penalty doesn't pull K* below 3.
    centers[0, :] = -30.0
    centers[1, :] = 0.0
    centers[2, :] = 30.0
    blobs = [rng.normal(centers[c], 1.0, size=(n_per, d)) for c in range(3)]
    X = np.vstack(blobs)
    return pd.DataFrame(X, columns=FEATURE_COLUMNS)


def test_k_sweep_recovers_three_clusters(cfg_regime):
    df = _make_three_cluster_features(n_per=300, seed=7)
    # Force k_grid to include 3 so BIC has a chance to pick it.
    cfg = OmegaConf.create({
        "k_grid": [2, 3, 4, 5],
        "n_ari_refits": 4,
        "ari_threshold": 0.85,
    })
    result = discover_regimes(df, cfg, seed=11)
    assert result.k_star == 3, f"expected K*=3, got {result.k_star}; BICs={result.bic_per_k}"


def test_ari_stability_on_clean_clusters(cfg_regime):
    df = _make_three_cluster_features(n_per=150, seed=13)
    cfg = OmegaConf.create({
        "k_grid": [3],
        "n_ari_refits": 5,
        "ari_threshold": 0.85,
    })
    result = discover_regimes(df, cfg, seed=17)
    assert result.stable, f"ARI {result.mean_ari} below threshold on clean clusters"
    assert result.mean_ari > 0.9


def test_attach_regime_posteriors_shape(cfg_regime):
    df = _make_three_cluster_features(n_per=50, seed=21)
    cfg = OmegaConf.create({
        "k_grid": [3],
        "n_ari_refits": 2,
        "ari_threshold": 0.85,
    })
    result = discover_regimes(df, cfg, seed=23)
    attached = attach_regime_posteriors(df, result.gmm)
    cols = regime_posterior_columns(result.k_star)
    for c in cols:
        assert c in attached.columns
    posteriors = attached[cols].to_numpy()
    assert posteriors.shape == (len(df), result.k_star)
    # Row sums of posteriors must equal 1.0 (GMM responsibility).
    assert np.allclose(posteriors.sum(axis=1), 1.0, atol=1e-9)


def test_missing_feature_column_raises():
    df = pd.DataFrame({"f_queue_length": [0.0, 1.0]})
    cfg = OmegaConf.create({
        "k_grid": [2],
        "n_ari_refits": 1,
        "ari_threshold": 0.85,
    })
    with pytest.raises(ValueError):
        discover_regimes(df, cfg, seed=0)
