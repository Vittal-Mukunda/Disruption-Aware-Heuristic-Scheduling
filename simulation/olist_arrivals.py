"""Empirical Olist inter-arrival pool for the non-Poisson arrival mode.

The simulator's default arrival process is homogeneous Poisson (exponential
gaps: CV=1, skew=2). The Olist Brazilian E-Commerce trace is heavy-tailed and
bursty (CV=2.68, skew=11). This module exposes the empirical mean-normalized
inter-arrival distribution so `WarehouseEnv` can bootstrap real burstiness
while holding the mean arrival rate fixed.

Pool construction mirrors `experiments.a_realdata_validation.load_olist_draws`:
  - inter-arrival gaps are diffs taken WITHIN each calendar day, which strips
    the multi-day growth trend (intra-day seasonality remains, a known
    deviation noted in Deliverable A);
  - gaps are mean-normalized to unit mean so they can be rescaled to any target
    rate by multiplying by 1 / base_rate_per_minute.

The normalized pool is cached to data/olist_interarrival_norm.npy on first
build and is also held in a module-level cache so repeated `WarehouseEnv`
construction does not re-read the CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_OLIST_DIR = _ROOT / "Olist Dataset"
_CACHE_PATH = _ROOT / "data" / "olist_interarrival_norm.npy"

_POOL: np.ndarray | None = None


def _build_pool() -> np.ndarray:
    """Extract mean-normalized within-day inter-arrival gaps from the Olist CSV."""
    orders = pd.read_csv(
        _OLIST_DIR / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp"],
    )
    o = orders.dropna(subset=["order_purchase_timestamp"]).copy()
    o["date"] = o["order_purchase_timestamp"].dt.date

    parts: list[np.ndarray] = []
    for _, g in o.groupby("date"):
        ts = np.sort(g["order_purchase_timestamp"].values).astype("datetime64[s]")
        if ts.size > 1:
            parts.append(np.diff(ts).astype(np.int64) / 60.0)  # minutes

    inter = np.concatenate(parts)
    inter = inter[inter > 0]
    return (inter / inter.mean()).astype(np.float64)


def load_interarrival_pool() -> np.ndarray:
    """Return the mean-normalized Olist inter-arrival pool (unit mean).

    Cached both on disk (data/olist_interarrival_norm.npy) and in process.
    """
    global _POOL
    if _POOL is not None:
        return _POOL
    if _CACHE_PATH.exists():
        _POOL = np.load(_CACHE_PATH)
    else:
        _POOL = _build_pool()
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(_CACHE_PATH, _POOL)
    return _POOL
