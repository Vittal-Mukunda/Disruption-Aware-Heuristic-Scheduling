"""Central seed registry for deterministic experiments.

All stochastic components must consume seeds derived from this module so that
runs are reproducible end-to-end.

Usage:
    from seed import SEED_SIM, SEED_MODEL, SEED_PPO, spawn_shift_seeds, rng

    rng_sim = rng(SEED_SIM)
    shift_seeds = spawn_shift_seeds(n=300, root=SEED_SIM)
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED_SIM: int = int(os.environ.get("SEED_SIM", 42))
SEED_MODEL: int = int(os.environ.get("SEED_MODEL", 1337))
SEED_PPO: int = int(os.environ.get("SEED_PPO", 2024))


def rng(seed: int) -> np.random.Generator:
    """Return a fresh numpy default_rng for the given seed."""
    return np.random.default_rng(seed)


SEED_ROLLOUT: int = int(os.environ.get("SEED_ROLLOUT", 20260804))


def spawn_shift_seeds(n: int, root: int = SEED_SIM) -> list[int]:
    """Spawn `n` independent integer seeds from a SeedSequence root."""
    seq = np.random.SeedSequence(root)
    children = seq.spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def shift_corpora(cfg) -> dict[str, list[int]]:
    """The three disjoint shift blocks: train, calibration, test.

    The calibration block is new in revision. The submitted version had only
    train and test, which left nowhere to fit rule hyperparameters — ATC's
    look-ahead scale `k` in particular (Reviewer 1, 4.c) — without contaminating
    one or the other. Blocks are contiguous slices of one SeedSequence spawn, so
    they are disjoint by construction and stable under changes to any one block
    size only if that block is last; changing `n_train` or `n_calib` therefore
    re-draws the downstream blocks and every corpus must be regenerated together.
    """
    s = cfg.shifts
    n_train, n_calib, n_test = int(s.n_train), int(s.n_calib), int(s.n_test)
    seeds = spawn_shift_seeds(n_train + n_calib + n_test, root=int(s.seed_seq_root))
    return {
        "train": seeds[:n_train],
        "calib": seeds[n_train : n_train + n_calib],
        "test": seeds[n_train + n_calib :],
    }


def data_efficiency_shift_indices(
    n_train: int, budget: int, rep: int, seed: int = 1337
) -> np.ndarray:
    """Shared train-shift subset for DAHS and FQI sample-efficiency curves.

    Indices are positional over the training block (0 .. n_train-1), which is
    also `shift_id` in data/train.parquet and in the FQI transition log.
    Independent RNG per (budget, rep) so adding a budget does not reshuffle
    the others.
    """
    if budget >= n_train:
        return np.arange(n_train, dtype=np.int64)
    rng = np.random.default_rng([int(seed), int(budget), int(rep)])
    return np.sort(rng.choice(n_train, size=int(budget), replace=False).astype(np.int64))


def seed_all(seed: int) -> None:
    """Seed Python `random`, numpy, and torch (if installed) in one call."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
