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


def spawn_shift_seeds(n: int, root: int = SEED_SIM) -> list[int]:
    """Spawn `n` independent integer seeds from a SeedSequence root.

    Used to generate the 300 shift seeds (250 train + 50 test) deterministically.
    """
    seq = np.random.SeedSequence(root)
    children = seq.spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


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
