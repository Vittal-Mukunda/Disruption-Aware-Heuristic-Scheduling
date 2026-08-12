"""Phase 4 — unit tests for the switching controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from models.switching_controller import PCT_PERISHABLE_IDX, SwitchingController
from simulation.heuristics import HEURISTIC_NAMES
from simulation.state_extractor import FEATURE_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
K = len(HEURISTIC_NAMES)
D = len(FEATURE_NAMES)


class _ConstantRanker:
    """Always returns the same (1, K) probability vector — for deterministic tests."""

    def __init__(self, probs: np.ndarray):
        self._probs = probs.reshape(1, -1).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.broadcast_to(self._probs, (n, self._probs.shape[1])).copy()


class _SequenceRanker:
    """Returns a queued probability vector on each `predict_proba` call."""

    def __init__(self, sequence: list[np.ndarray]):
        self._sequence = [s.astype(np.float64).reshape(1, -1) for s in sequence]
        self._i = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self._sequence[self._i]
        self._i += 1
        return out


@pytest.fixture(scope="module")
def cfg_switching():
    cfg = OmegaConf.load(CONFIG_PATH)
    return cfg.ranker.switching


def _zero_state(pct_perishable: float = 0.5) -> np.ndarray:
    feats = np.zeros(D, dtype=np.float64)
    feats[PCT_PERISHABLE_IDX] = pct_perishable
    return feats


def _peaked(idx: int, peak: float) -> np.ndarray:
    """A K-wide distribution with `peak` mass on `idx`, the rest spread evenly.

    Built at the pool's actual width. These fixtures were hard-coded to four
    columns, so they silently indexed the wrong rule once the pool grew to
    eight — and the mask's width check turned that into an error rather than a
    wrong answer, which is how it surfaced.
    """
    p = np.full(K, (1.0 - peak) / (K - 1), dtype=np.float64)
    p[idx] = peak
    return p / p.sum()


def test_fefo_mask_zeros_p_fefo(cfg_switching):
    probs = np.full(K, 1.0 / K)
    ranker = _ConstantRanker(probs)
    ctrl = SwitchingController(
        ranker, cfg_switching, fefo_threshold=0.05,
        heuristic_names=HEURISTIC_NAMES,
    )
    feats = _zero_state(pct_perishable=0.0)
    ctrl.select(feats)
    masked = ctrl.last_probs
    assert masked is not None
    fefo_idx = HEURISTIC_NAMES.index("FEFO")
    assert masked[fefo_idx] == 0.0
    assert np.isclose(masked.sum(), 1.0)


def test_dwell_keeps_heuristic_for_t_min(cfg_switching):
    """If T_min=2, a switch holds for 2 intervals when entropy stays high enough."""
    # Confident on FIFO at t=0, then confident on FEFO at t=1 (but high entropy
    # gate prevents the switch under default ratio=0.5 since H_max=log(K)=1.386
    # and we'll choose entropies just above and below the threshold).
    fifo_idx = HEURISTIC_NAMES.index("FIFO")
    fefo_idx = HEURISTIC_NAMES.index("FEFO")

    confident_fifo = _peaked(fifo_idx, 0.85)
    # Nearly flat, leaning to FEFO: entropy stays above the gate so the dwell,
    # not the gate, decides. Built at width K rather than hard-coded to four.
    flat_fefo_lean = np.full(K, 1.0 / K)
    flat_fefo_lean[fefo_idx] += 0.04
    flat_fefo_lean = flat_fefo_lean / flat_fefo_lean.sum()
    assert int(np.argmax(flat_fefo_lean)) == fefo_idx

    ranker = _SequenceRanker([confident_fifo, flat_fefo_lean, flat_fefo_lean])
    cfg = OmegaConf.create({"t_min_intervals": 2, "entropy_gate_ratio": 0.5})
    ctrl = SwitchingController(
        ranker, cfg, fefo_threshold=0.05, heuristic_names=HEURISTIC_NAMES,
    )

    feats = _zero_state(pct_perishable=0.5)
    h0 = ctrl.select(feats)
    h1 = ctrl.select(feats)
    h2 = ctrl.select(feats)

    assert h0 == "FIFO"
    # h1 is within dwell (t_min=2 means at least 2 consecutive). New dist is high
    # entropy -> gate does not fire -> keep FIFO.
    assert h1 == "FIFO"
    # h2: dwell exhausted -> switch is allowed since gate doesn't matter anymore.
    assert h2 == "FEFO"


def test_entropy_gate_overrides_dwell_when_confident(cfg_switching):
    fifo_idx = HEURISTIC_NAMES.index("FIFO")
    fefo_idx = HEURISTIC_NAMES.index("FEFO")
    confident_fifo = np.full(K, 0.05); confident_fifo[fifo_idx] = 0.85
    confident_fefo = np.full(K, 0.03); confident_fefo[fefo_idx] = 0.91

    ranker = _SequenceRanker([confident_fifo, confident_fefo])
    cfg = OmegaConf.create({"t_min_intervals": 3, "entropy_gate_ratio": 0.5})
    ctrl = SwitchingController(
        ranker, cfg, fefo_threshold=0.05, heuristic_names=HEURISTIC_NAMES,
    )

    feats = _zero_state(pct_perishable=0.5)
    assert ctrl.select(feats) == "FIFO"
    # Dwell remaining > 0, but the new distribution has H ~ 0.43 nats (below
    # 0.5 * log(4) = 0.693) -> entropy gate fires -> switch is allowed.
    assert ctrl.select(feats) == "FEFO"


def test_select_sequence_resets_state(cfg_switching):
    probs1 = _peaked(HEURISTIC_NAMES.index("FIFO"), 0.7)
    probs2 = _peaked(HEURISTIC_NAMES.index("FEFO"), 0.7)
    ranker = _SequenceRanker([probs1, probs2, probs1, probs2])
    cfg = OmegaConf.create({"t_min_intervals": 2, "entropy_gate_ratio": 0.5})
    ctrl = SwitchingController(
        ranker, cfg, fefo_threshold=0.05, heuristic_names=HEURISTIC_NAMES,
    )

    seq = np.tile(_zero_state(0.5), (2, 1))
    out_a = ctrl.select_sequence(seq)
    # After reset, the second sequence should reproduce the same logic.
    out_b = ctrl.select_sequence(seq)
    assert out_a == out_b


def test_fefo_mask_blocks_fefo_under_threshold(cfg_switching):
    """When FEFO would be argmax but the queue is non-perishable, FEFO is masked."""
    fefo_idx = HEURISTIC_NAMES.index("FEFO")
    probs = np.full(K, 0.1); probs[fefo_idx] = 0.7
    probs = probs / probs.sum()
    ranker = _ConstantRanker(probs)
    cfg = OmegaConf.create({"t_min_intervals": 1, "entropy_gate_ratio": 0.5})
    ctrl = SwitchingController(
        ranker, cfg, fefo_threshold=0.05, heuristic_names=HEURISTIC_NAMES,
    )

    feats = _zero_state(pct_perishable=0.0)  # below threshold
    chosen = ctrl.select(feats)
    assert chosen != "FEFO"
    masked = ctrl.last_probs
    assert masked[fefo_idx] == 0.0
