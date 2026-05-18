"""Phase 5 — snapshot-XGB baseline (tau=1 ablation of OUR controller).

This is the labeling-side ablation: identical to OUR controller in every
respect *except* that the soft labels come from a τ=1 rollout (next-interval
cost) rather than τ=4. It is the closest learned competitor in E2 and the
gap between `ours.parquet` and `snapshot_xgb.parquet` is the headline number
HANDOFF §3.3 asks for.

Wire-up mirrors `baselines.ours.load_ours` exactly — same regime/calibrator/
switching stack — but pointed at `runs/phase4_tau1/` rather than
`runs/phase4/`.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from baselines.ours import OursPolicy, load_ours

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "phase4_tau1"


def load_snapshot_xgb(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    cfg: DictConfig | None = None,
) -> OursPolicy:
    """Load the τ=1 ranker artifacts via the OURS loader (same pipeline)."""
    return load_ours(run_dir, cfg=cfg)
