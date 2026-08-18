"""Remove pre-revision labels, models and results. Keep current Stage-1 outputs.

    python scripts/clean_stale.py

Do NOT use `make clean` — that wipes results/ wholesale, including
results/S1_calibration and results/S1_perishability, which config.yaml's
fitted k and screened pool are derived from.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

KEEP = {
    "results/S1_calibration",
    "results/S1_perishability",
    "figures/S1_calibration",
}


def _rm(path: str) -> None:
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe even if current-revision Stage-2 labels are present.",
    )
    args = parser.parse_args()

    meta_path = Path("data/label_meta.json")
    if meta_path.exists() and not args.force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if int(meta.get("tau", 0) or 0) == 4 and not meta.get("provisional_scales"):
            print(
                "REFUSE: data/label_meta.json looks like current-revision Stage 2 "
                "(tau=4, not provisional). Wiping it would delete the labels and "
                "ranker. Pass --force if you really mean to start over."
            )
            return 1
    for p in glob.glob("data/*.parquet"):
        _rm(p)
    for p in glob.glob("data/*.npz"):
        _rm(p)
    _rm("data/label_meta.json")
    for d in (
        glob.glob("data/e3_*")
        + glob.glob("data/tau1")
        + glob.glob("data/tau2")
        + glob.glob("data/tau3")
        + glob.glob("data/e4_*")
        + glob.glob("data/smoke")
    ):
        _rm(d)

    shutil.rmtree("runs", ignore_errors=True)

    for top in ("results", "figures"):
        for d in glob.glob(top + "/*"):
            key = d.replace("\\", "/")
            if os.path.isdir(d) and key not in KEEP:
                _rm(d)
        for f in glob.glob(top + "/*"):
            if os.path.isfile(f):
                _rm(f)

    for d in ("runs", "results", "figures"):
        Path(d).mkdir(exist_ok=True)

    print("removed pre-revision labels, models and results; kept "
          + ", ".join(sorted(KEEP)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
