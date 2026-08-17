"""Everything checkable BEFORE the campaign starts, for a run you get once.

`scripts/preflight.py` proves the modules import. This proves the *campaign* can
start: the environment matches the lockfile, Stage 1's results are present and
agree with the config that will be labelled, every command in the recipe
resolves, no pre-revision artifact is lurking where a driver will read it, and
the Olist dataset is either present or its absence is acknowledged.

    python scripts/campaign_preflight.py

Exit 0 means start. Exit 1 means something will fail later, and later is
expensive: the labelling stage alone is over an hour, and a driver that dies at
hour fourteen on an argument typo costs the stage.

This does NOT prove the run will succeed. It proves the failures that are
knowable in advance are not present.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "RUN_CAMPAIGN.md"

FAIL: list[str] = []
WARN: list[str] = []


def check(ok: bool, msg: str, *, fatal: bool = True) -> None:
    if not ok:
        (FAIL if fatal else WARN).append(msg)
    print(f"  {'ok  ' if ok else ('FAIL' if fatal else 'warn')}  {msg}", flush=True)


def main() -> int:
    print("[1/7] interpreter and environment")
    v = sys.version_info
    check((3, 10) <= (v.major, v.minor) <= (3, 12),
          f"python {v.major}.{v.minor} is in the supported 3.10-3.12 range")

    lock = REPO / "requirements-lock.txt"
    check(lock.exists(), "requirements-lock.txt present")
    if lock.exists():
        pinned = dict(
            re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", lock.read_text(encoding="utf-8"), re.M)
        )
        import importlib.metadata as md
        drift = []
        for pkg in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy"):
            want = pinned.get(pkg)
            if not want:
                continue
            try:
                got = md.version(pkg)
            except md.PackageNotFoundError:
                drift.append(f"{pkg} missing"); continue
            if got != want:
                drift.append(f"{pkg} {got} != locked {want}")
        check(not drift,
              f"core packages match the lockfile{'' if not drift else ': ' + '; '.join(drift)}",
              fatal=False)

    print("\n[2/7] Stage 1 results present and consistent with the config")
    import json

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(REPO / "config.yaml")
    screen = REPO / "results" / "S1_calibration" / "pool_screening.json"
    calib = REPO / "results" / "S1_calibration" / "rule_calibration.json"
    perish = REPO / "results" / "S1_perishability" / "pivotality_summary.json"
    for p in (screen, calib, perish):
        check(p.exists(), f"{p.relative_to(REPO)} exists")

    if screen.exists():
        retained = json.loads(screen.read_text(encoding="utf-8"))["retained"]
        pool = list(cfg.heuristics.pool)
        check(pool == retained,
              f"config pool {pool} equals the Stage-1 retained pool"
              + ("" if pool == retained else f" (screen says {retained}; run scripts/apply_stage1.py)"))
    if calib.exists():
        chosen = json.loads(calib.read_text(encoding="utf-8"))["chosen"]
        for rule, e in chosen.items():
            key, want = e["config_key"], float(e["k_deployed"])
            got = cfg.heuristics.get(key)
            check(got is not None and float(got) == want,
                  f"{key} = {want} as fitted for {rule}")

    print("\n[3/7] no pre-revision artifact where a driver will read it")
    from simulation.state_extractor import N_FEATURES
    stale = []
    tp = REPO / "data" / "train.parquet"
    if tp.exists():
        import pandas as pd
        cols = set(pd.read_parquet(tp).columns)
        if "f_n_orders_breached_so_far" in cols or f"cost_FIFO" in cols:
            stale.append("data/train.parquet is pre-revision")
    for name in ("results/ours.parquet", "results/data_efficiency/data_efficiency_summary.json"):
        f = REPO / name
        if f.exists():
            body = f.read_text(encoding="utf-8", errors="ignore")[:4000] if f.suffix == ".json" else ""
            if f.suffix == ".json" and "sla_breach_rate_mean" in body:
                stale.append(f"{name} is pre-revision")
    npz = REPO / "data" / "offline_fqi_transitions.npz"
    if npz.exists():
        import numpy as np
        with np.load(npz, allow_pickle=True) as d:
            if "cache_stamp" not in d.files:
                stale.append("data/offline_fqi_transitions.npz has no schema stamp")
    check(not stale,
          "no stale artifacts detected" + ("" if not stale else f": {stale}; run `make clean-stale`"))

    print("\n[4/7] every campaign command resolves")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_campaign_commands.py", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    check(proc.returncode == 0,
          "all RUN_CAMPAIGN.md commands parse and every figure has a producer"
          + ("" if proc.returncode == 0 else f"\n{proc.stdout[-600:]}"))

    print("\n[5/7] optional inputs")
    olist = REPO / "Olist Dataset" / "olist_orders_dataset.csv"
    check(olist.exists(),
          "Olist dataset present"
          if olist.exists() else
          "Olist dataset MISSING — Figures 5-6 and all of R1.5b will not be "
          "produced. Download 'Brazilian E-Commerce Public Dataset by Olist' "
          "from Kaggle into 'Olist Dataset/' before starting.",
          fatal=False)

    print("\n[6/7] disk")
    free_gb = shutil.disk_usage(REPO).free / 1e9
    check(free_gb > 10, f"{free_gb:.0f} GB free (campaign artifacts need a few GB)")

    print("\n[7/7] estimated cost")
    subprocess.run([sys.executable, "scripts/campaign_budget.py"], cwd=REPO)

    print()
    for w in WARN:
        print(f"[warn ] {w}")
    if FAIL:
        print(f"\n[gate ] DO NOT START — {len(FAIL)} blocking problem(s):")
        for f in FAIL:
            print(f"   - {f}")
        return 1
    print("\n[gate ] READY TO START.")
    print("        If a stage fails mid-run, re-run THAT stage only — Stage 2 and 3")
    print("        outputs persist on disk, so nothing earlier is lost. Commit after")
    print("        each stage, as RUN_CAMPAIGN.md says, so a crash cannot cost more")
    print("        than the stage it happened in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
