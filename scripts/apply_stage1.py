"""Write Stage-1's fitted values back into config.yaml.

Stage 1 fits the ATC/COVERT look-ahead scales and screens the rule pool, then
prints "write these into config.yaml". Doing that by hand is a reproducibility
hole: the campaign has a manual step in the middle that nothing records, and a
mistyped `k` is indistinguishable from a fitted one afterwards.

    python scripts/apply_stage1.py            # apply
    python scripts/apply_stage1.py --dry-run  # show the edit only

Edits are surgical line rewrites, not a load-and-dump through OmegaConf: the
config's comments carry the justification for nearly every value in it
(Reviewer 1, 4.c among others) and a round-trip through a YAML emitter would
delete all of them. Each rewritten line gains a provenance comment naming the
artifact it came from.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
CALIB_JSON = REPO_ROOT / "results" / "S1_calibration" / "rule_calibration.json"
SCREEN_JSON = REPO_ROOT / "results" / "S1_calibration" / "pool_screening.json"


def _set_scalar(text: str, key: str, value: float, note: str) -> tuple[str, bool]:
    """Rewrite `  <key>: <anything>` in place, preserving indentation."""
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:[^\n#]*(?P<comment>#.*)?$",
                         re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return text, False
    new_line = f"{m.group('indent')}{key}: {value}   # {note}"
    return text[: m.start()] + new_line + text[m.end():], True


def _set_pool(text: str, pool: list[str], note: str) -> tuple[str, bool]:
    pattern = re.compile(r"^(?P<indent>\s*)pool:\s*\[[^\]]*\][^\n]*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return text, False
    new_line = f"{m.group('indent')}pool: [{', '.join(pool)}]   # {note}"
    return text[: m.start()] + new_line + text[m.end():], True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scale-mode", choices=("portfolio", "standalone"),
                    default="portfolio",
                    help="Which fitted k to deploy. The pool is what ships, so "
                         "the portfolio value is the default; both are reported.")
    args = ap.parse_args()

    if not CALIB_JSON.exists():
        raise SystemExit(
            f"missing {CALIB_JSON}\nRun `python -m experiments.calibrate_rules "
            f"calibrate` first."
        )
    calib = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    text = CONFIG_PATH.read_text(encoding="utf-8")
    changed: list[str] = []

    for rule, entry in calib["chosen"].items():
        key = entry["config_key"]
        k = float(entry[f"k_{args.scale_mode}"])
        text, ok = _set_scalar(
            text, key, k,
            f"FITTED by Stage 1 ({args.scale_mode}); standalone k*="
            f"{entry['k_standalone']}, portfolio k*={entry['k_portfolio']}",
        )
        if not ok:
            raise SystemExit(f"could not find `{key}:` in {CONFIG_PATH}")
        changed.append(f"{key} = {k}   ({rule})")

    if SCREEN_JSON.exists():
        screen = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
        retained = list(screen["retained"])
        if len(retained) < 2:
            raise SystemExit(
                f"Stage-1 screening retained {retained}, fewer than two rules. "
                f"The selection problem is degenerate under the current "
                f"objective — that is a finding to report, not a pool to deploy."
            )
        text, ok = _set_pool(
            text, retained,
            f"RETAINED by Stage 1 from {screen['candidates']}",
        )
        if not ok:
            raise SystemExit(f"could not find `pool:` in {CONFIG_PATH}")
        changed.append(f"pool = {retained}")
    else:
        print(f"[apply] {SCREEN_JSON.name} not found — leaving the pool alone. "
              f"Run `calibrate_rules screen` to settle it.")

    print("[apply] Stage-1 values:")
    for c in changed:
        print(f"  {c}")

    if args.dry_run:
        print("\n[apply] --dry-run: config.yaml not modified.")
        return 0

    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"\n[apply] wrote {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print("[apply] commit this — the deployed k and pool are results, not settings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
