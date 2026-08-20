"""Every command in RUN_CAMPAIGN.md must exist and parse its own arguments.

WHY THIS EXISTS. Three separate defects motivated it, all of the same shape — a
command that looks right in the recipe and dies on contact.

  * `a2_olist_arrivals` was invoked bare. It requires a subcommand, so it exits on
    argument parsing and Figure 6 is never produced.
  * Four of the manuscript's figures had no producing command anywhere in the
    campaign at all, including the one Section 6.3 calls "the central figure of
    the paper".
  * The script that added them made its edits and asserted afterwards, with a
    single write at the end. The last assert failed, the write never ran, and the
    console still reported the earlier edits as applied — so the fix silently
    no-op'd and the commands stayed missing.

The campaign is ~16 hours. A command that fails on argument parsing at hour
fourteen costs the stage; one that was never in the recipe costs a figure nobody
notices until submission. Both are cheap to catch here.

This checks that each command RESOLVES — module importable, subcommand and flags
accepted by its parser. It does not run them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "RUN_CAMPAIGN.md"

def _loop_values() -> dict[str, str]:
    """Map each shell-loop variable to its FIRST value, read from the recipe.

    `for a in no_regime hard_labels ...` tells us what `$a` can legally be, so a
    single hardcoded placeholder is wrong: `evaluate --method $m` wants a rule
    name and `e3_ablations retrain $a` wants an ablation name. Reading them from
    the loop keeps this correct when the recipe changes.
    """
    text = CAMPAIGN.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"^for\s+(\w+)\s+in\s+([^;]+);\s*do", text, re.M):
        values = m.group(2).split()
        if values:
            out[m.group(1)] = values[0]
    return out


def _commands() -> list[list[str]]:
    """Every `python -m experiments.X ...` invocation in the recipe."""
    text = CAMPAIGN.read_text(encoding="utf-8")
    out: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "python -m experiments." not in line:
            continue
        line = line.split("#", 1)[0].strip().rstrip("\\").strip()
        tokens = line.split()
        i = tokens.index(next(t for t in tokens if t.startswith("experiments.")))
        out.append(tokens[i:])
    return out


def _resolvable(cmd: list[str]) -> list[str]:
    """Substitute shell-loop placeholders with a legal value so the parser runs."""
    loops = _loop_values()
    return [loops.get(t.lstrip("$"), t) if t.startswith("$") else t for t in cmd]


@pytest.mark.parametrize("cmd", _commands(), ids=lambda c: " ".join(c))
def test_campaign_command_parses(cmd):
    """The command must reach its own argument parser without erroring."""
    argv = _resolvable(cmd)
    proc = subprocess.run(
        [sys.executable, "-m", *argv, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    combined = proc.stdout + proc.stderr
    # `--help` exits 0 for argparse-driven modules. A module with no CLI parser
    # ignores it and runs, which may exit non-zero for want of artifacts — that
    # is fine. What must NOT appear is an argument-parsing complaint.
    assert "error: " not in combined, (
        f"`python -m {' '.join(argv)}` fails argument parsing:\n"
        + combined.strip()[-400:]
    )
    assert "No module named" not in combined, (
        f"`python -m {' '.join(argv)}` names a module that does not exist"
    )


def test_every_manuscript_figure_has_a_producing_command():
    """A figure the paper references must be produced by something in the recipe."""
    ms = (REPO_ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")
    recipe = CAMPAIGN.read_text(encoding="utf-8")

    # Figure directory -> the driver that writes it.
    owners = {
        "S1_calibration": "calibrate_rules",
        "E2": "e2_main",
        "data_efficiency": "fig_data_efficiency",
        "E4": "e4_sensitivity",
        "E5": "e5_calibration",
        "E8": "e8_robustness_grid",
        "E9": "e9_offline_fqi",
        "A2": "a2_olist_arrivals",
        "A": "a_realdata_validation",
    }
    referenced = sorted(set(re.findall(r"\]\(\.\./figures/([^/]+)/", ms)))
    assert referenced, "no figures referenced — the regex is wrong, not the paper"

    orphans = []
    for d in referenced:
        driver = owners.get(d)
        if driver is None:
            orphans.append(f"{d} (no known producer)")
        elif f"experiments.{driver}" not in recipe:
            orphans.append(f"{d} <- experiments.{driver} not in RUN_CAMPAIGN.md")
    assert not orphans, (
        "figures referenced by the manuscript that the campaign never produces: "
        + "; ".join(orphans)
    )
